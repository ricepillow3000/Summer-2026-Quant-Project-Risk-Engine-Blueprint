"""
The Book: one dossier per stock, plus the chain of steps that builds it.

How it works:
the rest of the engine answers questions; this module answers them IN ORDER,
because the order is the point. a dashboard of sixteen charts hands you sixteen
facts to piece together yourself; a book states one fact, uses it to narrow
the next question, and repeats. no new math here - every number comes from
liquidity.py, grit.py, strategies.py, pairing.py or signals.py. what's new is
that each step's output LIMITS what the next step can pick from, so the steps
can't be reordered or skipped.

    book size -> per-name capacity -> eligible set -> flyers -> cushions -> pairs

why liquidity goes first: `days_to_liquidate` needs weights, weights come from
the optimizer, and the optimizer should already know what's tradable - a loop.
we break it by flipping the same math around instead of calling it:

    days_i = weight_i * book / (participation * ADV_i)
    => the weight that exits in exactly one day is  participation * ADV_i / book

so `capacity_weight_per_day` needs no weights, grows as the book shrinks, and
needs nothing from the optimizer. it's the plain physics: the slice of your
book this name can soak up per day of selling. times the exit limit you'll
accept, that's a box constraint in the optimizer's own units.

What it won't do:
- no free text. every field is a number, an Enum, a ticker or None. sentences
  are fill-in-the-blank templates keyed by a `SentenceKey`, so a renderer that
  only holds a dossier can't invent a claim - it has no prices to invent from.
- no silent clipping. when liquidity cuts a position, `liquidity_binding` says
  so and both weights are kept.
- no unknown passed off as zero. a number we couldn't measure is None, never
  0.0 - `grit._score01` already ranks a NaN last, and a dossier has to be able
  to say "not measured" instead of "measured as bad".
- no blended "short-term grit". see below.

Why there's no short-term grit:
`grit.py` measures recovery from drawdowns and needs MIN_HISTORY_DAYS of
history; a 60-day window holds about zero full fall-and-recover cycles, so a
"short-term grit score" would be a stat on a sample that doesn't exist. so the
two stay apart, never blended: grit is LONG-horizon and works as a gate ("this
name has a recovery record at all"), while the short leg ranks on
`momentum_per_vol`, a measured trailing number we never call grit.

Honest limit on the liquidity gate: ADV averages calm and busy days, but the
day you're forced out is the day volume dries up and everyone wants the same
door. each name's exit days are true alone and too rosy together. so
`adv_stressed` (the 5th percentile of the same window) rides along with the
mean and is what the gate actually uses - a stressed claim, not an average
one. it's still not a market-impact model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd

from src.analytics import covariance_matrix
from src.grit import grit_scores
from src.pairing import anchor_rank, tail_gap
from src.covariance import TRADING_DAYS
from src.signals import MOMENTUM_LOOKBACK, MOMENTUM_SKIP, momentum_signal
from src.strategies import risk_parity_weights

# judgement calls, not fitted - written down so they can be argued with.
# change any of these and who's eligible changes.
DEFAULT_PARTICIPATION = 0.20      # share of a name's daily volume we'd take
DEFAULT_MAX_EXIT_DAYS = 1.0       # can't get out in a day = too big
DEFAULT_N_FLYERS = 3
MIN_MEANINGFUL_WEIGHT = 0.01      # under 1% of the book is noise
ADV_LOOKBACK_DAYS = 63            # ~3 months, same as ingestion.average_dollar_volume
ADV_STRESS_QUANTILE = 0.05        # the quiet-day volume the gate is held to
# MOMENTUM_LOOKBACK / MOMENTUM_SKIP live in src.signals: one momentum window
VOL_WINDOW = 60
# a flyer has to actually be flying. rank is relative: in a falling market the
# "best" of a bad bunch is still down, and calling it a high-flyer would be lying
MIN_FLYER_MOMENTUM = 0.0
# we want the GRITTIEST cushion, but anchor_rank counts grit as just 1 of 4
# parts. so each pool gets cut to its grittiest half first, then anchor_rank
# picks inside it. the same cut can gate flyers too (`flyer_grit_floor`) - a
# toggle, not a default, since "grittiest high-flyer" reads two fair ways and
# it's better to show the difference than pick one quietly.
GRIT_TOP_FRACTION = 0.5

# words that sink you in an interview. tests/test_narrative.py greps SENTENCES for these.
FORBIDDEN_WORDS = (
    "capped", "guaranteed", "guarantee", "ensures", "ensure", "fortress",
    "risk-free", "riskless", "protects", "protected", "pairs trading",
    "automated investing", "manages capital", "will outperform", "should buy",
)


class Role(str, Enum):
    FLYER = "flyer"
    CUSHION = "cushion"
    EXCLUDED = "excluded"


class ExcludedReason(str, Enum):
    NONE = "none"
    NO_VOLUME = "no_volume"                # no volume feed at all
    SHORT_HISTORY = "short_history"        # below grit's MIN_HISTORY_DAYS
    CAPACITY_AT_BOOK = "capacity_at_book"  # too big to exit at THIS book size
    MOMENTUM_NEGATIVE = "momentum_negative"  # eligible, but not rising
    GRIT_BELOW_FLOOR = "grit_below_floor"  # rising, but not in the gritty half
    RANK_BELOW_CUT = "rank_below_cut"      # eligible, simply not selected


class ExitBucket(str, Enum):
    """How long getting out takes, in words. `days_to_exit` shows "0.00" for a
    mega-cap in a small book, which looks broken, not instant - so the sentence
    says the bucket and the number stays in the dossier for the table."""
    SAME_DAY = "the same day"
    DAYS = "within a few days"
    WEEKS = "over weeks"
    NEVER = "not at all on this feed"


class SentenceKey(str, Enum):
    FLYER_PAIRED = "flyer_paired"
    FLYER_PAIRED_DEEPER = "flyer_paired_deeper"
    FLYER_PAIRED_EXTERNAL = "flyer_paired_external"
    FLYER_UNPAIRED = "flyer_unpaired"
    CUSHION_ANCHOR = "cushion_anchor"
    CUSHION_ANCHOR_DEEPER = "cushion_anchor_deeper"
    CUSHION_ANCHOR_EXTERNAL = "cushion_anchor_external"
    EXCLUDED_NO_VOLUME = "excluded_no_volume"
    EXCLUDED_SHORT_HISTORY = "excluded_short_history"
    EXCLUDED_CAPACITY = "excluded_capacity"
    EXCLUDED_MOMENTUM = "excluded_momentum"
    EXCLUDED_GRIT_FLOOR = "excluded_grit_floor"
    EXCLUDED_RANK = "excluded_rank"


# one template per SentenceKey. slots are dossier field names ONLY - the renderer
# never sees prices, so it can't say anything the dossier doesn't already hold.
# every verb here describes something we measured.
SENTENCES: dict[SentenceKey, str] = {
    # John, 2026-11-09: "simplify this for non technical users ... fun to
    # read and they understand 100%". Short sentences, everyday words, the
    # same measured numbers.
    SentenceKey.FLYER_PAIRED: (
        "{ticker} leads: #{momentum_rank} of {n_eligible} for its climb over "
        "the {momentum_lookback} trading days that ended {momentum_skip} days "
        "ago, allowing for how much it swings ({momentum_per_vol:+.2f}). It "
        "gets {final_weight:.1%} of the book, could be sold {exit_bucket}, and "
        "scores {grit_score:.0f} of 100 for grit. Its partner {partner} has "
        "milder worst days: a tail {tail_gap_pts:.1f} points shallower."
    ),
    SentenceKey.FLYER_PAIRED_DEEPER: (
        "{ticker} leads: #{momentum_rank} of {n_eligible} for its climb over "
        "the {momentum_lookback} trading days that ended {momentum_skip} days "
        "ago, allowing for how much it swings ({momentum_per_vol:+.2f}). It "
        "gets {final_weight:.1%} of the book, could be sold {exit_bucket}, and "
        "scores {grit_score:.0f} of 100 for grit. Its partner {partner} moves "
        "apart from the rest, but its own tail is {tail_gap_abs_pts:.1f} "
        "points deeper, so the pairing does not soften the worst days."
    ),
    SentenceKey.FLYER_UNPAIRED: (
        "{ticker} leads: it gets {final_weight:.1%} of the book and could be "
        "sold {exit_bucket}. It is #{momentum_rank} of {n_eligible} for its "
        "climb over the {momentum_lookback} trading days that ended "
        "{momentum_skip} days ago, allowing for how much it swings "
        "({momentum_per_vol:+.2f}). No name qualified as its steady partner, "
        "so it stands alone."
    ),
    SentenceKey.CUSHION_ANCHOR: (
        "{ticker} anchors {partner}. From the grittiest half of the names you "
        "can hold, it scored best on the anchor test, {anchor_score:.0f} of "
        "100: it moves little with the rest, swings little, has mild worst "
        "days and grit ({grit_score:.0f} of 100). It gets {final_weight:.1%} "
        "of the book, could be sold {exit_bucket}, and its tail is "
        "{tail_gap_pts:.1f} points shallower than its partner's."
    ),
    SentenceKey.CUSHION_ANCHOR_DEEPER: (
        "{ticker} anchors {partner}. From the grittiest half of the names you "
        "can hold, it scored best on the anchor test, {anchor_score:.0f} of "
        "100, because it moves little with the rest and swings little (grit "
        "{grit_score:.0f} of 100). It gets {final_weight:.1%} of the book and "
        "could be sold {exit_bucket}. Its own tail is {tail_gap_abs_pts:.1f} "
        "points deeper than its partner's, so it helps by moving differently, "
        "not by milder worst days."
    ),
    SentenceKey.EXCLUDED_NO_VOLUME: (
        "{ticker} shows no trading volume on this feed, so how fast it could "
        "be sold is unknown. It is left out rather than given a made-up number."
    ),
    SentenceKey.EXCLUDED_SHORT_HISTORY: (
        "{ticker} is too new to have a recovery record, so it is left out of "
        "the ranking rather than scored on a record it does not have."
    ),
    SentenceKey.EXCLUDED_CAPACITY: (
        "{ticker} trades too thinly: only {capacity_weight_per_day:.2%} of a "
        "${book_value:,.0f} book could be sold a day, so 1% of the book could "
        "not be sold within {max_exit_days:.0f} day(s). It drops out once the "
        "book is above ${book_breakpoint:,.0f}."
    ),
    SentenceKey.EXCLUDED_MOMENTUM: (
        "{ticker} passes the checks (could be sold {exit_bucket}; grit "
        "{grit_score:.0f} of 100), but it is not rising: its climb over the "
        "{momentum_lookback} trading days that ended {momentum_skip} days ago "
        "is {momentum_per_vol:+.2f}, allowing for its swings. So it is not "
        "picked to lead."
    ),
    SentenceKey.FLYER_PAIRED_EXTERNAL: (
        "{ticker} leads: #{momentum_rank} of {n_eligible} for its climb over "
        "the {momentum_lookback} trading days that ended {momentum_skip} days "
        "ago, allowing for how much it swings ({momentum_per_vol:+.2f}). It "
        "gets {final_weight:.1%} of the book, could be sold {exit_bucket}, and "
        "scores {grit_score:.0f} of 100 for grit. Nothing in your list had "
        "milder worst days, so its partner is {partner}, a steady holding not "
        "on your list, with a tail {tail_gap_pts:.1f} points shallower."
    ),
    SentenceKey.CUSHION_ANCHOR_EXTERNAL: (
        "{ticker} is a steady holding from outside your list, added to anchor "
        "{partner} only because nothing in your list had milder worst days. "
        "It scores {anchor_score:.0f} of 100 on the anchor test, gets "
        "{final_weight:.1%} of the book, could be sold {exit_bucket}, and its "
        "tail is {tail_gap_pts:.1f} points shallower than its partner's."
    ),
    SentenceKey.EXCLUDED_GRIT_FLOOR: (
        "{ticker} is rising ({momentum_per_vol:+.2f} over the "
        "{momentum_lookback} trading days that ended {momentum_skip} days ago, "
        "allowing for its swings) and could be sold {exit_bucket}, but its "
        "grit, {grit_score:.0f} of 100, is outside the grittiest half, and the "
        "grit floor is on."
    ),
    SentenceKey.EXCLUDED_RANK: (
        "{ticker} passes every check (could be sold {exit_bucket}; grit "
        "{grit_score:.0f} of 100) but did not make the top {n_flyers} for its "
        "climb over the {momentum_lookback} trading days that ended "
        "{momentum_skip} days ago."
    ),
}


def exit_bucket(days: float) -> ExitBucket:
    """Bucket an exit time. the cutoffs are stated, not fitted."""
    if not np.isfinite(days):
        return ExitBucket.NEVER
    if days < 1.0:
        return ExitBucket.SAME_DAY
    if days < 5.0:
        return ExitBucket.DAYS
    return ExitBucket.WEEKS


@dataclass(frozen=True)
class TickerDossier:
    """Everything the renderer is allowed to know about one stock.

    every field is a number, an Enum, a ticker or None. no free-text field on
    purpose: that's what stops a sentence saying something we didn't measure.
    """

    # who and where from
    ticker: str
    as_of: str                      # ISO date of the last price bar used
    universe_hash: str              # which basket every rank is relative to
    n_universe: int
    n_eligible: int

    # the book this is about (a dossier means nothing without it)
    book_value: float
    participation_rate: float
    max_exit_days: float
    n_flyers: int

    # liquidity, measured
    adv: float                      # mean daily dollar volume, ADV_LOOKBACK_DAYS
    adv_stressed: float             # 5th percentile of the same window
    capacity_weight_per_day: float  # participation * adv_stressed / book
    book_breakpoint: float | None   # book size at which this name drops out

    # capital
    target_weight: float            # what the optimizer wanted
    final_weight: float             # what liquidity allows
    liquidity_binding: bool         # True when liquidity, not risk, set the size
    position_value: float
    days_to_exit: float             # inf when there is no volume feed

    # role inputs - measured separately, NEVER blended into one score
    grit_score: float | None        # long-horizon recovery record; None = unmeasured
    recovery_pct: float | None      # None = no drawdown episode deep enough yet
    momentum_60d: float | None
    vol_60d: float | None
    momentum_per_vol: float | None  # the flyer rank key
    momentum_rank: int | None       # 1 = strongest among eligible

    # outcome
    eligible: bool
    role: Role
    excluded_reason: ExcludedReason

    # pairing (pairing.py), populated only for flyers and their anchors
    partner: str | None
    tail_gap: float | None          # ES(self) - ES(partner); >0 = partner thinner
    anchor_score: float | None      # anchor_rank composite, 0-100; cushions only
    exit_bucket: ExitBucket
    is_fallback_anchor: bool        # a defensive name from outside the universe

    # narrative
    sentence_key: SentenceKey
    momentum_lookback: int = MOMENTUM_LOOKBACK
    momentum_skip: int = MOMENTUM_SKIP

    def sentence(self) -> str:
        """This dossier's one sentence, built only from its fields.

        enums go in as their text, and `tail_gap_abs` lets a template state a
        deeper tail as a positive size - never "-0.8% shallower", which says
        the opposite of the number.
        """
        d = {k: (v.value if isinstance(v, Enum) else v)
             for k, v in asdict(self).items()}
        d["tail_gap_abs"] = abs(self.tail_gap) if self.tail_gap is not None else None
        # a gap between two ES numbers is a difference of percents, so it's
        # printed in points ("2.4 points shallower"), like the map does
        d["tail_gap_pts"] = None if self.tail_gap is None else 100 * self.tail_gap
        d["tail_gap_abs_pts"] = (None if self.tail_gap is None
                                 else 100 * abs(self.tail_gap))
        return SENTENCES[self.sentence_key].format(**d)


def _universe_hash(tickers) -> str:
    """Short stable hash of the basket every rank is relative to - without it
    "ranks 1 of 12" has no stated "of what"."""
    return hashlib.sha1("|".join(sorted(tickers)).encode(), usedforsecurity=False).hexdigest()[:8]


def capacity_weight_per_day(adv_stressed, book_value: float,
                            participation_rate: float = DEFAULT_PARTICIPATION):
    """Share of the book this name can soak up per day of selling.

    `liquidity.days_to_liquidate` flipped: that asks how long a weight takes
    to sell, this asks what weight sells in a day. no weights needed, so it
    can gate the optimizer instead of waiting on it.
    """
    if book_value <= 0:
        raise ValueError("book_value must be positive")
    return participation_rate * np.asarray(adv_stressed, dtype=float) / book_value


def book_breakpoint(adv_stressed: float, participation_rate: float,
                    max_exit_days: float,
                    min_weight: float = MIN_MEANINGFUL_WEIGHT) -> float | None:
    """Book size past which this name can't hold a real position inside the
    exit limit. None if it has no volume feed."""
    if not np.isfinite(adv_stressed) or adv_stressed <= 0:
        return None
    return float(participation_rate * adv_stressed * max_exit_days / min_weight)


def build_book(prices: pd.DataFrame, dollar_volume: pd.DataFrame,
               book_value: float,
               participation_rate: float = DEFAULT_PARTICIPATION,
               max_exit_days: float = DEFAULT_MAX_EXIT_DAYS,
               n_flyers: int = DEFAULT_N_FLYERS,
               min_weight: float = MIN_MEANINGFUL_WEIGHT,
               flyer_grit_floor: bool = False,
               fallback_anchors: set[str] | None = None,
               grit: dict | None = None,
               fallback_grit: dict | None = None) -> list[TickerDossier]:
    """
    The chain, in the only order it can run.

    pure: no network, no streamlit, no caching. callers fetch `prices` and
    `dollar_volume` and pass them in, so the whole thing tests offline on
    synthetic data.

    Args:
        prices: dates x tickers price history.
        dollar_volume: dates x tickers daily DOLLAR volume (ingestion
            .fetch_dollar_volume). names with no feed report 0.0.
        book_value: dollars invested. the knob the whole story turns on -
            raise it and names drop out.
        flyer_grit_floor: if True a flyer must ALSO be in the grittiest
            GRIT_TOP_FRACTION of the eligible names, not just have a recovery
            record. off by default: "grittiest high-flyer" reads two ways and
            we'd rather show the difference than pick one quietly.
        fallback_anchors: columns of `prices` that are steady holdings from
            OUTSIDE the user's basket. they're never counted, ranked or scored
            next to the user's names, never flyers, and only get a dossier and
            a weight if picked as a cushion - which only happens when nothing
            inside the basket had a thinner tail. shown on screen when used.
        grit: a grit_scores() result on FULL history for the user's names
            (every column not in `fallback_anchors`) - same peer group the
            Grit Zone tab ranks. if left out, grit is scored on `prices`, which
            is only right when `prices` is long: a 2-year window holds no named
            crisis, so a third of every score (resilience) came out 0.
        fallback_grit: a grit_scores() result for the fallback columns, only
            used to screen them against each other. if left out, it's scored
            on `prices` over every column.

    Returns one TickerDossier per user name plus one per fallback actually
    picked as an anchor: flyers first, then cushions, then everyone excluded.
    """
    if book_value <= 0:
        raise ValueError("book_value must be positive")
    # user's names and the outside shelf stay apart from here on: lumped
    # together, a 12-name basket read "17 of 17 names" and its grit got ranked
    # against five ETFs nobody picked
    fallback = set(fallback_anchors or ()) & set(prices.columns)
    universe = [t for t in prices.columns if t not in fallback]
    shelf = [t for t in prices.columns if t in fallback]
    if not universe:
        raise ValueError("no tickers")

    returns = prices.pct_change().dropna(how="all")
    as_of = str(pd.Timestamp(prices.index[-1]).date())
    uhash, n_universe = _universe_hash(universe), len(universe)

    # 1. grit on the WHOLE basket, BEFORE any liquidity gate, so no score
    #    depends on book size. bringing more money doesn't change a recovery record
    gr = grit if grit is not None else grit_scores(universe, prices=prices[universe])
    gscores = gr["scores"]
    short_history = set(gr["excluded"])
    if shelf:
        fg = (fallback_grit if fallback_grit is not None
              else grit_scores(list(prices.columns), prices=prices))
        shelf_grit = fg["scores"]
        short_history |= set(fg["excluded"]) & set(shelf)
    else:
        shelf_grit = gscores

    # 2. liquidity, no weights needed. adv_stressed runs the gate; plain adv
    #    rides along so you can see how rosy the average day is
    window = dollar_volume.tail(ADV_LOOKBACK_DAYS)
    every = universe + shelf
    adv = window.mean().reindex(every).fillna(0.0)
    adv_stressed = window.quantile(ADV_STRESS_QUANTILE).reindex(every).fillna(0.0)
    cap_per_day = pd.Series(
        capacity_weight_per_day(adv_stressed, book_value, participation_rate),
        index=every)
    capacity = cap_per_day * max_exit_days

    # 3. who's eligible. order matters: no feed is a different story from
    #    just too big for this book
    reason = pd.Series(ExcludedReason.NONE, index=every, dtype=object)
    reason[adv_stressed <= 0] = ExcludedReason.NO_VOLUME
    for t in short_history:
        if t in reason.index and reason[t] == ExcludedReason.NONE:
            reason[t] = ExcludedReason.SHORT_HISTORY
    too_small = (capacity < min_weight) & (reason == ExcludedReason.NONE)
    reason[too_small] = ExcludedReason.CAPACITY_AT_BOOK
    eligible = reason == ExcludedReason.NONE

    # 4. the short leg: trailing return per unit of vol - NOT called grit on
    #    purpose (see the module docstring)
    mom = momentum_signal(prices, MOMENTUM_LOOKBACK, MOMENTUM_SKIP).iloc[-1]
    vol = returns.tail(VOL_WINDOW).std() * np.sqrt(TRADING_DAYS)
    with np.errstate(divide="ignore", invalid="ignore"):
        mpv = (mom / vol.replace(0.0, np.nan)).reindex(every)

    # ranked among the user's names only: a fallback never gets a flyer rank
    elig = [t for t in universe if bool(eligible.get(t, False))]
    ranked = mpv[elig].dropna().sort_values(ascending=False)
    mom_rank = {t: i + 1 for i, t in enumerate(ranked.index)}

    # rank is relative, "high-flyer" isn't: a name only leads if its return per
    # unit of vol is actually positive, so a falling market gives fewer flyers,
    # not a best-of-a-bad-bunch. fallbacks never lead at all.
    rising = ranked[ranked > MIN_FLYER_MOMENTUM]
    flyer_pool = list(rising.index)
    grit_floored = set()
    if flyer_grit_floor:
        fg = gscores["grit_score"].reindex(flyer_pool).dropna().sort_values(
            ascending=False)
        if not fg.empty:
            keep = max(1, int(round(len(fg) * GRIT_TOP_FRACTION)))
            grit_floored = set(fg.index[keep:])
            flyer_pool = [t for t in flyer_pool if t not in grit_floored]
    flyers = flyer_pool[:n_flyers]

    # 5. cushions. anchor_rank counts grit as just 1 of 4 parts and we want the
    #    GRITTIEST cushion, so each pool is cut to its grittiest half first. the
    #    user's basket goes FIRST; an outside holding is only used when nothing
    #    inside had a thinner tail, and we say so when it is.
    partner, gap, anchor, external = {}, {}, {}, set()

    def _grittiest_half(names, scores):
        g = scores["grit_score"].reindex(names).dropna().sort_values(
            ascending=False)
        if g.empty:
            return list(names)
        return list(g.index[:max(1, int(round(len(g) * GRIT_TOP_FRACTION)))])

    def _best_anchor(flyer, names, scores):
        """(ticker, gap, anchor_score) for the best anchor in `names`, or None.
        gap = ES(flyer) - ES(anchor): positive means the anchor's tail is thinner."""
        cands = [n for n in names if n != flyer and n not in partner]
        if not cands:
            return None
        sub = returns[[flyer] + cands].dropna()
        if sub.shape[0] < 2 or sub.shape[1] < 2:
            return None
        try:
            g_in = scores["grit_score"].reindex([flyer] + cands).dropna()
            ar = anchor_rank(sub, flyer, grit=g_in if len(g_in) >= 2 else None)
        except (ValueError, KeyError):
            return None
        if ar.empty:
            return None
        best = ar.index[0]
        try:
            g = tail_gap(sub[[flyer, best]], flyer, best)["gap"]
        except (ValueError, KeyError):
            return None
        if g is None or not np.isfinite(g):
            return None
        return best, float(g), float(ar.loc[best, "anchor_score"])

    inside = [t for t in elig if t not in flyers]
    outside = [t for t in shelf if bool(eligible.get(t, False))]
    for f in flyers:
        pick = _best_anchor(f, _grittiest_half(inside, gscores), gscores)
        # only look outside when the basket couldn't thin the tail, and only
        # take the outsider if it actually does better
        if outside and (pick is None or pick[1] <= 0):
            alt = _best_anchor(f, _grittiest_half(outside, shelf_grit), shelf_grit)
            # the outside sentences say "shallower", so an outsider has to be
            # actually thinner, not just less deep than the inside pick
            if (alt is not None and alt[1] > 0
                    and (pick is None or alt[1] > pick[1])):
                external.add(alt[0])
                pick = alt
        if pick is None:
            continue
        best, g, score = pick
        partner[f], partner[best] = best, f
        anchor[best] = score
        gap[f] = gap[best] = g

    cushions = {v for k, v in partner.items() if k in flyers}
    n_eligible = int(eligible[universe].sum())

    # 6. what risk alone wants, then what liquidity allows - over the user's
    #    names plus only the outside anchors actually picked (an unused fallback
    #    holds nothing). covariance uses long-history names only: a dropna()
    #    across everyone would let ONE new listing throw away years of data for
    #    every other name. short-history names are out anyway, so target 0.
    in_book = universe + [t for t in shelf if t in external]
    long_names = [t for t in in_book if t not in short_history]
    if len(long_names) >= 2:
        cov = covariance_matrix(returns[long_names].dropna())
        target = pd.Series(risk_parity_weights(cov), index=cov.index)
    else:
        target = pd.Series(dtype=float)
    target = target.reindex(in_book).fillna(0.0)
    final = pd.concat([target, capacity.reindex(in_book)], axis=1).min(axis=1)
    binding = capacity.reindex(in_book) < target

    def _num(series, t):
        v = series.get(t, np.nan)
        return None if v is None or not np.isfinite(v) else float(v)

    out = []
    for t in in_book:
        is_flyer, is_cushion = t in flyers, t in cushions
        # gap = ES(flyer) - ES(cushion) on BOTH legs, so > 0 always means the
        # anchor's tail is thinner. the wording follows the sign instead of
        # assuming it: "-0.8% shallower" says the opposite of the number
        g_t = gap.get(t)
        thinner = g_t is not None and g_t > 0
        if is_flyer:
            role, why = Role.FLYER, ExcludedReason.NONE
            if partner.get(t) is None:
                key = SentenceKey.FLYER_UNPAIRED
            elif partner[t] in external:
                key = SentenceKey.FLYER_PAIRED_EXTERNAL
            else:
                key = (SentenceKey.FLYER_PAIRED if thinner
                       else SentenceKey.FLYER_PAIRED_DEEPER)
        elif is_cushion:
            role, why = Role.CUSHION, ExcludedReason.NONE
            if t in external:
                key = SentenceKey.CUSHION_ANCHOR_EXTERNAL
            else:
                key = (SentenceKey.CUSHION_ANCHOR if thinner
                       else SentenceKey.CUSHION_ANCHOR_DEEPER)
        else:
            role = Role.EXCLUDED
            why = reason[t]
            if why == ExcludedReason.NONE:
                m = mpv.get(t, np.nan)
                if m is not None and np.isfinite(m) and m <= MIN_FLYER_MOMENTUM:
                    why = ExcludedReason.MOMENTUM_NEGATIVE
                elif t in grit_floored:
                    why = ExcludedReason.GRIT_BELOW_FLOOR
                else:
                    why = ExcludedReason.RANK_BELOW_CUT
            key = {
                ExcludedReason.NO_VOLUME: SentenceKey.EXCLUDED_NO_VOLUME,
                ExcludedReason.SHORT_HISTORY: SentenceKey.EXCLUDED_SHORT_HISTORY,
                ExcludedReason.CAPACITY_AT_BOOK: SentenceKey.EXCLUDED_CAPACITY,
                ExcludedReason.MOMENTUM_NEGATIVE: SentenceKey.EXCLUDED_MOMENTUM,
                ExcludedReason.GRIT_BELOW_FLOOR: SentenceKey.EXCLUDED_GRIT_FLOOR,
                ExcludedReason.RANK_BELOW_CUT: SentenceKey.EXCLUDED_RANK,
            }[why]

        # an outside anchor's grit comes from the shelf screen it won; no
        # sentence prints it next to the user's names
        gt = shelf_grit if t in fallback else gscores
        in_g = t in gt.index
        gs = (float(gt.loc[t, "grit_score"])
              if in_g and np.isfinite(gt.loc[t, "grit_score"]) else None)
        rec = (float(gt.loc[t, "pct_recovered"])
               if in_g and np.isfinite(gt.loc[t, "pct_recovered"]) else None)
        stressed = float(adv_stressed[t])
        w_final = float(final[t])
        days = float(w_final / cap_per_day[t]) if cap_per_day[t] > 0 else float("inf")

        out.append(TickerDossier(
            ticker=t, as_of=as_of, universe_hash=uhash, n_universe=n_universe,
            n_eligible=n_eligible,
            book_value=float(book_value), participation_rate=participation_rate,
            max_exit_days=max_exit_days, n_flyers=n_flyers,
            adv=float(adv[t]), adv_stressed=stressed,
            capacity_weight_per_day=float(cap_per_day[t]),
            book_breakpoint=book_breakpoint(stressed, participation_rate,
                                            max_exit_days, min_weight),
            target_weight=float(target[t]), final_weight=w_final,
            liquidity_binding=bool(binding[t]),
            position_value=w_final * float(book_value), days_to_exit=days,
            grit_score=gs, recovery_pct=rec,
            momentum_60d=_num(mom, t), vol_60d=_num(vol, t),
            momentum_per_vol=_num(mpv, t), momentum_rank=mom_rank.get(t),
            eligible=bool(eligible.get(t, False)), role=role, excluded_reason=why,
            partner=partner.get(t), tail_gap=gap.get(t),
            anchor_score=anchor.get(t), exit_bucket=exit_bucket(days),
            is_fallback_anchor=t in external, sentence_key=key,
        ))

    order = {Role.FLYER: 0, Role.CUSHION: 1, Role.EXCLUDED: 2}
    return sorted(out, key=lambda d: (order[d.role], d.momentum_rank or 10**6))


def role_label(d: TickerDossier) -> str:
    """The card's chip, from the dossier's role AND its weight.

    Role.EXCLUDED means "no lead or anchor role", not "not in the book": an
    eligible name under the momentum cut still holds its risk-parity size
    (capped by liquidity), and calling it "Left out" read like it held nothing.
    only a name with zero weight is left out. sentence case, like every label
    on the page.
    """
    if d.role is Role.FLYER:
        return "Leads"
    if d.role is Role.CUSHION:
        return "Anchor"
    return "No role" if d.final_weight > 0 else "Left out"


def role_note(d: TickerDossier) -> str:
    """What a "No role" card holds, so "still held" can be checked right there.
    a sliver under 0.1% reads "<0.1%", never "0.0%"."""
    if role_label(d) != "No role":
        return ""
    w = d.final_weight
    return (f"{w:.1%}" if w >= 0.0005 else "<0.1%") + " of the book"


def headline(book: list[TickerDossier]) -> str:
    """The one number the book leads with (constraint 3: lead with one number)."""
    if not book:
        return "No names in this universe."
    d = book[0]
    # an outside anchor isn't one of the user's names
    n_ok = sum(1 for x in book if x.eligible and not x.is_fallback_anchor)
    days = f"{d.max_exit_days:.0f} day" + ("" if f"{d.max_exit_days:.0f}" == "1" else "s")
    line = (f"${d.book_value:,.0f} book: {n_ok} of your {d.n_universe} names "
            f"can hold at least 1% of the book and sell it within {days}, "
            f"trading at most {d.participation_rate:.0%} of a quiet day's volume "
            f"(only 5 in 100 of the last {ADV_LOOKBACK_DAYS} trading days trade less).")
    # the count is the visitor's names only; an outside anchor still holds
    # part of the book, so it gets named with its weight
    out = [x for x in book if x.is_fallback_anchor]
    if out:
        names = " and ".join(x.ticker for x in out)
        weights = " and ".join(f"{x.final_weight:.1%}" for x in out)
        one = len(out) == 1
        line += (f" {names}, {'an anchor' if one else 'anchors'} from outside "
                 f"your list, {'holds' if one else 'hold'} {weights} of the book.")
    return line

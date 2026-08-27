"""
The Book: one dossier per stock, and the dependency chain that produces it.

Quant Deep Dive:
The rest of this engine answers questions. This module answers them IN ORDER,
because the order is the point. A dashboard of sixteen charts states sixteen
facts and leaves the reader to assemble them; a book states one fact, uses it
to narrow the next question, and repeats. Nothing here is new mathematics -
every number comes from liquidity.py, grit.py, strategies.py, pairing.py or
signals.py. What is new is that each stage's output RESTRICTS the next stage's
domain, so the sequence cannot be reordered or skipped.

    book size -> per-name capacity -> eligible set -> flyers -> cushions -> pairs

Why liquidity comes first. `days_to_liquidate` needs weights, and weights come
from the optimizer, which should already know which names are tradable - a
cycle. It is broken by inverting the same arithmetic instead of calling it:

    days_i = weight_i * book / (participation * ADV_i)
    => the weight that exits in exactly one day is  participation * ADV_i / book

so `capacity_weight_per_day` is weight-free, monotone in book size, and needs
no optimizer output. It is the honest statement of the physics: this is the
share of your book that this name can absorb per day of exiting. Multiply by
the exit limit you are willing to live with and you have a box constraint in
the same units the optimizer already speaks.

What this module refuses to do:
- No free text. Every field is a number, an Enum, a ticker, or None. Sentences
  are slot-filled templates keyed by a `SentenceKey`, so a renderer that holds
  only a dossier cannot invent a claim - it has no prices to invent from.
- No silent clipping. When liquidity cuts a position, `liquidity_binding` says
  so and both weights are kept.
- No unknown dressed as a zero. A number that could not be measured is None,
  never 0.0 - `grit._score01` already ranks a NaN last, and a dossier must be
  able to say "not measured" rather than "measured as bad".
- No blended "short-term grit". See the ruling below.

Ruling on short-horizon grit (council, 2026-08-26):
`grit.py` measures recovery from drawdown episodes and needs MIN_HISTORY_DAYS
of history; a 60-day window contains roughly zero complete drawdown-recovery
cycles, so a "short-term grit score" would be a statistic computed on a sample
that does not exist. The two axes are therefore kept apart and never blended:
grit stays LONG-horizon and acts as an eligibility gate ("this name has a
recovery record at all"), while the short leg is ranked on `momentum_per_vol`,
a measured trailing quantity that is never called grit.

Honest limit on the liquidity gate: ADV is a trailing average taken over calm
and busy days alike, but the day you are forced to exit is the day volume dries
up and every holder wants the same door. Per-name exit days are individually
true and jointly optimistic. `adv_stressed` (the 5th percentile of the same
window) is carried alongside the mean and is what the gate actually uses, which
makes the eligibility claim a stressed one rather than an average one. It is
still not a market-impact model.
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
from src.signals import momentum_signal
from src.strategies import risk_parity_weights

# --- Disclosed judgement calls (not fitted, not optimal - stated so they can
#     be argued with). Changing any of these changes who is eligible.
DEFAULT_PARTICIPATION = 0.20      # share of a name's daily volume we'd take
DEFAULT_MAX_EXIT_DAYS = 1.0       # a position we cannot leave in a day is too big
DEFAULT_N_FLYERS = 3
MIN_MEANINGFUL_WEIGHT = 0.01      # below 1% of the book a position is noise
ADV_LOOKBACK_DAYS = 63            # ~3 months, matches ingestion.average_dollar_volume
ADV_STRESS_QUANTILE = 0.05        # the quiet-tape volume the gate is held to
MOMENTUM_LOOKBACK = 60
MOMENTUM_SKIP = 5                 # signals.momentum_signal default: dodges reversal
VOL_WINDOW = 60
# A flyer must actually be flying. Rank alone is relative: in a falling market
# the "best" of a bad set still has a negative trailing return, and calling it a
# high-flyer would be the ranking dressing up as a fact.
MIN_FLYER_MOMENTUM = 0.0
# John's brief is the GRITTIEST cushion, but anchor_rank scores grit as only one
# of four components. So the cushion pool is cut to the grittiest half of the
# eligible names FIRST, and anchor_rank then picks within that.
CUSHION_GRIT_TOP_FRACTION = 0.5

# Interview-fatal vocabulary. tests/test_narrative.py greps SENTENCES for these.
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
    RANK_BELOW_CUT = "rank_below_cut"      # eligible, simply not selected


class SentenceKey(str, Enum):
    FLYER_PAIRED = "flyer_paired"
    FLYER_UNPAIRED = "flyer_unpaired"
    CUSHION_ANCHOR = "cushion_anchor"
    EXCLUDED_NO_VOLUME = "excluded_no_volume"
    EXCLUDED_SHORT_HISTORY = "excluded_short_history"
    EXCLUDED_CAPACITY = "excluded_capacity"
    EXCLUDED_MOMENTUM = "excluded_momentum"
    EXCLUDED_RANK = "excluded_rank"


# One template per SentenceKey. Slots are dossier field names ONLY - the
# renderer never sees prices, so it cannot say anything the dossier does not
# already contain. Every verb here describes something that has been measured.
SENTENCES: dict[SentenceKey, str] = {
    SentenceKey.FLYER_PAIRED: (
        "{ticker} carries {final_weight:.1%} of the book and would take "
        "{days_to_exit:.2f} days to leave at {participation_rate:.0%} of its "
        "quiet-tape volume. It ranks {momentum_rank} of {n_eligible} eligible "
        "names on {momentum_lookback}-day return per unit of volatility "
        "({momentum_per_vol:+.2f}), and it has a recovery record: grit "
        "{grit_score:.0f} of 100 across {n_universe} names. Paired with "
        "{partner}, whose tail sits {tail_gap:.1%} shallower at ES 97.5."
    ),
    SentenceKey.FLYER_UNPAIRED: (
        "{ticker} carries {final_weight:.1%} of the book and would take "
        "{days_to_exit:.2f} days to leave. It ranks {momentum_rank} of "
        "{n_eligible} on {momentum_lookback}-day return per unit of volatility "
        "({momentum_per_vol:+.2f}). No eligible name cleared the anchor screen "
        "beside it, so it is shown unpaired rather than paired with a name that "
        "did not qualify."
    ),
    SentenceKey.CUSHION_ANCHOR: (
        "{ticker} is the anchor beside {partner}. Among the grittiest half of "
        "the eligible names it came top of the anchor screen at "
        "{anchor_score:.0f} of 100 - least tied to the dominant factor, "
        "steadiest, thinnest own tail - with grit {grit_score:.0f} of 100 "
        "across {n_universe} names. It holds {final_weight:.1%} of the book, "
        "leaves in {days_to_exit:.2f} days, and its tail sits {tail_gap:.1%} "
        "shallower at ES 97.5 than the name beside it."
    ),
    SentenceKey.EXCLUDED_NO_VOLUME: (
        "{ticker} reports no volume on this feed, so the exit horizon cannot be "
        "computed at all. It is left out rather than assigned a filled-in number."
    ),
    SentenceKey.EXCLUDED_SHORT_HISTORY: (
        "{ticker} has too little price history to measure a recovery record, so "
        "it is left out of the ranking rather than scored as though it had one."
    ),
    SentenceKey.EXCLUDED_CAPACITY: (
        "{ticker} absorbs {capacity_weight_per_day:.2%} of a "
        "${book_value:,.0f} book per day of exiting, so a position worth "
        "holding cannot be left within {max_exit_days:.0f} day(s). It drops out "
        "of this book above ${book_breakpoint:,.0f}."
    ),
    SentenceKey.EXCLUDED_MOMENTUM: (
        "{ticker} clears the liquidity and history gates - {days_to_exit:.2f} "
        "days to exit, grit {grit_score:.0f} of 100 - but its "
        "{momentum_lookback}-day return per unit of volatility is "
        "{momentum_per_vol:+.2f}. It is not rising, so it is not offered as a "
        "high-flyer."
    ),
    SentenceKey.EXCLUDED_RANK: (
        "{ticker} clears every gate - {days_to_exit:.2f} days to exit, grit "
        "{grit_score:.0f} of 100 - and simply did not rank in the top "
        "{n_flyers} on {momentum_lookback}-day return per unit of volatility."
    ),
}


@dataclass(frozen=True)
class TickerDossier:
    """Everything the renderer is allowed to know about one stock.

    Every field is a number, an Enum, a ticker string, or None. There is no
    free-text field by construction: that is what stops a sentence from
    saying something the engine did not measure.
    """

    # identity and provenance
    ticker: str
    as_of: str                      # ISO date of the last price bar used
    universe_hash: str              # which universe every rank is relative to
    n_universe: int
    n_eligible: int

    # the book this dossier is about (a dossier is meaningless without it)
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

    # role inputs - separately measured, NEVER blended into one score
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
    tail_gap: float | None
    anchor_score: float | None      # anchor_rank composite, 0-100; cushions only

    # narrative
    sentence_key: SentenceKey
    momentum_lookback: int = MOMENTUM_LOOKBACK

    def sentence(self) -> str:
        """Render this dossier's one sentence. Pure function of the fields."""
        return SENTENCES[self.sentence_key].format(**asdict(self))


def _universe_hash(tickers) -> str:
    """Short stable digest of the universe every percentile rank is relative to.
    Without it, "ranks 1 of 12" is a claim with no stated denominator."""
    return hashlib.sha1("|".join(sorted(tickers)).encode()).hexdigest()[:8]


def capacity_weight_per_day(adv_stressed, book_value: float,
                            participation_rate: float = DEFAULT_PARTICIPATION):
    """Share of the book this name absorbs per day of exiting.

    The inversion of `liquidity.days_to_liquidate`: that function asks how long
    a given weight takes to sell, this one asks what weight sells in a day. It
    takes no weights, so it can gate the optimizer instead of depending on it.
    """
    if book_value <= 0:
        raise ValueError("book_value must be positive")
    return participation_rate * np.asarray(adv_stressed, dtype=float) / book_value


def book_breakpoint(adv_stressed: float, participation_rate: float,
                    max_exit_days: float,
                    min_weight: float = MIN_MEANINGFUL_WEIGHT) -> float | None:
    """The book size above which this name can no longer hold a meaningful
    position inside the exit limit. None when it has no volume feed."""
    if not np.isfinite(adv_stressed) or adv_stressed <= 0:
        return None
    return float(participation_rate * adv_stressed * max_exit_days / min_weight)


def build_book(prices: pd.DataFrame, dollar_volume: pd.DataFrame,
               book_value: float,
               participation_rate: float = DEFAULT_PARTICIPATION,
               max_exit_days: float = DEFAULT_MAX_EXIT_DAYS,
               n_flyers: int = DEFAULT_N_FLYERS,
               min_weight: float = MIN_MEANINGFUL_WEIGHT) -> list[TickerDossier]:
    """
    The chain, in the only order it can run.

    Pure: no network, no Streamlit, no caching. Callers fetch `prices` and
    `dollar_volume` and hand them in, which is what makes the whole thing
    testable offline against synthetic data.

    Args:
        prices: dates x tickers price history.
        dollar_volume: dates x tickers daily DOLLAR volume (ingestion
            .fetch_dollar_volume). Names with no feed report 0.0.
        book_value: dollars invested. This is the parameter the whole story
            turns on - raise it and names fall out.

    Returns one TickerDossier per column of `prices`, flyers first, then
    cushions, then everything excluded.
    """
    if book_value <= 0:
        raise ValueError("book_value must be positive")
    universe = list(prices.columns)
    if not universe:
        raise ValueError("no tickers")

    returns = prices.pct_change().dropna(how="all")
    as_of = str(pd.Timestamp(prices.index[-1]).date())
    uhash, n_universe = _universe_hash(universe), len(universe)

    # 1. Grit on the FULL universe, BEFORE any liquidity gate, so that no
    #    score is a function of book size. A name's recovery record does not
    #    change because you brought more money.
    gr = grit_scores(universe, prices=prices)
    gscores = gr["scores"]
    short_history = set(gr["excluded"])

    # 2. Liquidity, weight-free. adv_stressed drives the gate; adv is carried
    #    beside it so the optimism of an average tape is visible, not hidden.
    window = dollar_volume.tail(ADV_LOOKBACK_DAYS)
    adv = window.mean().reindex(universe).fillna(0.0)
    adv_stressed = window.quantile(ADV_STRESS_QUANTILE).reindex(universe).fillna(0.0)
    cap_per_day = pd.Series(
        capacity_weight_per_day(adv_stressed, book_value, participation_rate),
        index=universe)
    capacity = cap_per_day * max_exit_days

    # 3. What risk alone would have wanted, then what liquidity allows.
    #    Covariance is fitted on the long-history names only: complete-case
    #    dropna() across the whole universe would let ONE recent listing throw
    #    away years of overlap for every other name. Short-history names are
    #    excluded from the book anyway, so they get target 0.
    long_names = [t for t in universe if t not in short_history]
    if len(long_names) >= 2:
        cov = covariance_matrix(returns[long_names].dropna())
        target = pd.Series(risk_parity_weights(cov), index=cov.index)
    else:
        target = pd.Series(dtype=float)
    target = target.reindex(universe).fillna(0.0)
    final = pd.concat([target, capacity], axis=1).min(axis=1)
    binding = capacity < target

    # 4. Eligibility. Order matters: a name with no feed is a different story
    #    from one that is merely too big for this book.
    reason = pd.Series(ExcludedReason.NONE, index=universe, dtype=object)
    reason[adv_stressed <= 0] = ExcludedReason.NO_VOLUME
    for t in short_history:
        if t in reason.index and reason[t] == ExcludedReason.NONE:
            reason[t] = ExcludedReason.SHORT_HISTORY
    too_small = (capacity < min_weight) & (reason == ExcludedReason.NONE)
    reason[too_small] = ExcludedReason.CAPACITY_AT_BOOK
    eligible = reason == ExcludedReason.NONE

    # 5. The short leg. Measured trailing return per unit of measured
    #    volatility - deliberately NOT called grit (see module docstring).
    mom = momentum_signal(prices, MOMENTUM_LOOKBACK, MOMENTUM_SKIP).iloc[-1]
    vol = returns.tail(VOL_WINDOW).std() * np.sqrt(252)
    with np.errstate(divide="ignore", invalid="ignore"):
        mpv = (mom / vol.replace(0.0, np.nan)).reindex(universe)

    elig = [t for t in universe if bool(eligible.get(t, False))]
    ranked = mpv[elig].dropna().sort_values(ascending=False)
    mom_rank = {t: i + 1 for i, t in enumerate(ranked.index)}
    # Rank is relative; "high-flyer" is not. A name only leads the book if its
    # trailing return per unit of volatility is actually positive, so a falling
    # market produces fewer flyers rather than a best-of-a-bad-set.
    flyers = list(ranked[ranked > MIN_FLYER_MOMENTUM].index[:n_flyers])

    # 6. Cushions: for each flyer, the best anchor among eligible non-flyers.
    #    anchor_rank is reused unchanged and still ranks, never promises - but
    #    it scores grit as only one component of four, and the brief is the
    #    GRITTIEST cushion. So the pool is cut to the grittiest half first and
    #    anchor_rank then picks inside it. Both steps are disclosed on screen.
    partner, gap, anchor = {}, {}, {}
    pool = [t for t in elig if t not in flyers]
    pool_grit = gscores["grit_score"].reindex(pool).dropna().sort_values(
        ascending=False)
    if not pool_grit.empty:
        keep = max(1, int(round(len(pool_grit) * CUSHION_GRIT_TOP_FRACTION)))
        pool = list(pool_grit.index[:keep])
    for f in flyers:
        cands = [f] + [t for t in pool if t not in partner]
        if len(cands) < 2:
            continue
        sub = returns[cands].dropna()
        if sub.shape[0] < 2:
            continue
        try:
            g_in = gscores["grit_score"].reindex(cands).dropna()
            ar = anchor_rank(sub, f, grit=g_in if not g_in.empty else None)
        except (ValueError, KeyError):
            continue
        if ar.empty:
            continue
        best = ar.index[0]
        partner[f], partner[best] = best, f
        anchor[best] = float(ar.loc[best, "anchor_score"])
        try:
            g = tail_gap(sub[[f, best]], f, best)["gap"]
            g = None if g is None or not np.isfinite(g) else float(g)
        except (ValueError, KeyError):
            g = None
        gap[f] = gap[best] = g

    cushions = {v for k, v in partner.items() if k in flyers}
    n_eligible = int(eligible.sum())

    def _num(series, t):
        v = series.get(t, np.nan)
        return None if v is None or not np.isfinite(v) else float(v)

    out = []
    for t in universe:
        is_flyer, is_cushion = t in flyers, t in cushions
        if is_flyer:
            role = Role.FLYER
            key = (SentenceKey.FLYER_PAIRED if partner.get(t)
                   else SentenceKey.FLYER_UNPAIRED)
            why = ExcludedReason.NONE
        elif is_cushion:
            role, key = Role.CUSHION, SentenceKey.CUSHION_ANCHOR
            why = ExcludedReason.NONE
        else:
            role = Role.EXCLUDED
            why = reason[t]
            if why == ExcludedReason.NONE:
                m = mpv.get(t, np.nan)
                why = (ExcludedReason.MOMENTUM_NEGATIVE
                       if m is not None and np.isfinite(m)
                       and m <= MIN_FLYER_MOMENTUM
                       else ExcludedReason.RANK_BELOW_CUT)
            key = {
                ExcludedReason.NO_VOLUME: SentenceKey.EXCLUDED_NO_VOLUME,
                ExcludedReason.SHORT_HISTORY: SentenceKey.EXCLUDED_SHORT_HISTORY,
                ExcludedReason.CAPACITY_AT_BOOK: SentenceKey.EXCLUDED_CAPACITY,
                ExcludedReason.MOMENTUM_NEGATIVE: SentenceKey.EXCLUDED_MOMENTUM,
                ExcludedReason.RANK_BELOW_CUT: SentenceKey.EXCLUDED_RANK,
            }[why]

        in_g = t in gscores.index
        gs = (float(gscores.loc[t, "grit_score"])
              if in_g and np.isfinite(gscores.loc[t, "grit_score"]) else None)
        rec = (float(gscores.loc[t, "pct_recovered"])
               if in_g and np.isfinite(gscores.loc[t, "pct_recovered"]) else None)
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
            anchor_score=anchor.get(t), sentence_key=key,
        ))

    order = {Role.FLYER: 0, Role.CUSHION: 1, Role.EXCLUDED: 2}
    return sorted(out, key=lambda d: (order[d.role], d.momentum_rank or 10**6))


def headline(book: list[TickerDossier]) -> str:
    """The one number the book leads with, per Meleona constraint 3."""
    if not book:
        return "No names in this universe."
    d = book[0]
    n_ok = sum(1 for x in book if x.eligible)
    return (f"${d.book_value:,.0f} book - {n_ok} of {d.n_universe} names hold a "
            f"position worth keeping and leave within {d.max_exit_days:.0f} "
            f"day(s) at {d.participation_rate:.0%} of quiet-tape volume.")

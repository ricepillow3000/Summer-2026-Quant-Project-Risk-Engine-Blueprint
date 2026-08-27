"""
Regression tests for src/narrative.py - The Book.

Run standalone:  python -m tests.test_narrative
Or with pytest:  pytest tests/test_narrative.py

All synthetic and offline. The point of these tests is not that the numbers are
pretty but that the DEPENDENCY CHAIN holds: book size must be able to delete a
stock, a name with no volume feed must never be handed a role, and grit must be
computed before the liquidity gate so that no recovery score moves when the
book size changes.
"""

import math
from dataclasses import fields

import numpy as np
import pandas as pd

from src.liquidity import days_to_liquidate
from src.narrative import (
    SENTENCES, FORBIDDEN_WORDS, ExcludedReason, ExitBucket, Role, SentenceKey,
    TickerDossier, book_breakpoint, build_book, capacity_weight_per_day,
    exit_bucket, headline, MIN_MEANINGFUL_WEIGHT,
)

# Six names chosen to exercise every ExcludedReason:
#   LARGE1/2/3  full history, deep volume  -> always eligible
#   THIN        full history, thin volume  -> CAPACITY_AT_BOOK at a big book
#   NOVOL       full history, zero volume  -> NO_VOLUME always
#   NEWLY       ~100 days of history       -> SHORT_HISTORY always
_TICKERS = ["LARGE1", "LARGE2", "LARGE3", "THIN", "NOVOL", "NEWLY"]
_N_DAYS = 700


def _approx(x, tol=1e-9):
    """Tiny local helper so this file needs no pytest import to run standalone."""
    class _A:
        def __eq__(self, other):
            return abs(other - x) <= tol * max(1.0, abs(x))

        def __repr__(self):
            return f"~{x}"
    return _A()


def _book_fixture(seed: int = 7):
    """Deterministic prices with real drawdowns plus a dollar-volume table."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=_N_DAYS)

    px = {}
    for i, t in enumerate(_TICKERS):
        r = rng.normal(0.0004 + 0.0001 * i, 0.011 + 0.002 * i, _N_DAYS)
        r[120:150] -= 0.010          # a drawdown, so grit has episodes to score
        r[430:450] -= 0.008          # and a second one
        px[t] = 100.0 * np.exp(np.cumsum(r))
    prices = pd.DataFrame(px, index=idx)
    # NEWLY only lists near the end -> below grit's MIN_HISTORY_DAYS
    prices.loc[prices.index[:-100], "NEWLY"] = np.nan

    dv = pd.DataFrame(
        {"LARGE1": 9.0e8, "LARGE2": 7.5e8, "LARGE3": 6.0e8,
         "THIN": 2.0e5, "NOVOL": 0.0, "NEWLY": 5.0e8},
        index=idx)
    return prices, dv


def _by_ticker(book):
    return {d.ticker: d for d in book}


def test_capacity_inversion_agrees_with_days_to_liquidate():
    """The whole design rests on one claim: capacity_weight_per_day is the
    INVERSE of days_to_liquidate, not a second opinion about liquidity. If the
    two ever disagree, the gate is gating on arithmetic the rest of the engine
    does not use."""
    adv = pd.Series({"A": 5.0e8, "B": 1.0e7})
    book, rate = 25_000_000.0, 0.20

    cap = pd.Series(capacity_weight_per_day(adv, book, rate), index=adv.index)
    # A weight equal to one day of capacity must take exactly one day to sell.
    dtl = days_to_liquidate(cap.values, adv, book_value=book,
                            participation_rate=rate)
    assert np.allclose(dtl["days"].values, 1.0)

    # ...and three days of capacity must take exactly three days.
    dtl3 = days_to_liquidate((cap * 3).values, adv, book_value=book,
                             participation_rate=rate)
    assert np.allclose(dtl3["days"].values, 3.0)


def test_capacity_is_monotone_in_book_size():
    """Doubling the book halves what each name can absorb. This is what makes
    'raise the book and names fall out' a fact rather than a slogan."""
    adv = pd.Series({"A": 5.0e8})
    small = capacity_weight_per_day(adv, 10_000_000.0, 0.20)[0]
    large = capacity_weight_per_day(adv, 20_000_000.0, 0.20)[0]
    assert math.isclose(small, 2.0 * large)

    # book_breakpoint is the book at which capacity hits the meaningful floor.
    bp = book_breakpoint(5.0e8, 0.20, 1.0, MIN_MEANINGFUL_WEIGHT)
    assert capacity_weight_per_day(pd.Series([5.0e8]), bp, 0.20)[0] == \
        _approx(MIN_MEANINGFUL_WEIGHT)
    assert book_breakpoint(0.0, 0.20, 1.0) is None      # no feed -> no breakpoint


def test_eligibility_is_monotone_decreasing_in_book_value():
    """The council's headline invariant. More money can only ever remove names
    from the book, never add them."""
    prices, dv = _book_fixture()
    counts, seen = [], []
    for book in (1e6, 1e7, 1e8, 1e9):
        b = build_book(prices, dv, book_value=book)
        ok = {d.ticker for d in b if d.eligible}
        counts.append(len(ok))
        seen.append(ok)

    assert counts == sorted(counts, reverse=True), counts
    # Not merely fewer - the surviving set must be a strict subset each time.
    for a, b in zip(seen, seen[1:]):
        assert b <= a, (a, b)
    assert counts[0] > counts[-1], "book size never bound anything"


def test_no_volume_and_short_history_never_receive_a_role():
    """A name the engine cannot measure must not be narrated as a pick."""
    prices, dv = _book_fixture()
    book = _by_ticker(build_book(prices, dv, book_value=5_000_000))

    assert book["NOVOL"].role is Role.EXCLUDED
    assert book["NOVOL"].excluded_reason is ExcludedReason.NO_VOLUME
    assert not book["NOVOL"].eligible
    assert book["NOVOL"].partner is None
    assert math.isinf(book["NOVOL"].days_to_exit)

    assert book["NEWLY"].role is Role.EXCLUDED
    assert book["NEWLY"].excluded_reason is ExcludedReason.SHORT_HISTORY
    assert book["NEWLY"].grit_score is None       # unmeasured, NOT zero

    for d in book.values():
        if d.role is not Role.EXCLUDED:
            assert d.eligible, f"{d.ticker} has a role without being eligible"


def test_grit_is_computed_before_the_liquidity_gate():
    """A recovery record is a property of the stock, not of how much money you
    brought. Every grit score must be identical across book sizes."""
    prices, dv = _book_fixture()
    small = _by_ticker(build_book(prices, dv, book_value=1e6))
    huge = _by_ticker(build_book(prices, dv, book_value=1e9))

    for t in _TICKERS:
        assert small[t].grit_score == huge[t].grit_score, t
        assert small[t].recovery_pct == huge[t].recovery_pct, t

    # THIN is gated out by capacity at the big book but keeps its grit score,
    # which is exactly the ordering the council required.
    assert huge["THIN"].excluded_reason is ExcludedReason.CAPACITY_AT_BOOK
    assert huge["THIN"].grit_score is not None


def test_liquidity_binding_is_recorded_never_silent():
    """When liquidity cuts a position, both weights survive and a flag says
    which one won. A silently clipped weight is an unexplained number."""
    prices, dv = _book_fixture()
    book = build_book(prices, dv, book_value=5e8)

    bound = [d for d in book if d.liquidity_binding]
    assert bound, "no name was liquidity-bound at a $500M book"
    for d in bound:
        assert d.final_weight <= d.target_weight + 1e-12
        assert d.final_weight == _approx(
            min(d.target_weight, d.capacity_weight_per_day * d.max_exit_days))
    for d in book:
        assert d.final_weight <= d.target_weight + 1e-12
        assert d.position_value == _approx(d.final_weight * d.book_value)


def test_dossier_has_no_free_text_field():
    """The renderer holds only a dossier, so a dossier must contain nothing it
    could quote as an unverified claim. Ticker, as-of and the universe digest
    are the only strings allowed."""
    prices, dv = _book_fixture()
    allowed_str = {"ticker", "as_of", "universe_hash", "partner"}

    for d in build_book(prices, dv, book_value=5_000_000):
        for f in fields(TickerDossier):
            v = getattr(d, f.name)
            if v is None or isinstance(v, (bool, int, float, Role,
                                           ExcludedReason, SentenceKey,
                                           ExitBucket)):
                continue
            assert isinstance(v, str) and f.name in allowed_str, \
                f"{f.name} is a free-text field carrying {v!r}"


def test_every_template_renders_and_avoids_forbidden_vocabulary():
    """Two failure modes at once: a template that names a field the dossier
    does not have (crashes live), and a template that overclaims (fails an
    interview). Both are caught here rather than on screen."""
    for key, tpl in SENTENCES.items():
        low = tpl.lower()
        for word in FORBIDDEN_WORDS:
            assert word not in low, f"{key.value} uses forbidden word {word!r}"

    assert set(SENTENCES) == set(SentenceKey), "a SentenceKey has no template"

    prices, dv = _book_fixture()
    keys_seen = set()
    for book_value in (1e6, 5e8):
        for d in build_book(prices, dv, book_value=book_value):
            s = d.sentence()                      # raises on an unknown slot
            assert s and "{" not in s, s
            assert d.ticker in s
            keys_seen.add(d.sentence_key)

    # The exclusion paths are the ones most likely to rot unnoticed.
    assert SentenceKey.EXCLUDED_NO_VOLUME in keys_seen
    assert SentenceKey.EXCLUDED_SHORT_HISTORY in keys_seen
    assert SentenceKey.EXCLUDED_CAPACITY in keys_seen


def test_flyers_are_ranked_on_momentum_not_grit():
    """The council's ruling made executable: the flyer order must follow
    momentum_per_vol. If someone later blends grit into the rank key, this
    fails."""
    prices, dv = _book_fixture()
    book = build_book(prices, dv, book_value=2_000_000, n_flyers=2)
    flyers = [d for d in book if d.role is Role.FLYER]
    assert flyers, "no flyers selected"

    mpv = [d.momentum_per_vol for d in flyers]
    assert all(v is not None for v in mpv)
    assert mpv == sorted(mpv, reverse=True), mpv

    # No eligible name that was passed over ranked ABOVE a selected flyer.
    passed_over = [d.momentum_per_vol for d in book
                   if d.eligible and d.role is not Role.FLYER
                   and d.momentum_per_vol is not None]
    if passed_over:
        assert min(mpv) >= max(passed_over) - 1e-12

    for d in flyers:
        assert d.grit_score is not None, "a flyer cleared the gate unmeasured"
        assert d.momentum_rank is not None and d.momentum_rank >= 1


def test_tail_wording_follows_the_sign_of_the_gap():
    """The bug this test exists for: the paired templates asserted the anchor's
    tail was "shallower" unconditionally, so a negative gap rendered as
    "-0.8% shallower" - which states the opposite of what the number says.
    tail_gap is ES(flyer) - ES(cushion) for BOTH legs, so > 0 means the anchor
    is the thinner one, and the sentence key must follow that sign."""
    shallower = {SentenceKey.FLYER_PAIRED, SentenceKey.CUSHION_ANCHOR}
    deeper = {SentenceKey.FLYER_PAIRED_DEEPER, SentenceKey.CUSHION_ANCHOR_DEEPER}

    # The claim each template makes, checked against the sign it is used for.
    for k in shallower:
        assert "shallower" in SENTENCES[k] and "DEEPER" not in SENTENCES[k]
    for k in deeper:
        assert "DEEPER" in SENTENCES[k]
        assert "shallower" not in SENTENCES[k]
        # A deeper tail must be printed as a positive magnitude, never as a
        # negative number sitting next to the word DEEPER.
        assert "{tail_gap_abs" in SENTENCES[k] and "{tail_gap:" not in SENTENCES[k]

    prices, dv = _book_fixture()
    for book_value in (1e6, 5e8, 2e6):
        for d in build_book(prices, dv, book_value=book_value, n_flyers=2):
            if d.sentence_key in shallower:
                assert d.tail_gap is not None and d.tail_gap > 0, d.ticker
                assert "shallower" in d.sentence()
            elif d.sentence_key in deeper:
                assert d.tail_gap is not None and d.tail_gap <= 0, d.ticker
                s_out = d.sentence()
                assert "DEEPER" in s_out
                assert "-" not in s_out.split("DEEPER")[0].split("tail is")[-1]
            # A partner is only ever narrated when the gap was measurable.
            if d.partner is not None:
                assert d.tail_gap is not None, d.ticker


def test_exit_bucket_boundaries_and_no_collapsed_zero():
    """days_to_exit collapses to "0.00" for a mega-cap at a small book, which
    reads as broken rather than as instant. Sentences quote the bucket."""
    assert exit_bucket(0.004) is ExitBucket.SAME_DAY
    assert exit_bucket(0.99) is ExitBucket.SAME_DAY
    assert exit_bucket(1.0) is ExitBucket.DAYS
    assert exit_bucket(4.99) is ExitBucket.DAYS
    assert exit_bucket(5.0) is ExitBucket.WEEKS
    assert exit_bucket(float("inf")) is ExitBucket.NEVER

    prices, dv = _book_fixture()
    for d in build_book(prices, dv, book_value=1e6):
        assert d.exit_bucket is exit_bucket(d.days_to_exit)
        assert "0.00 days" not in d.sentence()


# --- the two behaviours John chose after the council ------------------------

def _anchor_fixture(seed: int = 3):
    """A universe where nothing INSIDE it can cushion.

    FLY rises steadily at moderate volatility. WILD1/WILD2 are far more
    volatile, so every in-universe anchor has a DEEPER tail than the flyer.
    DEFN is a calm defensive holding from outside the universe - and is given
    the strongest momentum per unit of volatility in the whole set, so the test
    also proves a fallback anchor can never lead the book.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=_N_DAYS)
    # (long-run drift, long-run vol, recent drift, recent vol). The recent
    # window is set explicitly because a 60-day return drawn at 3% daily vol is
    # mostly noise - left to the seed, a WILD name wins the momentum rank by
    # accident and the fixture stops testing what it claims to.
    spec = {"FLY":   (0.0015, 0.012,  0.0025, 0.0020),
            "WILD1": (0.0002, 0.030, -0.0015, 0.0020),
            "WILD2": (0.0003, 0.030, -0.0015, 0.0020),
            "DEFN":  (0.0020, 0.004,  0.0030, 0.0010)}
    px = {}
    for t, (mu, sd, mu_r, sd_r) in spec.items():
        r = rng.normal(mu, sd, _N_DAYS)
        r[120:150] -= 0.008          # drawdown episodes so grit can score
        r[430:450] -= 0.006
        # Deep tails come from the long history; the momentum ranking comes
        # from the last stretch, so the two can be set independently.
        r[-70:] = rng.normal(mu_r, sd_r, 70)
        px[t] = 100.0 * np.exp(np.cumsum(r))
    prices = pd.DataFrame(px, index=idx)
    dv = pd.DataFrame({t: 8.0e8 for t in spec}, index=idx)
    return prices, dv


def test_flyer_grit_floor_is_off_by_default_and_excludes_when_on():
    """Both readings of "grittiest high-flyer" must be available. Off: grit
    only proves a recovery record exists and momentum ranks. On: a flyer must
    also sit in the grittiest half, and anyone dropped for it says so."""
    prices, dv = _book_fixture()
    loose = build_book(prices, dv, book_value=2_000_000, n_flyers=2)
    strict = build_book(prices, dv, book_value=2_000_000, n_flyers=2,
                        flyer_grit_floor=True)

    loose_f = {d.ticker for d in loose if d.role is Role.FLYER}
    strict_f = {d.ticker for d in strict if d.role is Role.FLYER}
    assert loose_f, "no flyers without the floor"
    # The floor can only ever remove candidates, never invent new ones.
    assert strict_f <= loose_f, (loose_f, strict_f)

    grit_of = {d.ticker: d.grit_score for d in loose}
    ranked = sorted((g for g in grit_of.values() if g is not None), reverse=True)
    for t in strict_f:
        assert grit_of[t] is not None
        assert grit_of[t] >= ranked[len(ranked) // 2] - 1e-9

    for d in strict:
        if d.excluded_reason is ExcludedReason.GRIT_BELOW_FLOOR:
            assert d.eligible and d.momentum_per_vol > 0
            assert "grit floor is on" in d.sentence()


def test_fallback_anchor_is_used_only_when_the_universe_cannot_cushion():
    """Inside first, outside only as a fallback - and disclosed when used."""
    prices, dv = _anchor_fixture()

    # The user's CHOSEN universe is FLY/WILD1/WILD2; DEFN exists only when it
    # is offered as a fallback. Without it the book must NOT pretend it found
    # a cushion.
    chosen = prices.drop(columns=["DEFN"]), dv.drop(columns=["DEFN"])
    inside_only = _by_ticker(
        build_book(*chosen, book_value=2_000_000, n_flyers=1))
    fly = [d for d in inside_only.values() if d.role is Role.FLYER][0]
    assert fly.ticker == "FLY", fly.ticker
    assert fly.partner in ("WILD1", "WILD2"), fly.partner
    assert fly.tail_gap <= 0, "fixture no longer has a deeper in-universe anchor"
    assert fly.sentence_key is SentenceKey.FLYER_PAIRED_DEEPER
    assert "does not thin the tail" in fly.sentence()
    assert not any(d.is_fallback_anchor for d in inside_only.values())

    # With DEFN offered as a fallback the anchor switches, and says so.
    withfb = _by_ticker(build_book(prices, dv, book_value=2_000_000, n_flyers=1,
                                   fallback_anchors={"DEFN"}))
    fly2 = [d for d in withfb.values() if d.role is Role.FLYER][0]
    assert fly2.ticker == "FLY"
    assert fly2.partner == "DEFN", fly2.partner
    assert fly2.tail_gap > 0, "the fallback must actually thin the tail"
    assert fly2.sentence_key is SentenceKey.FLYER_PAIRED_EXTERNAL
    assert "outside that universe" in fly2.sentence()

    defn = withfb["DEFN"]
    assert defn.role is Role.CUSHION and defn.is_fallback_anchor
    assert defn.sentence_key is SentenceKey.CUSHION_ANCHOR_EXTERNAL
    assert "OUTSIDE the chosen universe" in defn.sentence()


def test_a_fallback_anchor_never_leads_the_book():
    """DEFN has the strongest momentum per unit of volatility in the fixture.
    It must still never be a flyer - a defensive holding brought in to cushion
    cannot become the thing being cushioned."""
    prices, dv = _anchor_fixture()
    book = build_book(prices, dv, book_value=2_000_000, n_flyers=3,
                      fallback_anchors={"DEFN"})

    mpv = {d.ticker: d.momentum_per_vol for d in book}
    best = max(v for v in mpv.values() if v is not None)
    assert mpv["DEFN"] == best, "fixture no longer makes DEFN the momentum leader"
    assert all(d.ticker != "DEFN" for d in book if d.role is Role.FLYER)


def test_pairs_are_reciprocal_and_never_self_referential():
    prices, dv = _book_fixture()
    book = _by_ticker(build_book(prices, dv, book_value=2_000_000, n_flyers=2))

    for t, d in book.items():
        if d.partner is None:
            continue
        assert d.partner != t, f"{t} paired with itself"
        assert book[d.partner].partner == t, "pairing is not reciprocal"
        if d.tail_gap is not None:
            assert math.isfinite(d.tail_gap)
            assert book[d.partner].tail_gap == d.tail_gap


def test_headline_counts_match_the_dossiers():
    prices, dv = _book_fixture()
    book = build_book(prices, dv, book_value=5_000_000)
    line = headline(book)
    n_ok = sum(1 for d in book if d.eligible)
    assert f"{n_ok} of {len(_TICKERS)} names" in line
    assert "{" not in line
    assert headline([]) == "No names in this universe."


if __name__ == "__main__":
    import sys

    tests = [v for k, v in sorted(dict(globals()).items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

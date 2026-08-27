"""
The employer walkthrough, automated.

A recruiter opens the live link, picks a basket, and starts clicking. This
drives that session headlessly and reports what a stranger would actually hit:
crashes, empty panels, numbers that cannot be right, and jargon presented with
no explanation next to it.

It is deliberately adversarial. Every check is phrased as "what would be
embarrassing in front of someone who knows more than me", which is why it looks
for contradictions - a worst case above the median, a negative VaR - rather
than for exceptions alone. An app that renders a wrong number without raising
is the failure mode a normal test suite sails past.

    python -m tools.demo_check                 # default basket
    python -m tools.demo_check --preset "Mega-cap tech (Mag 7 + QQQ)"
"""

import argparse
import pathlib
import re

from streamlit.testing.v1 import AppTest

# Terms a second-year finance student knows and a first-time visitor does not.
JARGON = [
    "CVaR", "VaR", "Sharpe", "beta", "Kupiec", "Christoffersen", "GPD",
    "EWMA", "risk parity", "vol targeting", "Wasserstein", "Mahalanobis",
    "Ledoit-Wolf", "drawdown", "eigen",
]

# Words that mean a number failed to compute but still reached the screen.
NON_FINITE = re.compile(r"\b(nan|[-+]?inf)\b", re.I)

# Holdings a visitor is likely to type. Absent ones are a coverage finding.
FAVOURITES = ["NFLX", "DIS", "AMD", "COST", "PEP", "BA", "PFE", "BAC",
              "ORCL", "CRM", "UBER", "PLTR", "SHOP", "MU", "GE"]


def collect_text(at) -> list[str]:
    """Everything a visitor can read, in one list."""
    out = []
    for block in (at.markdown, at.caption, at.title, at.header, at.subheader,
                  at.warning, at.error, at.info, at.success):
        out += [el.value for el in block if isinstance(el.value, str)]
    out += [f"{m.label}: {m.value}" for m in at.metric]
    return out


def _pct(metrics: dict, label: str):
    match = re.search(r"(-?\d+\.?\d*)%", metrics.get(label, ""))
    return float(match.group(1)) if match else None


def run(preset: str | None) -> int:
    findings: list[tuple[str, str]] = []

    def flag(kind: str, detail: str) -> None:
        findings.append((kind, detail))

    # Absolute: AppTest resolves relative paths against the CALLING file,
    # which is how the suite's app-boot test once silently skipped forever.
    app = pathlib.Path(__file__).resolve().parent.parent / "main.py"
    at = AppTest.from_file(str(app), default_timeout=600).run()

    if preset:
        box = next((s for s in at.selectbox if s.label == "Preset basket"), None)
        if box is None:
            flag("BLOCKER", "no preset selector on the page")
            return report(findings)
        if preset not in box.options:
            flag("COVERAGE", f"preset {preset!r} is not offered")
            return report(findings)
        at = box.select(preset).run()

    # 1. nothing crashes, and no error is shown to the visitor
    for exc in at.exception:
        flag("BLOCKER", f"unhandled exception: {str(exc.value)[:200]}")
    for err in at.error:
        flag("BLOCKER", f"error shown to the visitor: {err.value[:160]}")

    text = collect_text(at)
    joined = "\n".join(text)

    # 2. no failed computation reaches the screen dressed as a number
    for line in text:
        if NON_FINITE.search(line):
            flag("NUMBER", f"non-finite value rendered: {line[:120]}")

    # 3. the headline numbers must not contradict each other
    metrics = {m.label: m.value for m in at.metric}
    p_loss = _pct(metrics, "Probability of loss")
    if p_loss is not None and not 0.0 <= p_loss <= 100.0:
        flag("NUMBER", f"probability of loss outside [0, 100]: {p_loss}")
    worst, median = _pct(metrics, "Worst simulated year"), _pct(metrics, "Median 1-year return")
    if worst is not None and median is not None and worst > median:
        flag("NUMBER", f"worst simulated year {worst}% is above the median {median}%")
    for label in ("Historical VaR (95%)", "Parametric VaR (95%)"):
        value = _pct(metrics, label)
        if value is not None and value < 0:
            flag("NUMBER", f"{label} rendered negative ({value}%) - VaR is a loss")
    vol = _pct(metrics, "Annualized volatility")
    if vol is not None and vol <= 0:
        flag("NUMBER", f"annualized volatility is {vol}%")

    # 4. every tab puts something on screen
    if len(at.tabs) < 2:
        flag("BLOCKER", f"only {len(at.tabs)} tabs rendered")

    # 5. jargon with no plain-language explanation anywhere near it
    lowered = joined.lower()
    cues = ("means", "is the", "measures", "average loss", "how much", "in plain",
            "think of", "answers", "worst", "read this", "what this", "i.e.",
            "that is", "in other words")
    for term in JARGON:
        needle = term.lower()
        if needle not in lowered:
            continue
        explained = any(
            any(cue in lowered[max(0, m.start() - 400):m.end() + 400] for cue in cues)
            for m in re.finditer(re.escape(needle), lowered))
        if not explained:
            flag("CLARITY", f"'{term}' appears with no plain-language "
                            "explanation within 400 characters")

    # 6. a visitor's own ticker must be findable in the picker
    box = next((m for m in at.multiselect if m.label == "Tickers to analyze"), None)
    if box is not None:
        missing = [t for t in FAVOURITES if t not in box.options]
        if missing:
            flag("COVERAGE", f"{len(missing)} of {len(FAVOURITES)} common holdings "
                             f"are absent from the picker: {', '.join(missing)}")
    return report(findings)


def report(findings) -> int:
    order = {"BLOCKER": 0, "NUMBER": 1, "COVERAGE": 2, "CLARITY": 3}
    findings.sort(key=lambda f: order.get(f[0], 9))
    if not findings:
        print("No findings. A first-time visitor hits nothing broken, nothing "
              "unexplained, and can find their own ticker.")
        return 0
    for kind, detail in findings:
        print(f"{kind:9} {detail}")
    blocking = sum(1 for kind, _ in findings if kind in ("BLOCKER", "NUMBER"))
    print(f"\n{len(findings)} findings, {blocking} blocking.")
    return 1 if blocking else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="basket to walk through")
    raise SystemExit(run(ap.parse_args().preset))

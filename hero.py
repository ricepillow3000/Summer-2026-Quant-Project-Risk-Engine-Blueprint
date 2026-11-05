"""The hero's words and markup, in one place for both the app and the served page.

main.py renders the hero; tools/public_files.py writes its boot skeleton into
index.html, where it paints from the first frame - before Streamlit's own
JavaScript has started, so before anything main.py sends can arrive. Both
render the same copy through the same classes (the skeleton as transparent
ink), so every shimmer bar has the size and line breaks of the words it stands
in for, at every viewport and short-screen tier. A skeleton drawn with its own
boxes drifted out of shape the first time the hero changed; this one cannot.

Kept on one line per element - st.markdown truncates raw HTML at a blank line
(Lessons #5).
"""

from src.ingestion import MAX_UNIVERSE
from src.scenarios import HISTORICAL_REGIMES

HERO_TITLE_LINES = ("Meliona", "Portfolio Risk Engine")
HERO_SUB = ("How much could a portfolio lose in a bad year, and what did real "
            "crashes do to it? Worked out from live end-of-day prices. Every stock "
            "falls sometimes. What sets a name apart is how it gets back up: we "
            "call that grit.")
HERO_CTA = "Start with the Grit Zone"
# The two labels the banner's system adds (ME-28, John 2026-11-05: the site
# takes the share banner's design): a kicker over the name, and the caps line
# over the receipt of engine facts. The copy pass owns their words.
HERO_KICKER = ("Live end-of-day data", "from Yahoo Finance")
HERO_RECEIPT = "Inside the engine"
# The tiles quote the engine, so they read its constants rather than restate
# them: add a crisis to HISTORICAL_REGIMES and the hero, the footer and the
# replay list all move together.
N_PATHS = 10_000
# Each simulated path is one year (TRADING_DAYS), so a path is a simulated year.
HERO_STATS = ((f"{N_PATHS:,}", "Simulated years"), (str(len(HISTORICAL_REGIMES)), "Real crises replayed"),
              ("2", "Ways to simulate"), (str(MAX_UNIVERSE), "Most names per portfolio"))
_CTA_ARROW = ('<span class="cta-arrow" aria-hidden="true"><svg viewBox="0 0 16 30" fill="none" '
              'xmlns="http://www.w3.org/2000/svg"><path class="shaft" d="M8 1 V22" '
              'stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path '
              'class="head" d="M2.5 16.5 L8 22.5 L13.5 16.5" stroke="currentColor" '
              'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span>')


# The crest's five pieces (assets/logo.svg, drawn about 0,0), engraved as a
# faint gold outline into a charcoal block - the share banner's one ornament
# (tools/og_image.html). test_the_crest_engraving_is_the_banners_and_used_sparingly
# holds these to the logo's own points and the opacity to the banner's.
CREST_PIECES = (
    "0,-35 27.358,-19.205 0,-3.409 -27.358,-19.205", "0,-2.309 4,0 0,2.309 -4,0",
    "0,35 27.358,19.205 0,3.409 -27.358,19.205",
    "-5.905,0 -30.311,-14.091 -30.311,14.091", "5.905,0 30.311,-14.091 30.311,14.091",
)
ENGRAVING_OPACITY = ".1"


def crest_engraving() -> str:
    """The crest as a hairline engraving: no fill, gold at 10%, 1px at any size."""
    pieces = "".join(f'<polygon points="{p}" vector-effect="non-scaling-stroke"/>' for p in CREST_PIECES)
    return ('<svg class="crest-engraving" viewBox="-36 -36 72 72" aria-hidden="true" focusable="false" '
            f'xmlns="http://www.w3.org/2000/svg"><g fill="none" stroke="#C6A26B" '
            f'stroke-opacity="{ENGRAVING_OPACITY}" stroke-width="1">{pieces}</g></svg>')


def hero_html(skeleton: bool = False, crest: str = "") -> str:
    """The hero, or its boot skeleton: same classes, same copy, words masked.

    The banner's layout (tools/og_image.html): on the paper, the crest beside
    a kicker, the name, the annual-report rule, the sentence and the button;
    on a charcoal block to the right, the engine's facts as a receipt - a gold
    caps line, a gold rule, then rows of label and value - over the crest's
    faint engraving."""
    ink = (lambda t: f'<span class="sk-ink">{t}</span>') if skeleton else (lambda t: t)
    title = "".join(f'<span class="hline">{ink(line)}</span>' for line in HERO_TITLE_LINES)
    kicker = "<br>".join(ink(line) for line in HERO_KICKER)
    stats = "".join(f'<div class="hstat"><div class="l">{ink(label)}</div>'
                    f'<div class="n">{ink(n)}</div></div>' for n, label in HERO_STATS)
    if skeleton:
        # Streamlit appends its heading-link icon to a markdown <h1>, after the
        # two block lines, so the real title holds a third (empty) line box. The
        # skeleton is painted by index.html, outside Streamlit, so it holds the
        # same line itself - without it every box below sat one line too high.
        title += '<span class="sk-anchor">&#8203;</span>'
        head, crest = '<div class="hero-section">', ""
        cta = f'<span class="cta-btn">{HERO_CTA}{_CTA_ARROW}</span>'
    else:
        head = '<div class="hero-section" id="hero">'
        cta = f'<a href="#grit-showcase" class="cta-btn">{HERO_CTA}{_CTA_ARROW}</a>'
    return (f'{head}<div class="hero-left"><div class="hero-mark"><div class="hero-crest">{crest}</div>'
            f'<div class="kicker">{kicker}</div></div>'
            f'<h1 class="hero-title">{title}</h1><div class="ar-rule"></div>'
            f'<div class="hero-sub">{ink(HERO_SUB)}</div>{cta}</div>'
            f'<div class="hero-stats">{crest_engraving()}<div class="hreceipt-h">{ink(HERO_RECEIPT)}</div>'
            f'<div class="hstat-list">{stats}</div></div></div>')


def boot_skeleton_html() -> str:
    """The veil index.html paints first; the real hero lifts it (main.py)."""
    return ('<div id="boot-skel" aria-hidden="true">' + hero_html(skeleton=True)
            + '<div class="sk-band"></div><div class="sk-load">Loading Meliona</div></div>')

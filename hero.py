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

from src.scenarios import HISTORICAL_REGIMES

HERO_TITLE_LINES = ("Meleona", "Portfolio Risk Engine")
HERO_SUB = ("A hedge-fund-grade portfolio risk engine - VaR, CVaR, Monte Carlo stress "
            "testing, and named factor exposures, computed live from real market data. "
            "But every stock has drawdowns. What sets a name apart is what happens "
            "after one - that's what we call <strong>grit</strong>.")
HERO_CTA = "Explore what we do"
# The tiles quote the engine, so they read its constants rather than restate
# them: add a crisis to HISTORICAL_REGIMES and the hero, the footer and the
# replay list all move together.
N_PATHS = 10_000
HERO_STATS = ((f"{N_PATHS:,}", "Simulated paths"), (str(len(HISTORICAL_REGIMES)), "Crises replayed"),
              ("2", "Monte Carlo engines"), ("4", "Factor exposures"))
_CTA_ARROW = ('<span class="cta-arrow" aria-hidden="true"><svg viewBox="0 0 16 30" fill="none" '
              'xmlns="http://www.w3.org/2000/svg"><path class="shaft" d="M8 1 V22" '
              'stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path '
              'class="head" d="M2.5 16.5 L8 22.5 L13.5 16.5" stroke="currentColor" '
              'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span>')


def hero_html(skeleton: bool = False, crest: str = "") -> str:
    """The hero, or its boot skeleton: same classes, same copy, words masked."""
    ink = (lambda t: f'<span class="sk-ink">{t}</span>') if skeleton else (lambda t: t)
    title = "".join(f'<span class="hline">{ink(line)}</span>' for line in HERO_TITLE_LINES)
    stats = "".join(f'<div class="hstat"><div class="n">{ink(n)}</div>'
                    f'<div class="l">{ink(label)}</div></div>' for n, label in HERO_STATS)
    if skeleton:
        # Streamlit appends its heading-link icon to a markdown <h1>, after the
        # two block lines, so the real title holds a third (empty) line box. The
        # skeleton is painted by index.html, outside Streamlit, so it holds the
        # same line itself - without it every box below sat one line too high.
        title += '<span class="sk-anchor">&#8203;</span>'
        head, crest = '<div class="hero-section">', ""
        cta = f'<span class="cta-btn">{HERO_CTA}{_CTA_ARROW}</span>'
    else:
        head = '<div class="hero-section reveal" id="hero">'
        cta = f'<a href="#grit-showcase" class="cta-btn">{HERO_CTA}{_CTA_ARROW}</a>'
    return (f'{head}<div class="hero-left"><div class="hero-crest">{crest}</div>'
            f'<h1 class="hero-title">{title}</h1>'
            f'<div class="hero-sub">{ink(HERO_SUB)}</div>{cta}</div>'
            f'<div class="hero-stats">{stats}</div></div>')


def boot_skeleton_html() -> str:
    """The veil index.html paints first; the real hero lifts it (main.py)."""
    return ('<div id="boot-skel" aria-hidden="true">' + hero_html(skeleton=True)
            + '<div class="sk-band"></div><div class="sk-load">Loading Meleona</div></div>')

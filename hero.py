"""The page's shared words and markup: the hero, the builder and the questions.

The hero, the About block's facts and links, and the five questions live here
because two places print them: main.py renders them, and tools/public_files.py
writes them into the served index.html (the hero's boot skeleton, and the
Person and FAQPage JSON-LD a crawler reads).

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

import html as _html
import re
from urllib.parse import quote

from src.ingestion import MAX_UNIVERSE
from src.scenarios import HISTORICAL_REGIMES

HERO_TITLE_LINES = ("Meliona", "Portfolio Risk Engine")
HERO_SUB = ("How much could a portfolio lose in a bad year, and what did real "
            "crashes do to it? Worked out from live end-of-day prices. Every stock "
            "falls sometimes. What sets a name apart is how it gets back up: we "
            "call that grit.")
HERO_CTA = "Start with the Grit Zone"
# The second way in from the first screen (ME-28 part 4, John 2026-11-05: a
# contact CTA above the fold): a quiet link beside the button, gliding to the
# contact form at the foot of the page.
HERO_CONTACT = "Contact John"
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
        cta = (f'<div class="hero-ctas"><span class="cta-btn">{HERO_CTA}{_CTA_ARROW}</span>'
               f'<span class="cta-link">{ink(HERO_CONTACT)}{_CTA_ARROW}</span></div>')
    else:
        head = '<div class="hero-section" id="hero">'
        cta = (f'<div class="hero-ctas"><a href="#grit-showcase" class="cta-btn">{HERO_CTA}{_CTA_ARROW}</a>'
               f'<a href="#contact" class="cta-link">{HERO_CONTACT}{_CTA_ARROW}</a></div>')
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


# ---- The builder: who made this, and where to find him ----------------------
# ME-28 part 4 (John, 2026-11-05): "a photo of me where my LinkedIn, Handshake
# and GitHub can be clicked on". Only facts John gave are stated here. The
# same three addresses are the Person's sameAs in the served head's JSON-LD
# (tools/public_files.py), so the page and the schema cannot disagree.
PERSON_NAME = "John Nguyen"
PERSON_ROLE = "Computer Information Systems student specializing in AI & Analytics"
ABOUT_BIO = ("John Nguyen is a Computer Information Systems student specializing "
             "in AI & Analytics. He built Meliona: the price feed, the risk engine "
             "and this page. Its code is public, so every number can be checked.")
# Where the contact form's email goes (John, 2026-11-05: his personal inbox,
# "I'm more active there"), with LinkedIn as the second way to reach him.
# main.py reads it through host_setting("CONTACT_EMAIL", ...) so a deployment
# can change it without a code change. Support mail (the session reference,
# the kill switch page) stays on the project mailbox, SUPPORT_EMAIL.
CONTACT_EMAIL = "john4000.nguyen@gmail.com"
LINKEDIN_URL = "https://www.linkedin.com/in/johnnguyen3000/"
PORTRAIT = "john-nguyen"              # static/john-nguyen.{avif,webp,jpg}, 480x600
PORTRAIT_ALT = "John Nguyen, the builder of Meliona, in a navy blazer"
REPO_URL = "https://github.com/ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint"
# Monochrome marks on a 24-unit grid, drawn in currentColor so they take the
# ink or the gold of the ground they sit on - never a brand colour. GitHub's
# mark and Handshake's are Simple Icons' paths (CC0, simple-icons 16.36.0);
# Simple Icons carries no LinkedIn mark, so its "in" in a square is drawn here
# (the letters cut out of the square, fill-rule evenodd).
PROFILE_LINKS = (
    ("LinkedIn", LINKEDIN_URL,
     "M0 0h24v24H0zM5.4 3.3a2.05 2.05 0 1 0 0 4.1a2.05 2.05 0 1 0 0-4.1zM3.6 9h3.6v11.5H3.6z"
     "M9.5 9h3.45v1.6c.5-.95 1.7-1.9 3.5-1.9 3.75 0 4.45 2.45 4.45 5.65v6.15h-3.6v-5.45"
     "c0-1.3-.05-2.95-1.8-2.95-1.8 0-2.1 1.4-2.1 2.85v5.55H9.5z"),
    ("GitHub", "https://github.com/ricepillow3000",
     "M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 "
     "0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7"
     "c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 "
     "3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 "
     "1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 "
     "1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 "
     "1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 "
     "2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"),
    ("Handshake", "https://app.joinhandshake.com/profiles/74fwhv",
     "M20.728 0 16.49 24h-4.583l1.87-10.532-4.743 3.893L7.856 24H3.272L7.51 0h4.582L9.806 "
     "13.012l4.729-3.862L16.145 0h4.583z"),
)


# "Opens elsewhere": a short diagonal arrow in the CTA arrow's hand.
_OUT_ARROW = ('<svg class="pm-go" viewBox="0 0 16 16" aria-hidden="true" focusable="false" '
              'xmlns="http://www.w3.org/2000/svg"><path d="M4 12 L12 4 M6 4 H12 V10" fill="none" '
              'stroke="currentColor" stroke-width="1.4" stroke-linecap="round" '
              'stroke-linejoin="round"/></svg>')


def profile_marks(cls: str = "pmark", arrow: bool = False) -> str:
    """The three profile links: a mark, its name, a new tab. rel="me" tells a
    crawler the profiles are the same person as this page; noopener and
    noreferrer keep the new tab from reaching back into this one."""
    return "".join(
        f'<a class="{cls}" href="{url}" target="_blank" rel="noopener noreferrer me" '
        f'aria-label="{PERSON_NAME} on {name}"><svg viewBox="0 0 24 24" aria-hidden="true" '
        f'focusable="false" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" '
        f'fill-rule="evenodd" d="{path}"/></svg><span>{name}</span>'
        f'{_OUT_ARROW if arrow else ""}</a>'
        for name, url, path in PROFILE_LINKS)


def about_html() -> str:
    """About the builder, on a stone block: the portrait, the name and one line
    of facts, and a receipt of the three profiles - the hero's receipt, its
    rows the links. A real <img> (alt text, its own size, lazy) in a
    <picture>: AVIF first, then WebP, then JPEG."""
    base = f"./app/static/{PORTRAIT}"
    return (
        '<div class="about-band" id="about"><div class="about-grid">'
        f'<figure class="about-photo"><picture>'
        f'<source type="image/avif" srcset="{base}.avif">'
        f'<source type="image/webp" srcset="{base}.webp">'
        f'<img src="{base}.jpg" width="240" height="300" loading="lazy" decoding="async" '
        f'alt="{PORTRAIT_ALT}"></picture></figure>'
        '<div class="about-text"><div class="kicker">About the builder</div>'
        f'<h2 class="showcase-title">{PERSON_NAME}</h2><div class="ar-rule"></div>'
        f'<p class="about-bio">{_html.escape(ABOUT_BIO)}</p></div>'
        '<div class="about-find"><div class="about-find-h">Find John online</div>'
        f'{profile_marks(arrow=True)}</div></div></div>')


# ---- Why Meliona exists: John's story ---------------------------------------
# John, 2026-11-05: "a story place where we explain why we wanted to create
# this", beside a second photo of him "when I'm not in my suit". Every fact
# here is one John gave (his dad, the question, grit, the AI linkages); the
# promise stays honest - the page describes and simulates, it advises no one.
STORY_PHOTO = "john-nguyen-story"     # static/john-nguyen-story.{avif,webp,jpg}, 480x600
STORY_PHOTO_ALT = "John Nguyen out of the suit, smiling, in a maroon quarter-zip sweater"
STORY_TITLE = "Built for a question too many investors share"
STORY_QUOTE = "Not the best every year. Back up after every fall."
# John, 2026-11-05: his dad starts the story, but the story is not about him:
# it ends on everyone who faces the same problem.
STORY = (
    "It started at home. My dad talks about stocks a lot, and one question kept "
    "coming back: which stock can I actually rely on, and hold for years? He never "
    "found a good answer, and I came to see that he was far from alone.",
    "The more I looked, the more I noticed a pattern. The stocks worth holding for a "
    "long time are not always the best performers. They are the ones that fall and get "
    "back up. Crisis after crisis, they lose ground, win it back, and keep their numbers "
    "strong. I call that grit, and the place those stocks live, the Grit Zone.",
    "Grit was only half of it. Long-term investors also want a part in what is growing "
    "fastest today: the AI stocks. So I linked the two. A gritty stock is the anchor, an "
    "AI high-flyer sits beside it, and each carries the same share of the risk, so they "
    "hold each other up. If a flyer falls, another can take its place: the one that "
    "works best with the anchor when the two are tested together, against real crashes "
    "and simulated bad years.",
    "I saw a problem, so I put my effort into solving that one problem. Meliona is "
    "built for people like my dad: anyone looking for a stock they can count on, while "
    "still taking part in what comes next.",
)


def story_html() -> str:
    """The story on the paper ground: the out-of-suit photo under a charcoal
    cap on the left, the words on the right, one line pulled out between the
    second and third paragraphs, signed. A real <img>, like the portrait."""
    base = f"./app/static/{STORY_PHOTO}"
    paras = [f'<p class="story-p">{_html.escape(t)}</p>' for t in STORY]
    # a <p>, not a <blockquote>: Streamlit greys a blockquote and redraws its rule
    paras.insert(2, f'<p class="story-quote">{_html.escape(STORY_QUOTE)}</p>')
    return (
        '<div class="story-section" id="story"><div class="story-grid">'
        '<figure class="story-photo"><picture>'
        f'<source type="image/avif" srcset="{base}.avif">'
        f'<source type="image/webp" srcset="{base}.webp">'
        f'<img src="{base}.jpg" width="240" height="300" loading="lazy" decoding="async" '
        f'alt="{STORY_PHOTO_ALT}"></picture>'
        '<figcaption>John, out of the suit</figcaption></figure>'
        '<div class="story-text"><div class="kicker">Why I built this</div>'
        f'<h2 class="showcase-title">{_html.escape(STORY_TITLE)}</h2><div class="ar-rule"></div>'
        f'{"".join(paras)}'
        f'<p class="story-sign">{PERSON_NAME}, builder of Meliona</p></div></div></div>')


# ---- Five plain questions -------------------------------------------------
# Rendered on the page (main.py) and served as FAQPage JSON-LD in the head
# (tools/public_files.py) from this one tuple, so the two cannot drift
# (test_the_schema_parses_and_its_faq_is_the_pages). Each answer is held to
# the code: PRIVACY.md for storage, CLAUDE.md constraint #1 for the numbers.
FAQS = (
    ("What does Meliona tell me?",
     "How much a portfolio you pick could lose in a bad year, what real crashes did to it, "
     "and which of its names have got back up after their falls. The big number is the "
     "average loss in the worst 5 of every 100 simulated years."),
    ("Where do the numbers come from?",
     "From end-of-day prices pulled live from Yahoo Finance and worked out by the engine's "
     "own code, which is public. No number is typed in by hand or written by an AI model. "
     "The drawer at the end of the analysis shows the source and when each price was pulled."),
    ("Is this investment advice?",
     "No. Meliona is an educational student project. It describes what happened and "
     "simulates years that could happen; it does not tell anyone what to buy or sell, "
     "and nothing on it is a forecast."),
    ("What is grit?",
     "Our name for how a stock gets back up after a fall: how fast and how fully it won "
     "back its own falls of 5% or more, how often a year of holding it ended in a gain, "
     "and how it held up in the real crises we replay. Scores rank the names in your "
     "basket against each other."),
    ("Does the site store my data?",
     "No. There is no account and no tracking. Your choices are held in memory for "
     "this visit only and go when you close the tab. The contact form stores nothing: it only prepares "
     "an email in your own email app. The server keeps public price files and short "
     "error logs tagged with a random session reference."),
)


def faq_html() -> str:
    """The five questions as ruled rows: question in bold serif, answer under it."""
    rows = "".join(f'<div class="faq-row"><h3 class="faq-q">{_html.escape(q)}</h3>'
                   f'<p class="faq-a">{_html.escape(a)}</p></div>' for q, a in FAQS)
    return ('<div class="faq-section" id="faq"><div class="kicker">Questions, answered</div>'
            '<h2 class="showcase-title">Before you ask</h2><div class="ar-rule"></div>'
            f'<div class="faq-list">{rows}</div></div>')


# ---- Contact: the form's rules and the email it prepares --------------------
# There is no mail server and no database (CLAUDE.md, "Launch operations"), so
# the form sends nothing: it checks the words, then prepares an email the
# visitor sends from their own app. The visitor's text only ever leaves as
# percent-encoded mailto: parameters (quote(..., safe="") leaves A-Z, a-z,
# 0-9 and _.-~% - nothing an HTML attribute or a URL can be broken out of
# with); it is never printed into the page, never logged and never written
# to disk (test_the_contact_form_prepares_an_email_and_keeps_nothing).
CONTACT_TOPICS = ("Question", "Feedback", "Recruiting", "Report a problem")
CONTACT_MAX = 1000                      # characters; a mailto: link stays under 2,000 bytes for plain text
CONTACT_NAME_MAX = 80
RESPONSE_WINDOW = "within 2 business days"
RESPONSE_TIME = f"John replies {RESPONSE_WINDOW}."
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def contact_mailto(to: str, name: str, topic: str, message: str, ref: str):
    """(mailto URL, None) for a valid message, or (None, what to fix).

    Every rule is checked here, on the server, whatever the browser allowed:
    the topic is one of CONTACT_TOPICS, the message is not empty and at most
    CONTACT_MAX characters, the optional name at most CONTACT_NAME_MAX."""
    message = _CONTROL.sub("", message or "").strip()
    name = _CONTROL.sub("", name or "").strip()
    if topic not in CONTACT_TOPICS:
        return None, "Pick a topic from the list."
    if not message:
        return None, "Write a message first: the box is empty."
    if len(message) > CONTACT_MAX:
        return None, (f"That message is {len(message):,} characters; the limit is "
                      f"{CONTACT_MAX:,}. Shorten it and press the button again.")
    if len(name) > CONTACT_NAME_MAX:
        return None, f"Keep the name to {CONTACT_NAME_MAX} characters or fewer."
    subject = f"Meliona: {topic} (session {ref})"
    body = message + "\n\n" + (f"From: {name}\n" if name else "") + f"Session reference: {ref}\n"
    return f"mailto:{to}?subject={quote(subject, safe='')}&body={quote(body, safe='')}", None

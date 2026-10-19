"""
Serve the launch basics a crawler or a link preview needs, before the app starts.

Streamlit renders the page with JavaScript, and serves its own static folder
from the site root. Link-preview scrapers (LinkedIn, Slack, iMessage) and
search crawlers do not run JavaScript, so anything the app sets at runtime is
invisible to them. This step, run once at container start (Procfile), does two
things to Streamlit's static root:

1. copies `public/` - favicon, apple-touch icon, Open Graph image - and writes
   robots.txt and sitemap.xml for the deployed URL;
2. adds a title, meta description, canonical URL, Open Graph / Twitter tags and
   a <noscript> description to index.html's <head>, once (idempotent);
3. paints the hero's boot skeleton straight from index.html (with app.css
   linked in the head), so a visitor sees the page's shape from the first
   frame, before Streamlit's JavaScript has started.

It never blocks startup: any failure prints a warning and exits 0, and the app
runs exactly as before.

    python -m tools.public_files            # MELEONA_SITE_URL overrides the URL
"""

import datetime
import html
import os
import pathlib
import shutil
import sys

SITE_URL = os.environ.get(
    "MELEONA_SITE_URL",
    "https://summer-2026-quant-project-risk-engine-blueprint-production.up.railway.app",
).rstrip("/")
TITLE = "Meleona - Portfolio Risk Engine"
DESCRIPTION = ("VaR, CVaR, Monte Carlo stress testing and factor exposures, computed "
               "live from Yahoo Finance data - every number traceable to its source.")
MARK, END = "<!-- meleona-head -->", "<!-- /meleona-head -->"
BOOT_MARK, BOOT_END = "<!-- meleona-boot -->", "<!-- /meleona-boot -->"
# The commit this process was built from. Railway sets RAILWAY_GIT_COMMIT_SHA
# on a GitHub deploy and posts nothing back to GitHub, so "is the fix live?"
# was guesswork (2026-09-03, 2026-09-09) until the page said so itself.
BUILD = os.environ.get("RAILWAY_GIT_COMMIT_SHA") or os.environ.get("GITHUB_SHA") or ""
PUBLIC = pathlib.Path(__file__).resolve().parent.parent / "public"


# The plain screen index.html used to paint first (until 2026-10-19), removed
# from installs that still carry it.
OLD_PRE_BOOT = (
    '<div id="pre-boot" aria-hidden="true" style="position:fixed;inset:0;z-index:998;'
    'background:#EDE9E3;animation:pre-boot-off .4s ease 12s forwards">'
    '<style>@keyframes pre-boot-off{to{opacity:0;visibility:hidden}}</style>'
    '<div style="position:absolute;bottom:28px;left:max(7vw,calc(50vw - 744px));'
    "font:11px 'Helvetica Neue',Arial,sans-serif;letter-spacing:.24em;"
    'text-transform:uppercase;color:#6A5030">Loading Meleona</div></div>\n'
)


def boot_block() -> str:
    """The hero's boot skeleton, painted from the HTML before Streamlit starts.

    The first 2.5 s (laptop) to 4.4 s (phone, 9 Mbps) of a visit happen before
    Streamlit's JavaScript can show anything the app sends; a plain beige
    screen there read as "no skeleton" (John, 2026-10-19). The skeleton is the
    hero's own markup (hero.py), styled by app.css (linked in the head), so it
    has the hero's exact shape; the real hero lifts it (main.py). The inline
    rules only keep a veil if app.css ever fails to load - words stay
    invisible - and fade it after 30 s so it can never cover a Streamlit
    connection-error message for good."""
    fallback = ('<style>#boot-skel{position:fixed;inset:0;z-index:999;background:#EDE9E3;'
                'color:transparent;animation:boot-off .55s ease 30s forwards}'
                '@keyframes boot-off{to{opacity:0;visibility:hidden}}</style>')
    try:
        from hero import boot_skeleton_html
        skel = boot_skeleton_html()
    except Exception:  # noqa: BLE001 - never block startup: a plain veil instead
        skel = '<div id="boot-skel" aria-hidden="true"></div>'
    return f"{BOOT_MARK}\n{fallback}{skel}\n{BOOT_END}\n"


def head_tags(site_url: str, build: str = "") -> str:
    t, d, u = html.escape(TITLE), html.escape(DESCRIPTION), html.escape(site_url)
    b = html.escape(build[:12] or "unknown")
    return (
        f'{MARK}\n<title>{t}</title>\n'
        f'<meta name="meleona-build" content="{b}">\n'
        f'<meta name="description" content="{d}">\n'
        f'<link rel="canonical" href="{u}/">\n'
        f'<link rel="icon" type="image/png" href="/favicon.png">\n'
        f'<link rel="apple-touch-icon" href="/apple-touch-icon.png">\n'
        f'<meta name="theme-color" content="#EDE9E3">\n'
        # The boot skeleton below the <body> tag is styled by this sheet, so it
        # must be in place for the first paint (served by enableStaticServing).
        '<link rel="stylesheet" href="./app/static/app.css">\n'
        f'<meta property="og:type" content="website">\n'
        f'<meta property="og:title" content="{t}">\n'
        f'<meta property="og:description" content="{d}">\n'
        f'<meta property="og:url" content="{u}/">\n'
        f'<meta property="og:image" content="{u}/og-image.png">\n'
        f'<meta property="og:image:width" content="1200">\n'
        f'<meta property="og:image:height" content="627">\n'
        f'<meta property="og:image:alt" content="Meleona Portfolio Risk Engine: '
        f'Monte Carlo fan chart, simulated paths and live-data audit result">\n'
        f'<meta name="twitter:card" content="summary_large_image">\n'
        # Single-page app: every unknown path is served this same index.html,
        # whose relative script paths break below the root - a mistyped
        # /a/b link was a blank white page. Send it to the branded 404 instead.
        "<script>if(!/^\\/(index\\.html)?$/.test(location.pathname))"
        "location.replace('/404.html')</script>\n"
        f"{END}\n"
    )


def install(static_dir: pathlib.Path, site_url: str = SITE_URL,
            build: str | None = None) -> list[str]:
    """Copy public files and patch index.html in `static_dir`. Returns what changed."""
    done = []
    for f in sorted(PUBLIC.glob("*")):
        shutil.copyfile(f, static_dir / f.name)
        done.append(f.name)
    (static_dir / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\n\nSitemap: {site_url}/sitemap.xml\n", encoding="utf-8")
    today = datetime.date.today().isoformat()
    (static_dir / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{html.escape(site_url)}/</loc><lastmod>{today}</lastmod></url>\n"
        "</urlset>\n", encoding="utf-8")
    done += ["robots.txt", "sitemap.xml"]

    index = static_dir / "index.html"
    page = index.read_text(encoding="utf-8")
    block = head_tags(site_url, BUILD if build is None else build)
    if MARK in page:                       # replace our block in place
        i, j = page.index(MARK), page.index(END) + len(END) + 1
        new = page[:i] + block + page[j:]
    else:                                  # first run on this Streamlit build
        new = page.replace("<title>Streamlit</title>", "")
        new = new.replace("<head>", "<head>\n" + block, 1)
        new = new.replace(
            "<noscript>", f"<noscript><p>{html.escape(TITLE)} - {html.escape(DESCRIPTION)}</p>", 1)
        new = new.replace("<body>", "<body>\n" + boot_block(), 1)
    if BOOT_MARK in new:                   # the skeleton follows the hero's copy
        i, j = new.index(BOOT_MARK), new.index(BOOT_END) + len(BOOT_END) + 1
        new = new[:i] + boot_block() + new[j:]
    else:                                  # installed before the skeleton moved here
        new = new.replace(OLD_PRE_BOOT, "").replace("<body>", "<body>\n" + boot_block(), 1)
    if new != page:
        index.write_text(new, encoding="utf-8")
        done.append("index.html <head>")
    return done


def main() -> int:
    try:
        import streamlit
        static = pathlib.Path(streamlit.__file__).parent / "static"
        print("public_files:", ", ".join(install(static)) or "already installed")
    except Exception as exc:  # noqa: BLE001 - never block the app from starting
        print(f"public_files: skipped ({exc!r}); the app runs without them", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

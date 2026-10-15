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
   a <noscript> description to index.html's <head>, once (idempotent).

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
PUBLIC = pathlib.Path(__file__).resolve().parent.parent / "public"


# Painted straight from the HTML, before Streamlit's JavaScript bundle loads:
# without it the first ~3s of a cold visit were a blank page. Same beige and
# the same gutter label as the boot skeleton, one layer beneath it; app.css
# hides it the moment the app's own skeleton arrives, and it fades by itself
# after 12s so it can never cover a Streamlit connection-error message.
PRE_BOOT = (
    '<div id="pre-boot" aria-hidden="true" style="position:fixed;inset:0;z-index:998;'
    'background:#EDE9E3;animation:pre-boot-off .4s ease 12s forwards">'
    '<style>@keyframes pre-boot-off{to{opacity:0;visibility:hidden}}</style>'
    '<div style="position:absolute;bottom:28px;left:max(7vw,calc(50vw - 744px));'
    "font:11px 'Helvetica Neue',Arial,sans-serif;letter-spacing:.24em;"
    'text-transform:uppercase;color:#6A5030">Loading Meleona</div></div>\n'
)


def head_tags(site_url: str) -> str:
    t, d, u = html.escape(TITLE), html.escape(DESCRIPTION), html.escape(site_url)
    return (
        f'{MARK}\n<title>{t}</title>\n'
        f'<meta name="description" content="{d}">\n'
        f'<link rel="canonical" href="{u}/">\n'
        f'<link rel="icon" type="image/png" href="/favicon.png">\n'
        f'<link rel="apple-touch-icon" href="/apple-touch-icon.png">\n'
        f'<meta name="theme-color" content="#EDE9E3">\n'
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


def install(static_dir: pathlib.Path, site_url: str = SITE_URL) -> list[str]:
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
    block = head_tags(site_url)
    if MARK in page:                       # replace our block in place
        i, j = page.index(MARK), page.index(END) + len(END) + 1
        new = page[:i] + block + page[j:]
    else:                                  # first run on this Streamlit build
        new = page.replace("<title>Streamlit</title>", "")
        new = new.replace("<head>", "<head>\n" + block, 1)
        new = new.replace(
            "<noscript>", f"<noscript><p>{html.escape(TITLE)} - {html.escape(DESCRIPTION)}</p>", 1)
        new = new.replace("<body>", "<body>\n" + PRE_BOOT, 1)
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

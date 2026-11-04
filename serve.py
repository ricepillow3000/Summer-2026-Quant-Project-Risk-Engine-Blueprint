"""The production entry point: main.py, served with real HTTP security headers.

`streamlit run serve.py` finds the ASGI app below and serves main.py through
it - same page, same config, same server (Streamlit 1.66 already runs on
Starlette/uvicorn). What this file adds is what the page's <meta> tags cannot
do (LAUNCH_CHECKLIST #89, corrected 2026-11-04):

- Strict-Transport-Security: a browser that has seen the site once refuses to
  load it over plain HTTP for a year (Railway terminates TLS in front of us).
- X-Frame-Options + CSP frame-ancestors: no other site may frame this page and
  dress it up (clickjacking). 'self' rather than 'none', so a page of ours can
  still frame another, and the Risk Topology and glide iframes keep working.
- nosniff and the referrer policy. No Permissions-Policy: Streamlit's own
  frames ask for camera, microphone and location, and refusing them made the
  browser log a warning on every page (deep scan, 2026-11-04) while guarding
  nothing - the app calls none of them.
- /_stcore/metrics answers 404: Streamlit serves live session counts and cache
  sizes there to anyone, and nobody here reads them.

Pure ASGI rather than BaseHTTPMiddleware, so streamed responses and the
websocket pass straight through. Pinned by
test_every_response_carries_the_security_headers.
"""

from __future__ import annotations

from pathlib import Path

from starlette.middleware import Middleware
from streamlit.starlette import App

HEADERS = {
    "strict-transport-security": "max-age=31536000",
    "x-frame-options": "SAMEORIGIN",
    "content-security-policy": ("frame-ancestors 'self'; object-src 'none'; "
                                "base-uri 'self'; form-action 'self'"),
    "x-content-type-options": "nosniff",
    "referrer-policy": "strict-origin-when-cross-origin",
}
HIDDEN = ("/_stcore/metrics",)


class SecurityHeaders:
    """Adds HEADERS to every HTTP response and hides the paths in HIDDEN."""

    def __init__(self, app):
        self.app = app
        self.raw = [(k.encode(), v.encode()) for k, v in HEADERS.items()]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["path"].startswith(HIDDEN):
            await send({"type": "http.response.start", "status": 404,
                        "headers": [(b"content-type", b"text/plain"), *self.raw]})
            await send({"type": "http.response.body", "body": b"Not found"})
            return

        async def with_headers(message):
            if message["type"] == "http.response.start":
                names = {k for k, _ in self.raw}
                kept = [(k, v) for k, v in message.get("headers", []) if k.lower() not in names]
                message = {**message, "headers": kept + self.raw}
            await send(message)

        await self.app(scope, receive, with_headers)


app = App(Path(__file__).with_name("main.py"), middleware=[Middleware(SecurityHeaders)])

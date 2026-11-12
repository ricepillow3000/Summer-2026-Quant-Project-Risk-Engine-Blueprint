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

RateLimit (2026-11-04, John's pre-public list) caps what one network address
can do: new sessions a minute, sessions open at once, HTTP requests a minute,
and (2026-11-10) messages a minute over its open sessions - each click is a
message that reruns the page, and nothing capped them inside an open session. A visitor never comes near the caps - one page is one session and
about forty requests - so they only stop a script flooding the one process
every visitor shares. Over a cap: HTTP 429, or the websocket closed with
1013 ("try again later"). The address is read from the last hop Railway's
proxy appended to X-Forwarded-For (an address the client typed sits to its
left), counted in memory for a minute and never written down; a refusal is
logged with a scrambled form of it (PRIVACY.md). The health check is never
limited, so Railway's own probe cannot be refused. Server errors (5xx) are
logged too. Pinned by test_one_address_cannot_flood_the_shared_process.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import secrets
import sys
import time
from collections import defaultdict, deque
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
# metrics: live session counts, open to anyone by default. upload_file: the app
# has no uploader, but streamlit still takes files there, held in memory until
# the session ends, behind an xsrf check a script passes by sending the same
# made-up token as cookie and header (security review, 2026-11-12).
HIDDEN = ("/_stcore/metrics", "/_stcore/upload_file")


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


# Per address. A page load is one session and about forty requests, so a
# person - or a whole office behind one address - stays far below these.
NEW_SESSIONS_PER_MINUTE = 60
OPEN_SESSIONS = 30
REQUESTS_PER_MINUTE = 3000
MESSAGES_PER_MINUTE = 600      # one address's websocket messages; a person sends a few
MAX_MESSAGE_BYTES = 256 * 1024  # one inbound message; a click's widget state is a few KB
NEVER_LIMITED = ("/_stcore/health",)   # exact paths
WINDOW = 60.0
MAX_TRACKED = 20_000            # addresses held at once; the quietest go first

log = logging.getLogger("meliona.ratelimit")
if not log.handlers:            # serve.py runs before main.py's setup_logging
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    _h.setLevel(logging.WARNING)
    log.addHandler(_h)
    log.setLevel(logging.WARNING)
    log.propagate = False
_SALT = secrets.token_bytes(16)  # new every start: a logged tag cannot be reversed


def scrambled(addr: str) -> str:
    """A short tag for an address: the same within one run, never the address."""
    return hashlib.blake2s(addr.encode(), key=_SALT, digest_size=4).hexdigest()


def client_address(scope) -> str:
    """the last hop the proxy appended - read across every x-forwarded-for line,
    so a second line the client sent can't pick the address. an ipv6 /64 is one
    customer's block, so it counts as one address (else 2^64 fresh ones)."""
    hops = [h.strip() for name, value in scope.get("headers", []) if name == b"x-forwarded-for"
            for h in value.decode("latin-1").split(",") if h.strip()]
    client = scope.get("client")
    addr = hops[-1] if hops else (client[0] if client else "unknown")
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return addr
    return str(ipaddress.ip_network(f"{ip}/64", strict=False)) if ip.version == 6 else addr


class RateLimit:
    """Caps new sessions a minute, open sessions and requests a minute, per
    address. Refusals are 429 (HTTP) or close code 1013 (websocket)."""

    def __init__(self, app, clock=time.monotonic):
        self.app, self.clock = app, clock
        self.sessions = defaultdict(deque)   # address -> start times of sessions
        self.requests = defaultdict(deque)   # address -> times of HTTP requests
        self.messages = defaultdict(deque)   # address -> times of websocket messages
        self.open = defaultdict(int)          # address -> sessions open now
        self.warned = {}                      # address -> last refusal logged
        self.tidied = float("-inf")           # last time the tables were trimmed

    def _within(self, seen: deque, now: float, cap: int) -> bool:
        while seen and now - seen[0] > WINDOW:
            seen.popleft()
        if len(seen) >= cap:
            return False
        seen.append(now)
        return True

    def _refused(self, addr: str, what: str, now: float) -> None:
        if now - self.warned.get(addr, -WINDOW) >= WINDOW:     # once a minute
            self.warned[addr] = now
            log.warning("refused %s from client=%s", what, scrambled(addr))

    def _tidy(self) -> None:
        """keep the tables bounded: drop addresses idle a minute, then, if a
        flood of fresh addresses still overfills them, the quietest. at most
        once a second, so a big table isn't scanned on every request."""
        now = self.clock()
        tracked = len(self.requests) + len(self.sessions) + len(self.messages)
        if tracked <= MAX_TRACKED or now - self.tidied < 1.0:
            return
        self.tidied = now
        for table in (self.requests, self.sessions, self.messages):
            for addr in [a for a, q in table.items() if not q or now - q[-1] > WINDOW]:
                del table[addr]
            extra = len(table) - MAX_TRACKED // 3
            if extra > 0:
                for addr in sorted(table, key=lambda a: table[a][-1])[:extra]:
                    del table[addr]
        for addr in [a for a, n in self.open.items() if n <= 0]:
            del self.open[addr]
        for addr in [a for a, t in self.warned.items() if now - t > WINDOW]:
            del self.warned[addr]

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope["path"] in NEVER_LIMITED:
            await self.app(scope, receive, send)
            return
        addr, now = client_address(scope), self.clock()
        self._tidy()
        if scope["type"] == "http":
            if not self._within(self.requests[addr], now, REQUESTS_PER_MINUTE):
                self._refused(addr, "requests", now)
                await send({"type": "http.response.start", "status": 429,
                            "headers": [(b"content-type", b"text/plain"), (b"retry-after", b"60")]})
                await send({"type": "http.response.body", "body": b"Too many requests - try again in a minute."})
                return

            async def watch(message):
                if message["type"] == "http.response.start" and message["status"] >= 500:
                    log.error("HTTP %s on %s", message["status"], scope["path"])
                await send(message)

            await self.app(scope, receive, watch)
            return
        if self.open[addr] >= OPEN_SESSIONS or not self._within(
                self.sessions[addr], now, NEW_SESSIONS_PER_MINUTE):
            self._refused(addr, "a session", now)
            await send({"type": "websocket.close", "code": 1013})
            return
        self.open[addr] += 1
        closed = []

        async def counted():
            # past the cap: hang up (1013 "try again later") and hand the app a
            # disconnect, so the session ends the normal way
            # one too big gets 1009 ("too big"); compression means a small wire
            # frame can unpack to megabytes of widget state to chew through
            message = await receive()
            if message["type"] != "websocket.receive":
                return message
            size = len(message.get("bytes") or b"") + len(message.get("text") or "")
            code = 1009 if size > MAX_MESSAGE_BYTES else 1013
            if code == 1009 or not self._within(self.messages[addr], self.clock(), MESSAGES_PER_MINUTE):
                self._refused(addr, "messages", self.clock())
                if not closed:
                    closed.append(True)
                    await send({"type": "websocket.close", "code": code})
                return {"type": "websocket.disconnect", "code": code}
            return message

        async def quiet(message):                   # nothing goes out after the close
            if not closed:
                await send(message)

        try:
            await self.app(scope, counted, quiet)
        finally:
            self.open[addr] -= 1


app = App(Path(__file__).with_name("main.py"),
          middleware=[Middleware(SecurityHeaders), Middleware(RateLimit)])

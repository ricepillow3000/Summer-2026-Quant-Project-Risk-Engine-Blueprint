"""
Egress allowlist for every outbound request this engine makes.

Why this exists (the SSRF control, adapted to an app with no webhooks):
the classic SSRF sink is a URL the user supplies. This app has none - the only
user input is a ticker symbol, and `ingestion.VALID_TICKER` already restricts
it to `[A-Z0-9.\\-=]` at a single funnel, so it cannot carry a scheme, `//`,
`@`, or a path traversal. The host is chosen by `yfinance`, not by the user.

That makes the funnel the control and this module the second line: even if a
future feature takes a URL, a dependency is compromised, or a Yahoo endpoint
redirects somewhere unexpected, the request does not leave the allowlist. It
fails closed and names the host it refused.

Design notes:
- The session subclasses `curl_cffi.requests.Session` because that is what
  `yfinance` itself uses. A plain `requests.Session` is accepted by yfinance
  but draws `YFRateLimitError` immediately (Yahoo rejects the non-impersonated
  TLS fingerprint) - verified 2026-08-26, so swapping transports to add a
  security control would have broken the data path instead of hardening it.
- Both the requested URL and the FINAL url are checked, so a redirect chain
  cannot walk out of the allowlist.
- Scheme must be https: no http, no file://, no gopher://.
"""

from urllib.parse import urlparse

from curl_cffi import requests as _curl

# Hosts this engine is allowed to talk to. Yahoo serves quotes from
# query1/query2 and corporate actions from the main finance host; the rest are
# the endpoints yfinance uses for its cookie/crumb handshake.
ALLOWED_HOSTS = frozenset({
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",
    "finance.yahoo.com",
    "fc.yahoo.com",
    "guce.yahoo.com",
    "login.yahoo.com",
    # yfinance's `Ticker.isin` does NOT use Yahoo - it queries Business
    # Insider's symbol search. Found by this allowlist blocking it: every ISIN
    # silently came back "unavailable" (2026-08-26). Enumerated deliberately,
    # which is the point of an allowlist: the second host this app talks to is
    # now a reviewed decision instead of an unnoticed dependency.
    "markets.businessinsider.com",
})

# Browser fingerprint yfinance itself impersonates. Kept here so the guarded
# session behaves exactly like the unguarded one it replaces.
IMPERSONATE = "chrome"


class EgressBlocked(RuntimeError):
    """Raised when something tried to reach a host outside the allowlist."""


def host_allowed(url: str) -> bool:
    """True if `url` is https and its host is on the allowlist."""
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "") in ALLOWED_HOSTS


class _GuardedSession(_curl.Session):
    """A curl_cffi session that refuses to leave the allowlist."""

    def request(self, method, url, *args, **kwargs):
        if not host_allowed(url):
            raise EgressBlocked(
                f"blocked outbound {method} to {urlparse(url).hostname!r} - "
                "not on the market-data allowlist")
        response = super().request(method, url, *args, **kwargs)
        final = getattr(response, "url", url)
        if not host_allowed(str(final)):
            raise EgressBlocked(
                f"request to {urlparse(url).hostname!r} redirected to "
                f"{urlparse(str(final)).hostname!r}, outside the allowlist")
        return response


def guarded_session() -> _GuardedSession:
    """The session every market-data call in this engine goes through."""
    return _GuardedSession(impersonate=IMPERSONATE)


if __name__ == "__main__":
    assert host_allowed("https://query1.finance.yahoo.com/v8/finance/chart/SPY")
    assert not host_allowed("http://query1.finance.yahoo.com/x")      # plain http
    assert not host_allowed("https://169.254.169.254/latest/meta-data/")
    assert not host_allowed("https://evil.example.com/?x=1")
    assert not host_allowed("https://query1.finance.yahoo.com.evil.com/x")

    s = guarded_session()
    for bad in ("https://169.254.169.254/latest/meta-data/",
                "https://evil.example.com/"):
        try:
            s.get(bad)
        except EgressBlocked as exc:
            print(f"blocked: {exc}")
        else:                                  # pragma: no cover - would be a bug
            raise SystemExit(f"NOT BLOCKED: {bad}")
    print("egress allowlist holds")

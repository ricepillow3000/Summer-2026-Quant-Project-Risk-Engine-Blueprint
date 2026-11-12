"""
egress allowlist: every outbound request this engine makes goes through here.

the ssrf story: there's no user-supplied url anywhere. the only user input is a
ticker, and `ingestion.VALID_TICKER` already squeezes that to `[A-Z0-9.\\-=]`,
so it can't carry a scheme, `//`, `@` or a path. yfinance picks the host, not
the user. so the funnel is control #1 and this is the backup: if a future
feature takes a url, a dependency gets owned, or yahoo redirects somewhere
weird, the request still can't leave the list. fails closed, names the host.

notes:
- we subclass curl_cffi's Session because that's what yfinance uses. a plain
  requests.Session works but yahoo rate-limits it instantly (wrong tls
  fingerprint), so "safer" transport = broken data.
- redirects are followed here, one hop at a time, and each hop is checked
  BEFORE it's contacted - curl following them itself would reach a host first
  and only get refused after.
- https only. no http, file://, gopher://.
"""

import threading
import time
from urllib.parse import urljoin, urlparse

from curl_cffi import requests as _curl

# yahoo quotes come from query1/query2, corporate actions from finance., the
# rest is yfinance's cookie/crumb handshake. add a feed = add its host here.
ALLOWED_HOSTS = frozenset({
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",
    "finance.yahoo.com",
    "fc.yahoo.com",
    "guce.yahoo.com",
    "login.yahoo.com",
    # markets.businessinsider.com is gone with the isin lookup: a host nothing
    # uses is just an open door. add it back on purpose if isins return.
})

IMPERSONATE = "chrome"   # same browser fingerprint yfinance uses on its own
# yfinance's handshake sets no timeout, so a dead connect would hang a page forever
TIMEOUT_S = 30
MAX_HOPS = 5             # redirects followed per request
REDIRECTS = (301, 302, 303, 307, 308)


# --- outbound budget --------------------------------------------------------
# the spend cap. yahoo is free, so nothing bills us, but it rate-limits and then
# bans by ip - on shared free-tier egress that's the outage we can cause
# ourselves. so the cap is requests/day, checked at the same chokepoint, and
# running out falls back to cached data instead of hammering the feed.
MAX_REQUESTS_PER_DAY = 5000
_SPENT: dict[str, float] = {"day": 0.0, "count": 0.0}
_LOCK = threading.Lock()   # page scripts run on threads; the count must not drift


class EgressBlocked(RuntimeError):
    """something tried to reach a host that's not on the allowlist."""


class BudgetExhausted(RuntimeError):
    """today's outbound request budget is spent."""


def _day_number() -> float:
    return float(int(time.time() // 86400))


def budget_status() -> dict:
    """requests spent today and what's left (the provenance panel shows it)."""
    if _SPENT["day"] != _day_number():
        return {"spent": 0, "remaining": MAX_REQUESTS_PER_DAY,
                "cap": MAX_REQUESTS_PER_DAY}
    spent = int(_SPENT["count"])
    return {"spent": spent, "remaining": max(0, MAX_REQUESTS_PER_DAY - spent),
            "cap": MAX_REQUESTS_PER_DAY}


def check_budget(needed: int = 1) -> None:
    """
    raise BudgetExhausted BEFORE a fetch that has no budget left.

    the per-request charge below also guards every call, but yfinance swallows
    transport errors and calls them "possibly delisted", so without this up-front
    check a spent cap would look like a weird data error.
    """
    status = budget_status()
    if status["remaining"] < needed:
        raise BudgetExhausted(
            f"daily outbound budget of {status['cap']} market-data requests is "
            "spent; serving cached data until it resets at UTC midnight")


def _charge_budget() -> None:
    """count one outbound request, resets at utc midnight."""
    with _LOCK:
        today = _day_number()
        if _SPENT["day"] != today:
            _SPENT["day"], _SPENT["count"] = today, 0.0
        if _SPENT["count"] >= MAX_REQUESTS_PER_DAY:
            raise BudgetExhausted(
                f"daily outbound budget of {MAX_REQUESTS_PER_DAY} market-data "
                "requests is spent; serving cached data until it resets")
        _SPENT["count"] += 1.0


def host_allowed(url: str) -> bool:
    """true if `url` is https and its host is on the allowlist."""
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "") in ALLOWED_HOSTS


class _GuardedSession(_curl.Session):
    """a curl_cffi session that won't leave the allowlist."""

    def request(self, method, url, *args, **kwargs):
        follow = kwargs.pop("allow_redirects", True)
        for _ in range(MAX_HOPS + 1):
            if not host_allowed(url):       # stops ssrf, and a redirect walking out
                raise EgressBlocked(
                    f"blocked outbound {method} to {urlparse(url).hostname!r} - "
                    "not on the market-data allowlist")
            _charge_budget()                # stops us getting ip-banned by yahoo
            response = super().request(method, url, *args, allow_redirects=False, **kwargs)
            location = response.headers.get("location")
            if not follow or response.status_code not in REDIRECTS or not location:
                return response
            url = urljoin(url, location)
            if response.status_code in (301, 302, 303) and method.upper() != "HEAD":
                method = "GET"              # what browsers (and curl) do
                kwargs.pop("data", None)
                kwargs.pop("json", None)
        raise EgressBlocked(f"more than {MAX_HOPS} redirects")


def guarded_session() -> _GuardedSession:
    """the session every market-data call goes through."""
    return _GuardedSession(impersonate=IMPERSONATE, timeout=TIMEOUT_S)


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
    print("budget:", budget_status())

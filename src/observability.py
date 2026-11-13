"""
crash reporting for a single-process public app, no vendor sdk.

a hosted crash reporter gives you a stack trace, which user hit it, and a way
to find it later. this does the same 3 things with the stdlib. no sdk on
purpose: shipping user data to a vendor is exactly what the privacy note says
we don't do. no account, no device id, no beacon - so crashes stay local too.

- `setup_logging()` puts a size-capped rotating file on the ROOT logger, so it
  catches our own lines and the tracebacks streamlit already logs. nothing is
  logged twice.
- every browser session gets a random `session_ref`, tied to nobody, shown in
  the footer. a user emails it to support, we grep for it. no cookie needed.
- `log_incident()` logs a handled failure under that ref, so a degraded
  feature leaves a trail even when the visitor just saw a friendly message.

logs are capped by size and count, hold tickers and tracebacks (no personal
data) and rotate away on their own.

usage counts work the same way: two in-memory counters (sessions today, utc,
and how many sessions opened each preset), written as ONE summary line to the
host log at most once an hour, at the first event after utc midnight (the
finished day) and on exit. no sdk, cookie, session ref or ip, and preset names
only from the app's closed list so a typed ticker can never get counted.
PRIVACY.md "Analytics" says exactly this.
"""

import atexit
import logging
import os
import secrets
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable, Container
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "meliona.log")

# small on purpose: a debug tail, not an archive, on a tiny disk the cache shares
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_configured = False


def host_setting(name: str, default: str = "") -> str:
    """the host's MELIONA_<name> variable. the old MELEONA_ spelling still
    counts, so the rename can't quietly turn off the kill switch."""
    return os.getenv(f"MELIONA_{name}") or os.getenv(f"MELEONA_{name}") or default


def setup_logging(level: int = logging.INFO) -> str:
    """
    attach the rotating file (plus errors-only to the host's log) once per
    process. returns the log path.

    streamlit reruns the whole script on every click, so without the guard
    we'd add a handler each time, multiply every line and rotate the file
    away in minutes.
    """
    global _configured
    if _configured:
        return LOG_FILE
    os.makedirs(LOG_DIR, exist_ok=True)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES,
                                  backupCount=BACKUP_COUNT, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(level)
    root = logging.getLogger()
    root.addHandler(handler)
    # errors also go to the host log: every redeploy wipes logs/, and the
    # runbook needs those tracebacks. errors only - session starts stay on disk.
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_FORMAT))
    stream.setLevel(logging.ERROR)
    root.addHandler(stream)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    _configured = True
    return LOG_FILE


def new_session_ref() -> str:
    """a short random handle for one browser session, tied to nobody."""
    return secrets.token_hex(4).upper()


def log_session_start(session_ref: str, detail: str = "") -> None:
    logging.getLogger("meliona.session").info(
        "ref=%s session start %s", session_ref, detail)


def log_incident(session_ref: str, where: str, exc: BaseException) -> str:
    """
    log a handled failure under a session ref. returns the ref so the ui can
    say "something broke, quote REF when you email support".
    """
    logging.getLogger("meliona.incident").error(
        "ref=%s where=%s %s: %s\n%s", session_ref, where,
        type(exc).__name__, exc,
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return session_ref


def log_no_data(session_ref: str, where: str) -> None:
    """a typed basket yahoo had nothing for: a typo, not an incident. one INFO
    line, no traceback, so a script rerunning it can't bury real errors."""
    logging.getLogger("meliona.session").info("ref=%s where=%s no data", session_ref, where)


USAGE_EVERY_S = 3600       # one summary line an hour, max

# the usage line gets its own logger: INFO straight to the host's stderr, not
# the root's handlers (logs/ is wiped on redeploy, the root stream is errors only)
class _HostStream(logging.StreamHandler):
    """whatever stderr is when the line is written, not when we were made:
    the exit flush runs after a test runner already closed the one it lent."""

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, _value):
        pass


usage_log = logging.getLogger("meliona.usage")
if not usage_log.handlers:
    _uh = _HostStream()
    _uh.setFormatter(logging.Formatter(_FORMAT))
    _uh.setLevel(logging.INFO)
    usage_log.addHandler(_uh)
    usage_log.setLevel(logging.INFO)
    usage_log.propagate = False


def _utc_day(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).date().isoformat()


class UsageCounter:
    """
    today's anonymous totals, in memory: sessions started and preset picks.

    nothing here identifies anyone. a session is just +1 (its ref never comes
    here) and a preset only counts if it's in the closed list `known`, so a
    typed ticker or a forged url param gets dropped. streamlit runs each
    session on its own thread, hence the lock.

    no timer thread: we read the clock on each event. an event an hour+ after
    the last line writes the running totals, the first event of a new utc day
    writes the finished day and resets, and `flush()` (at exit) writes the rest.
    """

    def __init__(self, clock: Callable[[], float] = time.time,
                 write: Callable[[str], None] | None = None,
                 every_s: float = USAGE_EVERY_S):
        self._clock, self._every = clock, every_s
        self._write = write or usage_log.info
        self._lock = threading.Lock()
        self._day: str | None = None
        self._sessions = 0
        self._presets: Counter = Counter()
        self._last = 0.0
        self._dirty = False

    def session(self) -> None:
        """one browser session started."""
        self._count(None)

    def preset(self, name: str, known: Container[str]) -> None:
        """a session opened this preset. only a name in `known` counts."""
        if isinstance(name, str) and name in known:
            self._count(name)

    def line(self) -> str:
        with self._lock:
            return self._line()

    def flush(self) -> None:
        """write the running totals if anything got counted since the last line."""
        with self._lock:
            out = self._line() if self._dirty else None
            self._dirty = False
            self._last = self._clock()
        if out:
            self._write(out)

    def _line(self) -> str:
        top = sorted(self._presets.items(), key=lambda kv: (-kv[1], kv[0]))
        presets = ", ".join(f"{k} {v}" for k, v in top) or "none"
        n = self._sessions
        return f"usage {self._day}: {n} session{'' if n == 1 else 's'}; presets: {presets}"

    def _count(self, preset: str | None) -> None:
        now = self._clock()
        day = _utc_day(now)
        out = []
        with self._lock:
            if self._day is None:
                self._day, self._last = day, now
            elif day != self._day:
                if self._dirty:
                    out.append(self._line())          # the finished day's total
                self._day, self._sessions, self._presets = day, 0, Counter()
                self._last = now
            if preset is None:
                self._sessions += 1
            else:
                self._presets[preset] += 1
            self._dirty = True
            if now - self._last >= self._every:
                out.append(self._line())
                self._last, self._dirty = now, False
        for ln in out:
            self._write(ln)


usage = UsageCounter()
atexit.register(usage.flush)         # a redeploy keeps its last hour (only if something was counted)


def count_session() -> None:
    """main.py, where a session's ref is made: one session started."""
    usage.session()


def count_preset(name: str, known: Container[str]) -> None:
    """main.py, once per session per preset it opens."""
    usage.preset(name, known)


def recent_incidents(limit: int = 20) -> list[str]:
    """
    operator helper: the last incident lines, newest last. the runbook's
    triage step uses it, and so does the test that proves the wire works.
    """
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh if "meliona.incident" in ln]
    return lines[-limit:]


if __name__ == "__main__":
    path = setup_logging()
    ref = new_session_ref()
    log_session_start(ref, "smoke test")
    try:
        raise ValueError("synthetic failure for the crash wire")
    except ValueError as exc:
        log_incident(ref, "__main__ smoke test", exc)
    found = [ln for ln in recent_incidents() if ref in ln]
    assert found, "incident did not reach the log"
    lines: list[str] = []
    t = [datetime(2026, 11, 4, 23, 30, tzinfo=timezone.utc).timestamp()]
    u = UsageCounter(clock=lambda: t[0], write=lines.append)
    u.session(); u.preset("Mag-7", {"Mag-7"}); u.preset("NVDA", {"Mag-7"})
    t[0] += 3600
    u.session()
    assert lines == ["usage 2026-11-04: 1 session; presets: Mag-7 1"], lines
    print("usage line:", lines[-1])
    print(f"crash wire live: {path}")
    print(found[-1][:120])

"""
Crash reporting for a single-process public app - the beta-test wire.

What a hosted crash reporter (Sentry, Crashlytics) buys you is: a stack trace,
a way to tell WHICH user hit it, and a way to find it again later. This does
the same three things with the standard library and no third-party SDK, which
matters here for a specific reason: an SDK that ships user data to a vendor is
exactly what the privacy note promises this app does not do. No account, no
device id, no analytics beacon - so the crash wire stays local too.

How it works:
- `setup_logging()` attaches a size-capped rotating file handler to the ROOT
  logger, so it captures this engine's own log lines AND the tracebacks
  Streamlit already logs when a script run raises. Nothing is instrumented
  twice; the exception path that already exists just lands in a file.
- Every browser session gets a short `session_ref`. It is random, tied to no
  identity, and printed in the page footer. A user emailing support quotes it;
  the operator greps for it and gets that session's lines including the
  traceback. That is the whole correlation story - no cookie, no fingerprint.
- `log_incident()` records a handled failure with the same ref, so degraded
  features (a feed timeout, a refused calibration) leave a trail even when the
  page recovered and the visitor saw a friendly message.

Retention: files are capped by size and count below. They hold ticker symbols
and stack traces - no personal data - and rotate away on their own.

Usage counts (2026-11-04) are the analytics, built the same way: two
in-process counters - browser sessions started today (UTC) and how many
sessions opened each preset basket - written as ONE aggregate line to the
host's log once an hour, at the first event after midnight UTC (the finished
day's total) and when the process exits. No SDK, no cookie, no session
reference, no IP, and preset names only from the app's closed list, so a
typed ticker can never be counted. PRIVACY.md "Analytics" says exactly this.
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

# Small on purpose: this is a diagnostic tail, not an archive, and the host is
# an ephemeral free-tier disk that the cache budget already shares.
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_configured = False


def host_setting(name: str, default: str = "") -> str:
    """The host's MELIONA_<name> variable. The name was spelled Meleona until
    2026-10-28; a variable set on the host under that spelling still counts, so
    the rename cannot quietly disarm the kill switch."""
    return os.getenv(f"MELIONA_{name}") or os.getenv(f"MELEONA_{name}") or default


def setup_logging(level: int = logging.INFO) -> str:
    """
    Attach the rotating file handler (and an errors-only stream to the
    host's log) once per process. Returns the log path.

    Safe to call on every Streamlit rerun - the second call is a no-op, which
    is why the guard exists: Streamlit re-executes the whole script on every
    widget interaction, and re-adding a handler each time would multiply every
    line and rotate the file away in minutes.
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
    # Errors also go to the host's log stream (2026-11-04 security review):
    # the disk under logs/ is wiped by every redeploy, which took the
    # tracebacks RUNBOOK §3 triages with it. Errors only - session starts stay
    # on disk - and Streamlit already writes its own server errors there.
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_FORMAT))
    stream.setLevel(logging.ERROR)
    root.addHandler(stream)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    _configured = True
    return LOG_FILE


def new_session_ref() -> str:
    """A short, random, identity-free handle for one browser session."""
    return secrets.token_hex(4).upper()


def log_session_start(session_ref: str, detail: str = "") -> None:
    logging.getLogger("meliona.session").info(
        "ref=%s session start %s", session_ref, detail)


def log_incident(session_ref: str, where: str, exc: BaseException) -> str:
    """
    Record a handled failure against a session ref. Returns the ref so the UI
    can show it: "something broke, quote REF when you email support".
    """
    logging.getLogger("meliona.incident").error(
        "ref=%s where=%s %s: %s\n%s", session_ref, where,
        type(exc).__name__, exc,
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return session_ref


USAGE_EVERY_S = 3600       # one aggregate line an hour at most

# The usage line's own logger: INFO to the host's stream (stderr), like
# serve.py's rate-limit logger, and not into the root's handlers - the file
# under logs/ is wiped by every redeploy and the root's stream takes errors only.
class _HostStream(logging.StreamHandler):
    """stderr as it is when a line is written, not when the handler was made:
    the exit flush runs after a test runner has closed the stream it lent."""

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
    Today's anonymous totals, in memory: sessions started and preset picks.

    Nothing identifies anyone. A session is a +1 (its reference is never
    seen here) and a preset is a name from a closed list (`known`); anything
    else - a typed ticker, a forged address parameter - is dropped. Streamlit
    serves each session on its own thread, so every change holds a lock.

    There is no timer thread: the clock is read on each event. An event an
    hour or more after the last line writes the running totals; the first
    event of a new UTC day writes the finished day's totals and starts again
    at zero; `flush()` (registered at exit) writes what has not been written.
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
        """One browser session started."""
        self._count(None)

    def preset(self, name: str, known: Container[str]) -> None:
        """A session opened this preset basket. Only a name in `known` counts."""
        if isinstance(name, str) and name in known:
            self._count(name)

    def line(self) -> str:
        with self._lock:
            return self._line()

    def flush(self) -> None:
        """Write the running totals if anything was counted since the last line."""
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
atexit.register(usage.flush)         # a redeploy keeps its last hour (writes only if counted)


def count_session() -> None:
    """main.py, where a session's reference is minted: one session started."""
    usage.session()


def count_preset(name: str, known: Container[str]) -> None:
    """main.py, once per session per preset basket it opens."""
    usage.preset(name, known)


def recent_incidents(limit: int = 20) -> list[str]:
    """
    Operator helper: the last incident lines, newest last. Used by the
    runbook's triage step and by the test that proves the wire is live.
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

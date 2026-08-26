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
"""

import logging
import os
import secrets
import traceback
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "meleona.log")

# Small on purpose: this is a diagnostic tail, not an archive, and the host is
# an ephemeral free-tier disk that the cache budget already shares.
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_configured = False


def setup_logging(level: int = logging.INFO) -> str:
    """
    Attach the rotating file handler once per process. Returns the log path.

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
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    _configured = True
    return LOG_FILE


def new_session_ref() -> str:
    """A short, random, identity-free handle for one browser session."""
    return secrets.token_hex(4).upper()


def log_session_start(session_ref: str, detail: str = "") -> None:
    logging.getLogger("meleona.session").info(
        "ref=%s session start %s", session_ref, detail)


def log_incident(session_ref: str, where: str, exc: BaseException) -> str:
    """
    Record a handled failure against a session ref. Returns the ref so the UI
    can show it: "something broke, quote REF when you email support".
    """
    logging.getLogger("meleona.incident").error(
        "ref=%s where=%s %s: %s\n%s", session_ref, where,
        type(exc).__name__, exc,
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return session_ref


def recent_incidents(limit: int = 20) -> list[str]:
    """
    Operator helper: the last incident lines, newest last. Used by the
    runbook's triage step and by the test that proves the wire is live.
    """
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh if "meleona.incident" in ln]
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
    print(f"crash wire live: {path}")
    print(found[-1][:120])

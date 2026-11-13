# Support

**Email: meleona.support@gmail.com** · **Phone: [770-265-9930](tel:+17702659930)**

**Response time: John aims to reply within 2 business days.** This is a student
portfolio project run by one person, so that is a promise of effort, not a
service-level guarantee.

The quickest way to write is the **Contact** form at the foot of the page: pick
a topic (Question, Feedback, Recruiting or Report a problem), write up to 1,000
characters, and it opens an email in your own email app with your words and
your session reference already filled in. The site has no mail server, so you
send it yourself - nothing leaves until you press send. Email is best for a
problem report, because the session reference below travels with it.

## Reporting a problem

Include these four things and the problem is usually diagnosable without a
follow-up:

1. **The session reference** from the page footer (e.g. `9F3A2B7C`). It maps
   straight to that session's log lines, including any stack trace.
2. **What you clicked** - which preset or tickers, which tab, which controls.
3. **What you expected vs what happened** - a screenshot is ideal.
4. **Roughly when** (with your timezone), so the log window is easy to find.

## Expected behaviour, not bugs

- **"Ignored `XYZ` - not a valid Yahoo Finance symbol."** The ticker box only
  accepts Yahoo-shaped symbols (`BRK-B`, `EURUSD=X`, `GC=F`, `^IRX`).
- **A cap at 25 symbols.** A deliberate compute budget on a public single
  process, disclosed in the warning that fires.
- **"Excludes ..." in a scenario verdict.** Those assets did not trade in that
  historical window, so they are dropped and named rather than back-filled.
- **ISIN "unavailable".** The free feed does not publish one for every symbol.
  It is flagged, never guessed.
- **Data a few hours old.** The app serves live *end-of-day* data behind a
  freshness-aware cache; it is not, and never claims to be, real-time.
- **A model saying it refused to fit.** Several (EVT tail fit, state
  calibration) decline on thin data instead of returning a number they cannot
  support.

## Security reports

Email the same address with `SECURITY` in the subject: what the issue is and
how to reproduce it, with a reasonable window to fix it before publishing.
There is no bounty - this is a student project - but credit goes in the commit
that fixes it.

## Privacy requests

See `PRIVACY.md`. There is no account to delete. To have log lines carrying
your session reference purged early, email that reference.

## About this address

`meleona.support@gmail.com` is a project mailbox, not a personal one: it is
published on a public page, so it will attract spam, and a dedicated address
can be filtered, forwarded or handed on without touching a personal inbox.
Mail sent to it is **forwarded to the maintainer's personal inbox**, so there
is one place to watch.

The address lives in a single constant (`SUPPORT_EMAIL` in `main.py`) and can
be overridden per deployment with the `MELIONA_SUPPORT_EMAIL` environment
variable - no redeploy needed to change where support mail goes.

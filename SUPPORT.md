# Support

**Email: john4000.nguyen@gmail.com**

This is a student portfolio project run by one person. Support is best-effort:
expect a reply in days, not minutes, and no service-level guarantee.

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

## A note on this address

`john4000.nguyen@gmail.com` is a personal address published on a public page,
so it will attract spam and it ties the project to a personal inbox forever.
Two cheap alternatives, either of which keeps every link in the app working
(the address lives in ONE constant, `SUPPORT_EMAIL` in `main.py`):

- a Gmail **alias** - `john4000.nguyen+meleona@gmail.com` filters cleanly into
  its own label, but is trivially strippable by a spammer;
- a dedicated free address such as `meleona.support@gmail.com`, forwarded to
  the personal inbox - a clean break, and the one to pick if the project keeps
  going.

Changing it is a one-line edit; the footer, maintenance page and privacy panel
all follow.

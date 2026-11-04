# Privacy

_Last updated: 2026-11-02. Contact: meleona.support@gmail.com_

Meliona is run by John Nguyen, an individual, as a student portfolio project -
not a company, and nothing here is sold.

Meliona is a public, read-only market-risk dashboard. It has **no sign-up, no
login, no password, and no user accounts**. This note describes everything the
app holds, which is deliberately close to nothing.

## What is not collected

- No name, email, address, phone number or payment details. There is no form
  that asks for any of them.
- No analytics, advertising, attribution or crash-reporting SDK. No Google
  Analytics, no Sentry, no Firebase, no pixel, no fingerprinting script.
- No third-party JavaScript at all: the page loads no CDN scripts, no external
  fonts and no remote images. Your browser talks to this app and nothing else.
- No location data, no device identifiers, no advertising ID.
- No profile, and therefore no profiling or automated decision-making about
  you.

## What the app does hold

| Thing | Where it lives | How long | Contains |
|---|---|---|---|
| Your ticker selections and slider settings | Your browser session, server memory | Until you close the tab or press **Clear this session's data** | Public ticker symbols |
| A random session reference (e.g. `9F3A2B7C`) | Same session, shown in the footer | Same | Nothing linked to you - it is random |
| Cached market data | Server disk, filename is a hash of the ticker set | Rotates on size; oldest first | Public Yahoo Finance prices, shared by every visitor who picks the same basket |
| Error and session-start log lines | Server disk (`logs/`, size-capped, rotating) | Until rotated out (small), or the next redeploy | Timestamp, session reference, ticker symbols, stack trace |
| Error lines only (no session starts) | Also the host's log stream (Railway), where Streamlit writes its own server errors | The host's log retention | The same error lines |

The session reference exists for one purpose: if you email about a problem and
quote it, the operator can find the matching stack trace. It is not a user ID,
it is not stored anywhere beyond the logs above, and a new one is issued every visit.

## Cookies and browser storage

Measured in a browser on 2026-11-02, this is everything the app stores on your
device:

| Name | Kind | Set by | Purpose | Lasts |
|---|---|---|---|---|
| `_streamlit_xsrf` | Cookie | Streamlit | Anti-forgery token: proves a request came from this page, not another site | Until you close the browser |
| `stActiveTheme-/-v2` | Local storage | Streamlit | Remembers the light/dark theme setting | Until you clear site data |

Both are needed for the page to work as you asked it to, carry no identity and
no tracking value, and are never read by anyone but this app. That is why there
is no consent banner: there is nothing optional to consent to. No advertising,
analytics or third-party cookies are set.

## Third parties

The **server** (not your browser) fetches data from:

- **Yahoo Finance** (via the `yfinance` library) - prices (adjusted for
  dividends and splits), volumes, and the 13-week T-bill rate. It is the only
  outside service the app contacts.

These requests contain ticker symbols. They do not contain anything about you,
and they are made from the server, so your IP address is never exposed to
Yahoo. The full list of hosts the app may contact is enumerated in
`src/netguard.py`, which refuses every other destination.

The app is hosted on Railway, a commercial platform. Like any web host,
it processes connection metadata such as IP addresses for delivery and abuse
prevention under its own privacy policy.

## Children

Nothing here asks for, or keeps, information about anyone - child or adult - so
there is no age gate. If that ever changes, this section will say how age is
checked before anything is collected.

## Deleting your data

There is no account to delete. Closing the tab discards everything the session
held; **Clear this session's data** (Lineage & Audit tab) does it immediately
without closing the page. If you want the log lines carrying your session
reference removed sooner than rotation removes them, email
meleona.support@gmail.com with that reference and they will be purged from the
server disk. Error lines in the host's log stream cannot be edited one by one;
they expire with the host's retention.

## Changes

This file is versioned in the repository, so every change to it has a date and
a diff in the public commit history.

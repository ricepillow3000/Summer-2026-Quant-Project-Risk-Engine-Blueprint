# Privacy

_Last updated: 2026-08-26. Contact: meleona.support@gmail.com_

Meleona is a public, read-only market-risk dashboard. It has **no sign-up, no
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
| Error and session-start log lines | Server disk (`logs/`, size-capped, rotating) | Until rotated out (small) | Timestamp, session reference, ticker symbols, stack trace |

The session reference exists for one purpose: if you email about a problem and
quote it, the operator can find the matching stack trace. It is not a user ID,
it is not stored anywhere else, and a new one is issued every visit.

## Cookies

One: the session cookie Streamlit uses to keep your browser tab connected to
the server while you interact with the page. It carries no identity and no
tracking value. No advertising or analytics cookies are set.

## Third parties

The **server** (not your browser) fetches data from:

- **Yahoo Finance** (via the `yfinance` library) - prices, volumes, dividends,
  splits, the 13-week T-bill rate.
- **Business Insider** - ISIN lookups only, because that is where the `yfinance`
  ISIN call goes.

These requests contain ticker symbols. They do not contain anything about you,
and they are made from the server, so your IP address is never exposed to
either service. The full list of hosts the app may contact is enumerated in
`src/netguard.py`, which refuses every other destination.

The app is hosted on a commercial platform (Railway/Render). Like any web host,
it processes connection metadata such as IP addresses for delivery and abuse
prevention under its own privacy policy.

## Deleting your data

There is no account to delete. Closing the tab discards everything the session
held; **Clear this session's data** (Lineage & Audit tab) does it immediately
without closing the page. If you want the log lines carrying your session
reference removed sooner than rotation removes them, email
meleona.support@gmail.com with that reference and they will be purged.

## Changes

This file is versioned in the repository, so every change to it has a date and
a diff in the public commit history.

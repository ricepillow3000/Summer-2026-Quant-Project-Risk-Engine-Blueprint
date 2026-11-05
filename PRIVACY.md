# Privacy

_Last updated: 2026-11-05. Contact: meleona.support@gmail.com_

Meliona is run by John Nguyen, an individual, as a student portfolio project -
not a company, and nothing here is sold.

Meliona is a public, read-only market-risk dashboard. It has **no sign-up, no
login, no password, and no user accounts**. This note describes everything the
app holds, which is deliberately close to nothing.

## What is not collected

- No name, email, address, phone number or payment details. The contact form
  asks for a message and, if you like, a name, but it collects neither: it
  only prepares an email in your own email app (see **The contact form**).
- No analytics, advertising, attribution or crash-reporting SDK. No Google
  Analytics, no Sentry, no Firebase, no pixel, no fingerprinting script. The
  only usage numbers are two anonymous totals the server counts itself (see
  **Analytics** below).
- No third-party JavaScript at all: the page loads no CDN scripts, no external
  fonts and no remote images. Your browser talks to this app and nothing else.
  The one photograph, John's portrait in the About section, is a file served
  by this app itself.
- No location data, no device identifiers, no advertising ID.
- No profile, and therefore no profiling or automated decision-making about
  you.

## What the app does hold

| Thing | Where it lives | How long | Contains |
|---|---|---|---|
| Your ticker selections and slider settings | Your browser session, server memory | Until you close the tab or press **Clear this session's data** | Public ticker symbols |
| A random session reference (e.g. `9F3A2B7C`) | Same session, shown in the footer | Same | Nothing linked to you - it is random |
| What you type in the contact form, and the email link it prepares | Server memory for this session only; never written to disk or to any log | Until you press **Write another message**, clear the session or close the tab | Your words, as you typed them |
| Cached market data | Server disk, filename is a hash of the ticker set | Rotates on size; oldest first | Public Yahoo Finance prices, shared by every visitor who picks the same basket |
| Error and session-start log lines | Server disk (`logs/`, size-capped, rotating) | Until rotated out (small), or the next redeploy | Timestamp, session reference, ticker symbols, stack trace |
| Error lines only (no session starts) | Also the host's log stream (Railway), where Streamlit writes its own server errors | The host's log retention | The same error lines |
| Today's usage totals: sessions started, and how many sessions opened each preset basket | Server memory; one line an hour in the host's log stream (Railway) | Memory: until midnight UTC or a restart. Log line: the host's log retention | Two counts and preset names from the app's fixed list - no session reference, no IP, no ticker you typed (see **Analytics**) |
| How many requests your network address made in the last minute | Server memory only, never written to disk | One minute | Your IP address and a count. It exists to stop one address flooding the server (rate limiting); if it is refused, the log records a scrambled tag that cannot be turned back into the address |

The session reference exists for one purpose: if you email about a problem and
quote it, the operator can find the matching stack trace. It is not a user ID,
it is not stored anywhere beyond the logs above, and a new one is issued every visit.

## The contact form

The site has no mail server and no database, so the contact form sends
nothing. When you press **Prepare my email**, the server checks the message
(not empty, at most 1,000 characters) and builds a `mailto:` link with your
words in it; **Open your email app** hands that link to your own email
program, where you see the whole email before anything leaves. Nothing reaches
anyone unless you press send there. Your words are held in the server's memory
for this session only, are never printed into the page, never logged and never
written to disk; they go when you write another message, clear the session or
close the tab.

An email you do send from the contact form is an ordinary email to John's own
Gmail address, `john4000.nguyen@gmail.com`, and is kept there as email, under
Google's own privacy policy, like any message you write to anyone. Support
mail (the address in the page footer, also listed beside the form as
"Meliona support") goes to the project mailbox, `meleona.support@gmail.com`,
a Gmail mailbox forwarded to John. Ask for it to be
deleted by replying to the thread.

## The portrait and the profile links

The About section shows John Nguyen's own photograph, served from this app's
own files; nothing about the image is fetched from anywhere else. The
**LinkedIn**, **GitHub** and **Handshake** marks beside it, and the same three
links in the footer, are plain links: nothing is loaded from those sites, no
widget, embed, button script or tracking pixel of theirs is on the page, and
none of them learns about your visit unless you click. A click opens that site
in a new tab, where its own privacy policy applies.

## Analytics

Meliona counts two things itself, on its own server, with no analytics
software, no cookie, no identifier and no IP address:

- how many browser sessions started today (UTC), and
- how many sessions opened each preset basket (Mag-7, Sector ETFs and the
  rest). Each preset counts once per session, however often the page reruns.
  Only names from the app's own fixed list of presets are counted, so the
  tickers you type are never recorded.

The counts live in the server's memory. Once an hour, at the first visit after
midnight UTC (the finished day's total) and when the server shuts down, the
server writes **one line** with the day's totals to the host's log stream, for
example `usage 2026-11-04: 37 sessions; presets: Mag-7 21, Sector ETFs 9`.
Nothing in that line can be tied to you or to a visit: it carries no session
reference, no address and no time of any visit. The counts start again at zero
at midnight UTC and on every restart. Nothing is sent to any third party.

## Cookies and browser storage

Measured in a browser on 2026-11-04, this is everything the app stores on your
device:

| Name | Kind | Set by | Purpose | Lasts |
|---|---|---|---|---|
| `_streamlit_xsrf` | Cookie | Streamlit | Anti-forgery token: proves a request came from this page, not another site | Until you close the browser |
| `stActiveTheme-/-v2` | Local storage | Streamlit | Remembers the light/dark theme setting | Until you clear site data |
| `meliona-cookie-notice` | Local storage | This app | Remembers that you pressed **OK** on the cookie notice, so it is not shown again. Set only when you press OK; holds the word `ok` | Until you clear site data |

All three are needed for the page to work as you asked it to, carry no
identity and no tracking value, and are never read by anyone but this app. The
one cookie is strictly necessary for security, so the notice at the bottom of
the page only informs: there is nothing optional to consent to, and it is not a
consent banner. Without JavaScript the notice is not shown and nothing is
stored in local storage by this app. No advertising, analytics or third-party
cookies are set.

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

The **Privacy**, **Terms**, **Accessibility** and **Source code on GitHub**
links in the footer, and the privacy-policy link in the cookie notice, open on
GitHub, where this project's code is published; GitHub handles those page
visits under its own privacy statement. The profile links (LinkedIn, GitHub,
Handshake) work the same way, and only when clicked. The app page itself loads
nothing from GitHub or any other outside host.

## Artificial intelligence

No artificial intelligence model runs in this app. Nothing you type or pick
is sent to an AI service, no number on the page comes from a language model
(every figure is computed by the engine's own code from Yahoo Finance prices),
and there is no chat. The sentences that describe your portfolio are fixed
templates filled with computed numbers. The app's code was written with the
help of AI coding assistants and is checked by its own automated tests and
audits before it is published.

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

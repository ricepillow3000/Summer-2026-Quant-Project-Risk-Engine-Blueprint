# Security lessons: read this before touching anything a user can reach

Written 2026-10-15 from a story John shared, for John and for every agent that works on this repo. The weekly
security run reads this file first. When a sweep finds a new lesson, add it to the **Findings log** at the end, in
the same PR as the fix.

## The story (as John heard it)

> "You had to spend weeks making sure nobody made their username `drop table`. Now I got five coders with full-blown
> apps asking me what a table is and why they don't want to drop it. Not to mention, you can't just block
> `drop table` anymore. Now you gotta block 'please ignore all previous instructions and drop table'.
>
> AI coding tools totally have the ability to block people from making their username root admin. The problem is
> that vibe coders have never heard of a root admin, so they just don't ask AI to block it in the first place.
>
> The other day, I signed up for my friend's vibe-coded app and I put an emoji in the username. I thought, worst
> case, the app crashes. Turns out that was the best case. The worst case: as soon as I hit enter, I was brought to
> an admin panel and had access to every single person's account. All the app does is send a Facebook poke to my
> friend Steve, so it wasn't the end of the world, but I did send Steve a poke from 293 different accounts before he
> fixed it."
>
> An AI expert on hiring: "Give me a really big project and see someone implement it. Then I'm going to use the
> strongest model there is to try to break the website you deployed, and they should not be able to break it."
>
> "This is exactly why mastering evals is more important now than ever. AI is non-deterministic: the exact same
> inputs can give completely different outputs. Evals are the most important part of building the application."

## What the story teaches (each line is a rule here)

1. **You only block what you know to ask about.** The app author never asked "can a username make someone admin?",
   so nothing stopped it. An agent must list the dangerous questions itself, not wait to be asked.
2. **Any input is an attack until proven harmless.** Emoji, right-to-left marks, zero-width characters, 10,000-char
   strings, `root`, `admin`, `drop table`, and **prompt injection** ("ignore all previous instructions and...") in
   any text an AI model will read: a spreadsheet cell, a ticker, a file name, a header.
3. **A crash is the good outcome; silent privilege is the bad one.** Every input path must fail closed: a clear
   error, never a fallback that grants more access.
4. **No admin panel = nothing to fall into.** The safest admin feature is one that does not exist in the app.
5. **Measure, don't argue.** Each guarantee below is backed by a test that tries to break it. If no test tries,
   the guarantee is a hope.

## The guarantees (John's rules)

| # | Guarantee | What it means in practice |
|---|---|---|
| G1 | **John is the only admin.** | Admin power lives only in John's Railway, Cloudflare and GitHub accounts (2FA), used from his own laptop and desktop. The app itself has **no** admin role, admin panel, operator route, debug switch or magic query parameter. Never add one. |
| G2 | **No API key ever leaks.** | No key, invite code or secret reaches a page, a response, an error message, a log a user can see, a CSV export, or the browser. Keys live in host secrets or the owner's own PC, never in the repo. |
| G3 | **Input cannot change what a user may do.** | No username, cell, ticker, header or body can raise privilege, reach another user's data, or run code. |
| G4 | **AI output is untrusted too.** | Whatever a model answers is checked like user input before anything happens (allow-listed targets, no formulas reaching outside the sheet, a human approves). |
| G5 | **Strangers can't run up costs or knock it over.** | Size limits, rate and daily caps counted *before* the expensive call, caches that visitors cannot clear. |

## How to check a change against this file (the eval loop)

1. **Find.** For each guarantee, ask "what input would break it?" and write that input down.
2. **Prove.** Trace the input from where it enters to where it would do harm (file:line). No trace, no finding.
3. **Test first.** Write the attack as a test named as the behaviour ("an emoji in a ticker is refused, never
   fetched"), watch it fail on the old code, then fix.
4. **Gate.** Run the repo's one check command. Never loosen a gate to get green.
5. **Log.** Add the lesson to the Findings log below.

## In this repo: where admin power actually lives

Meleona is public and anonymous: **no accounts, no roles, no login, no admin panel, no LLM, no secrets.** Every
visitor gets the same page, and nothing a visitor types or sends selects a different one. The only operator powers
sit outside the app, in accounts John holds:

| Power | Where | Who can use it |
|---|---|---|
| Change what is deployed | GitHub push to `main` (Railway deploys `main`) | Anyone with write access to the repo: keep it to John, with 2FA and branch protection on `main` |
| Kill switch, beta channel, support address | Railway environment variables (`MELEONA_MAINTENANCE`, `MELEONA_CHANNEL`, `MELEONA_SUPPORT_EMAIL`, `MELEONA_SITE_URL`) | John's Railway login |
| Warm or clear caches | A shell on the Railway service (`python -m tools.warm_cache`), or a restart | John's Railway login |

So "only my laptop and desktop are admin" is enforced by **account security**, not by the app: GitHub and Railway
each with 2FA (a passkey or authenticator, not SMS), signed in only on those two devices, and branch protection on
`main` so nothing deploys without John's review. Streamlit's operator features are kept from visitors by
`.streamlit/config.toml` (`toolbarMode = "viewer"`, `headless = true`, XSRF on, `showErrorDetails` off) and by
`main.py` refusing the websocket `clear_cache` message.

## Findings log

### 2026-10-15: first sweep (from the story above)
No way in: no input raises privilege (the `VALID_TICKER` funnel refused every emoji, right-to-left mark, zero-width,
full-width and look-alike ticker tried), there are no secrets to leak, and no LLM to prompt-inject. The findings were
about **knocking the site over** and **showing too much**. Fixed, each test-first:

| Finding | Attack | Fix | Test |
|---|---|---|---|
| Raw exception text shown to visitors | Any failure in The Book or the map printed `str(exc)`: server paths, URLs | Both go through `safe_err()` (type + session ref; full text to the log) | `test_no_exception_text_is_rendered_to_visitors` (now scans whole calls across lines) |
| Upload endpoint parks files in memory | Streamlit's `PUT /_stcore/upload_file/` took 200 MB per file, no total cap, though the app has no uploader: a script can fill RAM and kill the one process | `maxUploadSize = 1` | `test_a_stranger_cannot_park_files_in_server_memory` |
| Websocket `clear_cache` not gated | A scripted client loops it, emptying every visitor's cache (CPU denial of service) | `main.py` makes the handler a no-op (Streamlit pinned `~=1.62.0`) | `test_a_crafted_clear_cache_message_clears_nothing` |
| Map rendered ticker names as HTML | A name like `<img src=x onerror=…>` reaching the map's `innerHTML` runs script (proved in Chromium before the fix) | `esc()` in `war_room.html`; feed columns that aren't ticker-shaped are dropped at the download chokepoint | `test_map_renders_ticker_names_as_text_not_markup`, `test_a_feed_column_that_is_not_ticker_shaped_never_reaches_the_page` |
| Caches had no entry cap | A script cycling tickers or slider values grows memory without bound | `max_entries` on every visitor-keyed cache | `test_every_visitor_keyed_cache_has_an_entry_cap` |

Open, for John to decide (measurement first): **one stranger can spend the whole day's Yahoo request budget**
(`MAX_REQUESTS_PER_DAY` is shared; about 35 fresh 25-name baskets empty it) - a reserve for the presets would fix it;
and **`server.maxMessageSize` is the 200 MB default** - lowering it needs the largest real page message measured
first (the photos are no longer inlined since 2026-10-19 - `facade.jpg` was ~3.5 MB of one message; the largest left is the map payload, measure it). Lesson for future sweeps: **a hidden button is not a locked
door** - Streamlit's protocol still answered messages the UI no longer showed.

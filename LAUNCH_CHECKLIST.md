# Launch checklist

Every pre-launch item asked for, mapped onto what this app actually is: a
public, anonymous, read-only Streamlit dashboard with **no accounts, no
database, no payments, no LLM calls, no mobile app and no outbound email**.
Items that map onto a real surface are built; items that do not are marked
N/A with the reason and the trigger that would make them apply. Building the
N/A ones anyway would be controls guarding nothing - the thing a reviewer
notices fastest.

| # | Asked for | Status | Where it lives / why not |
|---|---|---|---|
| 1 | Beta-test wire: crash reports | **Built** | `src/observability.py` - rotating local log, per-session reference shown in the footer, `log_incident()` for handled failures. No third-party SDK on purpose: an SDK shipping data to a vendor would contradict the privacy note. |
| 2 | Delete-account button | **Built as the honest equivalent** | There is no account to delete. Lineage & Audit tab now carries a "Your data" panel listing everything held, plus **Clear this session's data**. Log lines for a session reference are purged on request (`PRIVACY.md`). |
| 3 | Privacy policy | **Built** | `PRIVACY.md`, linked in the page footer. |
| 4 | Terms / policies | **Built** | `TERMS.md` - educational use, not investment advice, no warranty, acceptable use. Linked in the footer. |
| 5 | Declare SDKs and what is collected | **Built** | `PRIVACY.md` tables. Verified rather than asserted: the page loads **no** third-party JavaScript, fonts or images; the one server-side third party is Yahoo Finance (Business Insider left with the Security Master, 2026-10-29), and `src/netguard.py` enumerates every host the server may contact. |
| 6 | SPF / DKIM / DMARC | **N/A today, recipe ready** | The app sends no email and there is no sending domain - support is a Gmail address. `RUNBOOK.md` §7 has the exact records and the `p=none` -> `quarantine` -> `reject` order for the day a custom domain exists. |
| 7 | Test sign-up | **N/A - replaced by a beta feedback channel** | There is no sign-up to test. Instead, `MELIONA_CHANNEL=beta` marks a deployment as a beta and shows a one-click **report this session** link with the session reference already in the subject. |
| 8 | Cap API spend / caps on every API | **Built** | `src/netguard.MAX_REQUESTS_PER_DAY` (5000/day) charged at the one chokepoint every outbound call passes through. Exhaustion degrades to cached data instead of hammering the feed. `ingestion.MAX_UNIVERSE` (25) caps compute per visitor; `CACHE_MAX_FILES` (1500, a warm pass writes ~720)/`CACHE_MAX_MB` (200) cap disk. |
| 9 | LLM credit balance | **N/A by design** | The engine makes zero LLM calls - project constraint #1 is that no market number may originate from a language model. There is no balance to monitor. If a paid feed is ever added, its key goes behind the same counter as #8. |
| 10 | Database restore | **N/A - nothing to restore, and the drill is executable** | No database and no user data. The only durable state is git; the cache is derived. `python -m tools.warm_cache --drill` refetches every preset and prints the total, which IS the recovery time for the only regenerable state. `RUNBOOK.md` §0. |
| 11 | Kill switch | **Built** | `MELIONA_MAINTENANCE=1` on the host serves a maintenance page and stops before any market-data call - no redeploy. `RUNBOOK.md` §1. |
| 12 | OTA hotfix | **Built (web equivalent)** | No store review in the path: `git revert` + push redeploys, with a force-with-lease rollback documented. `RUNBOOK.md` §2. |
| 13 | Support email | **Built** | `meleona.support@gmail.com` in the page footer, `SUPPORT.md`, the maintenance page and the privacy panel - all from one constant in `main.py`. Users quote the footer's session reference; §3 of the runbook turns it into a stack trace. |
| 14 | Demo accounts | **N/A** | Nothing to log into. `RUNBOOK.md` §6 is the reviewer script instead. |
| 15 | Review notes | **Built** | `RUNBOOK.md` §6 - a five-minute path through the product for a reviewer or recruiter. |
| 16 | Restore purchases | **N/A** | No payments, no in-app purchases, no entitlements. |
| 17 | Reinstall behaviour | **N/A** | Web app - a "reinstall" is a page refresh. Nothing is installed on the device, and no state is lost. |
| 18 | Sandbox / test keys | **N/A** | The app holds no API keys at all: Yahoo Finance needs none. Nothing to rotate, sandbox or leak. |
| 19 | Phased release | **Built (process + flag)** | `RUNBOOK.md` §5 - local -> unlisted preview -> small beta -> public, kill switch one variable away at every stage. `MELIONA_CHANNEL=beta` puts the beta notice and feedback link on the pre-release deploys only. |
| 20 | Accessibility statement | **Built** | `ACCESSIBILITY.md`, linked in the page footer alongside Privacy and Terms, with a "report a barrier" route to the support mailbox. Seven text colours were measured failing WCAG AA and corrected; the canvas map gained a screen-reader description. Two regression tests keep both true. |
| 21 | Custom 404 page | **Built** | `public/404.html` in the house style with a way back. A single-page app answers every path with the same page, whose relative script paths broke below the root - a mistyped `/a/b` link was a blank white page. A one-line head script sends unknown paths to the 404. Honest limit: the status code stays 200 (Streamlit cannot set it), so the page is `noindex`. Checked by `tools/site_check.mjs`. |
| 22 | CTA above the fold | **Verified** | `site_check` asserts it at 1440x900 (bottom 587px) and 390x844 (454px). |
| 23 | Meta description | **Built** | One page, one description. `tools/public_files.py` writes it into the served HTML at start, with title and canonical URL - crawlers do not run the JavaScript that sets Streamlit's own title. |
| 24 | Open Graph image | **Built** | `public/og-image.png` (1200x627) plus og:/twitter: tags, so a LinkedIn post of the link shows the card. Its audit figure is the latest live run's, not a stale one. |
| 25 | Favicon set | **Built** | Crest rendered to `favicon.png` (64px, also the tab icon) and `apple-touch-icon.png` (180px). |
| 26 | robots.txt and sitemap.xml | **Built** | Written at start for `MELIONA_SITE_URL`; served from the site root. |
| 27 | Alt text on every image | **Verified** | `site_check` fails on any `<img>` without alt. Photos are decorative CSS backgrounds; the crest SVG carries `role="img"` and a label. |
| 28 | Mobile breakpoints | **Verified** | 390px: no sideways scroll, crest and CTA on screen; short-viewport tiers in `main.py`. |
| 29 | Sticky mobile CTA | **N/A by design** | The page has one job and its CTA is above the fold on a phone; a pinned bar would cover the headline verdict on a 390px screen. Revisit if a real visitor study says otherwise. |
| 30 | Loads under 2 seconds | **Built** | First paint under 200ms: the hero's boot skeleton, painted from the served HTML before Streamlit's JavaScript (2026-10-19; before that a plain screen, and on a phone ~7 s of blank page). Disclosed rather than hidden: the hero follows in ~2 s on a laptop, ~5.6 s on a 9 Mbps phone, and the live-data headline after that; an "Updating…" sign shows while the page works. |
| 31 | Form error states | **Built** | The ticker box is the form. Mistyped or dead symbols, more than 25 symbols, an invalid symbol, a constant-price asset and fewer than two usable symbols each get a specific message, each covered by a test. |
| 32 | Thank-you pages | **N/A** | Nothing is submitted: no sign-up, no order, no form post. |
| 33 | Privacy pages | **Built** | `PRIVACY.md`, `TERMS.md`, `ACCESSIBILITY.md`, linked from the footer. |
| 34 | Cookie banner | **N/A** | Measured 2026-11-02: one cookie (`_streamlit_xsrf`, anti-forgery, session) and one local-storage key (`stActiveTheme-/-v2`, the theme), both Streamlit's and strictly necessary; no analytics or advertising cookies exist. Adding analytics would make a banner necessary. |
| 35 | Analytics installed | **N/A by decision** | `PRIVACY.md` promises no analytics SDK and a test asserts Streamlit telemetry is off. Traffic is read from the host's own request metrics instead. |
| 36 | Real contact details | **Built** | `meleona.support@gmail.com`; `site_check` fails if the page stops showing it. |
| 37 | Security sweep | **Built** | XSS: `VALID_TICKER` funnel + escapes. CSRF: XSRF on. SSRF: `src/netguard.py` allowlist. Path traversal, exposed `.env`/`.git`/logs/source/secrets, permissive CORS, source maps: `tools/security_probe.py`, run on every live audit (all pass). Dependencies: `pip-audit` clean. Code: `bandit` clean. N/A, no surface: uploads, passwords and resets, sessions/JWT, default credentials, IDOR/BOLA, payments, webhooks. Headers: real HSTS, X-Frame-Options and `frame-ancestors` since 2026-11-04 (`serve.py`, #102); before that this row said Streamlit could not set them, which stopped being true with 1.66. |

### The 2026-11-02 sweep: performance, legal and security

A 60-item list, each checked against this app on 2026-11-02. Same rule as above:
built where there is a surface, N/A with the reason where there is none.

| # | Asked for | Status | Where it lives / why not |
|---|---|---|---|
| 38 | Images as WebP or AVIF | **Built** | AVIF first, JPEG/WebP fallback via `image-set()`: 4.4 MB of photos -> 0.3 MB (2026-10-30). The link-preview image and favicons stay PNG, which is what LinkedIn and iOS read. |
| 39 | No duplicate API calls | **Built** | Every fetch is behind `st.cache_data`, each name's download is kept on disk, and only names not held are fetched - in one `yf.download` per basket. |
| 40 | Debounce search and autocomplete | **N/A** | The ticker box (`st.multiselect`) filters its suggestions in the browser as you type and tells the server nothing until a symbol is picked; there is no autocomplete endpoint to call. |
| 41 | Paginate API results | **N/A** | The app serves no API. Long tables use Streamlit's virtualized grid. |
| 42 | Static files through a CDN | **N/A, decided** | RUNBOOK §4: Cloudflare was rejected for this app. Streamlit's own bundle is cached for a year; our files are gzipped and revalidated by ETag; photos are AVIF. |
| 43 | Slow tasks on a background queue | **Built (fitting form)** | The slow task is the Yahoo pull. `tools.warm_cache --every 5` refreshes every preset beside the app, so a visitor rarely waits on the network. One process: a job queue would have no workers to feed. |
| 44 | Fix N+1 queries | **N/A, analogue fixed** | No database. The analogue - one request per ticker - is already one download per basket (#39). |
| 45 | Brotli or gzip | **Built 2026-11-02** | `server.enableWebsocketCompression = true`: every chart, table and the map reach the browser over the websocket, and a page load's 107 KB of messages deflate to ~44 KB. Our static files are gzipped. Known limit: Streamlit serves its root HTML and its own bundle uncompressed by design (its source explains why); the bundle is cached for a year. |
| 46 | Cache hot data with Redis | **N/A** | One process: `st.cache_data` in memory plus the parquet disk cache. Redis would be a second service with nothing to share. |
| 47 | Remove unused dependencies | **Verified 2026-11-02** | Every line of `requirements.txt` is imported or a documented runtime need: `pyarrow` (parquet), `scikit-learn` (yfinance and Ledoit-Wolf), `pillow` (CVE floor for Streamlit). |
| 48 | Timeouts on external calls | **Built 2026-11-02** | Every outbound request goes through `netguard.guarded_session()`, now with an explicit 30 s timeout (yfinance's handshake passed none). Tested. |
| 49 | Stream AI responses | **N/A** | No AI anywhere (constraint #1). |
| 50 | Update the UI before the server answers | **Built (fitting form)** | The skeleton paints from the served HTML before the server answers, and an "Updating..." rail appears the instant a rerun starts. No write action exists to apply optimistically. |
| 51 | Lazy-load images below the screen | **Built** | The band photo and the topography attach within 600px of the screen (`static/reveal.js`). |
| 52 | Run independent requests in parallel | **Built** | `yf.download` fetches a basket's names in parallel threads (`threads=True`, yfinance's default). |
| 53 | Database indexes, only needed columns, pooling, server near the DB | **N/A** | No database. Downloads ask for prices and volume only. |
| 54 | Trace slow requests with OpenTelemetry | **N/A, decided** | A hosted tracer is a new processor of visitor data (`PRIVACY.md`). Slowness is measured by the stress test (RUNBOOK §4) and the deep scan instead. |
| 55 | Render on the server or at build time | **Built (fitting form)** | The hero's skeleton is in the served HTML (first paint under 200 ms, #30). Live market data cannot be prebuilt. |
| 56 | Load third-party scripts after the page | **N/A** | None exist: measured 2026-11-02, the browser contacts no host but this app. |
| 57 | Split JavaScript by page | **N/A** | One page. Streamlit already splits its bundle into chunks. |
| 58 | Virtualize long lists | **Built by the framework** | `st.dataframe` renders only the visible rows. |
| 59 | Prevent unnecessary re-renders | **Built** | Caches on every expensive call; elements keyed by position (`test_the_tab_strip_keeps_its_slot_whatever_the_basket`); reveals never replay. |
| 60 | Browser caching for static files | **Built, with a known limit** | Streamlit's bundle: `immutable, max-age=31536000`. Our `static/` files: ETag and Last-Modified, so repeat visits get 304s. Streamlit cannot set `Cache-Control` on app static files. |
| 61 | Load-balance several instances | **N/A, decided** | The cache is the container's own disk; one instance by design. Capacity is measured, not assumed (RUNBOOK §4). |
| 62 | Self-host fonts, few weights | **Built** | No web-font request at all: Georgia and system fonts, plus Streamlit's bundled ones. |
| 63 | Real-user speed monitoring | **N/A, decided** | `PRIVACY.md` promises no analytics (#35). |
| 64 | Test on slow phones and networks | **Built** | `tools/device_check.mjs`: 13 devices on every push; first-paint timing at phone speeds (#30). |
| 65 | Privacy policy | **Updated 2026-11-02** | `PRIVACY.md`: the measured cookie table, Yahoo as the only outside service, who runs the app, a Children section. |
| 66 | Terms of service | **Updated 2026-11-02** | `TERMS.md`: who runs it, Price, Images. |
| 67 | Refund policy | **N/A** | Free: nothing is sold, so nothing is refunded (`TERMS.md` "Price"). |
| 68 | Cookie policy | **Built** | `PRIVACY.md` "Cookies and browser storage", measured in a browser. |
| 69 | Cookie consent banner | **N/A** | See #34. |
| 70 | Form consents | **N/A** | No form collects personal data. |
| 71 | Collect no unnecessary data | **Verified** | Nothing personal is collected (#5). |
| 72 | Audit third-party SDKs | **Done 2026-11-02** | None in the browser. On the server only `yfinance`, to Yahoo; the allowlist dropped its unused Business Insider entry. |
| 73 | Dark patterns, hidden fees, fake reviews | **N/A, verified** | No prompts to trick, no fees, no testimonials or ratings. |
| 74 | Remove unsupported claims | **Fixed 2026-11-02** | README: "Institutional-Grade" and "the standard a quant desk would actually defend" gone; "Live demo: coming August 2026" replaced by the live URL. The hero was rewritten plainly on 2026-10-29. |
| 75 | Alt text | **Verified** | See #27. |
| 76 | Colour contrast | **Verified** | `test_text_colours_meet_wcag_aa`. |
| 77 | Keyboard navigation | **Fixed 2026-11-02** | The map's pairing rows were click-only: now focusable buttons that Enter and Space pick (`test_the_pairing_rows_answer_the_keyboard`). The rest: `ACCESSIBILITY.md`. |
| 78 | Business details | **Added 2026-11-02** | Operator named in `PRIVACY.md` and `TERMS.md`; contact address (#13). |
| 79 | Age consent for children's data | **N/A** | No data is collected from anyone (`PRIVACY.md` "Children"). |
| 80 | Unsubscribe link in emails | **N/A** | The app sends no email (#6). |
| 81 | License stock images | **Documented** | Unsplash License, in `TERMS.md` and the README's Credits. Open item: record each photo's Unsplash page link (below). |
| 82 | Data deletion requests | **Built** | See #2. |
| 83 | Hide API keys; check env variables | **N/A / verified** | The app holds no keys (#18). Its only host settings are non-secret `MELIONA_*` switches. |
| 84 | Protect admin routes | **N/A, by policy** | None exist and none may be added (`docs/security/SECURITY-LESSONS.md`). |
| 85 | Authentication and user access | **N/A** | No accounts: the app is public and read-only. |
| 86 | Sanitize forms; XSS | **Built** | The `VALID_TICKER` funnel and `esc()` (#37). |
| 87 | Rate limiting | **Built at the cost surface** | The outbound budget (#8) and cached compute (RUNBOOK §4). A per-IP limiter cannot see past Railway's proxy, and the load lives inside one websocket (RUNBOOK §4). |
| 88 | Secure API endpoints; CORS | **Built** | Streamlit's endpoints: XSRF and CORS on, uploads capped at 1 MB, the websocket `clear_cache` message refused. All pinned by tests. |
| 89 | Security headers | **Built, corrected 2026-11-04** | Real response headers on every path since 2026-11-04 (#102). The meta tags from 2026-11-02 stay as a second layer. The 2026-11-02 note that HSTS and frame-ancestors were impossible was wrong for Streamlit 1.66, which accepts an ASGI wrapper. |
| 90 | Debug mode off | **Verified** | `showErrorDetails = false`, `toolbarMode = "viewer"`, both tested. |
| 91 | Update dependencies; remove unused packages | **Built** | `pip-audit` in Live audit step 0, clean; see #47. |
| 92 | Exposed files | **Built** | `tools/security_probe.py` on every live audit; `static/` holds public files only (tested). |
| 93 | Secure the database; hash passwords | **N/A** | No database, no passwords. |
| 94 | Leaked secrets in git history | **Scanned 2026-11-02** | The whole history was searched for cloud, GitHub, Slack, Google and model-provider keys, private keys, password/secret/api-key assignments and `.env`/key files: none ever committed. |
| 95 | Full security audit | **Built** | `security_probe`, `bandit` and `pip-audit` on every push; the weekly sweep reads `docs/security/SECURITY-LESSONS.md`. |

### 2026-11-02: Docker or Kubernetes

| # | Asked for | Status | Where it lives / why not |
|---|---|---|---|
| 96 | Runs the same on any machine (containerization) | **Built** | `docker/Dockerfile` runs what the `Procfile` runs, as a non-root user, with a health check (`test_the_container_runs_what_the_procfile_runs`). The `Container` workflow builds and checks it on every push and publishes `ghcr.io/ricepillow3000/meliona` from `main` - free. One command: `docker run -p 8501:8501 ghcr.io/ricepillow3000/meliona:latest`. |
| 97 | Auto-healing | **Built, free** | `railway.toml`: a deploy takes traffic only once `/_stcore/health` answers; a crashed process restarts (up to 10 times). In the box: Docker's `HEALTHCHECK`, plus a restart policy. |
| 98 | Scaling up and load balancing (Kubernetes) | **N/A, decided** | One process by design: the cache is the container's disk and sessions live in memory on one websocket, so more copies mean cold caches and dropped sessions. Managed Kubernetes costs ~$70/month before servers. RUNBOOK §4, "Docker, not Kubernetes". |

### 2026-11-04: security lists (vibe-code holes, securitymaxxing, ship hardening)

Checked against the running app and its code by a security review on
2026-11-04. Every item that has a surface here was built, with a test.

| # | Asked for | Status | Where it lives / why not |
|---|---|---|---|
| 99 | SQL injection, parameterized queries, excessive database permissions, row-level security, public database key | **N/A** | No database. The only SQL anywhere is yfinance's own local timezone cache, written through its ORM with bound parameters, after tickers have passed `VALID_TICKER`. |
| 100 | Rate-limit the AI endpoint, keep the model key on the server, prompt injection, unpermissioned AI access | **N/A** | No model is called and no model reads anything (project constraint #1, #9). |
| 101 | Check that every package the AI chose exists (slopsquatting), malicious packages, scan dependencies | **Built 2026-11-04** | All ten packages, the gate's tools and the browser test's `playwright` were checked on PyPI and npm: each is the real project, none a look-alike. Every package is now pinned to one exact version (`pandas>=2.2` had been resolving to whatever was newest), the gate's tools too, and every GitHub Action to a commit (a tag can be moved). `pip-audit` runs on every push; Dependabot opens weekly bump PRs, each running every audit (`test_every_dependency_is_pinned_to_an_exact_version`). Known limit: second-level packages are not hashed. |
| 102 | Security headers, force HTTPS, clickjacking | **Built 2026-11-04** | `serve.py` runs `main.py` with real headers on every response: HSTS (a year), `X-Frame-Options: SAMEORIGIN` and `frame-ancestors 'self'`, `nosniff`, referrer and permissions policies. Railway terminates TLS. `test_every_response_carries_the_security_headers`; the security probe requires them of this commit's app on every run, and of the live site on `main`. |
| 103 | Exposed internal dashboards | **Fixed 2026-11-04** | Streamlit served `/_stcore/metrics` (live session counts, cache sizes) to anyone; it now answers 404. No operator menu (#90), no admin route (#84). |
| 104 | Insecure cookie settings, secure session cookies | **Verified, by design** | One cookie, `_streamlit_xsrf`: an anti-forgery token, not a login. `SameSite=Lax`; not `HttpOnly` because the page must read it to send it back (double-submit); not marked `Secure` because TLS ends at Railway, in front of the app, and HSTS now keeps the browser on HTTPS. No session or login cookie exists. |
| 105 | Missing audit logs, security monitoring, error monitoring | **Built 2026-11-04** | Admin actions are logged by GitHub and Railway, the only admin doors (`docs/security/SECURITY-LESSONS.md`). App errors now also go to the host's log, so a redeploy no longer erases them (RUNBOOK §3). Monitoring: CodeQL, secret scanning with push protection, Dependabot alerts, the weekly security sweep, and the Live audit, now also daily on `main` against the live site. |
| 106 | Backups | **N/A, verified** | Nothing to lose: code in git, the image on GHCR, the cache derived (#10). |
| 107 | Tenant isolation, lock record access, permission checks | **Verified** | No accounts or records. Each browser session's choices live in its own session state; the shared caches hold only public prices, keyed on validated tickers, with entry caps, and no visitor control clears them (#88). A stranger can still spend the web process's Yahoo budget; the presets are protected from that (own warmer budget, and `test_the_cache_budget_drops_typed_baskets_before_the_presets`). |
| 108 | Unreviewed AI code | **Built** | `main` takes changes only through a pull request whose CodeQL and Live audit checks pass (ruleset, RUNBOOK §8); `tools.check` (ruff, bandit, the engine tests) runs before every push and as step 0. |
| 109 | Mass assignment, block field tampering, validate all input, escape user content | **Verified** | No form maps fields onto an object. Every control but the ticker box is a closed list or a bounded number that Streamlit re-checks on the server; free text passes `VALID_TICKER` before any HTML, URL or path (#86). The only query parameter read, the preset, must name a preset or is ignored. |
| 110 | Trim API responses; oversized messages | **Built 2026-11-04** | No API. Error text is replaced by `safe_err()` (#90). The websocket's message cap fell from 200 MB to 16 MB; the largest page message is the Risk Topology page, about 0.1 MB with the full 25 names (measured 2026-11-04), so the cap leaves over a hundred times that. |
| 111 | Command injection, insecure deserialization | **Verified** | No shell, `eval` or `exec` on the production path; no pickle or YAML. The app reads back only the parquet and JSON files it wrote, named by a hash of validated tickers. |
| 112 | Hide API keys, purge git secrets, unencrypted sensitive data | **Verified 2026-11-04** | No keys and no sensitive data exist. All 196 commits rescanned, plus `detect-secrets` on the tree: nothing (#94). |
| 113 | Hash passwords, rate-limit login, bot protection on login, server-side auth, misconfigured OAuth, password recovery, confirmation emails | **N/A** | No accounts, no login, no email (#85, #93). |
| 114 | Restrict file uploads | **Built** | No uploader; Streamlit's upload endpoint is capped at 1 MB and needs the anti-forgery token (#88). |
| 115 | Rate limiting, bot protection | **Built at the cost surface** | #87, #8. Per-visitor limits would need the client IP Railway forwards, and there is nothing per visitor to protect; the expensive parts (downloads, the simulations) are budgeted and cached. |

### 2026-11-04: the vibe-coded tells, the 20 ship items and the AI-slop design list

Measured in Chromium at 390px and 1440px on 2026-11-04, then fixed. Each fix
has a regression test in `tests/test_engine.py`, so the tell cannot come back.

| # | Asked for | Status | Where it lives / why not |
|---|---|---|---|
| 116 | Purple gradients, gradient backgrounds | **Fixed** | Never purple. The decorative gradients (sheens, glints, a gradient-filled numeral, glass, page "lacquer", band fades, gold bars on the map) are gone; the only ones left do a job: the loading shimmer, the "Updating" rail and the scrims that keep text readable over photos (`test_no_decorative_gradients_glass_or_grain`). |
| 117 | Too much padding, floating elements | **Fixed** | Phone gutters 56px -> 20px, the map still 8px from the phone's edges; hover lifts and floating shadows removed. |
| 118 | The keyboard covering input fields; buttons too small | **Fixed** | No bar is pinned to the bottom of the screen. On touch, tabs, selects, number fields, drawer headers, radio and checkbox rows, buttons and the map's controls are 44px tall, and the small marks (help, clear-all, slider thumbs) get an invisible 44px hit area. Phone inputs are 16px, so iOS no longer zooms in on focus (`test_touch_targets_phone_fields_and_gutters`). |
| 119 | Text too small; text scaling breaks the UI | **Fixed** | Smallest text raised to 12px on the page (footer, session reference, Book role labels) and 10px on the map. Every font size is in rem, so a reader's own font-size setting reaches the page; the hero fits at 200% zoom on a phone (`test_type_follows_the_readers_font_size_and_stays_readable`). |
| 120 | No caching | **Built** | #39, #46, #60. |
| 121 | Loading spinners everywhere; skeleton loaders | **Built** | The first paint is a skeleton of the page (#30); no `st.spinner` exists. Streamlit's own cache notes ("Fetching market data") say what they are waiting for, under one "Updating" rail. |
| 122 | Emojis in buttons and icons | **Verified** | None in the app or the map (#73 pass, re-checked); the README's emoji checkmarks were removed too. |
| 123 | Tapping twice does the thing twice | **Verified** | The only action button, "Clear this session's data", gives the same result twice; the map's Run button disables itself while it runs; a double-clicked jump button lands once. |
| 124 | New users land on an empty screen; helpful empty states | **Built** | A first visit opens on a full preset basket. An emptied basket says to add two symbols or pick a preset again. |
| 125 | Dark and light mode break the UI | **Verified** | The theme is pinned light in `.streamlit/config.toml` and the menu that could switch it is hidden; a dark operating system and a forced dark setting both still render the beige page (screenshots in the 2026-11-04 audit). |
| 126 | Raw error text; error messages that explain | **Built** | `safe_err()` and `showErrorDetails = false` (#90). New: when Yahoo cannot be reached for a preset, the message says the feed is probably unreachable instead of asking the visitor to check the spelling. |
| 127 | No animation when tapping buttons | **Built** | Every button on the page and the map has a pressed state. |
| 128 | Preferences wiped on an update or reload | **Built** | The chosen preset rides in the address (`?preset=`), so a reload or a shared link opens the same basket. Only a real preset name is accepted; anything else is ignored (`test_the_preset_rides_in_the_address_and_only_a_known_one`). |
| 129 | Optimistic rendering; tooltips on icon buttons | **Built** | #50. Every icon-only control has a name: Streamlit's help marks show their text on hover and focus, and the rest carry a label a screen reader reads; the map's three switches had none and now do (`test_the_maps_switches_have_names`). |
| 130 | The 20 ship items | **Covered** | Mobile layout #28; loading #30; empty states #124; error messages #126; server-side validation #86; permission checks, password recovery, confirmation emails N/A #85/#113; keyboard #77 (and the page's main area now shows a focus ring, the hidden glide frame left out of Tab order); form labels: every field labelled, the map's switches fixed (#129); alt text #27; title and description #23; sitemap #26; privacy #3; HTTPS #102; analytics N/A #35; error monitoring #105; environment variables: RUNBOOK §0 table; backups #106; rollback #12. |
| 131 | Panel subtitles too faint | **Fixed** | About 25 subtitles were `#ABA499` on beige, 2.04:1: the contrast test passed them because it checked each colour against the darkest ground on the page, not its own. Now `#5D574D` (5.91:1), and `test_text_colours_meet_wcag_aa` checks every text colour on the ground it actually sits on. It also caught the CTA hover at 4.14:1 (now 6.42:1). |
| 132 | Tags and pills, squircles, containers with borders | **Fixed** | The pair diagram's rounded pill and Streamlit's rounded tags and alert boxes are square; cards keep only the house's top rule instead of a full box. |
| 133 | Excessive italics, Instrument Serif, excessive monospace | **Fixed / verified** | No italics left in the copy; no web font at all (Georgia, #62); ticker names and the session reference are plain text, not code font. |
| 134 | Negative kerning, all-caps subtitles | **Fixed** | No negative letter-spacing anywhere; capitals only on the headline verdict label (`test_no_negative_letter_spacing_anywhere`, `test_uppercase_is_kept_to_the_verdict_label`). |
| 135 | Over-detailed UI, overly detailed diagrams, dividers | **Fixed** | The pair diagram is down to two circles, a dashed line and the weights; the hero, band and footer lost their extra hairlines and the `<hr>`. |
| 136 | Many colour shades and opacities | **Fixed** | Nine beige grounds -> three, five muted inks -> two, five hairlines -> one. |
| 137 | Verbose text, code breadcrumbs, section numbering, eyebrows, periods in titles | **Fixed** | The ten longest captions were shortened, keeping every number and limit; the data-quality checks print plain names instead of code IDs; the 01/02/03 numbering, the eyebrows and the period on a title are gone, on the page, the map and the 404 page (`test_visitor_copy_has_no_numbers_eyebrows_honest_or_shouting`). |
| 138 | Dithering, ASCII images | **Fixed / verified** | The map's grain texture is gone; no ASCII art exists. |

## Before the link goes public

- [ ] `python -m tests.test_engine` green (includes the full-app boot test).
- [ ] `python -m tests.batch_audit` green (live-data math audit).
- [ ] Kill switch tested on the host: set `MELIONA_MAINTENANCE=1`, load the
      page, unset it, load again.
- [ ] Footer shows the support address, Privacy, Terms and a session reference.
- [ ] Trigger one deliberate error in the preview deploy and confirm the
      reference in the footer finds it in `logs/meliona.log`.
- [ ] Budget counter sane after a browsing session (`budget_status()`).
- [x] **Create `meleona.support@gmail.com`** - done 2026-08-26.
- [x] **Turn on forwarding to the personal inbox** - done 2026-08-26.
- [x] Test mail sent to the project address and confirmed arriving in both
      mailboxes (2026-08-26).
- [ ] Set `MELIONA_CHANNEL=beta` on the beta deploy, and unset it for public.
- [x] Warm cache: automatic. The Procfile runs `python -m tools.warm_cache --every 5`
      beside the app, so no preset expires before a visitor picks it - every pull
      its page makes, full histories included (RUNBOOK.md,
      *After every deploy*).
- [ ] Read `ACCESSIBILITY.md` once before sharing the link - it is the page a
      complainant's lawyer reads first, and it must stay true of the app.
- [ ] Deploy to Railway (`RUNBOOK.md` §6c) with `MELIONA_CHANNEL=beta`
      and confirm `/_stcore/health` answers.
- [ ] Walk the deployed page once end to end, then trigger one deliberate error
      and confirm the footer reference finds it in the logs.
- [ ] Record each photo's Unsplash page link (`static/facade.*`, `static/band.*`)
      beside its credit in the README - the licence needs no attribution, but the
      link is the proof of where each came from.
- [ ] After the first `Container` run on `main`, make the image public once:
      GitHub -> your profile -> Packages -> meliona -> Package settings ->
      Change visibility -> Public. (A new package starts private; until then
      `docker run` asks the visitor to sign in.)

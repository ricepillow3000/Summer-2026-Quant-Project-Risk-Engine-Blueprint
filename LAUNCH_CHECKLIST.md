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
| 5 | Declare SDKs and what is collected | **Built** | `PRIVACY.md` tables. Verified rather than asserted: the page loads **no** third-party JavaScript, fonts or images; server-side third parties are Yahoo Finance and Business Insider (ISIN), and `src/netguard.py` enumerates every host the server may contact. |
| 6 | SPF / DKIM / DMARC | **N/A today, recipe ready** | The app sends no email and there is no sending domain - support is a Gmail address. `RUNBOOK.md` §7 has the exact records and the `p=none` -> `quarantine` -> `reject` order for the day a custom domain exists. |
| 7 | Test sign-up | **N/A - replaced by a beta feedback channel** | There is no sign-up to test. Instead, `MELEONA_CHANNEL=beta` marks a deployment as a beta and shows a one-click **report this session** link with the session reference already in the subject. |
| 8 | Cap API spend / caps on every API | **Built** | `src/netguard.MAX_REQUESTS_PER_DAY` (5000/day) charged at the one chokepoint every outbound call passes through. Exhaustion degrades to cached data instead of hammering the feed. `ingestion.MAX_UNIVERSE` (25) caps compute per visitor; `CACHE_MAX_FILES`/`CACHE_MAX_MB` cap disk. |
| 9 | LLM credit balance | **N/A by design** | The engine makes zero LLM calls - project constraint #1 is that no market number may originate from a language model. There is no balance to monitor. If a paid feed is ever added, its key goes behind the same counter as #8. |
| 10 | Database restore | **N/A - nothing to restore, and the drill is executable** | No database and no user data. The only durable state is git; the cache is derived. `python -m tools.warm_cache --drill` refetches every preset and prints the total, which IS the recovery time for the only regenerable state. `RUNBOOK.md` §0. |
| 11 | Kill switch | **Built** | `MELEONA_MAINTENANCE=1` on the host serves a maintenance page and stops before any market-data call - no redeploy. `RUNBOOK.md` §1. |
| 12 | OTA hotfix | **Built (web equivalent)** | No store review in the path: `git revert` + push redeploys, with a force-with-lease rollback documented. `RUNBOOK.md` §2. |
| 13 | Support email | **Built** | `meleona.support@gmail.com` in the page footer, `SUPPORT.md`, the maintenance page and the privacy panel - all from one constant in `main.py`. Users quote the footer's session reference; §3 of the runbook turns it into a stack trace. |
| 14 | Demo accounts | **N/A** | Nothing to log into. `RUNBOOK.md` §6 is the reviewer script instead. |
| 15 | Review notes | **Built** | `RUNBOOK.md` §6 - a five-minute path through the product for a reviewer or recruiter. |
| 16 | Restore purchases | **N/A** | No payments, no in-app purchases, no entitlements. |
| 17 | Reinstall behaviour | **N/A** | Web app - a "reinstall" is a page refresh. Nothing is installed on the device, and no state is lost. |
| 18 | Sandbox / test keys | **N/A** | The app holds no API keys at all: Yahoo Finance needs none. Nothing to rotate, sandbox or leak. |
| 19 | Phased release | **Built (process + flag)** | `RUNBOOK.md` §5 - local -> unlisted preview -> small beta -> public, kill switch one variable away at every stage. `MELEONA_CHANNEL=beta` puts the beta notice and feedback link on the pre-release deploys only. |
| 20 | Accessibility statement | **Built** | `ACCESSIBILITY.md`, linked in the page footer alongside Privacy and Terms, with a "report a barrier" route to the support mailbox. Seven text colours were measured failing WCAG AA and corrected; the canvas map gained a screen-reader description. Two regression tests keep both true. |
| 21 | Custom 404 page | **Built** | `public/404.html` in the house style with a way back. A single-page app answers every path with the same page, whose relative script paths broke below the root - a mistyped `/a/b` link was a blank white page. A one-line head script sends unknown paths to the 404. Honest limit: the status code stays 200 (Streamlit cannot set it), so the page is `noindex`. Checked by `tools/site_check.mjs`. |
| 22 | CTA above the fold | **Verified** | `site_check` asserts it at 1440x900 (bottom 587px) and 390x844 (454px). |
| 23 | Meta description | **Built** | One page, one description. `tools/public_files.py` writes it into the served HTML at start, with title and canonical URL - crawlers do not run the JavaScript that sets Streamlit's own title. |
| 24 | Open Graph image | **Built** | `public/og-image.png` (1200x627) plus og:/twitter: tags, so a LinkedIn post of the link shows the card. Its audit figure is the latest live run's, not a stale one. |
| 25 | Favicon set | **Built** | Crest rendered to `favicon.png` (64px, also the tab icon) and `apple-touch-icon.png` (180px). |
| 26 | robots.txt and sitemap.xml | **Built** | Written at start for `MELEONA_SITE_URL`; served from the site root. |
| 27 | Alt text on every image | **Verified** | `site_check` fails on any `<img>` without alt. Photos are decorative CSS backgrounds; the crest SVG carries `role="img"` and a label. |
| 28 | Mobile breakpoints | **Verified** | 390px: no sideways scroll, crest and CTA on screen; short-viewport tiers in `main.py`. |
| 29 | Sticky mobile CTA | **N/A by design** | The page has one job and its CTA is above the fold on a phone; a pinned bar would cover the headline verdict on a 390px screen. Revisit if a real visitor study says otherwise. |
| 30 | Loads under 2 seconds | **Built** | First paint under 200ms: the hero's boot skeleton, painted from the served HTML before Streamlit's JavaScript (2026-10-19; before that a plain screen, and on a phone ~7 s of blank page). Disclosed rather than hidden: the hero follows in ~2 s on a laptop, ~5.6 s on a 9 Mbps phone, and the live-data headline after that; an "Updating…" sign shows while the page works. |
| 31 | Form error states | **Built** | The ticker box is the form. Mistyped or dead symbols, more than 25 symbols, an invalid symbol, a constant-price asset and fewer than two usable symbols each get a specific message, each covered by a test. |
| 32 | Thank-you pages | **N/A** | Nothing is submitted: no sign-up, no order, no form post. |
| 33 | Privacy pages | **Built** | `PRIVACY.md`, `TERMS.md`, `ACCESSIBILITY.md`, linked from the footer. |
| 34 | Cookie banner | **N/A** | The only cookie is Streamlit's strictly-necessary XSRF token, exempt from consent; no analytics or advertising cookies exist. Adding analytics would make a banner necessary. |
| 35 | Analytics installed | **N/A by decision** | `PRIVACY.md` promises no analytics SDK and a test asserts Streamlit telemetry is off. Traffic is read from the host's own request metrics instead. |
| 36 | Real contact details | **Built** | `meleona.support@gmail.com`; `site_check` fails if the page stops showing it. |
| 37 | Security sweep | **Built** | XSS: `VALID_TICKER` funnel + escapes. CSRF: XSRF on. SSRF: `src/netguard.py` allowlist. Path traversal, exposed `.env`/`.git`/logs/source/secrets, permissive CORS, source maps: `tools/security_probe.py`, run on every live audit (all pass). Dependencies: `pip-audit` clean. Code: `bandit` clean. N/A, no surface: uploads, passwords and resets, sessions/JWT, default credentials, IDOR/BOLA, payments, webhooks. Known limit: no CSP/X-Frame-Options headers - Streamlit cannot set them and the read-only page has nothing to clickjack. |

## Before the link goes public

- [ ] `python -m tests.test_engine` green (includes the full-app boot test).
- [ ] `python -m tests.batch_audit` green (live-data math audit).
- [ ] Kill switch tested on the host: set `MELEONA_MAINTENANCE=1`, load the
      page, unset it, load again.
- [ ] Footer shows the support address, Privacy, Terms and a session reference.
- [ ] Trigger one deliberate error in the preview deploy and confirm the
      reference in the footer finds it in `logs/meleona.log`.
- [ ] Budget counter sane after a browsing session (`budget_status()`).
- [x] **Create `meleona.support@gmail.com`** - done 2026-08-26.
- [x] **Turn on forwarding to the personal inbox** - done 2026-08-26.
- [x] Test mail sent to the project address and confirmed arriving in both
      mailboxes (2026-08-26).
- [ ] Set `MELEONA_CHANNEL=beta` on the beta deploy, and unset it for public.
- [x] Warm cache: automatic. The Procfile runs `python -m tools.warm_cache --every 5`
      beside the app, so no preset expires before a visitor picks it - every pull
      its page makes, full histories and the security master included (RUNBOOK.md,
      *After every deploy*).
- [ ] Read `ACCESSIBILITY.md` once before sharing the link - it is the page a
      complainant's lawyer reads first, and it must stay true of the app.
- [ ] Deploy to Railway (`RUNBOOK.md` §6c) with `MELEONA_CHANNEL=beta`
      and confirm `/_stcore/health` answers.
- [ ] Walk the deployed page once end to end, then trigger one deliberate error
      and confirm the footer reference finds it in the logs.

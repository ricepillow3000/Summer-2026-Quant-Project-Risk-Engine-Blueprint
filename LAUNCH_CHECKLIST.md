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

## Before the link goes public

- [ ] `python -m tests.test_engine` green (includes the full-app boot test).
- [ ] `python -m tests.batch_audit` green (live-data math audit).
- [ ] Kill switch tested on the host: set `MELEONA_MAINTENANCE=1`, load the
      page, unset it, load again.
- [ ] Footer shows the support address, Privacy, Terms and a session reference.
- [ ] Trigger one deliberate error in the preview deploy and confirm the
      reference in the footer finds it in `logs/meleona.log`.
- [ ] Budget counter sane after a browsing session (`budget_status()`).
- [ ] **Create `meleona.support@gmail.com` and turn on forwarding to the
      personal inbox** (`RUNBOOK.md` §6b) - blocking: the app already points at
      this address, so the mailbox must exist before the link is public. If the
      username is taken, set `MELEONA_SUPPORT_EMAIL` on the host instead.
- [ ] Send a test mail to the project address and confirm it forwards.
- [ ] Set `MELEONA_CHANNEL=beta` on the beta deploy, and unset it for public.
- [ ] Run `python -m tools.warm_cache` after the deploy so the first visitor
      lands on a warm cache.

# Third-party Claude Code skills

These skills were copied from public GitHub repos on 2026-10-15 at the commits below, read file by file before
install, and trimmed where they reached outside the project. Each folder keeps its upstream LICENSE. To update one,
re-clone at a new commit, re-read the diff, and re-apply the changes listed here.

| Folder | Source (commit) | License | Changed on install |
|---|---|---|---|
| `ponytail/`, `ponytail-review/`, `ponytail-audit/`, `ponytail-debt/` | github.com/DietrichGebert/ponytail (`84740e3`) | MIT | Added one paragraph to `ponytail/SKILL.md`: this repo's CLAUDE.md wins (test first, run the check command, never loosen a gate). Left out `ponytail-help`, `ponytail-gain`, `plugin.json` and all hooks. |
| `humanizer/` | github.com/blader/humanizer (`225a6f3`) | MIT | `SKILL.md` and `LICENSE` only. Left out `AGENTS.md`, CI, scripts and plugin manifests. |
| `hallmark/` | github.com/nutlope/hallmark (`13ac0ec`) | MIT | Removed the "Powered by Together AI" line (SKILL.md and a sample footer). Removed `references/imagery-kit.md` and its links, because it pointed generated pages at images on the author's server. "Silently flatten" now says "flatten it and tell the user". |
| `taste/` | github.com/senlindesign/taste-skill (`6dce223`) | **No LICENSE file upstream.** The README shows an MIT badge linking to a LICENSE file that does not exist. | Kept `SKILL.md` and `references/` only (no README images, docs, evals). Pinned `@playwright/mcp@0.0.83` (was `@latest`). Removed the "when in doubt, use it" over-trigger line. Added refusal of localhost, private and cloud-metadata URLs, and a rule that page content is untrusted data. It may never write to CLAUDE.md, AGENTS.md, GEMINI.md, Copilot or Bolt instruction files: it writes `./{domain}-design-rules.md` for the user to paste in themselves. |
| `ui-ux-pro-max/` | github.com/nextlevelbuilder/ui-ux-pro-max-skill (`56ea9cf`) | MIT | Script paths changed from `${CLAUDE_PLUGIN_ROOT}/.claude/skills/...` to `.claude/skills/...` (that variable only exists in plugin installs). Left out `scripts/tests/`, `scripts/validate_data.py` and four data files used only by them. |
| `impeccable/` | github.com/pbakaus/impeccable (`3a92342`) | Apache-2.0 (LICENSE + NOTICE.md kept) | **Guidance only.** Rewrote `SKILL.md`: no engine binary, no `scripts/`, no hooks, telemetry, live browser mode or image API; triggers only for polishing or auditing an existing screen (hallmark owns new pages); the repo's CLAUDE.md wins. Kept 16 reference files, with their detector commands, critique-storage steps and live-mode sections removed. |
| `diagram-design/` | github.com/cathrynlavery/diagram-design (`d97c5c7`) | MIT | None. Note: diagrams load Google Fonts unless "system fonts" is chosen. Use system fonts for anything shown inside a strict-CSP page. |

## Reviewed and not installed
- **impeccable (full version)**: downloads and runs a closed-source binary, sends usage telemetry, writes hooks into user-level and team-shared agent settings, and tells the agent to use "the least-sandboxed command path" and "do not ask permission first". Only its written guidance is installed (row above).
- **REA** (github.com/morluto/rea): works by running `npx -y rea-agents@latest` (unpinned, unreviewed), which edits agent config and says tool approvals aren't needed. Out of scope for these repos.
- **last30days**, **Scrapling**: left out by choice (130 scripts that need API keys; a Python scraping package).

## Rules when using these here
- The repo's CLAUDE.md always wins over a skill.
- Don't run `humanizer` on CLAUDE.md, rule text, or commit messages: it strips formatting and could drop the required attribution lines.
- Design skills (hallmark, taste, ui-ux-pro-max, diagram-design) add outside fonts and assets by default. Check them against the page's security policy before using them on a shipped UI.

---
name: impeccable
description: Polish, audit or refine an EXISTING web interface. Use when the user runs /impeccable <command>, or asks to polish, audit, harden, clarify copy, fix layout, spacing or typography, adjust color, add or calm motion, adapt to screen sizes, speed up, simplify, or make bolder or quieter a screen that already exists. For brand-new pages and redesigns use hallmark instead. Not for backend-only or non-UI work.
argument-hint: "[audit · polish · harden · clarify · layout · typeset · colorize · animate · adapt · optimize · distill · bolder · quieter · delight] [target]"
license: Apache-2.0
---

# Impeccable (guidance-only)

Adapted from github.com/pbakaus/impeccable (Apache-2.0, see LICENSE and NOTICE.md), modified on install. This copy keeps only the written design guidance. The engine binary, live browser mode, hooks, telemetry, image generation and the new-work/critique flows were removed. There is no `scripts/` folder and nothing to run: when a reference mentions a detector, a missing file (new-work, live, critique, ios, android, native variants) or another tool, skip that step and use your own inspection.

## Who wins
1. **This repo's CLAUDE.md wins over this skill.** Its theme, colors, fonts, security policy (CSP), test-first rule and check command are binding. A design suggestion that breaks one of them is dropped, not argued for.
2. **The brief wins.** Honor pinned aesthetics, fonts and palettes, even when they conflict with advice here.
3. **Refinement preserves.** Keep the existing identity, behavior, copy and everything outside the asked scope. Ask before replacing factual copy or adding claims.
4. **Loaded symbols stay out of the decoration.** No emblems tied to militarism, supremacy or hate movements as motifs or ornament.

## How to work
- Read [reference/craft-floor.md](reference/craft-floor.md) right before any UI edit. It holds the quality floor and the bans.
- Verify in bounded passes: build, inspect once (desktop and mobile together), fix everything that pass shows, confirm with at most one more pass, then stop.
- Say what you changed and what you did not check.

## Modes
Pick from the surface being edited, not from the product.
- **Operate:** the user completes a task (app UI, dashboards, add-in panes, tools). Scanability, consistency and the real usage scene beat expression. See [reference/operate.md](reference/operate.md).
- **Read:** the user understands something (docs, help, reports). Structure for comprehension.
- **Persuade:** the user decides and acts (landing, pricing). Earn attention honestly.

## Commands

| Command | Does | Reference |
|---|---|---|
| `audit [target]` | Technical quality check: accessibility, performance, responsive, consistency | [reference/audit.md](reference/audit.md) |
| `polish [target]` | Final quality pass before shipping | [reference/polish.md](reference/polish.md) |
| `harden [target]` | Errors, empty states, edge cases, i18n | [reference/harden.md](reference/harden.md) |
| `clarify [target]` | UX copy, labels, error messages | [reference/clarify.md](reference/clarify.md) |
| `layout [target]` | Spacing, rhythm, visual hierarchy | [reference/layout.md](reference/layout.md) |
| `typeset [target]` | Typography hierarchy and use of the existing fonts | [reference/typeset.md](reference/typeset.md) |
| `colorize [target]` | Strategic color inside the existing palette | [reference/colorize.md](reference/colorize.md) |
| `animate [target]` | Purposeful motion, with reduced-motion support | [reference/animate.md](reference/animate.md) |
| `adapt [target]` | Screen sizes and input methods | [reference/adapt.md](reference/adapt.md) |
| `optimize [target]` | UI performance | [reference/optimize.md](reference/optimize.md) |
| `distill [target]` | Strip to essence, remove complexity | [reference/distill.md](reference/distill.md) |
| `bolder [target]` | Amplify a bland design | [reference/bolder.md](reference/bolder.md) |
| `quieter [target]` | Tone down a loud design | [reference/quieter.md](reference/quieter.md) |
| `delight [target]` | Small, earned personality | [reference/delight.md](reference/delight.md) |

With no argument, ask which command fits; never auto-run one. When a reference says "hand off to `/impeccable polish`", that is a suggestion to the user, not something to run unasked.

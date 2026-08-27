# Security policy

_Contact: meleona.support@gmail.com (subject line: `SECURITY`)_

Meleona is a public, read-only market-risk dashboard run by one student. It has
no accounts, no roles, no secrets and no database, so the threat model is
resource exhaustion and injection, not authorization. `CLAUDE.md` lists the
controls that follow from that and the two that are deliberately absent.

## Reporting a vulnerability

Email the address above with what the issue is and how to reproduce it. Please
give a reasonable window to fix it before publishing. Expect a reply in days,
not minutes - there is no service-level guarantee and no bounty, but credit
goes in the commit that fixes it. `SUPPORT.md` covers non-security reports.

## Supported versions

Only the currently deployed `main` is supported. There are no releases,
branches or backports.

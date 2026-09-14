import { defineRailway, github, project, service } from "railway/iac";

// Railway Infrastructure as Code for the live Meleona service. Replaces the
// deprecated railway.json (Config as Code stops being read 2026-12-01).
//
// Generated from `railway config pull` on 2026-09-14, i.e. from what the
// production service ACTUALLY runs, not from railway.json - whose builder
// (NIXPACKS), health check and restart policy were never in effect: the live
// service builds with RAILPACK, has no health check, and restarts ON_FAILURE
// up to 10 times. Two corrections to the raw import:
//   - source follows branch `main` instead of pinning the commit that was live
//     at import time, so pushes keep deploying rather than freezing the site;
//   - builder and the public domain are stated explicitly, so an apply can
//     never silently switch builders or detach the production URL.
//
// This file is NOT read on push. It only takes effect through
// `railway config plan` (read-only preview) and `railway config apply`.
export default defineRailway(() => {
  const web = service("Summer-2026-Quant-Project-Risk-Engine-Blueprint", {
    source: github("ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint", {
      branch: "main",
    }),
    build: { builder: "RAILPACK", buildEnvironment: "V3" },
    replicas: { sfo: 1 },
    networking: {
      serviceDomains: {
        "summer-2026-quant-project-risk-engine-blueprint-production.up.railway.app": { port: 8080 },
      },
      privateNetworkEndpoint: "summer-2026-quant-project-risk-e",
    },
  });

  return project("adaptable-serenity", {
    resources: [web],
  });
});

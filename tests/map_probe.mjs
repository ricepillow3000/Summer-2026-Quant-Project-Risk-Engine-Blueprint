/*
  Map probe - runs the SHIPPED Risk Topology simulation code.

  It does not re-implement anything: it slices the deterministic math region
  out of prototypes/war_room.html (mulberry32 -> computeStats, i.e. the same
  text the browser executes), splices in a payload, and reports moments,
  ring masses and HUD statistics as JSON so tests/batch_audit.py can check
  them against closed-form OU results computed in Python.

  usage: node tests/map_probe.mjs <payload.json> <calName> <nPaths> <seed>
*/
import { readFileSync } from 'node:fs';

const [payloadPath, calName = 'base', nPaths = '20000', seed = '12345'] = process.argv.slice(2);
const html = readFileSync(new URL('../prototypes/war_room.html', import.meta.url), 'utf8');

const START = 'function mulberry32(';
const STOP = '/* ---------- canvas plumbing ---------- */';
const i = html.indexOf(START), j = html.indexOf(STOP);
if (i < 0 || j < 0) throw new Error('math region markers not found in war_room.html');
let code = html.slice(i, j);

// swap the file's demo block for the payload under audit, exactly the way
// main.py's war_room_html() splices the live engine payload
const B = '/* __PAYLOAD_BEGIN__ */', E = '/* __PAYLOAD_END__ */';
const b = code.indexOf(B), e = code.indexOf(E);
if (b < 0 || e < 0) throw new Error('payload markers not found inside the math region');
const payload = JSON.parse(readFileSync(payloadPath, 'utf8'));
code = code.slice(0, b + B.length) + '\nconst DEMO = ' + JSON.stringify(payload) + ';\n' + code.slice(e);

const probe = new Function(code + `
  return { simulate, fitCloud, computeStats, CAL, DAYS, DT, MU_V, MU_B, DEMO };
`)();

const N = Number(nPaths);
const sim = probe.simulate(N, Number(seed), calName, false);
const cloud = probe.fitCloud(sim.ends, N);
const stats = probe.computeStats(sim);

// moments of the simulated end-state cloud, in the coordinates the OU model
// is stated in: beta level, log volatility
let mb = 0, ml = 0;
for (let k = 0; k < N; k++) { mb += sim.ends[k * 2]; ml += Math.log(sim.ends[k * 2 + 1]); }
mb /= N; ml /= N;
let vb = 0, vl = 0, cbl = 0;
for (let k = 0; k < N; k++) {
  const db = sim.ends[k * 2] - mb, dl = Math.log(sim.ends[k * 2 + 1]) - ml;
  vb += db * db; vl += dl * dl; cbl += db * dl;
}
vb /= N - 1; vl /= N - 1; cbl /= N - 1;

// ring masses: what share of end-states actually sits inside each drawn ring
const mass = r => { let c = 0; for (let k = 0; k < N; k++) if (cloud.maha(sim.ends[k * 2], sim.ends[k * 2 + 1]) <= r) c++; return c / N; };

// breach flags recomputed from kept paths (does a flag mean "touched the
// perimeter at any point", as the caption claims, or only at T+30?)
const keep = probe.simulate(2000, Number(seed) + 1, calName, true);
let anyTouch = 0, endOnly = 0, flagged = 0;
for (let k = 0; k < 2000; k++) {
  const pts = keep.paths[k];
  const touched = pts.some(p => p.v > payload.hazard.volMax || p.b > payload.hazard.betaMax);
  const last = pts[pts.length - 1];
  if (touched) anyTouch++;
  if (last.v > payload.hazard.volMax || last.b > payload.hazard.betaMax) endOnly++;
  if (keep.breached[k]) flagged++;
}

// P&L tail statistics recomputed independently of computeStats
const pnl = Array.from(sim.pnls).sort((a, b) => a - b);
const q = p => pnl[Math.floor(p * N)];
let esSum = 0; const kk = Math.max(1, Math.floor(0.025 * N));
for (let k = 0; k < kk; k++) esSum += pnl[k];

const p = probe.CAL[calName];
console.log(JSON.stringify({
  cal: calName, n: N, days: probe.DAYS, dt: probe.DT,
  params: p, muV: probe.MU_V, muB: probe.MU_B,
  start: payload.base,
  moments: { beta_mean: mb, beta_var: vb, lnvol_mean: ml, lnvol_var: vl, cov_beta_lnvol: cbl },
  rings: { r68: cloud.r68, r95: cloud.r95, r997: cloud.r997,
           mass68: mass(cloud.r68), mass95: mass(cloud.r95), mass997: mass(cloud.r997) },
  stats: { es: stats.es, var95: stats.var95, sharpe: stats.sharpe, pBreach: stats.pBreach,
           hist_n: stats.hist.n, hist_sum: stats.hist.bins.reduce((a, x) => a + x, 0),
           hist_lo: stats.hist.lo, hist_hi: stats.hist.hi },
  pnl_check: { var95: -q(0.05), es975: -esSum / kk, mean: pnl.reduce((a, x) => a + x, 0) / N },
  breach_check: { flagged: flagged / 2000, any_touch: anyTouch / 2000, end_only: endOnly / 2000 },
}));

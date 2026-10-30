"""Monitoring dashboard: one self-contained HTML page built from a run's saved outputs (no model loading, no server).

Reads artifacts/<SUBSET>/{monitoring.json, fleet_status.csv, work_orders.csv, predictions_test.csv, replay_log.csv,
fleet_engine_stats.csv, fault_injection.csv} and artifacts/cost_optimization.json (optional), and writes
<run>/dashboard.html. Opens offline in any browser; all charts are plain SVG.

    python scripts/build_dashboard.py [run_dir]     # default: output/all_run_v3
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEVERITIES = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
EVENTS = ["created", "escalated", "downgraded", "closed"]


def _r(a, d=1):
    return [None if pd.isna(x) else round(float(x), d) for x in a]


def _clean(v):
    # json.dumps can't write NaN/inf/numpy scalars
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _records(df, cols):
    df = df[[c for c in cols if c in df.columns]].copy()
    return [{k: (None if (not isinstance(v, str) and pd.isna(v)) else v) for k, v in row.items()}
            for row in df.to_dict(orient="records")]


def subset_data(art):
    mon = json.loads((art / "monitoring.json").read_text())
    fleet = pd.read_csv(art / "fleet_status.csv")
    orders = pd.read_csv(art / "work_orders.csv")
    pred = pd.read_csv(art / "predictions_test.csv")
    replay = pd.read_csv(art / "replay_log.csv")
    stats = pd.read_csv(art / "fleet_engine_stats.csv")
    tickets = json.loads((art / "sensor_tickets.json").read_text()) if (art / "sensor_tickets.json").exists() else []

    test_traj = {}
    for u, g in pred.sort_values(["unit", "cycle"]).groupby("unit"):
        test_traj[int(u)] = {"c0": int(g.cycle.iloc[0]), "pred": _r(g.rul_pred), "lo": _r(g.rul_lo), "hi": _r(g.rul_hi),
                             "p": _r(g.p_fail, 3), "true": [int(x) for x in g.rul]}

    sev_ix = {s: i for i, s in enumerate(SEVERITIES)}
    replay_traj, events = {}, []
    for u, g in replay.sort_values(["unit", "cycle"]).groupby("unit"):
        replay_traj[int(u)] = {"c0": int(g.cycle.iloc[0]), "pred": _r(g.rul_pred), "lo": _r(g.rul_lo),
                               "hi": _r(g.rul_hi), "p": _r(g.p_fail, 3), "true": [int(x) for x in g.rul],
                               "sev": [sev_ix.get(s, 0) for s in g.severity],
                               "review": [int(bool(x)) for x in g.human_review]}
        ev = g[g.event.isin(EVENTS)]
        for row in ev.itertuples():
            events.append({"unit": int(u), "cycle": int(row.cycle), "event": row.event, "severity": row.severity,
                           "wo": row.work_order_id if isinstance(row.work_order_id, str) else "",
                           "rul": round(float(row.rul_pred), 1), "true": int(row.rul),
                           "why": row.reasons if isinstance(row.reasons, str) else ""})

    fault = pd.read_csv(art / "fault_injection.csv") if (art / "fault_injection.csv").exists() else pd.DataFrame()
    keep = ["subset", "generated_at", "registry_name", "champion_version", "champion_model", "production_version",
            "production_label", "production_engines", "production_metrics", "test_error_by_band", "cv_results",
            "cv_classifier", "cv_interval", "history", "drift_psi", "promotion", "agent_summary", "agent_confusion",
            "fleet_replay", "policy_comparison", "policy_fixed_intervals", "config", "agent_config", "ingest_report",
            "sensors_kept", "sensors_dropped", "selections"]
    meta = {k: mon.get(k) for k in keep}
    for s in (meta.get("selections") or {}).values():
        s.pop("cv_results", None)
    meta["history"] = [{k: h.get(k) for k in ["version", "label", "model", "trained_at", "train_engines",
                                               "train_seconds", "test_rmse_capped", "test_rmse", "test_nasa_score",
                                               "test_interval_coverage", "val_rmse_capped", "val_nasa_capped_mean"]}
                       for h in mon.get("history", [])]
    return _clean({
        **meta,
        "fleet": _records(fleet, ["unit", "cycle", "severity", "work_order_id", "rul_pred", "rul_lo", "rul_hi",
                                  "p_fail", "anomaly", "human_review", "sensor_ticket", "reasons", "review_reasons",
                                  "true_rul", "true_severity"]),
        "orders": _records(orders, ["work_order_id", "type", "unit", "at_cycle", "severity", "priority", "status",
                                    "human_review", "predicted_rul", "rul_lower", "rul_upper", "p_fail_window",
                                    "anomaly", "due_within_cycles", "recommended_action", "reasons",
                                    "review_reasons", "model_version", "agent_version"]),
        "tickets": tickets,
        "fault_injection": _records(fault, list(fault.columns)),
        "engine_stats": _records(stats, list(stats.columns)),
        "test_traj": test_traj,
        "replay_traj": replay_traj,
        "events": events,
    })


def cost_data(run):
    f = run / "artifacts" / "cost_optimization.json"
    if not f.exists():
        return None
    t = json.loads(f.read_text())["table"]
    return _clean([r for r in t if not r["policy"].startswith("agent, cost-tuned")])


def build(run, out=None):
    run = Path(run)
    data = {"subsets": {}, "cost": cost_data(run)}
    for s in SUBSETS:
        art = run / "artifacts" / s
        if (art / "monitoring.json").exists():
            data["subsets"][s] = subset_data(art)
    if not data["subsets"]:
        raise SystemExit(f"no artifacts/<subset>/monitoring.json under {run}")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = Path(out) if out else run / "dashboard.html"
    out.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fleet Monitor</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--raised:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--accent:#2a78d6;--accent-soft:#cde2fb;--band:rgba(42,120,214,.16);
--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;--low:#9a988f;--good-ink:#006300}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--raised:#222221;
--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--accent:#3987e5;--accent-soft:#184f95;
--band:rgba(57,135,229,.24);--low:#6f6d66;--good-ink:#0ca30c}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--raised:#222221;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--accent:#3987e5;--accent-soft:#184f95;--band:rgba(57,135,229,.24);
--low:#6f6d66;--good-ink:#0ca30c}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:1px solid var(--ring);padding:10px 16px;
display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
h1{font-size:17px;margin:0;font-weight:650}
.sub{color:var(--ink2);font-size:12.5px}
.seg{display:inline-flex;border:1px solid var(--ring);border-radius:8px;overflow:hidden;flex-wrap:wrap}
.seg button{border:0;background:transparent;color:var(--ink2);padding:6px 12px;font:inherit;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--accent);color:#fff}
nav.tabs{display:flex;gap:2px;flex-wrap:wrap;padding:0 16px;background:var(--surface);border-bottom:1px solid var(--ring)}
nav.tabs button{border:0;background:none;padding:10px 12px;font:inherit;color:var(--ink2);cursor:pointer;border-bottom:2px solid transparent}
nav.tabs button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
main{padding:16px;max-width:1400px;margin:0 auto}
.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr))}
.g3{grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr))}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px;min-width:0}
.card h2{font-size:14px;margin:0 0 2px;font-weight:620}
.card .note{color:var(--muted);font-size:12px;margin:0 0 10px}
.kpi .v{font-size:26px;font-weight:650;line-height:1.1}
.kpi .l{color:var(--ink2);font-size:12.5px}
.kpi .d{color:var(--muted);font-size:11.5px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--surface)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tbody tr.click{cursor:pointer}
tbody tr.click:hover,tbody tr.sel{background:var(--accent-soft)}
.scroll{max-height:520px;overflow:auto}
.chip{display:inline-flex;align-items:center;gap:5px;padding:1px 8px;border-radius:99px;font-size:11.5px;font-weight:650;
white-space:nowrap;color:#fff}
.chip::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor;opacity:.85}
.s-CRITICAL{background:var(--critical)}.s-HIGH{background:var(--serious)}.s-MEDIUM{background:var(--warning);color:#0b0b0b}
.s-LOW{background:var(--low)}.s-NORMAL{background:var(--good)}
.tag{display:inline-block;border:1px solid var(--ring);border-radius:6px;padding:0 6px;font-size:11.5px;color:var(--ink2);margin:1px 2px 1px 0}
.tag.warn{border-color:var(--serious);color:var(--ink)}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
.filters label{font-size:12.5px;color:var(--ink2);display:inline-flex;gap:5px;align-items:center}
select,input[type=search]{font:inherit;padding:4px 8px;border:1px solid var(--ring);border-radius:6px;background:var(--raised);color:var(--ink)}
.wo{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;background:var(--page);border:1px solid var(--ring);
border-radius:8px;padding:10px;white-space:pre-wrap}
.ok{color:var(--good-ink);font-weight:600}.bad{color:var(--critical);font-weight:600}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin-top:6px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:4px}
svg text{fill:var(--muted);font-size:11px}
.tip{position:fixed;pointer-events:none;z-index:20;background:var(--raised);color:var(--ink);border:1px solid var(--ring);
border-radius:8px;padding:6px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.14);display:none;max-width:280px}
.heat td{text-align:center;font-variant-numeric:tabular-nums;border:2px solid var(--surface);border-radius:4px}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(54px,1fr));gap:4px}
.cell{border-radius:6px;padding:4px 2px;text-align:center;font-size:11px;font-weight:650;color:#fff;cursor:pointer;border:2px solid transparent}
.cell.done{background:var(--grid)!important;color:var(--muted)}
.cell.rv{border-color:var(--ink)}
.feed{max-height:330px;overflow:auto;font-size:12.5px}
.feed div{padding:4px 0;border-bottom:1px solid var(--grid)}
.play{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.play button{font:inherit;padding:5px 14px;border-radius:6px;border:1px solid var(--ring);background:var(--accent);color:#fff;cursor:pointer}
input[type=range]{flex:1;min-width:160px;accent-color:var(--accent)}
.muted{color:var(--muted)}
.intro{background:var(--surface);border:1px solid var(--ring);border-left:3px solid var(--accent);border-radius:8px;padding:8px 12px;margin:0 0 14px;color:var(--ink2);font-size:13px}
.intro b{color:var(--ink)}
.helpbtn{border:1px solid var(--ring);background:var(--raised);color:var(--ink);border-radius:99px;padding:5px 12px;font:inherit;font-size:13px;cursor:pointer}
.helpbtn:hover{border-color:var(--accent)}
#tourSpot{position:fixed;z-index:30;border-radius:10px;box-shadow:0 0 0 3px var(--accent);pointer-events:none;display:none}
.tourDim{position:fixed;z-index:29;background:rgba(0,0,0,.5);display:none}
#tourCard{position:fixed;z-index:31;width:min(360px,calc(100vw - 32px));background:var(--raised);color:var(--ink);border:1px solid var(--ring);border-radius:12px;padding:14px 16px;box-shadow:0 10px 30px rgba(0,0,0,.25);display:none}
#tourCard h3{margin:0 0 6px;font-size:15px}
#tourCard p{margin:0 0 12px;font-size:13.5px;color:var(--ink2)}
#tourCard .row{display:flex;align-items:center;gap:8px}
#tourCard .row .muted{margin-right:auto;font-size:12px}
#tourCard button{font:inherit;font-size:13px;border-radius:6px;padding:5px 12px;cursor:pointer;border:1px solid var(--ring);background:var(--raised);color:var(--ink)}
#tourCard button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:14px}
@media (max-width:600px){header{padding:10px 16px}.kpi .v{font-size:22px}}
</style></head>
<body>
<header>
  <div><h1>Fleet Monitor</h1><div class="sub" id="runinfo"></div></div>
  <div class="seg" id="subsetSeg" role="group" aria-label="Subset"></div>
  <div class="sub" id="modelinfo" style="margin-left:auto"></div>
  <button class="helpbtn" id="helpBtn" aria-label="Start the guided tour">? Guide</button>
</header>
<nav class="tabs" role="tablist" id="tabs"></nav>
<main id="view"></main>
<div class="tip" id="tip"></div>
<div id="tourSpot"></div>
<div class="tourDim"></div><div class="tourDim"></div><div class="tourDim"></div><div class="tourDim"></div>
<div id="tourCard" role="dialog" aria-live="polite"></div>
<script>
const DATA = __DATA__;
const SEV = ["NORMAL","LOW","MEDIUM","HIGH","CRITICAL"];
const SEVVAR = {NORMAL:"--good",LOW:"--low",MEDIUM:"--warning",HIGH:"--serious",CRITICAL:"--critical"};
const TABS = [["overview","Overview"],["fleet","Fleet now"],["orders","Work orders"],["replay","Live replay"],
              ["mlops","Model & MLOps"],["quality","Data quality"],["impact","Agent & impact"]];
const subsets = Object.keys(DATA.subsets);
let state = {subset: subsets[0], tab: "overview", unit: null, sevFilter: "all", reviewOnly: false, status: "all", q: ""};
try { const s = JSON.parse(localStorage.getItem("fleetmon") || "{}"); if (s.tab) state.tab = s.tab; if (subsets.includes(s.subset)) state.subset = s.subset; } catch(e) {}
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const f1 = x => x == null ? "–" : (+x).toFixed(1);
const f0 = x => x == null ? "–" : Math.round(+x).toString();
const pct = (x, d=0) => x == null ? "–" : (100 * x).toFixed(d) + "%";
const chip = s => `<span class="chip s-${s}">${s}</span>`;
const D = () => DATA.subsets[state.subset];
const split = s => (s || "").split(";").map(x => x.trim()).filter(Boolean);

// ---------- tooltip ----------
const tip = $("tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.display = "block";
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8), y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px"; }
function hideTip() { tip.style.display = "none"; }
document.addEventListener("mousemove", e => { const t = e.target.closest("[data-tip]"); if (t) showTip(e, t.dataset.tip); else hideTip(); });

// ---------- SVG helpers ----------
function scale(d0, d1, r0, r1) { const k = (r1 - r0) / ((d1 - d0) || 1); return v => r0 + (v - d0) * k; }
function ticks(lo, hi, n=5) { const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n || 1))); const m = [1,2,5,10].find(m => (hi - lo) / (step * m) <= n) * step;
  const out = []; for (let v = Math.ceil(lo / m) * m; v <= hi + 1e-9; v += m) out.push(+v.toFixed(6)); return out; }

// horizontal bars: rows [{label, value, tip, color, ref}], one measure
function hbars(rows, {fmt = f1, max = null, refLine = null, refLabel = "", w = 520, color = "--accent"} = {}) {
  const rowH = 26, left = 150, right = 56, top = refLine != null ? 16 : 4, h = top + rows.length * rowH + 22;
  const mx = max ?? (Math.max(...rows.map(r => r.value), refLine ?? 0) * 1.05 || 1);
  const x = scale(0, mx, left, w - right);
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img">`;
  for (const t of ticks(0, mx, 4)) s += `<line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${h - 20}" stroke="${css("--grid")}"/><text x="${x(t)}" y="${h - 6}" text-anchor="middle">${fmt(t)}</text>`;
  rows.forEach((r, i) => { const y = top + i * rowH + 5, bw = Math.max(0, x(r.value) - left);
    s += `<text x="${left - 8}" y="${y + 12}" text-anchor="end" style="fill:${css("--ink2")}">${esc(r.label)}</text>`;
    s += `<g data-tip="${esc(r.tip || r.label + ": " + fmt(r.value))}"><rect x="${left}" y="${y - 3}" width="${w - right - left}" height="${rowH - 4}" fill="transparent"/>`;
    s += `<path d="M${left},${y} h${Math.max(0, bw - 4)} a4,4 0 0 1 4,4 v8 a4,4 0 0 1 -4,4 h-${Math.max(0, bw - 4)} z" fill="${css(r.color || color)}"/></g>`;
    s += `<text x="${left + bw + 6}" y="${y + 12}" style="fill:${css("--ink")}">${fmt(r.value)}</text>`; });
  if (refLine != null) s += `<line x1="${x(refLine)}" x2="${x(refLine)}" y1="${top - 4}" y2="${h - 20}" stroke="${css("--critical")}" stroke-dasharray="4 3" stroke-width="1.5"/><text x="${x(refLine)}" y="${top - 6}" text-anchor="middle" style="fill:${css("--ink2")}">${esc(refLabel)}</text>`;
  return s + "</svg>";
}

// RUL trajectory: predicted line + 80% band + actual (dashed); optional severity strip and cursor
function trajChart(t, {cursor = null, h = 260, w = 640} = {}) {
  const n = t.pred.length, c = i => t.c0 + i;
  const L = 40, R = 12, T = 10, B = t.sev ? 40 : 26, ph = h - T - B;
  const ymax = Math.max(130, ...t.true.slice(0, n).map(v => Math.min(v, 200)));
  const x = scale(t.c0, t.c0 + n - 1, L, w - R), y = scale(0, ymax, T + ph, T);
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="Predicted remaining useful life over cycles">`;
  for (const v of ticks(0, ymax, 5)) s += `<line x1="${L}" x2="${w - R}" y1="${y(v)}" y2="${y(v)}" stroke="${css("--grid")}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  for (const v of ticks(t.c0, t.c0 + n - 1, 6)) s += `<text x="${x(v)}" y="${T + ph + 14}" text-anchor="middle">${v}</text>`;
  s += `<line x1="${L}" x2="${w - R}" y1="${y(0)}" y2="${y(0)}" stroke="${css("--axis")}"/>`;
  [15, 30].forEach(v => s += `<line x1="${L}" x2="${w - R}" y1="${y(v)}" y2="${y(v)}" stroke="${css(v === 15 ? "--critical" : "--serious")}" stroke-dasharray="3 4" opacity=".7"/>`);
  let band = ""; for (let i = 0; i < n; i++) band += `${i ? "L" : "M"}${x(c(i)).toFixed(1)},${y(t.hi[i]).toFixed(1)}`;
  for (let i = n - 1; i >= 0; i--) band += `L${x(c(i)).toFixed(1)},${y(t.lo[i]).toFixed(1)}`;
  s += `<path d="${band}Z" fill="${css("--band")}"/>`;
  const line = arr => arr.map((v, i) => `${i ? "L" : "M"}${x(c(i)).toFixed(1)},${y(Math.min(v, ymax)).toFixed(1)}`).join("");
  s += `<path d="${line(t.true)}" fill="none" stroke="${css("--ink2")}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
  s += `<path d="${line(t.pred)}" fill="none" stroke="${css("--accent")}" stroke-width="2"/>`;
  if (t.sev) { const sw = (w - R - L) / n;
    for (let i = 0; i < n; i++) s += `<rect x="${(x(c(i)) - sw / 2).toFixed(1)}" y="${T + ph + 20}" width="${(sw + .6).toFixed(1)}" height="8" fill="${css(SEVVAR[SEV[t.sev[i]]])}"/>`;
    s += `<text x="${L - 6}" y="${T + ph + 28}" text-anchor="end">agent</text>`; }
  if (cursor != null && cursor >= t.c0 && cursor < t.c0 + n) s += `<line x1="${x(cursor)}" x2="${x(cursor)}" y1="${T}" y2="${T + ph}" stroke="${css("--ink")}" stroke-width="1"/>`;
  s += `<line class="hx" x1="0" x2="0" y1="${T}" y2="${T + ph}" stroke="${css("--muted")}" visibility="hidden"/>`;
  s += `<rect class="hit" x="${L}" y="${T}" width="${w - L - R}" height="${ph}" fill="transparent" data-c0="${t.c0}" data-n="${n}" data-l="${L}" data-w="${w - L - R}"/>`;
  return s + "</svg>";
}
function bindTraj(el, t) {
  const svg = el.querySelector("svg"); if (!svg) return; const hit = svg.querySelector(".hit"), hx = svg.querySelector(".hx");
  hit.addEventListener("mousemove", e => { const r = hit.getBoundingClientRect(); const n = +hit.dataset.n;
    const i = Math.max(0, Math.min(n - 1, Math.round((e.clientX - r.left) / r.width * (n - 1))));
    const vb = svg.viewBox.baseVal, px = +hit.dataset.l + i / Math.max(1, n - 1) * +hit.dataset.w;
    hx.setAttribute("x1", px); hx.setAttribute("x2", px); hx.setAttribute("visibility", "visible");
    showTip(e, `<b>cycle ${t.c0 + i}</b><br>predicted RUL ${f1(t.pred[i])} <span class="muted">(80% range ${f0(t.lo[i])}–${f0(t.hi[i])})</span><br>actual RUL ${t.true[i]}<br>P(fail ≤ 30) ${(+t.p[i]).toFixed(3)}${t.sev ? "<br>agent: " + SEV[t.sev[i]] + (t.review && t.review[i] ? " · review" : "") : ""}`);
    e.stopPropagation(); });
  hit.addEventListener("mouseleave", () => { hx.setAttribute("visibility", "hidden"); hideTip(); });
}
const trajLegend = sev => `<div class="legend"><span><i style="background:var(--accent)"></i>predicted RUL</span><span><i style="background:var(--band);height:10px"></i>80% range</span><span><i style="background:var(--ink2)"></i>actual RUL (dashed, known only in evaluation)</span><span><i style="background:var(--critical)"></i>15</span><span><i style="background:var(--serious)"></i>30 cycles</span>${sev ? SEV.map(s => `<span><i style="background:var(${SEVVAR[s]});height:8px"></i>${s}</span>`).join("") : ""}</div>`;

// ---------- header / nav ----------
function renderHeader() {
  $("subsetSeg").innerHTML = subsets.map(s => `<button aria-pressed="${s === state.subset}" data-s="${s}">${s}</button>`).join("");
  $("subsetSeg").onclick = e => { const b = e.target.closest("button"); if (!b) return; state.subset = b.dataset.s; state.unit = null; render(); };
  const d = D();
  $("runinfo").textContent = `Run ${String(d.generated_at || "").replace("T", " ").slice(0, 16)} UTC · NASA C-MAPSS`;
  $("modelinfo").innerHTML = `Production: <b>${esc(d.registry_name)} v${d.production_version}</b> (${esc(d.production_label)}) · champion v${d.champion_version} · ${esc(d.champion_model)}`;
  $("tabs").innerHTML = TABS.map(([k, l]) => `<button role="tab" aria-selected="${k === state.tab}" data-t="${k}">${l}</button>`).join("");
  $("tabs").onclick = e => { const b = e.target.closest("button"); if (!b) return; state.tab = b.dataset.t; render(); };
}
function render() {
  try { localStorage.setItem("fleetmon", JSON.stringify({tab: state.tab, subset: state.subset})); } catch(e) {}
  stopPlay(); renderHeader(); $("view").innerHTML = ""; VIEWS[state.tab]();
  $("view").insertAdjacentHTML("afterbegin", `<p class="intro">${INTRO[state.tab]}</p>`);
}
const kpi = (v, l, d = "") => `<div class="card kpi"><div class="v">${v}</div><div class="l">${l}</div>${d ? `<div class="d">${d}</div>` : ""}</div>`;

// ---------- views ----------
const VIEWS = {};
VIEWS.overview = () => {
  const rows = subsets.map(s => { const d = DATA.subsets[s], m = d.production_metrics || {}, a = d.agent_summary || {}, pc = d.policy_comparison || {};
    const ours = Object.entries(pc).find(([k]) => k.startsWith("predictive")), safe = Object.entries(pc).find(([k]) => k.includes("0% failures"));
    const open = d.orders.filter(o => o.status === "OPEN").length, rev = d.orders.filter(o => o.status === "NEEDS_REVIEW").length;
    const crit = d.fleet.filter(f => f.severity === "CRITICAL").length;
    return {s, d, m, a, ours: ours && ours[1], safe: safe && safe[1], open, rev, crit}; });
  const tot = k => rows.reduce((a, r) => a + r[k], 0);
  const engines = rows.reduce((a, r) => a + r.d.fleet.length, 0);
  const urgent = rows.reduce((a, r) => a + (r.a.urgent_high_or_critical || 0), 0), urgentN = rows.reduce((a, r) => a + (r.a.urgent_engines || 0), 0);
  const fa = rows.reduce((a, r) => a + (r.a.false_alarms_true_rul_gt_low || 0), 0);
  $("view").innerHTML = `<div class="kpis">
    ${kpi(engines, "engines monitored", "test fleets, all four subsets")}
    ${kpi(tot("open"), "work orders open", "created automatically")}
    ${kpi(tot("rev"), "need human review", "exceptions only")}
    ${kpi(tot("crit"), "CRITICAL now", "remove from service")}
    ${kpi(`${urgent}/${urgentN}`, "near-failure engines at HIGH+", "true RUL ≤ 30")}
    ${kpi(fa, "false alarms", "HIGH+ with true RUL > 100")}</div>
  <div class="card"><h2>All subsets at a glance</h2><p class="note">Production model per subset, scored on the held-out NASA test engines. Click a row to open that subset.</p>
  <div class="scroll"><table><thead><tr><th>Subset</th><th>Conditions · faults</th><th>Model</th><th class="n">RMSE (capped)</th><th class="n">RMSE (true)</th><th class="n">80% range coverage</th>
  <th class="n">Near-failure at HIGH+</th><th class="n">False alarms</th><th class="n">Open · review</th><th class="n">Life used: agent vs fixed</th></tr></thead><tbody>
  ${rows.map(r => `<tr class="click" data-s="${r.s}"><td><b>${r.s}</b></td><td>${r.d.config.condition_norm ? 6 : 1} · ${r.d.config.fault_modes}</td><td>${esc(r.d.champion_model)} v${r.d.production_version}</td>
  <td class="n">${f1(r.m.test_rmse_capped)}</td><td class="n">${f1(r.m.test_rmse)}</td><td class="n">${pct(r.m.test_interval_coverage)}</td>
  <td class="n">${r.a.urgent_high_or_critical}/${r.a.urgent_engines}</td><td class="n">${r.a.false_alarms_true_rul_gt_low}</td><td class="n">${r.open} · ${r.rev}</td>
  <td class="n">${r.ours ? f0(r.ours.life_used_pct) + "%" : "–"} vs ${r.safe ? f0(r.safe.life_used_pct) + "%" : "–"}</td></tr>`).join("")}</tbody></table></div></div>
  <div class="grid g2" style="margin-top:14px">
   <div class="card"><h2>Fleet severity now, by subset</h2><p class="note">Agent severity at each test engine's latest cycle.</p><div id="ovSev"></div></div>
   <div class="card"><h2>What happens to each engine</h2><p class="note">Replay of the validation engines to failure (not used in training). Fixed schedule = longest interval with no failures in past history.</p><div id="ovPol"></div></div></div>`;
  $("view").querySelector("tbody").onclick = e => { const tr = e.target.closest("tr"); if (tr) { state.subset = tr.dataset.s; state.tab = "fleet"; render(); } };
  $("ovSev").innerHTML = rows.map(r => stackRow(r.s, r.d.fleet)).join("") + sevLegend();
  $("ovPol").innerHTML = hbars(rows.flatMap(r => [
    {label: `${r.s} agent`, value: r.ours ? r.ours.life_used_pct : 0, tip: `${r.s} predictive agent: ${f1(r.ours?.life_used_pct)}% of life used, ${r.ours?.unplanned_failures} unplanned failures`},
    {label: `${r.s} fixed`, value: r.safe ? r.safe.life_used_pct : 0, color: "--low", tip: `${r.s} fixed interval: ${f1(r.safe?.life_used_pct)}% of life used, ${r.safe?.unplanned_failures} unplanned failures`}]),
    {max: 100, fmt: v => f0(v) + "%"}) + `<div class="legend"><span><i style="background:var(--accent);height:8px"></i>predictive agent</span><span><i style="background:var(--low);height:8px"></i>fixed interval</span><span>both: 0 unplanned failures</span></div>`;
};
function stackRow(label, fleet) {
  const n = fleet.length; let x = 0, s = `<div style="display:flex;align-items:center;gap:8px;margin:6px 0"><div style="width:52px;font-size:12px;color:var(--ink2)">${label}</div><div style="flex:1;display:flex;gap:2px;height:18px">`;
  for (const sv of SEV) { const k = fleet.filter(f => f.severity === sv).length; if (!k) continue;
    s += `<div data-tip="${label} · ${sv}: ${k} of ${n} engines" style="flex:${k};background:var(${SEVVAR[sv]});border-radius:4px;min-width:3px"></div>`; }
  return s + `</div><div style="width:34px;font-size:12px;text-align:right" class="muted">${n}</div></div>`;
}
const sevLegend = () => `<div class="legend">${SEV.map(s => `<span><i style="background:var(${SEVVAR[s]});height:8px"></i>${s}</span>`).join("")}</div>`;

VIEWS.fleet = () => {
  const d = D(), fl = d.fleet;
  const cnt = s => fl.filter(f => f.severity === s).length;
  $("view").innerHTML = `<div class="kpis">${kpi(fl.length, "engines monitored", "test fleet, latest cycle")}${kpi(cnt("CRITICAL"), "CRITICAL", "remove from service now")}
    ${kpi(cnt("HIGH"), "HIGH", "schedule before due date")}${kpi(cnt("MEDIUM"), "MEDIUM", "inspect")}${kpi(fl.filter(f => f.human_review).length, "flagged for review", "exceptions for a human")}
    ${kpi(fl.filter(f => f.sensor_ticket).length, "sensor tickets", "stuck / dead / out-of-range sensor")}</div>
  <div class="card" style="margin-bottom:14px"><h2>Severity mix</h2>${stackRow(state.subset, fl)}${sevLegend()}</div>
  <div class="two"><div class="card"><h2>Fleet status</h2><p class="note">One row per engine, latest cycle. Click an engine for its history.</p>
   <div class="filters"><label>Severity <select id="fSev"><option value="all">all</option>${SEV.slice().reverse().map(s => `<option ${state.sevFilter === s ? "selected" : ""}>${s}</option>`).join("")}</select></label>
   <label><input type="checkbox" id="fRev" ${state.reviewOnly ? "checked" : ""}> review only</label><span class="muted" id="fCount"></span></div>
   <div class="scroll"><table><thead><tr><th class="n">Engine</th><th class="n">Cycle</th><th>Severity</th><th class="n">RUL</th><th>80% range</th><th class="n">P(fail)</th><th>Flags</th><th class="n">Actual</th></tr></thead><tbody id="fBody"></tbody></table></div></div>
   <div class="card" id="engine"></div></div>`;
  const body = () => { const rows = fl.filter(f => (state.sevFilter === "all" || f.severity === state.sevFilter) && (!state.reviewOnly || f.human_review))
      .sort((a, b) => SEV.indexOf(b.severity) - SEV.indexOf(a.severity) || a.rul_pred - b.rul_pred);
    $("fCount").textContent = `${rows.length} engines`;
    $("fBody").innerHTML = rows.map(f => `<tr class="click ${f.unit === state.unit ? "sel" : ""}" data-u="${f.unit}"><td class="n">${f.unit}</td><td class="n">${f.cycle}</td><td>${chip(f.severity)}</td>
      <td class="n">${f0(f.rul_pred)}</td><td>${rangeBar(f.rul_lo, f.rul_pred, f.rul_hi)}</td><td class="n">${(+f.p_fail).toFixed(2)}</td>
      <td>${f.human_review ? `<span class="tag warn" data-tip="${esc(f.review_reasons)}">review</span>` : ""}${f.anomaly ? '<span class="tag">anomaly</span>' : ""}${f.sensor_ticket ? '<span class="tag warn">sensor</span>' : ""}</td>
      <td class="n muted">${f.true_rul}</td></tr>`).join(""); };
  body();
  $("fSev").onchange = e => { state.sevFilter = e.target.value; body(); };
  $("fRev").onchange = e => { state.reviewOnly = e.target.checked; body(); };
  $("fBody").onclick = e => { const tr = e.target.closest("tr"); if (!tr) return; state.unit = +tr.dataset.u; body(); engineView(); };
  if (state.unit == null || !fl.some(f => f.unit === state.unit)) state.unit = fl.slice().sort((a, b) => SEV.indexOf(b.severity) - SEV.indexOf(a.severity) || a.rul_pred - b.rul_pred)[0].unit;
  body(); engineView();
};
function rangeBar(lo, p, hi) { const x = v => Math.max(0, Math.min(100, v / 130 * 100));
  return `<div data-tip="80% range ${f0(lo)}–${f0(hi)} cycles, predicted ${f0(p)}" style="position:relative;width:110px;height:12px;margin-top:3px;background:var(--grid);border-radius:3px">
  <div style="position:absolute;left:${x(lo)}%;width:${Math.max(1, x(hi) - x(lo))}%;top:2px;height:8px;background:var(--band);border-radius:3px"></div>
  <div style="position:absolute;left:calc(${x(p)}% - 1px);width:3px;top:0;height:12px;background:var(--accent);border-radius:2px"></div></div>`; }
function engineView() {
  const d = D(), u = state.unit, f = d.fleet.find(x => x.unit === u), t = d.test_traj[u];
  const os = d.orders.filter(o => o.unit === u);
  $("engine").innerHTML = `<h2>Engine ${u} ${chip(f.severity)}</h2><p class="note">Cycle ${f.cycle} · actual RUL ${f.true_rul} (true severity ${f.true_severity}) · the actual line is test ground truth, never seen by the model.</p>
   <div id="engChart">${t ? trajChart(t) : ""}</div>${trajLegend(false)}
   <h2 style="margin-top:14px">Why</h2><div>${split(f.reasons).map(r => `<span class="tag">${esc(r)}</span>`).join("") || '<span class="muted">no rule fired</span>'}</div>
   ${f.review_reasons ? `<div style="margin-top:6px">${split(f.review_reasons).map(r => `<span class="tag warn">${esc(r)}</span>`).join("")}</div>` : ""}
   <h2 style="margin-top:14px">Work order</h2>${os.length ? os.map(woCard).join("") : '<p class="muted">No order: the agent logs and watches this engine.</p>'}`;
  if (t) bindTraj($("engChart"), t);
}
function woCard(o) {
  return `<div class="wo">${esc(o.work_order_id)} · ${o.type} · ${o.severity} · ${o.priority} · ${o.status}
Engine ${o.unit} · cycle ${o.at_cycle} · due within ${o.due_within_cycles} cycles
RUL ${f1(o.predicted_rul)} (range ${f0(o.rul_lower)}–${f0(o.rul_upper)}) · P(fail ≤ 30) = ${(+o.p_fail_window).toFixed(3)}
Action: ${esc(o.recommended_action)}
Why: ${esc(o.reasons)}${o.review_reasons ? "\nReview: " + esc(o.review_reasons) : ""}
Model ${esc(o.model_version)} · agent ${esc(o.agent_version)}</div>`;
}

VIEWS.orders = () => {
  const d = D(), os = d.orders;
  const by = s => os.filter(o => o.severity === s).length;
  $("view").innerHTML = `<div class="kpis">${kpi(os.length, "work orders", "all created by the agent")}${kpi(os.filter(o => o.status === "OPEN").length, "OPEN", "dispatched automatically")}
   ${kpi(os.filter(o => o.status === "NEEDS_REVIEW").length, "NEEDS REVIEW", "human exception")}${kpi(by("CRITICAL"), "P1 · CRITICAL")}${kpi(by("HIGH"), "P2 · HIGH")}${kpi(by("MEDIUM"), "P3 · MEDIUM")}</div>
  <div class="card"><h2>Work-order queue</h2><p class="note">Sorted by priority, then due date. Click an order to see the full card.</p>
  <div class="filters"><label>Status <select id="oSt"><option value="all">all</option><option ${state.status === "OPEN" ? "selected" : ""}>OPEN</option><option ${state.status === "NEEDS_REVIEW" ? "selected" : ""}>NEEDS_REVIEW</option></select></label>
  <input type="search" id="oQ" placeholder="search engine, reason…" value="${esc(state.q)}"><span class="muted" id="oCount"></span></div>
  <div class="scroll"><table><thead><tr><th>Order</th><th>Priority</th><th>Status</th><th class="n">Engine</th><th class="n">Due in</th><th class="n">RUL</th><th>Range</th><th>Why</th></tr></thead><tbody id="oBody"></tbody></table></div>
  <div id="oCard" style="margin-top:12px"></div></div>`;
  const body = () => { const q = state.q.toLowerCase();
    const rows = os.filter(o => (state.status === "all" || o.status === state.status) && (!q || JSON.stringify(o).toLowerCase().includes(q)))
      .sort((a, b) => a.priority.localeCompare(b.priority) || a.due_within_cycles - b.due_within_cycles);
    $("oCount").textContent = `${rows.length} orders`;
    $("oBody").innerHTML = rows.map(o => `<tr class="click" data-id="${esc(o.work_order_id)}"><td style="font-family:ui-monospace,Consolas,monospace;font-size:12px">${esc(o.work_order_id)}</td>
     <td>${chip(o.severity)} <span class="muted">${o.priority}</span></td><td>${o.status === "OPEN" ? "OPEN" : `<span class="tag warn">NEEDS REVIEW</span>`}</td><td class="n">${o.unit}</td>
     <td class="n">${o.due_within_cycles === 0 ? "<b>now</b>" : o.due_within_cycles + " cyc"}</td><td class="n">${f0(o.predicted_rul)}</td><td>${rangeBar(o.rul_lower, o.predicted_rul, o.rul_upper)}</td>
     <td style="max-width:360px">${split(o.review_reasons).map(r => `<span class="tag warn">${esc(r)}</span>`).join("")}${split(o.reasons).slice(0, 2).map(r => `<span class="tag">${esc(r)}</span>`).join("")}</td></tr>`).join(""); };
  body();
  $("oSt").onchange = e => { state.status = e.target.value; body(); };
  $("oQ").oninput = e => { state.q = e.target.value; body(); };
  $("oBody").onclick = e => { const tr = e.target.closest("tr"); if (!tr) return; $("oCard").innerHTML = woCard(os.find(o => o.work_order_id === tr.dataset.id)); };
};

// live replay of the validation engines, cycle by cycle
let timer = null;
function stopPlay() { if (timer) { clearInterval(timer); timer = null; } }
VIEWS.replay = () => {
  const d = D(), R = d.replay_traj, units = Object.keys(R).map(Number).sort((a, b) => a - b);
  const maxC = Math.max(...units.map(u => R[u].c0 + R[u].pred.length - 1));
  const stat = Object.fromEntries(d.engine_stats.map(s => [s.unit, s]));
  let cyc = Math.min(150, maxC), sel = state.unit != null && R[state.unit] ? state.unit : units[0];
  const fr = d.fleet_replay || {};
  $("view").innerHTML = `<div class="kpis">${kpi(units.length, "validation engines replayed", "streamed cycle by cycle; cell = predicted RUL")}${kpi(f0(fr.median_rul_at_first_order), "median RUL at first order", "cycles of warning")}
   ${kpi(f0(fr.median_rul_at_first_high), "median RUL at first HIGH")}${kpi(pct((fr.engines_critical_before_failure_pct ?? 0) / 100), "reached CRITICAL before failure")}
   ${kpi(f0(fr.min_cycles_left_at_service), "min cycles left at service", "0 would be a failure")}${kpi(fr.engines_with_downgrade_or_closure ?? "–", "orders downgraded or closed", "hysteresis: 5 calmer cycles")}</div>
  <div class="card" style="margin-bottom:14px"><div class="play"><button id="pBtn">▶ Play</button><b id="pCyc"></b><input type="range" id="pR" min="1" max="${maxC}" value="${cyc}" aria-label="Cycle">
   <label class="sub">speed <select id="pSp"><option value="120">1×</option><option value="40" selected>3×</option><option value="12">10×</option></select></label></div>
   <div class="strip" id="pStrip"></div><div class="legend">${SEV.map(s => `<span><i style="background:var(${SEVVAR[s]});height:8px"></i>${s}</span>`).join("")}<span><i style="background:var(--grid);height:8px"></i>serviced by the agent before failure</span><span>outlined = sent to human review</span></div></div>
  <div class="two"><div class="card"><h2 id="pTitle"></h2><p class="note" id="pNote"></p><div id="pChart"></div>${trajLegend(true)}</div>
   <div class="card"><h2>Agent event log</h2><p class="note">Order created / escalated / downgraded / closed, up to the current cycle (newest first).</p><div class="feed" id="pFeed"></div></div></div>`;
  const draw = () => {
    $("pCyc").textContent = `cycle ${cyc}`; $("pR").value = cyc;
    $("pStrip").innerHTML = units.map(u => { const t = R[u], st = stat[u], end = st && !st.failed ? st.service_cycle : t.c0 + t.pred.length - 1;
      const done = cyc >= end, ii = Math.max(0, Math.min(cyc, end) - t.c0), sv = SEV[t.sev[ii]];
      return `<div class="cell ${done ? "done" : ""} ${!done && t.review[ii] ? "rv" : ""}" data-u="${u}" style="background:var(${SEVVAR[sv]});${sv === "MEDIUM" && !done ? "color:#0b0b0b" : ""}${u === sel ? ";outline:2px solid var(--accent);outline-offset:1px" : ""}"
       data-tip="engine ${u}${done ? (st && !st.failed ? ` · serviced at cycle ${st.service_cycle}, ${st.cycles_left_at_service} cycles before it would have failed` : ` · failed at cycle ${end}`) : ` · ${sv}<br>predicted RUL ${f0(t.pred[ii])}, actual ${t.true[ii]}`}">${u}<br>${done ? (st && !st.failed ? "serviced" : "FAILED") : f0(t.pred[ii])}</div>`; }).join("");
    const t = R[sel], st = stat[sel];
    $("pTitle").innerHTML = `Engine ${sel}`;
    $("pNote").textContent = st ? `Life ${st.life} cycles · first order at RUL ${st.first_order_rul}, first HIGH at ${st.first_high_rul}, first CRITICAL at ${st.first_critical_rul ?? "–"} · agent service at cycle ${st.service_cycle} (${st.cycles_left_at_service} cycles before failure)` : "";
    $("pChart").innerHTML = trajChart(t, {cursor: cyc}); bindTraj($("pChart"), t);
    const ev = d.events.filter(e => e.cycle <= cyc).sort((a, b) => b.cycle - a.cycle || a.unit - b.unit).slice(0, 150);
    $("pFeed").innerHTML = ev.map(e => `<div><span class="muted">c${e.cycle}</span> · engine <b>${e.unit}</b> · ${e.event} ${chip(e.severity)} <span class="muted">RUL ${f0(e.rul)} (actual ${e.true})</span></div>`).join("") || '<p class="muted">No orders yet.</p>';
  };
  draw();
  $("pR").oninput = e => { cyc = +e.target.value; draw(); };
  $("pStrip").onclick = e => { const c = e.target.closest(".cell"); if (c) { sel = +c.dataset.u; state.unit = sel; draw(); } };
  $("pBtn").onclick = () => { if (timer) { stopPlay(); $("pBtn").textContent = "▶ Play"; return; }
    if (cyc >= maxC) cyc = 1; $("pBtn").textContent = "❚❚ Pause";
    timer = setInterval(() => { cyc++; if (cyc >= maxC) { stopPlay(); $("pBtn").textContent = "▶ Play"; } draw(); }, +$("pSp").value); };
};

VIEWS.mlops = () => {
  const d = D(), p = d.promotion || {}, cfg = d.config || {};
  const psi = Object.entries(d.drift_psi || {}).sort((a, b) => b[1] - a[1]);
  const maxPsi = psi.length ? psi[0][1] : 0;
  const test = p.promotion_test || "superiority (older run)";
  const bootRows = [["P(challenger better)", p.bootstrap_prob_new_better], ["P(worse by > margin)", p.bootstrap_prob_new_worse_by_margin], ["P(NASA clearly worse)", p.bootstrap_prob_nasa_clearly_worse]].filter(r => r[1] != null);
  const cv = d.cv_results || {};
  $("view").innerHTML = `<div class="kpis">${kpi("v" + d.production_version, "production model", esc(d.registry_name) + " · alias production")}${kpi("v" + d.champion_version, "champion", "alias champion")}
   ${kpi(f1((d.production_metrics || {}).test_rmse_capped), "test RMSE (capped at 125)", "true RMSE " + f1((d.production_metrics || {}).test_rmse))}
   ${kpi(pct((d.production_metrics || {}).test_interval_coverage), "80% range coverage", "on test engines")}
   ${kpi((+maxPsi).toFixed(2), "max drift (PSI)", maxPsi > cfg.psi_alert ? '<span class="bad">above ' + cfg.psi_alert + "</span>" : "alert at " + cfg.psi_alert)}
   ${kpi(p.promoted ? '<span class="ok">promoted</span>' : '<span class="bad">kept v' + p.champion_version + "</span>", "last retrain decision", esc(test))}</div>
  <div class="grid g2"><div class="card"><h2>Model versions</h2><p class="note">Every run is tracked (MLflow); rollback = repoint the production alias.</p>
   <div class="scroll"><table><thead><tr><th>Version</th><th>Model</th><th class="n">Train engines</th><th class="n">Val RMSE</th><th class="n">Test RMSE (capped)</th><th class="n">Test RMSE (true)</th><th class="n">Coverage</th><th>Trained</th></tr></thead><tbody>
   ${d.history.map(h => `<tr><td><b>v${h.version}</b> ${h.version === d.production_version ? '<span class="tag">production</span>' : ""}<div class="muted">${esc(h.label)}</div></td><td>${esc(h.model)}</td><td class="n">${h.train_engines}</td>
   <td class="n">${f1(h.val_rmse_capped)}</td><td class="n">${f1(h.test_rmse_capped)}</td><td class="n">${f1(h.test_rmse)}</td><td class="n">${pct(h.test_interval_coverage)}</td><td class="muted">${String(h.trained_at || "").slice(0, 16).replace("T", " ")}</td></tr>`).join("")}</tbody></table></div></div>
  <div class="card"><h2>Retrain &amp; promotion gate</h2><p class="note">Trigger: ≥ ${pct(cfg.retrain_min_new_frac)} new data or PSI > ${cfg.psi_alert}. Gate: ${esc(test)}, ${cfg.promote_min_prob ? "decided at " + pct(cfg.promote_min_prob) + " bootstrap confidence" : ""}.</p>
   <div>${(p.trigger_reasons || []).map(r => `<span class="tag">${esc(r)}</span>`).join("")}</div>
   <table style="margin-top:8px"><tbody>
   <tr><td>Validation RMSE (capped)</td><td class="n">v${p.champion_version} ${f1((p.val_rmse_capped || [])[0])} → v${p.challenger_version} <b>${f1((p.val_rmse_capped || [])[1])}</b></td></tr>
   <tr><td>NASA score per engine (capped)</td><td class="n">${f1((p.val_nasa_capped_mean || [])[0])} → <b>${f1((p.val_nasa_capped_mean || [])[1])}</b></td></tr>
   ${p.bootstrap_rmse_diff_ci90 ? `<tr><td>RMSE change, 90% bootstrap CI</td><td class="n">${f1(p.bootstrap_rmse_diff_ci90[0])} to ${f1(p.bootstrap_rmse_diff_ci90[1])}</td></tr>` : ""}
   ${bootRows.map(r => `<tr><td>${r[0]}</td><td class="n">${pct(r[1], 1)}</td></tr>`).join("")}
   <tr><td>Decision</td><td class="n">${p.promoted ? '<span class="ok">promote v' + p.challenger_version + "</span>" : '<span class="bad">keep v' + p.champion_version + "</span>"}</td></tr></tbody></table></div>
  <div class="card"><h2>Data drift (PSI) per sensor</h2><p class="note">New engines vs training data. Above ${cfg.psi_alert} triggers a retrain.</p><div id="mPsi"></div></div>
  <div class="card"><h2>Model selection (5-fold CV by engine)</h2><p class="note">Capped RMSE, lower is better. The winner becomes the candidate.</p><div id="mCv"></div></div>
  <div class="card"><h2>Error by true-RUL band (test)</h2><p class="note">Errors concentrate above the 125-cycle cap, where degradation hasn't started. Near failure the model is tight.</p><div id="mBand"></div></div>
  <div class="card"><h2>Calibration</h2><p class="note">80% range after conformal calibration (CV), and coverage by RUL band on test.</p><div id="mCal"></div></div></div>`;
  $("mPsi").innerHTML = hbars(psi.slice(0, 10).map(([k, v]) => ({label: k, value: v})), {refLine: cfg.psi_alert, refLabel: "alert " + cfg.psi_alert, fmt: v => (+v).toFixed(2)});
  $("mCv").innerHTML = hbars(Object.entries(cv).map(([k, v]) => ({label: k, value: v.rmse_capped, color: k === d.champion_model ? "--accent" : "--low", tip: `${k}: capped RMSE ${f1(v.rmse_capped)}, NASA/engine ${f1(v.nasa_capped_mean)}`})));
  $("mBand").innerHTML = hbars(Object.entries(d.test_error_by_band || {}).map(([k, v]) => ({label: `RUL ${k} (${v.engines})`, value: v.rmse, color: k.startsWith("(1") && +k.split(",")[0].slice(1) >= 125 ? "--low" : "--accent", tip: `${k}: RMSE ${v.rmse}, bias ${v.bias}, ${v.share_of_sq_error_pct}% of squared error, ${v.engines} engines`})));
  const m = d.production_metrics || {}, ci = d.cv_interval || {};
  $("mCal").innerHTML = hbars([{label: "CV raw", value: ci.raw_coverage * 100, color: "--low"}, {label: "CV calibrated", value: ci.calibrated_coverage * 100},
    {label: "test RUL 0–30", value: m.test_coverage_rul_0_30 * 100}, {label: "test RUL 30–60", value: m.test_coverage_rul_30_60 * 100}, {label: "test RUL 60–125", value: m.test_coverage_rul_60_125 * 100}],
    {max: 100, refLine: 80, refLabel: "target 80%", fmt: v => f0(v) + "%"});
};

VIEWS.quality = () => {
  const d = D(), ir = d.ingest_report || {}, fi = d.fault_injection || [];
  const det = fi.filter(f => f.detected).length;
  const flags = k => d.fleet.filter(f => String(f.review_reasons || "").includes(k)).length;
  $("view").innerHTML = `<div class="kpis">${kpi((ir.train || {}).rows ?? "–", "training rows ingested", `${(ir.train || {}).engines} engines`)}${kpi((ir.test || {}).rows ?? "–", "test rows ingested", `${(ir.test || {}).engines} engines`)}
   ${kpi(d.sensors_kept.length + "/21", "sensors kept", "uninformative ones dropped automatically")}${kpi(d.tickets.length, "sensor tickets (real data)", "CHECK_SENSOR work orders")}
   ${kpi(`${det}/${fi.length}`, "injected faults caught", "stuck / dead / out of range")}</div>
  <div class="grid g2"><div class="card"><h2>Ingest checks</h2><p class="note">Run on every batch before anything else.</p><table><thead><tr><th>Check</th><th class="n">Train</th><th class="n">Test</th></tr></thead><tbody>
   ${["rows", "engines", "missing_cells", "non_numeric_cells", "duplicate_unit_cycle", "cycle_gaps"].map(k => `<tr><td>${k.replace(/_/g, " ")}</td><td class="n">${(ir.train || {})[k] ?? "–"}</td><td class="n">${(ir.test || {})[k] ?? "–"}</td></tr>`).join("")}</tbody></table></div>
  <div class="card"><h2>Sensors</h2><p class="note">Dropped if constant within an operating condition; the rest are z-scored per condition.</p>
   <div>${d.sensors_kept.map(s => `<span class="tag">${s}</span>`).join("")}</div><p class="note" style="margin-top:10px">Dropped</p><div>${d.sensors_dropped.map(s => `<span class="tag muted">${s}</span>`).join("") || "none"}</div>
   <p class="note" style="margin-top:10px">Data-trust flags on the test fleet now: imputed ${flags("imputed")} · unseen condition ${flags("operating condition")} · sensor fault ${flags("sensor fault")}</p></div></div>
  <div class="card" style="margin-top:14px"><h2>Fault-injection test</h2><p class="note">Faults injected into real test engines. The gate must flag them, open a CHECK_SENSOR ticket and send any order to review instead of trusting the bad data.</p>
   <div class="scroll"><table><thead><tr><th class="n">Engine</th><th>Injected</th><th class="n">From cycle</th><th>Caught</th><th class="n">Flagged at</th><th>Gate says</th><th>Severity clean → faulty</th><th>Order</th><th>Review reason</th></tr></thead><tbody>
   ${fi.map(f => `<tr><td class="n">${f.unit}</td><td>${esc(f.injected)}</td><td class="n">${f.from_cycle}</td><td>${f.detected ? '<span class="ok">yes</span>' : '<span class="bad">no</span>'}</td><td class="n">${f.first_flagged_cycle ?? "–"}</td>
   <td>${esc(f.gate_says)}</td><td style="white-space:nowrap">${chip(f.severity_clean_data)} → ${chip(f.severity_faulty_data)}</td><td>${esc(f.order_status)}</td><td>${split(f.review_reasons).map(r => `<span class="tag warn">${esc(r)}</span>`).join("")}</td></tr>`).join("")}</tbody></table></div></div>`;
};

VIEWS.impact = () => {
  const d = D(), a = d.agent_summary || {}, cm = d.agent_confusion || {}, pc = d.policy_comparison || {};
  const mx = Math.max(...SEV.flatMap(t => SEV.map(p => (cm[t] || {})[p] || 0)));
  const cost = (DATA.cost || []).filter(r => r.subset === state.subset);
  const leads = [...new Set(cost.map(r => r.lead))];
  $("view").innerHTML = `<div class="kpis">${kpi(`${a.urgent_high_or_critical}/${a.urgent_engines}`, "near-failure engines at HIGH+", `95% lower bound ${pct(a.urgent_high_or_critical_ci95_low)}`)}
   ${kpi(a.urgent_missed, "near-failure engines missed", "no order at all")}${kpi(a.false_alarms_true_rul_gt_low, "false alarms", "HIGH+ while true RUL > 100")}
   ${kpi(f0(a.exact_severity_match_pct) + "%", "exact severity match")}${kpi(f0(a.review_rate_pct) + "%", "sent to human review")}${kpi(a.work_orders, "work orders created")}</div>
  <div class="grid g2"><div class="card"><h2>Agent severity vs true severity (test)</h2><p class="note">Rows: true severity from actual RUL. Columns: what the agent decided. Right of the diagonal = agent acted earlier than needed (safe side); left = later.</p>
   <table class="heat"><thead><tr><th>true ↓ / agent →</th>${SEV.map(s => `<th style="text-align:center">${s}</th>`).join("")}</tr></thead><tbody>
   ${SEV.map(t => `<tr><th>${t}</th>${SEV.map(p => { const v = (cm[t] || {})[p] || 0, k = v / (mx || 1);
     return `<td data-tip="true ${t}, agent ${p}: ${v} engines" style="background:color-mix(in srgb, var(--accent) ${Math.round(k * 85)}%, var(--surface));color:${k > .5 ? "#fff" : "var(--ink)"};${t === p ? "outline:1.5px solid var(--ink2)" : ""}">${v}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>
  <div class="card"><h2>Maintenance policy on the replayed fleet</h2><p class="note">Validation engines run to failure. Life used = share of each engine's life consumed before service.</p>
   <table><thead><tr><th>Policy</th><th class="n">Unplanned failures</th><th class="n">Life used</th><th class="n">Cycles wasted / engine</th></tr></thead><tbody>
   ${Object.entries(pc).map(([k, v]) => `<tr><td>${k.startsWith("predictive") ? "<b>" + esc(k) + "</b>" : esc(k)}</td><td class="n">${f0(v.unplanned_failures)}/${f0(v.engines)}</td><td class="n">${f1(v.life_used_pct)}%</td><td class="n">${f1(v.mean_cycles_wasted)}</td></tr>`).join("")}</tbody></table>
   <div id="iPol" style="margin-top:8px"></div></div></div>
  ${cost.length ? `<div class="card" style="margin-top:14px"><h2>Maintenance cost (renewal-reward model)</h2><p class="note">Cost per 1,000 operated cycles; planned service = 1, failure = ratio. Lead time: a service planned fewer cycles than this before failure can't be done in time and counts as a failure. Fixed interval T* is the cost-optimal schedule fitted on past history.</p>
   <div class="filters"><label>Lead time <select id="iLead">${leads.map(l => `<option ${l === 10 ? "selected" : ""}>${l}</option>`).join("")}</select> cycles</label></div><div id="iCost"></div></div>` : ""}`;
  $("iPol").innerHTML = hbars(Object.entries(pc).map(([k, v]) => ({label: `${k.replace(/ \(.*\)/, "").replace("fixed interval", "fixed").replace(" agent", "")} · ${f0(v.unplanned_failures)} fail`, value: v.life_used_pct, color: k.startsWith("predictive") ? "--accent" : "--low",
    tip: `${k}: ${f1(v.life_used_pct)}% life used, ${v.unplanned_failures} failures`})), {max: 100, fmt: v => f0(v) + "%", w: 520});
  const costTable = () => { const lead = +$("iLead").value, rs = cost.filter(r => r.lead === lead), ratios = [...new Set(rs.map(r => r.ratio))];
    const pol = [...new Set(rs.map(r => r.policy))];
    $("iCost").innerHTML = `<div class="scroll"><table><thead><tr><th>Policy</th>${ratios.map(r => `<th class="n">failure = ${r}× service</th>`).join("")}</tr></thead><tbody>
     ${pol.map(p => `<tr><td>${p.startsWith("agent") ? "<b>predictive agent</b>" : esc(p)}</td>${ratios.map(r => { const x = rs.find(z => z.policy === p && z.ratio === r);
       const fixed = rs.find(z => z.policy.startsWith("optimal") && z.ratio === r), save = p.startsWith("agent") && fixed ? 1 - x.cost_per_1000_cycles / fixed.cost_per_1000_cycles : null;
       return `<td class="n">${x.cost_per_1000_cycles.toFixed(2)} <span class="muted">(${x.unplanned_failures} fail${p.startsWith("optimal") ? ", T=" + x.t_star : ""})</span>${save != null ? `<div class="${save > 0 ? "ok" : "bad"}">${save > 0 ? "−" : "+"}${Math.abs(save * 100).toFixed(0)}% vs T*</div>` : ""}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`; };
  if (cost.length) { costTable(); $("iLead").onchange = costTable; }
};

const INTRO = {
  overview: "<b>Start here.</b> The whole system on one page: every engine in the four NASA test fleets, the work orders the agent raised, and how it compares with fixed-schedule maintenance. Click a fleet row to dig in.",
  fleet: "<b>Every engine right now</b>, most urgent first. Severity comes from the rule-based agent. Click an engine to see its predicted remaining life over time, why the rules fired and its work order.",
  orders: "<b>The maintenance queue the agent created on its own.</b> OPEN = sent automatically; NEEDS REVIEW = an exception for a human (uncertain prediction, short history, bad sensor data). Click an order for the full card.",
  replay: "<b>Watch the system work live.</b> Press Play: held-out engines run cycle by cycle until they fail. Colours show the agent's severity; an engine turns grey when the agent's planned service falls due, before it would have failed.",
  mlops: "<b>How the model is kept healthy.</b> Every training run is tracked and versioned; new data or drift triggers a retrain, and a statistical gate decides whether the new version replaces the old one.",
  quality: "<b>Bad data is caught before it reaches a decision.</b> Ingest checks, the sensors kept, and a test where real engines get broken sensors on purpose: the gate must flag them and send the work order to a human.",
  impact: "<b>Does it pay off?</b> The agent's severity against the truth, and maintenance on the replayed engines: unplanned failures, share of useful life used, and cost against the best possible fixed schedule.",
};

const TOUR = [
  {tab: "overview", sel: "#subsetSeg", title: "Four fleets", body: "The NASA C-MAPSS data has four fleets of increasing difficulty (FD001 easiest, FD004 hardest). Pick one here; every tab follows the selected fleet."},
  {tab: "overview", sel: "#view .kpis", title: "The headline numbers", body: "Engines monitored, work orders raised automatically, how many need a human, and how many near-failure engines the agent caught."},
  {tab: "overview", sel: "#view .card table", title: "All fleets side by side", body: "Prediction error (RMSE, in cycles), how often the 80% range holds the truth, failures caught, and life used compared with a fixed schedule."},
  {tab: "fleet", sel: "#fBody", title: "Engines, most urgent first", body: "Each row is one engine at its latest cycle: severity, predicted remaining life with its range, and failure probability. Click any row."},
  {tab: "fleet", sel: "#engine", title: "One engine in detail", body: "Blue line = predicted remaining life, shaded band = 80% range, dashed line = the real answer (known only because this is test data). Below: why the rules fired, and the work order."},
  {tab: "orders", sel: "#oBody tr:first-child", title: "The work-order queue", body: "Sorted by priority and due date, so this is the most urgent order. Orders marked NEEDS REVIEW are the only ones a human has to look at. Click any row for the full work-order card."},
  {tab: "replay", sel: "#view .play", title: "Press Play", body: "Engines the model never trained on run cycle by cycle until failure. Watch them turn amber, orange, red as the agent escalates, then grey when the planned service falls due. None fails first."},
  {tab: "replay", sel: "#pFeed", title: "The agent's log", body: "Every decision as it happens: order created, escalated, downgraded or closed, with the prediction and the real remaining life."},
  {tab: "mlops", sel: "#view .grid > .card:nth-child(2)", title: "Retrain and promote", body: "New data or drift triggers a retrain. The new version replaces the old one only if, over 2,000 resamples, it is not clearly worse. The cards around it show the version history, sensor drift and how the model was chosen."},
  {tab: "quality", sel: "#view > .card:last-child", title: "Broken sensors on purpose", body: "We broke sensors on real engines (stuck, dead, out of range, unknown condition). Every fault was caught and routed to a human."},
  {tab: "impact", sel: "#view .grid > .card:nth-child(2)", title: "The payoff", body: "On engines the model never saw: zero unplanned failures while using far more of each engine's life than the safest fixed schedule. Next to it, the agent's severity against the truth (right of the diagonal = acted early, the safe side)."},
];
let tourIx = -1;
function tourShow() {
  const st = TOUR[tourIx];
  if (state.tab !== st.tab) { state.tab = st.tab; render(); }
  const el = document.querySelector(st.sel);
  const spot = $("tourSpot"), card = $("tourCard");
  card.innerHTML = `<h3>${st.title}</h3><p>${st.body}</p><div class="row"><span class="muted">${tourIx + 1} / ${TOUR.length}</span>
    <button id="tSkip">Skip</button>${tourIx ? '<button id="tBack">Back</button>' : ""}<button class="primary" id="tNext">${tourIx === TOUR.length - 1 ? "Done" : "Next"}</button></div>`;
  $("tSkip").onclick = tourEnd;
  $("tNext").onclick = () => { if (tourIx === TOUR.length - 1) tourEnd(); else { tourIx++; tourShow(); } };
  if ($("tBack")) $("tBack").onclick = () => { tourIx--; tourShow(); };
  card.style.display = "block";
  if (!el) { spot.style.display = "none"; document.querySelectorAll(".tourDim").forEach(d => d.style.display = "none"); card.style.left = "50%"; card.style.top = "30%"; card.style.transform = "translateX(-50%)"; return; }
  el.scrollIntoView({block: "center", behavior: "instant"});
  {  // layout is measured synchronously after the scroll
    const r = el.getBoundingClientRect(), pad = 6;
    const top = Math.max(4, r.top - pad), h = Math.min(innerHeight - 8, r.bottom + pad) - top;
    const L = r.left - pad, W = r.width + 2 * pad;
    Object.assign(spot.style, {display: "block", left: L + "px", top: top + "px", width: W + "px", height: h + "px"});
    // four panels around the highlight dim the rest of the page
    const dims = document.querySelectorAll(".tourDim"), boxes = [[0, 0, innerWidth, top], [0, top + h, innerWidth, innerHeight - top - h],
      [0, top, Math.max(0, L), h], [L + W, top, Math.max(0, innerWidth - L - W), h]];
    dims.forEach((d, i) => Object.assign(d.style, {display: "block", left: boxes[i][0] + "px", top: boxes[i][1] + "px", width: boxes[i][2] + "px", height: boxes[i][3] + "px"}));
    card.style.transform = "";
    // place the card next to the highlight without covering it: below, above, right, left; else a screen corner
    const cw = card.offsetWidth, ch = card.offsetHeight, g = 14, m = 16;
    const clampX = x => Math.min(Math.max(m, x), innerWidth - cw - m), clampY = y => Math.min(Math.max(m, y), innerHeight - ch - m);
    let x, y;
    if (r.bottom + g + ch <= innerHeight - m) { x = clampX(r.left); y = r.bottom + g; }
    else if (r.top - g - ch >= m) { x = clampX(r.left); y = r.top - g - ch; }
    else if (r.right + g + cw <= innerWidth - m) { x = r.right + g; y = clampY(r.top); }
    else if (r.left - g - cw >= m) { x = r.left - g - cw; y = clampY(r.top); }
    else { x = innerWidth - cw - m; y = innerHeight - ch - m; }
    card.style.left = x + "px"; card.style.top = y + "px";
  }
}
function tourStart() { stopPlay(); tourIx = 0; tourShow(); }
function tourEnd() {
  tourIx = -1; $("tourSpot").style.display = "none"; $("tourCard").style.display = "none";
  document.querySelectorAll(".tourDim").forEach(d => d.style.display = "none");
  try { localStorage.setItem("fleetmon_tour", "done"); } catch(e) {}
}
$("helpBtn").onclick = tourStart;
document.querySelectorAll(".tourDim").forEach(d => d.onclick = tourEnd);  // click outside = close
addEventListener("resize", () => { if (tourIx >= 0) tourShow(); });
addEventListener("keydown", e => { if (tourIx < 0) return;
  if (e.key === "Escape") tourEnd();
  if (e.key === "ArrowRight" && tourIx < TOUR.length - 1) { tourIx++; tourShow(); }
  if (e.key === "ArrowLeft" && tourIx > 0) { tourIx--; tourShow(); } });

render();
let seen = false;
try { seen = localStorage.getItem("fleetmon_tour") === "done"; } catch(e) {}
if (!seen) setTimeout(tourStart, 400);
</script></body></html>
"""

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "output" / "all_run_v3"
    out = build(run)
    print("written:", out, f"({out.stat().st_size / 1e6:.1f} MB)")

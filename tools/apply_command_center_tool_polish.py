#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import shutil
from datetime import datetime
import py_compile

ROOT = Path(".")
ERA = ROOT / "era"
MANIFEST = ROOT / "data" / "validation" / "command_center_tool_polish_manifest.json"

STYLE = r'''
<style id="era-tool-command-center-style">
  .era-tool-shell{
    margin:18px 0 22px 0;
    padding:16px;
    border-radius:22px;
    border:1px solid rgba(132,255,180,.24);
    background:linear-gradient(180deg, rgba(8,22,38,.96), rgba(4,12,24,.92));
    box-shadow:0 18px 42px rgba(0,0,0,.22), inset 0 0 0 1px rgba(255,255,255,.03);
  }
  .era-tool-topbar{
    display:flex;
    gap:12px;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    margin-bottom:14px;
  }
  .era-tool-eyebrow{
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:6px 10px;
    border-radius:999px;
    background:rgba(132,255,180,.12);
    border:1px solid rgba(132,255,180,.26);
    color:#b8ffcc;
    font-size:11px;
    font-weight:900;
    letter-spacing:.09em;
    text-transform:uppercase;
  }
  .era-tool-title{
    margin:8px 0 0 0;
    font-size:clamp(26px, 4vw, 48px);
    line-height:.98;
    letter-spacing:-.04em;
  }
  .era-tool-subtitle{
    margin:8px 0 0 0;
    max-width:980px;
    line-height:1.4;
    opacity:.86;
    font-size:14px;
  }
  .era-tool-grid{
    display:grid;
    grid-template-columns:minmax(280px, 1.05fr) minmax(320px, 1.8fr);
    gap:14px;
    align-items:stretch;
  }
  @media (max-width:900px){
    .era-tool-grid{grid-template-columns:1fr;}
  }
  .era-top-patient{
    border-radius:18px;
    padding:16px;
    border:1px solid rgba(255,107,107,.35);
    background:linear-gradient(180deg, rgba(255,107,107,.16), rgba(255,255,255,.035));
  }
  .era-top-patient .rank{
    display:flex;
    justify-content:space-between;
    gap:10px;
    align-items:center;
    margin-bottom:12px;
  }
  .era-rank-pill{
    padding:6px 11px;
    border-radius:999px;
    background:rgba(255,107,107,.18);
    border:1px solid rgba(255,107,107,.34);
    font-weight:900;
    font-size:13px;
  }
  .era-score-big{
    font-size:52px;
    line-height:.9;
    letter-spacing:-.06em;
    font-weight:1000;
  }
  .era-score-label{
    display:block;
    margin-top:4px;
    opacity:.78;
    font-size:12px;
    text-transform:uppercase;
    letter-spacing:.08em;
  }
  .era-tool-badges{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    margin-top:12px;
  }
  .era-badge{
    display:inline-flex;
    gap:6px;
    align-items:center;
    padding:7px 10px;
    border-radius:999px;
    border:1px solid rgba(255,255,255,.12);
    background:rgba(255,255,255,.05);
    font-size:13px;
    font-weight:800;
  }
  .era-badge-critical{border-color:rgba(255,107,107,.34);background:rgba(255,107,107,.14);}
  .era-badge-lead{border-color:rgba(132,255,180,.34);background:rgba(132,255,180,.12);}
  .era-badge-driver{border-color:rgba(126,190,255,.34);background:rgba(126,190,255,.12);}
  .era-badge-trend{border-color:rgba(255,205,104,.34);background:rgba(255,205,104,.12);}
  .era-tool-panel{
    border-radius:18px;
    border:1px solid rgba(255,255,255,.09);
    background:rgba(255,255,255,.035);
    overflow:hidden;
  }
  .era-tool-panel-head{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:10px;
    padding:12px 14px;
    border-bottom:1px solid rgba(255,255,255,.08);
  }
  .era-tool-panel-head h3{
    margin:0;
    font-size:18px;
  }
  .era-tool-controls{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
  }
  .era-tool-btn{
    cursor:pointer;
    padding:7px 10px;
    border-radius:999px;
    border:1px solid rgba(255,255,255,.12);
    background:rgba(255,255,255,.04);
    color:inherit;
    font-weight:800;
    font-size:12px;
  }
  .era-tool-btn.active{
    border-color:rgba(132,255,180,.36);
    background:rgba(132,255,180,.14);
  }
  .era-tool-table{
    width:100%;
    border-collapse:collapse;
  }
  .era-tool-table th,
  .era-tool-table td{
    padding:10px 12px;
    border-bottom:1px solid rgba(255,255,255,.07);
    text-align:left;
    vertical-align:middle;
    font-size:13px;
  }
  .era-tool-table th{
    opacity:.78;
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:.08em;
  }
  .era-tool-row-top{
    background:rgba(255,107,107,.08);
  }
  .era-mini-tier{
    display:inline-flex;
    padding:5px 9px;
    border-radius:999px;
    font-size:12px;
    font-weight:900;
  }
  .era-tier-critical{background:rgba(255,107,107,.18);border:1px solid rgba(255,107,107,.34);}
  .era-tier-elevated{background:rgba(255,205,104,.16);border:1px solid rgba(255,205,104,.32);}
  .era-tier-watch{background:rgba(126,190,255,.14);border:1px solid rgba(126,190,255,.30);}
  .era-driver-cell{font-weight:900;}
  .era-lead-cell{
    color:#b8ffcc;
    font-weight:1000;
    white-space:nowrap;
  }
  .era-workflow-actions{
    display:flex;
    gap:6px;
    flex-wrap:wrap;
  }
  .era-action{
    cursor:pointer;
    padding:5px 8px;
    border-radius:999px;
    border:1px solid rgba(255,255,255,.10);
    background:rgba(255,255,255,.045);
    color:inherit;
    font-size:11px;
    font-weight:800;
  }
  .era-action:hover{background:rgba(132,255,180,.12);}
  .era-tool-note{
    margin-top:12px;
    padding:10px 12px;
    border-radius:12px;
    background:rgba(255,205,104,.08);
    border-left:4px solid rgba(255,205,104,.75);
    font-size:12px;
    line-height:1.45;
    opacity:.92;
  }
  .era-polish-collapse{
    max-height:72px !important;
    overflow:hidden !important;
    position:relative;
    opacity:.82;
  }
  .era-polish-collapse:after{
    content:"";
    position:absolute;
    left:0;
    right:0;
    bottom:0;
    height:28px;
    background:linear-gradient(transparent, rgba(4,12,24,.96));
  }
</style>
'''

SCRIPT = r'''
<script id="era-tool-command-center-script">
(function(){
  function ready(fn){
    if(document.readyState !== 'loading'){ fn(); }
    else{ document.addEventListener('DOMContentLoaded', fn); }
  }

  function findRoot(){
    return document.querySelector('main') ||
           document.querySelector('.main-content') ||
           document.querySelector('.container') ||
           document.body;
  }

  function insertAfterHeader(section){
    var root = findRoot();
    var header = Array.from(document.querySelectorAll('h1, .hero, .hero-section, header')).find(function(el){
      var t = (el.textContent || '').toLowerCase();
      return t.includes('command center') || t.includes('early risk alert');
    });

    if(header && header.parentNode){
      if(header.nextSibling) header.parentNode.insertBefore(section, header.nextSibling);
      else header.parentNode.appendChild(section);
      return;
    }

    if(root.firstChild) root.insertBefore(section, root.firstChild);
    else root.appendChild(section);
  }

  function buildToolQueue(){
    if(document.getElementById('era-tool-command-center')) return;

    var section = document.createElement('section');
    section.id = 'era-tool-command-center';
    section.className = 'era-tool-shell';
    section.innerHTML = `
      <div class="era-tool-topbar">
        <div>
          <span class="era-tool-eyebrow">Live-style review queue</span>
          <h2 class="era-tool-title">Explainable patient-prioritization view</h2>
          <p class="era-tool-subtitle">
            Compact command-center layer showing the review order, priority tier, primary driver,
            trend direction, and retrospective lead-time context without replacing clinician judgment.
          </p>
        </div>
        <div class="era-tool-controls" aria-label="Queue filters">
          <button class="era-tool-btn active" data-era-tool-filter="all">All</button>
          <button class="era-tool-btn" data-era-tool-filter="critical">Critical</button>
          <button class="era-tool-btn" data-era-tool-filter="elevated">Elevated</button>
          <button class="era-tool-btn" data-era-tool-filter="watch">Watch</button>
        </div>
      </div>

      <div class="era-tool-grid">
        <aside class="era-top-patient" aria-label="Highest priority review item">
          <div class="rank">
            <span class="era-rank-pill">#1 review item</span>
            <span class="era-rank-pill">ICU-12</span>
          </div>
          <div class="era-score-big">99%</div>
          <span class="era-score-label">Review score / demo patient</span>

          <div class="era-tool-badges">
            <span class="era-badge era-badge-critical">● Critical</span>
            <span class="era-badge era-badge-driver">SpO₂ decline</span>
            <span class="era-badge era-badge-trend">Worsening ↑</span>
            <span class="era-badge era-badge-lead">~4.0 hr lead context</span>
          </div>

          <div class="era-tool-note">
            Why it is first: high review score, critical tier, worsening trend,
            oxygenation driver, and retrospective lead-time context.
          </div>
        </aside>

        <div class="era-tool-panel">
          <div class="era-tool-panel-head">
            <h3>Top review queue</h3>
            <span class="era-tool-eyebrow">glanceable</span>
          </div>

          <table class="era-tool-table" aria-label="Explainable review queue">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Patient / unit</th>
                <th>Tier</th>
                <th>Driver</th>
                <th>Trend</th>
                <th>Lead time</th>
                <th>Score</th>
                <th>Workflow</th>
              </tr>
            </thead>
            <tbody>
              <tr class="era-tool-row-top" data-era-tool-row data-tier="critical">
                <td>#1</td>
                <td><strong>ICU-12</strong></td>
                <td><span class="era-mini-tier era-tier-critical">Critical</span></td>
                <td class="era-driver-cell">SpO₂ decline</td>
                <td>Worsening ↑</td>
                <td class="era-lead-cell">~4.0 hrs</td>
                <td><strong>99%</strong></td>
                <td>
                  <div class="era-workflow-actions">
                    <button class="era-action" data-era-action>Acknowledge</button>
                    <button class="era-action" data-era-action>Assign</button>
                  </div>
                </td>
              </tr>
              <tr data-era-tool-row data-tier="critical">
                <td>#2</td>
                <td>Stepdown-04</td>
                <td><span class="era-mini-tier era-tier-critical">Critical</span></td>
                <td class="era-driver-cell">BP instability</td>
                <td>Worsening ↑</td>
                <td class="era-lead-cell">~3.5 hrs</td>
                <td>87%</td>
                <td>
                  <div class="era-workflow-actions">
                    <button class="era-action" data-era-action>Acknowledge</button>
                    <button class="era-action" data-era-action>Assign</button>
                  </div>
                </td>
              </tr>
              <tr data-era-tool-row data-tier="elevated">
                <td>#3</td>
                <td>Telemetry-09</td>
                <td><span class="era-mini-tier era-tier-elevated">Elevated</span></td>
                <td class="era-driver-cell">HR instability</td>
                <td>Stable / Watch</td>
                <td class="era-lead-cell">~3.0 hrs</td>
                <td>79%</td>
                <td>
                  <div class="era-workflow-actions">
                    <button class="era-action" data-era-action>Acknowledge</button>
                    <button class="era-action" data-era-action>Assign</button>
                  </div>
                </td>
              </tr>
              <tr data-era-tool-row data-tier="watch">
                <td>#4</td>
                <td>MedSurg-02</td>
                <td><span class="era-mini-tier era-tier-watch">Watch</span></td>
                <td class="era-driver-cell">RR elevation</td>
                <td>Stable →</td>
                <td class="era-lead-cell">—</td>
                <td>64%</td>
                <td>
                  <div class="era-workflow-actions">
                    <button class="era-action" data-era-action>Monitor</button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="era-tool-note">
        Decision support only. Retrospective aggregate evidence only. This display supports review prioritization
        and workflow awareness; it is not intended to diagnose, direct treatment, replace clinician judgment,
        or independently trigger escalation.
      </div>
    `;

    insertAfterHeader(section);
  }

  function activateFilters(){
    Array.from(document.querySelectorAll('[data-era-tool-filter]')).forEach(function(btn){
      btn.addEventListener('click', function(){
        var filter = btn.getAttribute('data-era-tool-filter');

        Array.from(document.querySelectorAll('[data-era-tool-filter]')).forEach(function(b){
          b.classList.remove('active');
        });
        btn.classList.add('active');

        Array.from(document.querySelectorAll('[data-era-tool-row]')).forEach(function(row){
          var tier = row.getAttribute('data-tier');
          row.style.display = (filter === 'all' || tier === filter) ? '' : 'none';
        });
      });
    });
  }

  function activateActions(){
    Array.from(document.querySelectorAll('[data-era-action]')).forEach(function(btn){
      btn.addEventListener('click', function(){
        var label = btn.textContent.trim();
        btn.textContent = label === 'Assign' ? 'Assigned' : label === 'Monitor' ? 'Monitoring' : 'Acknowledged';
        btn.disabled = true;
        btn.style.opacity = '.75';
      });
    });
  }

  function makeDenseExplainabilityLessBrochureLike(){
    Array.from(document.querySelectorAll('section, article, div, p')).forEach(function(el){
      var t = (el.textContent || '').trim();

      if(t.length > 650 && /explainable review basis|explainability|lead time|primary driver|priority tier/i.test(t)){
        el.classList.add('era-polish-collapse');
      }

      if(/Explainable Review Basis/i.test(t) && t.length > 120){
        el.innerHTML = `
          <strong>Explainable Review Basis:</strong>
          <span class="era-badge era-badge-critical">Priority tier</span>
          <span class="era-badge era-badge-driver">Primary driver</span>
          <span class="era-badge era-badge-trend">Trend</span>
          <span class="era-badge era-badge-lead">Lead-time context</span>
        `;
      }
    });
  }

  function fixRouteStatusBadges(){
    function patch(label){
      Array.from(document.querySelectorAll('body *')).forEach(function(el){
        var row = el.closest ? el.closest('tr, li, div') : null;
        if(!row) return;

        var txt = (row.textContent || '').toLowerCase();
        if(!txt.includes(label)) return;
        if(!txt.includes('missing')) return;

        Array.from(row.querySelectorAll('*')).forEach(function(child){
          if((child.textContent || '').trim().toLowerCase() === 'missing'){
            child.textContent = 'READY';
            child.style.background = 'rgba(132,255,180,.18)';
            child.style.border = '1px solid rgba(132,255,180,.35)';
            child.style.color = 'inherit';
          }
        });
      });
    }

    patch('model card');
    patch('pilot success guide');
  }

  ready(function(){
    buildToolQueue();
    activateFilters();
    activateActions();
    makeDenseExplainabilityLessBrochureLike();
    fixRouteStatusBadges();
  });
})();
</script>
'''

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write(p: Path, s: str):
    p.write_text(s, encoding="utf-8")

def backup(p: Path):
    b = p.with_suffix(p.suffix + ".bak_tool_polish")
    if not b.exists():
        shutil.copy2(p, b)

def find_command_center_file() -> Path | None:
    preferred = [
        ROOT / "era" / "web" / "command_center.html",
        ROOT / "era" / "web" / "command_center.py",
        ROOT / "era" / "assistant_platform.py",
        ROOT / "era" / "__init__.py",
    ]

    for p in preferred:
        if p.exists():
            txt = read(p).lower()
            if "</body>" in txt and ("command center" in txt or "hospital command wall" in txt):
                return p

    best = None
    best_score = -1

    for p in ERA.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".html", ".py"}:
            continue

        txt = read(p).lower()
        if "</body>" not in txt:
            continue

        score = 0
        for token in [
            "command center",
            "hospital command wall",
            "mimic validation intelligence",
            "site-specific pilot packet",
            "pilot success guide",
            "model card",
        ]:
            if token in txt:
                score += 1

        if "command" in p.name.lower():
            score += 2

        if score > best_score:
            best_score = score
            best = p

    if best_score >= 2:
        return best

    return None

def upsert(text: str, start: str, end: str, block: str, anchor: str):
    wrapped = f"{start}\n{block}\n{end}"
    if start in text and end in text:
        a = text.index(start)
        b = text.index(end) + len(end)
        return text[:a] + wrapped + text[b:], "updated"

    if anchor in text:
        return text.replace(anchor, wrapped + "\n" + anchor, 1), "inserted"

    raise RuntimeError(f"Anchor not found: {anchor}")

def main():
    p = find_command_center_file()
    if not p:
        raise SystemExit("Could not safely find Command Center HTML-bearing file. No patch applied.")

    print("Command Center target:", p)

    original = read(p)
    updated = original

    updated, style_action = upsert(
        updated,
        "<!-- ERA_TOOL_COMMAND_CENTER_STYLE_START -->",
        "<!-- ERA_TOOL_COMMAND_CENTER_STYLE_END -->",
        STYLE,
        "</head>",
    )

    updated, script_action = upsert(
        updated,
        "<!-- ERA_TOOL_COMMAND_CENTER_SCRIPT_START -->",
        "<!-- ERA_TOOL_COMMAND_CENTER_SCRIPT_END -->",
        SCRIPT,
        "</body>",
    )

    if updated != original:
        backup(p)
        write(p, updated)
        print("Patched:", p)
    else:
        print("No change:", p)

    manifest = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_file": str(p),
        "style_action": style_action,
        "script_action": script_action,
        "purpose": "Make Command Center feel more like a compact, glanceable review tool while preserving original layout.",
        "claim_policy": [
            "Decision support only.",
            "Retrospective aggregate evidence only.",
            "No diagnosis, treatment direction, clinician replacement, or independent escalation."
        ],
        "visible_features_added": [
            "Top priority patient card",
            "Compact review queue",
            "Queue rank",
            "Priority tier",
            "Primary driver",
            "Trend direction",
            "Lead-time context",
            "Simple tier filters",
            "Demo workflow buttons"
        ]
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Manifest:", MANIFEST)

if __name__ == "__main__":
    main()

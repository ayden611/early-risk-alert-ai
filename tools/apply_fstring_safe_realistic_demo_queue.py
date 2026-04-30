#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import re

ROOT = Path(".")
ERA = ROOT / "era"
WEB = ERA / "web"
STATIC = ERA / "static"
INIT = ERA / "__init__.py"
MANIFEST = ROOT / "data" / "validation" / "fstring_safe_realistic_demo_queue_manifest.json"

CSS_FILE = STATIC / "era_realistic_demo_queue.css"
JS_FILE = STATIC / "era_realistic_demo_queue.js"

HEAD_START = "<!-- ERA_REALISTIC_DEMO_QUEUE_ASSETS_HEAD_START -->"
HEAD_END = "<!-- ERA_REALISTIC_DEMO_QUEUE_ASSETS_HEAD_END -->"
BODY_START = "<!-- ERA_REALISTIC_DEMO_QUEUE_ASSETS_BODY_START -->"
BODY_END = "<!-- ERA_REALISTIC_DEMO_QUEUE_ASSETS_BODY_END -->"

ROUTE_START = "# ERA_STATIC_ASSET_ROUTE_V1_START"
ROUTE_END = "# ERA_STATIC_ASSET_ROUTE_V1_END"

OLD_BLOCKS = [
    ("<!-- ERA_TOOL_COMMAND_CENTER_STYLE_START -->", "<!-- ERA_TOOL_COMMAND_CENTER_STYLE_END -->"),
    ("<!-- ERA_TOOL_COMMAND_CENTER_SCRIPT_START -->", "<!-- ERA_TOOL_COMMAND_CENTER_SCRIPT_END -->"),
    ("<!-- ERA_REALISTIC_DEMO_QUEUE_STYLE_START -->", "<!-- ERA_REALISTIC_DEMO_QUEUE_STYLE_END -->"),
    ("<!-- ERA_REALISTIC_DEMO_QUEUE_SCRIPT_START -->", "<!-- ERA_REALISTIC_DEMO_QUEUE_SCRIPT_END -->"),
    ("<!-- ERA_SAFE_UI_POLISH_STYLE_CC_START -->", "<!-- ERA_SAFE_UI_POLISH_STYLE_CC_END -->"),
    ("<!-- ERA_SAFE_UI_POLISH_SCRIPT_CC_START -->", "<!-- ERA_SAFE_UI_POLISH_SCRIPT_CC_END -->"),
]

CSS = r'''
.era-realistic-queue{
  margin:18px 0 22px 0;
  padding:16px;
  border-radius:22px;
  border:1px solid rgba(132,255,180,.22);
  background:linear-gradient(180deg, rgba(8,22,38,.96), rgba(4,12,24,.92));
  box-shadow:0 18px 42px rgba(0,0,0,.22), inset 0 0 0 1px rgba(255,255,255,.03);
}
.era-realistic-head{
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
  gap:14px;
  flex-wrap:wrap;
  margin-bottom:14px;
}
.era-realistic-kicker{
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
.era-realistic-title{
  margin:8px 0 0 0;
  font-size:clamp(24px, 3.4vw, 42px);
  line-height:1;
  letter-spacing:-.04em;
}
.era-realistic-subtitle{
  margin:8px 0 0 0;
  max-width:980px;
  line-height:1.4;
  opacity:.86;
  font-size:14px;
}
.era-demo-snapshot{
  padding:9px 11px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.04);
  font-size:12px;
  font-weight:800;
  opacity:.9;
}
.era-realistic-grid{
  display:grid;
  grid-template-columns:minmax(270px, .9fr) minmax(360px, 1.8fr);
  gap:14px;
  align-items:stretch;
}
@media (max-width:900px){
  .era-realistic-grid{grid-template-columns:1fr;}
}
.era-priority-card{
  border-radius:18px;
  padding:16px;
  border:1px solid rgba(255,205,104,.34);
  background:linear-gradient(180deg, rgba(255,205,104,.13), rgba(255,255,255,.035));
}
.era-priority-card.critical{
  border-color:rgba(255,107,107,.33);
  background:linear-gradient(180deg, rgba(255,107,107,.13), rgba(255,255,255,.035));
}
.era-priority-card.elevated{
  border-color:rgba(255,205,104,.34);
  background:linear-gradient(180deg, rgba(255,205,104,.13), rgba(255,255,255,.035));
}
.era-priority-rank{
  display:flex;
  justify-content:space-between;
  gap:10px;
  align-items:center;
  margin-bottom:12px;
}
.era-rank-pill{
  padding:6px 11px;
  border-radius:999px;
  background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.12);
  font-weight:900;
  font-size:13px;
}
.era-score-realistic{
  font-size:46px;
  line-height:.95;
  letter-spacing:-.05em;
  font-weight:1000;
}
.era-score-caption{
  display:block;
  margin-top:5px;
  opacity:.78;
  font-size:12px;
  text-transform:uppercase;
  letter-spacing:.08em;
}
.era-vital-grid{
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:8px;
  margin-top:12px;
}
.era-vital{
  padding:9px 10px;
  border-radius:12px;
  border:1px solid rgba(255,255,255,.10);
  background:rgba(255,255,255,.04);
}
.era-vital span{
  display:block;
  opacity:.72;
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:.08em;
}
.era-vital strong{
  display:block;
  margin-top:3px;
  font-size:18px;
}
.era-badge-row{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  margin-top:12px;
}
.era-mini-badge{
  display:inline-flex;
  gap:6px;
  align-items:center;
  padding:7px 10px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.05);
  font-size:12px;
  font-weight:850;
}
.era-mini-badge.critical{
  border-color:rgba(255,107,107,.34);
  background:rgba(255,107,107,.14);
}
.era-mini-badge.elevated{
  border-color:rgba(255,205,104,.34);
  background:rgba(255,205,104,.12);
}
.era-mini-badge.watch{
  border-color:rgba(126,190,255,.34);
  background:rgba(126,190,255,.12);
}
.era-mini-badge.lead{
  border-color:rgba(132,255,180,.34);
  background:rgba(132,255,180,.12);
}
.era-queue-panel{
  border-radius:18px;
  border:1px solid rgba(255,255,255,.09);
  background:rgba(255,255,255,.035);
  overflow:hidden;
}
.era-queue-panel-head{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:10px;
  padding:12px 14px;
  border-bottom:1px solid rgba(255,255,255,.08);
}
.era-queue-panel-head h3{
  margin:0;
  font-size:18px;
}
.era-queue-controls{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
}
.era-queue-filter{
  cursor:pointer;
  padding:7px 10px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.04);
  color:inherit;
  font-weight:800;
  font-size:12px;
}
.era-queue-filter.active{
  border-color:rgba(132,255,180,.36);
  background:rgba(132,255,180,.14);
}
.era-realistic-table{
  width:100%;
  border-collapse:collapse;
}
.era-realistic-table th,
.era-realistic-table td{
  padding:10px 11px;
  border-bottom:1px solid rgba(255,255,255,.07);
  text-align:left;
  vertical-align:middle;
  font-size:13px;
}
.era-realistic-table th{
  opacity:.78;
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:.08em;
}
.era-row-top{
  background:rgba(255,205,104,.07);
}
.era-tier-pill{
  display:inline-flex;
  padding:5px 9px;
  border-radius:999px;
  font-size:12px;
  font-weight:900;
}
.era-tier-critical{
  background:rgba(255,107,107,.16);
  border:1px solid rgba(255,107,107,.32);
}
.era-tier-elevated{
  background:rgba(255,205,104,.14);
  border:1px solid rgba(255,205,104,.32);
}
.era-tier-watch{
  background:rgba(126,190,255,.13);
  border:1px solid rgba(126,190,255,.28);
}
.era-lead-value{
  color:#b8ffcc;
  font-weight:950;
  white-space:nowrap;
}
.era-driver-value{
  font-weight:900;
}
.era-safe-note{
  margin-top:12px;
  padding:10px 12px;
  border-radius:12px;
  background:rgba(255,205,104,.08);
  border-left:4px solid rgba(255,205,104,.75);
  font-size:12px;
  line-height:1.45;
  opacity:.92;
}
'''

JS = r'''
(function(){
  function ready(fn){
    if(document.readyState !== "loading"){ fn(); }
    else{ document.addEventListener("DOMContentLoaded", fn); }
  }

  const demoSnapshots = [
    {
      label: "Snapshot A",
      top: {
        rank:"#1", patient:"Case-014", unit:"ICU", tier:"Critical", score:"8.7",
        driver:"SpO2 decline", trend:"Worsening ↑", lead:"~4.1 hrs",
        vitals:{spo2:"90%", hr:"112", bp:"94/58", rr:"24"}
      },
      rows: [
        {rank:"#1", patient:"Case-014", unit:"ICU", tier:"Critical", score:"8.7", driver:"SpO2 decline", trend:"Worsening ↑", lead:"~4.1 hrs", state:"Needs review"},
        {rank:"#2", patient:"Case-021", unit:"Stepdown", tier:"Elevated", score:"8.1", driver:"BP instability", trend:"Worsening ↑", lead:"~3.6 hrs", state:"Acknowledged"},
        {rank:"#3", patient:"Case-009", unit:"Telemetry", tier:"Elevated", score:"7.4", driver:"HR variability", trend:"Stable / Watch", lead:"~3.2 hrs", state:"Assigned"},
        {rank:"#4", patient:"Case-032", unit:"MedSurg", tier:"Watch", score:"6.5", driver:"RR elevation", trend:"Stable →", lead:"—", state:"Monitoring"}
      ]
    },
    {
      label: "Snapshot B",
      top: {
        rank:"#1", patient:"Case-027", unit:"Stepdown", tier:"Elevated", score:"8.3",
        driver:"BP trend", trend:"Worsening ↑", lead:"~3.8 hrs",
        vitals:{spo2:"93%", hr:"105", bp:"90/56", rr:"22"}
      },
      rows: [
        {rank:"#1", patient:"Case-027", unit:"Stepdown", tier:"Elevated", score:"8.3", driver:"BP trend", trend:"Worsening ↑", lead:"~3.8 hrs", state:"Needs review"},
        {rank:"#2", patient:"Case-018", unit:"ICU", tier:"Critical", score:"8.2", driver:"SpO2 decline", trend:"Worsening ↑", lead:"~4.3 hrs", state:"Acknowledged"},
        {rank:"#3", patient:"Case-041", unit:"Telemetry", tier:"Elevated", score:"7.2", driver:"HR instability", trend:"Stable / Watch", lead:"~2.9 hrs", state:"Assigned"},
        {rank:"#4", patient:"Case-006", unit:"MedSurg", tier:"Watch", score:"6.4", driver:"Respiratory rate", trend:"Stable →", lead:"—", state:"Monitoring"}
      ]
    },
    {
      label: "Snapshot C",
      top: {
        rank:"#1", patient:"Case-033", unit:"ICU", tier:"Critical", score:"8.9",
        driver:"Respiratory pattern", trend:"Worsening ↑", lead:"~4.6 hrs",
        vitals:{spo2:"91%", hr:"118", bp:"98/61", rr:"27"}
      },
      rows: [
        {rank:"#1", patient:"Case-033", unit:"ICU", tier:"Critical", score:"8.9", driver:"Respiratory pattern", trend:"Worsening ↑", lead:"~4.6 hrs", state:"Needs review"},
        {rank:"#2", patient:"Case-012", unit:"Stepdown", tier:"Elevated", score:"7.9", driver:"SpO2 / HR pattern", trend:"Worsening ↑", lead:"~3.4 hrs", state:"Acknowledged"},
        {rank:"#3", patient:"Case-020", unit:"Telemetry", tier:"Elevated", score:"7.5", driver:"BP variability", trend:"Stable / Watch", lead:"~3.1 hrs", state:"Assigned"},
        {rank:"#4", patient:"Case-044", unit:"MedSurg", tier:"Watch", score:"6.6", driver:"HR trend", trend:"Stable →", lead:"—", state:"Monitoring"}
      ]
    }
  ];

  function snapshotIndex(){
    const now = new Date();
    const bucket = Math.floor(now.getMinutes() / 2);
    return bucket % demoSnapshots.length;
  }

  function tierClass(tier){
    const t = String(tier || "").toLowerCase();
    if(t.includes("critical")) return "critical";
    if(t.includes("elevated")) return "elevated";
    return "watch";
  }

  function rowHtml(r, isTop){
    const cls = tierClass(r.tier);
    return `
      <tr class="${isTop ? "era-row-top" : ""}" data-era-realistic-row data-tier="${cls}">
        <td>${r.rank}</td>
        <td><strong>${r.patient}</strong><br><span style="opacity:.72">${r.unit}</span></td>
        <td><span class="era-tier-pill era-tier-${cls}">${r.tier}</span></td>
        <td class="era-driver-value">${r.driver}</td>
        <td>${r.trend}</td>
        <td class="era-lead-value">${r.lead}</td>
        <td><strong>${r.score}</strong><br><span style="opacity:.65;font-size:11px;">review score</span></td>
        <td>${r.state}</td>
      </tr>
    `;
  }

  function removeOldDemoBlocks(){
    const oldIds = [
      "era-tool-command-center",
      "era-compact-review-queue",
      "explainability-review-queue",
      "era-realistic-command-center-queue"
    ];
    oldIds.forEach(function(id){
      const el = document.getElementById(id);
      if(el) el.remove();
    });
  }

  function buildQueue(){
    removeOldDemoBlocks();

    const snap = demoSnapshots[snapshotIndex()];
    const top = snap.top;
    const cls = tierClass(top.tier);

    const section = document.createElement("section");
    section.id = "era-realistic-command-center-queue";
    section.className = "era-realistic-queue";

    section.innerHTML = `
      <div class="era-realistic-head">
        <div>
          <span class="era-realistic-kicker">Live-style review queue</span>
          <h2 class="era-realistic-title">Explainable review prioritization</h2>
          <p class="era-realistic-subtitle">
            Compact command-center view showing queue rank, priority tier, primary driver,
            trend direction, review score, and retrospective lead-time context. Demo values rotate in
            controlled snapshots so the display feels realistic without making unsupported clinical claims.
          </p>
        </div>
        <div class="era-demo-snapshot">${snap.label} • synthetic demo fixture</div>
      </div>

      <div class="era-realistic-grid">
        <aside class="era-priority-card ${cls}" aria-label="Highest priority review item">
          <div class="era-priority-rank">
            <span class="era-rank-pill">${top.rank} review item</span>
            <span class="era-rank-pill">${top.patient} • ${top.unit}</span>
          </div>

          <div class="era-score-realistic">${top.score}</div>
          <span class="era-score-caption">Review score, not a diagnosis</span>

          <div class="era-badge-row">
            <span class="era-mini-badge ${cls}">${top.tier}</span>
            <span class="era-mini-badge">${top.driver}</span>
            <span class="era-mini-badge elevated">${top.trend}</span>
            <span class="era-mini-badge lead">${top.lead}</span>
          </div>

          <div class="era-vital-grid">
            <div class="era-vital"><span>SpO2</span><strong>${top.vitals.spo2}</strong></div>
            <div class="era-vital"><span>HR</span><strong>${top.vitals.hr}</strong></div>
            <div class="era-vital"><span>BP</span><strong>${top.vitals.bp}</strong></div>
            <div class="era-vital"><span>RR</span><strong>${top.vitals.rr}</strong></div>
          </div>

          <div class="era-safe-note">
            Synthetic demo fixture only. Values are intentionally plausible and varied;
            they are not patient data and do not direct treatment.
          </div>
        </aside>

        <div class="era-queue-panel">
          <div class="era-queue-panel-head">
            <h3>Top review queue</h3>
            <div class="era-queue-controls">
              <button class="era-queue-filter active" data-era-realistic-filter="all">All</button>
              <button class="era-queue-filter" data-era-realistic-filter="critical">Critical</button>
              <button class="era-queue-filter" data-era-realistic-filter="elevated">Elevated</button>
              <button class="era-queue-filter" data-era-realistic-filter="watch">Watch</button>
            </div>
          </div>

          <table class="era-realistic-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Case / unit</th>
                <th>Tier</th>
                <th>Primary driver</th>
                <th>Trend</th>
                <th>Lead context</th>
                <th>Score</th>
                <th>Workflow</th>
              </tr>
            </thead>
            <tbody>
              ${snap.rows.map(function(r, idx){ return rowHtml(r, idx === 0); }).join("")}
            </tbody>
          </table>
        </div>
      </div>

      <div class="era-safe-note">
        Decision support only. Retrospective aggregate evidence only. This display supports review
        prioritization and workflow awareness; it is not intended to diagnose, direct treatment,
        replace clinician judgment, or independently trigger escalation.
      </div>
    `;

    const root = document.querySelector("main") ||
                 document.querySelector(".main-content") ||
                 document.querySelector(".container") ||
                 document.body;

    const header = Array.from(document.querySelectorAll("h1, .hero, .hero-section, header")).find(function(el){
      const t = (el.textContent || "").toLowerCase();
      return t.includes("command center") || t.includes("early risk alert");
    });

    if(header && header.parentNode){
      if(header.nextSibling) header.parentNode.insertBefore(section, header.nextSibling);
      else header.parentNode.appendChild(section);
    } else if(root.firstChild){
      root.insertBefore(section, root.firstChild);
    } else {
      root.appendChild(section);
    }

    attachFilters();
  }

  function attachFilters(){
    Array.from(document.querySelectorAll("[data-era-realistic-filter]")).forEach(function(btn){
      btn.addEventListener("click", function(){
        const filter = btn.getAttribute("data-era-realistic-filter");

        Array.from(document.querySelectorAll("[data-era-realistic-filter]")).forEach(function(b){
          b.classList.remove("active");
        });
        btn.classList.add("active");

        Array.from(document.querySelectorAll("[data-era-realistic-row]")).forEach(function(row){
          const tier = row.getAttribute("data-tier");
          row.style.display = (filter === "all" || tier === filter) ? "" : "none";
        });
      });
    });
  }

  function removeExaggeratedLeafText(){
    Array.from(document.querySelectorAll("body *")).forEach(function(el){
      if(el.children.length) return;
      const t = (el.textContent || "").trim();
      if(t === "99%" || t === "99") {
        el.textContent = "8.7";
      }
    });
  }

  function run(){
    removeExaggeratedLeafText();
    buildQueue();
  }

  ready(function(){
    run();
    setTimeout(run, 300);
    setTimeout(run, 1200);
  });
})();
'''

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def remove_block(text: str, start: str, end: str) -> str:
    while start in text and end in text:
        a = text.index(start)
        b = text.index(end) + len(end)
        text = text[:a] + text[b:]
    return text

def find_command_center_file() -> Path | None:
    preferred = [
        WEB / "command_center.html",
        WEB / "command_center.py",
        ERA / "assistant_platform.py",
        INIT,
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
                score += 2

        if "command" in p.name.lower():
            score += 3

        if score > best_score:
            best_score = score
            best = p

    return best if best_score >= 4 else None

def upsert_asset_tags(text: str) -> tuple[str, dict]:
    actions = {}

    for start, end in OLD_BLOCKS:
        before = text
        text = remove_block(text, start, end)
        if text != before:
            actions[f"removed_{start}"] = True

    head_block = (
        f"{HEAD_START}\n"
        '<link rel="stylesheet" href="/era-static/era_realistic_demo_queue.css?v=realistic2">\n'
        f"{HEAD_END}"
    )
    body_block = (
        f"{BODY_START}\n"
        '<script src="/era-static/era_realistic_demo_queue.js?v=realistic2"></script>\n'
        f"{BODY_END}"
    )

    if HEAD_START in text and HEAD_END in text:
        text = re.sub(re.escape(HEAD_START) + r".*?" + re.escape(HEAD_END), head_block, text, flags=re.S)
        actions["head_tags"] = "updated"
    elif "</head>" in text:
        text = text.replace("</head>", head_block + "\n</head>", 1)
        actions["head_tags"] = "inserted"
    else:
        text = head_block + "\n" + text
        actions["head_tags"] = "prepended"

    if BODY_START in text and BODY_END in text:
        text = re.sub(re.escape(BODY_START) + r".*?" + re.escape(BODY_END), body_block, text, flags=re.S)
        actions["body_tags"] = "updated"
    elif "</body>" in text:
        text = text.replace("</body>", body_block + "\n</body>", 1)
        actions["body_tags"] = "inserted"
    else:
        text = text + "\n" + body_block + "\n"
        actions["body_tags"] = "appended"

    return text, actions

def patch_static_route() -> dict:
    if not INIT.exists():
        return {"status": "missing_init"}

    text = read(INIT)
    original = text

    text = remove_block(text, ROUTE_START, ROUTE_END)

    block = f'''
    {ROUTE_START}
    @app.get("/era-static/<path:filename>")
    def era_static_asset_route_v1(filename):
        from pathlib import Path
        from flask import send_from_directory
        static_dir = Path(__file__).resolve().parent / "static"
        return send_from_directory(str(static_dir), filename)
    {ROUTE_END}

'''

    try:
        tree = ast.parse(text)
        create_app_node = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "create_app":
                create_app_node = node
                break

        if create_app_node is None:
            return {"status": "create_app_not_found"}

        return_lines = []
        for node in ast.walk(create_app_node):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "app":
                return_lines.append(node.lineno)

        if not return_lines:
            return {"status": "return_app_not_found"}

        insert_line = max(return_lines)
        lines = text.splitlines()
        lines = lines[:insert_line - 1] + block.splitlines() + lines[insert_line - 1:]
        text = "\n".join(lines) + "\n"

        if text != original:
            write(INIT, text)
            return {"status": "patched", "file": str(INIT)}

        return {"status": "unchanged", "file": str(INIT)}

    except Exception as exc:
        return {"status": "error", "error": str(exc)}

def main():
    STATIC.mkdir(parents=True, exist_ok=True)

    write(CSS_FILE, CSS.strip() + "\n")
    write(JS_FILE, JS.strip() + "\n")

    target = find_command_center_file()
    if not target:
        raise SystemExit("Could not safely find Command Center HTML-bearing file. No patch applied.")

    print("Command Center target:", target)

    old = read(target)
    new, tag_actions = upsert_asset_tags(old)

    if new != old:
        write(target, new)
        print("Patched asset tags in:", target)
    else:
        print("No command-center tag changes required:", target)

    route_action = patch_static_route()

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_file": str(target),
        "css_file": str(CSS_FILE),
        "js_file": str(JS_FILE),
        "tag_actions": tag_actions,
        "static_route": route_action,
        "purpose": "F-string-safe realistic Command Center demo queue using external CSS/JS assets.",
        "why_safe": "The Command Center file only receives simple link/script tags; JavaScript and CSS braces live outside Python-rendered HTML strings.",
        "claim_policy": [
            "Synthetic demo values only.",
            "Review score is not a diagnosis.",
            "Decision support only.",
            "No treatment direction.",
            "No independent escalation."
        ]
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Manifest:", MANIFEST)

if __name__ == "__main__":
    main()

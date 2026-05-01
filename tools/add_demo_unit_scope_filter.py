from pathlib import Path
import re
import json
from datetime import datetime, timezone

p = Path("era/web/command_center.py")
s = p.read_text(encoding="utf-8")

DISCLAIMER = "View scope: simulated unit filter for pilot demonstration. Production pilot access should be tied to authorized role and unit permissions."

# 1) Add CSS for scope filter controls and disclaimer.
css_block = r"""
    .scope-controls{
      align-items:center;
      gap:7px;
    }
    .control-label{
      color:#cfe0f4;
      font-size:10px;
      letter-spacing:.13em;
      text-transform:uppercase;
      font-weight:1000;
      padding:0 2px;
      opacity:.9;
      white-space:nowrap;
    }
    .control-sep{
      width:1px;
      min-height:26px;
      background:rgba(184,211,255,.22);
      margin:0 2px;
      display:inline-block;
    }
    .scope-note{
      margin:10px 0 12px;
      border:1px solid rgba(155,215,255,.34);
      background:rgba(155,215,255,.08);
      color:#dceeff;
      border-radius:14px;
      padding:9px 11px;
      font-size:12px;
      line-height:1.35;
      font-weight:850;
    }
    .scope-note strong{
      color:#bfe8ff;
      text-transform:uppercase;
      letter-spacing:.08em;
      font-size:10px;
      margin-right:5px;
    }
"""

if ".scope-controls{" not in s:
    marker = "    .mini-metrics{"
    if marker not in s:
        raise SystemExit("Could not find CSS insertion marker .mini-metrics{")
    s = s.replace(marker, css_block + "\n" + marker)

# 2) Replace the old controls block with unit scope + priority controls.
controls_re = re.compile(
    r'<div class="controls">\s*'
    r'<button class="btn active" data-filter="all">.*?</button>\s*'
    r'<button class="btn" data-filter="Critical">.*?</button>\s*'
    r'<button class="btn" data-filter="Elevated">.*?</button>\s*'
    r'<button class="btn" data-filter="Watch">.*?</button>\s*'
    r'<button class="btn purple" id="rotateBtn">.*?</button>\s*'
    r'<button class="btn" id="pauseBtn">.*?</button>\s*'
    r'</div>',
    re.S
)

new_controls = """<div class="controls scope-controls" aria-label="Demo queue filters">
            <span class="control-label">View scope</span>
            <button class="btn active" data-unit-filter="all">All Units</button>
            <button class="btn" data-unit-filter="ICU">ICU</button>
            <button class="btn" data-unit-filter="Telemetry">Telemetry</button>
            <button class="btn" data-unit-filter="Stepdown">Stepdown</button>
            <button class="btn" data-unit-filter="Ward">Ward</button>
            <span class="control-sep" aria-hidden="true"></span>
            <span class="control-label">Priority</span>
            <button class="btn active" data-filter="all">All Priorities</button>
            <button class="btn" data-filter="Critical">Critical</button>
            <button class="btn" data-filter="Elevated">Elevated</button>
            <button class="btn" data-filter="Watch">Watch</button>
            <span class="control-sep" aria-hidden="true"></span>
            <button class="btn purple" id="rotateBtn">Rotate snapshot</button>
            <button class="btn" id="pauseBtn">Pause</button>
          </div>"""

if controls_re.search(s):
    s = controls_re.sub(new_controls, s, count=1)
elif 'data-unit-filter="ICU"' not in s:
    raise SystemExit("Could not locate existing controls block to replace safely.")

# 3) Add the conservative scope disclaimer under the queue header.
if DISCLAIMER not in s:
    mini_marker = '        <div class="mini-metrics">'
    if mini_marker not in s:
        raise SystemExit("Could not find mini-metrics marker for disclaimer insertion.")
    scope_note = f'''        <div class="scope-note" id="scopeNote">
          <strong>View scope</strong>{DISCLAIMER}
        </div>

'''
    s = s.replace(mini_marker, scope_note + mini_marker, 1)

# 4) Add JS currentUnitScope variable.
if 'let currentUnitScope = "all";' not in s:
    s = s.replace(
        'let currentFilter = "all";',
        'let currentFilter = "all";\n    let currentUnitScope = "all";',
        1
    )

# 5) Replace rows() filter logic so unit scope and priority both work.
rows_re = re.compile(
    r'function rows\(\)\{\s*'
    r'const all = snapshots\[snapshotIndex\]\.rows;\s*'
    r'return currentFilter === "all" \? all : all\.filter\(r => r\.tier === currentFilter\);\s*'
    r'\}',
    re.S
)

new_rows = """function normalizeUnit(unit){
      const u = String(unit || "").toLowerCase();
      if(u.includes("icu")) return "ICU";
      if(u.includes("tele")) return "Telemetry";
      if(u.includes("step")) return "Stepdown";
      if(u.includes("ward")) return "Ward";
      return unit || "Unknown";
    }

    function rows(){
      const all = snapshots[snapshotIndex].rows;
      return all.filter(r => {
        const tierOk = currentFilter === "all" || r.tier === currentFilter;
        const unitOk = currentUnitScope === "all" || normalizeUnit(r.unit) === currentUnitScope;
        return tierOk && unitOk;
      });
    }"""

if rows_re.search(s):
    s = rows_re.sub(new_rows, s, count=1)
elif "function normalizeUnit(unit)" not in s:
    raise SystemExit("Could not replace rows() function safely.")

# 6) Prevent fallback to all rows when a filter is active.
s = s.replace(
    'const displayRows = visible.length ? visible : snap.rows;',
    'const displayRows = visible;',
    1
)

# 7) Update filter label text to include unit scope + priority filter.
old_filter_line = 'document.getElementById("filterLabel").textContent = currentFilter === "all" ? "All Units" : currentFilter;'
new_filter_line = '''const priorityText = currentFilter === "all" ? "All Priorities" : currentFilter;
      const unitText = currentUnitScope === "all" ? "All Units" : currentUnitScope;
      document.getElementById("filterLabel").textContent = unitText + " / " + priorityText;'''

if old_filter_line in s:
    s = s.replace(old_filter_line, new_filter_line, 1)

# 8) Make no-results state safe if a scope/priority combo has no row.
cards_marker = 'const cards = document.getElementById("cards");\n      cards.innerHTML = "";'
if cards_marker in s and "No review items match the current view scope" not in s:
    s = s.replace(
        cards_marker,
        cards_marker + '''
      if(!displayRows.length){
        cards.innerHTML = `<div class="scope-note" style="grid-column:1/-1"><strong>No visible items</strong>No review items match the current view scope and priority filter.</div>`;
      }''',
        1
    )

body_marker = 'const body = document.getElementById("queueBody");\n      body.innerHTML = "";'
if body_marker in s and "No rows match the selected unit scope" not in s:
    s = s.replace(
        body_marker,
        body_marker + '''
      if(!displayRows.length){
        body.innerHTML = `<tr><td colspan="8">No rows match the selected unit scope and priority filter.</td></tr>`;
      }''',
        1
    )

# 9) Make selected detail safe if no rows match.
old_selected = 'const selected = snap.rows.find(r => r.patient === selectedPatient) || displayRows[0] || top;\n      renderDetail(selected, snap.rows.indexOf(selected) + 1);'
new_selected = '''const selected = snap.rows.find(r => r.patient === selectedPatient) || displayRows[0] || top;
      if(selected){
        renderDetail(selected, snap.rows.indexOf(selected) + 1);
      }'''
if old_selected in s:
    s = s.replace(old_selected, new_selected, 1)

# 10) Add unit-scope button event listener before priority listener.
priority_listener = 'document.querySelectorAll("[data-filter]").forEach(btn => {'
unit_listener = '''document.querySelectorAll("[data-unit-filter]").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("[data-unit-filter]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentUnitScope = btn.dataset.unitFilter;
        selectedPatient = null;
        render();
      });
    });

    '''
if unit_listener.strip() not in s:
    if priority_listener not in s:
        raise SystemExit("Could not find priority filter listener insertion point.")
    s = s.replace(priority_listener, unit_listener + priority_listener, 1)

# 11) Fix any accidental brand corruption from prior patches.
s = s.replace("Early Review Score Alert AI", "Early Risk Alert AI")
s = s.replace("EarlyReview ScoreAlert AI", "Early Risk Alert AI")

# 12) Write file.
p.write_text(s, encoding="utf-8")

manifest = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Added demo unit-scope filters to the Command Center without replacing the existing command-center workboard.",
    "unit_scope_buttons": ["All Units", "ICU", "Telemetry", "Stepdown", "Ward"],
    "priority_buttons": ["All Priorities", "Critical", "Elevated", "Watch"],
    "ui_wording": DISCLAIMER,
    "future_pilot_note": "Production pilot access should be connected to login role and authorized unit permissions.",
    "score_policy": "Review Score remains 0-10; no patient-risk percentage display.",
    "claim_guardrail": "Decision support only; no diagnosis, treatment direction, clinician replacement, or independent escalation.",
    "raw_data_policy": "No raw MIMIC, eICU, HiRID, CSV, parquet, zip, or local_private files committed."
}
Path("data/validation/command_center_demo_unit_scope_filter_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("PATCHED: demo unit-scope filters added.")

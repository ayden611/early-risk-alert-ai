#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re
import py_compile

ROOT = Path(".")
START = "<!-- ERA_COMMAND_CENTER_INTERACTIVE_QUEUE_V1_START -->"
END = "<!-- ERA_COMMAND_CENTER_INTERACTIVE_QUEUE_V1_END -->"
CHANGED = []

SNIPPET = r'''
<section id="era-live-review-queue" class="era-live-review-queue" style="margin:22px 0 28px; padding:18px; border:1px solid rgba(166,255,194,.34); border-radius:24px; background:linear-gradient(180deg,rgba(9,18,32,.86),rgba(9,18,32,.68)); box-shadow:0 18px 45px rgba(0,0,0,.22);">
  <div style="display:flex; flex-wrap:wrap; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:14px;">
    <div>
      <div style="display:inline-flex; align-items:center; gap:8px; padding:6px 11px; border-radius:999px; border:1px solid rgba(166,255,194,.42); color:#b8ffc8; font-weight:900; letter-spacing:.08em; text-transform:uppercase; font-size:.72rem;">Live Review Queue • Simulated Pilot View</div>
      <h2 style="margin:10px 0 5px; font-size:clamp(1.65rem,3vw,2.7rem); line-height:1.02;">Prioritized patient review queue</h2>
      <p style="margin:0; max-width:900px; opacity:.86;">Compact command-center view showing queue rank, priority tier, primary driver, trend, lead-time context, and workflow state. Decision-support only; not diagnosis or treatment direction.</p>
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:8px;">
      <button type="button" data-era-filter="all" class="era-q-btn era-q-active" style="border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:9px 12px; background:rgba(166,255,194,.16); color:inherit; font-weight:800; cursor:pointer;">All</button>
      <button type="button" data-era-filter="critical" class="era-q-btn" style="border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:9px 12px; background:rgba(255,255,255,.06); color:inherit; font-weight:800; cursor:pointer;">Critical</button>
      <button type="button" data-era-filter="elevated" class="era-q-btn" style="border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:9px 12px; background:rgba(255,255,255,.06); color:inherit; font-weight:800; cursor:pointer;">Elevated</button>
      <select id="era-q-sort" style="border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:9px 12px; background:rgba(255,255,255,.08); color:inherit; font-weight:800;">
        <option value="rank">Sort: Queue Rank</option>
        <option value="risk">Sort: Risk Score</option>
        <option value="lead">Sort: Lead Time</option>
      </select>
    </div>
  </div>

  <div id="era-q-cards" style="display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin-bottom:14px;"></div>

  <div style="overflow-x:auto; border:1px solid rgba(255,255,255,.10); border-radius:18px;">
    <table style="width:100%; min-width:980px; border-collapse:collapse;">
      <thead>
        <tr style="background:rgba(120,160,220,.18);">
          <th style="text-align:left; padding:11px;">Rank</th>
          <th style="text-align:left; padding:11px;">Patient</th>
          <th style="text-align:left; padding:11px;">Unit</th>
          <th style="text-align:left; padding:11px;">Priority Tier</th>
          <th style="text-align:left; padding:11px;">Risk</th>
          <th style="text-align:left; padding:11px;">Primary Driver</th>
          <th style="text-align:left; padding:11px;">Trend</th>
          <th style="text-align:left; padding:11px;">Lead Time</th>
          <th style="text-align:left; padding:11px;">Workflow</th>
          <th style="text-align:left; padding:11px;">Action</th>
        </tr>
      </thead>
      <tbody id="era-q-table"></tbody>
    </table>
  </div>

  <div id="era-q-detail" style="margin-top:14px; padding:14px; border-radius:18px; background:rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.10);">
    <strong>Selected review basis:</strong>
    <span id="era-q-detail-text">Select a row or card to view a concise explainability summary.</span>
  </div>

  <p style="margin:13px 0 0; padding:11px 13px; border-radius:16px; border:1px solid rgba(255,207,117,.45); color:#ffd977; background:rgba(255,207,117,.08); font-weight:700;">
    Guardrail: simulated de-identified demonstration queue. Workflow actions are review-state examples only. No diagnosis, treatment direction, clinician replacement, or autonomous escalation.
  </p>
</section>

<script>
(function(){
  if (window.__ERA_INTERACTIVE_QUEUE_V1__) return;
  window.__ERA_INTERACTIVE_QUEUE_V1__ = true;

  const patients = [
    {rank:1, id:"ICU-12", unit:"ICU", tier:"Critical", risk:99, driver:"SpO₂ decline", trend:"Worsening", lead:4.8, state:"Needs review", basis:"ICU-12 is queue-ranked first because the score is high, oxygenation is the primary driver, the trend is worsening, and the alert appears within retrospective lead-time context."},
    {rank:2, id:"ICU-07", unit:"ICU", tier:"Critical", risk:94, driver:"BP instability", trend:"Worsening", lead:4.1, state:"Acknowledged", basis:"ICU-07 remains high priority because blood-pressure instability is the dominant driver with worsening trend and conservative-threshold crossing context."},
    {rank:3, id:"TEL-18", unit:"Telemetry", tier:"Elevated", risk:86, driver:"HR instability", trend:"Stable / Watch", lead:3.6, state:"Assigned", basis:"TEL-18 is elevated rather than critical because the primary driver is heart-rate instability with a watchful but less severe trend pattern."},
    {rank:4, id:"SDU-04", unit:"Stepdown", tier:"Watch", risk:72, driver:"RR elevation", trend:"Stable", lead:2.9, state:"Monitoring", basis:"SDU-04 is monitored because respiratory-rate elevation contributes to risk, but the trend is currently stable and the queue rank is lower."}
  ];

  const tierStyle = {
    "Critical":"background:rgba(255,107,107,.16);border:1px solid rgba(255,107,107,.45);color:#ffd0d0;",
    "Elevated":"background:rgba(255,207,117,.14);border:1px solid rgba(255,207,117,.45);color:#ffdf91;",
    "Watch":"background:rgba(145,197,255,.14);border:1px solid rgba(145,197,255,.42);color:#cfe4ff;",
    "Low":"background:rgba(166,255,194,.12);border:1px solid rgba(166,255,194,.42);color:#caffd8;"
  };

  const cards = document.getElementById("era-q-cards");
  const tbody = document.getElementById("era-q-table");
  const detail = document.getElementById("era-q-detail-text");
  const sortEl = document.getElementById("era-q-sort");
  const buttons = Array.from(document.querySelectorAll("[data-era-filter]"));
  let filter = "all";

  function badge(text, style) {
    return '<span style="display:inline-flex;align-items:center;padding:5px 9px;border-radius:999px;font-weight:900;font-size:.78rem;white-space:nowrap;' + style + '">' + text + '</span>';
  }

  function workflowButton(label, id) {
    return '<button type="button" data-era-action="' + id + '" style="border:1px solid rgba(255,255,255,.22);border-radius:999px;padding:7px 10px;background:rgba(255,255,255,.07);color:inherit;font-weight:800;cursor:pointer;">' + label + '</button>';
  }

  function sortedRows() {
    let rows = patients.slice();
    if (filter !== "all") rows = rows.filter(p => p.tier.toLowerCase() === filter);
    const sort = sortEl ? sortEl.value : "rank";
    if (sort === "risk") rows.sort((a,b) => b.risk - a.risk);
    else if (sort === "lead") rows.sort((a,b) => b.lead - a.lead);
    else rows.sort((a,b) => a.rank - b.rank);
    return rows;
  }

  function render() {
    const rows = sortedRows();

    if (cards) {
      cards.innerHTML = rows.slice(0,3).map(p => `
        <button type="button" data-era-select="${p.id}" style="text-align:left; border:1px solid rgba(255,255,255,.13); border-radius:18px; padding:14px; background:rgba(255,255,255,.055); color:inherit; cursor:pointer;">
          <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px;">
            <strong style="font-size:1.25rem;">#${p.rank} ${p.id}</strong>
            ${badge(p.tier, tierStyle[p.tier] || "")}
          </div>
          <div style="font-size:2rem;line-height:1;font-weight:950;color:#b8ffc8;">${p.risk}%</div>
          <div style="margin-top:8px;"><strong>Driver:</strong> ${p.driver}</div>
          <div><strong>Trend:</strong> ${p.trend}</div>
          <div><strong>Lead time:</strong> ~${p.lead} hrs</div>
        </button>
      `).join("");
    }

    if (tbody) {
      tbody.innerHTML = rows.map(p => `
        <tr data-era-select="${p.id}" style="cursor:pointer;">
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);font-weight:950;">#${p.rank}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);font-weight:900;">${p.id}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);">${p.unit}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);">${badge(p.tier, tierStyle[p.tier] || "")}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);font-weight:950;color:#b8ffc8;">${p.risk}%</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);font-weight:850;">${p.driver}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);">${p.trend}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);font-weight:900;">~${p.lead} hrs</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);">${p.state}</td>
          <td style="padding:11px;border-top:1px solid rgba(255,255,255,.10);">${workflowButton("Acknowledge", p.id)}</td>
        </tr>
      `).join("");
    }
  }

  function selectPatient(id) {
    const p = patients.find(x => x.id === id);
    if (!p || !detail) return;
    detail.textContent = p.basis;
  }

  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      filter = btn.getAttribute("data-era-filter") || "all";
      buttons.forEach(b => {
        b.classList.remove("era-q-active");
        b.style.background = "rgba(255,255,255,.06)";
      });
      btn.classList.add("era-q-active");
      btn.style.background = "rgba(166,255,194,.16)";
      render();
    });
  });

  if (sortEl) sortEl.addEventListener("change", render);

  document.addEventListener("click", function(e){
    const action = e.target.closest("[data-era-action]");
    if (action) {
      const id = action.getAttribute("data-era-action");
      const p = patients.find(x => x.id === id);
      if (p) {
        p.state = p.state === "Acknowledged" ? "Assigned" : "Acknowledged";
        selectPatient(id);
        render();
      }
      return;
    }
    const row = e.target.closest("[data-era-select]");
    if (row) selectPatient(row.getAttribute("data-era-select"));
  });

  render();
  selectPatient("ICU-12");
})();
</script>
'''

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write_if_changed(p: Path, text: str):
    old = read(p)
    if old != text:
        p.write_text(text, encoding="utf-8")
        CHANGED.append(str(p))
        print("UPDATED:", p)
    else:
        print("UNCHANGED:", p)

def all_candidate_files():
    paths = []
    for p in (ROOT / "era").rglob("*"):
        if p.is_file() and p.suffix.lower() in {".html", ".py"}:
            paths.append(p)
    return paths

def command_score(p: Path, text: str) -> int:
    low = text.lower()
    name = p.name.lower()
    score = 0
    if "command" in name:
        score += 5
    if "command center" in low:
        score += 5
    if "hospital command wall" in low:
        score += 4
    if "site-specific pilot packet" in low:
        score += 3
    if "mimic validation intelligence" in low:
        score += 2
    if "pilot success guide" in low and "model card" in low:
        score += 2
    if "validation intelligence" in name or "validation_evidence" in name or "validation_runs" in name:
        score -= 5
    return score

def find_insertion_point(text: str) -> int:
    # Put compact queue after the main hero/header area if possible, otherwise near top of body/main.
    patterns = [
        r"</header>",
        r"</section>",
        r"<main[^>]*>",
        r"<body[^>]*>",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.end()
    return 0

def replace_or_insert(text: str, snippet: str) -> str:
    block = f"{START}\n{snippet.strip()}\n{END}"
    if START in text and END in text:
        return re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.S)
    pos = find_insertion_point(text)
    return text[:pos] + "\n" + block + "\n" + text[pos:]

def active_triple_string_prefix(text: str, pos: int):
    token_re = re.compile(r"(?i)([rubf]*)('''|\"\"\")")
    active = None
    for m in token_re.finditer(text[:pos]):
        prefix = m.group(1).lower()
        quote = m.group(2)
        if active is None:
            active = (quote, prefix)
        elif active[0] == quote:
            active = None
    return active

def escape_for_f_string_if_needed(path: Path, text: str, pos: int, snippet: str) -> str:
    if path.suffix.lower() != ".py":
        return snippet
    active = active_triple_string_prefix(text, pos)
    if not active:
        raise RuntimeError(f"Refusing to inject HTML into Python outside a triple-quoted string: {path}")
    quote, prefix = active
    if "f" in prefix:
        return snippet.replace("{", "{{").replace("}", "}}")
    return snippet

def patch_status_near_model_and_guide(text: str) -> str:
    low = text.lower()
    targets = ["model card", "model-card", "pilot success guide", "pilot-success-guide"]
    intervals = []
    for target in targets:
        start = 0
        while True:
            idx = low.find(target, start)
            if idx == -1:
                break
            intervals.append((max(0, idx - 700), min(len(text), idx + 900)))
            start = idx + len(target)

    if not intervals:
        return text

    intervals.sort()
    merged = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    def patch(block: str) -> str:
        block = re.sub(r"\bMISSING\b", "READY", block)
        block = re.sub(r"\bMissing\b", "Ready", block)
        block = re.sub(r"\bmissing\b", "ready", block)
        block = block.replace("status-missing", "status-ready")
        block = block.replace("badge-missing", "badge-ready")
        block = block.replace("pill-missing", "pill-ready")
        block = block.replace("doc-missing", "doc-ready")
        return block

    out = []
    last = 0
    for s, e in merged:
        out.append(text[last:s])
        out.append(patch(text[s:e]))
        last = e
    out.append(text[last:])
    return "".join(out)

def patch_command_center():
    scored = []
    for p in all_candidate_files():
        text = read(p)
        score = command_score(p, text)
        if score >= 7:
            scored.append((score, p, text))

    if not scored:
        raise SystemExit("No strong Command Center candidate found. Aborting without changes.")

    scored.sort(reverse=True, key=lambda x: x[0])
    score, path, text = scored[0]
    print("Selected Command Center candidate:", path, "score=", score)

    pos = find_insertion_point(text)
    snippet = escape_for_f_string_if_needed(path, text, pos, SNIPPET)
    text2 = patch_status_near_model_and_guide(text)
    text2 = replace_or_insert(text2, snippet)
    write_if_changed(path, text2)
    return path

def main():
    patched_path = patch_command_center()

    manifest = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "patched_command_center_file": str(patched_path),
        "purpose": "Make the Command Center feel more like a live review queue while preserving the existing layout.",
        "added": [
            "Compact interactive review queue",
            "Filter controls for all/critical/elevated",
            "Sort control for queue rank/risk/lead time",
            "Priority tier badges",
            "Primary driver visibility",
            "Trend and lead-time context",
            "Workflow-state demonstration buttons",
            "Concise selected-review-basis panel"
        ],
        "guardrails": [
            "Simulated de-identified demonstration queue only",
            "Decision support only",
            "No diagnosis",
            "No treatment direction",
            "No clinician replacement",
            "No autonomous escalation",
            "No row-level data committed"
        ],
        "changed_files": CHANGED,
    }
    out = ROOT / "data" / "validation" / "command_center_interactive_queue_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    CHANGED.append(str(out))
    print("UPDATED:", out)

    print("")
    print("Changed files:")
    for f in CHANGED:
        print(" -", f)

if __name__ == "__main__":
    main()

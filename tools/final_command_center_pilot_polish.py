from pathlib import Path
import re
import json
from datetime import datetime, timezone

p = Path("era/web/command_center.py")
s = p.read_text(encoding="utf-8")

def target_is_f_string(text: str) -> bool:
    m = re.search(r"COMMAND_CENTER_HTML\s*=\s*([rubfRUBF]*)[\"']{3}", text)
    return bool(m and "f" in m.group(1).lower())

IS_F = target_is_f_string(s)

def safe_block(block: str) -> str:
    # If the HTML lives inside a Python f-string, inserted CSS/JS braces must be escaped.
    return block.replace("{", "{{").replace("}", "}}") if IS_F else block

POLISH_CSS = r"""
/* === ERA FINAL PILOT POLISH: compact clinical workboard hierarchy === */
.pilot-polish-marker{display:none}

.queue-panel{
  max-width:100%;
}

.queue-head h1{
  max-width:860px;
}

.queue-head p{
  max-width:760px;
}

.card-row{
  align-items:stretch;
}

.patient-card{
  position:relative;
  overflow:hidden;
  padding:10px 10px 11px;
  min-height:132px;
}

.patient-card .pill,
.patient-top .pill{
  position:relative;
  z-index:2;
  flex-shrink:0;
  max-width:104px;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  padding:5px 7px;
  font-size:9px;
}

.rank-title{
  font-size:16px;
  line-height:1.02;
  max-width:130px;
  overflow-wrap:anywhere;
}

.unit{
  font-size:11px;
}

.patient-card .score{
  font-size:30px;
  margin-top:8px;
}

.patient-card .score small{
  font-size:12px;
}

.patient-card .mini{
  font-size:11px;
  line-height:1.22;
  margin-top:6px;
}

.patient-card .mini b{
  color:#eaf3ff;
}

.card-row .patient-card:first-child{
  border-color:rgba(167,255,154,.78);
  background:
    radial-gradient(circle at 8% 0%,rgba(167,255,154,.18),transparent 36%),
    linear-gradient(180deg,rgba(30,48,73,.98),rgba(12,22,38,.98));
  box-shadow:
    0 0 0 1px rgba(167,255,154,.24),
    0 18px 48px rgba(0,0,0,.34);
}

.card-row .patient-card:first-child::before{
  content:"TOP REVIEW";
  position:absolute;
  top:8px;
  right:10px;
  border:1px solid rgba(167,255,154,.45);
  color:#d7ffd1;
  background:rgba(167,255,154,.12);
  border-radius:999px;
  padding:4px 7px;
  font-size:8px;
  font-weight:1000;
  letter-spacing:.11em;
  z-index:1;
}

.card-row .patient-card:first-child .rank-title{
  font-size:21px;
  max-width:170px;
}

.card-row .patient-card:first-child .score{
  font-size:40px;
  text-shadow:0 0 24px rgba(167,255,154,.22);
}

.card-row .patient-card:first-child .pill{
  margin-top:22px;
}

.detail{
  box-shadow:inset 4px 0 0 rgba(167,255,154,.35);
}

.detail-title h3{
  font-size:30px;
}

.detail .score{
  font-size:42px;
  margin:4px 0 8px;
}

.reason{
  font-size:12px;
  line-height:1.28;
  padding:10px;
}

.detail ul{
  font-size:11px;
  line-height:1.38;
  margin-top:8px;
}

.vital{
  padding:8px;
}

.vital b{
  font-size:16px;
}

.tier-critical{
  box-shadow:0 0 0 1px rgba(255,143,163,.18);
}

.tier-elevated{
  box-shadow:0 0 0 1px rgba(255,211,109,.18);
}

.tier-watch{
  box-shadow:0 0 0 1px rgba(155,215,255,.15);
}

.lower{
  margin-top:10px;
}

.about-demo{
  margin-top:12px;
  border:1px solid rgba(184,211,255,.16);
  background:rgba(255,255,255,.035);
  border-radius:18px;
  overflow:hidden;
}

.about-demo summary{
  cursor:pointer;
  list-style:none;
  padding:12px 14px;
  color:#dfeaff;
  font-size:12px;
  font-weight:1000;
  letter-spacing:.08em;
  text-transform:uppercase;
  display:flex;
  align-items:center;
  justify-content:space-between;
}

.about-demo summary::-webkit-details-marker{
  display:none;
}

.about-demo summary::after{
  content:"Open";
  border:1px solid rgba(184,211,255,.22);
  background:rgba(255,255,255,.06);
  border-radius:999px;
  padding:5px 9px;
  color:#cfe0f4;
  font-size:10px;
  letter-spacing:.08em;
}

.about-demo[open] summary::after{
  content:"Close";
}

.about-demo .lower{
  padding:0 12px 12px;
}

.about-demo .panel{
  box-shadow:none;
}

.footer-guardrail{
  margin-top:10px;
  padding:10px 12px;
  font-size:11px;
}

@media(min-width:1000px){
  .card-row .patient-card:first-child{
    grid-column:span 2;
  }
}

@media(max-width:1150px){
  .card-row .patient-card:first-child{
    grid-column:auto;
  }
  .card-row .patient-card:first-child .rank-title{
    font-size:18px;
  }
  .card-row .patient-card:first-child .score{
    font-size:34px;
  }
}
"""

POLISH_JS = r"""
/* === ERA FINAL PILOT POLISH RUNTIME === */
(function(){
  function polishCommandCenter(){
    try{
      var cards = Array.prototype.slice.call(document.querySelectorAll(".patient-card"));
      cards.forEach(function(card, index){
        card.classList.toggle("top-review-card", index === 0);
        var mini = card.querySelector(".mini");
        if(mini){
          mini.innerHTML = mini.innerHTML
            .replace(/<b>Driver:<\/b>/g, "<b>Driver</b>")
            .replace(/<b>Trend:<\/b>/g, "<b>Trend</b>")
            .replace(/<b>Lead:<\/b>/g, "<b>Lead</b>");
        }
      });

      var reason = document.getElementById("reason");
      if(reason){
        reason.innerHTML = reason.innerHTML
          .replace("is the selected primary driver.", "drives this review position.")
          .replace("Lead-time context:", "Lead:");
      }

      var lower = document.querySelector("section.lower");
      if(lower && !lower.closest("details.about-demo")){
        var details = document.createElement("details");
        details.className = "about-demo";
        var summary = document.createElement("summary");
        summary.textContent = "About this demo, metric clarity, and governance links";
        lower.parentNode.insertBefore(details, lower);
        details.appendChild(summary);
        details.appendChild(lower);
      }

      document.querySelectorAll(".patient-card .score, .score-cell, #detailScore").forEach(function(el){
        el.innerHTML = el.innerHTML
          .replace(/(\d{1,2}(?:\.\d)?)\s*\/\s*10/g, "$1<small>/10</small>")
          .replace(/%/g, "");
      });
    }catch(e){}
  }

  var originalRender = window.render;
  if(typeof originalRender === "function" && !window.__eraPilotPolishWrapped){
    window.render = function(){
      var result = originalRender.apply(this, arguments);
      setTimeout(polishCommandCenter, 0);
      return result;
    };
    window.__eraPilotPolishWrapped = true;
  }

  polishCommandCenter();
  setInterval(polishCommandCenter, 1200);
})();
"""

# 1) Insert CSS before </style>
if "ERA FINAL PILOT POLISH: compact clinical workboard hierarchy" not in s:
    if "</style>" not in s:
        raise SystemExit("Could not find </style> for CSS insertion.")
    s = s.replace("</style>", safe_block(POLISH_CSS) + "\n</style>", 1)

# 2) Insert JS before final </script>
if "ERA FINAL PILOT POLISH RUNTIME" not in s:
    if "</script>" not in s:
        raise SystemExit("Could not find </script> for JS insertion.")
    idx = s.rfind("</script>")
    s = s[:idx] + safe_block(POLISH_JS) + "\n" + s[idx:]

# 3) If lower section exists and is not already wrapped, do a source-level wrap too.
if '<details class="about-demo"' not in s and '<section class="lower">' in s:
    s = re.sub(
        r'(<section class="lower">.*?</section>)',
        r'<details class="about-demo">\n  <summary>About this demo, metric clarity, and governance links</summary>\n  \1\n</details>',
        s,
        count=1,
        flags=re.S
    )

# 4) Tighten visible copy where exact strings exist.
replacements = {
    "Compact workboard showing rank, unit, priority tier, review score, primary driver, trend, lead-time context, and workflow state in one glanceable view.":
    "Compact workboard for rank, tier, driver, trend, lead time, and workflow state.",

    "Decision support only. Does not diagnose, direct treatment, replace clinician judgment, or independently trigger escalation. Simulated demo data only. Retrospective MIMIC-IV + eICU validation remains aggregate and DUA-safe.":
    "Decision support only. No diagnosis, treatment direction, clinician replacement, or independent escalation. Simulated demo data only; validation evidence remains aggregate and DUA-safe.",

    "Workflow buttons are audit/workflow-state examples only: acknowledge, assign, escalate review, resolve.":
    "Workflow buttons are audit/workflow-state examples only."
}

for old, new in replacements.items():
    s = s.replace(old, new)

# 5) Keep brand correct.
s = s.replace("Early Review Score Alert AI", "Early Risk Alert AI")
s = s.replace("EarlyReview ScoreAlert AI", "Early Risk Alert AI")

# 6) Write updated module.
p.write_text(s, encoding="utf-8")

manifest = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Final pilot polish for Command Center UI: stronger top-patient hierarchy, tighter queue cards, collapsible explanatory sections, and clearer clinical tool feel.",
    "changes": [
        "Top review card receives stronger visual hierarchy and larger review score",
        "Queue cards use tighter text and badge containment",
        "Selected patient detail panel is more prominent and less wordy",
        "Long explanatory sections move into a collapsible About this demo area",
        "Tier badges and Review Score styling are strengthened",
        "Conservative decision-support guardrails remain visible"
    ],
    "score_policy": "Review Score remains 0-10 only; no patient-risk percentage display.",
    "claim_guardrail": "Decision support only. No diagnosis, treatment direction, clinician replacement, or independent escalation.",
    "raw_data_policy": "No raw MIMIC, eICU, HiRID, CSV, parquet, zip, or local_private files committed."
}

Path("data/validation/final_command_center_pilot_polish_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("PATCHED: final Command Center pilot polish applied.")

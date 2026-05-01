from __future__ import annotations

from pathlib import Path
import re
import json
import py_compile
from datetime import datetime, timezone

p = Path("era/web/command_center.py")
if not p.exists():
    raise SystemExit("ERROR: era/web/command_center.py not found.")

s = p.read_text(encoding="utf-8")
original = s

# Remove previous final polish markers if this is rerun.
s = re.sub(
    r"/\* ERA_FINAL_COMMAND_CENTER_POLISH_V4_START \*/.*?/\* ERA_FINAL_COMMAND_CENTER_POLISH_V4_END \*/",
    "",
    s,
    flags=re.S,
)
s = re.sub(
    r"<script>\s*// ERA_FINAL_COMMAND_CENTER_POLISH_JS_V4_START.*?// ERA_FINAL_COMMAND_CENTER_POLISH_JS_V4_END\s*</script>",
    "",
    s,
    flags=re.S,
)

# Correct accidental brand corruption if it exists.
s = s.replace("Early Review Score Alert AI", "Early Risk Alert AI")
s = s.replace("EarlyReview ScoreAlert AI", "Early Risk Alert AI")

# Remove the old unrealistic static ICU-12 99% display if any leftover text remains.
s = s.replace("99% risk", "9.2/10 review score")
s = s.replace("Risk 99%", "Review Score 9.2/10")
s = s.replace("RISK 99%", "REVIEW SCORE 9.2/10")
s = s.replace(">99%<", ">9.2/10<")
s = s.replace("Review score 99%", "Review Score 9.2/10")
s = s.replace("Review Score 99%", "Review Score 9.2/10")

css = r"""
/* ERA_FINAL_COMMAND_CENTER_POLISH_V4_START */

/* Final pilot polish: make the tool feel like a live workboard, not a brochure. */
body{
  background:
    radial-gradient(circle at 10% 0%,rgba(78,166,255,.14),transparent 30%),
    radial-gradient(circle at 88% 4%,rgba(167,255,154,.10),transparent 24%),
    linear-gradient(180deg,#06101d 0%,#07111f 46%,#050912 100%) !important;
}

/* Keep pilot banner readable and professional. */
.pilot-banner{
  background:linear-gradient(90deg,rgba(255,211,109,.08),rgba(167,255,154,.045)) !important;
  border-color:rgba(255,211,109,.32) !important;
  color:#f7fbff !important;
}
.pilot-banner strong,
.pilot-banner p{
  color:#f7fbff !important;
  text-shadow:none !important;
}
.pilot-banner p{
  max-width:980px;
}

/* Make the main queue feel like the hero. */
.queue-panel{
  border-color:rgba(167,255,154,.34) !important;
  box-shadow:0 22px 80px rgba(0,0,0,.42) !important;
}
.queue-head h1{
  letter-spacing:-.06em !important;
}
.queue-head p{
  max-width:760px !important;
  font-size:13px !important;
}

/* Reduce density in summary metrics. */
.mini-metrics{
  gap:8px !important;
  margin:10px 0 !important;
}
.metric{
  min-height:74px !important;
  padding:10px !important;
}
.metric b{
  font-size:24px !important;
}
.metric span{
  font-size:11px !important;
}

/* Stronger card hierarchy. First visible card becomes dominant. */
.card-row{
  grid-template-columns:1.55fr 1fr 1fr !important;
  align-items:stretch !important;
  gap:10px !important;
}
.patient-card{
  position:relative !important;
  overflow:hidden !important;
  min-height:118px !important;
  padding:12px !important;
  border-radius:17px !important;
}
.patient-card:first-child{
  grid-row:span 2 !important;
  min-height:210px !important;
  border:2px solid rgba(167,255,154,.78) !important;
  background:
    radial-gradient(circle at 20% 0%,rgba(167,255,154,.16),transparent 44%),
    linear-gradient(180deg,rgba(20,42,58,.98),rgba(10,22,37,.98)) !important;
  box-shadow:
    0 0 0 1px rgba(167,255,154,.18),
    0 20px 55px rgba(0,0,0,.38) !important;
}
.patient-card:first-child:before{
  content:"TOP PRIORITY";
  display:inline-flex;
  align-items:center;
  justify-content:center;
  margin-bottom:8px;
  padding:5px 8px;
  border-radius:999px;
  color:#d8ffd4;
  background:rgba(167,255,154,.14);
  border:1px solid rgba(167,255,154,.46);
  font-size:10px;
  letter-spacing:.14em;
  font-weight:1000;
}
.patient-card:first-child .rank-title{
  font-size:24px !important;
  line-height:1 !important;
}
.patient-card:first-child .score{
  font-size:42px !important;
  margin-top:10px !important;
}
.patient-card:first-child .mini{
  font-size:12px !important;
  line-height:1.32 !important;
}

/* Keep tier badges inside the cards and make tier primary. */
.patient-top{
  display:grid !important;
  grid-template-columns:minmax(0,1fr) auto !important;
  align-items:start !important;
  gap:8px !important;
}
.patient-card .pill,
.detail .pill,
td .pill{
  position:static !important;
  transform:none !important;
  max-width:100% !important;
  white-space:nowrap !important;
  flex-shrink:0 !important;
  font-size:11px !important;
  line-height:1 !important;
  padding:7px 9px !important;
  box-shadow:0 8px 22px rgba(0,0,0,.20) !important;
}
.patient-card:first-child .pill{
  font-size:12px !important;
  padding:8px 10px !important;
}
.tier-critical{
  background:rgba(255,143,163,.20) !important;
  border-color:rgba(255,143,163,.76) !important;
  color:#ffe3e8 !important;
}
.tier-elevated{
  background:rgba(255,211,109,.20) !important;
  border-color:rgba(255,211,109,.76) !important;
  color:#fff0bf !important;
}
.tier-watch{
  background:rgba(155,215,255,.17) !important;
  border-color:rgba(155,215,255,.62) !important;
  color:#d6f0ff !important;
}

/* Shorter, punchier queue-card text. */
.mini{
  font-size:11px !important;
  line-height:1.24 !important;
  margin-top:6px !important;
}
.rank-title{
  overflow-wrap:anywhere !important;
}
.score{
  margin-top:8px !important;
}

/* Make selected patient panel scan faster. */
.detail{
  padding:13px !important;
}
.detail-title h3{
  font-size:25px !important;
}
.detail-sub{
  font-size:11px !important;
}
.detail .score{
  font-size:34px !important;
  margin:8px 0 !important;
}
.vitals{
  gap:7px !important;
  margin:8px 0 !important;
}
.vital{
  padding:8px !important;
}
.vital b{
  font-size:16px !important;
}
.reason{
  font-size:12px !important;
  line-height:1.28 !important;
  padding:10px !important;
  margin-top:8px !important;
}
.detail ul{
  margin-top:7px !important;
  font-size:11.5px !important;
  line-height:1.36 !important;
}
.detail li:nth-child(n+4){
  display:none !important;
}

/* Keep explanatory sections available but reduce brochure feel. */
details.about-demo{
  margin-top:12px;
  border:1px solid rgba(184,211,255,.16);
  background:rgba(255,255,255,.035);
  border-radius:18px;
  padding:10px 12px;
}
details.about-demo > summary{
  cursor:pointer;
  color:#d9e7f7;
  font-size:12px;
  font-weight:1000;
  letter-spacing:.12em;
  text-transform:uppercase;
}
details.about-demo .lower{
  margin-top:10px !important;
}
details.about-demo:not([open]) .lower{
  display:none !important;
}

/* Cleaner guardrails. */
.footer-guardrail,
.guardrail{
  font-size:11.5px !important;
  line-height:1.34 !important;
}
.footer-guardrail{
  padding:10px 12px !important;
}

/* Keep table compact. */
table{
  font-size:11.5px !important;
}
th{
  font-size:9.5px !important;
  padding:9px 8px !important;
}
td{
  padding:9px 8px !important;
}

/* Responsive behavior: no broken badges/cards on smaller screens. */
@media(max-width:1200px){
  .card-row{
    grid-template-columns:repeat(2,minmax(0,1fr)) !important;
  }
  .patient-card:first-child{
    grid-column:1 / -1 !important;
    grid-row:auto !important;
    min-height:170px !important;
  }
}
@media(max-width:760px){
  .card-row{
    grid-template-columns:1fr !important;
  }
  .patient-card:first-child{
    grid-column:auto !important;
    min-height:150px !important;
  }
}

/* ERA_FINAL_COMMAND_CENTER_POLISH_V4_END */
"""

js = r"""
<script>
// ERA_FINAL_COMMAND_CENTER_POLISH_JS_V4_START
(function(){
  function tightenCommandCenter(){
    try{
      document.title = "Early Risk Alert AI — Command Center";

      // Collapse explanatory blocks into a small footer-style area if present.
      const lower = document.querySelector("section.lower");
      if(lower && !document.querySelector("details.about-demo")){
        const details = document.createElement("details");
        details.className = "about-demo";
        const summary = document.createElement("summary");
        summary.textContent = "About this demo, metric clarity, and governance links";
        lower.parentNode.insertBefore(details, lower);
        details.appendChild(summary);
        details.appendChild(lower);
      }

      // Make guardrail wording explicit and short.
      const footer = document.querySelector(".footer-guardrail");
      if(footer){
        footer.textContent = "No diagnosis. No treatment direction. Does not replace clinician judgment. Does not independently trigger escalation. Simulated demo queue only; validation evidence remains aggregate and DUA-safe.";
      }

      // Add a visible top-priority label to the first active card without duplicating it.
      const firstCard = document.querySelector(".patient-card");
      if(firstCard && !firstCard.dataset.finalPolished){
        firstCard.dataset.finalPolished = "true";
      }

      // Keep selected panel bullets tight.
      const detailList = document.querySelectorAll(".detail li");
      detailList.forEach((li, index) => {
        if(index > 2){ li.style.display = "none"; }
      });

      // Ensure no remaining visible percent scoring in queue cards.
      document.querySelectorAll(".score, .score-cell").forEach(el => {
        const text = (el.textContent || "").trim();
        if(/^\d{2,3}%$/.test(text)){
          el.textContent = text.replace("%","") + "/10";
        }
      });
    }catch(e){
      console.warn("ERA final command-center polish skipped:", e);
    }
  }

  if(typeof render === "function" && !window.__eraFinalPolishWrapped){
    const previousRender = render;
    render = function(){
      previousRender();
      tightenCommandCenter();
    };
    window.__eraFinalPolishWrapped = true;
  }

  if(document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", tightenCommandCenter);
  }else{
    tightenCommandCenter();
  }

  setTimeout(tightenCommandCenter, 250);
  setTimeout(tightenCommandCenter, 1000);
})();
// ERA_FINAL_COMMAND_CENTER_POLISH_JS_V4_END
</script>
"""

if "ERA_FINAL_COMMAND_CENTER_POLISH_V4_START" not in s:
    if "</style>" not in s:
        raise SystemExit("ERROR: Could not find </style> in command_center.py")
    s = s.replace("</style>", css + "\n</style>", 1)

if "ERA_FINAL_COMMAND_CENTER_POLISH_JS_V4_START" not in s:
    if "</body>" in s:
        s = s.replace("</body>", js + "\n</body>", 1)
    else:
        s += "\n" + js + "\n"

# Ensure there is explicit safe wording.
if "No diagnosis" not in s:
    s = s.replace(
        "Decision support only.",
        "No diagnosis. Decision support only.",
        1
    )

p.write_text(s, encoding="utf-8")
py_compile.compile(str(p), doraise=True)

s2 = p.read_text(encoding="utf-8")

required = [
    "ERA_FINAL_COMMAND_CENTER_POLISH_V4_START",
    "TOP PRIORITY",
    "About this demo, metric clarity, and governance links",
    "No diagnosis",
    "Review Score",
    "0–10",
    "patient-card:first-child",
    "tier-critical",
]

missing = [x for x in required if x not in s2]
if missing:
    raise SystemExit(f"Missing required final polish content: {missing}")

bad_brand = ["Early Review Score Alert AI", "EarlyReview ScoreAlert AI"]
found_brand = [x for x in bad_brand if x in s2]
if found_brand:
    raise SystemExit(f"Bad brand text still found: {found_brand}")

bad_static = ["Risk 99%", "RISK 99%", ">99%<", "ICU-12 99%"]
found_static = [x for x in bad_static if x in s2]
if found_static:
    raise SystemExit(f"Old static ICU-12 risk text still found: {found_static}")

manifest = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Final Command Center pilot polish based on assessment feedback.",
    "changes": [
        "Makes the #1 patient card visually dominant with TOP PRIORITY label.",
        "Keeps tier badges inside cards and makes tier visually primary.",
        "Tightens queue card text and selected-patient detail panel.",
        "Moves lower explanatory/demo sections into a collapsible About this demo area.",
        "Keeps Review Score on a 0-10 scale.",
        "Preserves conservative decision-support guardrail wording."
    ],
    "claims_guardrail": "No diagnosis. No treatment direction. Does not replace clinician judgment. Does not independently trigger escalation.",
    "raw_data_policy": "No raw MIMIC/eICU/HiRID files, local_private files, CSVs, parquet, archives, or row-level outputs committed."
}
Path("data/validation/final_command_center_pilot_polish_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("PATCHED: era/web/command_center.py")
print("PYTHON SYNTAX OK")
print("FINAL PILOT POLISH CHECK OK")

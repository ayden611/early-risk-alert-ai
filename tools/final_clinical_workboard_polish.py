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

# Detect whether COMMAND_CENTER_HTML is an f-string. If yes, escape injected CSS/JS braces.
m = re.search(r"COMMAND_CENTER_HTML\s*=\s*([rubfRUBF]*)[\"']{3}", s)
html_prefix = (m.group(1).lower() if m else "")
is_f_string_html = "f" in html_prefix

def html_safe(block: str) -> str:
    if is_f_string_html:
        return block.replace("{", "{{").replace("}", "}}")
    return block

# Remove older final-polish blocks if rerun.
s = re.sub(
    r"/\* ERA_CLINICAL_WORKBOARD_FINAL_POLISH_START \*/.*?/\* ERA_CLINICAL_WORKBOARD_FINAL_POLISH_END \*/",
    "",
    s,
    flags=re.S,
)
s = re.sub(
    r"<script>\s*// ERA_CLINICAL_WORKBOARD_FINAL_POLISH_JS_START.*?// ERA_CLINICAL_WORKBOARD_FINAL_POLISH_JS_END\s*</script>",
    "",
    s,
    flags=re.S,
)

# Correct accidental brand corruption.
s = s.replace("Early Review Score Alert AI", "Early Risk Alert AI")
s = s.replace("EarlyReview ScoreAlert AI", "Early Risk Alert AI")

# Remove old confusing static risk language if present.
replacements = {
    "99% risk": "9.2/10 review score",
    "Risk 99%": "Review Score 9.2/10",
    "RISK 99%": "REVIEW SCORE 9.2/10",
    "Review score 99%": "Review Score 9.2/10",
    "Review Score 99%": "Review Score 9.2/10",
    "patient-risk percentage": "review-prioritization score",
}
for old, new in replacements.items():
    s = s.replace(old, new)

css = r"""
/* ERA_CLINICAL_WORKBOARD_FINAL_POLISH_START */

/* Final direction: queue first, fewer words, clearer clinical workboard hierarchy. */
.queue-panel{
  border-color:rgba(167,255,154,.44) !important;
  box-shadow:0 22px 90px rgba(0,0,0,.46) !important;
}

/* Shorten visual footprint of descriptive copy near the title. */
.queue-head p{
  max-width:620px !important;
  font-size:12px !important;
  line-height:1.28 !important;
  color:#cbd8e8 !important;
}
.queue-head h1{
  margin-bottom:4px !important;
}

/* Make the #1 patient visually dominant. */
.card-row{
  grid-template-columns:1.75fr 1fr 1fr !important;
  gap:10px !important;
  align-items:stretch !important;
}
.patient-card{
  position:relative !important;
  min-height:112px !important;
  padding:11px !important;
  border-radius:16px !important;
  overflow:hidden !important;
}
.patient-card:first-child{
  grid-row:span 2 !important;
  min-height:218px !important;
  border:2px solid rgba(167,255,154,.88) !important;
  background:
    radial-gradient(circle at 18% 0%,rgba(167,255,154,.22),transparent 43%),
    linear-gradient(180deg,rgba(22,48,62,.98),rgba(9,20,36,.98)) !important;
  box-shadow:
    0 0 0 1px rgba(167,255,154,.20),
    0 22px 60px rgba(0,0,0,.45),
    inset 0 1px 0 rgba(255,255,255,.08) !important;
}
.patient-card:first-child::before{
  content:"TOP PRIORITY";
  display:inline-flex;
  align-items:center;
  justify-content:center;
  margin-bottom:8px;
  padding:5px 9px;
  border-radius:999px;
  color:#06101d;
  background:linear-gradient(90deg,#a7ff9a,#dfffd9);
  border:1px solid rgba(167,255,154,.9);
  font-size:10px;
  letter-spacing:.14em;
  font-weight:1000;
  box-shadow:0 0 22px rgba(167,255,154,.22);
}
.patient-card:first-child .rank-title{
  font-size:26px !important;
  line-height:.96 !important;
}
.patient-card:first-child .score{
  font-size:38px !important;
  margin-top:8px !important;
}

/* Tier is the primary visual cue; score is still visible but secondary. */
.patient-top{
  display:grid !important;
  grid-template-columns:minmax(0,1fr) auto !important;
  gap:8px !important;
  align-items:start !important;
}
.patient-card .pill,
.detail .pill,
td .pill{
  position:static !important;
  transform:none !important;
  white-space:nowrap !important;
  max-width:100% !important;
  flex-shrink:0 !important;
  font-size:11px !important;
  line-height:1 !important;
  padding:7px 10px !important;
  border-width:1.5px !important;
  box-shadow:0 8px 22px rgba(0,0,0,.18) !important;
}
.patient-card:first-child .pill{
  font-size:12px !important;
  padding:8px 11px !important;
}
.tier-critical{
  background:rgba(255,143,163,.22) !important;
  border-color:rgba(255,143,163,.86) !important;
  color:#ffe4ea !important;
}
.tier-elevated{
  background:rgba(255,211,109,.22) !important;
  border-color:rgba(255,211,109,.86) !important;
  color:#fff1c7 !important;
}
.tier-watch{
  background:rgba(155,215,255,.17) !important;
  border-color:rgba(155,215,255,.70) !important;
  color:#dbf3ff !important;
}

/* Tighten queue card text. */
.mini{
  font-size:10.8px !important;
  line-height:1.18 !important;
  margin-top:5px !important;
}
.patient-card:first-child .mini{
  font-size:11.5px !important;
  line-height:1.25 !important;
}
.score{
  color:#bfffac !important;
  text-shadow:0 0 18px rgba(167,255,154,.16) !important;
}
.score small{
  opacity:.82 !important;
}

/* Tighten summary metrics. */
.mini-metrics{
  margin:8px 0 10px !important;
  gap:8px !important;
}
.metric{
  min-height:68px !important;
  padding:9px 10px !important;
}
.metric small{
  font-size:9.5px !important;
  margin-bottom:4px !important;
}
.metric b{
  font-size:23px !important;
}
.metric span{
  font-size:10.8px !important;
  margin-top:3px !important;
}

/* Right-side selected patient detail: shorter, cleaner, more visual. */
.detail{
  padding:12px !important;
  border-color:rgba(167,255,154,.25) !important;
}
.detail-title h3{
  font-size:25px !important;
}
.detail-sub{
  font-size:11px !important;
  color:#c7d5e7 !important;
}
.detail .score{
  font-size:32px !important;
  margin:6px 0 8px !important;
}
.vitals{
  gap:7px !important;
  margin:8px 0 !important;
}
.vital{
  padding:8px !important;
  border-radius:11px !important;
}
.vital small{
  font-size:9px !important;
}
.vital b{
  font-size:15.5px !important;
}
.reason{
  margin-top:8px !important;
  padding:9px 10px !important;
  font-size:11.5px !important;
  line-height:1.25 !important;
}
.detail ul{
  margin:7px 0 0 16px !important;
  font-size:11px !important;
  line-height:1.32 !important;
}
.detail li:nth-child(n+4){
  display:none !important;
}

/* Convert long explanatory material into small collapsed footer. */
details.about-demo{
  margin-top:12px !important;
  border:1px solid rgba(184,211,255,.16) !important;
  background:rgba(255,255,255,.035) !important;
  border-radius:16px !important;
  padding:9px 11px !important;
}
details.about-demo > summary{
  cursor:pointer !important;
  color:#dbe8f9 !important;
  font-size:11px !important;
  font-weight:1000 !important;
  letter-spacing:.12em !important;
  text-transform:uppercase !important;
}
details.about-demo:not([open]) .lower{
  display:none !important;
}
details.about-demo[open] .lower{
  margin-top:10px !important;
}

/* Reduce visual noise from view-scope explanation while keeping the required disclaimer. */
.era-scope-note{
  margin:8px 0 10px !important;
  padding:8px 10px !important;
  border:1px solid rgba(155,215,255,.22) !important;
  background:rgba(155,215,255,.055) !important;
  border-radius:12px !important;
  color:#c9d8eb !important;
  font-size:11px !important;
  line-height:1.25 !important;
  font-weight:800 !important;
}

/* Compact guardrails. */
.guardrail,
.footer-guardrail{
  font-size:11px !important;
  line-height:1.28 !important;
  padding:9px 10px !important;
}
.footer-guardrail{
  text-align:center !important;
}

/* Table stays compact and scan-friendly. */
table{
  font-size:11px !important;
}
th{
  font-size:9px !important;
  padding:8px 7px !important;
}
td{
  padding:8px 7px !important;
}
.score-cell{
  color:#c8ffad !important;
}

/* Prevent badge spillover on narrower screens. */
@media(max-width:1250px){
  .card-row{
    grid-template-columns:repeat(2,minmax(0,1fr)) !important;
  }
  .patient-card:first-child{
    grid-column:1 / -1 !important;
    grid-row:auto !important;
    min-height:168px !important;
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

/* ERA_CLINICAL_WORKBOARD_FINAL_POLISH_END */
"""

js = r"""
<script>
// ERA_CLINICAL_WORKBOARD_FINAL_POLISH_JS_START
(function(){
  function textIncludes(el, phrase){
    return el && (el.textContent || "").toLowerCase().includes(phrase.toLowerCase());
  }

  function tightenVisibleText(){
    try{
      document.title = "Early Risk Alert AI — Command Center";

      // Shorten the repeated title paragraph once the visible queue is doing the work.
      var queueHead = document.querySelector(".queue-head p");
      if(queueHead){
        queueHead.textContent = "Live review workboard: tier, driver, trend, lead time, and workflow state.";
      }

      // Make the view-scope disclaimer smaller without removing it.
      var allNodes = Array.prototype.slice.call(document.querySelectorAll("p,div,section"));
      allNodes.forEach(function(el){
        if(textIncludes(el, "View scope: simulated unit filter for pilot demonstration")){
          el.classList.add("era-scope-note");
        }
      });

      // Convert lower explanatory sections into collapsed About area.
      var lower = document.querySelector("section.lower");
      if(lower && !document.querySelector("details.about-demo")){
        var details = document.createElement("details");
        details.className = "about-demo";
        var summary = document.createElement("summary");
        summary.textContent = "About this demo, metric clarity, and governance links";
        lower.parentNode.insertBefore(details, lower);
        details.appendChild(summary);
        details.appendChild(lower);
      }

      // Shorten selected-detail guardrail if present.
      var guardrails = Array.prototype.slice.call(document.querySelectorAll(".guardrail"));
      guardrails.forEach(function(el){
        if(textIncludes(el, "Selected patient detail is review-support context only")){
          el.textContent = "Review support only — clinician verifies patient context.";
        }
      });

      // Keep final footer guardrail short and conservative.
      var footer = document.querySelector(".footer-guardrail");
      if(footer){
        footer.textContent = "No diagnosis. No treatment direction. Does not replace clinician judgment. Does not independently trigger escalation. Simulated demo queue only; validation evidence remains aggregate and DUA-safe.";
      }

      // Tighten card microcopy. Keep driver/trend/lead only.
      Array.prototype.slice.call(document.querySelectorAll(".patient-card .mini")).forEach(function(el){
        var txt = el.innerHTML || "";
        txt = txt.replace(/<b>Driver:<\/b>/gi, "<b>Driver</b>");
        txt = txt.replace(/<b>Trend:<\/b>/gi, "<b>Trend</b>");
        txt = txt.replace(/<b>Lead:<\/b>/gi, "<b>Lead</b>");
        el.innerHTML = txt;
      });

      // Keep detail bullets to first three visible bullets.
      Array.prototype.slice.call(document.querySelectorAll(".detail li")).forEach(function(li, idx){
        if(idx > 2){ li.style.display = "none"; }
      });

      // Ensure visible scoring language stays 0-10, not patient-risk percent.
      Array.prototype.slice.call(document.querySelectorAll(".score, .score-cell")).forEach(function(el){
        var t = (el.textContent || "").trim();
        if(/^\d{2,3}%$/.test(t)){
          el.textContent = (parseFloat(t) / 10).toFixed(1) + "/10";
        }
      });
    }catch(e){
      console.warn("ERA clinical workboard final polish skipped:", e);
    }
  }

  if(typeof render === "function" && !window.__eraClinicalWorkboardFinalPolishWrapped){
    var previousRender = render;
    render = function(){
      previousRender();
      tightenVisibleText();
    };
    window.__eraClinicalWorkboardFinalPolishWrapped = true;
  }

  if(document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", tightenVisibleText);
  }else{
    tightenVisibleText();
  }

  setTimeout(tightenVisibleText, 250);
  setTimeout(tightenVisibleText, 1000);
})();
// ERA_CLINICAL_WORKBOARD_FINAL_POLISH_JS_END
</script>
"""

css = html_safe(css)
js = html_safe(js)

if "</style>" not in s:
    raise SystemExit("ERROR: Could not find </style> in command_center.py.")

s = s.replace("</style>", css + "\n</style>", 1)

if "</body>" in s:
    s = s.replace("</body>", js + "\n</body>", 1)
else:
    s += "\n" + js + "\n"

# Ensure conservative language exists somewhere.
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
    "ERA_CLINICAL_WORKBOARD_FINAL_POLISH_START",
    "TOP PRIORITY",
    "About this demo, metric clarity, and governance links",
    "No diagnosis",
    "Review Score",
    "0-10",
    "patient-card:first-child",
    "Review support only",
]

missing = [x for x in required if x not in s2]
if missing:
    raise SystemExit(f"Missing required final workboard polish content: {missing}")

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
    "purpose": "Final clinical workboard polish for the Command Center.",
    "assessment_agreement": [
        "The #1 patient needed stronger visual hierarchy.",
        "Queue card text needed to be shorter.",
        "Tier badges should be the primary visual cue and Review Score should remain secondary.",
        "Long explanatory sections should be collapsed so the queue feels more clinical and less brochure-like."
    ],
    "changes": [
        "Adds TOP PRIORITY visual treatment to the first queue card.",
        "Shortens queue header copy.",
        "Tightens queue-card microcopy.",
        "Strengthens Critical/Elevated/Watch badge styling.",
        "Shortens selected-patient explanation.",
        "Collapses lower explanatory sections into About this demo.",
        "Keeps View Scope disclaimer visible but visually smaller.",
        "Keeps conservative decision-support-only guardrail."
    ],
    "claim_guardrail": "No diagnosis. No treatment direction. Does not replace clinician judgment. Does not independently trigger escalation.",
    "raw_data_policy": "No raw MIMIC/eICU/HiRID/local_private/CSV/parquet/archive files committed."
}

Path("data/validation/final_clinical_workboard_polish_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("PATCHED: era/web/command_center.py")
print("PYTHON SYNTAX OK")
print("FINAL CLINICAL WORKBOARD POLISH CHECK OK")

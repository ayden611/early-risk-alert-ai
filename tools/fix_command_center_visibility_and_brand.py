from pathlib import Path
import re

p = Path("era/web/command_center.py")
s = p.read_text(encoding="utf-8")

print("Patching Command Center brand + visibility only...")

# 1) Correct corrupted brand text everywhere.
brand_patterns = [
    r"Early\s*Review\s*Score\s*Alert\s*AI",
    r"EarlyReview\s*Score\s*Alert\s*AI",
    r"Early\s*ReviewScore\s*Alert\s*AI",
    r"EarlyReviewScoreAlertAI",
]
for pat in brand_patterns:
    s = re.sub(pat, "Early Risk Alert AI", s, flags=re.I)

# Normalize title if it drifted.
s = re.sub(
    r"<title>.*?</title>",
    "<title>Early Risk Alert AI — Command Center</title>",
    s,
    count=1,
    flags=re.S | re.I,
)

# 2) Remove old visibility override block if it exists.
s = re.sub(
    r"/\* ERA_COMMAND_CENTER_VISIBILITY_FIX_START \*/.*?/\* ERA_COMMAND_CENTER_VISIBILITY_FIX_END \*/",
    "",
    s,
    flags=re.S,
)

# 3) Add a high-contrast CSS override before </style>.
# This does not change the queue logic. It fixes readability and the washed-out header/banner issue.
visibility_css = r"""
/* ERA_COMMAND_CENTER_VISIBILITY_FIX_START */

/* Dark production shell: prevents washed-out white/cream background behind the tool */
html,
body {
  background:
    radial-gradient(circle at 8% 0%, rgba(78,166,255,.16), transparent 30%),
    radial-gradient(circle at 90% 4%, rgba(167,255,154,.10), transparent 24%),
    linear-gradient(180deg, #050b16 0%, #06101d 48%, #030711 100%) !important;
  color: #f7fbff !important;
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(90deg, rgba(255,255,255,.025), transparent 18%, transparent 82%, rgba(255,255,255,.02)),
    radial-gradient(circle at 50% 0%, rgba(255,255,255,.045), transparent 34%);
  z-index: -1;
}

.wrap,
main.wrap {
  background: transparent !important;
}

/* Restore clean platform identity */
.brand-title,
.topbar .brand-title {
  color: #ffffff !important;
  text-shadow: 0 2px 18px rgba(0,0,0,.62) !important;
}

.brand-kicker,
.topbar .brand-kicker {
  color: #caffbf !important;
  text-shadow: 0 1px 10px rgba(0,0,0,.6) !important;
}

.brand-sub,
.topbar .brand-sub {
  color: #dbe8f8 !important;
}

/* Top bar should match the darker validation/evidence pages */
.topbar {
  background:
    linear-gradient(180deg, rgba(8,16,29,.98), rgba(6,13,24,.96)) !important;
  border-bottom: 1px solid rgba(184,211,255,.18) !important;
  box-shadow: 0 12px 35px rgba(0,0,0,.32) !important;
}

.topbar-inner {
  background: transparent !important;
}

.nav a {
  background: rgba(18,31,52,.94) !important;
  color: #f7fbff !important;
  border: 1px solid rgba(184,211,255,.22) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06) !important;
}

.nav a:hover {
  background: rgba(34,52,82,.98) !important;
  border-color: rgba(167,255,154,.38) !important;
}

/* Fix washed-out pilot banner: readable, compact, dark, and professional */
.pilot-banner {
  background:
    linear-gradient(90deg, rgba(20,34,56,.98), rgba(12,24,40,.96)) !important;
  border: 1px solid rgba(255,211,109,.35) !important;
  box-shadow: 0 16px 45px rgba(0,0,0,.30) !important;
  color: #f7fbff !important;
  opacity: 1 !important;
  filter: none !important;
}

.pilot-banner *,
.pilot-banner p,
.pilot-banner strong {
  color: #f7fbff !important;
  opacity: 1 !important;
  text-shadow: 0 1px 10px rgba(0,0,0,.58) !important;
}

.pilot-banner strong {
  color: #ffe6a2 !important;
}

.pilot-banner p {
  color: #e3ecf8 !important;
}

/* Ensure pills are readable on top banner */
.status-pills .pill,
.pilot-banner .pill {
  background: rgba(8,16,29,.88) !important;
  color: #f7fbff !important;
  border-color: rgba(184,211,255,.28) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.07) !important;
}

.status-pills .pill.green,
.pilot-banner .pill.green {
  color: #d7ffd1 !important;
  border-color: rgba(167,255,154,.42) !important;
  background: rgba(49,93,62,.48) !important;
}

.status-pills .pill.blue,
.pilot-banner .pill.blue {
  color: #d5e9ff !important;
  border-color: rgba(155,200,255,.42) !important;
  background: rgba(43,69,105,.50) !important;
}

.status-pills .pill.amber,
.pilot-banner .pill.amber {
  color: #ffe6a2 !important;
  border-color: rgba(255,211,109,.45) !important;
  background: rgba(102,73,21,.45) !important;
}

/* Match the clean dark card look used by validation pages */
.panel,
.queue-panel,
.side-panel,
.queue-box,
.detail,
.lower .panel {
  background:
    linear-gradient(180deg, rgba(20,34,56,.96), rgba(9,18,33,.96)) !important;
  border-color: rgba(184,211,255,.18) !important;
  color: #f7fbff !important;
}

.queue-panel {
  border-color: rgba(167,255,154,.28) !important;
}

.metric,
.side-card,
.patient-card,
.vital {
  background: rgba(255,255,255,.055) !important;
  border-color: rgba(184,211,255,.17) !important;
}

.patient-card.selected,
.patient-card:hover {
  border-color: rgba(167,255,154,.58) !important;
  box-shadow: 0 0 0 1px rgba(167,255,154,.16), 0 16px 45px rgba(0,0,0,.30) !important;
}

/* Prevent any pale page area from showing behind the right side */
.hero-grid,
.lower {
  background: transparent !important;
}

.footer-guardrail,
.guardrail {
  background: rgba(255,211,109,.085) !important;
  border-color: rgba(255,211,109,.45) !important;
  color: #ffe6a2 !important;
}

/* Make the table feel more like a tool */
table {
  background: rgba(8,16,29,.35) !important;
}

th {
  background: rgba(154,190,255,.20) !important;
  color: #f7fbff !important;
}

td {
  color: #f7fbff !important;
}

/* Mobile/smaller laptop readability */
@media(max-width: 1150px) {
  .hero-grid {
    grid-template-columns: 1fr !important;
  }

  .side-panel {
    max-width: none !important;
  }
}

/* ERA_COMMAND_CENTER_VISIBILITY_FIX_END */
"""

if "</style>" not in s:
    raise SystemExit("Could not find </style> in command_center.py HTML.")

s = s.replace("</style>", visibility_css + "\n</style>", 1)

# 4) Final safety corrections: prevent old static percent-risk wording from returning.
# Keep temperatures like 99.9°F. Only block the old patient risk percent labels.
s = s.replace("Early Review Score Alert AI", "Early Risk Alert AI")
s = s.replace("EarlyReview Score Alert AI", "Early Risk Alert AI")
s = s.replace("Risk 99%", "Review Score 9.2/10")
s = s.replace("RISK 99%", "REVIEW SCORE 9.2/10")
s = s.replace("Review Score 99%", "Review Score 9.2/10")
s = s.replace("Patient 1042 • p101 • Review score 99%", "Patient p101 • Review score 9.2/10")

p.write_text(s, encoding="utf-8")
print("Command Center visibility + brand patch applied.")

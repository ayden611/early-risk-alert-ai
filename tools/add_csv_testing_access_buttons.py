from __future__ import annotations

from pathlib import Path
import py_compile
import re
import json
from datetime import datetime, timezone

p = Path("era/web/command_center.py")
if not p.exists():
    raise SystemExit("ERROR: era/web/command_center.py not found.")

s = p.read_text(encoding="utf-8")
original = s

nav_block = """
        <a href="/data-ingest">CSV Upload / Data Testing</a>
        <a href="/retro-upload">Retrospective Upload</a>
"""

quick_access_block = """
    <!-- CSV_TESTING_ACCESS_V1_START -->
    <section class="pilot-banner" style="margin-top:12px;margin-bottom:14px;border-color:rgba(155,215,255,.38);background:linear-gradient(90deg,rgba(155,215,255,.10),rgba(167,255,154,.06));">
      <div>
        <strong>Data testing access • authorized pilot workflow</strong>
        <p>
          Use these tools to upload de-identified CSV files for controlled testing and retrospective review.
          Do not upload PHI, live patient data, or restricted row-level validation exports outside an approved pilot, data-use agreement, and security review.
        </p>
      </div>
      <div class="status-pills">
        <a class="pill blue" href="/data-ingest">CSV Upload / Data Testing</a>
        <a class="pill green" href="/retro-upload">Retrospective Upload</a>
      </div>
    </section>
    <!-- CSV_TESTING_ACCESS_V1_END -->
"""

# 1) Add visible nav links if not already present.
if "/data-ingest" not in s:
    if "</nav>" in s:
        s = s.replace("</nav>", nav_block + "\n      </nav>", 1)
    else:
        raise SystemExit("ERROR: Could not find </nav> in command_center.py to add CSV access links.")

# 2) Add a prominent quick-access panel below the pilot banner / before main queue.
if "CSV_TESTING_ACCESS_V1_START" not in s:
    # Best insertion point: before first hero-grid section.
    if '<section class="hero-grid">' in s:
        s = s.replace('<section class="hero-grid">', quick_access_block + '\n\n    <section class="hero-grid">', 1)
    # Fallback: after first pilot-banner section close.
    elif '</section>' in s:
        s = s.replace('</section>', '</section>\n\n' + quick_access_block, 1)
    else:
        raise SystemExit("ERROR: Could not find a safe insertion point for CSV access panel.")

# 3) Keep brand corrected.
s = s.replace("Early Review Score Alert AI", "Early Risk Alert AI")
s = s.replace("EarlyReview ScoreAlert AI", "Early Risk Alert AI")

p.write_text(s, encoding="utf-8")
py_compile.compile(str(p), doraise=True)

checks = [
    "/data-ingest",
    "/retro-upload",
    "CSV Upload / Data Testing",
    "Retrospective Upload",
    "Data testing access",
    "Do not upload PHI",
    "decision support",
]

s2 = p.read_text(encoding="utf-8")
missing = [x for x in checks if x not in s2]
if missing:
    raise SystemExit(f"Missing required access content: {missing}")

bad_brand = ["Early Review Score Alert AI", "EarlyReview ScoreAlert AI"]
found_brand = [x for x in bad_brand if x in s2]
if found_brand:
    raise SystemExit(f"Bad brand text still found: {found_brand}")

manifest = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Add visible direct access to existing CSV testing and retrospective upload routes.",
    "routes_added_to_command_center_navigation": ["/data-ingest", "/retro-upload"],
    "button_labels": ["CSV Upload / Data Testing", "Retrospective Upload"],
    "safety_wording": "Do not upload PHI, live patient data, or restricted row-level validation exports outside an approved pilot, data-use agreement, and security review.",
    "backend_change": "None. Existing upload routes remain unchanged."
}
Path("data/validation/csv_testing_access_buttons_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

print("PATCHED: command_center.py")
print("PYTHON SYNTAX OK")
print("CSV ACCESS BUTTONS OK")
print("MANIFEST WRITTEN")

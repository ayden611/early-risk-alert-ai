#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

ROOT = Path(".")
ERA = ROOT / "era"
DATA = ROOT / "data" / "validation"

START = "<!-- ERA_COMMAND_CENTER_TOOL_MODE_START -->"
END = "<!-- ERA_COMMAND_CENTER_TOOL_MODE_END -->"

SNIPPET = f"""
{START}
<link rel="stylesheet" href="/static/era_command_center_tool_mode.css?v=toolmode1">
<script src="/static/era_command_center_tool_mode.js?v=toolmode1"></script>
{END}
""".strip()

OLD_BLOCKS = [
    ("<!-- ERA_COMMAND_CENTER_TOOL_MODE_START -->", "<!-- ERA_COMMAND_CENTER_TOOL_MODE_END -->"),
    ("<!-- ERA_COMMAND_CENTER_SAFE_INPLACE_FIX_START -->", "<!-- ERA_COMMAND_CENTER_SAFE_INPLACE_FIX_END -->"),
    ("<!-- ERA_LIVE_UNIT_ROTATION_V2_START -->", "<!-- ERA_LIVE_UNIT_ROTATION_V2_END -->"),
    ("<!-- ERA_DEMO_QUEUE_POLISH_START -->", "<!-- ERA_DEMO_QUEUE_POLISH_END -->"),
]

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def remove_blocks(s: str) -> str:
    for a, b in OLD_BLOCKS:
        s = re.sub(re.escape(a) + r".*?" + re.escape(b), "", s, flags=re.S)
    return s

def score_file(p: Path) -> int:
    if not p.exists() or p.suffix not in {".py", ".html"}:
        return -1
    s = read(p)
    low = s.lower()
    score = 0
    if "command center" in low or "command-center" in low:
        score += 10
    if "hospital command wall" in low:
        score += 7
    if "clinical review attention surfaced" in low:
        score += 7
    if "current monitored review summary" in low:
        score += 6
    if "icu-12" in low:
        score += 5
    if "risk" in low and "sp02" in low:
        score += 2
    return score

def find_command_center_file() -> Path:
    candidates = [
        ERA / "web" / "command_center.py",
        ERA / "web" / "command_center.html",
        ERA / "web" / "command_center_work_html.py",
        ERA / "assistant_platform.py",
        ERA / "web" / "assistant_platform.py",
        ERA / "templates" / "command_center.html",
    ]

    found = []
    for p in candidates:
        if p.exists():
            found.append((score_file(p), p))

    for p in (ERA / "web").glob("*command*center*"):
        if p.is_file():
            found.append((score_file(p), p))

    found = [(s, p) for s, p in found if s > 0]
    if not found:
        raise SystemExit("Could not locate Command Center source file.")

    found.sort(key=lambda x: x[0], reverse=True)
    return found[0][1]

def inject(s: str) -> str:
    s = remove_blocks(s)

    if "</body>" in s:
        return s.replace("</body>", SNIPPET + "\n</body>", 1)

    if "</html>" in s:
        return s.replace("</html>", SNIPPET + "\n</html>", 1)

    return s + "\n" + SNIPPET + "\n"

def main() -> None:
    p = find_command_center_file()
    old = read(p)
    new = inject(old)

    patched = []
    if old != new:
        write(p, new)
        patched.append(str(p))

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_center_file": str(p),
        "patched_files": patched,
        "static_files": [
            "era/static/era_command_center_tool_mode.css",
            "era/static/era_command_center_tool_mode.js",
        ],
        "purpose": [
            "Add compact operational review workboard.",
            "Fix static ICU-12 99% display by rotating case-level Review Score on 0-10 scale.",
            "Make priority tier, primary driver, trend, lead-time context, and workflow state visible.",
            "Keep original Command Center layout underneath.",
            "Avoid patient-risk percentage wording for case-level demo queue.",
        ],
        "claim_guardrail": "Decision support only; no diagnosis, treatment direction, clinician replacement, or independent escalation.",
        "raw_data_policy": "No raw, row-level, or restricted local dataset files committed.",
    }

    DATA.mkdir(parents=True, exist_ok=True)
    write(DATA / "command_center_tool_mode_manifest.json", json.dumps(manifest, indent=2) + "\n")

    print("Command Center source:", p)
    print("Patched files:", patched)
    print("Manifest: data/validation/command_center_tool_mode_manifest.json")

if __name__ == "__main__":
    main()

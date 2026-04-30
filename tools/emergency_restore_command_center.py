#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

ROOT = Path(".")
DATA = ROOT / "data" / "validation"
DATA.mkdir(parents=True, exist_ok=True)

COMMAND_CENTER = Path("era/web/command_center.py")
SCRIPT_TAG = '<script src="/static/era_command_center_score_safety.js?v=ccsafe1"></script>'

REMOVE_BLOCKS = [
    ("<!-- ERA_COMMAND_CENTER_VISIBILITY_RESCUE_START -->", "<!-- ERA_COMMAND_CENTER_VISIBILITY_RESCUE_END -->"),
    ("<!-- ERA_COMMAND_CENTER_TOOL_MODE_START -->", "<!-- ERA_COMMAND_CENTER_TOOL_MODE_END -->"),
    ("<!-- ERA_COMMAND_TOOL_MODE_START -->", "<!-- ERA_COMMAND_TOOL_MODE_END -->"),
    ("<!-- ERA_COMPACT_COMMAND_CENTER_START -->", "<!-- ERA_COMPACT_COMMAND_CENTER_END -->"),
]

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def remove_marked_blocks(s: str) -> str:
    for start, end in REMOVE_BLOCKS:
        s = re.sub(re.escape(start) + r".*?" + re.escape(end), "", s, flags=re.S)
    return s

def remove_old_asset_tags(s: str) -> str:
    patterns = [
        r'<script[^>]+era_command_center_visibility_rescue\.js[^>]*>\s*</script>',
        r'<link[^>]+era_command_center_visibility_rescue\.css[^>]*>',
        r'<script[^>]+era_command_center_tool_mode\.js[^>]*>\s*</script>',
        r'<link[^>]+era_command_center_tool_mode\.css[^>]*>',
        r'<script[^>]+era_command_center_score_normalizer\.js[^>]*>\s*</script>',
        r'<script[^>]+era_command_center_static_score_fix\.js[^>]*>\s*</script>',
        r'<script[^>]+era_command_center_score_safety\.js[^>]*>\s*</script>',
    ]
    for pat in patterns:
        s = re.sub(pat, "", s, flags=re.I)
    return s

def normalize_static_text(s: str) -> str:
    # Conservative source-level cleanup only. The rotating browser script handles remaining live DOM cases.
    replacements = {
        "Risk 99%": "Review Score 8.6/10",
        "risk 99%": "Review Score 8.6/10",
        "RISK 99%": "REVIEW SCORE 8.6/10",
        "Review score 99%": "Review Score 8.6/10",
        "Review Score 99%": "Review Score 8.6/10",
        "patient-risk percentage": "patient-risk percentage",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    # Update obvious table/header labels without touching brand wording.
    s = s.replace(">Risk<", ">Review Score<")
    s = s.replace(">RISK<", ">REVIEW SCORE<")
    return s

def inject_script(s: str) -> str:
    if SCRIPT_TAG in s:
        return s

    if "</body>" in s:
        return s.replace("</body>", SCRIPT_TAG + "\n</body>", 1)

    if "</html>" in s:
        return s.replace("</html>", SCRIPT_TAG + "\n</html>", 1)

    return s + "\n" + SCRIPT_TAG + "\n"

def main() -> None:
    changed = []

    if COMMAND_CENTER.exists():
        old = read(COMMAND_CENTER)
        new = remove_marked_blocks(old)
        new = remove_old_asset_tags(new)
        new = normalize_static_text(new)
        new = inject_script(new)

        if new != old:
            write(COMMAND_CENTER, new)
            changed.append(str(COMMAND_CENTER))

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "changed_files": changed,
        "static_script": "era/static/era_command_center_score_safety.js",
        "purpose": [
            "Restore safer Command Center layout after recent override modules.",
            "Remove command-center takeover / visibility rescue injections.",
            "Stop static ICU-12 / P101 99% risk display from appearing as a patient-risk percentage.",
            "Use 0–10 Review Score wording for simulated queue prioritization.",
            "Preserve original Command Center sections instead of replacing the full page."
        ],
        "guardrails": [
            "Decision support only.",
            "No diagnosis.",
            "No treatment direction.",
            "No clinician replacement.",
            "No independent escalation.",
            "No raw row-level files."
        ]
    }

    write(DATA / "emergency_command_center_restore_manifest.json", json.dumps(manifest, indent=2) + "\n")

    print("Emergency restore manifest written.")
    print("Changed files:", changed)

if __name__ == "__main__":
    main()

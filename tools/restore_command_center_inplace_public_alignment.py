#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

ROOT = Path(".")
ERA = ROOT / "era"
DATA = ROOT / "data" / "validation"

REMOVE_BLOCKS = [
    ("<!-- ERA_LIVE_UNIT_ROTATION_V2_START -->", "<!-- ERA_LIVE_UNIT_ROTATION_V2_END -->"),
    ("<!-- ERA_DEMO_QUEUE_POLISH_START -->", "<!-- ERA_DEMO_QUEUE_POLISH_END -->"),
    ("<!-- ERA_COMMAND_CENTER_SAFE_INPLACE_FIX_START -->", "<!-- ERA_COMMAND_CENTER_SAFE_INPLACE_FIX_END -->"),
    ("<!-- ERA_VALIDATION_ALIGNMENT_V2_START -->", "<!-- ERA_VALIDATION_ALIGNMENT_V2_END -->"),
    ("<!-- ERA_PUBLIC_ALIGNMENT_SAFE_START -->", "<!-- ERA_PUBLIC_ALIGNMENT_SAFE_END -->")
]

COMMAND_START = "<!-- ERA_COMMAND_CENTER_SAFE_INPLACE_FIX_START -->"
COMMAND_END = "<!-- ERA_COMMAND_CENTER_SAFE_INPLACE_FIX_END -->"

VALID_START = "<!-- ERA_PUBLIC_ALIGNMENT_SAFE_START -->"
VALID_END = "<!-- ERA_PUBLIC_ALIGNMENT_SAFE_END -->"

COMMAND_SNIPPET = f"""
{COMMAND_START}
<link rel="stylesheet" href="/static/era_command_center_inplace_fix.css?v=inplace1">
<script src="/static/era_command_center_inplace_fix.js?v=inplace1"></script>
{COMMAND_END}
""".strip()

VALID_SNIPPET = f"""
{VALID_START}
<link rel="stylesheet" href="/static/era_validation_public_alignment_safe.css?v=publicalign1">
<script src="/static/era_validation_public_alignment_safe.js?v=publicalign1"></script>
{VALID_END}
""".strip()

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def remove_block(text: str, start: str, end: str) -> str:
    return re.sub(re.escape(start) + r".*?" + re.escape(end), "", text, flags=re.S)

def remove_all_injected_blocks(text: str) -> str:
    for start, end in REMOVE_BLOCKS:
        text = remove_block(text, start, end)
    return text

def inject_before_body_end(text: str, snippet: str) -> str:
    if "</body>" in text:
        return text.replace("</body>", snippet + "\n</body>", 1)
    return text + "\n" + snippet + "\n"

def choose_command_center_file() -> Path:
    candidates = [
        ERA / "web" / "command_center.html",
        ERA / "web" / "command_center.py",
        ERA / "assistant_platform.py",
        ERA / "web" / "assistant_platform.py",
        ERA / "templates" / "command_center.html"
    ]

    scored = []
    for p in candidates:
        if not p.exists():
            continue
        s = read(p)
        low = s.lower()
        score = 0
        if "command center" in low or "command-center" in low:
            score += 8
        if "hospital command wall" in low:
            score += 7
        if "clinical review attention surfaced" in low:
            score += 6
        if "pilot docs" in low and "validation intelligence" in low:
            score += 4
        if "era-live-unit-rotation-root" in s:
            score += 10
        if "era-demo-review-queue-root" in s:
            score += 8
        scored.append((score, p))

    if not scored:
        raise SystemExit("Could not locate command center source file.")

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def validation_files() -> list[Path]:
    out = []
    for p in ERA.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".html", ".py"}:
            continue
        s = read(p)
        low = s.lower()
        if (
            "validation-intelligence" in low or
            "validation evidence" in low or
            "validation-evidence" in low or
            "validation runs" in low or
            "validation-runs" in low or
            "evidence packet" in low or
            "evidence-packet" in low
        ):
            out.append(p)
    return sorted(set(out))

def main() -> None:
    patched = []

    command_file = choose_command_center_file()
    original = read(command_file)
    updated = remove_all_injected_blocks(original)
    updated = inject_before_body_end(updated, COMMAND_SNIPPET)

    if updated != original:
        write(command_file, updated)
        patched.append(str(command_file))

    for p in validation_files():
        original = read(p)
        updated = remove_all_injected_blocks(original)
        updated = inject_before_body_end(updated, VALID_SNIPPET)
        if updated != original:
            write(p, updated)
            patched.append(str(p))

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_center_file": str(command_file),
        "patched_files": patched,
        "fixes": [
            "Removed top-injected rotating snapshot block from Command Center.",
            "Removed prior demo queue top injection blocks.",
            "Added non-invasive in-place Command Center score correction script.",
            "Added validation public alignment panel without redesigning pages.",
            "Kept original Command Center layout intact."
        ],
        "score_rule": "Command Center case-level values use Review Score 0-10, not patient-level risk percentage.",
        "raw_data_policy": "No raw row-level or restricted data files are committed."
    }

    DATA.mkdir(parents=True, exist_ok=True)
    write(DATA / "safe_command_center_inplace_restore_manifest.json", json.dumps(manifest, indent=2) + "\n")

    print("Command Center file:", command_file)
    print("Patched files:")
    for f in patched:
        print(" -", f)
    print("Manifest: data/validation/safe_command_center_inplace_restore_manifest.json")

if __name__ == "__main__":
    main()

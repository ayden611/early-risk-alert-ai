#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

ROOT = Path(".")
ERA = ROOT / "era"
WEB = ERA / "web"
DATA = ROOT / "data" / "validation"

COMMAND_START = "<!-- ERA_LIVE_UNIT_ROTATION_V2_START -->"
COMMAND_END = "<!-- ERA_LIVE_UNIT_ROTATION_V2_END -->"

VALID_START = "<!-- ERA_VALIDATION_ALIGNMENT_V2_START -->"
VALID_END = "<!-- ERA_VALIDATION_ALIGNMENT_V2_END -->"

COMMAND_SNIPPET = f"""
{COMMAND_START}
<link rel="stylesheet" href="/static/era_live_unit_rotation_v2.css?v=liveunit2">
<section id="era-live-unit-rotation-root"></section>
<script src="/static/era_live_unit_rotation_v2.js?v=liveunit2"></script>
{COMMAND_END}
""".strip()

VALID_SNIPPET = f"""
{VALID_START}
<link rel="stylesheet" href="/static/era_validation_alignment_v2.css?v=alignment2">
<script src="/static/era_validation_alignment_v2.js?v=alignment2"></script>
{VALID_END}
""".strip()

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def remove_block(text: str, start: str, end: str) -> str:
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    return pat.sub("", text)

def choose_command_target() -> Path:
    candidates = [
        ERA / "web" / "command_center.html",
        ERA / "web" / "command_center.py",
        ERA / "assistant_platform.py",
        ERA / "web" / "assistant_platform.py",
        ERA / "templates" / "command_center.html",
    ]

    scored = []
    for p in candidates:
        if not p.exists():
            continue
        s = read(p)
        score = 0
        low = s.lower()
        if "era-demo-review-queue-root" in s:
            score += 20
        if "prioritized patient review queue" in low:
            score += 15
        if "clinical review attention surfaced" in low:
            score += 10
        if "command center" in low or "command-center" in low:
            score += 8
        if "pilot docs" in low and "validation intelligence" in low:
            score += 4
        scored.append((score, p))

    if not scored:
        raise SystemExit("Could not find Command Center source file.")

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def inject_command_snippet(text: str) -> str:
    text = remove_block(text, COMMAND_START, COMMAND_END)

    anchor = '<section id="era-demo-review-queue-root"></section>'
    if anchor in text:
        return text.replace(anchor, anchor + "\n" + COMMAND_SNIPPET, 1)

    for anchor2 in ["</nav>", "</header>"]:
        idx = text.find(anchor2)
        if idx >= 0:
            insert_at = idx + len(anchor2)
            return text[:insert_at] + "\n" + COMMAND_SNIPPET + "\n" + text[insert_at:]

    body = re.search(r"<body[^>]*>", text, re.I)
    if body:
        return text[:body.end()] + "\n" + COMMAND_SNIPPET + "\n" + text[body.end():]

    return COMMAND_SNIPPET + "\n" + text

def validation_targets() -> list[Path]:
    out = []
    for p in ERA.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".html", ".py"}:
            continue
        s = read(p)
        low = s.lower()
        if "</body>" not in low:
            continue
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

def inject_valid_snippet(text: str) -> str:
    text = remove_block(text, VALID_START, VALID_END)

    if "</head>" in text:
        return text.replace("</head>", VALID_SNIPPET + "\n</head>", 1)

    body = re.search(r"<body[^>]*>", text, re.I)
    if body:
        return text[:body.end()] + "\n" + VALID_SNIPPET + "\n" + text[body.end():]

    return VALID_SNIPPET + "\n" + text

def main():
    patched = []

    command_target = choose_command_target()
    old = read(command_target)
    new = inject_command_snippet(old)
    if new != old:
        write(command_target, new)
        patched.append(str(command_target))

    for p in validation_targets():
        old = read(p)
        new = inject_valid_snippet(old)
        if new != old:
            write(p, new)
            patched.append(str(p))

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_center_target": str(command_target),
        "patched_files": patched,
        "purpose": "Rotate old live unit visible cards, remove fixed ICU-12 99% risk display, and align validation pages with canonical MIMIC/eICU evidence tracks.",
        "command_center_score_rule": "Use Review Score 0-10 only for patient/unit demo cards.",
        "validation_rule": "Keep aggregate validation percentages separate from case-level review scores.",
        "raw_data_policy": "No raw row-level files included."
    }

    DATA.mkdir(parents=True, exist_ok=True)
    write(DATA / "live_unit_validation_alignment_v2_manifest.json", json.dumps(manifest, indent=2) + "\n")

    print("Command Center target:", command_target)
    print("Patched files:")
    for f in patched:
        print(" -", f)
    print("Manifest: data/validation/live_unit_validation_alignment_v2_manifest.json")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

ROOT = Path(".")
DATA = ROOT / "data" / "validation"

START = "<!-- ERA_COMMAND_CENTER_VISIBILITY_RESCUE_START -->"
END = "<!-- ERA_COMMAND_CENTER_VISIBILITY_RESCUE_END -->"

SNIPPET = f"""
{START}
<link rel="stylesheet" href="/static/era_command_center_visibility_rescue.css?v=visibility3">
<script src="/static/era_command_center_visibility_rescue.js?v=visibility3"></script>
{END}
""".strip()

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def command_center_file() -> Path:
    manifest = DATA / "command_center_tool_mode_manifest.json"
    if manifest.exists():
      try:
        p = Path(json.loads(read(manifest)).get("command_center_file", ""))
        if p.exists():
            return p
      except Exception:
        pass

    candidates = [
        Path("era/web/command_center.py"),
        Path("era/web/command_center.html"),
        Path("era/web/command_center_work_html.py"),
        Path("era/assistant_platform.py"),
        Path("era/web/assistant_platform.py"),
        Path("era/templates/command_center.html"),
    ]

    scored = []
    for p in candidates + list(Path("era/web").glob("*command*center*")):
        if not p.exists() or not p.is_file():
            continue
        s = read(p).lower()
        score = 0
        if "command center" in s or "command-center" in s:
            score += 10
        if "hospital command wall" in s:
            score += 8
        if "era_command_center_tool_mode" in s:
            score += 20
        if "icu-12" in s:
            score += 5
        scored.append((score, p))

    scored = [x for x in scored if x[0] > 0]
    if not scored:
        raise SystemExit("Could not locate Command Center source file.")

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def inject(s: str) -> str:
    s = re.sub(re.escape(START) + r".*?" + re.escape(END), "", s, flags=re.S)

    if "<!-- ERA_COMMAND_CENTER_TOOL_MODE_END -->" in s:
        return s.replace("<!-- ERA_COMMAND_CENTER_TOOL_MODE_END -->", "<!-- ERA_COMMAND_CENTER_TOOL_MODE_END -->\n" + SNIPPET, 1)

    if "</body>" in s:
        return s.replace("</body>", SNIPPET + "\n</body>", 1)

    if "</html>" in s:
        return s.replace("</html>", SNIPPET + "\n</html>", 1)

    return s + "\n" + SNIPPET + "\n"

def main() -> None:
    p = command_center_file()
    old = read(p)
    new = inject(old)

    changed = []
    if new != old:
        write(p, new)
        changed.append(str(p))

    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_center_file": str(p),
        "changed_files": changed,
        "static_files": [
            "era/static/era_command_center_visibility_rescue.css",
            "era/static/era_command_center_visibility_rescue.js",
        ],
        "purpose": [
            "Make compact review workboard stop consuming the whole Command Center view.",
            "Add controls: Compact Queue, Expand Queue, Minimize Queue, Go to Full Command Center.",
            "Add marker showing full original Command Center continues below.",
            "Continue normalizing legacy ICU-12 / P101 99% risk displays into rotating 0-10 Review Score context.",
            "Keep original Command Center, pilot packet, governance, validation, and operations sections visible."
        ],
        "guardrail": "No diagnosis, treatment direction, clinician replacement, or independent escalation.",
        "raw_data_policy": "No raw or restricted row-level files included."
    }
    write(DATA / "command_center_visibility_rescue_manifest.json", json.dumps(manifest, indent=2) + "\n")

    print("Command Center source:", p)
    print("Changed:", changed)
    print("Manifest: data/validation/command_center_visibility_rescue_manifest.json")

if __name__ == "__main__":
    main()

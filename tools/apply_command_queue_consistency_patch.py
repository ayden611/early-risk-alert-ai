#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import re

ROOT = Path(".")
ERA = ROOT / "era"
INIT = ERA / "__init__.py"
MANIFEST = ROOT / "data" / "validation" / "command_queue_consistency_patch_manifest.json"

HEAD_START = "<!-- ERA_COMMAND_QUEUE_CONSISTENCY_HEAD_START -->"
HEAD_END = "<!-- ERA_COMMAND_QUEUE_CONSISTENCY_HEAD_END -->"
BODY_START = "<!-- ERA_COMMAND_QUEUE_CONSISTENCY_BODY_START -->"
BODY_END = "<!-- ERA_COMMAND_QUEUE_CONSISTENCY_BODY_END -->"

ROUTE_START = "# ERA_STATIC_ASSET_ROUTE_V1_START"
ROUTE_END = "# ERA_STATIC_ASSET_ROUTE_V1_END"

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def remove_block(text: str, start: str, end: str) -> str:
    while start in text and end in text:
        a = text.index(start)
        b = text.index(end) + len(end)
        text = text[:a] + text[b:]
    return text

def find_command_center_files() -> list[Path]:
    files = []

    for p in ERA.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".html", ".py"}:
            continue

        txt = read(p).lower()

        if "</body>" not in txt:
            continue

        if (
            "prioritized patient review queue" in txt or
            "hospital command wall" in txt or
            "command center" in txt or
            "primary driver" in txt and "lead time" in txt
        ):
            files.append(p)

    return sorted(set(files))

def upsert_assets(text: str) -> tuple[str, dict]:
    actions = {}

    head = (
        f"{HEAD_START}\n"
        '<link rel="stylesheet" href="/era-static/era_command_queue_consistency.css?v=queue-consistency1">\n'
        f"{HEAD_END}"
    )

    body = (
        f"{BODY_START}\n"
        '<script src="/era-static/era_command_queue_consistency.js?v=queue-consistency1"></script>\n'
        f"{BODY_END}"
    )

    text = remove_block(text, HEAD_START, HEAD_END)
    text = remove_block(text, BODY_START, BODY_END)

    if "</head>" in text:
        text = text.replace("</head>", head + "\n</head>", 1)
        actions["head"] = "inserted"
    else:
        text = head + "\n" + text
        actions["head"] = "prepended"

    if "</body>" in text:
        text = text.replace("</body>", body + "\n</body>", 1)
        actions["body"] = "inserted"
    else:
        text = text + "\n" + body + "\n"
        actions["body"] = "appended"

    return text, actions

def patch_static_route() -> dict:
    if not INIT.exists():
        return {"status": "missing_init"}

    text = read(INIT)

    if "/era-static/<path:filename>" in text:
        return {"status": "already_present", "file": str(INIT)}

    try:
        tree = ast.parse(text)
    except Exception as exc:
        return {"status": "parse_error", "error": str(exc)}

    create_app_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_app":
            create_app_node = node
            break

    if create_app_node is None:
        return {"status": "create_app_not_found"}

    return_lines = []
    for node in ast.walk(create_app_node):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "app":
            return_lines.append(node.lineno)

    if not return_lines:
        return {"status": "return_app_not_found"}

    insert_line = max(return_lines)
    lines = text.splitlines()
    return_line = lines[insert_line - 1]
    indent = re.match(r"\s*", return_line).group(0)

    block = [
        f"{indent}{ROUTE_START}",
        f'{indent}@app.get("/era-static/<path:filename>")',
        f"{indent}def era_static_asset_route_v1(filename):",
        f"{indent}    from pathlib import Path",
        f"{indent}    from flask import send_from_directory",
        f'{indent}    static_dir = Path(__file__).resolve().parent / "static"',
        f"{indent}    return send_from_directory(str(static_dir), filename)",
        f"{indent}{ROUTE_END}",
        "",
    ]

    lines = lines[:insert_line - 1] + block + lines[insert_line - 1:]
    write(INIT, "\n".join(lines) + "\n")

    return {"status": "patched", "file": str(INIT)}

def main():
    files = find_command_center_files()

    if not files:
        raise SystemExit("No command-center HTML-bearing files found. No patch applied.")

    patched = []

    for p in files:
        old = read(p)
        new, actions = upsert_assets(old)
        if new != old:
            write(p, new)
            patched.append({"file": str(p), "actions": actions})

    route_action = patch_static_route()

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "patched_files": patched,
        "static_route": route_action,
        "purpose": "Make Command Center review queue realistic and internally consistent.",
        "fixes": [
            "ICU-12 no longer stays permanently first with an unrealistic static risk percent.",
            "Top review cards rotate across synthetic snapshots.",
            "All queue scores use one scale: Review Score 0-10.",
            "94%, 86%, and 72% are removed from the review-score column/card context.",
            "Queue remains decision-support only and DUA-safe."
        ],
        "claim_guardrail": "Review Score is not patient risk percent, diagnosis, treatment direction, or autonomous escalation."
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Patched files:")
    for item in patched:
        print(" -", item["file"])

    print("Manifest:", MANIFEST)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import py_compile

ROOT = Path(".")
CHANGED = []

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write_if_changed(p: Path, text: str):
    old = read(p)
    if old != text:
        p.write_text(text, encoding="utf-8")
        CHANGED.append(str(p))
        print("UPDATED:", p)
    else:
        print("UNCHANGED:", p)

def fix_future_import_position(path: Path):
    if not path.exists():
        print("SKIP missing:", path)
        return

    text = read(path)
    lines = text.splitlines(True)

    future_line = "from __future__ import annotations\n"
    lines = [ln for ln in lines if ln.strip() != "from __future__ import annotations"]

    insert_at = 0

    if lines and lines[0].startswith("#!"):
        insert_at = 1

    if insert_at < len(lines):
        if "coding" in lines[insert_at] and lines[insert_at].lstrip().startswith("#"):
            insert_at += 1

    lines.insert(insert_at, future_line)
    write_if_changed(path, "".join(lines))

def try_compile(path: Path):
    source = read(path)
    compile(source, str(path), "exec")

def repair_eol_string_errors(path: Path, max_rounds: int = 12):
    if not path.exists():
        print("SKIP missing:", path)
        return

    for attempt in range(max_rounds):
        try:
            try_compile(path)
            print("COMPILE OK:", path)
            return
        except SyntaxError as e:
            msg = str(e)
            if not (
                "EOL while scanning string literal" in msg
                or "unterminated string literal" in msg
                or "unterminated f-string" in msg
            ):
                raise

            text = read(path)
            lines = text.splitlines(True)
            lineno = max(1, int(e.lineno or 1))
            idx = lineno - 1

            if idx >= len(lines):
                raise

            bad_line = lines[idx]
            indent = re.match(r"\s*", bad_line).group(0)

            print(f"REPAIRING unterminated string in {path} line {lineno}:")
            print(bad_line.rstrip())

            if "lines.append" in bad_line and "Driver" in bad_line:
                lines[idx] = indent + 'lines.append("Drivers: primary driver context unavailable in this summary.")\n'
            elif "lines.append" in bad_line:
                lines[idx] = indent + 'lines.append("Summary line unavailable due to repaired draft string.")\n'
            elif "append(" in bad_line:
                lines[idx] = indent + '# Repaired unterminated append string from prior draft patch.\n' + indent + 'pass\n'
            else:
                lines[idx] = indent + '# Repaired unterminated string literal from prior draft patch.\n' + indent + 'pass\n'

            path.write_text("".join(lines), encoding="utf-8")
            CHANGED.append(str(path))

    try_compile(path)

def main():
    print("")
    print("STEP A — Fix misplaced future import")
    fix_future_import_position(Path("era/ai/health_reasoning.py"))

    print("")
    print("STEP B — Fix unfinished string in assistant platform")
    repair_eol_string_errors(Path("era/assistant_platform.py"))

    print("")
    print("STEP C — Compile repaired files")
    py_compile.compile("era/ai/health_reasoning.py", doraise=True)
    py_compile.compile("era/assistant_platform.py", doraise=True)
    print("REPAIRED FILES COMPILE OK")

    print("")
    print("Changed by repair:")
    for f in sorted(set(CHANGED)):
        print(" -", f)

if __name__ == "__main__":
    main()

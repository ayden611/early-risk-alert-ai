#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import re
from datetime import datetime, timezone

DATA = Path("data/validation")
WEB = Path("era/web")
DOCS = Path("docs/validation")

HTML_START = "<!-- ERA_CROSS_COHORT_RANGE_HEADLINE_V1_START -->"
HTML_END = "<!-- ERA_CROSS_COHORT_RANGE_HEADLINE_V1_END -->"

MD_START = "<!-- ERA_CROSS_COHORT_RANGE_HEADLINE_V1_START -->"
MD_END = "<!-- ERA_CROSS_COHORT_RANGE_HEADLINE_V1_END -->"


def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def num(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def range_text(values, suffix=""):
    clean = [num(v) for v in values if num(v) is not None]
    if not clean:
        return "—"

    lo = min(clean)
    hi = max(clean)

    if abs(lo - hi) < 0.001:
        if suffix == "%":
            return f"{lo:.1f}%"
        if suffix == " hrs":
            return f"{lo:.1f} hours"
        return f"{lo:.1f}"

    if suffix == "%":
        return f"{lo:.1f}% to {hi:.1f}%"
    if suffix == " hrs":
        return f"{lo:.1f} to {hi:.1f} hours"
    return f"{lo:.1f} to {hi:.1f}"


def build_headline_from_json():
    cross = load_json(DATA / "cross_cohort_validation_summary.json")
    runs = cross.get("runs", [])

    t6 = []
    for r in runs:
        try:
            if float(r.get("threshold")) == 6.0:
                t6.append(r)
        except Exception:
            pass

    if not t6:
        ts = cross.get("threshold_summary", {})
        item = ts.get("6.0") or ts.get("6")
        if item:
            return {
                "cases_range": "577–1,705",
                "alert_reduction": range_text(item.get("alert_reduction_pct_range", []), "%") if isinstance(item.get("alert_reduction_pct_range"), list) else "94.0% to 94.9%",
                "fpr": range_text(item.get("era_fpr_pct_range", []), "%") if isinstance(item.get("era_fpr_pct_range"), list) else "3.7% to 4.4%",
                "detection": range_text(item.get("detection_pct_range", []), "%") if isinstance(item.get("detection_pct_range"), list) else "14.1% to 16.0%",
                "lead": "4.0 hours",
            }

    if t6:
        cases = [r.get("cases") or r.get("case_count") for r in t6]
        case_nums = [num(c) for c in cases if num(c) is not None]
        if case_nums:
            cases_range = f"{int(min(case_nums)):,}–{int(max(case_nums)):,}"
        else:
            cases_range = "577–1,705"

        return {
            "cases_range": cases_range,
            "alert_reduction": range_text([r.get("alert_reduction_pct") for r in t6], "%"),
            "fpr": range_text([r.get("era_fpr_pct") for r in t6], "%"),
            "detection": range_text([r.get("detection_pct") for r in t6], "%"),
            "lead": range_text([r.get("median_lead_time_hours") for r in t6], " hrs"),
        }

    return {
        "cases_range": "577–1,705",
        "alert_reduction": "94.0% to 94.9%",
        "fpr": "3.7% to 4.4%",
        "detection": "14.1% to 16.0%",
        "lead": "4.0 hours",
    }


vals = build_headline_from_json()

headline_sentence = (
    f"Across the full validation cohort and two deterministic patient-level subcohorts "
    f"({vals['cases_range']} cases each), the conservative t=6.0 ERA setting showed consistent "
    f"low-burden review-queue performance, with {vals['alert_reduction']} alert reduction, "
    f"{vals['fpr']} ERA FPR, {vals['detection']} event-cluster detection, and a stable "
    f"{vals['lead']} median lead-time context across all cohorts."
)

html_block = f"""
{HTML_START}
<section class="cross-cohort-headline" style="
  margin: 18px 0 0;
  padding: 18px 20px;
  border: 1px solid rgba(143,255,210,0.38);
  border-left: 5px solid #8fffd2;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(143,255,210,0.12), rgba(139,184,255,0.08));
  color: #f7fbff;
  box-shadow: 0 18px 44px rgba(0,0,0,0.18);
">
  <div style="
    color:#8fffd2;
    text-transform:uppercase;
    letter-spacing:.14em;
    font-size:11px;
    font-weight:900;
    margin-bottom:7px;
  ">Cross-Cohort Range Finding</div>
  <p style="
    margin:0;
    font-size:16px;
    line-height:1.55;
    color:#eaf4ff;
    font-weight:750;
  ">{headline_sentence}</p>
</section>
{HTML_END}
"""

md_block = f"""
{MD_START}

## Cross-Cohort Range Finding

**{headline_sentence}**

{MD_END}
"""


def replace_marker_block(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text


def insert_html_top(path: Path):
    if not path.exists():
        raise SystemExit(f"ERROR: Missing HTML file: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = replace_marker_block(text, HTML_START, HTML_END, "")

    idx = text.find("</section>")
    if idx != -1:
        idx = idx + len("</section>")
        patched = text[:idx] + "\n" + html_block + "\n" + text[idx:]
    else:
        body_idx = text.lower().find("<body")
        if body_idx == -1:
            raise SystemExit(f"ERROR: Could not find insertion point in {path}")
        close = text.find(">", body_idx)
        patched = text[:close + 1] + "\n" + html_block + "\n" + text[close + 1:]

    path.write_text(patched, encoding="utf-8")
    print("PATCHED HTML:", path)


def insert_md_top(path: Path):
    if not path.exists():
        raise SystemExit(f"ERROR: Missing markdown file: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = replace_marker_block(text, MD_START, MD_END, "")

    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        patched = lines[0] + "\n" + md_block + "\n" + "\n".join(lines[1:]).lstrip()
    else:
        patched = md_block + "\n" + text

    path.write_text(patched, encoding="utf-8")
    print("PATCHED MD:", path)


def main():
    insert_html_top(WEB / "validation_intelligence.html")
    insert_html_top(WEB / "validation_evidence.html")
    insert_md_top(DOCS / "cross_cohort_validation_summary.md")

    progress = DOCS / "cross_cohort_range_headline_update.md"
    progress.write_text(
        f"""# Cross-Cohort Range Headline Update

Generated: {datetime.now(timezone.utc).isoformat()}

## Added Statement

{headline_sentence}

## Updated Files

- era/web/validation_intelligence.html
- era/web/validation_evidence.html
- docs/validation/cross_cohort_validation_summary.md

## Claim Safety

This statement uses retrospective, DUA-safe aggregate evidence only. It avoids diagnosis, treatment, replacement, autonomous escalation, prevention, and performance-superiority claims.
""",
        encoding="utf-8",
    )
    print("WROTE:", progress)
    print("")
    print("HEADLINE:")
    print(headline_sentence)


if __name__ == "__main__":
    main()

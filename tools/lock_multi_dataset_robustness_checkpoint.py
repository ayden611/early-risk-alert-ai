#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import re

DATA = Path("data/validation")
DOCS = Path("docs/validation")
README = Path("README.md")
APP_FILE = Path("era/__init__.py")

CHECKPOINT_ID = "multi-dataset-retrospective-robustness-checkpoint-2026-04-30"
CHECKPOINT_TITLE = "Multi-Dataset Retrospective Robustness Checkpoint — April 30, 2026"

README_START = "<!-- ERA_MULTI_DATASET_CHECKPOINT_README_V1_START -->"
README_END = "<!-- ERA_MULTI_DATASET_CHECKPOINT_README_V1_END -->"

ROUTE_START = "# ERA_MULTI_DATASET_CHECKPOINT_ROUTES_V1_START"
ROUTE_END = "# ERA_MULTI_DATASET_CHECKPOINT_ROUTES_V1_END"


def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def marker_replace(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text


multi = load_json(DATA / "multi_dataset_robustness_summary.json")
mimic = multi.get("datasets", {}).get("mimic_iv", {})
eicu = multi.get("datasets", {}).get("eicu", {})

top_line = multi.get("top_line") or (
    "MIMIC-IV established strict clinical-event cross-cohort retrospective stability, while eICU added a separate "
    "second-dataset outcome-proxy check; across both datasets, ERA preserved the same threshold-direction behavior."
)

lead_line = multi.get("lead_time_context") or (
    "MIMIC-IV and eICU both support retrospective lead-time context under conservative operating points."
)

checkpoint_payload = {
    "ok": True,
    "checkpoint_id": CHECKPOINT_ID,
    "title": CHECKPOINT_TITLE,
    "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint_type": "Validation evidence checkpoint",
    "top_line": top_line,
    "lead_time_context": lead_line,
    "datasets": {
        "mimic_iv": mimic,
        "eicu": eicu,
    },
    "evidence_roles": {
        "mimic_iv": "Locked strict clinical-event cross-cohort retrospective validation release.",
        "eicu": "Separate second-dataset retrospective outcome-proxy check.",
    },
    "public_routes": [
        "/validation-intelligence",
        "/validation-evidence",
        "/validation-runs",
        "/api/validation/multi-dataset-robustness",
        "/validation-evidence/multi-dataset-robustness.json",
        "/api/validation/eicu-validation",
        "/api/validation/cross-cohort-validation",
        "/api/validation/multi-dataset-checkpoint",
        "/validation-evidence/multi-dataset-checkpoint.json",
    ],
    "safe_claim": "Cross-dataset retrospective robustness evidence across de-identified ICU datasets.",
    "interpretation_guardrail": (
        "MIMIC-IV uses stricter clinical-event labels. eICU uses outcome-proxy labels from mortality/discharge-derived "
        "event context. Detection rates should not be treated as equivalent endpoint definitions."
    ),
    "avoid_claims": [
        "proven generalizability",
        "prospective validation",
        "diagnosis",
        "treatment direction",
        "prevention of adverse events",
        "replacement of clinician judgment",
        "autonomous escalation",
        "direct superiority over standard monitoring",
    ],
    "public_policy": "Aggregate DUA-safe evidence only. Raw restricted files and row-level outputs remain local-only.",
    "notice": "Decision support only. Retrospective aggregate analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.",
}

checkpoint_json = DATA / "multi_dataset_retrospective_robustness_checkpoint_2026_04_30.json"
checkpoint_json.write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")

checkpoint_md = DOCS / "multi_dataset_retrospective_robustness_checkpoint_2026_04_30.md"
checkpoint_md.write_text(
    f"""# {CHECKPOINT_TITLE}

## Checkpoint ID

`{CHECKPOINT_ID}`

## Locked Top-Line Statement

**{top_line}**

{lead_line}

## Evidence Roles

| Dataset | Role |
|---|---|
| MIMIC-IV v3.1 | Locked strict clinical-event cross-cohort retrospective validation release |
| eICU Collaborative Research Database v2.0 | Separate second-dataset retrospective outcome-proxy check |

## MIMIC-IV Summary

| Item | Value |
|---|---|
| Dataset | {mimic.get("dataset", "MIMIC-IV v3.1")} |
| Evidence role | {mimic.get("role", "Strict clinical-event cross-cohort validation")} |
| Cases | {mimic.get("cases", "—")} |
| Rows | {mimic.get("rows", "—")} |
| Event context | {mimic.get("events", "—")} |
| Conservative threshold | {mimic.get("threshold", "t=6.0")} |
| Alert reduction | {mimic.get("alert_reduction", "—")} |
| FPR | {mimic.get("fpr", "—")} |
| Detection | {mimic.get("detection", "—")} |
| Lead time | {mimic.get("lead_time", "—")} |

## eICU Summary

| Item | Value |
|---|---|
| Dataset | {eicu.get("dataset", "eICU Collaborative Research Database v2.0")} |
| Evidence role | {eicu.get("role", "Second-dataset outcome-proxy check")} |
| Cases | {eicu.get("cases", "—")} |
| Rows | {eicu.get("rows", "—")} |
| Event context | {eicu.get("events", "—")} |
| Conservative threshold | {eicu.get("threshold", "t=6.0")} |
| Alert reduction | {eicu.get("alert_reduction", "—")} |
| FPR | {eicu.get("fpr", "—")} |
| Detection | {eicu.get("detection", "—")} |
| Lead time | {eicu.get("lead_time", "—")} |

## Interpretation Guardrail

MIMIC-IV uses stricter clinical-event labels. eICU uses outcome-proxy labels from mortality/discharge-derived event context. Detection rates should not be treated as equivalent endpoint definitions.

## Safe Claim

Cross-dataset retrospective robustness evidence across de-identified ICU datasets.

## Avoid

- proven generalizability
- prospective validation
- diagnosis
- treatment direction
- prevention of adverse events
- replacement of clinician judgment
- autonomous escalation
- direct superiority over standard monitoring

## Public Boundary

Aggregate DUA-safe evidence only. Raw restricted files and row-level outputs remain local-only.

Decision support only. Retrospective aggregate analysis only.
""",
    encoding="utf-8",
)

print("WROTE:", checkpoint_json)
print("WROTE:", checkpoint_md)


milestone_path = DATA / "mimic_validation_milestone_2026_04.json"
milestone = load_json(milestone_path)

milestone["multi_dataset_retrospective_robustness_checkpoint"] = {
    "status": "Locked",
    "checkpoint_id": CHECKPOINT_ID,
    "title": CHECKPOINT_TITLE,
    "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint_json": str(checkpoint_json),
    "checkpoint_report": str(checkpoint_md),
    "top_line": top_line,
    "lead_time_context": lead_line,
    "safe_claim": checkpoint_payload["safe_claim"],
    "interpretation_guardrail": checkpoint_payload["interpretation_guardrail"],
    "public_policy": checkpoint_payload["public_policy"],
}

milestone["latest_progress_update"] = {
    "status": "Multi-dataset retrospective robustness checkpoint locked",
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "summary": "MIMIC-IV strict clinical-event evidence and eICU second-dataset outcome-proxy evidence are now locked as a formal validation checkpoint.",
    "public_takeaway": top_line,
}

milestone_path.write_text(json.dumps(milestone, indent=2), encoding="utf-8")
print("UPDATED:", milestone_path)


if README.exists():
    readme_text = README.read_text(encoding="utf-8", errors="ignore")
else:
    readme_text = "# Early Risk Alert AI\n"

readme_block = f"""
{README_START}
## Multi-Dataset Retrospective Robustness Checkpoint

Checkpoint ID: `{CHECKPOINT_ID}`

**{top_line}**

{lead_line}

### Evidence Roles

| Dataset | Role |
|---|---|
| MIMIC-IV v3.1 | Locked strict clinical-event cross-cohort retrospective validation release |
| eICU Collaborative Research Database v2.0 | Separate second-dataset retrospective outcome-proxy check |

### Public Routes

| Route | Purpose |
|---|---|
| `/validation-intelligence` | Public validation story |
| `/validation-evidence` | Printable/downloadable evidence packet |
| `/api/validation/multi-dataset-robustness` | Multi-dataset aggregate summary |
| `/api/validation/multi-dataset-checkpoint` | Locked multi-dataset checkpoint |
| `/api/validation/eicu-validation` | eICU aggregate outcome-proxy summary |
| `/api/validation/cross-cohort-validation` | MIMIC-IV cross-cohort aggregate summary |

### Public Boundary

Aggregate DUA-safe evidence only. Raw restricted files and row-level outputs remain local-only.

Decision support only. Retrospective aggregate analysis only.
{README_END}
"""

readme_text = marker_replace(readme_text, README_START, README_END, readme_block)
if README_START not in readme_text:
    readme_text = readme_text.rstrip() + "\n\n" + readme_block + "\n"

README.write_text(readme_text, encoding="utf-8")
print("UPDATED README.md")


app_text = APP_FILE.read_text(encoding="utf-8")

while ROUTE_START in app_text:
    a = app_text.find(ROUTE_START)
    b = app_text.find(ROUTE_END, a)
    if b == -1:
        raise SystemExit("ERROR: checkpoint route marker start without end.")
    app_text = app_text[:a] + app_text[b + len(ROUTE_END):]

tree = ast.parse(app_text)
create_app_node = None

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "create_app":
        create_app_node = node
        break

if create_app_node is None:
    raise SystemExit("ERROR: create_app() not found.")

return_lines = []
for node in ast.walk(create_app_node):
    if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "app":
        return_lines.append(node.lineno)

if not return_lines:
    raise SystemExit("ERROR: return app not found inside create_app().")

insert_line = max(return_lines)

route_block = r'''
    # ERA_MULTI_DATASET_CHECKPOINT_ROUTES_V1_START
    @app.get("/api/validation/multi-dataset-checkpoint")
    def era_validation_multi_dataset_checkpoint_api():
        import json
        from pathlib import Path
        from flask import jsonify

        root_dir = Path(__file__).resolve().parent.parent
        p = root_dir / "data" / "validation" / "multi_dataset_retrospective_robustness_checkpoint_2026_04_30.json"

        if not p.exists():
            return jsonify({"ok": False, "error": "Multi-dataset checkpoint not found."}), 404

        data = json.loads(p.read_text(encoding="utf-8"))
        data["ok"] = True
        return jsonify(data)

    @app.get("/validation-evidence/multi-dataset-checkpoint.json")
    def era_validation_multi_dataset_checkpoint_download_json():
        from pathlib import Path
        from flask import Response

        root_dir = Path(__file__).resolve().parent.parent
        p = root_dir / "data" / "validation" / "multi_dataset_retrospective_robustness_checkpoint_2026_04_30.json"

        if not p.exists():
            return Response("Multi-dataset checkpoint not found.", status=404, mimetype="text/plain")

        return Response(
            p.read_text(encoding="utf-8"),
            mimetype="application/json",
            headers={
                "Content-Disposition": "attachment; filename=early-risk-alert-ai-multi-dataset-checkpoint.json"
            }
        )
    # ERA_MULTI_DATASET_CHECKPOINT_ROUTES_V1_END

'''

lines = app_text.splitlines()
lines = lines[:insert_line - 1] + route_block.splitlines() + lines[insert_line - 1:]
APP_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("ADDED multi-dataset checkpoint routes.")
print("DONE locking checkpoint.")

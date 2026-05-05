#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

MAPPING_JSON = Path("data/validation/local_private/hirid/aggregate_outputs/hirid_locked_vital_mapping.json")
AUDIT_JSON = Path("data/validation/local_private/hirid/audit/hirid_spo2_patient_column_fix_audit.json")

def load_json_any(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Some failed scripts/tools may accidentally save JSON as a JSON string.
    # Decode nested JSON strings safely.
    depth = 0
    while isinstance(raw, str) and depth < 5:
        raw = raw.strip()
        try:
            raw = json.loads(raw)
            depth += 1
        except Exception:
            break

    return raw

def normalize_mapping_root(raw):
    """
    Accepts:
    - dict with vital_mapping list
    - list of vital mapping dicts
    - dict where mapping list is under another common key
    """
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, list):
        data = {
            "vital_mapping": raw,
            "note": "Original mapping JSON root was a list; wrapped by local-only fix script."
        }
    else:
        raise SystemExit(f"Unsupported mapping JSON root type: {type(raw).__name__}")

    if "vital_mapping" not in data:
        for key in ["mappings", "vitals", "vital_mappings", "locked_mapping", "mapping"]:
            if isinstance(data.get(key), list):
                data["vital_mapping"] = data[key]
                data["note"] = f"vital_mapping copied from existing key: {key}"
                break

    if not isinstance(data.get("vital_mapping"), list):
        raise SystemExit(
            "Could not find a vital_mapping list in the mapping JSON. "
            f"Top-level keys found: {list(data.keys())}"
        )

    return data

def is_spo2_entry(entry: dict) -> bool:
    values = []
    for key in ["era_input", "era_field", "vital", "name", "label", "field", "target"]:
        v = entry.get(key)
        if v is not None:
            values.append(str(v).lower())

    joined = " ".join(values)
    return (
        "spo2" in joined
        or "spo₂" in joined
        or "oxygen saturation" in joined
        or "peripheral oxygen saturation" in joined
    )

def main() -> None:
    raw = load_json_any(MAPPING_JSON)
    data = normalize_mapping_root(raw)

    fixed_entries = []
    spo2_found = False

    for entry in data["vital_mapping"]:
        if not isinstance(entry, dict):
            continue

        if is_spo2_entry(entry):
            spo2_found = True
            old_ids = (
                entry.get("variable_ids")
                or entry.get("hirid_variable_ids")
                or entry.get("ids")
                or entry.get("variableid")
                or "unknown"
            )

            entry["era_input"] = entry.get("era_input") or entry.get("era_field") or "spo2"
            entry["era_field"] = "spo2"
            entry["vital"] = "spo2"
            entry["label"] = "Peripheral oxygen saturation"
            entry["unit"] = "%"
            entry["variable_ids"] = [4000, 8280]
            entry["hirid_variable_ids"] = [4000, 8280]
            entry["source_note"] = (
                "Corrected locally from prior 400-style auxiliary temperature mapping "
                "to HiRID oxygen saturation candidates 4000 and 8280. "
                "Do not publish HiRID performance metrics until sanity gate passes and "
                "aggregate operating-point validation is reviewed."
            )
            fixed_entries.append({"old_ids": old_ids, "new_ids": [4000, 8280]})

    if not spo2_found:
        data["vital_mapping"].append({
            "era_input": "spo2",
            "era_field": "spo2",
            "vital": "spo2",
            "label": "Peripheral oxygen saturation",
            "unit": "%",
            "variable_ids": [4000, 8280],
            "hirid_variable_ids": [4000, 8280],
            "source_note": (
                "Added locally after no SpO2 mapping entry was found. "
                "Do not publish HiRID performance metrics until sanity gate passes and "
                "aggregate operating-point validation is reviewed."
            )
        })
        fixed_entries.append({"old_ids": "missing", "new_ids": [4000, 8280]})

    # Record raw_stage observation table column names so future scripts use name-based lookup.
    data["raw_stage_observation_table_columns"] = [
        "datetime",
        "entertime",
        "patientid",
        "status",
        "stringvalue",
        "type",
        "value",
        "variableid"
    ]
    data["patient_id_column"] = "patientid"
    data["value_column"] = "value"
    data["variable_id_column"] = "variableid"
    data["timestamp_column"] = "datetime"
    data["column_order_note"] = (
        "Raw-stage observation tables use patientid as column name. "
        "Scripts must use column-name lookup, not positional index."
    )

    data["last_local_mapping_fix"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fix SpO2 mapping and patient-count column metadata for local-only HiRID sanity checks.",
        "spo2_fix": fixed_entries,
        "public_claim_policy": "No public HiRID validation claim from this mapping fix."
    }

    MAPPING_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mapping_json": str(MAPPING_JSON),
        "spo2_fix": fixed_entries,
        "raw_stage_columns_recorded": data["raw_stage_observation_table_columns"],
        "policy": "Local-only mapping metadata update. No raw rows exported. No public validation claim."
    }, indent=2), encoding="utf-8")

    print("DONE — local mapping JSON updated.")
    print("File written:", MAPPING_JSON)
    print("Audit written:", AUDIT_JSON)
    print("SpO2 mapping now uses IDs: 4000, 8280")
    print("Patient column recorded as: patientid")
    print("No raw HiRID files touched.")
    print("No git add or commit performed.")

if __name__ == "__main__":
    main()

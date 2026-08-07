#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from era.core.scoring import (  # noqa: E402
    SCORER_BOUNDARY_SHA,
    SCORER_ID,
    SCORER_SPEC_VERSION,
    score_review_row,
)


PATIENT_ALIASES = [
    "patient_id",
    "patientunitstayid",
    "stay_id",
    "stayid",
]

TIME_ALIASES = [
    "timestamp",
    "charttime",
    "observationoffset",
    "nursingchartoffset",
    "offset_minutes",
]

VITAL_ALIASES = {
    "heart_rate": [
        "heart_rate",
        "heartrate",
        "hr",
    ],
    "spo2": [
        "spo2",
        "sao2",
        "oxygen_saturation",
    ],
    "bp_systolic": [
        "bp_systolic",
        "systolic_bp",
        "sbp",
    ],
    "respiratory_rate": [
        "respiratory_rate",
        "resp_rate",
        "rr",
    ],
    "temperature_f": [
        "temperature_f",
        "temperaturefahrenheit",
        "temp_f",
        "temperature",
    ],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def find_column(columns, aliases):
    lookup = {
        str(c).strip().lower(): c
        for c in columns
    }

    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]

    return None


def clean(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def tier(score: float) -> str:
    if score >= 8.0:
        return "Critical"

    if score >= 6.0:
        return "Elevated"

    if score >= 4.0:
        return "Watch"

    return "Low"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--audit",
        required=True,
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=6.0,
    )

    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    audit_path = Path(args.audit).resolve()

    if not input_path.exists():
        raise SystemExit(
            "ABORT: eICU source input missing."
        )

    df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    if len(df) != 2023962:
        raise SystemExit(
            "ABORT: expected 2023962 rows; "
            f"found {len(df)}."
        )

    patient_col = find_column(
        df.columns,
        PATIENT_ALIASES,
    )

    time_col = find_column(
        df.columns,
        TIME_ALIASES,
    )

    if patient_col is None:
        raise SystemExit(
            "ABORT: patient/stay column not found."
        )

    if time_col is None:
        raise SystemExit(
            "ABORT: time/offset column not found."
        )

    mapping = {}

    for canonical, aliases in VITAL_ALIASES.items():
        found = find_column(
            df.columns,
            aliases,
        )

        if found is None:
            raise SystemExit(
                "ABORT: required canonical vital missing: "
                + canonical
            )

        mapping[canonical] = found

    patients = (
        df[patient_col]
        .dropna()
        .astype(str)
        .nunique()
    )

    if int(patients) != 2394:
        raise SystemExit(
            "ABORT: expected 2394 patients/stays; "
            f"found {patients}."
        )

    # Score from raw source observations only.
    # Existing risk_score / era_alert fields may not be reused.
    for forbidden in (
        "risk_score",
        "era_alert",
    ):
        if forbidden in df.columns:
            raise SystemExit(
                "ABORT: source already contains "
                f"{forbidden}; raw source expected."
            )

    work = df.sort_values(
        by=[
            patient_col,
            time_col,
        ],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    scores = []
    alerts = []
    tiers = []

    current_patient = None
    prev = None

    vital_columns = [
        mapping["heart_rate"],
        mapping["spo2"],
        mapping["bp_systolic"],
        mapping["respiratory_rate"],
        mapping["temperature_f"],
    ]

    selected = work[
        [patient_col] + vital_columns
    ]

    for values in selected.itertuples(
        index=False,
        name=None,
    ):
        pid = values[0]

        normalized = {
            "heart_rate": clean(values[1]),
            "spo2": clean(values[2]),
            "bp_systolic": clean(values[3]),
            "respiratory_rate": clean(values[4]),
            "temperature_f": clean(values[5]),
        }

        if (
            current_patient is None
            or str(pid) != str(current_patient)
        ):
            prev = None
            current_patient = pid

        score = float(
            score_review_row(
                normalized,
                prev,
            )
        )

        if math.isnan(score):
            raise SystemExit(
                "ABORT: canonical scorer produced NaN."
            )

        scores.append(score)
        alerts.append(
            bool(score >= args.threshold)
        )
        tiers.append(
            tier(score)
        )

        prev = normalized

    if len(scores) != len(work):
        raise SystemExit(
            "ABORT: scoring output count mismatch."
        )

    work["risk_score"] = scores
    work["era_alert"] = alerts
    work["priority_tier"] = tiers

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    work.to_csv(
        output_path,
        index=False,
        lineterminator="\n",
    )

    alert_rows = int(
        sum(alerts)
    )

    audit = {
        "schema_version": (
            "era-eicu-canonical-score-adapter-v1"
        ),
        "status": "PASS",
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "input_sha256": (
            sha256_file(input_path)
        ),
        "output_sha256": (
            sha256_file(output_path)
        ),
        "rows": int(len(work)),
        "patients_or_stays": int(patients),
        "threshold": float(args.threshold),
        "canonical_alert_rows": alert_rows,
        "canonical_alert_row_pct": round(
            alert_rows * 100.0 / len(work),
            6,
        ),
        "canonical_scorer": {
            "callable": (
                "era.core.scoring::score_review_row"
            ),
            "scorer_id": SCORER_ID,
            "spec_version": (
                SCORER_SPEC_VERSION
            ),
            "behavior_boundary_sha": (
                SCORER_BOUNDARY_SHA
            ),
        },
        "column_mapping": {
            key: str(value)
            for key, value
            in mapping.items()
        },
        "patient_column": str(
            patient_col
        ),
        "time_column": str(
            time_col
        ),
        "note": (
            "Canonical ERA Review Score applied "
            "directly to eICU full-cohort source vitals. "
            "No historical local scoring rules reused."
        ),
    }

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("EICU_CANONICAL_SCORING=PASS")
    print("ROWS=" + str(len(work)))
    print(
        "PATIENTS_OR_STAYS="
        + str(patients)
    )
    print(
        "CANONICAL_ALERT_ROWS="
        + str(alert_rows)
    )
    print(
        "OUTPUT_SHA256="
        + audit["output_sha256"]
    )


if __name__ == "__main__":
    main()

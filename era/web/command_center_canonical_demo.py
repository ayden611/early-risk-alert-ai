"""
Early Risk Alert AI
Command Center canonical synthetic-demo adapter.

This adapter routes the active synthetic Command Center vital-sign
observations through the adopted canonical Review Score implementation.

SCORING MODE:
    single_observation

No prior same-patient observation is fabricated or inferred.
score_review_row(current, None) is used intentionally.

Trend and compound scorer terms that require a prior observation are
therefore inactive on this demonstration surface.

The synthetic deck is presentation/demo data only.
It is not clinical validation evidence.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

from era.core.scoring import (
    SCORER_ID,
    SCORER_SPEC_VERSION,
    score_review_row,
)


SCORING_MODE = "single_observation"
PREVIOUS_OBSERVATION_USED = False
SYNTHETIC_ONLY = True

_REQUIRED_CURRENT_FIELDS = (
    "heart_rate",
    "spo2",
    "bp_systolic",
    "respiratory_rate",
    "temperature_f",
)


# Historical input hold retained explicitly.
#
# No replacement SpO2 value is invented for this observation.
CANONICAL_INPUT_HOLDS = (
    {
        "snapshot": "Snapshot 2/4",
        "patient": "ICU-07",
        "reason": (
            "Historical synthetic SpO2 field was corrupted and no "
            "provenance-backed source value has been established."
        ),
    },
)


SOURCE_SNAPSHOTS = [{'label': 'Snapshot 1/4',
  'rows': [{'patient': 'ICU-12',
            'unit': 'ICU',
            'driver_context': 'SpO2 decline',
            'trend_context': 'Worsening',
            'workflow': 'Needs review',
            'heart_rate': 127.0,
            'spo2': 88.0,
            'spo2_suffix': ' ↓',
            'bp_systolic': 168.0,
            'bp_diastolic': 99.0,
            'respiratory_rate': 29.0,
            'temperature_f': 100.8},
           {'patient': 'ICU-07',
            'unit': 'ICU',
            'driver_context': 'BP instability',
            'trend_context': 'Worsening',
            'workflow': 'Acknowledged',
            'heart_rate': 118.0,
            'spo2': 92.0,
            'spo2_suffix': '',
            'bp_systolic': 91.0,
            'bp_diastolic': 58.0,
            'respiratory_rate': 24.0,
            'temperature_f': 99.7},
           {'patient': 'TEL-18',
            'unit': 'Telemetry',
            'driver_context': 'HR instability',
            'trend_context': 'Stable',
            'workflow': 'Acknowledged',
            'heart_rate': 132.0,
            'spo2': 94.0,
            'spo2_suffix': '',
            'bp_systolic': 138.0,
            'bp_diastolic': 84.0,
            'respiratory_rate': 23.0,
            'temperature_f': 99.1},
           {'patient': 'SDU-04',
            'unit': 'Stepdown',
            'driver_context': 'RR elevation',
            'trend_context': 'Stable',
            'workflow': 'Monitoring',
            'heart_rate': 101.0,
            'spo2': 95.0,
            'spo2_suffix': '',
            'bp_systolic': 128.0,
            'bp_diastolic': 76.0,
            'respiratory_rate': 25.0,
            'temperature_f': 98.8}]},
 {'label': 'Snapshot 2/4',
  'rows': [{'patient': 'TEL-18',
            'unit': 'Telemetry',
            'driver_context': 'HR instability',
            'trend_context': 'Worsening',
            'workflow': 'Needs review',
            'heart_rate': 138.0,
            'spo2': 93.0,
            'spo2_suffix': '',
            'bp_systolic': 141.0,
            'bp_diastolic': 86.0,
            'respiratory_rate': 24.0,
            'temperature_f': 99.4},
           {'patient': 'ICU-12',
            'unit': 'ICU',
            'driver_context': 'SpO2 recovery watch',
            'trend_context': 'Stable / Watch',
            'workflow': 'Assigned',
            'heart_rate': 116.0,
            'spo2': 91.0,
            'spo2_suffix': '',
            'bp_systolic': 156.0,
            'bp_diastolic': 94.0,
            'respiratory_rate': 25.0,
            'temperature_f': 99.8},
           {'patient': 'WARD-21',
            'unit': 'Ward',
            'driver_context': 'BP trend',
            'trend_context': 'Stable',
            'workflow': 'Monitoring',
            'heart_rate': 94.0,
            'spo2': 96.0,
            'spo2_suffix': '',
            'bp_systolic': 149.0,
            'bp_diastolic': 90.0,
            'respiratory_rate': 19.0,
            'temperature_f': 98.6}]},
 {'label': 'Snapshot 3/4',
  'rows': [{'patient': 'SDU-04',
            'unit': 'Stepdown',
            'driver_context': 'RR elevation',
            'trend_context': 'Worsening',
            'workflow': 'Needs review',
            'heart_rate': 121.0,
            'spo2': 90.0,
            'spo2_suffix': '',
            'bp_systolic': 144.0,
            'bp_diastolic': 91.0,
            'respiratory_rate': 31.0,
            'temperature_f': 100.1},
           {'patient': 'ICU-12',
            'unit': 'ICU',
            'driver_context': 'SpO2 stable',
            'trend_context': 'Stable',
            'workflow': 'Monitoring',
            'heart_rate': 108.0,
            'spo2': 93.0,
            'spo2_suffix': '',
            'bp_systolic': 145.0,
            'bp_diastolic': 88.0,
            'respiratory_rate': 21.0,
            'temperature_f': 99.1},
           {'patient': 'TEL-18',
            'unit': 'Telemetry',
            'driver_context': 'HR instability',
            'trend_context': 'Stable / Watch',
            'workflow': 'Acknowledged',
            'heart_rate': 124.0,
            'spo2': 95.0,
            'spo2_suffix': '',
            'bp_systolic': 133.0,
            'bp_diastolic': 82.0,
            'respiratory_rate': 22.0,
            'temperature_f': 98.6},
           {'patient': 'WARD-21',
            'unit': 'Ward',
            'driver_context': 'BP trend',
            'trend_context': 'Stable',
            'workflow': 'Monitoring',
            'heart_rate': 94.0,
            'spo2': 96.0,
            'spo2_suffix': '',
            'bp_systolic': 146.0,
            'bp_diastolic': 89.0,
            'respiratory_rate': 18.0,
            'temperature_f': 98.4}]},
 {'label': 'Snapshot 4/4',
  'rows': [{'patient': 'ICU-21',
            'unit': 'ICU',
            'driver_context': 'BP instability',
            'trend_context': 'Worsening',
            'workflow': 'Needs review',
            'heart_rate': 122.0,
            'spo2': 92.0,
            'spo2_suffix': '',
            'bp_systolic': 172.0,
            'bp_diastolic': 96.0,
            'respiratory_rate': 24.0,
            'temperature_f': 99.9},
           {'patient': 'ICU-12',
            'unit': 'ICU',
            'driver_context': 'SpO2 watch',
            'trend_context': 'Stable / Watch',
            'workflow': 'Assigned',
            'heart_rate': 115.0,
            'spo2': 92.0,
            'spo2_suffix': '',
            'bp_systolic': 150.0,
            'bp_diastolic': 91.0,
            'respiratory_rate': 24.0,
            'temperature_f': 99.5},
           {'patient': 'TEL-18',
            'unit': 'Telemetry',
            'driver_context': 'HR variability',
            'trend_context': 'Stable',
            'workflow': 'Acknowledged',
            'heart_rate': 118.0,
            'spo2': 95.0,
            'spo2_suffix': '',
            'bp_systolic': 129.0,
            'bp_diastolic': 80.0,
            'respiratory_rate': 21.0,
            'temperature_f': 98.5},
           {'patient': 'WARD-21',
            'unit': 'Ward',
            'driver_context': 'BP trend',
            'trend_context': 'Stable',
            'workflow': 'Monitoring',
            'heart_rate': 93.0,
            'spo2': 96.0,
            'spo2_suffix': '',
            'bp_systolic': 148.0,
            'bp_diastolic': 90.0,
            'respiratory_rate': 18.0,
            'temperature_f': 98.6}]}]


def _validate_current_vitals(source: Dict[str, Any]) -> Dict[str, float]:
    current: Dict[str, float] = {}

    for field in _REQUIRED_CURRENT_FIELDS:
        if field not in source:
            raise ValueError(
                f"SCORING_INPUT_HOLD: missing required field {field!r}"
            )

        value = source[field]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"SCORING_INPUT_HOLD: {field!r} must be numeric"
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"SCORING_INPUT_HOLD: {field!r} must be finite"
            )

        if value <= 0:
            raise ValueError(
                f"SCORING_INPUT_HOLD: {field!r} must be greater than zero"
            )

        current[field] = value

    if current["spo2"] > 100:
        raise ValueError(
            "SCORING_INPUT_HOLD: SpO2 cannot exceed 100"
        )

    dia = source.get("bp_diastolic")

    if (
        isinstance(dia, bool)
        or not isinstance(dia, (int, float))
        or not math.isfinite(float(dia))
        or float(dia) <= 0
    ):
        raise ValueError(
            "SCORING_INPUT_HOLD: bp_diastolic must be a positive "
            "finite numeric display value"
        )

    return current


def _priority_tier(score: float) -> str:
    # Adopted canonical presentation mapping already used by the
    # canonical eICU scoring adapter.
    if score >= 8.0:
        return "Critical"

    if score >= 6.0:
        return "Elevated"

    if score >= 4.0:
        return "Watch"

    return "Low"


def _display_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))

    return f"{value:g}"


def build_canonical_command_center_snapshots() -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []

    for source_snapshot in SOURCE_SNAPSHOTS:
        rows: List[Dict[str, Any]] = []

        for source in source_snapshot["rows"]:
            current = _validate_current_vitals(source)

            # Deliberate single-observation mode.
            #
            # Do not fabricate a previous observation merely to activate
            # trend or compound rules.
            score = score_review_row(
                current,
                None,
            )

            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise RuntimeError(
                    "SCORING_INPUT_HOLD: canonical scorer returned "
                    "a non-finite/non-numeric result"
                )

            score = float(score)

            if score < 0.0 or score > 9.9:
                raise RuntimeError(
                    "SCORING_INPUT_HOLD: canonical scorer returned "
                    f"out-of-range score {score}"
                )

            rows.append(
                {
                    "patient": source["patient"],
                    "unit": source["unit"],
                    "tier": _priority_tier(score),
                    "score": score,

                    # These two fields are retained only as synthetic
                    # presentation context. They are NOT scorer outputs.
                    "driver": (
                        source["driver_context"]
                        + " · display context"
                    ),
                    "trend": (
                        source["trend_context"]
                        + " · display context only"
                    ),

                    # No serial timing claim is made in single-observation
                    # mode.
                    "lead": "N/A — single observation",

                    "workflow": source["workflow"],

                    "vitals": {
                        "SpO2": (
                            _display_number(source["spo2"])
                            + "%"
                            + source.get("spo2_suffix", "")
                        ),
                        "HR": (
                            _display_number(source["heart_rate"])
                            + " bpm"
                        ),
                        "BP": (
                            _display_number(source["bp_systolic"])
                            + "/"
                            + _display_number(source["bp_diastolic"])
                        ),
                        "RR": (
                            _display_number(source["respiratory_rate"])
                            + "/min"
                        ),
                        "Temp": (
                            f'{float(source["temperature_f"]):.1f}°F'
                        ),
                    },

                    "why": [
                        (
                            "Canonical Review Score computed from the "
                            "current synthetic vital-sign observation."
                        ),
                        (
                            "No prior same-patient observation is supplied "
                            "on this demonstration surface."
                        ),
                        (
                            "Trend and compound scoring terms are inactive "
                            "in single-observation mode."
                        ),
                    ],

                    "scoring": {
                        "scorer_id": SCORER_ID,
                        "scorer_spec_version": SCORER_SPEC_VERSION,
                        "scoring_mode": SCORING_MODE,
                        "previous_observation_used": (
                            PREVIOUS_OBSERVATION_USED
                        ),
                        "synthetic_only": SYNTHETIC_ONLY,
                    },
                }
            )

        # Queue rank is presentation order, so canonical score is now the
        # authority for queue ordering.
        rows.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["patient"]),
            )
        )

        snapshots.append(
            {
                "label": source_snapshot["label"],
                "scoring_mode": SCORING_MODE,
                "previous_observation_used": (
                    PREVIOUS_OBSERVATION_USED
                ),
                "synthetic_only": SYNTHETIC_ONLY,
                "rows": rows,
            }
        )

    counts = [len(snapshot["rows"]) for snapshot in snapshots]

    if counts != [4, 3, 4, 4]:
        raise RuntimeError(
            f"SCORING_INPUT_HOLD: unexpected active row counts {counts}"
        )

    return snapshots


def build_canonical_command_center_snapshots_json() -> str:
    payload = build_canonical_command_center_snapshots()

    # Safe for direct embedding into an inline JavaScript assignment.
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")

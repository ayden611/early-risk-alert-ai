"""
Early Risk Alert AI
Canonical Review Score

This module is initially an exact extraction of the corrected
era/__init__.py::_score_row implementation at the documented
2026-08-07 scorer behavior boundary.

No validation claims are attached to this module merely because
it exists. Dataset performance must be regenerated after integration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


SCORER_ID = "era-review-score-v1"
SCORER_BOUNDARY_SHA = "4b792cc745bca5db03e8dfcb58c62cb22c3bef79"
SCORER_SPEC_VERSION = "1.0.0"

_W_HR_BASE = 0.065
_W_SPO2 = 0.85
_W_SBP = 0.030
_W_RR = 0.18
_W_TEMP = 0.75


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_review_row(
    row: Dict[str, Any],
    prev_row: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Calculate the deterministic ERA 0–9.9 Review Score.

    Inputs:
      Current vital-sign observation and, when available,
      the immediately preceding observation for the same patient.

    This is prioritization-support scoring logic.
    It is not an individual probability of deterioration.
    """

    try:
        hr = float(row.get("heart_rate", 0) or 0)
        spo2 = float(row.get("spo2", 100) or 100)
        sbp = float(row.get("bp_systolic", 120) or 120)
        rr = float(row.get("respiratory_rate", 16) or 16)
        temp = float(row.get("temperature_f", 98.6) or 98.6)

        risk = max(0, hr - 90) * _W_HR_BASE
        risk += max(0, 94 - spo2) * _W_SPO2
        risk += max(0, sbp - 140) * _W_SBP
        risk += max(0, rr - 20) * _W_RR
        risk += max(0, temp - 99.0) * _W_TEMP

        if prev_row:
            try:
                dhr = hr - float(
                    prev_row.get("heart_rate", hr) or hr
                )
                dspo = spo2 - float(
                    prev_row.get("spo2", spo2) or spo2
                )
                drr = rr - float(
                    prev_row.get("respiratory_rate", rr) or rr
                )
                dsbp = sbp - float(
                    prev_row.get("bp_systolic", sbp) or sbp
                )
                dtemp = temp - float(
                    prev_row.get("temperature_f", temp) or temp
                )

                hr_rising = dhr > 5
                spo2_drop = dspo < -1.5
                rr_rising = drr > 2
                sbp_drop = dsbp < -10
                temp_rising = dtemp > 0.3

                det_count = sum(
                    [
                        hr_rising,
                        spo2_drop,
                        rr_rising,
                        sbp_drop,
                        temp_rising,
                    ]
                )

                if det_count >= 3:
                    risk += 1.5
                elif det_count == 2:
                    risk += 0.9
                elif det_count == 1:
                    risk += 0.4

                if hr_rising and spo2_drop:
                    risk += 0.6

                if spo2_drop and rr_rising:
                    risk += 0.7

                if hr_rising and rr_rising and not spo2_drop:
                    risk += 0.4

                if sbp_drop and hr_rising:
                    risk += 0.6

                if hr_rising and temp_rising:
                    risk += 0.3

            except Exception:
                pass

        return round(
            _clamp(risk, 0.0, 9.9),
            2,
        )

    except Exception:
        return 0.0

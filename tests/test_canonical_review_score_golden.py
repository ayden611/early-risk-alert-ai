from __future__ import annotations

import csv
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load the canonical scorer directly from its frozen module path.
# This contract intentionally avoids importing era/__init__.py so
# application/web dependencies cannot affect scorer verification.
import importlib.util

SCORING_MODULE = REPO_ROOT / "era" / "core" / "scoring.py"

spec = importlib.util.spec_from_file_location(
    "era_canonical_scoring_contract_target",
    SCORING_MODULE,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Unable to load canonical scorer module: {SCORING_MODULE}"
    )

scoring = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scoring
spec.loader.exec_module(scoring)

SCORER_BOUNDARY_SHA = scoring.SCORER_BOUNDARY_SHA
SCORER_ID = scoring.SCORER_ID
score_review_row = scoring.score_review_row


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "canonical_review_score_golden.csv"
)


NUMERIC_FIELDS = [
    "heart_rate",
    "spo2",
    "bp_systolic",
    "respiratory_rate",
    "temperature_f",
]


def parse_value(value: str):
    value = value.strip()
    if value == "":
        return None
    return float(value)


class CanonicalReviewScoreGoldenTest(unittest.TestCase):
    def test_frozen_golden_contract(self):
        self.assertTrue(
            FIXTURE.exists(),
            f"Missing fixture: {FIXTURE}",
        )

        tested = 0

        with FIXTURE.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            for record in reader:
                case_id = record["case_id"]

                self.assertEqual(
                    record["scorer_id"],
                    SCORER_ID,
                    case_id,
                )

                self.assertEqual(
                    record["boundary_sha"],
                    SCORER_BOUNDARY_SHA,
                    case_id,
                )

                row = {}
                prev = {}

                for field in NUMERIC_FIELDS:
                    value = parse_value(
                        record[field]
                    )

                    if value is not None:
                        row[field] = value

                    prev_value = parse_value(
                        record["prev_" + field]
                    )

                    if prev_value is not None:
                        prev[field] = prev_value

                expected_text = (
                    record["expected_score"].strip()
                )

                expected = float(expected_text)

                actual = score_review_row(
                    row,
                    prev if prev else None,
                )

                # Exact numeric equality is intentional:
                # the canonical scorer returns a value
                # rounded to two decimal places.
                self.assertEqual(
                    actual,
                    expected,
                    case_id,
                )

                # Also freeze its serialized 2-decimal
                # representation for audit artifacts.
                self.assertEqual(
                    f"{actual:.2f}",
                    expected_text,
                    case_id,
                )

                tested += 1

        self.assertGreaterEqual(
            tested,
            15,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from era.core.scoring import (  # noqa: E402
    SCORER_BOUNDARY_SHA,
    SCORER_ID,
    SCORER_SPEC_VERSION,
    score_review_row,
)


LEGACY_PATH = (
    ROOT
    / "tools"
    / "generate_real_era_validation_export.py"
)

EXPECTED_LEGACY_SHA256 = (
    "2936fd21d2737642e55281a43eb3f3d6"
    "dfed7b550b13a428b86ce2048fdad24f"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def sha256_source(obj) -> str:
    source = inspect.getsource(obj)

    return hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def load_legacy():
    actual = sha256_file(LEGACY_PATH)

    if actual != EXPECTED_LEGACY_SHA256:
        raise SystemExit(
            "ABORT: frozen metric helper SHA changed. "
            f"expected={EXPECTED_LEGACY_SHA256} "
            f"actual={actual}"
        )

    spec = importlib.util.spec_from_file_location(
        "era_legacy_metric_helper",
        LEGACY_PATH,
    )

    if spec is None or spec.loader is None:
        raise SystemExit(
            "ABORT: unable to load legacy metric helper."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def aggregate_scalars(result):
    safe = {}

    for key, value in result.items():
        if (
            value is None
            or isinstance(
                value,
                (
                    bool,
                    int,
                    float,
                    str,
                ),
            )
        ):
            safe[key] = value

    return safe


def quantile(values, q):
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return ordered[lo]

    frac = pos - lo

    return (
        ordered[lo] * (1 - frac)
        + ordered[hi] * frac
    )


def score_summary(values):
    if not values:
        return {
            "count": 0,
        }

    return {
        "count": len(values),
        "min": min(values),
        "p25": round(
            quantile(values, 0.25),
            4,
        ),
        "median": round(
            statistics.median(values),
            4,
        ),
        "mean": round(
            statistics.fmean(values),
            4,
        ),
        "p75": round(
            quantile(values, 0.75),
            4,
        ),
        "p90": round(
            quantile(values, 0.90),
            4,
        ),
        "p95": round(
            quantile(values, 0.95),
            4,
        ),
        "max": max(values),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        default=(
            "data/validation/reverified_runs"
        ),
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=6.0,
    )

    ap.add_argument(
        "--window-hours",
        type=float,
        default=6.0,
    )

    ap.add_argument(
        "--event-gap-hours",
        type=float,
        default=6.0,
    )

    ap.add_argument(
        "--source-manifest",
        default="",
    )

    args = ap.parse_args()

    legacy = load_legacy()

    input_path = Path(args.input).resolve()

    if not input_path.exists():
        raise SystemExit(
            "ABORT: input does not exist."
        )

    out_dir = (
        ROOT / args.out_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with input_path.open(
        "r",
        encoding="utf-8-sig",
        errors="strict",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        fieldnames = list(
            reader.fieldnames or []
        )

        rows = list(reader)

    patient_col = legacy.find_col(
        fieldnames,
        legacy.PATIENT_COLS,
    )

    time_col = legacy.find_col(
        fieldnames,
        legacy.TIME_COLS,
    )

    event_col = legacy.find_col(
        fieldnames,
        legacy.EVENT_FLAG_COLS,
    )

    event_group_col = legacy.find_col(
        fieldnames,
        legacy.EVENT_GROUP_COLS,
    )

    if not patient_col:
        raise SystemExit(
            "ABORT: patient column not found."
        )

    if not time_col:
        raise SystemExit(
            "ABORT: time column not found."
        )

    if not event_col:
        raise SystemExit(
            "ABORT: event column not found."
        )

    vital_cols = {
        key: legacy.find_col(
            fieldnames,
            candidates,
        )
        for key, candidates
        in legacy.VITAL_COLS.items()
    }

    required_vitals = {
        "heart_rate",
        "spo2",
        "bp_systolic",
        "respiratory_rate",
        "temperature_f",
    }

    missing_vitals = [
        key
        for key in sorted(required_vitals)
        if not vital_cols.get(key)
    ]

    if missing_vitals:
        raise SystemExit(
            "ABORT: missing canonical vital columns: "
            + ", ".join(missing_vitals)
        )

    by_patient = {}
    invalid_pid_or_time_rows = 0

    for row in rows:
        pid = str(
            row.get(patient_col, "")
        ).strip()

        ts = legacy.parse_time(
            row.get(time_col)
        )

        row["_parsed_time"] = ts

        if pid and ts:
            by_patient.setdefault(
                pid,
                [],
            ).append(row)
        else:
            invalid_pid_or_time_rows += 1

    scored_rows = []

    canonical_scores = []
    legacy_scores = []
    absolute_diffs = []

    for pid, patient_rows in by_patient.items():
        patient_rows.sort(
            key=lambda r: r["_parsed_time"]
        )

        prev = None

        for row in patient_rows:
            canonical_score = (
                score_review_row(
                    row,
                    prev,
                )
            )

            legacy_score, _, _ = (
                legacy.score_row(
                    row,
                    vital_cols,
                    prev,
                )
            )

            standard_alert = (
                legacy.standard_threshold_alert(
                    row,
                    vital_cols,
                )
            )

            row["_canonical_score"] = (
                canonical_score
            )

            row["_legacy_score"] = (
                legacy_score
            )

            row["_standard_alert"] = (
                standard_alert
            )

            row["_hour_bucket"] = (
                legacy.hour_bucket(
                    row["_parsed_time"]
                )
            )

            canonical_scores.append(
                canonical_score
            )

            legacy_scores.append(
                legacy_score
            )

            absolute_diffs.append(
                abs(
                    canonical_score
                    - legacy_score
                )
            )

            scored_rows.append(row)
            prev = row

    if len(scored_rows) != 456453:
        raise SystemExit(
            "ABORT: scored row count is "
            f"{len(scored_rows)}, expected 456453."
        )

    if len(by_patient) != 1705:
        raise SystemExit(
            "ABORT: scored patient count is "
            f"{len(by_patient)}, expected 1705."
        )

    raw_event_rows = sum(
        1
        for row in scored_rows
        if legacy.parse_bool(
            row.get(event_col)
        )
    )

    standard_alert_rows = sum(
        1
        for row in scored_rows
        if row["_standard_alert"]
    )

    thresholds = [
        4.0,
        5.0,
        6.0,
    ]

    canonical_metrics = {}
    legacy_metrics = {}
    divergence_by_threshold = {}

    for threshold in thresholds:

        for row in scored_rows:
            row["_era_alert"] = (
                float(
                    row["_canonical_score"]
                )
                >= threshold
            )

            row["risk_score"] = (
                row["_canonical_score"]
            )

        canonical_result = (
            legacy.compute_metrics(
                scored_rows,
                patient_col,
                time_col,
                event_col,
                event_group_col,
                threshold,
                args.window_hours,
                args.event_gap_hours,
            )
        )

        canonical_metrics[
            str(threshold)
        ] = aggregate_scalars(
            canonical_result
        )

        canonical_alerts = sum(
            1
            for row in scored_rows
            if (
                float(
                    row["_canonical_score"]
                )
                >= threshold
            )
        )

        for row in scored_rows:
            row["_era_alert"] = (
                float(
                    row["_legacy_score"]
                )
                >= threshold
            )

            row["risk_score"] = (
                row["_legacy_score"]
            )

        legacy_result = (
            legacy.compute_metrics(
                scored_rows,
                patient_col,
                time_col,
                event_col,
                event_group_col,
                threshold,
                args.window_hours,
                args.event_gap_hours,
            )
        )

        legacy_metrics[
            str(threshold)
        ] = aggregate_scalars(
            legacy_result
        )

        legacy_alerts = sum(
            1
            for row in scored_rows
            if (
                float(
                    row["_legacy_score"]
                )
                >= threshold
            )
        )

        canonical_only = 0
        legacy_only = 0
        both = 0
        neither = 0

        for row in scored_rows:
            c = (
                float(
                    row["_canonical_score"]
                )
                >= threshold
            )

            l = (
                float(
                    row["_legacy_score"]
                )
                >= threshold
            )

            if c and l:
                both += 1
            elif c:
                canonical_only += 1
            elif l:
                legacy_only += 1
            else:
                neither += 1

        divergence_by_threshold[
            str(threshold)
        ] = {
            "rows": len(scored_rows),
            "canonical_alert_rows": (
                canonical_alerts
            ),
            "legacy_alert_rows": (
                legacy_alerts
            ),
            "standard_threshold_alert_rows": (
                standard_alert_rows
            ),
            "both_alert": both,
            "canonical_only_alert": (
                canonical_only
            ),
            "legacy_only_alert": (
                legacy_only
            ),
            "neither_alert": neither,
            "alert_classification_disagreement_rows": (
                canonical_only
                + legacy_only
            ),
            "alert_classification_disagreement_pct": (
                round(
                    (
                        canonical_only
                        + legacy_only
                    )
                    * 100.0
                    / len(scored_rows),
                    4,
                )
            ),
        }

    selected_key = str(
        float(args.threshold)
    )

    if selected_key not in canonical_metrics:
        raise SystemExit(
            "ABORT: selected threshold result missing."
        )

    generated = datetime.now(
        timezone.utc
    )

    repo_sha = git_sha()
    repo_short = repo_sha[:12]
    input_sha = sha256_file(
        input_path
    )

    scorer_path = (
        ROOT
        / "era"
        / "core"
        / "scoring.py"
    )

    scorer_sha = sha256_file(
        scorer_path
    )

    run_id = (
        "mimic_canonical_"
        + generated.strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_t"
        + str(args.threshold).replace(
            ".",
            "_",
        )
        + "_"
        + repo_short
        + "_"
        + input_sha[:12]
    )

    try:
        input_identity = str(
            input_path.relative_to(ROOT)
        )
    except ValueError:
        input_identity = input_path.name

    source_manifest = None

    if args.source_manifest:
        manifest_path = Path(
            args.source_manifest
        )

        if manifest_path.exists():
            try:
                manifest_identity = str(
                    manifest_path.resolve().relative_to(
                        ROOT
                    )
                )
            except ValueError:
                manifest_identity = (
                    manifest_path.name
                )

            source_manifest = {
                "path": manifest_identity,
                "sha256": sha256_file(
                    manifest_path
                ),
            }

    metric_helper_functions = {
        name: sha256_source(
            getattr(
                legacy,
                name,
            )
        )
        for name in [
            "standard_threshold_alert",
            "collapse_event_times",
            "compute_metrics",
            "parse_time",
            "parse_bool",
        ]
    }

    run_payload = {
        "schema_version": (
            "era-canonical-reverification-v1"
        ),
        "status": (
            "candidate_canonical_reverification_unreviewed"
        ),
        "run_id": run_id,
        "generated_at_utc": (
            generated.isoformat()
        ),
        "dataset": "MIMIC-IV",
        "cohort": (
            "historical full validation cohort "
            "reconstructed from deterministic "
            "cohort folds 1/2/3"
        ),
        "input": {
            "path": input_identity,
            "sha256": input_sha,
            "source_manifest": source_manifest,
            "rows_loaded": len(rows),
            "rows_scored": len(
                scored_rows
            ),
            "patients": len(
                by_patient
            ),
            "raw_event_flag_rows": (
                raw_event_rows
            ),
            "invalid_patient_or_time_rows": (
                invalid_pid_or_time_rows
            ),
        },
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
            "scorer_file_sha256": (
                scorer_sha
            ),
        },
        "repository": {
            "git_sha": repo_sha,
        },
        "event_metric_method": {
            "helper_path": (
                "tools/"
                "generate_real_era_validation_export.py"
            ),
            "helper_file_sha256": (
                EXPECTED_LEGACY_SHA256
            ),
            "function_source_sha256": (
                metric_helper_functions
            ),
            "window_hours": (
                args.window_hours
            ),
            "event_gap_hours": (
                args.event_gap_hours
            ),
            "note": (
                "Historical event clustering and "
                "metric definitions are reused; "
                "historical scoring is not reused."
            ),
        },
        "selected_threshold": (
            args.threshold
        ),
        "selected_canonical_metrics": (
            canonical_metrics[
                selected_key
            ]
        ),
        "canonical_threshold_results": (
            canonical_metrics
        ),
        "aggregate_counts": {
            "rows": len(
                scored_rows
            ),
            "patients": len(
                by_patient
            ),
            "raw_event_flag_rows": (
                raw_event_rows
            ),
            "standard_threshold_alert_rows": (
                standard_alert_rows
            ),
        },
        "claim_boundary": (
            "Retrospective aggregate "
            "reverification only. "
            "Not prospective clinical validation."
        ),
    }

    divergence_payload = {
        "schema_version": (
            "era-canonical-vs-legacy-divergence-v1"
        ),
        "status": (
            "historical_method_comparison"
        ),
        "run_id": run_id,
        "generated_at_utc": (
            generated.isoformat()
        ),
        "dataset": "MIMIC-IV",
        "input_sha256": input_sha,
        "repository_git_sha": repo_sha,
        "canonical_scorer_id": (
            SCORER_ID
        ),
        "legacy_scorer": {
            "callable": (
                "tools/"
                "generate_real_era_validation_export.py"
                "::score_row"
            ),
            "function_source_sha256": (
                sha256_source(
                    legacy.score_row
                )
            ),
            "role": (
                "historical comparison only; "
                "not canonical product evidence"
            ),
        },
        "score_distribution": {
            "canonical": (
                score_summary(
                    canonical_scores
                )
            ),
            "legacy": (
                score_summary(
                    legacy_scores
                )
            ),
            "absolute_difference": (
                score_summary(
                    absolute_diffs
                )
            ),
            "exact_score_equality_rows": (
                sum(
                    1
                    for c, l
                    in zip(
                        canonical_scores,
                        legacy_scores,
                    )
                    if c == l
                )
            ),
        },
        "threshold_divergence": (
            divergence_by_threshold
        ),
        "canonical_metrics": (
            canonical_metrics
        ),
        "legacy_metrics": (
            legacy_metrics
        ),
        "interpretation": (
            "This artifact quantifies how far "
            "historical validation-local scoring "
            "diverged from the adopted canonical "
            "ERA Review Score. Historical values "
            "are not targets and are not promoted "
            "as canonical results."
        ),
    }

    run_path = (
        out_dir
        / (
            run_id
            + "_aggregate.json"
        )
    )

    divergence_path = (
        out_dir
        / (
            run_id
            + "_legacy_divergence.json"
        )
    )

    run_text = json.dumps(
        run_payload,
        indent=2,
        sort_keys=True,
    ) + "\n"

    divergence_text = json.dumps(
        divergence_payload,
        indent=2,
        sort_keys=True,
    ) + "\n"

    forbidden = [
        '"patient_id"',
        '"subject_id"',
        '"hadm_id"',
        '"stay_id"',
        '"case_id"',
        "/Users/",
    ]

    for marker in forbidden:
        if (
            marker in run_text
            or marker in divergence_text
        ):
            raise SystemExit(
                "ABORT: restricted-looking "
                f"identifier leaked: {marker}"
            )

    run_path.write_text(
        run_text,
        encoding="utf-8",
    )

    divergence_path.write_text(
        divergence_text,
        encoding="utf-8",
    )

    print("CANONICAL_MIMIC_RUN=PASS")
    print(f"RUN_ID={run_id}")
    print(
        "REPOSITORY_GIT_SHA="
        + repo_sha
    )
    print(
        "INPUT_SHA256="
        + input_sha
    )
    print(
        "SCORER_SHA256="
        + scorer_sha
    )
    print(
        "ROWS="
        + str(len(scored_rows))
    )
    print(
        "PATIENTS="
        + str(len(by_patient))
    )
    print(
        "RAW_EVENT_FLAG_ROWS="
        + str(raw_event_rows)
    )
    print(
        "CANONICAL_RESULT="
        + str(run_path.relative_to(ROOT))
    )
    print(
        "DIVERGENCE_RESULT="
        + str(
            divergence_path.relative_to(
                ROOT
            )
        )
    )


if __name__ == "__main__":
    main()

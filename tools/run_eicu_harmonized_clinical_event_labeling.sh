#!/usr/bin/env bash
set -euo pipefail

echo "RUNNING HARMONIZED eICU CLINICAL-EVENT LABELING"
echo "Row-level output stays local-only."

INPUT="${EICU_ERA_COHORT:-}"
MAX_ROWS="${MAX_ROWS:-0}"

if [ -n "$INPUT" ] && [ "$MAX_ROWS" != "0" ]; then
  python3 tools/eicu_harmonized_clinical_event_labeler.py --input "$INPUT" --max-rows "$MAX_ROWS"
elif [ -n "$INPUT" ]; then
  python3 tools/eicu_harmonized_clinical_event_labeler.py --input "$INPUT"
elif [ "$MAX_ROWS" != "0" ]; then
  python3 tools/eicu_harmonized_clinical_event_labeler.py --max-rows "$MAX_ROWS"
else
  python3 tools/eicu_harmonized_clinical_event_labeler.py
fi

echo ""
echo "DONE — harmonized eICU clinical-event labeling completed."
echo "Local row-level file:"
echo " - data/validation/local_private/eicu/harmonized/eicu_harmonized_clinical_event_labeled_era_cohort.csv"
echo "Public aggregate files:"
echo " - data/validation/eicu_harmonized_clinical_event_summary.json"
echo " - docs/validation/eicu_harmonized_clinical_event_summary.md"

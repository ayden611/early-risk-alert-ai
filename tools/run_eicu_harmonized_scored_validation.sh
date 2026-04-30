#!/usr/bin/env bash
set -euo pipefail

echo "RUNNING LOCAL-ONLY eICU SCORING + HARMONIZED CLINICAL-EVENT VALIDATION"
echo "Row-level scored output stays local-only."

INPUT="${EICU_ERA_COHORT:-}"
SCORED="data/validation/local_private/eicu/working/eicu_era_scored_harmonized_input.csv"

if [ -n "$INPUT" ]; then
  python3 tools/eicu_add_era_scores.py --input "$INPUT" --output "$SCORED"
else
  python3 tools/eicu_add_era_scores.py --output "$SCORED"
fi

EICU_ERA_COHORT="$SCORED" bash tools/run_eicu_harmonized_clinical_event_labeling.sh

echo ""
echo "DONE — scored harmonized eICU validation completed."
echo "Local-only scored cohort:"
echo " - $SCORED"
echo "Public aggregate review files:"
echo " - data/validation/eicu_local_scoring_adapter_audit_summary.json"
echo " - data/validation/eicu_harmonized_clinical_event_summary.json"
echo " - docs/validation/eicu_harmonized_clinical_event_summary.md"

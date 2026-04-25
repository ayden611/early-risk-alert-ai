#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  tools/run_mimic_validation_dua_safe.sh /path/to/mimic_strict_event_labeled_era_cohort.csv --threshold 6.0 --window-hours 6 --event-gap-hours 6"
  exit 1
fi

python3 tools/generate_real_era_validation_export.py "$@"
python3 tools/mimic_dua_public_safety.py

echo ""
echo "DUA-safe validation run complete."
echo "Public files are sanitized."
echo "Row-level enriched CSV remains local-only and ignored by Git."

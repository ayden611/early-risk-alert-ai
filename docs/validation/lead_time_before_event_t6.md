# Lead Time Before Event — ERA Retrospective Validation

## Summary

Among detected event clusters, ERA flags appeared a median of 4.0 hours before documented event timestamps within the 6-hour pre-event window.

## Calculation Mode

derived_vitals_proxy_score

This CSV did not contain era_alert or risk_score columns, so the script used a transparent vitals-based proxy score.

## Configuration

- Source file: `/Users/andreasmith/Desktop/mimic_strict_event_labeled_era_cohort.csv`
- ERA threshold: 6.0
- Pre-event window: 6.0 hours
- Event clustering gap: 6.0 hours

## Results

- Total rows: 456,453
- Total patients: 1,705
- Total event patients: 1,649
- Detected event patients: 253
- Patient-level detection: 15.3%
- Total event clusters: 1,658
- Detected event clusters: 253
- Event-cluster detection: 15.3%
- Median lead time before event: 4.0 hours
- Median patient-level lead time: 4.0 hours
- Detected events flagged over 1 hour before event: 95.7%
- Detected events flagged over 2 hours before event: 83.4%
- Detected events flagged over 3 hours before event: 71.1%
- Alerts per patient-day: 0.6467
- Alerts per event patient: 11.6525

## Explainability Summary

Top primary drivers:

- no dominant driver: 111480
- RR instability: 60490
- RR elevation: 45383
- borderline SpO2: 40995
- BP trend concern: 38708
- SpO2 decline: 33256
- HR elevation/depression: 31567
- severe RR instability: 23135
- HR instability: 15230
- severe BP instability: 8979
- BP instability: 8940
- severe HR instability: 7905

Priority tier counts:

- Low: 390632
- Watch: 46606
- Critical: 9768
- Elevated: 9447

## Pilot-Safe Interpretation

This metric should be framed as retrospective timing analysis on de-identified data. It supports pilot planning and clinical review, but does not establish prospective performance.

Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.

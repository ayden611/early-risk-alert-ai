# MIMIC DUA Public Safety Policy

## Public-Facing Rule

Only aggregate metrics, sanitized case examples, validation methodology, and code should be public-facing.

## Safe to Publish

- Aggregate validation metrics
- Code used to generate results
- High-level threshold tables
- Alert reduction percentage
- False-positive rate
- Patient detection percentage
- Median lead-time summary
- Pilot-safe evidence packet
- Validation methodology
- Sanitized case examples using Case-001 style labels

## Local-Only / Not Public

- Raw MIMIC CSV files
- Row-level enriched CSV files
- Patient-level rows
- MIMIC patient IDs
- Subject IDs
- Hospital admission IDs
- Stay IDs
- Exact timestamps tied to cases/patients
- Representative examples with real MIMIC IDs or exact timestamps
- Any file that could be treated as derived restricted data

## Public Example Format

- Case ID, such as Case-001
- Lead time only
- Priority tier
- Primary driver
- Trend direction
- Score
- Queue rank

No MIMIC IDs. No exact timestamps. No row-level export.

## Notice

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.

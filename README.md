# Early Risk Alert AI

## Current Platform Summary

Early Risk Alert AI is a pilot-stage, pre-commercial healthcare-professional-facing decision-support and workflow-support software platform.

The platform is designed to organize already-acquired numeric vital-sign observations into pattern-based review notifications and explainable review context for authorized healthcare professionals. It supports independent professional review and monitored-workflow prioritization.

### Current Input Boundary

The pilot-stage software may display already-acquired numeric observations, including:

* heart-rate observations
* non-invasive blood-pressure observations
* SpO2 observations
* respiratory-rate observations
* temperature observations, where available
* relevant review-context and workflow-state information

The platform does not directly acquire physiologic signals from patients and is not intended to process raw continuous waveform data.

### Current Output Boundary

The pilot-stage software may present:

* pattern-based review notifications
* contributing numeric variables
* available trend context
* review-queue organization
* workflow-state visibility
* data-freshness and limitation context

### Decision-Support Boundary

Decision support only. Early Risk Alert AI does not diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.

The platform is not intended to replace bedside monitoring systems, immediate physiologic alarms, code-blue systems, cardiac-arrest alarms, or hospital emergency-response systems. Healthcare professionals independently review the underlying numeric observations, relevant context, and patient record before determining whether any action is appropriate.

### Evidence Boundary

Current evidence is retrospective, de-identified, and aggregate only.

Internal retrospective aggregate analyses have used critical-care datasets including MIMIC-IV and eICU. Performance characteristics vary by dataset, event definition, and operating threshold. Metrics must be interpreted within their respective evidence-track definitions and must not be merged across tracks.

These analyses do not constitute prospective clinical validation, diagnostic-performance claims, or proof of patient-outcome improvement.

HiRID status: Access approved; local/private retrospective aggregate evaluation pending.

### Pilot and Integration Boundary

The current pilot portals are limited to de-identified evaluation data. No live PHI flows are active.

Future integration planning may consider standard health-data exchange protocols, such as HL7 and FHIR, where appropriate. Live hospital integration is not currently active.

### Regulatory Status

Early Risk Alert AI is preparing a Section 513(g) Request for Information to seek FDA information regarding classification and applicable regulatory requirements.

A Section 513(g) request is not FDA clearance, FDA approval, or a determination of clinical performance. Any future FDA Q-Submission or pre-submission engagement will be handled as a separate regulatory interaction.

### Historical Repository Note

Earlier repository sections document the original educational prototype and the development history of the platform. They should be interpreted as historical records, not as the current intended-use statement or current public positioning.

---

<!-- ERA_VALIDATION_ROUTES_V2_START -->
## Validation Intelligence and Pilot Evidence Routes

The platform includes a pilot-safe retrospective validation and evidence workflow:

| Route | Purpose |
|---|---|
| `/command-center` | Live command-center demo with validation intelligence and patient-card explainability context |
| `/validation-intelligence` | Hospital-facing validation showcase |
| `/validation-evidence` | Printable Pilot Evidence Packet |
| `/validation-evidence/download.md` | Downloadable Markdown evidence packet |
| `/validation-evidence/download.json` | Downloadable validation JSON |
| `/validation-evidence/examples.csv` | Representative detected review examples CSV |
| `/api/validation/evidence` | JSON evidence API |
| `/api/validation/milestone` | Validation milestone API |
| `/data-ingest` | De-identified retrospective CSV upload and validation workflow |

Pilot-safe framing: retrospective analysis on de-identified MIMIC data showed ERA can reduce alert burden while maintaining configurable patient-level detection in a 6-hour pre-event window.

Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
<!-- ERA_VALIDATION_ROUTES_V2_END -->


<!-- ERA_VALIDATION_ROUTES_V3_START -->
## Validation Intelligence, Evidence Export, and Pilot Routes

Early Risk Alert AI includes a pilot-safe retrospective validation and evidence workflow.

### Key Validation Results

- Rows analyzed: 456,453
- Patients: 1,705
- Clinical events: 21,396
- Conservative threshold: t=6.0
- Alert reduction: 81%
- ERA false-positive rate: 4.5%
- Patient-level detection: 36.6%
- Median first-flag timing among detected event clusters: approximately 4.0 hours

### Routes

| Route | Purpose |
|---|---|
| `/command-center` | Live command-center demo with validation intelligence and patient-card explainability context |
| `/validation-intelligence` | Hospital-facing validation showcase |
| `/validation-evidence` | Printable Pilot Evidence Packet |
| `/validation-evidence/download.md` | Downloadable Markdown evidence packet |
| `/validation-evidence/download.json` | Downloadable validation JSON |
| `/validation-evidence/examples.csv` | Representative detected review examples CSV |
| `/api/validation/evidence` | JSON evidence API |
| `/api/validation/milestone` | Validation milestone API |
| `/data-ingest` | De-identified retrospective CSV upload and validation workflow |

Pilot-safe framing: retrospective analysis on de-identified MIMIC data showed ERA can reduce alert burden while maintaining configurable patient-level detection in a 6-hour pre-event window.

Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
<!-- ERA_VALIDATION_ROUTES_V3_END -->


<!-- ERA_REAL_VALIDATION_EXPORT_ROUTES_START -->
## Real ERA Validation Export and Run Registry

The validation workflow now supports enriched retrospective exports for repeated MIMIC testing.

### Enriched Export Columns

- risk_score
- era_alert
- priority_tier
- primary_driver
- trend_direction
- threshold_crossed_at
- queue_rank
- standard_threshold_alert

### Run Registry Fields

- Run ID
- Dataset
- Rows
- Patients
- Events
- Threshold
- Alert reduction
- FPR
- Patient detection
- Median lead time
- Date generated
- Validation status

### Routes

| Route | Purpose |
|---|---|
| `/validation-runs` | Validation Run Registry page |
| `/api/validation/runs` | Validation Run Registry API |
| `/validation-evidence/runs.json` | Downloadable run registry JSON |
| `/validation-evidence/latest-enriched.csv` | Downloadable latest enriched ERA validation CSV |

### Manual MIMIC Test Command

```bash
python3 tools/generate_real_era_validation_export.py /path/to/mimic_strict_event_labeled_era_cohort.csv --threshold 6.0 --window-hours 6 --event-gap-hours 6
Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
<!-- ERA_REAL_VALIDATION_EXPORT_ROUTES_END -->


<!-- ERA_MIMIC_DUA_PUBLIC_SAFETY_V1_START -->
## MIMIC / PhysioNet Data-Use Safety

Public-facing validation evidence is limited to aggregate metrics, sanitized case examples, validation methodology, and code.

### Public-Safe

- Aggregate validation metrics
- High-level threshold tables
- Alert reduction percentage
- False-positive rate
- Patient detection percentage
- Median lead-time summary
- Pilot-safe evidence packet
- Validation methodology
- Sanitized Case-001 style examples

### Local-Only / Not Public

- Raw MIMIC CSV files
- Row-level enriched CSV files
- Patient-level rows
- MIMIC patient IDs
- Subject IDs
- Hospital admission IDs
- Stay IDs
- Exact timestamps tied to cases/patients
- Representative examples with real MIMIC IDs or exact timestamps

### Safer Testing Command

Use the DUA-safe wrapper for future local MIMIC testing:

```bash
tools/run_mimic_validation_dua_safe.sh ~/Desktop/mimic_strict_event_labeled_era_cohort.csv --threshold 6.0 --window-hours 6 --event-gap-hours 6
The wrapper generates local/private outputs, sanitizes public artifacts, and keeps row-level MIMIC-derived CSV exports out of Git.
Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
<!-- ERA_MIMIC_DUA_PUBLIC_SAFETY_V1_END -->


<!-- ERA_REAL_ENGINE_PUBLIC_WORDING_V2_START -->
## Real-Engine DUA-Safe Validation Framing

The current public validation pages use the latest DUA-safe real-engine retrospective validation run.

### Conservative t=6.0 Review Queue

- Alert reduction: 94.3%
- ERA FPR: 4.2%
- Event-cluster detection: 15.3%
- Median lead time among detected event clusters: 4.0 hours
- ERA alerts per patient-day: 0.6467
- Standard threshold alerts per patient-day: 11.2743

t=6.0 is intentionally selective and should be framed as a conservative telemetry / stepdown review queue optimized for alert-burden reduction and low false positives.

### High-Acuity t=4.0 Review Queue

- Alert reduction: 80.3%
- ERA FPR: 14.4%
- Event-cluster detection: 37.7%
- Median lead time: 4.0 hours
- ERA alerts per patient-day: 2.2154

t=4.0 is better for ICU / high-acuity review when the goal is higher detection with more alert volume.

### Pilot-Safe Claim

Retrospective analysis on de-identified MIMIC data showed ERA can support configurable review-prioritization workflows with substantially reduced alert burden and retrospective lead-time context.

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
<!-- ERA_REAL_ENGINE_PUBLIC_WORDING_V2_END -->


<!-- ERA_LEAD_TIME_ROBUSTNESS_PROGRESS_V1_START -->
## Lead-Time Robustness / Event-Window Sensitivity Progress

Early Risk Alert AI completed a DUA-safe lead-time robustness matrix across:

- 3-hour retrospective event window
- 6-hour retrospective event window
- 12-hour retrospective event window

Each window was tested at:

- t=4.0 — ICU / high-acuity
- t=5.0 — mixed-unit balanced
- t=6.0 — telemetry / stepdown conservative

Public outputs remain aggregate-only and DUA-safe. Row-level MIMIC-derived exports remain local-only.

### New Routes

| Route | Purpose |
|---|---|
| `/api/validation/lead-time-sensitivity` | Aggregate lead-time robustness API |
| `/validation-evidence/lead-time-sensitivity.json` | Downloadable aggregate lead-time robustness JSON |

Decision support only. Retrospective analysis only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
<!-- ERA_LEAD_TIME_ROBUSTNESS_PROGRESS_V1_END -->


<!-- ERA_CROSS_COHORT_RELEASE_README_V1_START -->
## Stable Cross-Cohort Validation Evidence Release

Release ID: `stable-cross-cohort-validation-release-2026-04-25`

**Across the full validation cohort and two deterministic patient-level subcohorts (577–1,705 cases each), the conservative t=6.0 ERA setting showed consistent low-burden review-queue performance, with 94.0%–94.9% alert reduction, 3.7%–4.4% ERA FPR, 14.1%–16.0% event-cluster detection, and a stable 4.0 hours median lead-time context across all cohorts.**

### Locked Public Routes

| Route | Purpose |
|---|---|
| `/validation-intelligence` | Cross-cohort validation story |
| `/validation-evidence` | Printable evidence packet |
| `/validation-runs` | Validation run registry |
| `/command-center` | Live command-center demo aligned to release evidence |
| `/api/validation/cross-cohort-validation` | Aggregate cross-cohort validation JSON |
| `/validation-evidence/cross-cohort-validation.json` | Downloadable aggregate cross-cohort evidence |

### Public Evidence Boundary

Aggregate DUA-safe evidence only. Row-level MIMIC-derived exports, raw restricted CSVs, real restricted identifiers, exact case-linked timestamps, and patient-level rows remain local-only.

Decision support only. Retrospective aggregate analysis only.
<!-- ERA_CROSS_COHORT_RELEASE_README_V1_END -->


<!-- ERA_MULTI_DATASET_ROBUSTNESS_README_V1_START -->
## Multi-Dataset Robustness Summary

**MIMIC-IV established strict clinical-event cross-cohort retrospective stability, while eICU added a separate second-dataset outcome-proxy check; across both datasets, ERA preserved the same threshold-direction behavior: lower thresholds increased detection, while conservative thresholds reduced review burden and false positives.**

At the conservative t=6.0 operating point, MIMIC-IV showed 4 hrs median lead-time context across locked cross-cohort evidence, while eICU showed 3.41 hrs median lead-time context in the outcome-proxy check.

### Evidence Roles

| Dataset | Role |
|---|---|
| MIMIC-IV v3.1 | Locked strict clinical-event cross-cohort retrospective validation release |
| eICU Collaborative Research Database v2.0 | Separate second-dataset retrospective outcome-proxy check |

### Citation Story

- MIMIC-IV v3.1: Johnson et al. (2024), PhysioNet, DOI 10.13026/kpb9-mt58.
- MIMIC-IV Scientific Data: Johnson et al. (2023), DOI 10.1038/s41597-022-01899-x.
- eICU v2.0: Pollard et al. (2019), PhysioNet, DOI 10.13026/C2WM1R.
- eICU Scientific Data: Pollard et al. (2018), DOI 10.1038/sdata.2018.178.
- PhysioNet standard citation: Goldberger et al. (2000), Circulation.

### Public Boundary

Aggregate DUA-safe evidence only. Raw restricted files and row-level outputs remain local-only.

Decision support only. Retrospective aggregate analysis only.
<!-- ERA_MULTI_DATASET_ROBUSTNESS_README_V1_END -->


<!-- ERA_MULTI_DATASET_CHECKPOINT_README_V1_START -->
## Multi-Dataset Retrospective Robustness Checkpoint

Checkpoint ID: `multi-dataset-retrospective-robustness-checkpoint-2026-04-30`

**MIMIC-IV established strict clinical-event cross-cohort retrospective stability, while eICU added a separate second-dataset outcome-proxy check; across both datasets, ERA preserved the same threshold-direction behavior: lower thresholds increased detection, while conservative thresholds reduced review burden and false positives.**

At the conservative t=6.0 operating point, MIMIC-IV showed 4 hrs median lead-time context across locked cross-cohort evidence, while eICU showed 3.41 hrs median lead-time context in the outcome-proxy check.

### Evidence Roles

| Dataset | Role |
|---|---|
| MIMIC-IV v3.1 | Locked strict clinical-event cross-cohort retrospective validation release |
| eICU Collaborative Research Database v2.0 | Separate second-dataset retrospective outcome-proxy check |

### Public Routes

| Route | Purpose |
|---|---|
| `/validation-intelligence` | Public validation story |
| `/validation-evidence` | Printable/downloadable evidence packet |
| `/api/validation/multi-dataset-robustness` | Multi-dataset aggregate summary |
| `/api/validation/multi-dataset-checkpoint` | Locked multi-dataset checkpoint |
| `/api/validation/eicu-validation` | eICU aggregate outcome-proxy summary |
| `/api/validation/cross-cohort-validation` | MIMIC-IV cross-cohort aggregate summary |

### Public Boundary

Aggregate DUA-safe evidence only. Raw restricted files and row-level outputs remain local-only.

Decision support only. Retrospective aggregate analysis only.
<!-- ERA_MULTI_DATASET_CHECKPOINT_README_V1_END -->


<!-- ERA_MULTI_DATASET_PUBLIC_FRAMING_POLISH_README_V1_START -->
## Multi-Dataset Public Framing

**Early Risk Alert AI now has retrospective evidence across two de-identified ICU datasets: MIMIC-IV strict clinical-event cross-cohort validation plus a separate eICU outcome-proxy check.**

Across both datasets, ERA preserved the same threshold-direction behavior: lower thresholds increased detection, while conservative thresholds reduced review burden and false positives.

At the conservative t=6.0 setting, MIMIC-IV showed 4 hrs median lead-time context across the locked cross-cohort release, while eICU showed 3.41 hrs median lead-time context in the outcome-proxy check.

### Critical Interpretation Guardrail

MIMIC-IV and eICU detection rates should not be treated as equivalent endpoint definitions because MIMIC-IV uses stricter clinical-event labels, while eICU uses outcome-proxy event labels derived from mortality/discharge context.

### Approved Claim

Cross-dataset retrospective robustness evidence across de-identified ICU datasets.

### Do Not Claim

- validated on two datasets without qualification
- proven generalizability
- prospective validation
- diagnosis, treatment direction, prevention, or autonomous escalation

Public evidence remains aggregate-only. Raw restricted files and row-level outputs remain local-only.
<!-- ERA_MULTI_DATASET_PUBLIC_FRAMING_POLISH_README_V1_END -->


<!-- ERA_FINAL_FRONTEND_EVIDENCE_POLISH_README_V1_START -->
## Final Frontend Evidence Polish

The live platform now includes:

- `/validation-intelligence`
- `/validation-evidence`
- `/validation-runs`
- `/model-card`
- `/pilot-success-guide`
- `/api/validation/final-frontend-polish`

The public story is:

**Early Risk Alert AI now has retrospective evidence across two de-identified ICU datasets: MIMIC-IV strict clinical-event cross-cohort validation plus a separate eICU outcome-proxy check.**

Across both datasets, ERA preserved the same threshold-direction behavior: lower thresholds increased detection, while conservative thresholds reduced review burden and false positives.

Important guardrail:

MIMIC-IV and eICU detection rates should not be treated as equivalent endpoint definitions because MIMIC-IV uses stricter clinical-event labels, while eICU uses outcome-proxy event labels derived from mortality/discharge context.

No raw restricted files or row-level outputs are published.
<!-- ERA_FINAL_FRONTEND_EVIDENCE_POLISH_README_V1_END -->


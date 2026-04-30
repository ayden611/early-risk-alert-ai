# Early Risk Alert AI
## Summary

Early Risk Alert AI is a web-based machine learning application that predicts whether a person may be at low or high cardiovascular risk using health indicators such as age, BMI, exercise level, blood pressure, heart rate, and smoking status. The system provides a risk classification and probability score, stores predictions in a cloud database, and is deployed as a scalable web service.

Author: Milton Munroe  
Course: Elements of AI – University of Helsinki  
Project Type: Probabilistic Classification (Naive Bayes)

1. Your idea in a nutshell
Project Name: Early Risk Alert AI
Early Risk Alert AI is a web-based machine learning application that predicts whether a person may be at low or high cardiovascular risk based on health indicators such as age, BMI, exercise level, blood pressure, and heart rate. The system provides a risk classification and probability score and stores prediction history in a database.

2. Background
Cardiovascular disease is one of the leading causes of death worldwide. Many individuals are unaware of their potential risk level until symptoms become serious.
This project addresses the problem of early awareness. By using health indicators that people commonly know (age, blood pressure, heart rate, BMI), the system provides an immediate estimate of risk.
The motivation behind this project is to demonstrate how AI can assist in preventative health awareness. Even simple models can help individuals better understand how lifestyle and measurable health metrics influence risk.
This topic is important because early detection and awareness can lead to better health decisions and potentially prevent serious outcomes.

3. Data and AI Techniques
This project uses a supervised machine learning classification model trained on structured health-related data.
AI techniques used:
Supervised learning
Binary classification
Probability prediction
Feature-based modeling
NumPy for feature arrays
Joblib for model loading
SQLAlchemy + PostgreSQL for storing prediction logs
The model takes numerical inputs:
Age
BMI
Exercise level
Systolic blood pressure
Diastolic blood pressure
Heart rate
The system outputs:
Risk classification (Low / High)
Probability score
The application is implemented using:
Python
Flask (web framework)
PostgreSQL (cloud database)
Deployed on Render

4. How it is used
The application is used through a web interface.
A user:
Enters their health metrics
Submits the form
Receives a risk classification and probability
The prediction is stored in a database
The user can view previous predictions in a history page
The affected users are:
Individuals interested in understanding cardiovascular risk
Students learning about AI in healthcare
Developers studying ML deployment

5. Challenges
This project does not replace medical professionals. It is not a diagnostic tool.
Limitations include:
Model accuracy depends on training data quality
Limited feature set
No integration with real medical records
No personalization beyond input variables
Risk of over-reliance on simplified predictions
AI models must be interpreted carefully and responsibly.

6. Whats next?
Future improvements could include:
Larger and more diverse datasets
Integration with wearable devices
Personalized health tracking
Mobile app version
Improved UI/UX design
Model retraining pipeline
Security and authentication
HIPAA-compliant architecture
API for mobile apps
Cloud scaling and monitoring
The project could evolve into a full preventive health monitoring platform.

7. Acknowledgments
This project was developed using:
Python
Flask
SQLAlchemy
NumPy
Joblib
PostgreSQL
Render for deployment
Open-source machine learning tools
Inspiration from introductory AI coursework and machine learning education materials.
## 8. Demo Code

A simple Python implementation using Gaussian Naive Bayes is included in `demo_model.py`.

This demonstrates:
- Feature structuring
- Model training
- Risk classification
- Probability estimation

## 9. How to Run This Project

1. Clone the repository:
   git clone https://github.com/ayden611/early-risk-alert-ai.git

2. Navigate into the folder:
   cd early-risk-alert-ai

3. Install dependencies:
   pip install -r requirements.txt

4. Run the demo model:
   python demo_model.py

## 10. Model Evaluation

The model can be evaluated using:

- Accuracy score
- Confusion matrix
- Classification report (precision, recall, F1-score)

Future versions will include train/test split validation
and performance metrics visualization.

Mermaid:
flowchart LR
  subgraph Clients
    M[Mobile App / Devices]
    W[Web Dashboard]
  end

  subgraph API[early-risk-alert-mobile-api]
    V1[POST /vitals]
    AL[GET /alerts]
    SSE[GET /stream/alerts (SSE)]
  end

  subgraph Stream[Streaming Layer]
    RS[(Redis Streams)]
    PUB[(Redis PubSub)]
    K[(Kafka - optional)]
  end

  subgraph Worker[early-risk-alert-ai-worker]
    C1[Consumer Group: vitals-workers]
    DET[Anomaly Detection]
    WR[Write Alerts]
    PUSH[Publish Realtime Alerts]
  end

  subgraph DB[(Postgres)]
    VE[vitals_events]
    A[alerts]
  end

  M --> V1
  V1 --> VE
  V1 --> RS
  RS --> C1 --> DET --> WR --> A
  WR --> PUSH --> PUB --> SSE --> W
  AL --> A

- Open-source Python libraries such as NumPy and scikit-learn


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


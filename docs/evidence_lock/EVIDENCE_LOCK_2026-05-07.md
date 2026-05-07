# Early Risk Alert AI Stable Evidence Lock — May 7, 2026

**Lock ID:** `stable-evidence-lock-2026-05-07`  
**Date:** 2026-05-07  
**Generated UTC:** 2026-05-07T12:03:49.261147+00:00

## 1. Locked Intended Use

Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized health care professionals identify patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness. It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.

## 2. Current Evidence Status

Early Risk Alert AI should currently be presented as a **pilot-stage HCP-facing decision-support and workflow-support command-center platform** supported by **retrospective aggregate evidence only**.

This evidence lock does **not** claim:

- Prospective clinical validation
- Clinical outcome improvement
- FDA clearance or approval
- Diagnosis, treatment direction, or independent escalation
- Final HiRID performance
- Replacement of clinician judgment or standard monitoring

## 3. Dataset-Specific Evidence Boundary

### MIMIC-IV

Status: retrospective de-identified dataset used for aggregate-only evidence.

Safe framing:

> Retrospective aggregate evidence supporting operational-burden and prioritization analysis.

Current conservative internal headline figures must be verified against the local aggregate validation registry before external publication:

- Median retrospective lead-time context around **4.0 hours** among detected clusters across locked threshold views.
- Conservative **t=6.0** operating point used as a lower-burden posture.
- Prior internal summaries referenced approximately **0.65 ERA alerts per patient-day** versus approximately **11.27 standard threshold-rule alerts per patient-day**, roughly **94.3% operational alert reduction**.
- Prior internal summaries referenced approximately **4.2% false-positive behavior** and approximately **15.3% patient-level detection** under strict MIMIC t=6.0 framing.

### eICU

Status: second retrospective de-identified dataset used for aggregate-only robustness framing.

Safe framing:

> Retrospective aggregate evidence supporting cross-dataset robustness of the operational-burden and prioritization story.

Guardrail:

Do not overstate eICU as prospective clinical validation.

### HiRID

Only safe public wording:

> HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.

Guardrail:

Do not publish HiRID performance, FPR, detection rate, lead time, final validation, diagnosis, treatment, or escalation claims until local evaluation is complete and reviewed.

## 4. Approved Public Claims

- HCP-facing decision-support and workflow-support platform.
- Designed to support authorized health care professionals with patient prioritization and command-center operational awareness.
- Provides explainable context such as priority tier, queue rank, primary driver, trend direction, and retrospective lead-time context.
- Supports controlled pilot evaluation and operational review workflows.
- Uses retrospective, aggregate-only validation evidence for current evidence materials.
- Workflow actions are operational state tracking only and do not independently trigger clinical escalation.
- HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.

## 5. Banned or Restricted Claims

- Do not say the platform is clinically validated.
- Do not say the platform predicts deterioration as a clinical claim.
- Do not say the platform prevents adverse events.
- Do not say the platform replaces standard monitoring.
- Do not say the platform replaces clinician judgment.
- Do not say the platform independently triggers escalation.
- Do not say the platform diagnoses, treats, or directs treatment.
- Do not say the platform is FDA cleared, FDA approved, or FDA exempt.
- Do not publish HiRID performance, FPR, detection rate, lead time, or final validation language until local evaluation is complete and reviewed.
- Do not upload raw rows, patient-level outputs, timestamps, identifiers, or restricted dataset files to any public site, repo, demo, investor page, or production platform.

## 6. RN / Pilot Review Boundary

The RN/pilot review packet is for **private clinical-advisor and pilot-readiness review only**.

Allowed review topics:

- Clinical framing
- Decision-support-only clarity
- Workflow clarity
- Limitations and guardrails
- RN/pilot operations feedback

Not allowed:

- Public clinical validation claim
- Final HiRID claim
- Clinical performance guarantee
- Diagnosis/treatment/escalation instruction

## 7. Public-Safe Manifest

This manifest intentionally excludes local_private, raw, row-level, patient-level, timestamp, identifier, and restricted dataset material.

Total public-safe manifest items found: **112**

See JSON file:

`data/evidence_lock/evidence_lock_2026-05-07.json`

## 8. Lock Rule

Before updating the website, investor materials, pilot materials, or regulatory materials, wording should be checked against this evidence lock.

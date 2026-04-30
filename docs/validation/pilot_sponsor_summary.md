# Early Risk Alert AI — Pilot Sponsor Summary

## Purpose

Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform designed to help authorized healthcare professionals identify patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.

Decision support only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.

## Current Evidence Posture

Early Risk Alert AI has retrospective aggregate evidence across:

1. MIMIC-IV v3.1 strict clinical-event cohorts.
2. eICU v2.0 harmonized clinical-event cohorts.

The evidence supports retrospective robustness under a harmonized threshold framework. It does not establish prospective clinical performance or full clinical validation.

## Conservative t=6.0 Summary

- MIMIC-IV strict clinical-event cohorts: 94.0%–94.9% alert reduction, 3.7%–4.4% FPR, 14.1%–16.0% detection, and 4.0 hours of retrospective lead-time context.
- eICU harmonized clinical-event cohorts: 94.25%–95.64% alert reduction, 0.98%–2.02% FPR, 24.66%–64.49% detection, and 3.13–4.83 hours of retrospective lead-time context.

Detection rates are reported separately because event definitions differ across datasets.

## Command-Center Explainability

The pilot interface should visibly show:

- Priority tier.
- Queue rank.
- Primary driver.
- Trend direction.
- Lead-time context.
- Workflow state.

These outputs are review-prioritization context only, not diagnosis or treatment direction.

## Suggested Pilot Success Metrics

- Review burden / alert burden.
- False-positive burden.
- Reviewer understanding of why a case may warrant review.
- Workflow-state completion.
- Safe use of decision-support-only guardrails.

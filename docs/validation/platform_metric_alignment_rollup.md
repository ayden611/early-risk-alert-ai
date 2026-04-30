# Platform Metric Alignment Rollup

## Purpose

This document locks the conservative public metric framing across MIMIC-IV and eICU while preventing confusion between:

1. Patient-level Command Center review scores.
2. Aggregate validation percentages.
3. eICU outcome-proxy checks.
4. eICU harmonized clinical-event labeling.

## Command Center Score Rule

The Command Center should use:

**Review Score: 0-10**

It should not show patient-level values as “99% risk.”  
Validation percentages such as 94.3% or 96.8% are aggregate retrospective metrics, not patient-level risk percentages.

## Conservative t=6.0 Comparison

| Evidence Track | Event Definition | Alert Reduction | FPR | Detection | Lead-Time Context |
|---|---|---:|---:|---:|---:|
| MIMIC-IV strict clinical-event evidence | Strict clinical-event labels / event clusters | 94.0%-94.9% | 3.7%-4.4% | 14.1%-16.0% | 4.0 hrs |
| eICU harmonized clinical-event full cohort | Harmonized clinical-event labeling pass | 94.25% | 0.98% | 24.66% | 4.83 hrs |
| eICU harmonized clinical-event subcohorts B/C | Harmonized clinical-event labeling pass | 95.26%-95.64% | 1.78%-2.02% | 64.03%-64.49% | 3.13-3.69 hrs |
| eICU outcome-proxy check | Mortality/discharge-derived outcome-proxy context | 96.8% | 1.8% | 66.6% | 3.41 hrs |

## Approved Wording

Early Risk Alert AI has retrospective aggregate evidence across MIMIC-IV and eICU. The strongest conservative framing is that both datasets preserve the same threshold-direction behavior: lower thresholds increase detection, while conservative thresholds reduce alert burden and false positives. Detection rates must be interpreted with event-definition context.

## Claim Guardrail

Do not claim unqualified clinical validation or prospective performance.

Approved claims remain:

- Retrospective aggregate evidence.
- De-identified datasets.
- Decision support only.
- Workflow support only.
- Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.

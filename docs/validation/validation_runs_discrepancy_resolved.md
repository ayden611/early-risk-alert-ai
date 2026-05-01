# Validation Runs Page Discrepancy Resolved

## Status

Resolved.

The `/validation-runs` page has been corrected so the eICU outcome-proxy row matches the authoritative terminal/evidence output:

- t=6.0 alert reduction: 96.8%
- t=6.0 FPR: 1.8%
- t=6.0 detection: 66.6%
- t=6.0 lead-time context: 3.41 hours

## Important Separation

The prior issue occurred because the page displayed eICU harmonized clinical-event numbers as if they were the same as the eICU outcome-proxy terminal/evidence output.

The corrected page now separates:

1. MIMIC-IV strict clinical-event evidence.
2. eICU outcome-proxy check.
3. eICU harmonized clinical-event pass.
4. eICU harmonized subcohort B/C robustness checkpoint.

## Guardrail

Detection rates must not be merged across event-definition tracks. These are retrospective aggregate results only and are not prospective clinical validation.

## Public Wording

Approved conservative wording:

Early Risk Alert AI has retrospective aggregate evidence across MIMIC-IV and eICU, with event-definition tracks clearly separated. Alert-reduction and lead-time behavior are directionally consistent, while detection rates should be interpreted only within their respective event-definition context.

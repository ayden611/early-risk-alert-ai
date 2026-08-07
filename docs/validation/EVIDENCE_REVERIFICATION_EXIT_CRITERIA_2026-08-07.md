# ERA Evidence Reverification — Exit Criteria
## 2026-08-07

The current evidence-repair program is complete when all required items
below are satisfied.

## Required

1. A single canonical scoring module is committed and protected by a frozen
   golden input/output CI contract.

2. The retrospective validation pipeline imports and executes the canonical
   scoring module rather than maintaining a copied scoring implementation.

3. MIMIC-IV is rerun from source observations through the canonical scorer.

4. The MIMIC run writes an immutable run-specific artifact containing at
   minimum:
   - dataset/cohort identity
   - input SHA-256
   - row count
   - patient count
   - event definition
   - operating threshold
   - scorer ID
   - scorer specification version
   - scorer source hash
   - repository Git SHA
   - exact numerators and denominators
   - metric definitions
   - generated timestamp

5. The same MIMIC execution produces a separate aggregate-only
   canonical-versus-legacy divergence artifact describing scorer disagreement
   without modifying historical evidence.

6. eICU full-cohort evaluation is rerun through the same canonical scorer and
   written to an equivalent immutable run-specific artifact.

7. Public validation pages and the Model Card are rebuilt only from accepted
   canonical run artifacts and explicitly identify scorer path/version/hash
   and repository SHA.

8. The evidence integrity hold remains active until items 1 through 7 are
   accepted.

## Command Center demonstration gate

Before the next external Command Center demonstration, synthetic vital-sign
observations must be scored through the canonical module rather than using
predetermined demonstration scores.

## Not blockers for the current evidence-repair exit

The following are deferred unless new evidence makes them necessary:

- refactoring `_risk_from_vitals`;
- refactoring `detect_risk`;
- unrelated demo/API scorer cleanup;
- historical eICU subcohort archaeology;
- reconciliation of obsolete historical subcohort outputs that will no
  longer be cited;
- broader architecture cleanup not needed for the canonical evidence path.

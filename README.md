# Early Risk Alert AI

## Current Platform Summary

Early Risk Alert AI is a **pilot-stage, pre-commercial** healthcare-professional-facing decision-support and workflow-support software platform.

The platform is designed to organize **already-acquired numeric vital-sign observations** into pattern-based review notifications and explainable review context for authorized health care professionals. It supports independent professional review and monitored-workflow prioritization.

### Input Boundary

The platform may display already-acquired numeric observations, including:

- heart-rate observations
- non-invasive blood-pressure observations
- SpO₂ observations
- respiratory-rate observations
- temperature observations, where available
- relevant review-context and workflow-state information

The platform does not directly acquire physiologic signals from patients and is not intended to process raw continuous waveform data.

### Output Boundary

The platform may present:

- pattern-based review notifications
- contributing numeric variables
- available trend context
- review-queue organization
- workflow-state visibility
- data-freshness and limitation context

### Decision-Support Boundary

**Decision support only.** Early Risk Alert AI does not diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.

The platform is not intended to replace bedside monitoring systems, immediate physiologic alarms, code-blue systems, cardiac-arrest alarms, or hospital emergency-response systems. Healthcare professionals independently review the underlying numeric observations, relevant context, and patient record before determining whether any action is appropriate.

### Evidence Boundary

Current evidence is **retrospective, de-identified, and aggregate only**.

Internal retrospective aggregate analyses have used critical-care datasets including MIMIC-IV and eICU. Performance characteristics vary by dataset, event definition, and operating threshold. Metrics must be interpreted within their respective evidence-track definitions and must not be merged across tracks.

These analyses do not constitute prospective clinical validation, diagnostic-performance claims, or proof of patient-outcome improvement.

**HiRID status:** Access approved; local/private retrospective aggregate evaluation pending.

### Pilot and Integration Boundary

Current pilot portals are limited to de-identified evaluation data. No live PHI flows are active.

Future integration planning may consider standard health-data exchange protocols, such as HL7 and FHIR, where appropriate. Live hospital integration is not currently active.

### Regulatory Status

Early Risk Alert AI is preparing a Section 513(g) Request for Information to seek FDA information regarding classification and applicable regulatory requirements.

A Section 513(g) request is not FDA clearance, FDA approval, or a determination of clinical performance. Any future FDA Q-Submission or pre-submission engagement will be handled as a separate regulatory interaction.

---

**Decision support only.** Early Risk Alert AI does not diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.

<!-- ERA_INSURANCE_READINESS_STATUS_START -->
## Insurance Readiness

Active Technology Errors & Omissions / Cyber Liability coverage is in place as part of the platform's pilot-readiness governance controls.

Detailed policy documents, certificate records, premiums, private contact details, and insurer records are retained privately and are not published in this repository.

Insurance readiness does not alter the platform boundary: decision support only; no diagnosis, treatment direction, replacement of clinician judgment, or autonomous escalation.
<!-- ERA_INSURANCE_READINESS_STATUS_END -->

## Future Integration Discovery Status

Early Risk Alert AI is not currently connected to Epic or any live hospital EHR environment.

Future institutional discussions may consider standards-based, read-only integration pathways for already-recorded numeric observations, including FHIR where applicable. No live PHI flow, production endpoint, write-back function, autonomous escalation function, or interruptive EHR alert is active.

Retrospective bulk-data analysis is treated as a separate future discovery pathway and is not represented as an active operational integration.

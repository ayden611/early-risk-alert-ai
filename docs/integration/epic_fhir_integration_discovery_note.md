# Epic / FHIR Integration-Discovery Note

## Status
**Future integration planning only**

Early Risk Alert AI is not currently connected to Epic or any live hospital electronic health record environment.

No production endpoint, live protected health information flow, write-back function, autonomous escalation function, or interruptive EHR alert is active.

## Current Platform Boundary
Early Risk Alert AI remains a pilot-stage, pre-commercial healthcare-professional-facing decision-support and workflow-support software platform.

The platform organizes already-acquired numeric vital-sign observations into pattern-based review notifications and explainable review context for independent clinician review.

It does not diagnose, direct treatment, replace clinician judgment, independently trigger escalation, or replace bedside monitoring systems, immediate physiologic alarms, or hospital emergency-response systems.

## Purpose of This Note
This note documents potential standards-based integration pathways for future institutional discussions.

It is not:

- a production-integration specification;
- a claim of Epic compatibility certification;
- a claim of active Epic integration;
- a live-data implementation plan;
- a regulatory classification conclusion;
- authorization to process PHI;
- authorization to modify the current application.

## Future Discovery Pathways

### Pathway 1 — Standalone Controlled Workflow Demonstration
Potential initial configuration:

- standalone review dashboard;
- synthetic, retrospective, or appropriately approved de-identified evaluation data;
- no hospital write-back;
- no production endpoint;
- no autonomous escalation.

This remains the preferred initial discovery posture.

### Pathway 2 — Future Hospital-Approved Read-Only FHIR Access
Potential future discussion topic:

- authorized read-only access to already-recorded numeric observations;
- hospital-approved authentication and authorization controls;
- no record modification;
- no write-back;
- no autonomous escalation;
- no assumption that read-only access eliminates privacy, security, contractual, or regulatory review.

### Pathway 3 — Retrospective Bulk-Data Export
Potential future discussion topic:

- approved retrospective export of larger datasets;
- separate from operational workflow integration;
- appropriate only where the institution authorizes the data-use scope and technical method.

### Pathway 4 — Future In-Workflow Presentation
Potential later-stage discussion topic only:

- EHR-adjacent or in-workflow presentation;
- institution-selected integration method;
- separate privacy, security, usability, contractual, and regulatory assessment;
- no current implementation claim.

## Conceptual FHIR Resources for Discovery Discussions

| FHIR resource | Conceptual purpose | Current implementation status |
|---|---|---|
| Observation | Already-recorded numeric vital-sign observations | Conceptual mapping only |
| Patient | Authorized patient-context association where permitted | Conceptual mapping only |
| Encounter | Authorized care-setting or encounter context where permitted | Conceptual mapping only |

## Discovery Questions for Hospital IT and Clinical-Informatics Review
1. Which monitored-care workflows should be evaluated first?
2. Would the initial evaluation remain retrospective and de-identified?
3. Would the organization prefer a standalone demonstration before an EHR-connected pathway?
4. Which approved interface types are available for already-recorded numeric observations?
5. Which authentication, authorization, audit, privacy, and security controls are required?
6. Which stakeholders must approve any future data access?
7. Would any future evaluation require a BAA, data-use agreement, interface agreement, security assessment, or additional regulatory review?
8. Should retrospective bulk-data analysis remain separate from any future operational-workflow discussion?

## Controlled Language
Approved wording:

> Future institutional discussions may consider standards-based, read-only integration pathways for already-recorded numeric observations, including FHIR where applicable. No live EHR integration is currently active.

Do not claim:

- active Epic integration;
- live hospital deployment;
- real-time disease prediction;
- impending-disease detection;
- autonomous escalation;
- automatic IT-security clearance;
- guaranteed non-device status;
- replacement of immediate physiologic alarms or emergency-response systems.

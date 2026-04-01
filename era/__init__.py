from __future__ import annotations

import json
import os
import random
import secrets
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List

from werkzeug.security import check_password_hash

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_from_directory,
    session,
    url_for,
)

from era.web.command_center import COMMAND_CENTER_HTML


INFO_EMAIL = "info@earlyriskalertai.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"
FOUNDER_ROLE = "Founder & CEO, Early Risk Alert AI"


PILOT_VERSION = os.getenv("PILOT_VERSION", "stable-pilot-1.0.4")
PILOT_BUILD_STATE = "Locked Stable Pilot Build"
INTENDED_USE_STATEMENT = (
    "Early Risk Alert AI is an HCP-facing decision-support and workflow-support software platform intended to assist authorized health care professionals in identifying patients who may warrant further clinical evaluation, supporting patient prioritization, and improving command-center operational awareness. It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation."
)
PILOT_SUPPORT_LANGUAGE = [
    "supports monitored patient visibility and supportive review prioritization",
    "assists health care professionals in prioritizing monitored patients",
    "provides explainable review-support context",
    "supports command-center workflow awareness",
    "helps identify patients who may warrant further review",
]
PILOT_SUPPORTED_INPUTS = [
    "structured patient medical information",
    "trended vital-sign summaries",
    "monitored patient context",
    "workflow state and review context",
    "approved medical information available to the HCP",
]
PILOT_SUPPORTED_OUTPUTS = [
    "patient prioritization support",
    "review-support context",
    "explainable contributing factors",
    "trend and freshness information",
    "workflow-support visibility",
    "supportive workflow note for further clinical evaluation",
]
PILOT_AVOID_CLAIMS = [
    "detects deterioration autonomously",
    "identifies who needs immediate escalation",
    "predicts who will clinically crash",
    "directs bedside intervention",
    "determines which patients need escalation now",
]
PILOT_CHANGE_CONTROL = [
    "Freeze one intended-use statement everywhere in the pilot build.",
    "Keep claims narrow and supportive rather than directive.",
    "Make review basis, confidence, limitations, freshness, and unknowns visible for each patient.",
    "Maintain role scoping, unit scoping, and audit visibility across routes.",
    "Record each approved pilot change in release notes before deployment.",
]
PILOT_LIMITATIONS_TEXT = [
    "Output is decision support only and not a diagnosis.",
    "Incomplete or delayed data may affect outputs.",
    "Clinicians must independently review the patient and current clinical context.",
    "Hospital policy governs escalation, response timing, and treatment decisions.",
    "The platform is not intended to diagnose, direct treatment, or independently trigger escalation.",
]
PILOT_RISK_REGISTER = [
    {"id": "R-001", "area": "Claims", "risk": "Autonomous or directive claims could shift the platform toward a higher-risk regulatory posture.", "mitigation": "Freeze one intended-use statement and review UI, sales copy, and decks before each release.", "owner": "Founder/Product", "status": "Open"},
    {"id": "R-002", "area": "Explainability", "risk": "Users may over-rely on outputs if the basis, confidence, limitations, and unknowns are not visible.", "mitigation": "Keep review basis, freshness, confidence, limitations, and unknowns visible in the patient detail workflow.", "owner": "Engineering", "status": "In Place"},
    {"id": "R-003", "area": "Workflow Separation", "risk": "Operational buttons could be misread as machine-issued clinical directives.", "mitigation": "Keep acknowledge/assign/escalate/resolve framed as workflow state and user action logging only.", "owner": "Product", "status": "In Place"},
    {"id": "R-004", "area": "Scope Control", "risk": "Improper access scope could expose records outside the intended role or unit context.", "mitigation": "Maintain role restrictions, unit restrictions, and filtered workflow/audit visibility across routes.", "owner": "Engineering", "status": "In Place"},
    {"id": "R-005", "area": "Change Control", "risk": "Pilot drift could occur if live edits are made without a stable version marker and release notes.", "mitigation": "Keep one stable pilot version, maintain release notes, and update a simple change log for each approved bundle.", "owner": "Founder/Engineering", "status": "Open"},
]
PILOT_VNV_LITE = [
    {"id": "VV-001", "check": "Access context respects login, role, and unit scope.", "method": "Route check across /login, /pilot-access, /api/access-context, and filtered views.", "evidence": "Implemented in current pilot bundle.", "status": "Pass"},
    {"id": "VV-002", "check": "Workflow actions remain operational and auditable rather than directive.", "method": "Exercise ack/assign/escalate/resolve and confirm workflow/audit separation in API and UI.", "evidence": "Workflow and audit routes updated in current pilot bundle.", "status": "Pass"},
    {"id": "VV-003", "check": "Explainability fields are visible for patient review.", "method": "Open patient drawer and verify review basis, confidence, limitations, freshness, what changed, and unknowns.", "evidence": "Visible in patient detail drawer.", "status": "Pass"},
    {"id": "VV-004", "check": "Threshold and trend routes return scoped data without breaking the command center.", "method": "Call /api/thresholds and /api/trends/<patient_id> under pilot roles.", "evidence": "Routes available in current pilot bundle.", "status": "Pass"},
    {"id": "VV-005", "check": "Pilot governance artifacts are visible from the app.", "method": "Open /pilot-docs and verify intended use, risk register, V&V-lite, and release notes render.", "evidence": "Added in current pilot bundle.", "status": "Pass"},
]
PILOT_RELEASE_NOTES = [
    {
        "version": "stable-pilot-1.0.0",
        "date": "2026-03-25",
        "summary": "Locked stable pilot-safe positioning bundle.",
        "changes": [
            "Frozen intended-use statement added across platform context and pilot documentation.",
            "Safer support-language and supportive-output framing tightened in command center copy and form routes.",
            "Risk register, V&V-lite sheet, release notes, and pilot docs route added.",
            "Role/unit scoping, workflow/audit separation, and explainability-first presentation retained."
        ],
    },
    {
        "version": "stable-pilot-1.0.1",
        "date": "2026-03-26",
        "summary": "Expanded pilot governance package and hospital-review readiness materials.",
        "changes": [
            "Complaint / issue log, change approvals, cybersecurity summary, access policies, retention policy, pilot scope, training sheet, and dated validation evidence added.",
            "Approved and banned claims lists added to the governance set.",
            "Pilot docs and platform-positioning routes expanded for governance visibility."
        ],
    },
    {
        "version": "stable-pilot-1.0.2",
        "date": "2026-03-26",
        "summary": "Strengthened controlled pilot governance, document control, and supporting process detail.",
        "changes": [
            "Expanded complaint handling with investigation, corrective action, and closure tracking fields.",
            "Expanded change approval evidence with approver, date, reason, what changed, and impact assessment.",
            "Added document-control index and claims-control sheet.",
            "Expanded cybersecurity, access, data handling, training, and validation evidence details.",
            "Softened remaining review-language labels to keep the pilot posture supportive rather than directive."
        ],
    },
    {
        "version": "stable-pilot-1.0.3",
        "date": "2026-03-31",
        "summary": "Softened remaining predictive UI labels and expanded documented security / insurance-readiness controls.",
        "changes": [
            "Renamed remaining predictive command-center labels to review-oriented language.",
            "Replaced forecast percentages and exact time windows with review-attention levels and monitored-trend language.",
            "Added advisory structure with Technical Infrastructure & Security Advisor support.",
            "Added documented MFA, endpoint security, immutable backup, incident response, business continuity, patching, tabletop, and security training controls for pilot review."
        ],
    },
    {
        "version": "stable-pilot-1.0.4",
        "date": "2026-04-01",
        "summary": "Stability-first polish bundle with readiness evidence, route checks, and site-packet structure.",
        "changes": [
            "Added route and document status visibility for core command-center stability checks.",
            "Added operational readiness evidence logs for MFA, backup/restore, patching, access reviews, tabletop exercises, and training acknowledgments.",
            "Added named support owners, technical infrastructure coverage, and site-specific pilot packet placeholders.",
            "Added command-center UI visibility for pilot docs, readiness evidence, and site-packet structure without changing core workflow behavior."
        ],
    },
]

PILOT_APPROVED_CLAIMS = [
    "assists authorized health care professionals in identifying patients who may warrant further clinical evaluation",
    "supports patient prioritization across monitored patients",
    "provides explainable risk-support context",
    "supports command-center workflow awareness and operational visibility",
    "supports monitored patient visibility and reviewable trend context",
    "supports further clinical evaluation through reviewable contributing factors, confidence, limitations, and data freshness",
    "supports controlled hospital pilot evaluation and workflow-aware review by authorized users",
]

PILOT_BANNED_CLAIMS = [
    "detects deterioration autonomously",
    "identifies who needs immediate escalation",
    "predicts who will clinically crash",
    "directs bedside intervention",
    "determines which patients need escalation now",
    "replaces clinician judgment",
    "provides a diagnosis",
    "triggers escalation independently",
    "forces immediate escalation decisions",
    "acts as the primary clinical decision-maker",
]

PILOT_CLAIMS_CONTROL_SHEET = [
    {
        "claim": "Assists authorized health care professionals in identifying patients who may warrant further clinical evaluation",
        "status": "Approved",
        "category": "Intended use / supportive output",
        "owner": "Founder/Product",
        "last_reviewed": "2026-03-26",
        "rationale": "Keeps the HCP as decision-maker and frames output as supportive review context."
    },
    {
        "claim": "Supports patient prioritization across monitored patients",
        "status": "Approved",
        "category": "Workflow-support / prioritization",
        "owner": "Founder/Product",
        "last_reviewed": "2026-03-26",
        "rationale": "Positions the platform as prioritization support rather than autonomous action."
    },
    {
        "claim": "Provides explainable risk-support context",
        "status": "Approved",
        "category": "Explainability / transparency",
        "owner": "Founder/Product",
        "last_reviewed": "2026-03-26",
        "rationale": "Supports independent HCP review of the basis for outputs."
    },
    {
        "claim": "Supports command-center workflow awareness and operational visibility",
        "status": "Approved",
        "category": "Operational awareness",
        "owner": "Founder/Product",
        "last_reviewed": "2026-03-26",
        "rationale": "Frames the system as workflow-support rather than diagnosis or treatment direction."
    },
    {
        "claim": "Detects deterioration autonomously",
        "status": "Banned",
        "category": "Automation / device-risk",
        "owner": "Founder/Product",
        "last_reviewed": "2026-03-26",
        "rationale": "Would overstate autonomy and weaken the HCP-facing support posture."
    },
    {
        "claim": "Identifies who needs immediate escalation",
        "status": "Banned",
        "category": "Time-critical directive",
        "owner": "Founder/Product",
        "last_reviewed": "2026-03-26",
        "rationale": "Sounds like independent clinical direction and should not be used in pilot materials."
    },
    {
        "claim": "Predicts who will clinically crash",
        "status": "Banned",
        "category": "Outcome prediction",
        "owner": "Founder/Product",
        "last_reviewed": "2026-03-26",
        "rationale": "Creates a stronger device-side impression than the intended supportive pilot posture."
    },
    {
        "claim": "Directs bedside intervention",
        "status": "Banned",
        "category": "Treatment direction",
        "owner": "Founder/Product",
        "last_reviewed": "2026-03-26",
        "rationale": "Would position the platform as giving treatment instructions rather than review support."
    },
]

PILOT_COMPLAINT_ISSUE_LOG = [
    {
        "id": "CI-001",
        "date_opened": "2026-03-25",
        "source": "Internal pilot review",
        "category": "Claims / Messaging",
        "issue": "Legacy predictive or directive language could remain in customer-facing materials after UI updates.",
        "severity": "Moderate",
        "owner": "Founder/Product",
        "investigation_summary": "Reviewed platform screens, website copy, forms, and demo script against the frozen intended-use statement and claims controls.",
        "corrective_action": "Replace non-approved language, re-check command-center wording, and require pre-release wording review.",
        "status": "Open",
        "escalation_level": "Review before external release",
        "next_step": "Review website, video, deck, and templates against the frozen intended-use statement.",
        "closure_date": ""
    },
    {
        "id": "CI-002",
        "date_opened": "2026-03-25",
        "source": "Platform validation review",
        "category": "Explainability",
        "issue": "If review basis, limitations, freshness, or unknowns fail to render, users could over-rely on summary outputs.",
        "severity": "High",
        "owner": "Engineering",
        "investigation_summary": "Confirmed that the patient detail workflow must surface review basis, confidence, limitations, freshness, and unknowns on each reviewed patient.",
        "corrective_action": "Validate patient detail drawer and guardrail panels before each approved bundle and log failures as release blockers.",
        "status": "Monitoring",
        "escalation_level": "Immediate engineering review",
        "next_step": "Verify visibility checks during each release validation cycle.",
        "closure_date": ""
    },
    {
        "id": "CI-003",
        "date_opened": "2026-03-25",
        "source": "Operational workflow review",
        "category": "Workflow separation",
        "issue": "Workflow buttons could be interpreted as machine-issued clinical directives if labels drift from approved wording.",
        "severity": "High",
        "owner": "Product/Engineering",
        "investigation_summary": "Reviewed workflow labels, audit entries, and recommendation text to confirm operational handling remains separate from supportive clinical review context.",
        "corrective_action": "Keep ACK, Assign, Escalate, and Resolve framed as workflow logging only and require wording review on each release.",
        "status": "Implemented",
        "escalation_level": "Product sign-off required",
        "next_step": "Audit workflow labels and patient-detail copy before external pilot demonstrations.",
        "closure_date": "2026-03-26"
    },
    {
        "id": "CI-004",
        "date_opened": "2026-03-26",
        "source": "Hospital readiness review",
        "category": "Documentation control",
        "issue": "Governance artifacts may drift across versions without a single document-control index.",
        "severity": "Moderate",
        "owner": "Founder/Admin",
        "investigation_summary": "Identified need for a master index covering governance documents, owners, versions, and pilot applicability.",
        "corrective_action": "Create and maintain a document-control index in pilot docs and governance routes.",
        "status": "Implemented",
        "escalation_level": "Change approval required",
        "next_step": "Review index entries whenever pilot documents or approved claims are updated.",
        "closure_date": "2026-03-26"
    },
]

PILOT_ESCALATION_PROCESS = [
    {"stage": "1. Record issue", "trigger": "A complaint, defect, security concern, claims drift, or validation failure is identified.", "owner": "Issue reporter / admin", "target_time": "Same business day", "action": "Create or update a complaint / issue log entry with date opened, source, severity, and owner."},
    {"stage": "2. Triage severity", "trigger": "Issue is logged and scoped.", "owner": "Founder/Product", "target_time": "Within 1 business day", "action": "Assign severity, determine if the issue affects claims, access, data handling, workflow separation, or hospital pilot operation."},
    {"stage": "3. Investigate", "trigger": "Issue has an assigned owner and severity.", "owner": "Assigned owner", "target_time": "Within 2 business days for moderate/high issues", "action": "Document investigation summary, root cause, impacted routes/screens/docs, and expected corrective action."},
    {"stage": "4. Escalate if required", "trigger": "High-severity issue, security incident, pilot outage, or claims/control drift affecting hospital-facing use.", "owner": "Founder/Product", "target_time": "Immediate for high severity", "action": "Pause affected pilot use if needed, notify required stakeholders, and require change approval before release restoration."},
    {"stage": "5. Correct and verify", "trigger": "Corrective action is implemented.", "owner": "Engineering / Product", "target_time": "Before next approved release", "action": "Validate the fix, capture evidence, update release notes/change approval, and record pass/fail verification."},
    {"stage": "6. Close", "trigger": "Evidence supports resolution and no open blockers remain.", "owner": "Founder/Admin", "target_time": "After verification", "action": "Record closure date, summarize outcome, and retain issue history in the pilot governance set."},
]

PILOT_CHANGE_APPROVAL_LOG = [
    {
        "version": "stable-pilot-1.0.0",
        "approval_date": "2026-03-25",
        "approver_name": "Milton Munroe",
        "reason_for_approval": "Required to align UI, forms, pilot docs, and platform positioning to the HCP-facing support posture.",
        "what_changed": "Frozen intended-use language, explainability guardrails, workflow separation, and governance docs baseline.",
        "impact_assessment": "Reduced risk of overstatement and improved consistency across pilot materials.",
        "status": "Approved"
    },
    {
        "version": "stable-pilot-1.0.1",
        "approval_date": "2026-03-26",
        "approver_name": "Milton Munroe",
        "reason_for_approval": "Required to strengthen pilot operating discipline and hospital-facing review readiness without shifting claims into device territory.",
        "what_changed": "Expanded complaint log, change approval log, cybersecurity summary, access policies, retention policy, pilot scope, training sheet, and dated validation evidence.",
        "impact_assessment": "Improved hospital-review credibility and governance coverage around the stable pilot build.",
        "status": "Approved"
    },
    {
        "version": "stable-pilot-1.0.2",
        "approval_date": "2026-03-26",
        "approver_name": "Milton Munroe",
        "reason_for_approval": "Required to formalize complaint handling process detail, document control, validation backing packet, and claims-control discipline.",
        "what_changed": "Added document-control index, expanded complaint handling fields, expanded cybersecurity/access/data policies, strengthened validation evidence fields, and softened remaining directive wording.",
        "impact_assessment": "Further reduced claims drift risk, improved traceability, and strengthened the credibility of the HCP-facing pilot posture for hospital review.",
        "status": "Approved"
    },
]

PILOT_CYBERSECURITY_SUMMARY = [
    {
        "domain": "Access control",
        "control": "Authenticated pilot access with role-based permissions and unit-scoped visibility",
        "summary": "User login, session access, and backend filtering limit access to approved roles and assigned units.",
        "evidence": "Pilot access screens, access-context route, unit filtering, and role-aware workflow routes.",
        "status": "Implemented"
    },
    {
        "domain": "Password policy",
        "control": "Protected admin credentials and controlled pilot account management",
        "summary": "Administrative access should use configured credentials, limited approved emails, and controlled account provisioning.",
        "evidence": "Admin login validation and pilot-account governance review.",
        "status": "Structured control"
    },
    {
        "domain": "Patching approach",
        "control": "Approved release and change logging",
        "summary": "Pilot updates are intended to be deployed only through approved bundles tracked in release notes and change approval logs.",
        "evidence": "Release notes, change approval log, stable build marker.",
        "status": "Implemented"
    },
    {
        "domain": "Vulnerability handling",
        "control": "Complaint / issue log and escalation process",
        "summary": "Security, reliability, or access issues are logged, triaged, escalated to owner, and verified before closure.",
        "evidence": "Complaint / issue log, escalation process, and corrective-action workflow.",
        "status": "Implemented"
    },
    {
        "domain": "Backup / restore",
        "control": "Operational backup and recovery summary",
        "summary": "Pilot data artifacts, configuration, and operational files are covered by documented backup, restore, and immutability requirements before hospital deployment.",
        "evidence": "Backup / restore standard, closeout documentation, and restore-test tracking requirements.",
        "status": "Documented and available"
    },
    {
        "domain": "Incident handling",
        "control": "Documented incident response path",
        "summary": "Security or operational incidents are expected to follow documented triage, escalation, remediation, verification, and closure steps.",
        "evidence": "Escalation process and complaint / issue handling fields.",
        "status": "Implemented"
    },
    {
        "domain": "Pilot pause / breach response",
        "control": "Controlled pilot hold and stakeholder notification",
        "summary": "If a hospital pilot is paused for security, reliability, or governance reasons, affected users should be notified, access reviewed, and restoration approved before pilot resumption.",
        "evidence": "Escalation process stage for pause/restore decision-making.",
        "status": "Documented and available"
    },
]

PILOT_USER_PROVISIONING_POLICY = [
    {
        "step": "Provision request",
        "policy": "User access is provisioned only for approved pilot participants with a defined role and scope.",
        "owner": "Founder/Admin",
        "sla": "Before pilot access is granted",
        "evidence": "Pilot access screen, role selection, assigned-unit control"
    },
    {
        "step": "Role assignment",
        "policy": "Each user receives the minimum role and unit scope required for pilot evaluation.",
        "owner": "Founder/Admin",
        "sla": "At account creation",
        "evidence": "Session role and unit scoping reflected in access-context route"
    },
    {
        "step": "Unit-scope review",
        "policy": "Assigned unit scope should be reviewed whenever pilot responsibilities change or when broader scope is requested.",
        "owner": "Founder/Admin",
        "sla": "Within 1 business day of scope request",
        "evidence": "Pilot account review and updated access assignment"
    },
    {
        "step": "Provisioning owner",
        "policy": "Only the designated admin / founder or approved pilot administrator may create or modify pilot accounts.",
        "owner": "Founder/Admin",
        "sla": "Per documented approval path",
        "evidence": "Controlled pilot-account list and login governance"
    },
    {
        "step": "Deprovision",
        "policy": "Pilot access should be disabled or removed when a pilot ends, a user leaves the pilot, or access is no longer required.",
        "owner": "Founder/Admin",
        "sla": "Same business day when feasible; no later than next business day",
        "evidence": "Pilot account removal / session invalidation / access policy review"
    },
    {
        "step": "Pilot closeout",
        "policy": "At pilot end, users should be reviewed, accounts disabled or archived, and remaining access documented as part of pilot closeout.",
        "owner": "Founder/Admin",
        "sla": "At pilot closeout",
        "evidence": "Pilot closeout checklist and account review record"
    },
]

PILOT_DATA_RETENTION_POLICY = [
    {
        "topic": "Pilot data stored",
        "policy": "Pilot request records, workflow artifacts, audit logs, threshold settings, trend histories, and governance documents may be stored for pilot operation and review.",
        "owner": "Founder/Product",
        "note": "Hospital-specific storage and handling expectations should be confirmed before deployment.",
        "pilot_applicability": "All controlled pilots"
    },
    {
        "topic": "Retention period",
        "policy": "Pilot data should be retained only for the approved pilot review period and business, legal, or contractual needs agreed for the pilot.",
        "owner": "Founder/Product",
        "note": "Hospital-specific retention schedules should be confirmed in writing.",
        "pilot_applicability": "All controlled pilots"
    },
    {
        "topic": "Return / export",
        "policy": "If hospital or partner data is supplied for pilot use, return / export terms should be documented in the pilot agreement, including format and timing expectations.",
        "owner": "Founder/Product",
        "note": "Customer-specific handling should be confirmed in writing.",
        "pilot_applicability": "Pilots using customer-supplied data"
    },
    {
        "topic": "Deletion",
        "policy": "Pilot data should be deleted or archived in accordance with the agreed pilot closeout process and documented by date, owner, and scope.",
        "owner": "Founder/Product",
        "note": "Deletion or archive status should be recorded in pilot closeout materials.",
        "pilot_applicability": "All controlled pilots"
    },
    {
        "topic": "Pilot closeout",
        "policy": "At pilot closeout, the team should document what data is retained, exported, returned, archived, or deleted and who approved the outcome.",
        "owner": "Founder/Product",
        "note": "Closeout summary should align with the pilot agreement and any hospital requirements.",
        "pilot_applicability": "All controlled pilots"
    },
]

PILOT_SCOPE_DOCUMENT = [
    {"section": "Purpose", "summary": "Controlled pilot evaluation of an HCP-facing decision-support and workflow-support platform for monitored patient visibility, patient prioritization support, explainable review context, and command-center operational awareness."},
    {"section": "Intended users", "summary": "Authorized health care professionals, approved hospital stakeholders, clinical operations teams, and approved pilot reviewers."},
    {"section": "Pilot boundaries", "summary": "The platform does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation."},
    {"section": "Operational scope", "summary": "Role-based access, unit-scoped visibility, workflow logging, explainability panels, audit visibility, and governance documentation are included in the current pilot bundle."},
    {"section": "Pilot review materials", "summary": "The pilot bundle includes claims controls, complaint handling, change approvals, cybersecurity summary, MFA and access-control requirements, backup / restore requirements, incident-response and business-continuity materials, data handling policy, training sheet, validation evidence, and document-control index."},
    {"section": "Hospital responsibilities", "summary": "Hospitals retain oversight of clinical decision-making, local policy, and any site-specific access, data, and operational requirements."},
]

PILOT_TRAINING_USE_INSTRUCTIONS = [
    {"step": "1. Intended users", "instruction": "Use is limited to authorized health care professionals, approved hospital stakeholders, clinical operations teams, and approved pilot reviewers.", "owner": "Founder/Admin"},
    {"step": "2. Access the pilot", "instruction": "Log in with an approved pilot account and confirm role and assigned unit scope before review.", "owner": "Founder/Admin"},
    {"step": "3. Understand the command center", "instruction": "Patient lists and command-center panels support monitored patient visibility, review priority, trend review, workflow status, and audit visibility across the visible scope.", "owner": "Pilot trainer"},
    {"step": "4. Review the basis", "instruction": "Before any workflow action, review patient summary, contributing factors, confidence, limitations, freshness, what changed, and what the software does not know.", "owner": "Clinical reviewer"},
    {"step": "5. Interpret workflow controls correctly", "instruction": "ACK, Assign, Escalate, and Resolve record workflow handling and audit history only; they are not machine-issued clinical orders.", "owner": "Clinical reviewer"},
    {"step": "6. What not to use the platform for", "instruction": "Do not use the platform as a diagnosis engine, treatment director, or independent escalation trigger.", "owner": "Clinical reviewer"},
    {"step": "7. Follow hospital policy", "instruction": "Any clinical action, escalation timing, or treatment decision remains governed by licensed clinician judgment and hospital policy.", "owner": "Clinical reviewer"},
    {"step": "8. Escalate product issues", "instruction": "Report reliability, access, claims, or governance concerns through the complaint / issue handling process so they can be triaged and documented.", "owner": "All pilot users"},
    {"step": "9. Follow security operating requirements", "instruction": "Administrative and support personnel should follow documented access-control, MFA, endpoint-security, patching, backup / restore, incident-response, and training requirements for the pilot environment.", "owner": "Founder/Admin"},
]

PILOT_ADVISORY_STRUCTURE = [
    {
        "name": "Milton Munroe",
        "title": "Founder & CEO, Early Risk Alert AI",
        "scope": "Product leadership, pilot operations coordination, release control, hospital-facing pilot management, and governance ownership.",
        "status": "In place"
    },
    {
        "name": "Uche Anosike",
        "title": "Technical Infrastructure & Security Advisor",
        "scope": "Infrastructure architecture, platform security posture, environment configuration, backup / recovery planning, patch-management review, and deployment-readiness guidance for controlled pilot environments.",
        "status": "In place"
    },
]

PILOT_SECURITY_PROGRAM_DOCUMENTS = [
    {
        "document_name": "Access Control & MFA Standard",
        "version": "1.0",
        "owner": "Founder/Admin",
        "last_updated": "2026-03-31",
        "status": "Approved",
        "pilot_applicability": "Administrative access and pilot deployment review"
    },
    {
        "document_name": "Endpoint Security & EDR Requirement Summary",
        "version": "1.0",
        "owner": "Founder/Admin",
        "last_updated": "2026-03-31",
        "status": "Approved",
        "pilot_applicability": "Administrative and support endpoints"
    },
    {
        "document_name": "Backup, Restore & Immutability Standard",
        "version": "1.0",
        "owner": "Founder/Admin",
        "last_updated": "2026-03-31",
        "status": "Approved",
        "pilot_applicability": "Controlled pilot data protection and closeout"
    },
    {
        "document_name": "Incident Response Plan",
        "version": "1.0",
        "owner": "Founder/Admin",
        "last_updated": "2026-03-31",
        "status": "Approved",
        "pilot_applicability": "All controlled pilots"
    },
    {
        "document_name": "Business Continuity & Disaster Recovery Plan",
        "version": "1.0",
        "owner": "Founder/Admin",
        "last_updated": "2026-03-31",
        "status": "Approved",
        "pilot_applicability": "All controlled pilots"
    },
    {
        "document_name": "Patch Management Standard",
        "version": "1.0",
        "owner": "Founder/Admin",
        "last_updated": "2026-03-31",
        "status": "Approved",
        "pilot_applicability": "All controlled pilots"
    },
    {
        "document_name": "Security Awareness & Phishing Training Standard",
        "version": "1.0",
        "owner": "Founder/Admin",
        "last_updated": "2026-03-31",
        "status": "Approved",
        "pilot_applicability": "Administrative and support personnel"
    },
    {
        "document_name": "Tabletop Exercise Record Template",
        "version": "1.0",
        "owner": "Founder/Admin",
        "last_updated": "2026-03-31",
        "status": "Approved",
        "pilot_applicability": "Incident response and business continuity review"
    },
]

PILOT_INSURANCE_READINESS_CONTROLS = [
    {
        "control": "Phishing-Resistant MFA",
        "requirement": "Multi-factor authentication is required for administrative accounts, remote access, and administrative consoles. Phishing-resistant methods are the preferred configuration where applicable before hospital deployment.",
        "documentation": "Documented in access-control and deployment requirements.",
        "status": "Documented requirement"
    },
    {
        "control": "Endpoint Detection & Response (EDR)",
        "requirement": "Administrative and support endpoints used to manage the pilot environment are expected to use continuously monitored endpoint security tooling with isolation capability before hospital deployment.",
        "documentation": "Documented in endpoint-security requirement summary.",
        "status": "Documented requirement"
    },
    {
        "control": "Immutable Backups",
        "requirement": "Automated off-site or separate-cloud backups with immutability expectations and restore-test tracking are structured for pilot deployment where applicable.",
        "documentation": "Formalized in backup / restore policy and closeout documentation.",
        "status": "Structured and documented"
    },
    {
        "control": "Documented Incident Response Plan",
        "requirement": "A written incident response path with triage, escalation, remediation, verification, notification, and closure steps is documented and available for pilot review.",
        "documentation": "Documented and available.",
        "status": "In place"
    },
    {
        "control": "Business Continuity & Disaster Recovery",
        "requirement": "Recovery objectives, service continuity expectations, pause / restore controls, and backup / restore responsibilities are documented for pilot operations.",
        "documentation": "Documented and available.",
        "status": "Documented and available"
    },
    {
        "control": "Patch Management",
        "requirement": "A formal process for identifying, prioritizing, and remediating critical vulnerabilities within a defined window is structured and documented for pilot environments.",
        "documentation": "Structured in patch management standard.",
        "status": "Structured and documented"
    },
    {
        "control": "Employee Security Awareness Training",
        "requirement": "Cybersecurity awareness training and phishing-awareness requirements for administrative and support personnel are documented for the pilot operating environment.",
        "documentation": "Documented in training requirement summary.",
        "status": "Documented requirement"
    },
    {
        "control": "Tabletop Exercise Readiness",
        "requirement": "Incident-response tabletop exercise procedures and recording templates are documented for pilot security review and evidence tracking.",
        "documentation": "Documented and available.",
        "status": "Documented and available"
    },
]

PILOT_SUPPORT_OWNERS = [
    {
        "area": "Product & Pilot Operations",
        "owner": "Milton Munroe",
        "title": FOUNDER_ROLE,
        "coverage": "Release control, pilot operations coordination, stakeholder communication, and governance ownership.",
        "status": "In place"
    },
    {
        "area": "Technical Infrastructure & Security",
        "owner": "Uche Anosike",
        "title": "Technical Infrastructure & Security Advisor",
        "coverage": "Infrastructure review, environment hardening guidance, deployment security, backup / recovery planning, and patch-management review.",
        "status": "In place"
    },
    {
        "area": "Clinical Review",
        "owner": "Clinical advisor / site champion placeholder",
        "title": "Clinical Review Lead",
        "coverage": "Clinical workflow review, hospital-facing feedback, and pilot-use oversight at the site level.",
        "status": "Site-specific placeholder"
    },
    {
        "area": "Hospital Sponsor",
        "owner": "Executive sponsor placeholder",
        "title": "Hospital Sponsor",
        "coverage": "Executive support, pilot escalation awareness, and closeout sign-off for the participating site.",
        "status": "Site-specific placeholder"
    },
]

PILOT_MFA_ACCESS_EVIDENCE_LOG = [
    {
        "control": "Admin application access",
        "implementation": "Allowed-admin email validation and password or hash verification before an admin session is created.",
        "owner": "Founder/Admin",
        "last_reviewed": "2026-04-01",
        "evidence": "Implemented in the Flask admin login path and session creation flow.",
        "status": "Implemented in app"
    },
    {
        "control": "Session protection baseline",
        "implementation": "HTTPOnly session cookie, SameSite=Lax, secure-cookie environment toggle, and no-store headers.",
        "owner": "Engineering",
        "last_reviewed": "2026-04-01",
        "evidence": "Configured in application settings and response headers.",
        "status": "Implemented in app"
    },
    {
        "control": "Administrative MFA confirmation",
        "implementation": "External MFA for admin email, remote access, and administrative consoles.",
        "owner": "Founder/Admin",
        "last_reviewed": "",
        "evidence": "Deployment confirmation required before hospital pilot start.",
        "status": "Pending external verification"
    },
    {
        "control": "Front-door / WAF review",
        "implementation": "Deployment-layer traffic protection and access review for the public front door.",
        "owner": "Technical Infrastructure & Security Advisor",
        "last_reviewed": "",
        "evidence": "Confirm hosting or edge-layer configuration before hospital deployment.",
        "status": "Pending deployment review"
    },
]

PILOT_BACKUP_RESTORE_LOG = [
    {
        "system": "Pilot data files and governance artifacts",
        "owner": "Technical Infrastructure & Security Advisor",
        "backup_scope": "Workflow records, thresholds, trends, pilot request logs, and governance packet artifacts.",
        "last_tested": "",
        "evidence": "Record the first restore test date and storage location before hospital pilot start.",
        "notes": "Immutable or separate-environment backup evidence still needs operational confirmation.",
        "status": "Pending external verification"
    },
    {
        "system": "Application bundle / release artifact",
        "owner": "Founder/Admin",
        "backup_scope": "Current stable pilot bundle and release-controlled source snapshot.",
        "last_tested": "2026-04-01",
        "evidence": "Current bundle version and release notes preserved in pilot governance materials.",
        "notes": "Application release control is in place; off-platform backup evidence should still be recorded.",
        "status": "Structured and documented"
    },
]

PILOT_PATCH_LOG = [
    {
        "component": "Stable pilot application bundle",
        "owner": "Founder/Admin",
        "cadence": "Approved bundle updates with critical fixes prioritized inside a 30-day window.",
        "last_patched": "2026-04-01",
        "evidence": "stable-pilot-1.0.4 release note and change-controlled bundle update.",
        "status": "Current bundle updated"
    },
    {
        "component": "Administrative and support endpoints",
        "owner": "Technical Infrastructure & Security Advisor",
        "cadence": "Monthly review and critical out-of-band remediation as needed.",
        "last_patched": "",
        "evidence": "Endpoint patch evidence must be recorded outside the application before hospital pilot start.",
        "status": "Pending external verification"
    },
]

PILOT_ACCESS_REVIEW_LOG = [
    {
        "scope": "Application roles and unit scoping baseline",
        "owner": "Founder/Admin",
        "last_reviewed": "2026-04-01",
        "evidence": "Role, unit, workflow, and audit scoping reviewed against current routes and session context.",
        "status": "Pass"
    },
    {
        "scope": "Site-specific user list and revocation plan",
        "owner": "Founder/Admin",
        "last_reviewed": "",
        "evidence": "Complete before site activation and again at pilot closeout.",
        "status": "Pending site setup"
    },
]

PILOT_TABLETOP_LOG = [
    {
        "exercise_name": "Security incident and pilot pause tabletop",
        "owner": "Founder/Admin",
        "exercise_date": "",
        "scenario": "Credential compromise, pilot pause decision, stakeholder notification, and restore approval path.",
        "evidence": "Run and record before hospital pilot start.",
        "status": "Pending"
    }
]

PILOT_TRAINING_ACK_LOG = [
    {
        "audience": "Founder/Admin baseline review",
        "owner": "Founder/Admin",
        "ack_date": "2026-04-01",
        "evidence": "Current pilot operating, claims-control, and access-review packet updated with this bundle.",
        "status": "Updated"
    },
    {
        "audience": "Site-specific pilot users",
        "owner": "Clinical Review Lead / Site Champion",
        "ack_date": "",
        "evidence": "Collect signed or documented acknowledgment before site go-live.",
        "status": "Pending site setup"
    },
]

PILOT_SITE_PACKET_TEMPLATE = [
    {
        "section": "Hospital / Site Identification",
        "summary": "Hospital name, primary unit, pilot start target, sponsor, clinical champion, technical contact, and success-review cadence.",
        "owner": "Founder/Admin + Site Sponsor",
        "status": "Fill before pilot start"
    },
    {
        "section": "Telemetry / ICU Packet",
        "summary": "Identify whether the pilot is telemetry, ICU, stepdown, ward, or RPM, and define the included monitoring scope and workflow users.",
        "owner": "Clinical Review Lead",
        "status": "Fill before pilot start"
    },
    {
        "section": "Hospital / Vendor Responsibilities",
        "summary": "Document who manages clinical review, user provisioning, access revocation, deployment coordination, support contacts, and change approval.",
        "owner": "Founder/Admin + Site Sponsor",
        "status": "Fill before pilot start"
    },
    {
        "section": "Success Metrics",
        "summary": "Define the site’s pilot success measures such as workflow adoption, review timeliness, explainability usefulness, and operational acceptance.",
        "owner": "Site Sponsor / Clinical Review Lead",
        "status": "Fill before pilot start"
    },
    {
        "section": "Support / Escalation Plan",
        "summary": "Record technical support owner, clinical review owner, escalation path, incident contact route, and expected response windows.",
        "owner": "Founder/Admin + Technical Infrastructure & Security Advisor",
        "status": "Fill before pilot start"
    },
    {
        "section": "Pilot Closeout",
        "summary": "Document closeout review, user deprovisioning, data return or deletion, retained artifacts, and sign-off responsibilities.",
        "owner": "Founder/Admin + Site Sponsor",
        "status": "Fill at pilot closeout"
    },
]

PILOT_VALIDATION_EVIDENCE = [
    {
        "date_tested": "2026-03-25",
        "tested_by": "Founder/Engineering",
        "test_case_id": "VAL-001",
        "test_case": "Access scope respects login, role, and assigned unit",
        "expected_result": "Authorized users see only approved role and unit scope in access context and command-center views.",
        "actual_result": "Verified through /login, /pilot-access, /api/access-context, unit filter behavior, and scoped command-center views.",
        "screenshot_evidence": "Secure pilot access screen and role/unit-scoped command-center screenshots.",
        "status": "Pass"
    },
    {
        "date_tested": "2026-03-25",
        "tested_by": "Founder/Engineering",
        "test_case_id": "VAL-002",
        "test_case": "Explainability guardrails are visible in the patient review workflow",
        "expected_result": "Patient detail workflow shows review basis, contributing factors, confidence, limitations, freshness, what changed, and unknowns.",
        "actual_result": "Verified review basis, confidence, limitations, freshness, workflow-vs-recommendation separation, and unknowns display in patient detail and guardrail sections.",
        "screenshot_evidence": "Patient drawer screenshots showing review basis and guardrail panels.",
        "status": "Pass"
    },
    {
        "date_tested": "2026-03-26",
        "tested_by": "Founder/Product",
        "test_case_id": "VAL-003",
        "test_case": "Pilot docs render current governance package",
        "expected_result": "Pilot docs should display intended use, claims controls, risk register, issue handling, change approvals, access/data policies, training materials, and release-controlled documentation.",
        "actual_result": "Verified /pilot-docs page renders current governance package and release-controlled documentation.",
        "screenshot_evidence": "Pilot docs screenshots showing governance tables and document sections.",
        "status": "Pass"
    },
    {
        "date_tested": "2026-03-26",
        "tested_by": "Founder/Product",
        "test_case_id": "VAL-004",
        "test_case": "Claims-control and wording cleanup reflected in command center",
        "expected_result": "Supportive wording should use review-oriented language and avoid autonomous or directive phrasing.",
        "actual_result": "Verified near-term review, review-priority summary, workflow tracking status, and supportive review visibility language in the command center bundle.",
        "screenshot_evidence": "Command-center screenshots showing updated labels and governance summary cards.",
        "status": "Pass"
    },
]

PILOT_DOCUMENT_CONTROL_INDEX = [
    {"document_name": "Frozen Intended-Use Statement", "version": "1.0", "owner": "Founder/Product", "last_updated": "2026-03-26", "status": "Approved", "pilot_applicability": "All controlled pilots"},
    {"document_name": "Approved Support Language", "version": "1.0", "owner": "Founder/Product", "last_updated": "2026-03-26", "status": "Approved", "pilot_applicability": "All controlled pilots"},
    {"document_name": "Claims Control Sheet", "version": "1.0", "owner": "Founder/Product", "last_updated": "2026-03-26", "status": "Approved", "pilot_applicability": "All controlled pilots"},
    {"document_name": "Complaint / Issue Log", "version": "1.0", "owner": "Founder/Admin", "last_updated": "2026-03-26", "status": "Active", "pilot_applicability": "All controlled pilots"},
    {"document_name": "Issue Escalation Process", "version": "1.0", "owner": "Founder/Admin", "last_updated": "2026-03-26", "status": "Approved", "pilot_applicability": "All controlled pilots"},
    {"document_name": "Change Approval Log", "version": "1.0", "owner": "Founder/Admin", "last_updated": "2026-03-26", "status": "Active", "pilot_applicability": "All controlled pilots"},
    {"document_name": "Cybersecurity Summary", "version": "1.0", "owner": "Founder/Admin", "last_updated": "2026-03-26", "status": "Approved", "pilot_applicability": "All controlled pilots"},
    {"document_name": "User Provisioning / Deprovisioning Policy", "version": "1.0", "owner": "Founder/Admin", "last_updated": "2026-03-26", "status": "Approved", "pilot_applicability": "All controlled pilots"},
    {"document_name": "Data Retention / Deletion / Return Policy", "version": "1.0", "owner": "Founder/Product", "last_updated": "2026-03-26", "status": "Approved", "pilot_applicability": "All controlled pilots"},
    {"document_name": "Pilot Agreement / Scope Document", "version": "1.0", "owner": "Founder/Product", "last_updated": "2026-03-26", "status": "Approved", "pilot_applicability": "All controlled pilots"},
    {"document_name": "Training / Use Instructions", "version": "1.0", "owner": "Founder/Admin", "last_updated": "2026-03-26", "status": "Approved", "pilot_applicability": "All pilot users"},
    {"document_name": "Dated Validation Evidence", "version": "1.0", "owner": "Founder/Engineering", "last_updated": "2026-03-26", "status": "Active", "pilot_applicability": "All controlled pilots"},
    {"document_name": "Release Notes", "version": "1.0", "owner": "Founder/Product", "last_updated": "2026-03-26", "status": "Active", "pilot_applicability": "All controlled pilots"},
]

DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "icu": {"spo2_low": 92, "hr_high": 120, "sbp_high": 160},
    "telemetry": {"spo2_low": 93, "hr_high": 110, "sbp_high": 150},
    "stepdown": {"spo2_low": 93, "hr_high": 115, "sbp_high": 155},
    "ward": {"spo2_low": 94, "hr_high": 110, "sbp_high": 150},
    "rpm": {"spo2_low": 94, "hr_high": 105, "sbp_high": 145},
    "all": {"spo2_low": 93, "hr_high": 112, "sbp_high": 152},
}

VALID_UNITS = {"all", "icu", "telemetry", "stepdown", "ward", "rpm"}

ROLE_ACTIONS: Dict[str, set[str]] = {
    "viewer": {"view"},
    "operator": {"view", "ack", "assign"},
    "physician": {"view", "ack", "assign", "escalate", "resolve"},
    "admin": {"view", "ack", "assign", "escalate", "resolve", "admin"},
}

HOSPITAL_BRANDS = {
    "early-risk-alert-ai": {
        "hospital_name": "Early Risk Alert AI",
        "brand_name": "Early Risk Alert AI",
        "brand_tagline": "Clinical Command Center",
        "brand_primary": "#7aa2ff",
        "brand_secondary": "#5bd4ff",
    }
}

PILOT_ACCOUNTS = {
    "admin@erapilot.com": {
        "full_name": "ERA Pilot Admin",
        "email": "admin@erapilot.com",
        "user_role": "admin",
        "assigned_unit": "all",
        "hospital_brand": "early-risk-alert-ai",
    },
    "pilot@earlyriskalertai.com": {
        "full_name": "Early Risk Alert AI Pilot User",
        "email": "pilot@earlyriskalertai.com",
        "user_role": "operator",
        "assigned_unit": "all",
        "hospital_brand": "early-risk-alert-ai",
    },
}


LOGIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Secure Pilot Access</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;
      --panel:#101a2d;
      --line:rgba(255,255,255,.08);
      --text:#eef4ff;
      --muted:#9fb4d6;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --warn:#f4bd6a;
    }
    *{box-sizing:border-box}
    body{
      margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at 10% 10%, rgba(91,212,255,.12), transparent 20%),
        linear-gradient(180deg, #07101c, #0b1528);
    }
    .card{
      width:min(640px,100%);
      border:1px solid var(--line);border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    .k{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff;margin-bottom:10px}
    h1{margin:0 0 10px;font-size:40px;line-height:.95;letter-spacing:-.05em}
    p{margin:0 0 18px;color:var(--muted);line-height:1.6}
    .callout{
      margin:0 0 18px;padding:14px 16px;border-radius:16px;
      background:rgba(91,212,255,.08);border:1px solid rgba(91,212,255,.16);color:#dce9ff;
      line-height:1.6;
    }
    .disclaimer{
      margin:16px 0 18px;padding:14px 16px;border-radius:16px;
      background:rgba(244,189,106,.10);border:1px solid rgba(244,189,106,.22);color:#ffe7bf;
      line-height:1.6;font-size:14px;
    }
    label{display:block;font-size:13px;font-weight:900;margin-bottom:8px}
    input, select{
      width:100%;padding:14px 16px;border-radius:16px;border:1px solid var(--line);
      background:#0d1728;color:var(--text);font:inherit;margin-bottom:14px;
    }
    button{
      width:100%;padding:14px 18px;border:none;border-radius:16px;cursor:pointer;
      font:inherit;font-weight:1000;color:#07101c;
      background:linear-gradient(135deg,var(--blue),var(--blue2));
    }
    .links{display:flex;gap:14px;flex-wrap:wrap;margin-top:16px}
    a{color:#cfe7ff;text-decoration:none;font-weight:800}
    .error{
      margin-bottom:14px;padding:12px 14px;border-radius:14px;
      background:rgba(255,102,125,.12);border:1px solid rgba(255,102,125,.24);color:#ffd8de;
    }
    .footer-note{
      margin-top:18px;color:#9fb4d6;font-size:13px;line-height:1.6;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="k">Controlled pilot environment</div>
    <h1>Early Risk Alert AI Secure Pilot Access</h1>
    <p>Access the Early Risk Alert AI command center for controlled pilot evaluation, role-based workflow review, and unit-scoped visibility.</p>

    <div class="callout">
      Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform intended to assist authorized health care professionals in identifying patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.
    </div>

    <div class="disclaimer">
      The platform does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.
    </div>

    __ERROR__

    <form method="post" action="/login">
      <label>Full Name</label>
      <input name="full_name" placeholder="Your full name" required>

      <label>Work Email</label>
      <input name="email" type="email" placeholder="you@hospital.org" required>

      <label>Role</label>
      <select name="user_role" required>
        <option value="viewer">Viewer</option>
        <option value="operator">Operator</option>
        <option value="physician">Physician</option>
        <option value="admin">Admin</option>
      </select>

      <label>Admin Password (required for Admin role)</label>
      <input name="admin_password" type="password" placeholder="Enter admin password if admin role is selected">

      <label>Hospital Brand</label>
      <select name="hospital_brand" required>
        <option value="early-risk-alert-ai">Early Risk Alert AI</option>
      </select>

      <label>Assigned Unit</label>
      <select name="assigned_unit" required>
        <option value="all">All Units</option>
        <option value="icu">ICU</option>
        <option value="telemetry">Telemetry</option>
        <option value="stepdown">Stepdown</option>
        <option value="ward">Ward</option>
        <option value="rpm">RPM / Home</option>
      </select>

      <button type="submit">Enter Secure Pilot Access</button>
    </form>

    <div class="links">
      <a href="/pilot-access">Pilot Access</a>
      <a href="/hospital-demo">Hospital Demo</a>
      <a href="/investor-intake">Investor Access</a>
      <a href="/executive-walkthrough">Executive Walkthrough</a>
    </div>

    <div class="footer-note">
      Controlled Pilot Evaluation | Secure Cloud Deployment | Hospital-Facing Workflow Support
    </div>
  </div>
</body>
</html>
"""


PILOT_ACCESS_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pilot Access — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{
      margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528);
    }
    .card{
      width:min(560px,100%);
      border:1px solid rgba(255,255,255,.08);border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    h1{margin:0 0 10px;font-size:38px;line-height:.95;letter-spacing:-.05em}
    p{color:#9fb4d6;line-height:1.6}
    .callout{
      margin:0 0 18px;padding:14px 16px;border-radius:16px;
      background:rgba(91,212,255,.08);border:1px solid rgba(91,212,255,.16);color:#dce9ff;
      line-height:1.6;
    }
    .disclaimer{
      margin:0 0 18px;padding:14px 16px;border-radius:16px;
      background:rgba(244,189,106,.10);border:1px solid rgba(244,189,106,.22);color:#ffe7bf;
      line-height:1.6;font-size:14px;
    }
    input{
      width:100%;padding:14px 16px;border-radius:16px;border:1px solid rgba(255,255,255,.08);
      background:#0d1728;color:#eef4ff;font:inherit;margin-bottom:14px;
    }
    button{
      width:100%;padding:14px 18px;border:none;border-radius:16px;cursor:pointer;
      font:inherit;font-weight:1000;color:#07101c;
      background:linear-gradient(135deg,#7aa2ff,#5bd4ff);
    }
    .error{
      margin-bottom:14px;padding:12px 14px;border-radius:14px;
      background:rgba(255,102,125,.12);border:1px solid rgba(255,102,125,.24);color:#ffd8de;
    }
    .footer-note{
      margin-top:18px;color:#9fb4d6;font-size:13px;line-height:1.6;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Branded Pilot Access</h1>
    <p>Enter your authorized pilot email to load your configured command-center experience, assigned unit, and pilot access scope.</p>

    <div class="callout">
      Early Risk Alert AI supports controlled pilot evaluation through an HCP-facing decision-support and workflow-support platform designed to assist authorized health care professionals with patient prioritization, monitored patient visibility, and command-center operational awareness.
    </div>

    <div class="disclaimer">
      It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.
    </div>

    __ERROR__

    <form method="post" action="/pilot-access">
      <input name="pilot_email" type="email" placeholder="pilot@hospital.com" required>
      <button type="submit">Enter Pilot Account</button>
    </form>

    <div class="footer-note">
      Controlled Pilot Evaluation | Role-Based Access | Unit-Scoped Visibility
    </div>
  </div>
</body>
</html>
"""


FORM_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{
      --bg:#08111f;
      --panel:#101a2d;
      --line:rgba(255,255,255,.08);
      --text:#eef4ff;
      --muted:#9fb4d6;
      --blue:#7aa2ff;
      --blue2:#5bd4ff;
      --warn:#f4bd6a;
    }
    *{box-sizing:border-box}
    body{
      margin:0;padding:24px;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:var(--text);
      background:linear-gradient(180deg,#07101c,#0b1528);
    }
    .wrap{max-width:920px;margin:0 auto}
    .card{
      border:1px solid var(--line);border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    .k{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff;margin-bottom:10px}
    h1{margin:0 0 10px;font-size:44px;line-height:.95;letter-spacing:-.05em}
    p{margin:0 0 18px;color:var(--muted);line-height:1.6}
    .callout{
      margin:0 0 18px;padding:14px 16px;border-radius:16px;
      background:rgba(91,212,255,.08);border:1px solid rgba(91,212,255,.16);color:#dce9ff;
      line-height:1.6;
    }
    .disclaimer{
      margin:0 0 20px;padding:14px 16px;border-radius:16px;
      background:rgba(244,189,106,.10);border:1px solid rgba(244,189,106,.22);color:#ffe7bf;
      line-height:1.6;font-size:14px;
    }
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .field{margin-bottom:14px}
    .full{grid-column:1 / -1}
    label{display:block;font-size:13px;font-weight:900;margin-bottom:8px}
    input, select, textarea{
      width:100%;padding:14px 16px;border-radius:16px;border:1px solid var(--line);
      background:#0d1728;color:var(--text);font:inherit;
    }
    textarea{min-height:120px;resize:vertical}
    .check{
      display:flex;gap:12px;align-items:flex-start;padding:14px 16px;border-radius:16px;
      border:1px solid var(--line);background:#0d1728;
    }
    .check input{
      width:auto;margin:4px 0 0;
    }
    .check span{
      color:#dce9ff;line-height:1.6;font-size:14px;
    }
    button{
      padding:14px 18px;border:none;border-radius:16px;cursor:pointer;
      font:inherit;font-weight:1000;color:#07101c;
      background:linear-gradient(135deg,var(--blue),var(--blue2));
    }
    .links{display:flex;gap:14px;flex-wrap:wrap;margin-top:16px}
    a{color:#cfe7ff;text-decoration:none;font-weight:800}
    .footer-note{
      margin-top:18px;color:#9fb4d6;font-size:13px;line-height:1.6;
    }
    @media (max-width:700px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="k">Early Risk Alert AI</div>
      <h1>__HEADING__</h1>
      <p>__COPY__</p>

      <div class="callout">
        Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform intended to assist authorized health care professionals in identifying patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.
      </div>

      <div class="disclaimer">
        The platform is intended for controlled pilot evaluation and hospital-facing workflow support. It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.
      </div>

      <form method="post">
        <div class="grid">
          __FIELDS__
        </div>
        <button type="submit">__BUTTON__</button>
      </form>

      <div class="links">
        <a href="/command-center">Command Center</a>
        <a href="/admin/review">Admin Review</a>
      </div>

      <div class="footer-note">
        Controlled Pilot Evaluation | Secure Cloud Deployment | Hospital-Facing Workflow Support
      </div>
    </div>
  </div>
</body>
</html>
"""


THANK_YOU_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Request Received</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{
      margin:0;padding:24px;display:grid;place-items:center;min-height:100vh;
      font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528);
    }
    .card{
      width:min(760px,100%);
      border:1px solid rgba(255,255,255,.08);border-radius:24px;
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));
      padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34);
    }
    h1{margin:0 0 10px;font-size:40px;line-height:.95;letter-spacing:-.05em}
    p{margin:0 0 18px;color:#9fb4d6;line-height:1.6}
    .callout{
      margin:0 0 18px;padding:14px 16px;border-radius:16px;
      background:rgba(91,212,255,.08);border:1px solid rgba(91,212,255,.16);color:#dce9ff;
      line-height:1.6;
    }
    .disclaimer{
      margin:0 0 18px;padding:14px 16px;border-radius:16px;
      background:rgba(244,189,106,.10);border:1px solid rgba(244,189,106,.22);color:#ffe7bf;
      line-height:1.6;font-size:14px;
    }
    .box{
      border:1px solid rgba(255,255,255,.08);border-radius:18px;background:rgba(255,255,255,.03);
      padding:16px;line-height:1.8;color:#dce9ff;margin:16px 0;
    }
    a{
      display:inline-flex;align-items:center;justify-content:center;
      padding:12px 16px;border-radius:16px;font-size:14px;font-weight:900;
      background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;text-decoration:none;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Thank You</h1>
    <p>__MESSAGE__</p>

    <div class="callout">
      A member of the Early Risk Alert AI team will review your submission and follow up regarding pilot evaluation, platform demonstration, executive walkthrough, or investor access.
    </div>

    <div class="disclaimer">
      Early Risk Alert AI is intended for controlled pilot evaluation and hospital-facing workflow support. It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.
    </div>

    <div class="box">__DETAILS__</div>
    <a href="/command-center">Return to Command Center</a>
  </div>
</body>
</html>
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.getenv("SECRET_KEY", "early-risk-alert-dev-secret")
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"
    app.config["SESSION_COOKIE_NAME"] = os.getenv("SESSION_COOKIE_NAME", "eraa_session")
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=int(os.getenv("SESSION_LIFETIME_HOURS", "12")))
    app.config["SESSION_REFRESH_EACH_REQUEST"] = True

    runtime_state = {
        "last_update": datetime.now(timezone.utc),
        "heartbeat_bucket": -1,
    }

    in_memory_audit: List[Dict[str, Any]] = []

    data_dir = Path(app.instance_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    DECK_FILE_CANDIDATES = {
        "main": [
            "Early_Risk_Alert_AI_Pitch_Deck.pdf",
            "Early_Risk_Alert_AI_Pitch_Deck.pdf",
            "Early_Risk_Alert_AI_Hospital_Pilot_Deck.pdf",
        ],
        "hospital": [
            "Early_Risk_Alert_AI_Pitch_Deck.pdf",
            "Early_Risk_Alert_AI_Hospital_Pilot_Deck.pdf",
            "Early_Risk_Alert_AI_Pitch_Deck.pdf",
        ],
        "investor": [
            "Early_Risk_Alert_AI_Investor_Deck.pdf",
            "Early_Risk_Alert_AI_Pitch_Deck.pdf",
        ],
        "executive": [
            "Early_Risk_Alert_AI_Pitch_Deck.pdf",
            "Early_Risk_Alert_AI_Hospital_Pilot_Deck.pdf",
            "Early_Risk_Alert_AI_Pitch_Deck.pdf",
        ],
        "legacy": [
            "Early_Risk_Alert_AI_Pitch_Deck.pdf",
            "Early_Risk_Alert_AI_Pitch_Deck.pdf",
            "Early_Risk_Alert_AI_Hospital_Pilot_Deck.pdf",
        ],
    }



    DOC_FILE_NAMES = {
        "pilot_success_guide": "Early_Risk_Alert_AI_Pilot_Success_Criteria_and_Workflow_Integration_Guide.docx",
        "model_card": "Early_Risk_Alert_AI_Model_Card.docx",
    }

    def _deck_search_roots() -> List[Path]:
        roots = [
            Path(app.root_path) / "static",
            Path(app.root_path).parent / "static",
            Path.cwd() / "static",
        ]
        unique: List[Path] = []
        seen = set()
        for root in roots:
            key = str(root.resolve()) if root.exists() else str(root)
            if key not in seen:
                seen.add(key)
                unique.append(root)
        return unique

    def _resolve_deck_file(kind: str = "main") -> tuple[Path, str]:
        names = DECK_FILE_CANDIDATES.get(kind, DECK_FILE_CANDIDATES["main"])
        roots = _deck_search_roots()
        for root in roots:
            for name in names:
                if (root / name).exists():
                    return root, name
        return roots[0], names[0]

    def _serve_deck(kind: str = "main"):
        root, filename = _resolve_deck_file(kind)
        response = send_from_directory(
            str(root),
            filename,
            mimetype="application/pdf",
            as_attachment=False,
            download_name=filename,
        )
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


    def _serve_static_doc(kind: str):
        filename = DOC_FILE_NAMES[kind]
        roots = _deck_search_roots()
        for root in roots:
            candidate = root / filename
            if candidate.exists():
                response = send_from_directory(
                    str(root),
                    filename,
                    as_attachment=True,
                    download_name=filename,
                    max_age=0,
                )
                response.headers["Cache-Control"] = "no-store, max-age=0"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                return response
        raise FileNotFoundError(f"Missing static document: {filename}")

    def _route_status_snapshot() -> Dict[str, Any]:
        route_set = {rule.rule for rule in app.url_map.iter_rules()}
        tracked_routes = [
            "/",
            "/login",
            "/pilot-access",
            "/logout",
            "/command-center",
            "/pilot-docs",
            "/admin/review",
            "/hospital-demo",
            "/executive-walkthrough",
            "/investor-intake",
            "/api/access-context",
            "/api/workflow",
            "/api/audit",
            "/api/thresholds",
            "/api/system-health",
            "/api/v1/live-snapshot",
            "/healthz",
        ]
        missing_routes = [route for route in tracked_routes if route not in route_set]

        def _deck_exists(kind: str) -> bool:
            root, filename = _resolve_deck_file(kind)
            return (root / filename).exists()

        def _doc_exists(kind: str) -> bool:
            filename = DOC_FILE_NAMES[kind]
            return any((root / filename).exists() for root in _deck_search_roots())

        documents = {
            "pitch_deck": _deck_exists("main"),
            "hospital_deck": _deck_exists("hospital"),
            "investor_deck": _deck_exists("investor"),
            "pilot_success_guide": _doc_exists("pilot_success_guide"),
            "model_card": _doc_exists("model_card"),
        }

        return {
            "checked_at": _utc_now_iso(),
            "routes_checked": len(tracked_routes),
            "routes_available": len(tracked_routes) - len(missing_routes),
            "missing_routes": missing_routes,
            "docs_checked": len(documents),
            "docs_available": sum(1 for ok in documents.values() if ok),
            "documents": documents,
        }

    hospital_file = data_dir / "hospital_demo_requests.json"
    exec_file = data_dir / "executive_walkthrough_requests.json"
    investor_file = data_dir / "investor_intake_requests.json"
    workflow_file = data_dir / "command_workflow.json"
    thresholds_file = data_dir / "risk_thresholds.json"
    trends_file = data_dir / "patient_trends.json"

    def _read_json_list(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, list) else []
        except Exception:
            return []

    def _write_json_list(path: Path, payload: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _read_json_dict(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        if not path.exists():
            return json.loads(json.dumps(default))
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else json.loads(json.dumps(default))
        except Exception:
            return json.loads(json.dumps(default))

    def _write_json_dict(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _load_workflow() -> Dict[str, Any]:
        default = {"pilot_mode": True, "records": {}, "audit_log": []}
        store = _read_json_dict(workflow_file, default)
        store.setdefault("pilot_mode", True)
        store.setdefault("records", {})
        store.setdefault("audit_log", [])
        return store

    def _save_workflow(store: Dict[str, Any]) -> None:
        _write_json_dict(workflow_file, store)

    def _load_thresholds() -> Dict[str, Dict[str, float]]:
        stored = _read_json_dict(thresholds_file, {})
        merged = json.loads(json.dumps(DEFAULT_THRESHOLDS or {
            "icu": {"spo2_low": 92, "hr_high": 120, "sbp_high": 160},
            "telemetry": {"spo2_low": 93, "hr_high": 110, "sbp_high": 150},
            "stepdown": {"spo2_low": 93, "hr_high": 115, "sbp_high": 155},
            "ward": {"spo2_low": 94, "hr_high": 110, "sbp_high": 150},
            "rpm": {"spo2_low": 94, "hr_high": 105, "sbp_high": 145},
            "all": {"spo2_low": 93, "hr_high": 112, "sbp_high": 152},
        }))
        for unit, values in stored.items():
            if unit in merged and isinstance(values, dict):
                merged[unit].update(
                    {
                        "spo2_low": _safe_float(values.get("spo2_low"), merged[unit]["spo2_low"]),
                        "hr_high": _safe_float(values.get("hr_high"), merged[unit]["hr_high"]),
                        "sbp_high": _safe_float(values.get("sbp_high"), merged[unit]["sbp_high"]),
                    }
                )
        return merged

    def _save_thresholds(payload: Dict[str, Dict[str, float]]) -> None:
        clean: Dict[str, Dict[str, float]] = {}
        for unit, values in payload.items():
            if unit not in DEFAULT_THRESHOLDS or not isinstance(values, dict):
                continue
            clean[unit] = {
                "spo2_low": round(_safe_float(values.get("spo2_low"), DEFAULT_THRESHOLDS[unit]["spo2_low"]), 1),
                "hr_high": round(_safe_float(values.get("hr_high"), DEFAULT_THRESHOLDS[unit]["hr_high"]), 1),
                "sbp_high": round(_safe_float(values.get("sbp_high"), DEFAULT_THRESHOLDS[unit]["sbp_high"]), 1),
            }
        _write_json_dict(thresholds_file, clean)

    def _load_trends() -> Dict[str, List[Dict[str, Any]]]:
        raw = _read_json_dict(trends_file, {})
        cleaned: Dict[str, List[Dict[str, Any]]] = {}
        for patient_id, rows in raw.items():
            if isinstance(rows, list):
                cleaned[str(patient_id)] = [row for row in rows if isinstance(row, dict)]
        return cleaned

    def _save_trends(payload: Dict[str, List[Dict[str, Any]]]) -> None:
        _write_json_dict(trends_file, payload)

    def _logged_in() -> bool:
        return bool(session.get("logged_in"))

    def _current_user() -> str:
        return str(session.get("full_name", "Pilot User")).strip() or "Pilot User"

    def _current_role() -> str:
        role = str(session.get("user_role", "viewer")).strip().lower()
        return role if role in ROLE_ACTIONS else "viewer"

    def _current_unit_access() -> str:
        unit = str(session.get("assigned_unit", "all")).strip().lower()
        return unit if unit in VALID_UNITS else "all"

    def _current_hospital_brand_key() -> str:
        brand = str(session.get("hospital_brand", "early-risk-alert-ai")).strip().lower()
        return brand if brand in HOSPITAL_BRANDS else "early-risk-alert-ai"

    def _current_brand() -> Dict[str, str]:
        return HOSPITAL_BRANDS[_current_hospital_brand_key()]

    def _has_permission(action: str) -> bool:
        return action in ROLE_ACTIONS.get(_current_role(), {"view"})

    def _can_view_unit(unit: str) -> bool:
        current = _current_unit_access()
        role = _current_role()
        if role == "admin" or current == "all":
            return True
        return current == unit

    def _login_required(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not _logged_in():
                return redirect(url_for("login"))
            return fn(*args, **kwargs)
        return wrapper

    def _admin_required(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not _logged_in():
                return redirect(url_for("login"))
            if _current_role() != "admin":
                return redirect(url_for("command_center"))
            return fn(*args, **kwargs)
        return wrapper

    def _detail_html(payload: Dict[str, Any], keys: List[str]) -> str:
        return "<br>".join(
            f"<strong>{key.replace('_', ' ').title()}:</strong> {payload.get(key, '')}"
            for key in keys
        )

    def _lead_score(payload: Dict[str, Any], lead_type: str) -> int:
        score = 0
        timeline = str(payload.get("timeline", "")).strip().lower()
        if "immediate" in timeline:
            score += 3
        elif "30-60" in timeline:
            score += 2
        elif "quarter" in timeline:
            score += 1

        if lead_type == "investor":
            stage = str(payload.get("stage", "")).strip().lower()
            if "institutional" in stage or "strategic" in stage:
                score += 3
            elif "angel" in stage or "seed" in stage:
                score += 2

        return score

    def _append_row(path: Path, row: Dict[str, Any]) -> None:
        rows = _read_json_list(path)
        rows.append(row)
        _write_json_list(path, rows)

    def _normalize_room_to_unit(room: str) -> str:
        value = str(room or "").lower()
        if "icu" in value:
            return "icu"
        if "telemetry" in value:
            return "telemetry"
        if "stepdown" in value:
            return "stepdown"
        if "ward" in value:
            return "ward"
        if "rpm" in value or "home" in value:
            return "rpm"
        return "telemetry"

    def _thresholds_for_unit(unit: str, threshold_map: Dict[str, Dict[str, float]] | None = None) -> Dict[str, float]:
        store = threshold_map or _load_thresholds()
        key = unit if unit in store else "all"
        return store.get(key, DEFAULT_THRESHOLDS["all"])

    def _build_review_basis(hr: int, sbp: int, spo2: int, thresholds: Dict[str, float], reasons: List[str]) -> List[str]:
        basis = [
            "HCP-facing decision-support and workflow-support display generated from structured patient medical information, trended vital sign summaries, and monitored patient context.",
            f"Current reviewed inputs: HR {hr}, SBP {sbp}, SpO₂ {spo2}. Reference thresholds: SpO₂ low < {thresholds['spo2_low']}, HR high > {thresholds['hr_high']}, SBP high > {thresholds['sbp_high']}.",
        ]
        if reasons:
            basis.append("Top contributing factors reviewed: " + "; ".join(reasons))
        else:
            basis.append("No trigger-level threshold breach detected in the current summary window.")
        basis.append(INTENDED_USE_STATEMENT)
        return basis

    def _build_current_review_inputs(hr: int, sbp: int, dbp: int, spo2: int, thresholds: Dict[str, float], generated_at: str) -> List[str]:
        return [
            f"Snapshot time reviewed: {generated_at}",
            f"Heart rate: {hr} bpm compared with review threshold > {thresholds['hr_high']}",
            f"Blood pressure: {sbp}/{dbp} mmHg compared with review threshold SBP > {thresholds['sbp_high']}",
            f"SpO₂: {spo2}% compared with review threshold < {thresholds['spo2_low']}",
        ]

    def _build_change_summary(item: Dict[str, Any], hr: int, sbp: int, dbp: int, spo2: int) -> List[str]:
        baseline = item.get("baseline", {})
        changes = [
            f"Heart rate changed from baseline {baseline.get('heart_rate', '--')} to {hr}",
            f"Systolic blood pressure changed from baseline {baseline.get('systolic_bp', '--')} to {sbp}",
            f"Diastolic blood pressure changed from baseline {baseline.get('diastolic_bp', '--')} to {dbp}",
            f"SpO₂ changed from baseline {baseline.get('spo2', '--')} to {spo2}",
        ]
        return changes

    def _build_unknowns(unit: str) -> List[str]:
        unit_label = unit.upper() if unit != "rpm" else "RPM"
        return [
            "This display does not know bedside examination findings, clinician impression, code status, active symptoms, or patient-reported changes unless they are entered elsewhere.",
            "This display does not know full chart context such as laboratory data, medications, imaging, fluid balance, device settings, consultant input, or undocumented workflow events.",
            f"This display does not know whether local {unit_label} escalation policy, staffing coverage, or physician instructions change the next appropriate step.",
        ]

    def _build_limitations(unit: str) -> List[str]:
        unit_label = unit.upper() if unit != "rpm" else "RPM"
        return [
            "HCP-facing pilot / evaluation environment; not a production bedside deployment or system of record.",
            "Output is decision-support and workflow-support only and is intended to assist further clinical evaluation, prioritization, and command-center awareness.",
            "Displayed output uses structured vital sign summaries and monitored context only; it is not intended to diagnose, direct treatment, or independently trigger escalation.",
            f"Unit-specific thresholds for {unit_label} inform the display, but clinician judgment, local escalation policy, and hospital protocol remain required.",
            "Incomplete, delayed, or missing data may affect the displayed output.",
        ]

    def _build_confidence_payload(severity: str, reasons: List[str], hr: int, sbp: int, spo2: int, thresholds: Dict[str, float]) -> Dict[str, Any]:
        supporting_signals = 0
        if spo2 < thresholds["spo2_low"]:
            supporting_signals += 1
        if hr > thresholds["hr_high"]:
            supporting_signals += 1
        if sbp > thresholds["sbp_high"]:
            supporting_signals += 1
        if any("combined" in str(reason).lower() for reason in reasons):
            supporting_signals += 1

        score = 56 + (supporting_signals * 8) + (4 if severity in {"high", "critical"} else 0)
        score = int(max(52, min(86, score)))

        if score >= 78:
            label = "supportive"
            narrative = "Multiple reviewed inputs support this HCP-facing output, but the display still requires independent clinical review before any action."
        elif score >= 66:
            label = "moderate"
            narrative = "Some reviewed inputs support this HCP-facing output; clinicians should confirm it against additional context and hospital policy."
        else:
            label = "limited"
            narrative = "This display provides a limited attention flag and should not be used as a standalone conclusion or directive."

        return {
            "score": score,
            "label": label,
            "narrative": narrative,
        }

    def _bounded(value: float, low: int, high: int) -> int:
        return int(max(low, min(high, round(value))))

    def _simulated_patients(threshold_map: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        tick = int(time.time() // 5)

        base = [
            {
                "patient_id": "p101",
                "patient_name": "Patient 1042",
                "room": "ICU-12",
                "program": "Cardiac",
                "baseline": {"heart_rate": 124, "systolic_bp": 160, "diastolic_bp": 98, "spo2": 90},
                "volatility": {"heart_rate": 8, "systolic_bp": 10, "diastolic_bp": 6, "spo2": 2},
                "trend_bias": {"heart_rate": 4, "systolic_bp": 5, "diastolic_bp": 2, "spo2": -1},
            },
            {
                "patient_id": "p202",
                "patient_name": "Patient 2188",
                "room": "Telemetry-04",
                "program": "Pulmonary",
                "baseline": {"heart_rate": 101, "systolic_bp": 149, "diastolic_bp": 89, "spo2": 93},
                "volatility": {"heart_rate": 6, "systolic_bp": 8, "diastolic_bp": 5, "spo2": 2},
                "trend_bias": {"heart_rate": 2, "systolic_bp": 2, "diastolic_bp": 1, "spo2": -1},
            },
            {
                "patient_id": "p303",
                "patient_name": "Patient 3045",
                "room": "Stepdown-09",
                "program": "Cardiac",
                "baseline": {"heart_rate": 110, "systolic_bp": 156, "diastolic_bp": 95, "spo2": 95},
                "volatility": {"heart_rate": 7, "systolic_bp": 9, "diastolic_bp": 6, "spo2": 1},
                "trend_bias": {"heart_rate": 3, "systolic_bp": 4, "diastolic_bp": 2, "spo2": 0},
            },
            {
                "patient_id": "p404",
                "patient_name": "Patient 4172",
                "room": "Ward-21",
                "program": "Recovery",
                "baseline": {"heart_rate": 84, "systolic_bp": 128, "diastolic_bp": 82, "spo2": 97},
                "volatility": {"heart_rate": 4, "systolic_bp": 6, "diastolic_bp": 4, "spo2": 1},
                "trend_bias": {"heart_rate": 0, "systolic_bp": 0, "diastolic_bp": 0, "spo2": 0},
            },
            {
                "patient_id": "p505",
                "patient_name": "Patient 5117",
                "room": "RPM-Home-03",
                "program": "Home Monitoring",
                "baseline": {"heart_rate": 106, "systolic_bp": 146, "diastolic_bp": 87, "spo2": 93},
                "volatility": {"heart_rate": 5, "systolic_bp": 7, "diastolic_bp": 4, "spo2": 2},
                "trend_bias": {"heart_rate": 1, "systolic_bp": 1, "diastolic_bp": 1, "spo2": -1},
            },
        ]

        patients: List[Dict[str, Any]] = []

        for item in base:
            unit = _normalize_room_to_unit(item["room"])
            thresholds = _thresholds_for_unit(unit, threshold_map)

            rng = random.Random(f"{item['patient_id']}:{tick}")
            phase = tick % 6
            drift_multiplier = {
                0: -0.45,
                1: -0.15,
                2: 0.10,
                3: 0.35,
                4: 0.20,
                5: -0.05,
            }[phase]

            baseline = item["baseline"]
            volatility = item["volatility"]
            trend_bias = item["trend_bias"]

            hr = _bounded(
                baseline["heart_rate"]
                + (trend_bias["heart_rate"] * drift_multiplier)
                + rng.randint(-volatility["heart_rate"], volatility["heart_rate"]),
                58,
                155,
            )
            sbp = _bounded(
                baseline["systolic_bp"]
                + (trend_bias["systolic_bp"] * drift_multiplier)
                + rng.randint(-volatility["systolic_bp"], volatility["systolic_bp"]),
                95,
                210,
            )
            dbp = _bounded(
                baseline["diastolic_bp"]
                + (trend_bias["diastolic_bp"] * drift_multiplier)
                + rng.randint(-volatility["diastolic_bp"], volatility["diastolic_bp"]),
                55,
                130,
            )
            spo2 = _bounded(
                baseline["spo2"]
                + (trend_bias["spo2"] * drift_multiplier)
                + rng.randint(-volatility["spo2"], volatility["spo2"]),
                84,
                100,
            )

            score = 0.0
            reasons: List[str] = []

            if spo2 < thresholds["spo2_low"]:
                gap = thresholds["spo2_low"] - spo2
                score += 3.2 + min(gap * 0.7, 2.2)
                reasons.append(f"SpO₂ below {thresholds['spo2_low']}")

            if hr > thresholds["hr_high"]:
                gap = hr - thresholds["hr_high"]
                score += 1.8 + min(gap * 0.08, 1.8)
                reasons.append(f"HR above {thresholds['hr_high']}")

            if sbp > thresholds["sbp_high"]:
                gap = sbp - thresholds["sbp_high"]
                score += 1.9 + min(gap * 0.06, 1.9)
                reasons.append(f"SBP above {thresholds['sbp_high']}")

            if spo2 <= max(thresholds["spo2_low"] - 3, 88) and hr >= thresholds["hr_high"]:
                score += 0.8
                reasons.append("combined oxygen and heart rate deterioration pattern")

            if spo2 <= max(thresholds["spo2_low"] - 2, 89) and sbp >= thresholds["sbp_high"]:
                score += 0.6
                reasons.append("combined oxygen and blood pressure deterioration pattern")

            score = round(min(score, 9.7), 1)

            if score >= 7:
                severity = "critical"
                action = "Suggested review context: prioritize independent clinical reassessment and senior clinician review according to clinician judgment and hospital protocol."
            elif score >= 4:
                severity = "high"
                action = "Suggested review context: prioritize clinical follow-up and reassess within the next monitoring interval according to local workflow and policy."
            elif score >= 2:
                severity = "moderate"
                action = "Suggested review context: continue monitored review and evaluate additional clinical context before deciding next steps."
            else:
                severity = "stable"
                action = "Suggested review context: continue routine monitored review within standard workflow."

            if not reasons:
                reasons = ["Combined reviewed summary pattern within acceptable thresholds"]

            generated_at = _utc_now_iso()
            review_basis = _build_review_basis(hr, sbp, spo2, thresholds, reasons)
            current_review_inputs = _build_current_review_inputs(hr, sbp, dbp, spo2, thresholds, generated_at)
            change_summary = _build_change_summary(item, hr, sbp, dbp, spo2)
            limitations = _build_limitations(unit)
            confidence = _build_confidence_payload(severity, reasons, hr, sbp, spo2, thresholds)
            unknowns = _build_unknowns(unit)

            story_map = {
                "critical": "HCP-facing decision-support highlights a high-priority patient for further clinical evaluation and command-center awareness.",
                "high": "HCP-facing decision-support shows elevated monitored risk and supports prioritization for additional review.",
                "moderate": "Monitored trend changes support continued review and workflow awareness.",
                "stable": "Structured patient summaries remain within the current monitored review range.",
            }

            alert_message = "Clinical review attention surfaced" if severity != "stable" else "Current review summary stable"

            patients.append(
                {
                    "patient_id": item["patient_id"],
                    "patient_name": item["patient_name"],
                    "room": item["room"],
                    "program": item["program"],
                    "heart_rate": hr,
                    "bp_systolic": sbp,
                    "bp_diastolic": dbp,
                    "spo2": spo2,
                    "vitals": {
                        "heart_rate": hr,
                        "systolic_bp": sbp,
                        "diastolic_bp": dbp,
                        "spo2": spo2,
                    },
                    "story": story_map[severity],
                    "data_freshness": {
                        "generated_at": generated_at,
                        "age_seconds": 0,
                        "source_mode": "structured-vital-summary-pilot",
                        "refresh_interval_seconds": 5,
                    },
                    "risk": {
                        "risk_score": score,
                        "severity": severity,
                        "alert_message": alert_message,
                        "recommended_action": action,
                        "clinical_recommendation": action,
                        "reasons": reasons,
                        "top_contributing_factors": reasons,
                        "review_basis": review_basis,
                        "current_review_inputs": current_review_inputs,
                        "what_changed": change_summary,
                        "what_software_does_not_know": unknowns,
                        "confidence": confidence,
                        "limitations": limitations,
                        "decision_support_disclaimer": INTENDED_USE_STATEMENT,
                    },
                }
            )

        return patients

    def _update_trend_history(patients: List[Dict[str, Any]]) -> None:
        history = _load_trends()
        bucket = int(time.time() // 5)

        for patient in patients:
            patient_id = str(patient["patient_id"])
            series = history.setdefault(patient_id, [])
            point = {
                "time": _utc_now_iso(),
                "bucket": bucket,
                "hr": patient["heart_rate"],
                "spo2": patient["spo2"],
                "risk": patient["risk"]["risk_score"],
            }

            if series and int(series[-1].get("bucket", -1)) == bucket:
                series[-1] = point
            else:
                series.append(point)

            history[patient_id] = series[-60:]

        _save_trends(history)

    def _simulated_snapshot() -> Dict[str, Any]:
        threshold_map = _load_thresholds()
        patients = _simulated_patients(threshold_map)
        alerts: List[Dict[str, Any]] = []

        for patient in patients:
            risk = patient["risk"]
            if risk["severity"] != "stable":
                alerts.append(
                    {
                        "patient_id": patient["patient_id"],
                        "severity": risk["severity"],
                        "message": risk["alert_message"],
                        "room": patient["room"],
                    }
                )

        _update_trend_history(patients)
        runtime_state["last_update"] = datetime.now(timezone.utc)

        avg = sum(_safe_float(p["risk"]["risk_score"]) for p in patients) / len(patients) if patients else 0.0
        generated_at = _utc_now_iso()
        return {
            "generated_at": generated_at,
            "patients": patients,
            "alerts": alerts,
            "summary": {
                "patient_count": len(patients),
                "open_alerts": len(alerts),
                "critical_alerts": sum(1 for alert in alerts if alert["severity"] == "critical"),
                "avg_risk_score": round(avg, 1),
            },
            "meta": {
                "platform_positioning": [
                    "HCP-facing",
                    "decision-support and workflow-support",
                    "assists further clinical evaluation",
                    "supports prioritization and command-center awareness",
                    "does not replace clinician judgment",
                    "not intended to diagnose, direct treatment, or independently trigger escalation",
                ],
                "intended_use_statement": INTENDED_USE_STATEMENT,
                "decision_support_disclaimer": INTENDED_USE_STATEMENT + " Review the basis, confidence, limitations, data freshness, what changed, what the software does not know, and hospital policy before acting.",
                "workflow_disclaimer": "Acknowledge, assign, escalate, and resolve controls record operational workflow state and user action logs only. They are not machine-issued medical orders, treatment directives, or mandatory escalation commands.",
                "limitations_banner": "Incomplete or delayed data may affect outputs. Clinicians must independently review the patient, and hospital policy governs escalation.",
                "pilot_version": PILOT_VERSION,
                "pilot_build_state": PILOT_BUILD_STATE,
                "supported_inputs": PILOT_SUPPORTED_INPUTS,
                "supported_outputs": PILOT_SUPPORTED_OUTPUTS,
                "avoid_claims": PILOT_AVOID_CLAIMS,
                "change_control": PILOT_CHANGE_CONTROL,
                "limitations": PILOT_LIMITATIONS_TEXT,
                "data_freshness": {
                    "generated_at": generated_at,
                    "age_seconds": 0,
                    "refresh_interval_seconds": 5,
                    "source_mode": "structured-vital-summary-pilot",
                },
            },
        }

    def _workflow_record(store: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        if patient_id not in store["records"]:
            store["records"][patient_id] = {
                "ack": False,
                "assigned": False,
                "escalated": False,
                "resolved": False,
                "state": "new",
                "assigned_label": "",
                "updated_at": _utc_now_iso(),
                "role": "",
            }
        return store["records"][patient_id]

    def _audit(store: Dict[str, Any], patient_id: str, action: str, role: str, note: str = "") -> None:
        patient_unit = ""
        for patient in _simulated_snapshot()["patients"]:
            if patient["patient_id"] == patient_id:
                patient_unit = _normalize_room_to_unit(patient.get("room", ""))
                break

        row = {
            "time": _utc_now_iso(),
            "patient_id": patient_id,
            "action": action,
            "role": role,
            "note": note,
            "unit": patient_unit or _current_unit_access(),
            "user": _current_user(),
        }
        store["audit_log"].append(row)
        store["audit_log"] = store["audit_log"][-200:]
        in_memory_audit.append(row)
        del in_memory_audit[:-200]

    def _filtered_patients(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        patients = snapshot["patients"]
        allowed_unit = _current_unit_access()
        role = _current_role()
        if role != "admin" and allowed_unit != "all":
            patients = [
                patient
                for patient in patients
                if _normalize_room_to_unit(patient.get("room", "")) == allowed_unit
            ]
        return patients

    def _filtered_alerts(snapshot: Dict[str, Any], patients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        alerts = snapshot["alerts"]
        allowed_ids = {patient["patient_id"] for patient in patients}
        return [alert for alert in alerts if alert.get("patient_id") in allowed_ids]

    def _visible_audit_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if _current_role() == "admin" or _current_unit_access() == "all":
            return rows[-100:]
        return [row for row in rows if row.get("unit") == _current_unit_access()][-100:]

    def _visible_workflow_records(records: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = _simulated_snapshot()
        visible_ids = {patient["patient_id"] for patient in _filtered_patients(snapshot)}
        if _current_role() == "admin" or _current_unit_access() == "all":
            return records
        return {patient_id: row for patient_id, row in records.items() if patient_id in visible_ids}

    def _validate_admin_login(email: str, password: str) -> str | None:
        admin_email = str(os.getenv("ADMIN_EMAIL", "")).strip().lower()
        allowed_admin_emails = {
            item.strip().lower()
            for item in os.getenv("ALLOWED_ADMIN_EMAILS", admin_email).split(",")
            if item.strip()
        }

        if not email or email not in allowed_admin_emails:
            return "Admin access denied for this email."

        admin_password_hash = str(os.getenv("ADMIN_PASSWORD_HASH", "")).strip()
        admin_password_plain = str(os.getenv("ADMIN_PASSWORD", "")).strip()

        if not password:
            return "Admin password is required."

        if admin_password_hash:
            try:
                if not check_password_hash(admin_password_hash, password):
                    return "Invalid admin password."
                return None
            except Exception:
                # Safe fallback when a plain password was mistakenly stored in ADMIN_PASSWORD_HASH
                if admin_password_plain and secrets.compare_digest(admin_password_plain, password):
                    return None
                return "Admin password hash is invalid. Move the plain password to ADMIN_PASSWORD or replace ADMIN_PASSWORD_HASH with a valid Werkzeug hash."

        if admin_password_plain:
            if not secrets.compare_digest(admin_password_plain, password):
                return "Invalid admin password."
            return None

        return "Admin password is not configured."

    @app.after_request
    def add_security_headers(response):
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: https:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; font-src 'self' data: https:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response

    @app.get("/")
    def home():
        return redirect("/command-center")

    @app.get("/deck")
    def deck():
        return _serve_deck("main")

    @app.get("/pitch-deck")
    def pitch_deck():
        return _serve_deck("main")

    @app.get("/pilot-deck")
    def pilot_deck():
        return _serve_deck("hospital")

    @app.get("/hospital-deck")
    def hospital_deck():
        return _serve_deck("hospital")

    @app.get("/investor-deck")
    def investor_deck():
        return _serve_deck("investor")

    @app.get("/executive-deck")
    def executive_deck():
        return _serve_deck("executive")

    @app.get("/legacy-deck")
    def legacy_deck():
        return _serve_deck("legacy")



    @app.get("/pilot-success-guide")
    def pilot_success_guide():
        return _serve_static_doc("pilot_success_guide")

    @app.get("/model-card")
    def model_card():
        return _serve_static_doc("model_card")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            form = request.form or {}
            full_name = str(form.get("full_name", "")).strip()
            email = str(form.get("email", "")).strip().lower()
            user_role = str(form.get("user_role", "viewer")).strip().lower()
            hospital_brand = str(form.get("hospital_brand", "early-risk-alert-ai")).strip().lower()
            assigned_unit = str(form.get("assigned_unit", "all")).strip().lower()
            admin_password = str(form.get("admin_password", "")).strip()

            if user_role not in ROLE_ACTIONS:
                user_role = "viewer"
            if hospital_brand not in HOSPITAL_BRANDS:
                hospital_brand = "early-risk-alert-ai"
            if assigned_unit not in VALID_UNITS:
                assigned_unit = "all"
            if not full_name or not email:
                return render_template_string(
                    LOGIN_HTML.replace("__ERROR__", "<div class='error'>Full name and email are required.</div>")
                )

            if user_role == "admin":
                admin_error = _validate_admin_login(email, admin_password)
                if admin_error:
                    return render_template_string(
                        LOGIN_HTML.replace("__ERROR__", f"<div class='error'>{admin_error}</div>")
                    )

            session.clear()
            session.permanent = True
            session["logged_in"] = True
            session["full_name"] = full_name
            session["email"] = email
            session["user_role"] = user_role
            session["hospital_brand"] = hospital_brand
            session["assigned_unit"] = "all" if user_role == "admin" else assigned_unit
            return redirect("/command-center")

        return render_template_string(LOGIN_HTML.replace("__ERROR__", ""))

    @app.route("/pilot-access", methods=["GET", "POST"])
    def pilot_access():
        if request.method == "POST":
            pilot_email = str(request.form.get("pilot_email", "")).strip().lower()
            account = PILOT_ACCOUNTS.get(pilot_email)

            if not account:
                return render_template_string(
                    PILOT_ACCESS_HTML.replace("__ERROR__", "<div class='error'>Pilot account not found.</div>")
                )

            session.clear()
            session.permanent = True
            session["logged_in"] = True
            session["full_name"] = account["full_name"]
            session["email"] = account["email"]
            session["user_role"] = account["user_role"]
            session["assigned_unit"] = "all" if account["user_role"] == "admin" else account["assigned_unit"]
            session["hospital_brand"] = account["hospital_brand"]
            return redirect("/command-center")

        return render_template_string(PILOT_ACCESS_HTML.replace("__ERROR__", ""))

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect("/login")

    @app.get("/command-center")
    @_login_required
    def command_center():
        return Response(COMMAND_CENTER_HTML, mimetype="text/html; charset=utf-8")

    @app.get("/api/access-context")
    @_login_required
    def access_context():
        brand = _current_brand()
        return jsonify(
            {
                "logged_in": _logged_in(),
                "role": _current_role(),
                "user_name": _current_user(),
                "assigned_unit": _current_unit_access(),
                "can_view_all_units": _current_role() == "admin" or _current_unit_access() == "all",
                "pilot_mode": True,
                "pilot_version": PILOT_VERSION,
                "pilot_build_state": PILOT_BUILD_STATE,
                "intended_use_statement": INTENDED_USE_STATEMENT,
                "pilot_docs_url": "/pilot-docs",
                "deck_url": "/deck",
                "hospital_deck_url": "/hospital-deck",
                "investor_deck_url": "/investor-deck",
                "executive_deck_url": "/executive-deck",
                **brand,
            }
        )

    @app.get("/api/workflow")
    @_login_required
    def get_workflow():
        store = _load_workflow()
        visible_records = _visible_workflow_records(store.get("records", {}))
        return jsonify(
            {
                "records": visible_records,
                "audit_log": _visible_audit_rows(store.get("audit_log", [])),
                "access_scope": {
                    "role": _current_role(),
                    "assigned_unit": _current_unit_access(),
                    "record_count": len(visible_records),
                },
            }
        )

    @app.post("/api/workflow/action")
    @_login_required
    def workflow_action():
        data = request.get_json(silent=True) or {}
        patient_id = str(data.get("patient_id", "")).strip()
        action = str(data.get("action", "")).strip().lower()
        note = str(data.get("note", "")).strip()
        role = _current_role()

        if not patient_id:
            return jsonify({"ok": False, "error": "patient_id required"}), 400

        snapshot = _simulated_snapshot()
        visible_ids = {patient["patient_id"] for patient in _filtered_patients(snapshot)}
        if patient_id not in visible_ids and _current_role() != "admin":
            return jsonify({"ok": False, "error": "patient not available in current scope"}), 403

        store = _load_workflow()
        record = _workflow_record(store, patient_id)

        if action == "ack":
            if not _has_permission("ack"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["ack"] = True
            record["state"] = "acknowledged"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "ACK", role, note or "Alert acknowledged")
        elif action == "assign_nurse":
            if not _has_permission("assign"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["assigned"] = True
            record["assigned_label"] = note or "Assigned Nurse"
            record["state"] = "assigned"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "ASSIGN", role, note or "Assigned Nurse")
        elif action == "escalate":
            if not _has_permission("escalate"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["escalated"] = True
            record["state"] = "escalated"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "ESCALATE", role, note or "Escalated patient")
        elif action == "resolve":
            if not _has_permission("resolve"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["resolved"] = True
            record["state"] = "resolved"
            record["role"] = role
            record["updated_at"] = _utc_now_iso()
            _audit(store, patient_id, "RESOLVE", role, note or "Resolved workflow")
        else:
            return jsonify({"ok": False, "error": "invalid action"}), 400

        _save_workflow(store)
        return jsonify({"ok": True, "record": record})

    @app.get("/api/audit")
    @_login_required
    def get_audit_logs():
        store = _load_workflow()
        combined = list(store.get("audit_log", [])) + list(in_memory_audit)
        unique: Dict[str, Dict[str, Any]] = {}
        for row in combined:
            key = "|".join(
                [
                    str(row.get("time", "")),
                    str(row.get("patient_id", "")),
                    str(row.get("action", "")),
                    str(row.get("role", "")),
                    str(row.get("note", "")),
                ]
            )
            unique[key] = row
        rows = list(unique.values())
        rows.sort(key=lambda item: item.get("time", ""), reverse=False)
        return jsonify(_visible_audit_rows(rows))

    @app.post("/api/action/<action>/<patient_id>")
    @_login_required
    def take_action(action: str, patient_id: str):
        action = str(action).strip().lower()
        action_map = {
            "ack": "ack",
            "assign": "assign_nurse",
            "escalate": "escalate",
            "resolve": "resolve",
        }
        mapped_action = action_map.get(action)
        if not mapped_action:
            return jsonify({"ok": False, "error": "invalid action"}), 400

        note_map = {
            "ack": "Alert acknowledged",
            "assign_nurse": "Assigned Nurse",
            "escalate": "Escalated patient",
            "resolve": "Resolved workflow",
        }

        data = {"patient_id": patient_id, "action": mapped_action, "note": note_map[mapped_action]}
        original_get_json = request.get_json

        def _patched_get_json(*args, **kwargs):
            return data

        request.get_json = _patched_get_json  # type: ignore[assignment]
        try:
            return workflow_action()
        finally:
            request.get_json = original_get_json  # type: ignore[assignment]

    @app.get("/api/thresholds")
    @_login_required
    def get_thresholds():
        return jsonify(_load_thresholds())

    @app.post("/api/thresholds")
    @_login_required
    def update_thresholds():
        if _current_role() != "admin":
            return jsonify({"ok": False, "error": "permission denied"}), 403

        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict) or not payload:
            return jsonify({"ok": False, "error": "threshold payload required"}), 400

        current = _load_thresholds()
        changed_units: List[str] = []

        for unit, values in payload.items():
            if unit not in DEFAULT_THRESHOLDS or not isinstance(values, dict):
                continue

            spo2_low = round(_safe_float(values.get("spo2_low"), current[unit]["spo2_low"]), 1)
            hr_high = round(_safe_float(values.get("hr_high"), current[unit]["hr_high"]), 1)
            sbp_high = round(_safe_float(values.get("sbp_high"), current[unit]["sbp_high"]), 1)

            if not (80 <= spo2_low <= 100 and 60 <= hr_high <= 180 and 90 <= sbp_high <= 220):
                return jsonify({"ok": False, "error": f"invalid threshold range for {unit}"}), 400

            current[unit] = {
                "spo2_low": spo2_low,
                "hr_high": hr_high,
                "sbp_high": sbp_high,
            }
            changed_units.append(unit)

        if not changed_units:
            return jsonify({"ok": False, "error": "no valid threshold updates supplied"}), 400

        persisted = {unit: values for unit, values in current.items() if unit in DEFAULT_THRESHOLDS}
        _save_thresholds(persisted)
        return jsonify({"ok": True, "updated_units": changed_units, "thresholds": current})

    @app.get("/api/trends/<patient_id>")
    @_login_required
    def get_patient_trends(patient_id: str):
        snapshot = _simulated_snapshot()
        visible_ids = {patient["patient_id"] for patient in _filtered_patients(snapshot)}
        if patient_id not in visible_ids and _current_role() != "admin":
            return jsonify({"ok": False, "error": "patient not available in current scope"}), 403

        trends = _load_trends().get(patient_id, [])
        response = []
        for row in trends[-24:]:
            response.append(
                {
                    "time": row.get("time"),
                    "hr": row.get("hr"),
                    "spo2": row.get("spo2"),
                    "risk": row.get("risk"),
                }
            )
        return jsonify(response)

    @app.get("/api/system-health")
    @_login_required
    def api_system_health():
        snapshot = _simulated_snapshot()
        return jsonify(
            {
                "status": "ok",
                "time": _utc_now_iso(),
                "patient_count": snapshot["summary"]["patient_count"],
                "open_alerts": snapshot["summary"]["open_alerts"],
                "critical_alerts": snapshot["summary"]["critical_alerts"],
            }
        )

    @app.get("/api/system/health")
    @_login_required
    def api_system_health_legacy():
        now = datetime.now(timezone.utc)
        delta = (now - runtime_state["last_update"]).total_seconds()
        status = "connected" if delta < 12 else "degraded"
        return jsonify(
            {
                "status": status,
                "last_update": runtime_state["last_update"].isoformat(),
                "seconds_since_update": round(delta, 2),
            }
        )

    @app.get("/api/v1/live-snapshot")
    @_login_required
    def live_snapshot():
        snapshot = _simulated_snapshot()
        patients = _filtered_patients(snapshot)
        alerts = _filtered_alerts(snapshot, patients)
        avg_risk = (
            round(sum(_safe_float(patient.get("risk", {}).get("risk_score", 0)) for patient in patients) / len(patients), 1)
            if patients
            else 0.0
        )

        return jsonify(
            {
                "generated_at": snapshot["generated_at"],
                "patients": patients,
                "alerts": alerts,
                "summary": {
                    "patient_count": len(patients),
                    "open_alerts": len(alerts),
                    "critical_alerts": sum(1 for alert in alerts if str(alert.get("severity", "")).lower() == "critical"),
                    "avg_risk_score": avg_risk,
                },
                "access": {
                    "role": _current_role(),
                    "assigned_unit": _current_unit_access(),
                },
                "meta": snapshot.get("meta", {}),
            }
        )

    @app.get("/api/command-center-stream")
    @_login_required
    def command_center_stream():
        def generate():
            while True:
                snapshot = _simulated_snapshot()
                patients = _filtered_patients(snapshot)
                alerts = _filtered_alerts(snapshot, patients)
                payload = {
                    "generated_at": snapshot["generated_at"],
                    "patients": patients,
                    "alerts": alerts,
                    "meta": snapshot.get("meta", {}),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(5)

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.route("/hospital-demo", methods=["GET", "POST"])
    def hospital_demo():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "email": request.form.get("email", "").strip(),
                "organization_type": request.form.get("organization_type", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "role": request.form.get("role", "").strip(),
                "department_unit": request.form.get("department_unit", "").strip(),
                "evaluation_interest": request.form.get("evaluation_interest", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "additional_notes": request.form.get("additional_notes", "").strip(),
                "acknowledgment": request.form.get("acknowledgment", "").strip(),
                "lead_score": 0,
            }
            scoring_payload = {
                "timeline": payload["timeline"],
                "organization": payload["organization"],
                "role": payload["role"],
                "message": f"{payload['evaluation_interest']} {payload['additional_notes']}".strip(),
            }
            payload["lead_score"] = _lead_score(scoring_payload, "hospital")
            _append_row(hospital_file, payload)
            return render_template_string(
                THANK_YOU_HTML
                .replace("__MESSAGE__", "Your hospital demonstration request has been received.")
                .replace(
                    "__DETAILS__",
                    _detail_html(
                        payload,
                        [
                            "full_name",
                            "email",
                            "organization_type",
                            "organization",
                            "role",
                            "department_unit",
                            "evaluation_interest",
                            "timeline",
                        ],
                    ),
                )
            )

        fields = """
        <div class="field"><label>Work Email</label><input type="email" name="email" placeholder="Enter your work email" required></div>
        <div class="field"><label>Organization Type</label>
          <select name="organization_type" required>
            <option value="Hospital">Hospital</option>
            <option value="Health System">Health System</option>
            <option value="Remote Patient Monitoring Program">Remote Patient Monitoring Program</option>
            <option value="Care Network">Care Network</option>
            <option value="Clinical Operations Team">Clinical Operations Team</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div class="field"><label>Organization Name</label><input name="organization" placeholder="Enter your hospital, health system, or organization name" required></div>
        <div class="field"><label>Full Name</label><input name="full_name" placeholder="Enter your full name" required></div>
        <div class="field"><label>Title / Role</label><input name="role" placeholder="Enter your title or role" required></div>
        <div class="field"><label>Department / Unit</label>
          <select name="department_unit" required>
            <option value="ICU">ICU</option>
            <option value="Telemetry">Telemetry</option>
            <option value="Stepdown">Stepdown</option>
            <option value="Med-Surg">Med-Surg</option>
            <option value="Remote Monitoring">Remote Monitoring</option>
            <option value="Operations / Command Center">Operations / Command Center</option>
            <option value="Executive Leadership">Executive Leadership</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div class="field"><label>What are you interested in evaluating?</label>
          <select name="evaluation_interest" required>
            <option value="Patient prioritization support">Patient prioritization support</option>
            <option value="Command-center operational awareness">Command-center operational awareness</option>
            <option value="Explainable review-basis visibility">Explainable review-basis visibility</option>
            <option value="Workflow-state and audit visibility">Workflow-state and audit visibility</option>
            <option value="Controlled pilot evaluation">Controlled pilot evaluation</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div class="field"><label>Pilot Timeline</label>
          <select name="timeline" required>
            <option value="Immediate">Immediate</option>
            <option value="30-60 days">30-60 days</option>
            <option value="This quarter">This quarter</option>
            <option value="Exploratory">Exploratory</option>
          </select>
        </div>
        <div class="field full"><label>Additional Notes</label><textarea name="additional_notes" placeholder="Share your pilot goals, care setting, or workflow interests"></textarea></div>
        <div class="field full">
          <label>Acknowledgment</label>
          <div class="check">
            <input type="checkbox" name="acknowledgment" value="yes" required>
            <span>I understand Early Risk Alert AI is intended for HCP-facing decision-support and workflow-support pilot evaluation. It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.</span>
          </div>
        </div>
        """
        page = FORM_HTML.replace("__TITLE__", "Request a Live Command Center Demonstration")
        page = page.replace("__HEADING__", "Request a Live Command Center Demonstration")
        page = page.replace(
            "__COPY__",
            "Schedule a guided demonstration of Early Risk Alert AI’s HCP-facing decision-support and workflow-support platform for monitored patient visibility, patient prioritization support, explainable review context, and command-center operational awareness.",
        )
        page = page.replace("__FIELDS__", fields)
        page = page.replace("__BUTTON__", "Request Demo Access")
        return render_template_string(page)

    @app.route("/executive-walkthrough", methods=["GET", "POST"])
    def executive_walkthrough():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "title": request.form.get("title", "").strip(),
                "email": request.form.get("email", "").strip(),
                "leadership_area": request.form.get("leadership_area", "").strip(),
                "review_focus": request.form.get("review_focus", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
                "acknowledgment": request.form.get("acknowledgment", "").strip(),
                "lead_score": 0,
            }
            payload["lead_score"] = _lead_score(payload, "executive")
            _append_row(exec_file, payload)
            return render_template_string(
                THANK_YOU_HTML
                .replace("__MESSAGE__", "Your executive walkthrough request has been received.")
                .replace(
                    "__DETAILS__",
                    _detail_html(
                        payload,
                        ["full_name", "organization", "title", "email", "leadership_area", "review_focus", "timeline"],
                    ),
                )
            )

        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" placeholder="Enter your full name" required></div>
        <div class="field"><label>Organization Name</label><input name="organization" placeholder="Enter your organization name" required></div>
        <div class="field"><label>Title</label><input name="title" placeholder="Enter your title" required></div>
        <div class="field"><label>Work Email</label><input type="email" name="email" placeholder="Enter your work email" required></div>
        <div class="field"><label>Leadership Area</label>
          <select name="leadership_area" required>
            <option value="Clinical Leadership">Clinical Leadership</option>
            <option value="Hospital Operations">Hospital Operations</option>
            <option value="Digital Health">Digital Health</option>
            <option value="Innovation">Innovation</option>
            <option value="Executive Administration">Executive Administration</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div class="field"><label>Timeline</label>
          <select name="timeline" required>
            <option value="Immediate">Immediate</option>
            <option value="30-60 days">30-60 days</option>
            <option value="This quarter">This quarter</option>
            <option value="Exploratory">Exploratory</option>
          </select>
        </div>
        <div class="field full"><label>What would you like reviewed?</label>
          <select name="review_focus" required>
            <option value="Pilot readiness">Pilot readiness</option>
            <option value="Platform overview">Platform overview</option>
            <option value="Operational workflow support">Operational workflow support</option>
            <option value="Command-center model">Command-center model</option>
            <option value="Security and pilot controls">Security and pilot controls</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div class="field full"><label>Additional Notes</label><textarea name="message" placeholder="Share any leadership, pilot, or operational review priorities"></textarea></div>
        <div class="field full">
          <label>Acknowledgment</label>
          <div class="check">
            <input type="checkbox" name="acknowledgment" value="yes" required>
            <span>I understand Early Risk Alert AI is intended for controlled pilot evaluation and hospital-facing workflow support. It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.</span>
          </div>
        </div>
        """
        page = FORM_HTML.replace("__TITLE__", "Request an Executive Walkthrough")
        page = page.replace("__HEADING__", "Request an Executive Walkthrough")
        page = page.replace(
            "__COPY__",
            "Request a leadership-level walkthrough of Early Risk Alert AI’s hospital-facing platform, pilot readiness, operational workflow support, and command-center visibility model.",
        )
        page = page.replace("__FIELDS__", fields)
        page = page.replace("__BUTTON__", "Request Executive Walkthrough")
        return render_template_string(page)

    @app.route("/investor-intake", methods=["GET", "POST"])
    def investor_intake():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "role": request.form.get("role", "").strip(),
                "email": request.form.get("email", "").strip(),
                "stage": request.form.get("stage", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "interest_area": request.form.get("interest_area", "").strip(),
                "message": request.form.get("message", "").strip(),
                "acknowledgment": request.form.get("acknowledgment", "").strip(),
                "lead_score": 0,
            }
            payload["lead_score"] = _lead_score(payload, "investor")
            _append_row(investor_file, payload)
            return render_template_string(
                THANK_YOU_HTML
                .replace("__MESSAGE__", "Your investor access request has been received.")
                .replace(
                    "__DETAILS__",
                    _detail_html(payload, ["full_name", "organization", "role", "email", "stage", "timeline", "interest_area"]),
                )
            )

        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" placeholder="Enter your full name" required></div>
        <div class="field"><label>Organization</label><input name="organization" placeholder="Enter your organization" required></div>
        <div class="field"><label>Role</label><input name="role" placeholder="Enter your role" required></div>
        <div class="field"><label>Work Email</label><input type="email" name="email" placeholder="Enter your work email" required></div>
        <div class="field"><label>Investor Stage</label>
          <select name="stage" required>
            <option value="Angel">Angel</option>
            <option value="Seed">Seed</option>
            <option value="Institutional">Institutional</option>
            <option value="Strategic">Strategic</option>
          </select>
        </div>
        <div class="field"><label>Timeline</label>
          <select name="timeline" required>
            <option value="Immediate">Immediate</option>
            <option value="30-60 days">30-60 days</option>
            <option value="This quarter">This quarter</option>
            <option value="Exploratory">Exploratory</option>
          </select>
        </div>
        <div class="field full"><label>Areas of Interest</label>
          <select name="interest_area" required>
            <option value="Platform overview">Platform overview</option>
            <option value="Hospital pilot model">Hospital pilot model</option>
            <option value="Commercial model">Commercial model</option>
            <option value="Market opportunity">Market opportunity</option>
            <option value="Founder discussion">Founder discussion</option>
            <option value="Partnership discussion">Partnership discussion</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div class="field full"><label>Message</label><textarea name="message" placeholder="Share your investor or partnership interests"></textarea></div>
        <div class="field full">
          <label>Acknowledgment</label>
          <div class="check">
            <input type="checkbox" name="acknowledgment" value="yes" required>
            <span>I understand Early Risk Alert AI is positioned as an HCP-facing decision-support and workflow-support platform for controlled pilot evaluation and hospital-facing workflow support.</span>
          </div>
        </div>
        """
        page = FORM_HTML.replace("__TITLE__", "Request Investor Access")
        page = page.replace("__HEADING__", "Request Investor Access")
        page = page.replace(
            "__COPY__",
            "Request investor materials, platform overview, and partnership discussion access for Early Risk Alert AI’s HCP-facing decision-support and workflow-support platform.",
        )
        page = page.replace("__FIELDS__", fields)
        page = page.replace("__BUTTON__", "Request Investor Access")
        return render_template_string(page)

    @app.get("/api/pilot-governance")
    @_login_required
    def pilot_governance():
        return jsonify(
            {
                "pilot_version": PILOT_VERSION,
                "pilot_build_state": PILOT_BUILD_STATE,
                "intended_use_statement": INTENDED_USE_STATEMENT,
                "support_language": PILOT_SUPPORT_LANGUAGE,
                "supported_inputs": PILOT_SUPPORTED_INPUTS,
                "supported_outputs": PILOT_SUPPORTED_OUTPUTS,
                "avoid_claims": PILOT_AVOID_CLAIMS,
                "change_control": PILOT_CHANGE_CONTROL,
                "limitations": PILOT_LIMITATIONS_TEXT,
                "risk_register": PILOT_RISK_REGISTER,
                "vnv_lite": PILOT_VNV_LITE,
                "release_notes": PILOT_RELEASE_NOTES,
                "approved_claims": PILOT_APPROVED_CLAIMS,
                "banned_claims": PILOT_BANNED_CLAIMS,
                "claims_control_sheet": PILOT_CLAIMS_CONTROL_SHEET,
                "document_control_index": PILOT_DOCUMENT_CONTROL_INDEX,
                "complaint_issue_log": PILOT_COMPLAINT_ISSUE_LOG,
                "escalation_process": PILOT_ESCALATION_PROCESS,
                "change_approval_log": PILOT_CHANGE_APPROVAL_LOG,
                "cybersecurity_summary": PILOT_CYBERSECURITY_SUMMARY,
                "advisory_structure": PILOT_ADVISORY_STRUCTURE,
                "support_owners": PILOT_SUPPORT_OWNERS,
                "security_program_documents": PILOT_SECURITY_PROGRAM_DOCUMENTS,
                "insurance_readiness_controls": PILOT_INSURANCE_READINESS_CONTROLS,
                "mfa_access_log": PILOT_MFA_ACCESS_EVIDENCE_LOG,
                "backup_restore_log": PILOT_BACKUP_RESTORE_LOG,
                "patch_log": PILOT_PATCH_LOG,
                "access_review_log": PILOT_ACCESS_REVIEW_LOG,
                "tabletop_log": PILOT_TABLETOP_LOG,
                "training_ack_log": PILOT_TRAINING_ACK_LOG,
                "site_packet_template": PILOT_SITE_PACKET_TEMPLATE,
                "route_status": _route_status_snapshot(),
                "user_provisioning_policy": PILOT_USER_PROVISIONING_POLICY,
                "data_retention_policy": PILOT_DATA_RETENTION_POLICY,
                "pilot_scope_document": PILOT_SCOPE_DOCUMENT,
                "training_use_instructions": PILOT_TRAINING_USE_INSTRUCTIONS,
                "dated_validation_evidence": PILOT_VALIDATION_EVIDENCE,
                "document_control_index": PILOT_DOCUMENT_CONTROL_INDEX,
            }
        )

    @app.get("/api/platform-positioning")
    @_login_required
    def platform_positioning():
        return jsonify(
            {
                "pilot_version": PILOT_VERSION,
                "pilot_build_state": PILOT_BUILD_STATE,
                "intended_use_statement": INTENDED_USE_STATEMENT,
                "support_language": PILOT_SUPPORT_LANGUAGE,
                "supported_inputs": PILOT_SUPPORTED_INPUTS,
                "supported_outputs": PILOT_SUPPORTED_OUTPUTS,
                "avoid_claims": PILOT_AVOID_CLAIMS,
                "limitations": PILOT_LIMITATIONS_TEXT,
                "approved_claims": PILOT_APPROVED_CLAIMS,
                "banned_claims": PILOT_BANNED_CLAIMS,
                "claims_control_sheet": PILOT_CLAIMS_CONTROL_SHEET,
                "document_control_index": PILOT_DOCUMENT_CONTROL_INDEX,
                "advisory_structure": PILOT_ADVISORY_STRUCTURE,
                "support_owners": PILOT_SUPPORT_OWNERS,
                "security_program_documents": PILOT_SECURITY_PROGRAM_DOCUMENTS,
                "insurance_readiness_controls": PILOT_INSURANCE_READINESS_CONTROLS,
                "route_status": _route_status_snapshot(),
            }
        )

    @app.get("/api/pilot-readiness")
    @_login_required
    def pilot_readiness():
        return jsonify(
            {
                "pilot_version": PILOT_VERSION,
                "pilot_build_state": PILOT_BUILD_STATE,
                "route_status": _route_status_snapshot(),
                "support_owners": PILOT_SUPPORT_OWNERS,
                "mfa_access_log": PILOT_MFA_ACCESS_EVIDENCE_LOG,
                "backup_restore_log": PILOT_BACKUP_RESTORE_LOG,
                "patch_log": PILOT_PATCH_LOG,
                "access_review_log": PILOT_ACCESS_REVIEW_LOG,
                "tabletop_log": PILOT_TABLETOP_LOG,
                "training_ack_log": PILOT_TRAINING_ACK_LOG,
                "site_packet_template": PILOT_SITE_PACKET_TEMPLATE,
            }
        )

    @app.get("/pilot-docs")
    @_login_required
    def pilot_docs():
        def _render_simple_list(items: List[str]) -> str:
            return "".join(
                f"<div style='padding:10px 12px;border:1px solid rgba(255,255,255,.08);border-radius:14px;background:rgba(255,255,255,.03);margin-bottom:10px;color:#dce9ff'>{item}</div>"
                for item in items
            )

        def _render_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
            out = ["<table style='width:100%;border-collapse:collapse'>", "<thead><tr>"]
            for header in headers:
                out.append(f"<th style='padding:12px;text-align:left;color:#9adfff;border-bottom:1px solid rgba(255,255,255,.08)'>{header.replace('_', ' ').title()}</th>")
            out.append("</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>")
                for header in headers:
                    value = row.get(header, "")
                    if isinstance(value, list):
                        value = "<br>".join(f"• {item}" for item in value)
                    out.append(f"<td style='padding:12px;vertical-align:top;border-bottom:1px solid rgba(255,255,255,.08);color:#dce9ff'>{value}</td>")
                out.append("</tr>")
            out.append("</tbody></table>")
            return "".join(out)

        html = f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Pilot Docs — Early Risk Alert AI</title>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            body{{margin:0;padding:24px;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528)}}
            .wrap{{max-width:1280px;margin:0 auto}}
            .card{{border:1px solid rgba(255,255,255,.08);border-radius:24px;background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));padding:24px;margin-bottom:18px;box-shadow:0 20px 60px rgba(0,0,0,.28)}}
            .btn{{display:inline-flex;align-items:center;justify-content:center;padding:12px 16px;border-radius:16px;font-weight:900;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;text-decoration:none}}
            .sub{{color:#9fb4d6;line-height:1.7}}
            .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
            .pill{{display:inline-flex;align-items:center;padding:10px 14px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);font-size:12px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;margin-right:8px;margin-bottom:8px}}
            @media (max-width:840px){{.grid{{grid-template-columns:1fr}}}}
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="card">
              <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center">
                <div>
                  <div class="pill">Pilot Docs</div>
                  <div class="pill">Version {PILOT_VERSION}</div>
                  <h1 style="margin:12px 0 10px;font-size:42px;line-height:.95;letter-spacing:-.05em">Stable Pilot Positioning Bundle</h1>
                  <div class="sub">Freeze one intended-use statement everywhere, keep outputs supportive rather than directive, and keep explainability, limitations, scoping, audit visibility, and operating controls easy to review.</div>
                </div>
                <div style="display:flex;gap:12px;flex-wrap:wrap">
                  <a class="btn" href="/command-center">Back to Command Center</a>
                  <a class="btn" href="/pilot-docs">Open Pilot Docs</a>
                  <a class="btn" href="/deck" target="_blank" rel="noopener">Open Pitch Deck</a>
                  <a class="btn" href="/pilot-success-guide">Pilot Success Guide</a>
                  <a class="btn" href="/model-card">Model Card</a>
                  <a class="btn" href="/investor-deck" target="_blank" rel="noopener">Investor Deck</a>
                </div>
              </div>
            </div>

            <div class="card">
              <h2 style="margin:0 0 10px;font-size:30px">Frozen Intended Use</h2>
              <div class="sub" style="font-size:18px;color:#eef4ff">{INTENDED_USE_STATEMENT}</div>
              <div style="margin-top:12px;display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;background:rgba(181,140,255,.14);border:1px solid rgba(181,140,255,.28);font-weight:900;color:#f0e5ff">{PILOT_BUILD_STATE} · {PILOT_VERSION}</div>
            </div>

            <div class="grid">
              <div class="card"><h2 style="margin:0 0 10px;font-size:26px">Approved Support Language</h2><div class="sub">{_render_simple_list(PILOT_SUPPORT_LANGUAGE)}</div></div>
              <div class="card"><h2 style="margin:0 0 10px;font-size:26px">Visible Limitations</h2><div class="sub">{_render_simple_list(PILOT_LIMITATIONS_TEXT)}</div></div>
            </div>

            <div class="grid">
              <div class="card"><h2 style="margin:0 0 10px;font-size:26px">Supported Inputs</h2><div class="sub">{_render_simple_list(PILOT_SUPPORTED_INPUTS)}</div></div>
              <div class="card"><h2 style="margin:0 0 10px;font-size:26px">Supported Outputs</h2><div class="sub">{_render_simple_list(PILOT_SUPPORTED_OUTPUTS)}</div></div>
            </div>

            <div class="grid">
              <div class="card"><h2 style="margin:0 0 10px;font-size:26px">Approved Claims</h2><div class="sub">{_render_simple_list(PILOT_APPROVED_CLAIMS)}</div></div>
              <div class="card"><h2 style="margin:0 0 10px;font-size:26px">Banned Claims</h2><div class="sub">{_render_simple_list(PILOT_BANNED_CLAIMS)}</div></div>
            </div>

            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Claims Control Sheet</h2>{_render_table(PILOT_CLAIMS_CONTROL_SHEET, ["claim", "status", "category", "owner", "last_reviewed", "rationale"])}</div>
            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Document Control Index</h2>{_render_table(PILOT_DOCUMENT_CONTROL_INDEX, ["document_name", "version", "owner", "last_updated", "status", "pilot_applicability"])}</div>
            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Risk Register</h2>{_render_table(PILOT_RISK_REGISTER, ["id", "area", "risk", "mitigation", "owner", "status"])}</div>
            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Complaint / Issue Log</h2>{_render_table(PILOT_COMPLAINT_ISSUE_LOG, ["id", "date_opened", "source", "category", "issue", "severity", "owner", "investigation_summary", "corrective_action", "status", "escalation_level", "next_step", "closure_date"])}</div>
            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Issue Escalation Process</h2>{_render_table(PILOT_ESCALATION_PROCESS, ["stage", "trigger", "owner", "target_time", "action"])}</div>
            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Change Approval Log</h2>{_render_table(PILOT_CHANGE_APPROVAL_LOG, ["version", "approval_date", "approver_name", "reason_for_approval", "what_changed", "impact_assessment", "status"])}</div>
            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Cybersecurity Summary</h2>{_render_table(PILOT_CYBERSECURITY_SUMMARY, ["domain", "control", "summary", "evidence", "status"])}</div>

            <div class="grid">
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Advisory & Support Structure</h2>{_render_table(PILOT_ADVISORY_STRUCTURE, ["name", "title", "scope", "status"])}</div>
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Insurance-Readiness Controls</h2>{_render_table(PILOT_INSURANCE_READINESS_CONTROLS, ["control", "requirement", "documentation", "status"])}</div>
            </div>

            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Named Support Owners</h2>{_render_table(PILOT_SUPPORT_OWNERS, ["area", "owner", "title", "coverage", "status"])}</div>
            <div class="grid">
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">MFA / Access Evidence Log</h2>{_render_table(PILOT_MFA_ACCESS_EVIDENCE_LOG, ["control", "implementation", "owner", "last_reviewed", "evidence", "status"])}</div>
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Backup / Restore Log</h2>{_render_table(PILOT_BACKUP_RESTORE_LOG, ["system", "owner", "backup_scope", "last_tested", "evidence", "notes", "status"])}</div>
            </div>
            <div class="grid">
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Patch Log</h2>{_render_table(PILOT_PATCH_LOG, ["component", "owner", "cadence", "last_patched", "evidence", "status"])}</div>
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Access Review Log</h2>{_render_table(PILOT_ACCESS_REVIEW_LOG, ["scope", "owner", "last_reviewed", "evidence", "status"])}</div>
            </div>
            <div class="grid">
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Tabletop Exercise Log</h2>{_render_table(PILOT_TABLETOP_LOG, ["exercise_name", "owner", "exercise_date", "scenario", "evidence", "status"])}</div>
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Training Acknowledgment Log</h2>{_render_table(PILOT_TRAINING_ACK_LOG, ["audience", "owner", "ack_date", "evidence", "status"])}</div>
            </div>

            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Security Program Documents</h2>{_render_table(PILOT_SECURITY_PROGRAM_DOCUMENTS, ["document_name", "version", "owner", "last_updated", "status", "pilot_applicability"])}</div>
            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Route + Document Status Snapshot</h2>{_render_table([_route_status_snapshot()], ["checked_at", "routes_checked", "routes_available", "missing_routes", "docs_checked", "docs_available", "documents"])}</div>

            <div class="grid">
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">User Provisioning / Deprovisioning Policy</h2>{_render_table(PILOT_USER_PROVISIONING_POLICY, ["step", "policy", "owner", "sla", "evidence"])}</div>
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Data Retention / Deletion / Return Policy</h2>{_render_table(PILOT_DATA_RETENTION_POLICY, ["topic", "policy", "owner", "note", "pilot_applicability"])}</div>
            </div>

            <div class="grid">
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Pilot Agreement / Scope Document</h2>{_render_table(PILOT_SCOPE_DOCUMENT, ["section", "summary"])}</div>
              <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Training / Use Instructions</h2>{_render_table(PILOT_TRAINING_USE_INSTRUCTIONS, ["step", "instruction", "owner"])}</div>
            </div>
            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Site-Specific Pilot Packet Template</h2>{_render_table(PILOT_SITE_PACKET_TEMPLATE, ["section", "summary", "owner", "status"])}</div>

            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Dated Validation Evidence</h2>{_render_table(PILOT_VALIDATION_EVIDENCE, ["date_tested", "tested_by", "test_case_id", "test_case", "expected_result", "actual_result", "screenshot_evidence", "status"])}</div>
            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">V&amp;V-Lite Sheet</h2>{_render_table(PILOT_VNV_LITE, ["id", "check", "method", "evidence", "status"])}</div>
            <div class="card"><h2 style="margin:0 0 12px;font-size:28px">Release Notes</h2>{_render_table(PILOT_RELEASE_NOTES, ["version", "date", "summary", "changes"])}</div>
          </div>
        </body>
        </html>
        """
        return render_template_string(html)

    @app.get("/admin/review")
    @_admin_required
    def admin_review():
        hospitals = _read_json_list(hospital_file)
        executives = _read_json_list(exec_file)
        investors = _read_json_list(investor_file)

        def _table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
            if not rows:
                return "<p style='color:#9fb4d6;'>No submissions yet.</p>"

            out = ["<table style='width:100%;border-collapse:collapse'><thead><tr>"]
            for header in headers:
                out.append(
                    f"<th style='padding:12px;text-align:left;color:#9adfff;border-bottom:1px solid rgba(255,255,255,.08);'>{header.replace('_',' ')}</th>"
                )
            out.append("</tr></thead><tbody>")
            for row in rows[::-1]:
                out.append("<tr>")
                for header in headers:
                    out.append(
                        f"<td style='padding:12px;border-bottom:1px solid rgba(255,255,255,.08);'>{row.get(header, '')}</td>"
                    )
                out.append("</tr>")
            out.append("</tbody></table>")
            return "".join(out)

        html = f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Admin Review — Early Risk Alert AI</title>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            body{{margin:0;padding:24px;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528)}}
            .wrap{{max-width:1200px;margin:0 auto}}
            .card{{border:1px solid rgba(255,255,255,.08);border-radius:24px;background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));padding:24px;margin-bottom:18px}}
            .btn{{display:inline-flex;align-items:center;justify-content:center;padding:12px 16px;border-radius:16px;font-weight:900;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;text-decoration:none}}
            .top{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center}}
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="card">
              <div class="top">
                <div>
                  <h1 style="margin:0 0 8px;font-size:42px;line-height:.95;">Admin Review</h1>
                  <div style="color:#9fb4d6;">Role: {_current_role()} · User: {_current_user()}</div>
                </div>
                <div style="display:flex;gap:12px;flex-wrap:wrap">
                  <a class="btn" href="/command-center">Back to Command Center</a>
                  <a class="btn" href="/deck" target="_blank" rel="noopener">Open Pitch Deck</a>
                  <a class="btn" href="/pilot-docs">Pilot Docs</a>
                </div>
              </div>
            </div>

            <div class="card"><h2>Hospital Demo Requests</h2>{_table(hospitals, ["submitted_at", "full_name", "email", "organization_type", "organization", "role", "department_unit", "evaluation_interest", "timeline", "lead_score", "status"])}</div>
            <div class="card"><h2>Executive Walkthrough Requests</h2>{_table(executives, ["submitted_at", "full_name", "organization", "title", "email", "leadership_area", "review_focus", "timeline", "lead_score", "status"])}</div>
            <div class="card"><h2>Investor Access Requests</h2>{_table(investors, ["submitted_at", "full_name", "organization", "role", "email", "stage", "timeline", "interest_area", "lead_score", "status"])}</div>
          </div>
        </body>
        </html>
        """
        return render_template_string(html)

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "service": "early-risk-alert-ai", "time": _utc_now_iso(), "version": PILOT_VERSION, "build_state": PILOT_BUILD_STATE})

    return app

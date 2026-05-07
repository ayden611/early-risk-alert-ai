#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(".")
DATE = "2026-05-07"
LOCK_ID = "stable-evidence-lock-2026-05-07"
LOCK_TITLE = "Early Risk Alert AI Stable Evidence Lock — May 7, 2026"

DOC_DIR = ROOT / "docs" / "evidence_lock"
DATA_DIR = ROOT / "data" / "evidence_lock"
GOV_DIR = ROOT / "docs" / "governance" / "evidence"

DOC_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
GOV_DIR.mkdir(parents=True, exist_ok=True)

generated_at = datetime.now(timezone.utc).isoformat()

INTENDED_USE = (
    "Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform "
    "designed to help authorized health care professionals identify patients who may warrant "
    "further clinical evaluation, support patient prioritization, and improve command-center "
    "operational awareness. It does not replace clinician judgment and is not intended to "
    "diagnose, direct treatment, or independently trigger escalation."
)

SAFE_HIRID_WORDING = (
    "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation."
)

APPROVED_PUBLIC_CLAIMS = [
    "HCP-facing decision-support and workflow-support platform.",
    "Designed to support authorized health care professionals with patient prioritization and command-center operational awareness.",
    "Provides explainable context such as priority tier, queue rank, primary driver, trend direction, and retrospective lead-time context.",
    "Supports controlled pilot evaluation and operational review workflows.",
    "Uses retrospective, aggregate-only validation evidence for current evidence materials.",
    "Workflow actions are operational state tracking only and do not independently trigger clinical escalation.",
    "HiRID access approved; HiRID retrospective aggregate validation pending local evaluation."
]

BANNED_OR_RESTRICTED_CLAIMS = [
    "Do not say the platform is clinically validated.",
    "Do not say the platform predicts deterioration as a clinical claim.",
    "Do not say the platform prevents adverse events.",
    "Do not say the platform replaces standard monitoring.",
    "Do not say the platform replaces clinician judgment.",
    "Do not say the platform independently triggers escalation.",
    "Do not say the platform diagnoses, treats, or directs treatment.",
    "Do not say the platform is FDA cleared, FDA approved, or FDA exempt.",
    "Do not publish HiRID performance, FPR, detection rate, lead time, or final validation language until local evaluation is complete and reviewed.",
    "Do not upload raw rows, patient-level outputs, timestamps, identifiers, or restricted dataset files to any public site, repo, demo, investor page, or production platform."
]

EVIDENCE_BASELINE = {
    "lock_id": LOCK_ID,
    "lock_title": LOCK_TITLE,
    "date": DATE,
    "generated_at_utc": generated_at,
    "product_stage": "Pilot-stage command-center platform.",
    "regulatory_posture": "Conservative HCP-facing decision-support/workflow-support framing. Not diagnosis, treatment direction, independent escalation, FDA clearance, FDA approval, or prospective clinical validation.",
    "intended_use_lock": INTENDED_USE,
    "evidence_state": {
        "current_status": "Retrospective aggregate evidence only.",
        "not_currently_claimed": [
            "Prospective clinical validation",
            "Clinical outcome improvement",
            "Real-time clinical deployment validation",
            "FDA clearance or approval",
            "Final HiRID performance"
        ],
        "source_of_truth_rule": "Public or external-facing evidence numbers should be reconciled against the locked validation registry and aggregate evidence files before publication."
    },
    "datasets": {
        "MIMIC_IV": {
            "status": "Retrospective de-identified dataset used for aggregate-only evidence.",
            "safe_framing": "Can be described as retrospective aggregate evidence supporting operational-burden and prioritization analysis.",
            "current_conservative_headlines_to_verify_before_publication": [
                "Median retrospective lead-time context around 4.0 hours among detected clusters across locked threshold views.",
                "Conservative t=6.0 operating point has been used as a lower-burden posture.",
                "Prior internal summaries referenced approximately 0.65 ERA alerts per patient-day versus approximately 11.27 standard threshold-rule alerts per patient-day, roughly 94.3% operational alert reduction.",
                "Prior internal summaries referenced approximately 4.2% false-positive behavior and approximately 15.3% patient-level detection under strict MIMIC t=6.0 framing."
            ],
            "guardrail": "Verify exact figures against local aggregate validation registry before external publication."
        },
        "eICU": {
            "status": "Second retrospective de-identified dataset used for aggregate-only robustness framing.",
            "safe_framing": "Can be described as supporting cross-dataset robustness of the operational-burden/prioritization story when presented as aggregate-only retrospective analysis.",
            "guardrail": "Use harmonized/outcome-proxy framing carefully. Do not overstate as prospective clinical validation."
        },
        "HiRID": {
            "status": "Access approved; local/private retrospective aggregate evaluation pending or in progress.",
            "only_safe_public_wording": SAFE_HIRID_WORDING,
            "guardrail": "No public HiRID performance, FPR, detection rate, lead-time, final validation, diagnosis, treatment, or escalation claim."
        }
    },
    "approved_public_claims": APPROVED_PUBLIC_CLAIMS,
    "banned_or_restricted_claims": BANNED_OR_RESTRICTED_CLAIMS,
    "rn_pilot_review_boundary": {
        "purpose": "Private clinical-advisor and pilot-readiness review only.",
        "allowed": [
            "Clinical framing review",
            "Decision-support-only clarity review",
            "Workflow clarity review",
            "Limitations and guardrails review",
            "RN/pilot operations feedback"
        ],
        "not_allowed": [
            "Public clinical validation claim",
            "Final HiRID claim",
            "Clinical performance guarantee",
            "Diagnosis/treatment/escalation instruction"
        ]
    }
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def should_manifest(path: Path) -> bool:
    p = str(path).lower()

    # Exclude raw/restricted/local-private material from this public-safe manifest.
    blocked_parts = [
        "local_private",
        "/raw/",
        "\\raw\\",
        "patient_level",
        "row_level",
        "timestamp",
        "timestamps",
        "identifier",
        "restricted",
        "mimic_raw",
        "eicu_raw",
        "hirid_raw",
    ]
    if any(x in p for x in blocked_parts):
        return False

    allowed_suffixes = {".md", ".json", ".txt", ".pdf", ".docx", ".pptx"}
    if path.suffix.lower() not in allowed_suffixes:
        return False

    safe_roots = [
        ROOT / "docs" / "validation",
        ROOT / "docs" / "evidence_lock",
        ROOT / "docs" / "governance",
        ROOT / "data" / "validation",
        ROOT / "data" / "evidence_lock",
    ]

    try:
        resolved = path.resolve()
        return any(resolved.is_relative_to(root.resolve()) for root in safe_roots if root.exists())
    except AttributeError:
        # Python < 3.9 fallback
        resolved_str = str(path.resolve())
        return any(resolved_str.startswith(str(root.resolve())) for root in safe_roots if root.exists())

def build_manifest() -> list[dict]:
    candidates = []
    for base in [
        ROOT / "docs" / "validation",
        ROOT / "docs" / "evidence_lock",
        ROOT / "docs" / "governance",
        ROOT / "data" / "validation",
        ROOT / "data" / "evidence_lock",
    ]:
        if base.exists():
            candidates.extend([p for p in base.rglob("*") if p.is_file()])

    items = []
    for p in sorted(set(candidates)):
        if should_manifest(p):
            try:
                items.append({
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })
            except Exception as exc:
                items.append({
                    "path": str(p),
                    "error": str(exc),
                })
    return items

manifest = build_manifest()

json_out = DATA_DIR / f"evidence_lock_{DATE}.json"
md_out = DOC_DIR / f"EVIDENCE_LOCK_{DATE}.md"
claims_out = DOC_DIR / f"CLAIMS_CONTROL_LOCK_{DATE}.md"
rn_out = DOC_DIR / f"RN_PILOT_PACKET_MANIFEST_{DATE}.md"
gov_out = GOV_DIR / f"GOVERNANCE_PROOF_INDEX_{DATE}.md"

json_out.write_text(json.dumps({
    "evidence_baseline": EVIDENCE_BASELINE,
    "public_safe_manifest": manifest,
    "manifest_note": "Manifest intentionally excludes local_private, raw, row-level, patient-level, timestamp, identifier, and restricted dataset material."
}, indent=2), encoding="utf-8")

md_out.write_text(f"""# {LOCK_TITLE}

**Lock ID:** `{LOCK_ID}`  
**Date:** {DATE}  
**Generated UTC:** {generated_at}

## 1. Locked Intended Use

{INTENDED_USE}

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

> {SAFE_HIRID_WORDING}

Guardrail:

Do not publish HiRID performance, FPR, detection rate, lead time, final validation, diagnosis, treatment, or escalation claims until local evaluation is complete and reviewed.

## 4. Approved Public Claims

""" + "\n".join([f"- {x}" for x in APPROVED_PUBLIC_CLAIMS]) + f"""

## 5. Banned or Restricted Claims

""" + "\n".join([f"- {x}" for x in BANNED_OR_RESTRICTED_CLAIMS]) + f"""

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

Total public-safe manifest items found: **{len(manifest)}**

See JSON file:

`{json_out}`

## 8. Lock Rule

Before updating the website, investor materials, pilot materials, or regulatory materials, wording should be checked against this evidence lock.
""", encoding="utf-8")

claims_out.write_text(f"""# Claims Control Lock — {DATE}

## Locked Intended Use

{INTENDED_USE}

## Approved Website / Public Wording

Use this language:

> HCP-facing decision-support and workflow-support platform.

> Designed to support authorized health care professionals with patient prioritization and command-center operational awareness.

> Provides explainable context such as priority tier, queue rank, primary driver, trend direction, and retrospective lead-time context.

> Supports controlled pilot evaluation and operational review workflows.

> Retrospective aggregate evidence only; not prospective clinical validation.

> Workflow actions are operational state tracking only and do not independently trigger clinical escalation.

For HiRID, use only:

> {SAFE_HIRID_WORDING}

## Do Not Use

""" + "\n".join([f"- {x}" for x in BANNED_OR_RESTRICTED_CLAIMS]) + """

## Replacement Rules

Replace **clinically validated** with:

> supported by retrospective aggregate evidence

Replace **predicts deterioration** with:

> helps authorized health care professionals identify patients who may warrant further clinical evaluation

Replace **real-time alerting** with:

> command-center operational awareness and prioritization support

Replace **automatic escalation** with:

> workflow state tracking for authorized clinical review

Replace **HiRID validated** with:

> HiRID access approved; HiRID retrospective aggregate validation pending local evaluation.
""", encoding="utf-8")

rn_out.write_text(f"""# RN / Pilot Packet Manifest — {DATE}

This is the private RN/pilot review packet structure.

## Purpose

Private clinical-advisor and pilot-readiness review only.

## Include

1. Private RN/pilot review deck  
2. Evidence Lock: `{md_out}`  
3. Claims Control Lock: `{claims_out}`  
4. Current MIMIC-IV/eICU aggregate validation summaries  
5. Command Center screenshots  
6. Homepage screenshots after wording cleanup  
7. Pilot workflow notes  
8. Governance Proof Index: `{gov_out}`  
9. HiRID methodology note, local/private only

## Do Not Include

- Raw rows
- Patient-level outputs
- Timestamps
- Identifiers
- Restricted dataset files
- Public HiRID performance claims
- Final clinical validation claims
- Diagnosis, treatment, or independent escalation language

## Safe Cover Note

This packet is for private RN/pilot review only. The evidence is retrospective and aggregate-only. It should not be interpreted as public clinical validation, diagnosis, treatment direction, final HiRID performance, or independent escalation support.

## Questions for RN / Clinical Advisor

1. Is the clinical framing appropriate for an RN / hospital pilot reviewer?
2. Is the decision-support-only language clear?
3. Are the limitations and guardrails understandable?
4. Does the Command Center workflow make sense from an RN / clinical operations perspective?
5. Is there anything that should be changed before private pilot discussions?
""", encoding="utf-8")

gov_out.write_text(f"""# Governance Proof Index — {DATE}

This index tracks operational proof that supports pilot readiness, insurance readiness, and hospital-review readiness.

## Current Priority

Evidence maturity is now less about adding features and more about proving that controls are real.

## Proof Items

| Control Area | Current Status | Proof Needed |
|---|---:|---|
| MFA | Substantially complete based on prior setup work | Screenshots / inventory of GitHub, Render, email, Namecheap, admin accounts |
| Backup Codes | Complete based on prior setup work | Secure record location confirmation |
| Access Review | Needs dated log | List users/accounts, roles, and access justification |
| Patch Process | Needs dated log | Recent commits, dependency updates, Render/GitHub activity |
| Backup / Restore | Needs test evidence | Screenshot or note confirming restore test |
| Incident Response | Needs tabletop record | One-page incident-response tabletop exercise |
| Business Continuity | Needs summary | What happens if founder is unavailable, platform down, or account locked |
| Data Governance | Partially complete | DUA-safe data handling, local-only restricted datasets, aggregate-only publication rule |
| Claims Control | Complete with this lock | Evidence Lock and Claims Control Lock |
| RN / Clinical Review | Pending advisor response | Andrene Louison/RN written feedback record |

## Next Governance Files To Create

- Access Review Log
- Patch Log
- Backup Restore Test Log
- Incident Response Tabletop Log
- Business Continuity Note
- Security Control Inventory

## Pilot-Readiness Rule

For hospital/pilot review, every important claim should have either:

1. A document,
2. A screenshot,
3. A dated log,
4. A validation artifact, or
5. A written advisor/reviewer note.
""", encoding="utf-8")

print("")
print("EVIDENCE LOCK CREATED")
print("=====================")
print(f"Lock ID: {LOCK_ID}")
print(f"Evidence lock markdown: {md_out}")
print(f"Evidence lock JSON:     {json_out}")
print(f"Claims control lock:    {claims_out}")
print(f"RN packet manifest:     {rn_out}")
print(f"Governance proof index: {gov_out}")
print(f"Public-safe manifest items: {len(manifest)}")
print("")
print("IMPORTANT:")
print("- No raw rows were created.")
print("- No patient-level outputs were created.")
print("- No timestamps/identifiers were created.")
print("- local_private/raw/restricted materials were excluded from the manifest.")
print("- Do not git push until you review the generated files.")

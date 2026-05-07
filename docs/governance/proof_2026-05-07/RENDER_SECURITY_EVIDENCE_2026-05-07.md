# Render Security Evidence — 2026-05-07

## Purpose

Document Render account, workspace, deployment, and environment-variable controls for Early Risk Alert AI platform governance, pilot readiness, and insurance-readiness evidence.

## Account / System

| System | Purpose | Owner / Role | Status |
|---|---|---|---|
| Render Admin | Hosting, deployment, service runtime, database/worker/Redis services | Milton Munroe / Founder & CEO | Verified privately |

## Controls Verified Privately

- [x] Render workspace reviewed
- [x] Workspace member list reviewed
- [x] Only one workspace member observed
- [x] Owner/admin access verified privately
- [x] Production service list reviewed
- [x] Main production web service deployed
- [x] Connected GitHub repository / branch reviewed
- [x] Recent deploy history reviewed
- [x] Environment variables reviewed with values masked/hidden
- [x] No environment values, secrets, tokens, API keys, or screenshots committed to GitHub
- [x] Logs reviewed only for deployment/runtime status; logs retained privately only if needed

## Evidence Observed Privately

- Workspace team member count: one visible member
- Visible role: Admin
- Active services visible in Render dashboard
- Main production service visible as deployed
- Environment variable values masked
- Recent deploys visible and live
- Runtime/log output reviewed for service availability only

## Evidence Rules

Do not commit or upload:

- Render screenshots
- Environment variable values
- API keys
- Database URLs
- Secret keys
- SMTP credentials
- SendGrid/Twilio credentials
- Session secrets
- Log screenshots containing IP addresses, request IDs, tokens, payloads, or private data
- Billing or payment information

## Public-Safe Evidence Statement

Render Admin security and deployment controls were reviewed privately for the current governance checkpoint. Private evidence confirms workspace membership review, admin access, production service deployment status, GitHub-linked deployment flow, and masked environment-variable handling. Screenshot proof is retained privately and is not committed to the repository.

## Status

Verified privately on 2026-05-07.

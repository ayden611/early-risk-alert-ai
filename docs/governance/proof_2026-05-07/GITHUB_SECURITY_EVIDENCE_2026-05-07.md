# GitHub Security Evidence — 2026-05-07

## Purpose

Document GitHub account-security controls for Early Risk Alert AI code, evidence-lock, governance, and deployment-source protection.

## Account / System

| System | Purpose | Owner / Role | Status |
|---|---|---|---|
| GitHub Admin | Repository, evidence-lock, governance docs, deployment-source control | Milton Munroe / Founder & CEO | Verified privately |

## Controls Verified Privately

- [x] GitHub two-factor authentication reviewed
- [x] Preferred 2FA method set to security keys
- [x] Authenticator app configured
- [x] Security keys configured
- [x] Two security keys present
- [x] Recovery codes viewed
- [x] Screenshot proof saved privately
- [x] No recovery codes, tokens, SSH private keys, QR codes, or secrets committed to GitHub

## Notes

SMS/Text was not added and is labeled less secure by GitHub. This is acceptable because stronger methods are already configured: security keys and authenticator app.

## Evidence Rules

Do not commit or upload:

- Recovery codes
- Authenticator QR codes
- TOTP secret keys
- Personal access tokens
- SSH private keys
- Secret keys
- Full password-manager screenshots showing sensitive details
- Screenshots showing account recovery codes or secrets

## Public-Safe Evidence Statement

GitHub Admin account-security review is complete for the current governance checkpoint. Private evidence confirms two-factor authentication, security keys, authenticator app configuration, and recovery-code review. Screenshot proof is retained privately and is not committed to the repository.

## Status

Verified privately on 2026-05-07.

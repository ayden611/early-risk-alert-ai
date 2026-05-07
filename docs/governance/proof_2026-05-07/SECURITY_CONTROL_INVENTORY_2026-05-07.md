# Security Control Inventory — 2026-05-07

## Purpose

Track security controls that support pilot readiness and insurance readiness.

| Control | Status | Evidence Needed | Owner |
|---|---|---|---|
| MFA on core accounts | To verify | Screenshots / account-security pages | Milton Munroe |
| Backup codes stored securely | To verify | Secure location confirmation, not public | Milton Munroe |
| Access review | Drafted | Access Review Log completed | Milton Munroe |
| Patch/change log | Drafted | Commit history and Wix publish record | Milton Munroe |
| Backup/restore test | Pending | Restore test screenshot/note | Milton Munroe |
| Incident response tabletop | Drafted | Completed tabletop note | Milton Munroe |
| Business continuity note | Drafted | Recovery contact plan | Milton Munroe |
| DUA-safe data handling | Partially complete | Local-only path + gitignore protections | Milton Munroe |
| Claims control | Complete baseline | Evidence Lock and Claims Control Lock | Milton Munroe |
| Clinical-advisor review | Pending reply | Written notes from Andrene Louison, RN | Milton Munroe |

## High-Priority Next Evidence

- MFA screenshots
- GitHub commit screenshot
- Wix publish screenshot
- Restore test note
- Andrene Louison RN feedback email


## Wix Admin Security Evidence

| Control | Status | Evidence Needed | Owner |
|---|---|---|---|
| Wix Admin strong unique password | Verified privately | Password-manager proof retained privately | Milton Munroe |
| Wix Admin passkey | Verified privately | Wix Account Settings screenshot retained privately | Milton Munroe |
| Wix Admin SMS 2-step verification | Verified privately | Wix Account Settings screenshot retained privately | Milton Munroe |
| Wix Admin authenticator app verification | Verified privately | Wix Account Settings screenshot retained privately | Milton Munroe |
| Wix collaborator access policy | Draft | No shared passwords; collaborators only if needed | Milton Munroe |


## GitHub Admin Security Evidence

| Control | Status | Evidence Needed | Owner |
|---|---|---|---|
| GitHub 2FA | Verified privately | GitHub Password and authentication screenshot retained privately | Milton Munroe |
| GitHub preferred 2FA method | Verified privately | Security keys preferred | Milton Munroe |
| GitHub authenticator app | Verified privately | Configured; screenshot retained privately | Milton Munroe |
| GitHub security keys | Verified privately | Two keys configured; screenshot retained privately | Milton Munroe |
| GitHub recovery codes | Reviewed privately | Recovery codes viewed/saved privately; do not commit | Milton Munroe |
| GitHub SMS/Text | Not enabled | Acceptable because security keys and authenticator app are configured | Milton Munroe |


## Render Admin Security Evidence

| Control | Status | Evidence Needed | Owner |
|---|---|---|---|
| Render workspace access review | Verified privately | Workspace/team screenshot retained privately | Milton Munroe |
| Render member list | Verified privately | One visible member; admin role retained privately | Milton Munroe |
| Render production services | Verified privately | Service dashboard screenshot retained privately | Milton Munroe |
| Render GitHub-linked deployment | Verified privately | Production service connected to GitHub repo/branch; screenshot retained privately | Milton Munroe |
| Render deploy history | Verified privately | Recent successful deploys retained privately | Milton Munroe |
| Render environment values masked | Verified privately | Environment page screenshot retained privately with values hidden | Milton Munroe |
| Render logs | Reviewed cautiously | Do not commit logs; retain only redacted/private runtime proof if needed | Milton Munroe |

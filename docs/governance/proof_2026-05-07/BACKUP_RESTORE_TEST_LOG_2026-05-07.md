# Backup / Restore Test Log — 2026-05-07

## Purpose

Show that Early Risk Alert AI can recover key materials if files, code, or accounts are disrupted.

## Backup Areas

| Area | Backup Method | Last Backup Date | Restore Tested | Evidence Needed |
|---|---|---:|---:|---|
| GitHub repository | Git remote + local copy | To verify | To test | Screenshot of repo + local clone |
| Wix website | Wix site history / duplicate site | To verify | To test | Screenshot of site history |
| Evidence lock docs | GitHub + local copy | 2026-05-07 | To test | Commit `bb0b0bf` |
| Private RN packet | Local-only folder | 2026-05-07 | To test | Folder screenshot |
| HiRID private outputs | Local-only restricted workspace | To verify | To test | Local-only manifest, no public upload |

## Simple Restore Test To Perform

- [ ] Create a temporary copy of one governance file
- [ ] Delete the temporary copy
- [ ] Restore it from backup/local repo
- [ ] Record date/time and screenshot

## Result

- Restore test status: Pending
- Notes:

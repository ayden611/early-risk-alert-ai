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

- Restore test status: Passed for lightweight git restore test
- Notes: Lightweight git restore test completed. Additional offsite/Wix/private-workspace backup evidence still needed.

## Restore Test Record — 2026-05-07T13:33:44.642017+00:00

**Test type:** Lightweight git restore test  
**File tested:** `docs/governance/proof_2026-05-07/ACCESS_REVIEW_LOG_2026-05-07.md`  
**Restore source:** `HEAD` commit in local git repository  
**Result:** Passed  
**Private/raw data involved:** No  
**Patient-level data involved:** No  
**Restricted dataset material involved:** No  

### Procedure

1. Confirmed the access review log was tracked in git.
2. Captured the original SHA-256 checksum.
3. Temporarily deleted the local file.
4. Restored the file from `HEAD`.
5. Confirmed the restored SHA-256 checksum matched the original file.

### Interpretation

This confirms that public-safe governance documentation can be restored from the committed repository state. This is a lightweight repository restore test only. Separate evidence is still needed for offsite backups, Wix site history, restricted local workspace backups, and account recovery.


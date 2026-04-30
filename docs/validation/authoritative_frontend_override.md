# Authoritative Frontend Override

Generated: 2026-04-30T12:38:38.351098+00:00

## Purpose

The backend evidence was correct, but the live app was still serving older frontend route content on some URLs.

This patch forces the public routes to serve corrected frontend files before older route handlers can respond.

## Forced Routes

- `/validation-intelligence`
- `/validation-evidence`
- `/validation-runs`
- `/command-center`
- `/model-card`
- `/pilot-success-guide`

## What This Fixes

- Removes stale `Loading...` placeholders.
- Makes Model Card and Pilot Success Guide show as ready.
- Makes Command Center visibly show priority tier, queue rank, primary driver, trend, and lead-time context.
- Keeps MIMIC-IV and eICU event definitions separate.

## Not Added

- No hard ROI claims.
- No proven generalizability claim.
- No direct equality claim between MIMIC-IV and eICU detection rates.
- No diagnosis, treatment, prevention, or autonomous escalation claims.

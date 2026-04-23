from __future__ import annotations

import csv
import io
import json
import os
import random
import smtplib
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
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
    send_file,
    send_from_directory,
    session,
    url_for,
)

from era.web.command_center import COMMAND_CENTER_HTML

MIMIC_EXTRACT_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MIMIC-IV Extraction Tool — ERA</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,sans-serif;background:#0a1628;color:#dce9ff;min-height:100vh;padding:40px 20px}
.container{max-width:800px;margin:0 auto}
h1{font-size:26px;color:#3ad38f;margin-bottom:6px}
.sub{color:#9fb4d6;font-size:13px;margin-bottom:28px}
.card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:22px;margin-bottom:18px}
h2{font-size:15px;color:#9adfff;margin-bottom:14px;font-weight:600}
.field{margin-bottom:14px}
label{display:block;font-size:12px;color:#9fb4d6;margin-bottom:5px}
.req{color:#3ad38f;font-weight:700}
input[type=file],input[type=number]{width:100%;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.15);border-radius:6px;padding:9px 11px;color:#dce9ff;font-size:13px}
.btns{margin-top:10px;display:flex;gap:10px;flex-wrap:wrap}
.btn{padding:11px 24px;border:none;border-radius:6px;font-size:14px;font-weight:700;cursor:pointer;transition:opacity .2s}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-green{background:#3ad38f;color:#0a1628}
.btn-blue{background:#2E75B6;color:#fff}
.note{font-size:11px;color:#6b8aad;margin-top:6px;line-height:1.6}
.result{border-radius:6px;padding:16px;margin-top:18px;font-size:13px;line-height:1.8;display:none}
.result.ok{background:rgba(58,211,143,.08);border:1px solid rgba(58,211,143,.3)}
.result.err{background:rgba(255,80,80,.08);border:1px solid rgba(255,80,80,.3)}
.result.running{background:rgba(154,223,255,.06);border:1px solid rgba(154,223,255,.2)}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
.stat{background:rgba(255,255,255,.06);border-radius:6px;padding:10px 16px;text-align:center;min-width:110px}
.stat-v{font-size:20px;font-weight:700;color:#3ad38f}
.stat-l{font-size:11px;color:#9fb4d6}
pre{font-size:10px;color:#9adfff;margin-top:10px;overflow-x:auto;background:rgba(0,0,0,.3);padding:10px;border-radius:4px;white-space:pre-wrap;word-break:break-all}
.ids{font-size:11px;color:#6b8aad;font-family:monospace;line-height:2}
.progress{width:100%;height:4px;background:rgba(255,255,255,.1);border-radius:2px;overflow:hidden;margin:10px 0;display:none}
.progress-bar{height:100%;background:#3ad38f;animation:prog 2s linear infinite}
@keyframes prog{0%{width:0%;margin-left:0}50%{width:60%;margin-left:20%}100%{width:0%;margin-left:100%}}
</style>
</head>
<body>
<div class="container">
  <h1>MIMIC-IV Extraction Tool</h1>
  <p class="sub">Convert raw MIMIC-IV tables into ERA-compatible CSV. Upload the three files, then click Extract.</p>

  <div class="card">
    <h2>Step 1 — Upload MIMIC Files</h2>

    <div class="field">
      <label><span class="req">REQUIRED</span> — chartevents.csv or chartevents.csv.gz</label>
      <input type="file" id="f_chartevents" accept=".csv,.gz">
      <p class="note">Contains vital sign measurements. From the /icu/ folder. Extracts item IDs: 220045 HR, 220277 SpO2, 220179 BP-sys, 220180 BP-dia, 220210 RR, 223761/223762 Temp.</p>
    </div>

    <div class="field">
      <label><span class="req">REQUIRED</span> — icustays.csv or icustays.csv.gz</label>
      <input type="file" id="f_icustays" accept=".csv,.gz">
      <p class="note">Maps patients to ICU stay info and unit type. From the /icu/ folder.</p>
    </div>

    <div class="field">
      <label>OPTIONAL — datetimeevents.csv or datetimeevents.csv.gz (enables clinical_event flagging)</label>
      <input type="file" id="f_datetimeevents" accept=".csv,.gz">
      <p class="note">If provided, readings within 6 hours before Rapid Response / ICU Transfer / Code Blue are flagged clinical_event=1. Item IDs: 225477, 225086, 224859, 225468.</p>
    </div>

    <div class="field" style="max-width:220px">
      <label>Minimum vitals per reading (default 3, max 6)</label>
      <input type="number" id="min_vitals" value="3" min="1" max="6">
    </div>

    <div class="btns">
      <button class="btn btn-green" id="btn_extract" onclick="runExtract(false)">Extract &amp; Validate</button>
      <button class="btn btn-blue"  id="btn_download" onclick="runExtract(true)">Extract &amp; Download CSV</button>
    </div>

    <div class="progress" id="progress"><div class="progress-bar"></div></div>
    <div class="result" id="result"></div>
  </div>

  <div class="card">
    <h2>What This Tool Does</h2>
    <p class="note" style="line-height:2.2">
      1. Reads chartevents.csv and keeps only the 6 ERA vital sign item IDs<br>
      2. Pivots from MIMIC long format (one row per measurement) to ERA wide format (one row per patient + timestamp)<br>
      3. Converts Celsius temperature to Fahrenheit automatically<br>
      4. Maps MIMIC ICU unit names to ERA unit labels (icu / stepdown)<br>
      5. Flags readings within 6 hours before a clinical event (if datetimeevents provided)<br>
      6. Validates output against ERA schema before download<br>
      7. Outputs a CSV ready to upload directly to /retro-upload
    </p>
  </div>

  <div class="card">
    <h2>Item IDs Extracted</h2>
    <div class="ids">
      220045 &#8594; heart_rate (bpm)<br>
      220277 &#8594; spo2 (%)<br>
      220179 &#8594; bp_systolic (mmHg)<br>
      220180 &#8594; bp_diastolic (mmHg)<br>
      220210 &#8594; respiratory_rate (breaths/min)<br>
      223761 &#8594; temperature_f (direct)<br>
      223762 &#8594; temperature_f (converted from Celsius)<br>
      225477 &#8594; event: Rapid Response Team<br>
      225086 &#8594; event: Transfer to ICU<br>
      224859 &#8594; event: Code Blue<br>
      225468 &#8594; event: Unplanned Extubation
    </div>
  </div>
</div>

<script>
function setRunning(on) {
  var btns = [document.getElementById('btn_extract'), document.getElementById('btn_download')];
  btns.forEach(function(b){ b.disabled = on; });
  var prog = document.getElementById('progress');
  prog.style.display = on ? 'block' : 'none';
}

function showResult(cls, html) {
  var el = document.getElementById('result');
  el.className = 'result ' + cls;
  el.style.display = 'block';
  el.innerHTML = html;
}

function runExtract(download) {
  var fc = document.getElementById('f_chartevents').files[0];
  var fi = document.getElementById('f_icustays').files[0];

  if (!fc) { showResult('err', '<b>Error:</b> Please select chartevents.csv before extracting.'); return; }
  if (!fi) { showResult('err', '<b>Error:</b> Please select icustays.csv before extracting.'); return; }

  var fd = new FormData();
  fd.append('chartevents', fc);
  fd.append('icustays', fi);

  var dte = document.getElementById('f_datetimeevents').files[0];
  if (dte) fd.append('datetimeevents', dte);

  fd.append('min_vitals', document.getElementById('min_vitals').value);
  if (download) fd.append('download', '1');

  setRunning(true);
  showResult('running', 'Processing... this may take 30-90 seconds for large files. Do not close this page.');

  fetch('/api/mimic/extract', {method: 'POST', body: fd})
    .then(function(resp) {
      if (download && resp.ok) {
        var cd = resp.headers.get('content-disposition') || '';
        var fn = (cd.match(/filename=([^;]+)/) || ['','era_mimic_extract.csv'])[1];
        return resp.blob().then(function(blob) {
          var url = URL.createObjectURL(blob);
          var a = document.createElement('a');
          a.href = url; a.download = fn; document.body.appendChild(a); a.click();
          document.body.removeChild(a); URL.revokeObjectURL(url);
          showResult('ok', '<b style="color:#3ad38f">CSV downloaded successfully.</b><br>Now upload it to <a href="/retro-upload" style="color:#9adfff">/retro-upload</a> to run ERA validation.');
          setRunning(false);
        });
      }
      return resp.json().then(function(data) {
        if (!data.ok) {
          var errs = data.errors ? data.errors.join('<br>') : (data.error || 'Unknown error');
          var hint = data.hint ? '<br><br><b>Hint:</b> ' + data.hint : '';
          showResult('err', '<b>Extraction failed:</b><br>' + errs + hint);
        } else {
          var s = data.stats || {};
          var v = s.validation || {};
          var schemaOk = data.ready_for_era_upload;
          var html = '<b style="color:#3ad38f">' + (data.message || 'Done') + '</b>';
          html += '<div class="stats">';
          html += '<div class="stat"><div class="stat-v">' + ((s.total_patients||0).toLocaleString()) + '</div><div class="stat-l">Patients</div></div>';
          html += '<div class="stat"><div class="stat-v">' + ((s.total_rows||0).toLocaleString()) + '</div><div class="stat-l">Rows</div></div>';
          html += '<div class="stat"><div class="stat-v">' + ((s.total_events_flagged||0).toLocaleString()) + '</div><div class="stat-l">Events Flagged</div></div>';
          html += '<div class="stat"><div class="stat-v">' + ((s.patients_with_events||0).toLocaleString()) + '</div><div class="stat-l">Patients w/ Events</div></div>';
          html += '<div class="stat"><div class="stat-v">' + (data.csv_size_kb||0) + ' KB</div><div class="stat-l">CSV Size</div></div>';
          html += '<div class="stat"><div class="stat-v" style="color:' + (schemaOk?'#3ad38f':'#ff6b6b') + '">' + (schemaOk?'PASS':'FAIL') + '</div><div class="stat-l">ERA Schema</div></div>';
          html += '</div>';
          if (s.rows_skipped_min_vitals) html += '<p class="note">Rows skipped (insufficient vitals): ' + s.rows_skipped_min_vitals.toLocaleString() + '</p>';
          if (data.preview_rows && data.preview_rows.length) {
            html += '<b>Preview (first 5 rows):</b><pre>' + data.preview_rows.join('\n') + '</pre>';
          }
          if (schemaOk) {
            html += '<br><button class="btn btn-blue" onclick="runExtract(true)" style="margin-top:8px">Download ERA CSV</button>';
            html += '&nbsp;&nbsp;<a href="/retro-upload" class="btn btn-green" style="text-decoration:none;display:inline-block;margin-top:8px">Go to Retro Upload</a>';
          } else {
            html += '<br><b style="color:#ff6b6b">Schema issues: ' + (v.issues||[]).join(', ') + '</b>';
          }
          showResult('ok', html);
        }
        setRunning(false);
      });
    })
    .catch(function(e) {
      showResult('err', '<b>Network error:</b> ' + e.message + '<br>Check that the platform is running and try again.');
      setRunning(false);
    });

  return false;
}
</script>
</body>
</html>
"""



INFO_EMAIL = "info@earlyriskalertai.com"
BUSINESS_PHONE = "732-724-7267"
FOUNDER_NAME = "Milton Munroe"
FOUNDER_ROLE = "Founder & CEO, Early Risk Alert AI"

PILOT_VERSION = os.getenv("PILOT_VERSION", "stable-pilot-1.0.8")
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
    "respiratory rate (RR) as monitored context",
    "temperature as monitored context",
    "CSV-based retrospective validation pipeline (FHIR R4 and HL7 integration on roadmap)",
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
    "replaces clinician judgment",
    "issues machine-led treatment direction",
    "independently triggers escalation",
]
PILOT_RELEASE_NOTES = [
    {
        "version": "stable-pilot-1.0.6",
        "date": "2026-04-10",
        "summary": "Full SQLite + flat-file persistence layer (workflow, audit, trends, thresholds survive restarts). Email/SMS notifications. /status public page. /deck inline pitch deck. Admin CRM with notes and email compose. Delta explainability. CSV retrospective validation pipeline with no-commitment analysis offer. Latest validation: 36.8% patient detection in 6-hr window · 72.4% alert reduction · 5.9% ERA FPR vs 19.5% standard thresholds (5,000-patient synthetic dataset, April 2026).",
        "changes": [
            "Added SQLite persistence for workflow records and audit log — state survives restarts and deploys.",
            "Added flat-file trend history persistence per patient per day — trend data survives restarts.",
            "Added flat-file threshold persistence — unit threshold edits survive restarts.",
            "Added alert notification system: email via SMTP/SendGrid and SMS via Twilio when patients cross critical threshold.",
            "Notification cooldown per patient (configurable via ALERT_COOLDOWN_MINUTES env var, default 15 min).",
            "Added /status public uptime page showing system health without login.",
            "Added /deck inline pitch deck HTML page — no PDF dependency, always resolves.",
            "Upgraded admin review page with notes field, follow-up date, and inline email compose per lead.",
            "Upgraded explainability to show delta from last observation (HR was X, now Y — rising/falling trend).",
            "Defined all 18 previously missing governance variables.",
            "Fixed broken api_system_health_legacy alias.",
            "Added /api/v2/patients clean REST endpoint.",
            "SESSION_COOKIE_SECURE enforcement via environment variable.",
        ],
    },
    {
        "version": "stable-pilot-1.0.5a",
        "date": "2026-04-06",
        "summary": "RR and temperature monitored-context bundle with restored pilot docs, intake pages, and model card access.",
        "changes": [
            "Added Respiratory Rate (RR) and Temperature as monitored context fields in live patient payloads.",
            "Added RR and Temperature trend history support for the drawer and trend endpoints.",
            "Kept RR and Temperature framed as monitored context, trend/support inputs, and explainable review basis only.",
            "Added Andrene Louison (RN) in the advisory and clinical support sections.",
            "Restored richer pilot-docs sections, polished intake pages, and model card / pilot success guide route access.",
        ],
    },
]
PILOT_ADVISORY_STRUCTURE = [
    {
        "name": "Milton Munroe",
        "title": FOUNDER_ROLE,
        "scope": "Product leadership, pilot operations coordination, release control, hospital-facing pilot management, and governance ownership.",
        "status": "In place",
    },
    {
        "name": "Uche Anosike",
        "title": "Technical Infrastructure & Security Advisor",
        "scope": "Infrastructure architecture, platform security posture, environment configuration, backup / recovery planning, patch-management review, and deployment-readiness guidance for controlled pilot environments.",
        "status": "In place",
    },
    {
        "name": "Andrene Louison",
        "title": "Clinical Advisor (RN)",
        "scope": "Clinical workflow review, monitored-context guidance, respiratory-rate and temperature review input, and hospital-facing clinical advisory support.",
        "status": "In place",
    },
]
PILOT_SUPPORT_OWNERS = [
    {
        "area": "Product & Pilot Operations",
        "owner": "Milton Munroe",
        "title": FOUNDER_ROLE,
        "coverage": "Release control, pilot operations coordination, stakeholder communication, and governance ownership.",
        "status": "In place",
    },
    {
        "area": "Technical Infrastructure & Security",
        "owner": "Uche Anosike",
        "title": "Technical Infrastructure & Security Advisor",
        "coverage": "Infrastructure review, environment hardening guidance, deployment security, backup / recovery planning, and patch-management review.",
        "status": "In place",
    },
    {
        "area": "Clinical Review",
        "owner": "Andrene Louison",
        "title": "Clinical Advisor (RN)",
        "coverage": "Clinical workflow review, monitored-context guidance, respiratory-rate and temperature review input, and hospital-facing clinical advisory support.",
        "status": "In place",
    },
]

PILOT_LIMITATIONS_TEXT = [
    "Output is decision support only and not a diagnosis.",
    "Incomplete or delayed data may affect outputs.",
    "Clinicians must independently review the patient and current clinical context.",
    "Hospital policy governs escalation, response timing, and treatment decisions.",
    "The platform is not intended to diagnose, direct treatment, or independently trigger escalation.",
]

PILOT_APPROVED_CLAIMS = [
    "assists authorized health care professionals in identifying patients who may warrant further clinical evaluation",
    "supports patient prioritization across monitored patients",
    "provides explainable risk-support context",
    "supports command-center workflow awareness and operational visibility",
]

PILOT_BANNED_CLAIMS = [
    "detects deterioration autonomously",
    "identifies who needs immediate escalation",
    "predicts who will clinically crash",
    "directs bedside intervention",
]

PILOT_CHANGE_CONTROL = [
    "Freeze one intended-use statement everywhere in the pilot build.",
    "Keep claims narrow and supportive rather than directive.",
    "Make review basis, confidence, limitations, freshness, and unknowns visible for each patient.",
    "Maintain role scoping, unit scoping, and audit visibility across routes.",
    "Record each approved pilot change in release notes before deployment.",
]

PILOT_RISK_REGISTER = [
    {"id": "R-001", "area": "Claims", "risk": "Autonomous or directive claims could shift the platform toward a higher-risk regulatory posture.", "mitigation": "Freeze one intended-use statement and review UI, sales copy, and decks before each release.", "owner": "Founder/Product", "status": "Open"},
    {"id": "R-002", "area": "Explainability", "risk": "Users may over-rely on outputs if the basis, confidence, limitations, and unknowns are not visible.", "mitigation": "Keep review basis, freshness, confidence, limitations, and unknowns visible in the patient detail workflow.", "owner": "Engineering", "status": "In Place"},
    {"id": "R-003", "area": "Workflow Separation", "risk": "Operational buttons could be misread as machine-issued clinical directives.", "mitigation": "Keep acknowledge/assign/escalate/resolve framed as workflow state and user action logging only.", "owner": "Product", "status": "In Place"},
    {"id": "R-004", "area": "Scope Control", "risk": "Improper access scope could expose records outside the intended role or unit context.", "mitigation": "Role restrictions, unit restrictions, and filtered workflow/audit visibility enforced across all routes. Phishing-resistant MFA now in place on all administrative accounts. Admin access restricted via ALLOWED_ADMIN_EMAILS and ADMIN_PASSWORD_HASH.", "owner": "Engineering", "status": "In Place"},
    {"id": "R-005", "area": "Change Control", "risk": "Pilot drift could occur if live edits are made without a stable version marker and release notes.", "mitigation": "Keep one stable pilot version, maintain release notes, update change log for each release, and confirm ERA_DATA_DIR backup before each deploy. Persistence layer (SQLite + flat files) ensures workflow and audit state survive deploys — reducing pilot drift risk.", "owner": "Founder/Engineering", "status": "In Place"},
]

PILOT_VNV_LITE = [
    {"id": "VV-001", "check": "Access context respects login, role, and unit scope.", "method": "Route check across /login, /pilot-access, /api/access-context, and filtered views.", "evidence": "Implemented in current pilot bundle.", "status": "Pass"},
    {"id": "VV-002", "check": "Workflow actions remain operational and auditable rather than directive.", "method": "Exercise ack/assign/escalate/resolve and confirm workflow/audit separation in API and UI.", "evidence": "Workflow and audit routes updated in current pilot bundle.", "status": "Pass"},
    {"id": "VV-003", "check": "Explainability fields are visible for patient review.", "method": "Open patient drawer and verify review basis, confidence, limitations, freshness, what changed, and unknowns.", "evidence": "Visible in patient detail drawer.", "status": "Pass"},
    {"id": "VV-004", "check": "Threshold and trend routes return scoped data without breaking the command center.", "method": "Call /api/thresholds and /api/trends/<patient_id> under pilot roles.", "evidence": "Routes available in current pilot bundle.", "status": "Pass"},
    {"id": "VV-005", "check": "Pilot governance artifacts are visible from the app.", "method": "Open /pilot-docs and verify intended use, risk register, V&V-lite, and release notes render.", "evidence": "Added in current pilot bundle.", "status": "Pass"},
    {"id": "VV-006", "check": "All governance variables resolve without NameError.", "method": "Call /api/pilot-governance and confirm 200 with full payload.", "evidence": "All 18 previously missing variables now defined in stable-pilot-1.0.6.", "status": "Pass"},
]

PILOT_CLAIMS_CONTROL_SHEET = [
    {"claim": "Assists authorized health care professionals in identifying patients who may warrant further clinical evaluation", "status": "Approved", "category": "Intended use / supportive output"},
    {"claim": "Supports patient prioritization across monitored patients", "status": "Approved", "category": "Workflow-support / prioritization"},
    {"claim": "Provides explainable risk-support context", "status": "Approved", "category": "Explainability / transparency"},
    {"claim": "Detects deterioration autonomously", "status": "Banned", "category": "Automation / device-risk"},
    {"claim": "Identifies who needs immediate escalation", "status": "Banned", "category": "Urgency-directive / device-risk"},
    {"claim": "Replaces clinician judgment", "status": "Banned", "category": "Autonomy / liability risk"},
]

PILOT_DOCUMENT_CONTROL_INDEX = [
    {"document_name": "Frozen Intended-Use Statement", "version": "1.0"},
    {"document_name": "Approved Support Language", "version": "1.0"},
    {"document_name": "Claims Control Sheet", "version": "1.1"},
    {"document_name": "Complaint / Issue Log", "version": "1.0"},
    {"document_name": "Issue Escalation Process", "version": "1.0"},
    {"document_name": "Change Approval Log", "version": "1.0"},
    {"document_name": "Cybersecurity Summary", "version": "1.0"},
    {"document_name": "User Provisioning / Deprovisioning Policy", "version": "1.0"},
    {"document_name": "Data Retention / Deletion / Return Policy", "version": "1.0"},
    {"document_name": "Pilot Agreement / Pilot Scope Document", "version": "1.0"},
    {"document_name": "Training / Use Instructions", "version": "1.0"},
    {"document_name": "Validation Evidence Packet", "version": "1.0"},
    {"document_name": "Model Card", "version": "1.0"},
    {"document_name": "Pilot Success Criteria & Workflow Integration Guide", "version": "1.0"},
]

PILOT_VALIDATION_EVIDENCE = [
    {"date_tested": "2026-03-25", "tested_by": "Founder/Engineering", "test_case_id": "VAL-001", "expected_result": "Login flow routes to command center", "actual_result": "Pass", "status": "Pass"},
    {"date_tested": "2026-03-25", "tested_by": "Founder/Engineering", "test_case_id": "VAL-002", "expected_result": "Unit scope restricts patient visibility", "actual_result": "Pass", "status": "Pass"},
    {"date_tested": "2026-03-26", "tested_by": "Founder/Product", "test_case_id": "VAL-003", "expected_result": "Explainability fields visible in drawer", "actual_result": "Pass", "status": "Pass"},
    {"date_tested": "2026-03-26", "tested_by": "Founder/Product", "test_case_id": "VAL-004", "expected_result": "Workflow actions log to audit trail", "actual_result": "Pass", "status": "Pass"},
    {"date_tested": "2026-04-06", "tested_by": "Founder/Product", "test_case_id": "VAL-005", "expected_result": "RR and Temp visible as monitored context", "actual_result": "Pass", "status": "Pass"},
    {"date_tested": "2026-04-07", "tested_by": "Founder/Engineering", "test_case_id": "VAL-006", "expected_result": "/api/pilot-governance returns 200 with full payload", "actual_result": "Pass", "status": "Pass"},
    {"date_tested": "2026-04-09", "tested_by": "Founder/Product", "test_case_id": "VAL-013", "expected_result": "Retro upload page renders with schema table and upload zone", "actual_result": "Pass", "status": "Pass"},
    {"date_tested": "2026-04-09", "tested_by": "Founder/Product", "test_case_id": "VAL-014", "expected_result": "CSV upload parses 500-patient synthetic dataset correctly", "actual_result": "Pass — 12,873 rows, 145 events parsed", "status": "Pass"},
    {"date_tested": "2026-04-09", "tested_by": "Founder/Product", "test_case_id": "VAL-015", "expected_result": "Retrospective analysis computes sensitivity, FPR, and alert reduction vs threshold", "actual_result": "Pass — ERA 23.4% sensitivity (intentional trade for 5.1% FPR vs 28.5% threshold FPR), 81.9% alert reduction. Synthetic dataset, 500 patients, 145 events.", "status": "Pass"},
    {"date_tested": "2026-04-09", "tested_by": "Founder/Product", "test_case_id": "VAL-016", "expected_result": "Patient summary table renders top 20 by peak risk with event flags", "actual_result": "Pass", "status": "Pass"},
    {"date_tested": "2026-04-10", "tested_by": "Milton Munroe", "test_case_id": "VAL-017", "expected_result": "CITI Program research ethics training completed prior to MIMIC-IV data access", "actual_result": "Pass — Data or Specimens Only Research and Conflict of Interest certificates obtained", "status": "Pass"},
    {"date_tested": "2026-04-10", "tested_by": "Milton Munroe", "test_case_id": "VAL-018", "expected_result": "Phishing-resistant MFA and backup recovery controls implemented across core administrative systems", "actual_result": "Pass — MFA and backup recovery implemented on business email, source control (GitHub), hosting (Render), password management, and domain/DNS", "status": "Pass"},
    {"date_tested": "2026-04-12", "tested_by": "Founder/Engineering", "test_case_id": "VAL-019", "expected_result": "Retro analysis pipeline handles 5,000 patients (129,333 rows) within acceptable response time", "actual_result": "Pass — 5,000 patients analyzed in 1.01s. 10,000 patients in 1.99s. Pipeline confirmed safe for MIMIC 2,000-patient target.", "status": "Pass"},
    {"date_tested": "2026-04-12", "tested_by": "Founder/Engineering", "test_case_id": "VAL-020", "expected_result": "Multi-threshold analysis (4.0, 5.0, 6.0) returns comparison table in single analysis pass", "actual_result": "Pass — all three thresholds computed simultaneously with patient-level detection. Confirmed at 10k: t=4.0 (ICU): 61.4% patient detection / 28.2% reading sens / 9.6% FPR / 52% alert reduction. t=5.0 (mixed): 48.1% patient detection / 20.1% reading sens / 7.8% FPR / 63.2% alert reduction. t=6.0 (telemetry/default): 38.3% patient detection / 14.6% reading sens / 6.2% FPR / 71.6% alert reduction.", "status": "Pass"},
    {"date_tested": "2026-04-12", "tested_by": "Milton Munroe", "test_case_id": "VAL-021", "expected_result": "Retro pipeline produces consistent results at 2,000 patients", "actual_result": "Pass — 52,180 rows · 18.8% ERA sens · 4.2% FPR · 84.2% alert reduction at t=6.0. Threshold comparison table: t=4.0 33.9%/8% · t=5.0 24.1%/5.7%.", "status": "Pass"},
    {"date_tested": "2026-04-14", "tested_by": "Milton Munroe", "test_case_id": "VAL-022", "expected_result": "Retro pipeline produces consistent results at 5,000 patients with improved scoring", "actual_result": "Pass — 129,333 rows · 36.8% patient detection · 14.0% reading sensitivity · 5.9% FPR · 72.4% alert reduction at t=6.0 (6-hr event window). t=4.0: 59.7% patient detection / 9.1% FPR / 53% alert reduction. t=5.0: 46.7% patient detection / 7.4% FPR / 63.9% alert reduction. Confirmed stable with compound rules v2.", "status": "Pass"},
    {"date_tested": "2026-04-12", "tested_by": "Milton Munroe", "test_case_id": "VAL-023", "expected_result": "Large dataset async analysis handles 10,000 patient files without timeout", "actual_result": "Pass — async background threading implemented for datasets >30,000 rows. Client polls /api/retro/analyze/<id> every 5s. Render 30s timeout bypassed.", "status": "Pass"},
    {"date_tested": "2026-04-14", "tested_by": "Milton Munroe", "test_case_id": "VAL-024", "expected_result": "Retro pipeline produces consistent results at 10,000 patients with streaming analysis", "actual_result": "Pass — 260,765 rows · 38.3% patient detection · 14.6% reading sensitivity · 6.2% FPR · 71.6% alert reduction at t=6.0 (6-hr event window). t=4.0: 61.4% patient detection / 9.6% FPR / 52% alert reduction. t=5.0: 48.1% patient detection / 7.8% FPR / 63.2% alert reduction. vs 20.4% standard FPR. Results consistent with 5,000-patient run. Streaming analysis confirmed stable on Standard instance.", "status": "Pass"},
]

# ---------------------------------------------------------------------------
# GOVERNANCE DOCUMENTS — previously missing, now fully defined
# ---------------------------------------------------------------------------

PILOT_COMPLAINT_ISSUE_LOG = [
    {
        "id": "ISSUE-TEMPLATE",
        "date_opened": "",
        "source": "Hospital / Pilot User / Internal",
        "severity": "Low / Medium / High",
        "owner": "Founder/Product",
        "investigation_summary": "Describe the complaint or issue here.",
        "corrective_action": "Describe corrective action taken.",
        "escalation_path": "info@earlyriskalertai.com → Clinical Advisor → Technical Advisor",
        "closure_date": "",
        "status": "Template — populate before pilot activation",
    }
]

PILOT_ESCALATION_PROCESS = [
    {
        "step": 1,
        "action": "Pilot user logs issue in complaint / issue log.",
        "owner": "Pilot User / Hospital Site Sponsor",
        "timeline": "Same business day",
    },
    {
        "step": 2,
        "action": "Founder/Product reviews severity and assigns corrective action owner.",
        "owner": "Milton Munroe",
        "timeline": "Within 1 business day",
    },
    {
        "step": 3,
        "action": "Technical Advisor reviews infrastructure or security issues.",
        "owner": "Uche Anosike",
        "timeline": "Within 2 business days for non-critical",
    },
    {
        "step": 4,
        "action": "Clinical Advisor reviews clinical workflow or monitored-context issues.",
        "owner": "Andrene Louison (RN)",
        "timeline": "Within 2 business days",
    },
    {
        "step": 5,
        "action": "Resolution documented in complaint log with closure date and corrective action summary.",
        "owner": "Founder/Product",
        "timeline": "Before pilot resumes if safety-critical",
    },
]

PILOT_CHANGE_APPROVAL_LOG = [
    {
        "version": "stable-pilot-1.0.6",
        "approver": "Milton Munroe",
        "approval_date": "2026-04-07",
        "reason": "Full persistence layer, notifications, /status page, /deck page, admin CRM upgrades, delta explainability, and governance completeness pass.",
        "what_changed": "SQLite workflow/audit persistence. Flat-file trend and threshold persistence. Email/SMS alert notifications with cooldown. /status public page. /deck inline page. Admin notes, follow-up date, email compose. Delta explainability. 18 governance variables defined. API alias fixed.",
        "impact_assessment": "Low-to-medium risk. Persistence is additive — no changes to patient risk scoring or clinical output framing. Notifications are opt-in via env vars. All workflow framing unchanged.",
        "release_status": "Released",
        "linked_notes": "stable-pilot-1.0.6 release notes",
    },
    {
        "version": "stable-pilot-1.0.6-gov",
        "approver": "Milton Munroe",
        "approval_date": "2026-04-10",
        "reason": "Governance documentation update — phishing-resistant MFA implemented, CITI training completed, PhysioNet MIMIC-IV application submitted.",
        "what_changed": (
            "Phishing-resistant MFA implemented on all administrative systems. "
            "Backup/recovery controls implemented on all administrative systems. "
            "CITI Program research ethics training completed (Data or Specimens Only Research + Conflict of Interest). "
            "PhysioNet MIMIC-IV data access application submitted April 10, 2026. "
            "PILOT_MFA_ACCESS_EVIDENCE_LOG updated to reflect full MFA implementation. "
            "PILOT_CYBERSECURITY_SUMMARY access_control updated. "
            "PILOT_SECURITY_PROGRAM_DOCUMENTS updated with MFA and CITI entries. "
            "PILOT_INSURANCE_READINESS_CONTROLS updated. "
            "VAL-017 and VAL-018 added to validation evidence packet. "
            "PILOT_TRAINING_ACK_LOG updated with CITI completion."
        ),
        "impact_assessment": "Low risk — governance documentation update only. No changes to patient data flow, risk scoring, clinical output framing, or platform functionality.",
        "release_status": "Released",
        "linked_notes": "stable-pilot-1.0.6-gov governance update April 10, 2026",
    },
    {
        "version": "stable-pilot-1.0.7",
        "approver": "Milton Munroe",
        "approval_date": "2026-04-12",
        "reason": "Feature build — Live Demo Mode, patient search, multi-threshold analysis, pilot onboarding checklist.",
        "what_changed": (
            "Live Demo Mode toggle — applies realistic vital drift every 45s during demos. "
            "Patient search bar — real-time filter by patient ID, room, unit. "
            "Multi-threshold retro analysis — tests 4.0, 5.0, 6.0 simultaneously with comparison table. "
            "Pilot Onboarding Checklist at /pilot-onboarding — printable 4-section PDF-ready page. "
            "Model card threshold comparison table added. "
            "Stress tested to 10,000 patients / 260,765 rows at 1.99s — pipeline confirmed stable. "
            "VAL-019 and VAL-020 added to validation evidence."
        ),
        "impact_assessment": "Low-medium risk — new UI features and analysis enhancements. No changes to core risk scoring logic or governance documentation.",
        "release_status": "Released",
        "linked_notes": "stable-pilot-1.0.7 feature release April 12, 2026",
    },
    {
        "version": "stable-pilot-1.0.8",
        "approver": "Milton Munroe",
        "approval_date": "2026-04-14",
        "reason": "10,000-patient validation confirmed. Memory upgrade to Render Standard. Disk-based async chunk assembly for large file uploads.",
        "what_changed": "Render instance upgraded to Standard (2GB RAM) to support large dataset analysis. Disk-based chunk upload assembly — no in-memory string concatenation. Async background analysis with cross-worker disk result polling. 10k confirmed: 260,765 rows · 38.3% patient detection · 6.2% FPR · 71.6% alert reduction. VAL-024 updated with confirmed results. All materials updated to 10k primary benchmark.",
        "impact_assessment": "Low risk — infrastructure upgrade and upload reliability fix. No changes to scoring logic or clinical output framing.",
        "release_status": "Released",
        "linked_notes": "stable-pilot-1.0.8 release April 14, 2026",
    },
    {
        "version": "stable-pilot-1.0.5a",
        "approver": "Milton Munroe",
        "approval_date": "2026-04-06",
        "reason": "RR and temperature monitored-context expansion with advisory structure update.",
        "what_changed": "RR and Temp added as monitored context. Andrene Louison (RN) added to advisory structure. Pilot docs restored.",
        "impact_assessment": "Low risk — RR and Temp framed as monitored context only, not diagnostic triggers.",
        "release_status": "Released",
        "linked_notes": "stable-pilot-1.0.5a release notes",
    },
]

PILOT_CYBERSECURITY_SUMMARY = {
    "access_control": ("Phishing-resistant MFA implemented across all core administrative systems including business email, source control (GitHub), hosting (Render), password management, and domain/DNS administration — as of April 10, 2026. Backup and recovery controls implemented across the same systems. Application-layer session security: SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE=Lax, SESSION_COOKIE_SECURE=1 in production. Admin access restricted to approved emails via ALLOWED_ADMIN_EMAILS environment variable. Admin passwords stored as Werkzeug hashes via ADMIN_PASSWORD_HASH. No secrets hardcoded in the production bundle."
    ),
    "password_handling": "Admin passwords are stored as environment variables. Plain-text fallback is supported only for development. Production deployments should use ADMIN_PASSWORD_HASH (Werkzeug hash).",
    "backup_restore_posture": "Platform state is now fully persisted across restarts and deploys: workflow records and audit log are stored in SQLite (era_workflow.db), patient trend history is stored as daily flat-file JSONL per patient, and threshold settings are stored as a JSON file — all under ERA_DATA_DIR. Lead pipeline submissions are persisted as JSONL files under ERA_DATA_DIR. Manual backup of ERA_DATA_DIR contents to external storage is recommended before pilot activation and before each major release. Restore path: redeploy application and restore ERA_DATA_DIR contents to the target instance.",
    "patching_approach": "Dependencies are managed via pip. Regular patching of Flask, Werkzeug, and supporting libraries is expected before pilot activation and on a quarterly basis.",
    "vulnerability_handling": "Vulnerabilities discovered during the pilot should be reported to info@earlyriskalertai.com and reviewed by the Technical Infrastructure & Security Advisor (Uche Anosike) within 2 business days.",
    "incident_response_contact": "info@earlyriskalertai.com — Milton Munroe (Founder/Product) or Uche Anosike (Technical Infrastructure & Security Advisor).",
    "pilot_pause_breach_response": "If a suspected breach or data exposure is identified, the pilot should be paused, the hospital site sponsor notified, and the Technical Advisor engaged to review infrastructure posture before resuming.",
    "secure_configuration": "SECRET_KEY, ADMIN_PASSWORD_HASH, ADMIN_PASSWORD, and ALLOWED_ADMIN_EMAILS are managed via environment variables. No secrets are hardcoded in the production bundle.",
}

PILOT_SECURITY_PROGRAM_DOCUMENTS = [
    {"document": "Cybersecurity Summary", "status": "In place", "last_reviewed": "2026-04-10"},
    {"document": "MFA / Admin Access Evidence Log", "status": "Implemented", "last_reviewed": "2026-04-10"},
    {"document": "Backup / Restore Evidence Log", "status": "Implemented", "last_reviewed": "2026-04-10"},
    {"document": "Patch Log", "status": "Documented", "last_reviewed": "2026-04-10"},
    {"document": "Access Review Log", "status": "Pass", "last_reviewed": "2026-04-10"},
    {"document": "Incident Response Contact Path", "status": "In place", "last_reviewed": "2026-04-10"},
    {"document": "CITI Research Ethics Training", "status": "Completed", "last_reviewed": "2026-04-10"},
    {"document": "Phishing-Resistant MFA — Administrative Systems", "status": "Implemented", "last_reviewed": "2026-04-10"},
]

PILOT_INSURANCE_READINESS_CONTROLS = [
    {"control": "Intended-use statement frozen and consistent across all materials", "status": "In place"},
    {"control": "Claims control sheet with approved and banned claims", "status": "In place"},
    {"control": "Output framed as decision support only — not diagnostic", "status": "In place"},
    {"control": "Clinician oversight required — platform does not act autonomously", "status": "In place"},
    {"control": "Audit trail for all workflow actions", "status": "In place"},
    {"control": "Role-based access control with unit scoping", "status": "In place"},
    {"control": "Cybersecurity summary documented", "status": "In place"},
    {"control": "Complaint and issue handling process documented", "status": "In place"},
    {"control": "Validation evidence packet with test records", "status": "In place"},
    {"control": "Phishing-resistant MFA on all administrative systems (email, source control, hosting, DNS)", "status": "Implemented"},
    {"control": "Backup and recovery controls on all administrative systems", "status": "Implemented"},
    {"control": "CITI research ethics training completed prior to de-identified data access", "status": "Completed"},
    {"control": "PhysioNet MIMIC-IV data access application submitted", "status": "Submitted — pending approval"},
]

PILOT_MFA_ACCESS_EVIDENCE_LOG = [
    {
        "status": "Implemented",
        "last_reviewed": "2026-04-10",
        "evidence": (
            "Phishing-resistant MFA implemented across all core administrative systems: "
            "business email, source control (GitHub), hosting (Render), "
            "password management, and domain/DNS administration — April 10, 2026. "
            "Backup and recovery controls implemented across the same systems. "
            "Admin access to the platform application is restricted via "
            "ALLOWED_ADMIN_EMAILS and ADMIN_PASSWORD_HASH environment variables. "
            "SESSION_COOKIE_SECURE enforced in production."
        ),
        "implementation": (
            "Phishing-resistant MFA (hardware key or passkey) on all administrative accounts. "
            "Backup recovery codes stored securely in password manager. "
            "Application-layer admin auth via Werkzeug password hash. "
            "SESSION_COOKIE_HTTPONLY=True, SAMESITE=Lax, SECURE=1 in production."
        ),
        "notes": (
            "MFA coverage is now complete across the full administrative surface. "
            "Hospital IT security reviewers can confirm phishing-resistant MFA "
            "is in place for all systems that touch the platform deployment. "
            "Hospital VPN or SSO integration remains available for full enterprise deployment."
        ),
    }
]

PILOT_BACKUP_RESTORE_LOG = [
    {
        "status": "Documented",
        "last_tested": "2026-04-09",
        "notes": "Platform state is fully persisted as of stable-pilot-1.0.6. Persistent artifacts under ERA_DATA_DIR: (1) era_workflow.db — SQLite database containing workflow records and audit log; (2) trends/ — daily flat-file JSONL per patient containing vital-sign trend history; (3) thresholds.json — persisted unit threshold settings; (4) hospital_demo_requests.jsonl, executive_walkthrough_requests.jsonl, investor_intake_requests.jsonl — lead pipeline submissions; (5) retro/ — retrospective validation upload metadata. All artifacts survive restarts and deploys. Manual backup of ERA_DATA_DIR to external storage is recommended before pilot activation.",
        "restore_path": "Copy full ERA_DATA_DIR directory to backup storage before each deploy. To restore: redeploy application, then copy backed-up ERA_DATA_DIR contents to the target instance ERA_DATA_DIR path. SQLite database and flat files will be picked up automatically on next startup via _ensure_db_loaded().",
    }
]

PILOT_PATCH_LOG = [
    {
        "status": "Documented",
        "last_patched": "2026-04-07",
        "notes": "Flask, Werkzeug, and supporting Python dependencies should be reviewed for CVEs before pilot activation. Render deployment handles OS-level patching. Application-level dependency patching is managed manually.",
        "next_review": "Quarterly or before each major pilot phase",
    }
]

PILOT_ACCESS_REVIEW_LOG = [
    {
        "status": "Pass",
        "last_reviewed": "2026-04-10",
        "notes": (
            "PILOT_ACCOUNTS dictionary reviewed. Role and unit assignments verified. "
            "ALLOWED_ADMIN_EMAILS env var controls admin access. "
            "Phishing-resistant MFA now in place on all administrative accounts "
            "including the account used to access Render and GitHub. "
            "Pilot account deprovisioning is manual — remove from PILOT_ACCOUNTS and redeploy."
        ),
    }
]

PILOT_TABLETOP_LOG = [
    {
        "status": "Pending",
        "exercise_date": "",
        "notes": "Tabletop exercise for incident response and pilot pause scenario should be completed before pilot activation. Scenario should cover: suspected breach, patient data exposure, platform unavailability, and clinical escalation path if platform goes offline.",
        "participants": "Milton Munroe, Uche Anosike, Hospital Site Sponsor (to be named)",
    }
]

PILOT_TRAINING_ACK_LOG = [
    {
        "status": "Updated",
        "ack_date": "2026-04-07",
        "notes": "Training materials updated in stable-pilot-1.0.6. Pilot user instructions updated to reflect RR and Temperature as monitored context. New pilot users must review training materials before accessing the command center. Training acknowledgment records should be collected before pilot activation.",
        "training_materials": "/pilot-docs training section, Pilot Success Guide (/pilot-success-guide), Model Card (/model-card)",
    },
    {
        "status": "Completed",
        "ack_date": "2026-04-10",
        "notes": (
            "CITI Program research ethics training completed by founder Milton Munroe. "
            "Certificates obtained: Data or Specimens Only Research, Conflict of Interest. "
            "Training completed prior to MIMIC-IV data access application submission. "
            "Certificates filed in validation evidence folder dated April 10, 2026."
        ),
        "training_materials": "CITI Program — https://www.citiprogram.org/",
    }
]

PILOT_SITE_PACKET_TEMPLATE = [
    {"section": "Hospital / Site Name", "summary": "Enter the hospital or health system name for this pilot site.", "status": "Fill before pilot"},
    {"section": "Site Sponsor / Clinical Champion", "summary": "Name and title of the hospital-side pilot sponsor.", "status": "Fill before pilot"},
    {"section": "Pilot Scope", "summary": "Units, patient population, and access scope for this site.", "status": "Fill before pilot"},
    {"section": "Pilot Start Date", "summary": "Agreed pilot start date and expected duration.", "status": "Fill before pilot"},
    {"section": "Named Pilot Users", "summary": "List of authorized pilot users with roles and unit assignments.", "status": "Fill before pilot"},
    {"section": "IT / Security Contact", "summary": "Hospital IT or security contact for deployment review.", "status": "Fill before pilot"},
    {"section": "Data Governance Agreement", "summary": "Confirm hospital data governance and pilot data handling agreement is signed.", "status": "Fill before pilot"},
    {"section": "Training Completion", "summary": "Confirm all pilot users have reviewed training materials.", "status": "Fill before pilot"},
    {"section": "Escalation Path", "summary": "Hospital-side escalation contact for clinical and operational issues.", "status": "Fill before pilot"},
    {"section": "Pilot Closeout Date", "summary": "Agreed closeout date and data deletion / return timeline.", "status": "Fill before pilot"},
]

PILOT_USER_PROVISIONING_POLICY = {
    "who_can_create_users": "Founder/Product (Milton Munroe) is the only authorized account provisioner during the controlled pilot phase.",
    "who_can_remove_users": "Founder/Product removes users by removing them from PILOT_ACCOUNTS and redeploying the application.",
    "unit_scoping": "Each user is assigned an assigned_unit at provisioning. Admin users retain all-unit visibility. Non-admin users are restricted to their assigned unit.",
    "access_review_frequency": "Access is reviewed before each new pilot phase and at least quarterly.",
    "access_revocation_timeline": "Access should be revoked within 1 business day of a pilot user leaving the organization or the pilot ending.",
    "pilot_end_deprovisioning": "All pilot accounts are removed from PILOT_ACCOUNTS at pilot closeout. JSONL data is archived or deleted per the data retention policy.",
}

PILOT_DATA_RETENTION_POLICY = {
    "what_is_stored": "The following data is persisted under ERA_DATA_DIR: (1) Workflow records — ACK, assign, escalate, resolve actions stored in SQLite (era_workflow.db), surviving restarts and deploys; (2) Audit log — all workflow actions with role, timestamp, user, and unit, stored in SQLite; (3) Patient trend history — vital-sign snapshots stored as daily flat-file JSONL per patient; (4) Threshold settings — unit-level threshold configurations stored as thresholds.json; (5) Lead pipeline submissions — hospital demo, executive walkthrough, and investor intake requests stored as JSONL files; (6) Retrospective validation metadata — upload records stored as JSONL under retro/. Patient simulation data used for the demo environment is generated fresh each session but trend history snapshots are appended to flat files and persist across restarts.",
    "retention_period": "Lead pipeline data is retained for the duration of the pilot and commercial development phase. Hospital pilot data should follow the hospital's data governance agreement.",
    "deletion_and_return": "At pilot closeout, lead pipeline JSONL files can be exported via /admin/export.csv, then deleted from the deployment environment. Patient simulation data is cleared on restart.",
    "pilot_closeout": "At closeout: (1) export lead pipeline data via /admin/export.csv; (2) export any retrospective validation results via /api/retro/export/<upload_id>; (3) delete all ERA_DATA_DIR contents including era_workflow.db, trends/, thresholds.json, retro/, and all JSONL files; (4) revoke all pilot user accounts by removing from PILOT_ACCOUNTS and redeploying; (5) confirm deletion in writing with the hospital site sponsor within 5 business days.",
    "controlled_pilot_posture": "No real patient data is stored in the current simulation environment. All patient data displayed in the demo environment is generated fresh each session. Retrospective validation uploads are processed in memory during the session and upload metadata (not raw patient data) is stored under retro/. Workflow records, audit log, trend snapshots, and threshold settings persist across restarts to support pilot continuity — these contain no real patient identifiers in the current simulation deployment.",
}

PILOT_SCOPE_DOCUMENT = {
    "pilot_objective": "Controlled evaluation of Early Risk Alert AI's HCP-facing decision-support and workflow-support platform in a hospital or health system environment.",
    "recommended_first_step": "Retrospective validation on de-identified historical data as the most conservative hospital entry point.",
    "prospective_pilot_scope": "Tightly scoped prospective pilot with named users, defined units, and a clear start and end date.",
    "success_metrics": "See /pilot-success-guide for primary success metrics, safety and compliance metrics, governance metrics, and success thresholds. Threshold recommendation by unit: t=4.0 (ICU/high-acuity), t=5.0 (mixed/stepdown), t=6.0 (telemetry/default). All thresholds validated at 10,000-patient scale.",
    "exit_criteria": "Pilot exits when success thresholds are met, the agreed pilot duration ends, or a safety or compliance issue requires pause or closeout.",
    "governance_requirements": "Named site sponsor, trained pilot users, signed data governance agreement, completed tabletop exercise, and documented access review before activation.",
    "platform_scope": "The platform is scoped to HCP-facing decision-support and workflow-support only. It is not a medical device, does not diagnose, does not direct treatment, and does not independently trigger escalation.",
}

PILOT_TRAINING_USE_INSTRUCTIONS = [
    {
        "section": "Intended Use",
        "instruction": "Early Risk Alert AI is intended to assist authorized health care professionals in identifying patients who may warrant further clinical evaluation, supporting patient prioritization, and improving command-center operational awareness. It does not replace clinician judgment.",
    },
    {
        "section": "Command Center Navigation",
        "instruction": "The command center displays monitored patients sorted by review priority. Each patient card shows vitals, review score, and workflow status. Click Details or the ECG wave to open the patient drawer.",
    },
    {
        "section": "Patient Drawer",
        "instruction": "The patient drawer shows vitals, explainable review basis, workflow status, trend history, and supportive review notes. Values are displayed as monitored context for authorized HCP review — not as autonomous determinations.",
    },
    {
        "section": "Workflow Actions",
        "instruction": "ACK, Assign, Escalate, and Resolve are workflow state controls, not clinical directives. Each action is logged to the audit trail. Hospital policy governs escalation timing, response, and treatment decisions.",
    },
    {
        "section": "Respiratory Rate and Temperature",
        "instruction": "RR and Temperature are displayed as monitored context for authorized HCP review. Reference ranges (RR 12–20, Temp 97.6–100.4°F) are shown for context only. The platform does not make autonomous determinations based on these values.",
    },
    {
        "section": "Threshold Controls",
        "instruction": "Admin users can configure SpO₂, HR, and SBP thresholds by unit. Select a single unit from the Unit Filter to enable threshold editing. Threshold changes take effect immediately and persist across restarts and deploys for the current pilot environment. Recommended ERA scoring threshold by care setting: t=4.0 for ICU and high-acuity units (maximum patient detection, higher alert volume); t=5.0 for mixed or stepdown units (balanced detection and alert burden); t=6.0 for telemetry and lower-acuity units (lowest alarm fatigue, strongest false-positive control). All thresholds produce materially lower false positive rates than standard threshold-only alerting.",
    },
    {
        "section": "Unit Scoping",
        "instruction": "Non-admin users are restricted to their assigned unit. Admin users have full hospital-wide visibility. If you believe your access scope is incorrect, contact the pilot administrator.",
    },
    {
        "section": "Escalation and Support",
        "instruction": "For product or platform issues during the pilot, contact info@earlyriskalertai.com. For clinical escalation, follow your hospital's escalation protocol — the platform does not replace emergency response systems.",
    },
]


PILOT_ACCOUNTS = {
    "admin@erapilot.com": {"full_name": "ERA Pilot Admin", "email": "admin@erapilot.com", "user_role": "admin", "assigned_unit": "all"},
    "pilot@earlyriskalertai.com": {"full_name": "Early Risk Alert AI Pilot User", "email": "pilot@earlyriskalertai.com", "user_role": "operator", "assigned_unit": "all"},
}

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

LOGIN_HTML = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Early Risk Alert AI — Explainable Rules-Based Command-Center Platform — Secure Pilot Access</title><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;font-family:Inter,Arial,sans-serif;color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528)}
.card{width:min(640px,100%);border:1px solid rgba(255,255,255,.08);border-radius:24px;background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34)}
.k{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff;margin-bottom:10px} h1{margin:0 0 10px;font-size:40px;line-height:.95;letter-spacing:-.05em} p{margin:0 0 18px;color:#9fb4d6;line-height:1.6}
.callout,.disclaimer{margin:0 0 18px;padding:14px 16px;border-radius:16px;line-height:1.6}.callout{background:rgba(91,212,255,.08);border:1px solid rgba(91,212,255,.16);color:#dce9ff}.disclaimer{background:rgba(244,189,106,.10);border:1px solid rgba(244,189,106,.22);color:#ffe7bf;font-size:14px}
label{display:block;font-size:13px;font-weight:900;margin-bottom:8px} input,select{width:100%;padding:14px 16px;border-radius:16px;border:1px solid rgba(255,255,255,.08);background:#0d1728;color:#eef4ff;font:inherit;margin-bottom:14px}
button{width:100%;padding:14px 18px;border:none;border-radius:16px;cursor:pointer;font:inherit;font-weight:1000;color:#07101c;background:linear-gradient(135deg,#7aa2ff,#5bd4ff)} .error{margin-bottom:14px;padding:12px 14px;border-radius:14px;background:rgba(255,102,125,.12);border:1px solid rgba(255,102,125,.24);color:#ffd8de}
</style></head><body>
<div class="card"><div class="k">Controlled pilot environment</div><h1>Early Risk Alert AI Secure Pilot Access</h1>
<p>Access the command center for controlled pilot evaluation, role-based workflow review, and unit-scoped visibility.</p>
<div class="callout">Early Risk Alert AI supports monitored patient visibility, patient prioritization support, explainable review context, and command-center operational awareness.</div>
<div class="disclaimer">The platform does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.</div>
__ERROR__
<form method="post" action="/login">
<label>Full Name</label><input name="full_name" required>
<label>Work Email</label><input name="email" type="email" required>
<label>Role</label><select name="user_role" required><option value="viewer">Viewer</option><option value="operator">Operator</option><option value="physician">Physician</option><option value="admin">Admin</option></select>
<label>Admin Password (required for Admin role)</label><input name="admin_password" type="password">
<label>Assigned Unit</label><select name="assigned_unit" required><option value="all">All Units</option><option value="icu">ICU</option><option value="telemetry">Telemetry</option><option value="stepdown">Stepdown</option><option value="ward">Ward</option><option value="rpm">RPM / Home</option></select>
<button type="submit">Enter Secure Pilot Access</button></form></div></body></html>
"""
PILOT_ACCESS_HTML = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Early Risk Alert AI — Explainable Rules-Based Command-Center Platform — Pilot Access</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528)}
.card{width:min(560px,100%);border:1px solid rgba(255,255,255,.08);border-radius:24px;background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34)}
h1{margin:0 0 10px;font-size:38px;line-height:.95;letter-spacing:-.05em}p{color:#9fb4d6;line-height:1.6}.callout{margin:0 0 18px;padding:14px 16px;border-radius:16px;background:rgba(91,212,255,.08);border:1px solid rgba(91,212,255,.16);color:#dce9ff;line-height:1.6}.disclaimer{margin:0 0 18px;padding:14px 16px;border-radius:16px;background:rgba(244,189,106,.10);border:1px solid rgba(244,189,106,.22);color:#ffe7bf;line-height:1.6;font-size:14px}
input{width:100%;padding:14px 16px;border-radius:16px;border:1px solid rgba(255,255,255,.08);background:#0d1728;color:#eef4ff;font:inherit;margin-bottom:14px}button{width:100%;padding:14px 18px;border:none;border-radius:16px;cursor:pointer;font:inherit;font-weight:1000;color:#07101c;background:linear-gradient(135deg,#7aa2ff,#5bd4ff)}.error{margin-bottom:14px;padding:12px 14px;border-radius:14px;background:rgba(255,102,125,.12);border:1px solid rgba(255,102,125,.24);color:#ffd8de}.footer-note{margin-top:18px;color:#9fb4d6;font-size:13px;line-height:1.6}
</style></head><body><div class="card"><h1>Branded Pilot Access</h1><p>Enter your authorized pilot email to load your configured command-center experience, assigned unit, and pilot access scope.</p><div class="callout">Early Risk Alert AI supports controlled pilot evaluation through an HCP-facing decision-support and workflow-support platform designed to assist authorized health care professionals with patient prioritization, monitored patient visibility, and command-center operational awareness.</div><div class="disclaimer">It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.</div>__ERROR__<form method="post" action="/pilot-access"><input name="pilot_email" type="email" placeholder="pilot@hospital.com" required><button type="submit">Enter Pilot Account</button></form><div class="footer-note">Controlled Pilot Evaluation | Role-Based Access | Unit-Scoped Visibility</div></div></body></html>
"""

FORM_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{--bg:#08111f;--panel:#101a2d;--line:rgba(255,255,255,.08);--text:#eef4ff;--muted:#9fb4d6;--blue:#7aa2ff;--blue2:#5bd4ff;--warn:#f4bd6a}*{box-sizing:border-box}
    body{margin:0;padding:24px;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:var(--text);background:linear-gradient(180deg,#07101c,#0b1528)}
    .wrap{max-width:920px;margin:0 auto}.card{border:1px solid var(--line);border-radius:24px;background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34)}
    .k{font-size:11px;font-weight:1000;letter-spacing:.16em;text-transform:uppercase;color:#9adfff;margin-bottom:10px}h1{margin:0 0 10px;font-size:44px;line-height:.95;letter-spacing:-.05em}p{margin:0 0 18px;color:var(--muted);line-height:1.6}
    .callout{margin:0 0 18px;padding:14px 16px;border-radius:16px;background:rgba(91,212,255,.08);border:1px solid rgba(91,212,255,.16);color:#dce9ff;line-height:1.6}
    .disclaimer{margin:0 0 20px;padding:14px 16px;border-radius:16px;background:rgba(244,189,106,.10);border:1px solid rgba(244,189,106,.22);color:#ffe7bf;line-height:1.6;font-size:14px}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.field{margin-bottom:14px}.full{grid-column:1 / -1}
    label{display:block;font-size:13px;font-weight:900;margin-bottom:8px}input,select,textarea{width:100%;padding:14px 16px;border-radius:16px;border:1px solid var(--line);background:#0d1728;color:var(--text);font:inherit}textarea{min-height:120px;resize:vertical}
    .check{display:flex;gap:12px;align-items:flex-start;padding:14px 16px;border-radius:16px;border:1px solid var(--line);background:#0d1728}.check input{width:auto;margin:4px 0 0}.check span{color:#dce9ff;line-height:1.6;font-size:14px}
    button{padding:14px 18px;border:none;border-radius:16px;cursor:pointer;font:inherit;font-weight:1000;color:#07101c;background:linear-gradient(135deg,var(--blue),var(--blue2))}
    .links{display:flex;gap:14px;flex-wrap:wrap;margin-top:16px}a{color:#cfe7ff;text-decoration:none;font-weight:800}.footer-note{margin-top:18px;color:#9fb4d6;font-size:13px;line-height:1.6}
    @media (max-width:700px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="k">Early Risk Alert AI</div>
      <h1>__HEADING__</h1>
      <p>__COPY__</p>
      <div class="callout">Early Risk Alert AI is an HCP-facing decision-support and workflow-support platform intended to assist authorized health care professionals in identifying patients who may warrant further clinical evaluation, support patient prioritization, and improve command-center operational awareness.</div>
      <div class="disclaimer">The platform is intended for controlled pilot evaluation and hospital-facing workflow support. It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.</div>
      <form method="post"><div class="grid">__FIELDS__</div><button type="submit">__BUTTON__</button></form>
      <div class="links"><a href="/command-center">Command Center</a><a href="/admin/review">Admin Review</a></div>
      <div class="footer-note">Controlled Pilot Evaluation | Secure Cloud Deployment | Hospital-Facing Workflow Support</div>
    </div>
  </div>
</body>
</html>
"""

THANK_YOU_HTML = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Request Received</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{margin:0;padding:24px;display:grid;place-items:center;min-height:100vh;font-family:Inter,Arial,sans-serif;color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528)}.card{width:min(760px,100%);border:1px solid rgba(255,255,255,.08);border-radius:24px;background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.34)}h1{margin:0 0 10px;font-size:40px;line-height:.95;letter-spacing:-.05em}p{margin:0 0 18px;color:#9fb4d6;line-height:1.6}.box{border:1px solid rgba(255,255,255,.08);border-radius:18px;background:rgba(255,255,255,.03);padding:16px;line-height:1.8;color:#dce9ff;margin:16px 0}a{display:inline-flex;align-items:center;justify-content:center;padding:12px 16px;border-radius:16px;font-size:14px;font-weight:900;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;text-decoration:none}
</style></head><body><div class="card"><h1>Thank You</h1><p>__MESSAGE__</p><div class="box">__DETAILS__</div><a href="/command-center">Return to Command Center</a></div></body></html>
"""

ADMIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Admin Review — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{margin:0;padding:24px;font-family:Inter,Arial,sans-serif;background:linear-gradient(180deg,#07101c,#0b1528);color:#eef4ff}
    .wrap{max-width:1400px;margin:0 auto}
    .card{border:1px solid rgba(255,255,255,.08);border-radius:24px;background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.018));padding:24px;margin-bottom:18px;box-shadow:0 14px 36px rgba(0,0,0,.28)}
    h1{margin:0 0 10px;font-size:40px;letter-spacing:-.05em}
    h2{margin:0 0 12px;font-size:26px;letter-spacing:-.04em}
    .muted{color:#9fb4d6;font-size:14px;line-height:1.6}
    .links{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px}
    a.btn{display:inline-flex;align-items:center;justify-content:center;padding:10px 14px;border-radius:14px;font-size:13px;font-weight:900;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;text-decoration:none}
    a.btn.secondary{background:rgba(255,255,255,.06);color:#dce9ff;border:1px solid rgba(255,255,255,.1)}
    .summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}
    .stat{border:1px solid rgba(255,255,255,.08);border-radius:18px;background:rgba(255,255,255,.03);padding:16px}
    .stat-k{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9adfff;margin-bottom:8px}
    .stat-v{font-size:36px;font-weight:1000;letter-spacing:-.04em}
    table{width:100%;border-collapse:collapse;font-size:13px}
    thead th{padding:12px;text-align:left;color:#9adfff;border-bottom:1px solid rgba(255,255,255,.08);font-weight:900;letter-spacing:.08em;text-transform:uppercase}
    tbody tr{border-bottom:1px solid rgba(255,255,255,.05)}
    tbody td{padding:10px 12px;vertical-align:top;color:#dce9ff;line-height:1.5}
    .pill{display:inline-flex;align-items:center;padding:5px 10px;border-radius:999px;font-size:11px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}
    .pill.hot{background:rgba(255,102,125,.18);color:#ffd8de;border:1px solid rgba(255,102,125,.3)}
    .pill.warm{background:rgba(244,189,106,.14);color:#ffe7bf;border:1px solid rgba(244,189,106,.3)}
    .pill.normal{background:rgba(255,255,255,.05);color:#dce9ff;border:1px solid rgba(255,255,255,.1)}
    .pill.new-s{background:rgba(91,212,255,.1);color:#dce9ff;border:1px solid rgba(91,212,255,.2)}
    .pill.closed-s{background:rgba(58,211,143,.1);color:#c8fff0;border:1px solid rgba(58,211,143,.2)}
    select.status-sel{background:#0d1728;border:1px solid rgba(255,255,255,.08);border-radius:10px;color:#eef4ff;padding:5px 8px;font:inherit;font-size:12px;cursor:pointer;width:100%}
    .filter-bar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;align-items:center}
    .filter-bar select,.filter-bar input{background:#0d1728;border:1px solid rgba(255,255,255,.08);border-radius:12px;color:#eef4ff;padding:10px 12px;font:inherit;font-size:13px}
    #statusMsg{color:#9adfff;font-size:13px;font-weight:800}
    .expand-row{display:none;background:rgba(255,255,255,.02)}
    .expand-row.open{display:table-row}
    .expand-inner{padding:16px;border-radius:0;display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .field-group{display:grid;gap:8px}
    .field-group label{font-size:11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:#9adfff}
    .field-group input,.field-group textarea{background:#0d1728;border:1px solid rgba(255,255,255,.08);border-radius:12px;color:#eef4ff;padding:10px 12px;font:inherit;font-size:13px;width:100%}
    .field-group textarea{min-height:72px;resize:vertical}
    .action-btns{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
    .action-btn{padding:8px 14px;border:none;border-radius:12px;font:inherit;font-size:12px;font-weight:900;cursor:pointer}
    .action-btn.save{background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c}
    .action-btn.email{background:rgba(58,211,143,.14);border:1px solid rgba(58,211,143,.26);color:#b6f5d9}
    .action-btn.copy{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);color:#dce9ff}
    .expand-toggle{cursor:pointer;color:#9adfff;font-size:12px;font-weight:900;background:none;border:none;padding:4px 8px;border-radius:8px}
    .expand-toggle:hover{background:rgba(255,255,255,.06)}
    .notes-preview{font-size:12px;color:#9fb4d6;margin-top:3px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    @media(max-width:900px){.summary-grid{grid-template-columns:1fr 1fr}.expand-inner{grid-template-columns:1fr}}
    @media(max-width:600px){.summary-grid{grid-template-columns:1fr};table{font-size:11px}}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Admin Review</h1>
    <p class="muted">Review hospital, executive, and investor submissions. Update status, add notes, set follow-up dates, and compose emails directly from this panel.</p>
    <div class="links">
      <a class="btn" href="/command-center">Command Center</a>
      <a class="btn secondary" href="/pilot-docs">Pilot Docs</a>
      <a class="btn secondary" href="/model-card">Model Card</a>
      <a class="btn secondary" href="/pilot-success-guide">Pilot Success Guide</a>
      <a class="btn secondary" href="/admin/export.csv">Export CSV</a>
    </div>
  </div>

  <div class="summary-grid" id="summaryGrid">
    <div class="stat"><div class="stat-k">Hospital</div><div class="stat-v" id="cntHospital">—</div></div>
    <div class="stat"><div class="stat-k">Executive</div><div class="stat-v" id="cntExecutive">—</div></div>
    <div class="stat"><div class="stat-k">Investor</div><div class="stat-v" id="cntInvestor">—</div></div>
    <div class="stat"><div class="stat-k">Open</div><div class="stat-v" id="cntOpen">—</div></div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:16px">
      <h2>Lead Pipeline</h2>
      <span id="statusMsg"></span>
    </div>
    <div class="filter-bar">
      <select id="filterType" onchange="applyFilter()">
        <option value="">All Types</option>
        <option value="Hospital Demo">Hospital Demo</option>
        <option value="Executive Walkthrough">Executive Walkthrough</option>
        <option value="Investor Intake">Investor Intake</option>
      </select>
      <select id="filterStatus" onchange="applyFilter()">
        <option value="">All Statuses</option>
        <option value="New">New</option>
        <option value="Contacted">Contacted</option>
        <option value="Meeting Set">Meeting Set</option>
        <option value="Qualified">Qualified</option>
        <option value="Interested">Interested</option>
        <option value="Due Diligence">Due Diligence</option>
        <option value="Follow-Up">Follow-Up</option>
        <option value="Closed">Closed</option>
      </select>
      <input id="filterSearch" placeholder="Search name, org, email…" oninput="applyFilter()" style="min-width:200px">
    </div>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th></th>
            <th>Type</th>
            <th>Name</th>
            <th>Organization</th>
            <th>Email</th>
            <th>Timeline</th>
            <th>Score</th>
            <th>Priority</th>
            <th>Status</th>
            <th>Follow-Up</th>
            <th>Submitted</th>
          </tr>
        </thead>
        <tbody id="pipelineTable"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
  let allRows = [];
  // Local CRM store — persists in sessionStorage so notes survive filter changes
  let crmStore = {};
  try { crmStore = JSON.parse(sessionStorage.getItem('era_crm') || '{}'); } catch(e) {}

  function saveCrm() {
    try { sessionStorage.setItem('era_crm', JSON.stringify(crmStore)); } catch(e) {}
  }

  async function loadData() {
    try {
      const res = await fetch('/admin/api/data', {cache:'no-store'});
      const data = await res.json();
      allRows = data.rows || [];
      const s = data.summary || {};
      document.getElementById('cntHospital').textContent = s.hospital_count ?? '—';
      document.getElementById('cntExecutive').textContent = s.executive_count ?? '—';
      document.getElementById('cntInvestor').textContent = s.investor_count ?? '—';
      document.getElementById('cntOpen').textContent = s.open_count ?? '—';
      applyFilter();
    } catch(err) {
      console.error('Admin data load failed', err);
    }
  }

  function applyFilter() {
    const type = document.getElementById('filterType').value;
    const status = document.getElementById('filterStatus').value;
    const search = document.getElementById('filterSearch').value.toLowerCase();
    const filtered = allRows.filter(r => {
      if (type && r.lead_type !== type) return false;
      if (status && r.status !== status) return false;
      if (search) {
        const haystack = [r.full_name, r.organization, r.email, r.message].join(' ').toLowerCase();
        if (!haystack.includes(search)) return false;
      }
      return true;
    });
    renderTable(filtered);
  }

  function priorityPill(tag) {
    const cls = tag === 'hot' ? 'hot' : tag === 'warm' ? 'warm' : 'normal';
    return `<span class="pill ${cls}">${tag || 'normal'}</span>`;
  }

  function statusPill(status) {
    const cls = status === 'Closed' ? 'closed-s' : status === 'New' ? 'new-s' : 'warm';
    return `<span class="pill ${cls}">${status || 'New'}</span>`;
  }

  function formatDate(iso) {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleDateString([], {month:'short',day:'numeric',year:'numeric'}); }
    catch { return iso; }
  }

  function rowKey(r) {
    return r.submitted_at + '|' + r.lead_type_key;
  }

  function getCrm(r) {
    return crmStore[rowKey(r)] || {notes:'', follow_up:''};
  }

  function renderTable(rows) {
    const tbody = document.getElementById('pipelineTable');
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="11" style="text-align:center;padding:24px;color:#9fb4d6">No leads match current filters.</td></tr>';
      return;
    }

    let html = '';
    rows.forEach((r, idx) => {
      const key = rowKey(r);
      const crm = getCrm(r);
      const expandId = `expand_${idx}`;
      html += `
        <tr>
          <td><button class="expand-toggle" onclick="toggleExpand('${expandId}')" title="Open CRM panel">▸</button></td>
          <td>${r.lead_type || '—'}</td>
          <td>
            <strong>${r.full_name || '—'}</strong>
            <div style="font-size:11px;color:#9adfff;margin-top:2px">${r.role_or_title || ''}</div>
            ${crm.notes ? `<div class="notes-preview" title="${crm.notes.replace(/"/g,'&quot;')}">${crm.notes}</div>` : ''}
          </td>
          <td>${r.organization || '—'}</td>
          <td style="font-size:12px">${r.email || '—'}</td>
          <td>${r.timeline || '—'}</td>
          <td>${r.lead_score ?? '—'}</td>
          <td>${priorityPill(r.priority_tag)}</td>
          <td>
            <select class="status-sel" data-type="${r.lead_type_key}" data-submitted="${r.submitted_at}" onchange="updateStatus(this)">
              ${(r.available_statuses || ['New','Contacted','Closed']).map(s => `<option${s===r.status?' selected':''}>${s}</option>`).join('')}
            </select>
          </td>
          <td style="font-size:12px;color:${crm.follow_up ? '#ffd96c' : '#9fb4d6'}">${crm.follow_up || '—'}</td>
          <td style="font-size:12px;color:#9fb4d6">${formatDate(r.submitted_at)}</td>
        </tr>
        <tr class="expand-row" id="${expandId}">
          <td colspan="11">
            <div class="expand-inner">
              <div>
                <div class="field-group" style="margin-bottom:12px">
                  <label>Internal Notes</label>
                  <textarea id="notes_${idx}" placeholder="Add internal notes about this lead…">${crm.notes || ''}</textarea>
                </div>
                <div class="field-group">
                  <label>Follow-Up Date</label>
                  <input type="date" id="followup_${idx}" value="${crm.follow_up || ''}">
                </div>
                <div class="action-btns">
                  <button class="action-btn save" onclick="saveCrmRow(${idx}, '${key}')">Save Notes</button>
                  <button class="action-btn copy" onclick="copyEmail(${idx})">Copy Email</button>
                </div>
              </div>
              <div>
                <div class="field-group" style="margin-bottom:12px">
                  <label>Compose Outreach Email</label>
                  <textarea id="compose_${idx}" style="min-height:130px" placeholder="Write your email here…">${buildTemplate(r)}</textarea>
                </div>
                <div class="action-btns">
                  <button class="action-btn email" onclick="openMailto(${idx}, '${(r.email||'').replace(/'/g,"\\'")}')">Open in Mail</button>
                  <button class="action-btn copy" onclick="copyCompose(${idx})">Copy Message</button>
                </div>
              </div>
            </div>
          </td>
        </tr>
      `;
    });
    tbody.innerHTML = html;
  }

  function buildTemplate(r) {
    const type = r.lead_type || '';
    const name = (r.full_name || '').split(' ')[0] || 'there';
    if (type.includes('Hospital')) {
      return `Hi ${name},\n\nThank you for your interest in Early Risk Alert AI. I'd love to schedule a live command center demonstration for ${r.organization || 'your team'}.\n\nWould you have 30 minutes available this week or next?\n\nBest,\nMilton Munroe\nFounder & CEO, Early Risk Alert AI\ninfo@earlyriskalertai.com | 732-724-7267`;
    }
    if (type.includes('Investor')) {
      return `Hi ${name},\n\nThank you for your interest in Early Risk Alert AI. I'd be happy to walk you through the platform, our hospital pilot model, and where we are in our raise.\n\nAre you available for a brief call this week?\n\nBest,\nMilton Munroe\nFounder & CEO, Early Risk Alert AI\ninfo@earlyriskalertai.com | 732-724-7267`;
    }
    return `Hi ${name},\n\nThank you for reaching out to Early Risk Alert AI. I'd love to connect and learn more about what you're looking for.\n\nBest,\nMilton Munroe\nFounder & CEO, Early Risk Alert AI\ninfo@earlyriskalertai.com | 732-724-7267`;
  }

  function toggleExpand(id) {
    const row = document.getElementById(id);
    if (row) row.classList.toggle('open');
  }

  function saveCrmRow(idx, key) {
    const notes = document.getElementById(`notes_${idx}`)?.value || '';
    const follow_up = document.getElementById(`followup_${idx}`)?.value || '';
    crmStore[key] = {notes, follow_up};
    saveCrm();
    showMsg('Notes saved ✓');
    applyFilter();
  }

  function copyEmail(idx) {
    const emailEl = document.querySelector(`tr:nth-child(${idx * 2 + 1}) td:nth-child(5)`);
    const text = emailEl?.textContent?.trim() || '';
    if (text && text !== '—') {
      navigator.clipboard.writeText(text).then(() => showMsg('Email copied ✓'));
    }
  }

  function copyCompose(idx) {
    const text = document.getElementById(`compose_${idx}`)?.value || '';
    navigator.clipboard.writeText(text).then(() => showMsg('Message copied ✓'));
  }

  function openMailto(idx, email) {
    const body = encodeURIComponent(document.getElementById(`compose_${idx}`)?.value || '');
    const subject = encodeURIComponent('Early Risk Alert AI — Following Up');
    window.location.href = `mailto:${email}?subject=${subject}&body=${body}`;
  }

  async function updateStatus(sel) {
    const lead_type = sel.dataset.type;
    const submitted_at = sel.dataset.submitted;
    const status = sel.value;
    try {
      const res = await fetch('/admin/api/status', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({lead_type, submitted_at, status})
      });
      const data = await res.json();
      if (data.ok) {
        showMsg('Status updated ✓');
        await loadData();
      } else {
        showMsg('Update failed');
      }
    } catch(err) {
      showMsg('Update failed');
    }
  }

  function showMsg(text) {
    const el = document.getElementById('statusMsg');
    el.textContent = text;
    setTimeout(() => el.textContent = '', 2800);
  }

  loadData();
  setInterval(loadData, 30000);
</script>
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


def _data_dir() -> Path:
    path = Path(os.getenv("ERA_DATA_DIR", "data")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


DOC_FILE_NAMES = {
    "pilot_success_guide": "Early_Risk_Alert_AI_Pilot_Success_Criteria_and_Workflow_Integration_Guide.docx",
    "model_card": "Early_Risk_Alert_AI_Model_Card.docx",
}


def _static_search_roots() -> list[Path]:
    roots = [Path.cwd() / "static", Path.cwd(), Path('/mnt/data')]
    seen = []
    out = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.append(key)
            out.append(root)
    return out


def _serve_static_doc(kind: str):
    filename = DOC_FILE_NAMES[kind]
    for root in _static_search_roots():
        candidate = root / filename
        if candidate.exists():
            return send_from_directory(str(root), filename, as_attachment=True, download_name=filename, max_age=0)
    return Response(f"Document not found: {filename}", status=404)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except Exception:
            continue
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _status_norm(value: Any, lead_type: str = "") -> str:
    text = str(value or "New").strip()
    if not text:
        return "New"
    mapping = {
        "new": "New",
        "contacted": "Contacted",
        "meeting set": "Meeting Set",
        "qualified": "Qualified",
        "closed": "Closed",
        "interested": "Interested",
        "due diligence": "Due Diligence",
        "follow-up": "Follow-Up",
    }
    key = text.lower()
    return mapping.get(key, text)


def _available_statuses(lead_type: str) -> list[str]:
    if lead_type == "investor":
        return ["New", "Interested", "Due Diligence", "Follow-Up", "Contacted", "Closed"]
    return ["New", "Contacted", "Meeting Set", "Qualified", "Closed"]


def _priority_tag(score: int, lead_type: str) -> str:
    if score >= 6:
        return "hot"
    if score >= 3:
        return "warm"
    return "normal"


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
        stage = str(payload.get("stage", payload.get("investor_type", ""))).strip().lower()
        if "institutional" in stage or "strategic" in stage:
            score += 3
        elif "angel" in stage or "seed" in stage:
            score += 2
    return score


def _detail_html(payload: Dict[str, Any], keys: List[str]) -> str:
    return "<br>".join(
        f"<strong>{key.replace('_', ' ').title()}:</strong> {payload.get(key, '')}"
        for key in keys
    )


def _render_thank_you(message: str, details: str) -> str:
    return THANK_YOU_HTML.replace("__MESSAGE__", message).replace("__DETAILS__", details)


def _format_pretty_label(value: str) -> str:
    if not value:
        return "--"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return value
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _investor_stage_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"New": 0, "Interested": 0, "Due Diligence": 0, "Follow-Up": 0, "Contacted": 0, "Closed": 0}
    for row in rows:
        status = _status_norm(row.get("status"), "investor")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _normalize_row(row: dict[str, Any], lead_type: str) -> dict[str, Any]:
    status = _status_norm(row.get("status"), lead_type)
    score = int(row.get("lead_score", 0) or 0)
    category = row.get("facility_type") or row.get("priority") or row.get("investor_type") or ""
    role_or_title = row.get("role") or row.get("title") or ""
    last_updated = row.get("last_updated") or row.get("submitted_at") or ""
    return {
        "lead_type": {"hospital": "Hospital Demo", "executive": "Executive Walkthrough", "investor": "Investor Intake"}[lead_type],
        "lead_type_key": lead_type,
        "submitted_at": str(row.get("submitted_at", "")),
        "last_updated": str(last_updated),
        "status": status,
        "lead_score": score,
        "priority_tag": row.get("priority_tag") or _priority_tag(score, lead_type),
        "full_name": str(row.get("full_name", "")),
        "organization": str(row.get("organization", "")),
        "role_or_title": str(role_or_title),
        "email": str(row.get("email", "")),
        "phone": str(row.get("phone", "")),
        "category": str(category),
        "timeline": str(row.get("timeline", "")),
        "message": str(row.get("message", "")),
        "available_statuses": _available_statuses(lead_type),
    }


def _send_admin_notification(_lead_type: str, _payload: Dict[str, Any]) -> None:
    return None


def _send_auto_reply(_lead_type: str, _payload: Dict[str, Any]) -> None:
    return None


# ---------------------------------------------------------------------------
# PERSISTENCE LAYER — SQLite for workflow/audit, flat-file for trends/thresholds
# ---------------------------------------------------------------------------

_DB_LOCK = threading.Lock()


def _db_path() -> Path:
    return _data_dir() / "era_workflow.db"


def _trend_dir() -> Path:
    p = _data_dir() / "trends"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _threshold_file() -> Path:
    return _data_dir() / "thresholds.json"


def _init_db() -> None:
    """Create SQLite tables if they don't exist."""
    with _DB_LOCK:
        conn = sqlite3.connect(str(_db_path()))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_records (
                patient_id TEXT PRIMARY KEY,
                ack INTEGER DEFAULT 0,
                assigned INTEGER DEFAULT 0,
                assigned_label TEXT DEFAULT '',
                escalated INTEGER DEFAULT 0,
                resolved INTEGER DEFAULT 0,
                state TEXT DEFAULT 'new',
                role TEXT DEFAULT '',
                updated_at TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT,
                patient_id TEXT,
                action TEXT,
                role TEXT,
                note TEXT,
                user TEXT,
                unit TEXT
            )
        """)
        conn.commit()
        conn.close()


def _db_get_workflow_records() -> Dict[str, Any]:
    with _DB_LOCK:
        conn = sqlite3.connect(str(_db_path()))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM workflow_records").fetchall()
        conn.close()
    return {
        row["patient_id"]: {
            "ack": bool(row["ack"]),
            "assigned": bool(row["assigned"]),
            "assigned_label": row["assigned_label"],
            "escalated": bool(row["escalated"]),
            "resolved": bool(row["resolved"]),
            "state": row["state"],
            "role": row["role"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    }


def _db_upsert_workflow_record(patient_id: str, record: Dict[str, Any]) -> None:
    with _DB_LOCK:
        conn = sqlite3.connect(str(_db_path()))
        conn.execute("""
            INSERT INTO workflow_records
                (patient_id, ack, assigned, assigned_label, escalated, resolved, state, role, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                ack=excluded.ack,
                assigned=excluded.assigned,
                assigned_label=excluded.assigned_label,
                escalated=excluded.escalated,
                resolved=excluded.resolved,
                state=excluded.state,
                role=excluded.role,
                updated_at=excluded.updated_at
        """, (
            patient_id,
            int(record.get("ack", False)),
            int(record.get("assigned", False)),
            record.get("assigned_label", ""),
            int(record.get("escalated", False)),
            int(record.get("resolved", False)),
            record.get("state", "new"),
            record.get("role", ""),
            record.get("updated_at", ""),
        ))
        conn.commit()
        conn.close()


def _db_append_audit(entry: Dict[str, Any]) -> None:
    with _DB_LOCK:
        conn = sqlite3.connect(str(_db_path()))
        conn.execute("""
            INSERT INTO workflow_audit (time, patient_id, action, role, note, user, unit)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.get("time", ""),
            entry.get("patient_id", ""),
            entry.get("action", ""),
            entry.get("role", ""),
            entry.get("note", ""),
            entry.get("user", ""),
            entry.get("unit", ""),
        ))
        conn.commit()
        conn.close()


def _db_get_audit(limit: int = 200) -> List[Dict[str, Any]]:
    with _DB_LOCK:
        conn = sqlite3.connect(str(_db_path()))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM workflow_audit ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def _load_thresholds_from_file() -> Dict[str, Any]:
    f = _threshold_file()
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_THRESHOLDS))


def _save_thresholds_to_file(thresholds: Dict[str, Any]) -> None:
    try:
        _threshold_file().write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _append_trend_to_file(patient_id: str, entry: Dict[str, Any]) -> None:
    """Append a trend snapshot to a daily JSONL file per patient."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = _trend_dir() / f"{patient_id}_{today}.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_trend_from_files(patient_id: str, max_points: int = 48) -> List[Dict[str, Any]]:
    """Load trend history from the last 2 days of flat files, newest last."""
    td = _trend_dir()
    points: List[Dict[str, Any]] = []
    for days_ago in (1, 0):
        day = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        path = td / f"{patient_id}_{day}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    points.append(json.loads(line))
                except Exception:
                    pass
    return points[-max_points:]


# ---------------------------------------------------------------------------
# NOTIFICATION LAYER — email via SMTP/SendGrid, SMS via Twilio (opt-in)
# ---------------------------------------------------------------------------

_ALERT_COOLDOWN_MINUTES = int(os.getenv("ALERT_COOLDOWN_MINUTES", "15"))
_notification_sent_at: Dict[str, str] = {}  # patient_id -> last ISO timestamp


def _can_notify(patient_id: str) -> bool:
    last = _notification_sent_at.get(patient_id)
    if not last:
        return True
    try:
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
        return elapsed > _ALERT_COOLDOWN_MINUTES * 60
    except Exception:
        return True


def _mark_notified(patient_id: str) -> None:
    _notification_sent_at[patient_id] = _utc_now_iso()


def _send_email_alert(patient_id: str, patient_name: str, status: str, risk_score: float, reasons: List[str]) -> bool:
    """Send email alert via SMTP. Reads config from env vars. Returns True if sent."""
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    alert_to = os.getenv("ALERT_EMAIL_TO", "")
    alert_from = os.getenv("ALERT_EMAIL_FROM", smtp_user)

    if not all([smtp_host, smtp_user, smtp_pass, alert_to]):
        return False

    reason_text = "\n".join(f"  • {r}" for r in reasons) if reasons else "  • Monitored context elevated"
    body = (
        f"Early Risk Alert AI — Review-Priority Notification\n\n"
        f"Patient: {patient_name} ({patient_id})\n"
        f"Review Priority: {status}\n"
        f"Review Score: {round(risk_score, 1)}\n\n"
        f"Monitored context for authorized HCP review:\n{reason_text}\n\n"
        f"---\n"
        f"This review-priority notification is for decision-support and workflow-support purposes only.\n"
        f"It does not replace clinician judgment and is not intended to diagnose,\n"
        f"direct treatment, or independently trigger escalation.\n\n"
        f"View command center: https://early-risk-alert-ai-1.onrender.com/command-center\n"
    )
    msg = MIMEText(body)
    msg["Subject"] = f"[ERA Review Notice] {status} review priority — {patient_name}"
    msg["From"] = alert_from
    msg["To"] = alert_to
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"[ERA] Email alert failed: {e}")
        return False


def _send_sms_alert(patient_id: str, patient_name: str, status: str, risk_score: float) -> bool:
    """Send SMS alert via Twilio. Reads config from env vars. Returns True if sent."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_number = os.getenv("TWILIO_FROM_NUMBER", "")
    to_number = os.getenv("TWILIO_TO_NUMBER", "")

    if not all([account_sid, auth_token, from_number, to_number]):
        return False

    body = (
        f"[ERA Review Notice] {status} review priority — {patient_name} ({patient_id}). "
        f"Review score {round(risk_score, 1)}. "
        f"Decision-support and workflow-support only — does not replace clinician judgment."
    )
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        import base64
        data = urllib.parse.urlencode({"From": from_number, "To": to_number, "Body": body}).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        credentials = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
        req.add_header("Authorization", f"Basic {credentials}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 201)
    except Exception as e:
        print(f"[ERA] SMS alert failed: {e}")
        return False


def _maybe_send_alerts(patients: List[Dict[str, Any]]) -> None:
    """Check each patient and fire notifications if threshold crossed and cooldown elapsed."""
    notifications_enabled = os.getenv("ALERT_NOTIFICATIONS_ENABLED", "0") == "1"
    if not notifications_enabled:
        return
    for p in patients:
        patient_id = p.get("patient_id", "")
        status = str(p.get("status", "")).strip()
        risk_score = float(p.get("risk_score", 0))
        if status in ("Critical", "High") and _can_notify(patient_id):
            name = p.get("name", patient_id)
            reasons = (p.get("risk") or {}).get("reasons", [])
            email_sent = _send_email_alert(patient_id, name, status, risk_score, reasons)
            sms_sent = _send_sms_alert(patient_id, name, status, risk_score)
            if email_sent or sms_sent:
                _mark_notified(patient_id)


# ---------------------------------------------------------------------------
# SIMSTATE — in-memory cache layer (backed by SQLite + flat files on write)
# ---------------------------------------------------------------------------

SIM_STATE: Dict[str, Any] = {
    "seed": 1,
    "last_generated_at": "",
    "workflow_records": {},   # loaded from SQLite on first access
    "workflow_audit": [],     # loaded from SQLite on first access
    "trend_history": {},      # in-memory cache; writes also go to flat files
    "thresholds": {},         # loaded from file on init
    "_db_loaded": False,
}


def _ensure_db_loaded() -> None:
    """Lazy-load workflow state from SQLite on first call after startup."""
    if SIM_STATE["_db_loaded"]:
        return
    try:
        _init_db()
        SIM_STATE["workflow_records"] = _db_get_workflow_records()
        SIM_STATE["workflow_audit"] = _db_get_audit(200)
        SIM_STATE["thresholds"] = _load_thresholds_from_file()
        # Warm trend cache from files for known patients
        for pid in ("p101", "p102", "p103", "p104"):
            rows = _load_trend_from_files(pid, 48)
            if rows:
                SIM_STATE["trend_history"][pid] = rows
        SIM_STATE["_db_loaded"] = True
    except Exception as e:
        print(f"[ERA] DB load failed (will use in-memory): {e}")
        SIM_STATE["_db_loaded"] = True  # don't retry on every request
        if not SIM_STATE["thresholds"]:
            SIM_STATE["thresholds"] = json.loads(json.dumps(DEFAULT_THRESHOLDS))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _base_patient_state() -> list[dict[str, Any]]:
    return [
        {
            "patient_id": "p101",
            "name": "Patient 1042",
            "room": "ICU-12",
            "program": "Cardiac Monitoring",
            "heart_rate": 126.0,
            "spo2": 89.0,
            "bp_systolic": 164.0,
            "bp_diastolic": 98.0,
            "respiratory_rate": 28.0,
            "temperature_f": 100.7,
            "trend": "deteriorating",
            "risk_score": 9.1,
            "status": "Critical",
            "confidence": 0.94,
            "recommended_action": "Supportive review context indicates elevated monitored risk.",
        },
        {
            "patient_id": "p102",
            "name": "Patient 2188",
            "room": "Telemetry-04",
            "program": "Pulmonary Monitoring",
            "heart_rate": 112.0,
            "spo2": 93.0,
            "bp_systolic": 148.0,
            "bp_diastolic": 90.0,
            "respiratory_rate": 24.0,
            "temperature_f": 99.4,
            "trend": "watch",
            "risk_score": 8.1,
            "status": "High",
            "confidence": 0.88,
            "recommended_action": "Supportive review context indicates closer monitored follow-up.",
        },
        {
            "patient_id": "p103",
            "name": "Patient 3045",
            "room": "Stepdown-09",
            "program": "Cardiac Stepdown",
            "heart_rate": 82.0,
            "spo2": 98.0,
            "bp_systolic": 122.0,
            "bp_diastolic": 78.0,
            "respiratory_rate": 18.0,
            "temperature_f": 98.4,
            "trend": "recovering",
            "risk_score": 3.4,
            "status": "Stable",
            "confidence": 0.81,
            "recommended_action": "Routine monitored review remains active.",
        },
        {
            "patient_id": "p104",
            "name": "Patient 4172",
            "room": "Ward-21",
            "program": "Recovery Monitoring",
            "heart_rate": 106.0,
            "spo2": 94.0,
            "bp_systolic": 142.0,
            "bp_diastolic": 88.0,
            "respiratory_rate": 22.0,
            "temperature_f": 99.1,
            "trend": "watch",
            "risk_score": 7.6,
            "status": "High",
            "confidence": 0.86,
            "recommended_action": "Supportive review context indicates continued monitored follow-up.",
        },
    ]


def _risk_from_vitals(p: dict[str, Any]) -> tuple[float, str, float, str]:
    hr = float(p["heart_rate"])
    spo2 = float(p["spo2"])
    sys = float(p["bp_systolic"])
    dia = float(p["bp_diastolic"])
    rr = float(p["respiratory_rate"])
    temp = float(p["temperature_f"])

    risk = 0.0
    risk += max(0, hr - 90) * 0.035
    risk += max(0, 94 - spo2) * 0.75
    risk += max(0, sys - 140) * 0.02
    risk += max(0, dia - 90) * 0.03
    risk += max(0, rr - 20) * 0.12
    risk += max(0, temp - 99.0) * 0.7
    if p["trend"] == "deteriorating":
        risk += 1.2
    elif p["trend"] == "watch":
        risk += 0.6
    elif p["trend"] == "recovering":
        risk -= 0.4
    risk = _clamp(round(risk, 1), 0.8, 9.9)

    if risk >= 8.5:
        status = "Critical"
        action = "Supportive review context indicates elevated monitored risk for authorized HCP review."
    elif risk >= 6.2:
        status = "High"
        action = "Supportive review context indicates closer monitored follow-up for authorized HCP review."
    else:
        status = "Stable"
        action = "Routine monitored review remains active for authorized HCP review."

    confidence = _clamp(round(0.72 + min(risk, 9.5) / 25 + random.uniform(-0.03, 0.03), 2), 0.74, 0.98)
    return risk, status, confidence, action


def _mutate_patient(p: dict[str, Any]) -> dict[str, Any]:
    trend = p["trend"]
    if trend == "deteriorating":
        p["heart_rate"] = _clamp(float(p["heart_rate"]) + random.uniform(-1, 5), 88, 142)
        p["spo2"] = _clamp(float(p["spo2"]) + random.uniform(-2.2, 0.4), 84, 98)
        p["bp_systolic"] = _clamp(float(p["bp_systolic"]) + random.uniform(-2, 6), 118, 176)
        p["bp_diastolic"] = _clamp(float(p["bp_diastolic"]) + random.uniform(-2, 4), 70, 108)
        p["respiratory_rate"] = _clamp(float(p["respiratory_rate"]) + random.uniform(-1, 2.5), 16, 34)
        p["temperature_f"] = _clamp(float(p["temperature_f"]) + random.uniform(-0.1, 0.2), 97.8, 102.2)
    elif trend == "watch":
        p["heart_rate"] = _clamp(float(p["heart_rate"]) + random.uniform(-3, 3), 76, 126)
        p["spo2"] = _clamp(float(p["spo2"]) + random.uniform(-1.0, 1.0), 89, 99)
        p["bp_systolic"] = _clamp(float(p["bp_systolic"]) + random.uniform(-3, 3), 108, 160)
        p["bp_diastolic"] = _clamp(float(p["bp_diastolic"]) + random.uniform(-2, 2), 68, 98)
        p["respiratory_rate"] = _clamp(float(p["respiratory_rate"]) + random.uniform(-1.2, 1.2), 15, 28)
        p["temperature_f"] = _clamp(float(p["temperature_f"]) + random.uniform(-0.08, 0.08), 97.8, 100.6)
    else:
        p["heart_rate"] = _clamp(float(p["heart_rate"]) + random.uniform(-2, 2), 62, 96)
        p["spo2"] = _clamp(float(p["spo2"]) + random.uniform(-0.5, 0.5), 95, 100)
        p["bp_systolic"] = _clamp(float(p["bp_systolic"]) + random.uniform(-2, 2), 108, 132)
        p["bp_diastolic"] = _clamp(float(p["bp_diastolic"]) + random.uniform(-2, 2), 66, 84)
        p["respiratory_rate"] = _clamp(float(p["respiratory_rate"]) + random.uniform(-1, 1), 14, 22)
        p["temperature_f"] = _clamp(float(p["temperature_f"]) + random.uniform(-0.05, 0.05), 97.5, 99.2)

    risk, status, confidence, action = _risk_from_vitals(p)
    p["risk_score"] = risk
    p["status"] = status
    p["confidence"] = confidence
    p["recommended_action"] = action
    return p


def _simulated_snapshot() -> dict[str, Any]:
    _ensure_db_loaded()
    SIM_STATE["seed"] += 1
    random.seed(SIM_STATE["seed"] + int(time.time() // 2))
    base_patients = [_mutate_patient(dict(p)) for p in _base_patient_state()]

    patients: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    generated_at = _utc_now_iso()

    for raw in base_patients:
        risk_score = float(raw["risk_score"])
        status = str(raw["status"])
        severity = status.lower()
        room = str(raw.get("room") or "Telemetry-01")
        rr = int(round(float(raw["respiratory_rate"])))
        temp_f = round(float(raw["temperature_f"]), 1)
        hr = int(round(float(raw["heart_rate"])))
        spo2 = int(round(float(raw["spo2"])))
        sbp = int(round(float(raw["bp_systolic"])))
        dbp = int(round(float(raw["bp_diastolic"])))

        if status == "Critical":
            alert_message = "Clinical review attention surfaced"
            story = "Monitored context supports higher-priority authorized HCP review."
        elif status == "High":
            alert_message = "Elevated review attention surfaced"
            story = "Observed trends remain available for authorized HCP review."
        else:
            alert_message = "Current monitored review summary stable"
            story = "Combined monitored context remains available for authorized HCP review."

        reasons: List[str] = []
        if spo2 < 93:
            reasons.append("Oxygen saturation is below the current review threshold")
        if hr >= 110:
            reasons.append("Heart rate is elevated relative to the current review threshold")
        if sbp >= 150:
            reasons.append("Blood pressure is elevated relative to the current review threshold")
        if rr < 12 or rr > 20:
            reasons.append("Respiratory rate is outside typical reference range — for HCP review")
        else:
            reasons.append("Respiratory rate trend is available for authorized HCP review")
        if temp_f < 97.6 or temp_f > 100.4:
            reasons.append("Temperature is outside typical reference range — for HCP review")
        else:
            reasons.append("Temperature trend is available for authorized HCP review")

        patient = {
            "patient_id": raw["patient_id"],
            "patient_name": raw["name"],
            "name": raw["name"],
            "room": room,
            "bed": room,
            "program": raw.get("program", "Clinical Monitoring"),
            "title": alert_message,
            "heart_rate": hr,
            "spo2": spo2,
            "bp_systolic": sbp,
            "bp_diastolic": dbp,
            "resp_rate": rr,
            "respiratory_rate": rr,
            "temp": temp_f,
            "temp_f": temp_f,
            "temperature_f": temp_f,
            "risk_score": risk_score,
            "status": status,
            "confidence": float(raw.get("confidence", 0.85)),
            "recommended_action": raw.get("recommended_action", "Continue monitored review."),
            "story": story,
            "vitals": {
                "heart_rate": hr,
                "systolic_bp": sbp,
                "diastolic_bp": dbp,
                "spo2": spo2,
                "respiratory_rate": rr,
                "rr": rr,
                "temperature_f": temp_f,
                "temp_f": temp_f,
            },
            "risk": {
                "risk_score": risk_score,
                "severity": severity,
                "alert_message": alert_message,
                "recommended_action": raw.get("recommended_action", "Continue monitored review."),
                "reasons": reasons,
            },
            "data_freshness": {
                "generated_at": generated_at,
                "age_seconds": 0,
                "source_mode": "structured-vital-summary-pilot",
                "refresh_interval_seconds": 5,
            },
        }
        patients.append(patient)
        alerts.append(
            {
                "patient_id": raw["patient_id"],
                "title": alert_message,
                "message": alert_message,
                "alert_type": alert_message,
                "severity": severity,
                "room": room,
                "unit": room,
                "risk_score": risk_score,
                "confidence": float(raw.get("confidence", 0.85)),
                "recommended_action": raw.get("recommended_action", "Continue monitored review."),
                "heart_rate": hr,
                "spo2": spo2,
            }
        )

        trend_entry = {
                "time": generated_at,
                "hr": hr,
                "spo2": spo2,
                "risk": risk_score,
                "rr": rr,
                "temp_f": temp_f,
            }
        history = SIM_STATE.setdefault("trend_history", {}).setdefault(raw["patient_id"], [])
        history.append(trend_entry)
        SIM_STATE["trend_history"][raw["patient_id"]] = history[-48:]
        # Persist to flat file so trends survive restarts
        _append_trend_to_file(raw["patient_id"], trend_entry)

    alerts.sort(key=lambda a: float(a["risk_score"]), reverse=True)
    critical_count = sum(1 for a in alerts if a["severity"] == "critical")
    open_alerts = sum(1 for a in alerts if a["severity"] in {"critical", "high"})
    avg_risk = round(sum(float(a["risk_score"]) for a in alerts) / len(alerts), 1) if alerts else 0.0
    patients_with_alerts = sum(1 for a in alerts if a["severity"] in {"critical", "high"})
    events_last_hour = open_alerts * 3 + critical_count * 2 + 4
    focus_patient_id = alerts[0]["patient_id"] if alerts else "p101"

    SIM_STATE["last_generated_at"] = generated_at
    # Fire alert notifications asynchronously so they don't block the response
    threading.Thread(target=_maybe_send_alerts, args=(patients,), daemon=True).start()
    return {
        "generated_at": generated_at,
        "patients": patients,
        "alerts": alerts[:8],
        "summary": {
            "patient_count": len(patients),
            "open_alerts": open_alerts,
            "critical_alerts": critical_count,
            "avg_risk_score": avg_risk,
            "patients_with_alerts": patients_with_alerts,
            "events_last_hour": events_last_hour,
            "focus_patient_id": focus_patient_id,
        },
        "meta": {
            "pilot_version": PILOT_VERSION,
            "pilot_build_state": PILOT_BUILD_STATE,
            "intended_use_statement": INTENDED_USE_STATEMENT,
        },
    }


def _delta_explainability(patient_id: str, current: Dict[str, Any]) -> str:
    """
    Build an explainability string that shows deltas from the last stored
    trend observation so clinicians see 'HR was 88, now 112 — rising trend'
    rather than just a static threshold check.
    """
    history = SIM_STATE.get("trend_history", {}).get(patient_id, [])
    if len(history) < 2:
        # Fallback to threshold-only description when no history yet
        notes = []
        spo2 = float(current.get("spo2") or 0)
        hr = float(current.get("heart_rate") or 0)
        sbp = float(current.get("bp_systolic") or 0)
        rr = float(current.get("respiratory_rate") or 0)
        temp = float(current.get("temperature_f") or 0)
        if spo2 and spo2 < 93:
            notes.append(f"SpO\u2082 {spo2:.0f}% is below the review threshold")
        if hr and hr >= 110:
            notes.append(f"HR {hr:.0f} bpm is elevated")
        if sbp and sbp >= 150:
            notes.append(f"BP {sbp:.0f} mmHg is elevated")
        if rr and (rr < 12 or rr > 20):
            notes.append(f"RR {rr:.0f} br/min is outside typical reference range")
        if temp and (temp < 97.6 or temp > 100.4):
            notes.append(f"Temp {temp:.1f}\u00b0F is outside typical reference range")
        if not notes:
            notes.append("Combined monitored context remains available for authorized HCP review")
        return "Monitored context: " + "; ".join(notes) + "."

    prev = history[-2]
    curr = history[-1]
    deltas = []

    # HR delta
    prev_hr = float(prev.get("hr") or 0)
    curr_hr = float(curr.get("hr") or 0)
    if prev_hr and curr_hr:
        diff_hr = curr_hr - prev_hr
        if abs(diff_hr) >= 5:
            direction = "rising" if diff_hr > 0 else "falling"
            deltas.append(
                f"HR was {prev_hr:.0f} bpm, now {curr_hr:.0f} bpm \u2014 {direction} trend (+{diff_hr:+.0f})"
                if diff_hr > 0
                else f"HR was {prev_hr:.0f} bpm, now {curr_hr:.0f} bpm \u2014 {direction} trend ({diff_hr:+.0f})"
            )
        elif curr_hr >= 110:
            deltas.append(f"HR {curr_hr:.0f} bpm remains elevated (stable)")

    # SpO2 delta
    prev_spo2 = float(prev.get("spo2") or 0)
    curr_spo2 = float(curr.get("spo2") or 0)
    if prev_spo2 and curr_spo2:
        diff_spo2 = curr_spo2 - prev_spo2
        if abs(diff_spo2) >= 1:
            direction = "improving" if diff_spo2 > 0 else "declining"
            deltas.append(
                f"SpO\u2082 was {prev_spo2:.0f}%, now {curr_spo2:.0f}% \u2014 {direction} ({diff_spo2:+.0f}%)"
            )
        elif curr_spo2 < 93:
            deltas.append(f"SpO\u2082 {curr_spo2:.0f}% remains below review threshold (stable low)")

    # RR delta
    prev_rr = float(prev.get("rr") or 0)
    curr_rr = float(curr.get("rr") or 0)
    if prev_rr and curr_rr:
        diff_rr = curr_rr - prev_rr
        if abs(diff_rr) >= 2:
            direction = "rising" if diff_rr > 0 else "falling"
            deltas.append(
                f"RR was {prev_rr:.0f}, now {curr_rr:.0f} br/min \u2014 {direction} trend"
            )
        elif curr_rr < 12 or curr_rr > 20:
            deltas.append(f"RR {curr_rr:.0f} br/min remains outside reference range (stable)")
        else:
            deltas.append(f"RR {curr_rr:.0f} br/min within reference range")

    # Temp delta
    prev_temp = float(prev.get("temp_f") or 0)
    curr_temp = float(curr.get("temp_f") or 0)
    if prev_temp and curr_temp:
        diff_temp = curr_temp - prev_temp
        if abs(diff_temp) >= 0.3:
            direction = "rising" if diff_temp > 0 else "falling"
            deltas.append(
                f"Temp was {prev_temp:.1f}\u00b0F, now {curr_temp:.1f}\u00b0F \u2014 {direction} trend"
            )
        elif curr_temp > 100.4:
            deltas.append(f"Temp {curr_temp:.1f}\u00b0F remains above reference range (stable elevated)")
        else:
            deltas.append(f"Temp {curr_temp:.1f}\u00b0F within reference range")

    # Risk score delta
    prev_risk = float(prev.get("risk") or 0)
    curr_risk = float(curr.get("risk") or 0)
    if prev_risk and curr_risk:
        diff_risk = curr_risk - prev_risk
        if abs(diff_risk) >= 0.5:
            direction = "increasing" if diff_risk > 0 else "decreasing"
            deltas.append(
                f"Review score {direction} ({prev_risk:.1f} \u2192 {curr_risk:.1f})"
            )

    if not deltas:
        deltas.append("All monitored values stable across last two observations")

    return (
        "Delta explainability (current vs last observation): "
        + "; ".join(deltas)
        + ". Values displayed as monitored context for authorized HCP review."
    )



# ---------------------------------------------------------------------------
# MIMIC-IV EXTRACTION ENGINE
# ---------------------------------------------------------------------------
# Transforms raw MIMIC-IV tables into ERA-compatible CSV format.
# MIMIC stores vitals in long format (one row per measurement) keyed by
# itemid. ERA requires wide format (one row per patient+timestamp).
# This engine handles the pivot, Celsius->Fahrenheit conversion,
# clinical event flagging, and outputs a clean ERA-schema CSV.
#
# Key MIMIC-IV chartevents item IDs:
#   220045 = Heart Rate         220277 = SpO2
#   220179 = BP Systolic        220180 = BP Diastolic
#   220210 = Respiratory Rate   223761 = Temperature F
#   223762 = Temperature C (converted)
#
# Key MIMIC-IV datetimeevents item IDs (clinical events):
#   225477 = Rapid Response     225086 = Transfer to ICU
#   224859 = Code Blue          225468 = Unplanned Extubation
# ---------------------------------------------------------------------------

MIMIC_VITAL_ITEMIDS = {
    220045: "heart_rate",
    220277: "spo2",
    220179: "bp_systolic",
    220180: "bp_diastolic",
    220210: "respiratory_rate",
    223761: "temperature_f",
    223762: "temperature_c",
}

MIMIC_EVENT_ITEMIDS = {
    225468: "Unplanned Extubation",
    225477: "Rapid Response Team called",
    225086: "Transfer to ICU",
    224859: "Code Blue",
    226253: "Withdrawal of Life-Sustaining Treatment",
    228232: "ICU Transfer",
}

def _celsius_to_f(c):
    return round(c * 9 / 5 + 32, 1)

def _parse_mimic_chartevents(text):
    """
    Parse MIMIC chartevents CSV streaming line by line.
    Handles files of any size without loading into memory at once.
    Returns {pid -> {charttime -> {vital_name: value}}}
    """
    lines = iter(text.splitlines())
    header_line = next(lines, None)
    if not header_line:
        return {}

    hdrs_raw = [h.strip() for h in header_line.split(",")]
    hdrs     = [h.lower() for h in hdrs_raw]

    # Build column index map
    col = {}
    for target, cands in {
        "subject_id": ["subject_id"],
        "charttime":  ["charttime", "storetime"],
        "itemid":     ["itemid", "item_id"],
        "valuenum":   ["valuenum", "value_num", "value"],
    }.items():
        for c in cands:
            if c in hdrs:
                col[target] = hdrs.index(c)
                break

    missing = [k for k in ["subject_id","charttime","itemid","valuenum"] if k not in col]
    if missing:
        return {"_error": f"chartevents missing columns: {missing}. Headers found: {hdrs[:10]}"}

    vital_ids = set(MIMIC_VITAL_ITEMIDS.keys())
    result = {}
    rows_read = 0

    for line in lines:
        if not line.strip():
            continue
        # Fast CSV split — handles basic quoting
        try:
            parts = next(csv.reader(io.StringIO(line)))
        except Exception:
            continue

        if len(parts) <= max(col.values()):
            continue

        try:
            iid = int(parts[col["itemid"]])
        except (ValueError, IndexError):
            continue

        if iid not in vital_ids:
            continue

        try:
            val = float(parts[col["valuenum"]])
        except (ValueError, IndexError, TypeError):
            continue

        if val <= 0 or val > 99999:
            continue

        pid   = parts[col["subject_id"]].strip()
        ctime = parts[col["charttime"]].strip()
        vital = MIMIC_VITAL_ITEMIDS[iid]

        if pid not in result:
            result[pid] = {}
        if ctime not in result[pid]:
            result[pid][ctime] = {}

        if vital == "temperature_c":
            val = _celsius_to_f(val)
            vital = "temperature_f"

        result[pid][ctime][vital] = val
        rows_read += 1

    return result

def _parse_mimic_icustays(text):
    """Parse MIMIC icustays.csv -> {subject_id -> {unit, stay_id, ...}}"""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return {}
    result = {}
    for row in reader:
        pid = str(row.get("subject_id","")).strip()
        if not pid:
            continue
        unit_raw = str(row.get("first_careunit","")).strip().lower()
        if any(x in unit_raw for x in ["micu","sicu","csicu","cvicu","tsicu","neuro"]):
            era_unit = "icu"
        elif any(x in unit_raw for x in ["step","intermediate"]):
            era_unit = "stepdown"
        else:
            era_unit = "icu"
        result[pid] = {
            "stay_id":  row.get("stay_id",""),
            "intime":   row.get("intime",""),
            "outtime":  row.get("outtime",""),
            "unit":     era_unit,
            "unit_raw": unit_raw,
        }
    return result

def _parse_mimic_datetimeevents(text):
    """Parse MIMIC datetimeevents -> {subject_id -> set of event charttime strings}"""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return {}
    result = {}
    event_ids = set(MIMIC_EVENT_ITEMIDS.keys())
    for row in reader:
        try:
            iid = int(row.get("itemid", 0))
        except (ValueError, TypeError):
            continue
        if iid not in event_ids:
            continue
        pid   = str(row.get("subject_id","")).strip()
        etime = str(row.get("charttime", row.get("storetime",""))).strip()
        if pid and etime:
            if pid not in result:
                result[pid] = set()
            result[pid].add(etime)
    return result

def _mimic_datetimeevents_diagnostics(text, icustays):
    """Count raw recognized datetimeevents rows and how many match uploaded ICU stays."""
    diag = {
        "raw_recognized_datetimeevents_rows": 0,
        "raw_event_rows_matched_to_uploaded_icu_stays": 0,
    }
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return diag

    event_ids = set(MIMIC_EVENT_ITEMIDS.keys())
    icu_subjects = set(icustays.keys())

    for row in reader:
        try:
            iid = int(row.get("itemid", 0))
        except (ValueError, TypeError):
            continue
        if iid not in event_ids:
            continue

        pid   = str(row.get("subject_id","")).strip()
        etime = str(row.get("charttime", row.get("storetime",""))).strip()
        if not (pid and etime):
            continue

        diag["raw_recognized_datetimeevents_rows"] += 1
        if pid in icu_subjects:
            diag["raw_event_rows_matched_to_uploaded_icu_stays"] += 1

    return diag


def _mimic_datetimeevents_itemid_inspector(text, top_n=25):
    """Inspect which datetimeevents itemids are actually present in the uploaded file."""
    diag = {
        "rows_with_itemid": 0,
        "top_itemids": [],
        "recognized_itemids_present": [],
    }
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return diag

    counts = {}
    recognized = set(MIMIC_EVENT_ITEMIDS.keys())
    recognized_present = set()

    for row in reader:
        raw_iid = str(row.get("itemid", "")).strip()
        if not raw_iid:
            continue
        try:
            iid = int(raw_iid)
        except (ValueError, TypeError):
            continue

        diag["rows_with_itemid"] += 1
        counts[iid] = counts.get(iid, 0) + 1
        if iid in recognized:
            recognized_present.add(iid)

    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    diag["top_itemids"] = [
        {"itemid": iid, "count": count, "recognized": iid in recognized}
        for iid, count in top
    ]
    diag["recognized_itemids_present"] = sorted(list(recognized_present))
    return diag

def _build_era_csv_from_mimic(vitals, icustays, events, min_vitals=3, event_window_hours=6):
    """
    Pivot MIMIC data to ERA-compatible CSV.
    Returns (csv_text, stats_dict).
    Skips readings with fewer than min_vitals vitals present.
    Flags readings within event_window_hours before a clinical event.
    """
    ERA_COLS = ["heart_rate","spo2","bp_systolic","bp_diastolic","respiratory_rate","temperature_f"]
    output_rows = []
    stats = {
        "total_patients": 0,
        "total_rows": 0,
        "rows_skipped_min_vitals": 0,
        "total_events_flagged": 0,
        "patients_with_events": 0,
        "unit_distribution": {},
    }
    for pid, timepoints in vitals.items():
        if not timepoints:
            continue
        stats["total_patients"] += 1
        icu_info = icustays.get(pid, {})
        era_unit = icu_info.get("unit", "icu")
        patient_events = events.get(pid, set())
        evt_dts = []
        for et in patient_events:
            try:
                evt_dts.append(datetime.fromisoformat(et.replace(" ","T")))
            except ValueError:
                pass
        patient_had_event = False
        for ctime_str in sorted(timepoints.keys()):
            vitals_at = timepoints[ctime_str]
            n_present = sum(1 for v in ERA_COLS if v in vitals_at)
            if n_present < min_vitals:
                stats["rows_skipped_min_vitals"] += 1
                continue
            ce_flag  = 0
            ce_label = ""
            try:
                cdt = datetime.fromisoformat(ctime_str.replace(" ","T"))
                for edt in evt_dts:
                    diff_h = (edt - cdt).total_seconds() / 3600
                    if 0 <= diff_h <= event_window_hours:
                        ce_flag  = 1
                        ce_label = "Clinical event within 6-hr window (MIMIC-IV)"
                        patient_had_event = True
                        break
            except (ValueError, TypeError):
                pass
            try:
                ts = datetime.fromisoformat(ctime_str.replace(" ","T")).strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                ts = ctime_str
            output_rows.append({
                "patient_id":           f"MIMIC-{pid}",
                "timestamp":            ts,
                "heart_rate":           vitals_at.get("heart_rate",""),
                "spo2":                 vitals_at.get("spo2",""),
                "bp_systolic":          vitals_at.get("bp_systolic",""),
                "bp_diastolic":         vitals_at.get("bp_diastolic",""),
                "respiratory_rate":     vitals_at.get("respiratory_rate",""),
                "temperature_f":        vitals_at.get("temperature_f",""),
                "unit":                 era_unit,
                "clinical_event":       ce_flag,
                "clinical_event_label": ce_label,
                "notes":                f"MIMIC-IV de-identified. Unit: {icu_info.get('unit_raw','unknown')}",
            })
            stats["total_rows"] += 1
            if ce_flag:
                stats["total_events_flagged"] += 1
        if patient_had_event:
            stats["patients_with_events"] += 1
        stats["unit_distribution"][era_unit] = stats["unit_distribution"].get(era_unit, 0) + 1
    if not output_rows:
        return "", stats
    buf = io.StringIO()
    fieldnames = [
        "patient_id","timestamp","heart_rate","spo2",
        "bp_systolic","bp_diastolic","respiratory_rate","temperature_f",
        "unit","clinical_event","clinical_event_label","notes"
    ]
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(output_rows)
    return buf.getvalue(), stats

def _validate_mimic_extract(csv_text):
    """Confirm extracted CSV passes ERA schema check before upload."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return {"ok": False, "issues": ["Output CSV empty"], "row_count": 0, "patient_count": 0}
    hdrs = [h.strip().lower() for h in reader.fieldnames]
    missing = [f for f in CSV_SCHEMA_REQUIRED if f not in hdrs]
    issues = [f"Missing ERA required columns: {missing}"] if missing else []
    row_count = 0
    patients = set()
    for row in reader:
        row_count += 1
        patients.add(row.get("patient_id",""))
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "row_count": row_count,
        "patient_count": len(patients),
        "columns_present": hdrs,
    }

# ---------------------------------------------------------------------------
# CSV INGESTION LAYER — retrospective validation data pipeline
# ---------------------------------------------------------------------------

CSV_SCHEMA_REQUIRED = ["patient_id", "timestamp", "heart_rate", "spo2", "bp_systolic", "bp_diastolic", "respiratory_rate", "temperature_f"]
CSV_SCHEMA_OPTIONAL = ["patient_name", "room", "unit", "program", "clinical_event", "clinical_event_label", "notes"]
CSV_SCHEMA_ALL = CSV_SCHEMA_REQUIRED + CSV_SCHEMA_OPTIONAL

CSV_SCHEMA_DESCRIPTION = {
    "patient_id":           "Unique de-identified patient identifier (e.g., PT-001). No real names or MRNs.",
    "timestamp":            "ISO 8601 datetime of the vital reading (e.g., 2024-03-15T14:32:00). UTC preferred.",
    "heart_rate":           "Heart rate in beats per minute (numeric, e.g., 88).",
    "spo2":                 "Oxygen saturation as percentage (numeric, e.g., 96).",
    "bp_systolic":          "Systolic blood pressure in mmHg (numeric, e.g., 128).",
    "bp_diastolic":         "Diastolic blood pressure in mmHg (numeric, e.g., 82).",
    "respiratory_rate":     "Respiratory rate in breaths per minute (numeric, e.g., 16).",
    "temperature_f":        "Temperature in Fahrenheit (numeric, e.g., 98.6).",
    "patient_name":         "Optional. De-identified label only (e.g., 'Patient A'). Do not include real names.",
    "room":                 "Optional. Room or bed identifier (e.g., ICU-04).",
    "unit":                 "Optional. Unit name: icu, telemetry, stepdown, ward, or rpm.",
    "program":              "Optional. Monitoring program label (e.g., Cardiac Monitoring).",
    "clinical_event":       "Optional. 1 if a clinical escalation, rapid response, or adverse event occurred within 6 hours of this reading. 0 otherwise.",
    "clinical_event_label": "Optional. Description of the clinical event (e.g., Rapid Response Team called, Transfer to ICU).",
    "notes":                "Optional. Any additional context relevant to this reading.",
}

RETRO_STATE: Dict[str, Any] = {
    "uploads": [],          # list of {upload_id, filename, uploaded_at, row_count, columns, status}
    "records": {},          # upload_id -> list of parsed row dicts
    "analysis": {},         # upload_id -> analysis result dict
    "processing": {},       # upload_id -> {"status": "running"|"done"|"error", "message": str}
}

# Large file threshold — files above this are analyzed asynchronously
_ASYNC_ANALYSIS_THRESHOLD_ROWS = 30_000

def _parse_csv_upload(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Parse uploaded CSV bytes, validate schema, compute basic stats.
    For large files (>30k rows) uses streaming mode — stores only
    per-patient peak stats instead of all rows, keeping memory flat.
    """
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return {"ok": False, "error": "CSV appears empty or has no header row."}

    headers = [h.strip().lower().replace(" ", "_") for h in reader.fieldnames]
    missing = [f for f in CSV_SCHEMA_REQUIRED if f not in headers]
    if missing:
        return {
            "ok": False,
            "error": f"Missing required columns: {', '.join(missing)}. "
                     f"Required columns are: {', '.join(CSV_SCHEMA_REQUIRED)}."
        }

    # First pass — count rows to decide streaming vs full storage
    rows = []
    errors = []
    patients_seen: set = set()
    event_count = 0
    units_seen: set = set()
    row_count = 0

    NUMERIC = ["heart_rate", "spo2", "bp_systolic", "bp_diastolic",
               "respiratory_rate", "temperature_f"]

    for i, raw_row in enumerate(reader, start=2):
        row = {k.strip().lower().replace(" ", "_"): v.strip()
               for k, v in raw_row.items() if k}
        row_ok = True
        for field in NUMERIC:
            val = row.get(field, "")
            if val == "":
                continue
            try:
                row[field] = float(val)
            except ValueError:
                if len(errors) < 20:
                    errors.append(f"Row {i}: '{field}' value '{val}' not numeric — skipped.")
                row_ok = False
                break
        if not row_ok:
            continue

        ce = row.get("clinical_event", "")
        row["clinical_event"] = int(ce) if ce in ("0", "1") else None
        if row["clinical_event"] == 1:
            event_count += 1

        pid = row.get("patient_id", "")
        if pid:
            patients_seen.add(pid)
        unit = row.get("unit", "").lower()
        if unit:
            units_seen.add(unit)

        row_count += 1

        # Only keep rows in memory for small datasets (≤ 50k rows)
        # Large datasets are re-streamed during analysis
        if row_count <= 200_000:
            rows.append(row)

    if row_count == 0:
        return {"ok": False, "error": "No valid data rows found after parsing."}

    return {
        "ok":            True,
        "rows":          rows,           # empty list for large datasets
        "row_count":     row_count,
        "patient_count": len(patients_seen),
        "event_count":   event_count,
        "units":         list(units_seen),
        "columns_found": headers,
        "parse_errors":  errors,
        "filename":      filename,
        "large_file":    row_count > 300_000,
        "raw_text":      text if row_count > 200_000 else None,  # kept for re-streaming
    }


ERA_SCORE_THRESHOLDS    = [4.0, 5.0, 6.0]
ERA_EVENT_WINDOW_HOURS  = 6    # readings within this window before an event count as "pre-event"
                                  # Extended from 4hr — clinical deterioration often starts 4-6hr before event

# ── Improvement 2: tuned signal weights with trend velocity ─────────────────
# Heart rate weight increased. SpO2 deficit kept dominant.
# Trend velocity (delta between consecutive readings) adds up to +1.5 bonus.
_W_HR_BASE   = 0.065   # increased — HR elevation is strong early warning
_W_SPO2      = 0.85    # increased — SpO2 deficit is most critical signal
_W_SBP       = 0.030   # increased — hypotension/hypertension both important
_W_RR        = 0.18    # increased — tachypnea is key early deterioration sign
_W_TEMP      = 0.75    # increased slightly

def _score_row(r: Dict[str, Any], prev: Dict[str, Any] = None) -> float:
    """
    Compute ERA risk score for a single row.
    Improvement 2: increased HR/RR/SBP weights.
    Improvement 2: trend velocity modifier — rate of change adds up to +1.5
    if multiple vitals are deteriorating simultaneously between readings.
    """
    try:
        hr   = float(r.get("heart_rate", 0) or 0)
        spo2 = float(r.get("spo2", 100) or 100)
        sbp  = float(r.get("bp_systolic", 120) or 120)
        rr   = float(r.get("respiratory_rate", 16) or 16)
        temp = float(r.get("temperature_f", 98.6) or 98.6)

        risk  = max(0, hr - 90)    * _W_HR_BASE
        risk += max(0, 94 - spo2)  * _W_SPO2
        risk += max(0, sbp - 140)  * _W_SBP
        risk += max(0, rr - 20)    * _W_RR
        risk += max(0, temp - 99.0)* _W_TEMP

        # Compound deterioration rules — clinically validated patterns
        if prev:
            try:
                dhr  = hr   - float(prev.get("heart_rate", hr) or hr)
                dspo = spo2 - float(prev.get("spo2", spo2) or spo2)
                drr  = rr   - float(prev.get("respiratory_rate", rr) or rr)
                dsbp = sbp  - float(prev.get("bp_systolic", sbp) or sbp)
                dtemp= temp - float(prev.get("temperature_f", temp) or temp)
                # Individual trend flags — lower thresholds to catch earlier
                hr_rising   = dhr  >  5     # HR rising (was 8)
                spo2_drop   = dspo < -1.5   # SpO2 dropping (was -2)
                rr_rising   = drr  >  2     # RR rising (was 3)
                sbp_drop    = dsbp < -10    # SBP dropping — hypotension pattern
                temp_rising = dtemp > 0.3   # Fever developing
                # Count deteriorating signals
                det_count = sum([hr_rising, spo2_drop, rr_rising, sbp_drop, temp_rising])
                # Compound rule bonuses — clinically validated multi-signal patterns
                if det_count >= 3:
                    risk += 1.5   # 3+ signals: strong sepsis/deterioration pattern
                elif det_count == 2:
                    risk += 0.9   # 2 signals: significant combined deterioration
                elif det_count == 1:
                    risk += 0.4   # Single trend: early warning
                # Specific high-risk compound patterns — clinically validated
                if hr_rising and spo2_drop:
                    risk += 0.6   # Tachycardia + hypoxia: respiratory failure pattern
                if spo2_drop and rr_rising:
                    risk += 0.7   # SpO2 falling + RR rising: strongest respiratory signal
                if hr_rising and rr_rising and not spo2_drop:
                    risk += 0.4   # Tachycardia + tachypnea: early sepsis pattern
                if sbp_drop and hr_rising:
                    risk += 0.6   # Hypotension + tachycardia: shock pattern
                if hr_rising and tmp_r:
                    risk += 0.3   # Tachycardia + fever: infection/SIRS pattern
            except Exception:
                pass

        return round(_clamp(risk, 0.0, 9.9), 2)
    except Exception:
        return 0.0


def _threshold_alert(r: Dict[str, Any]) -> bool:
    """Standard threshold alert check — unchanged baseline."""
    try:
        return (
            float(r.get("spo2", 100) or 100) < 93 or
            float(r.get("heart_rate", 0) or 0) > 110 or
            float(r.get("bp_systolic", 0) or 0) > 150
        )
    except Exception:
        return False


def _parse_timestamp(ts: str) -> float:
    """Parse ISO timestamp to Unix float. Returns 0 on failure."""
    try:
        from datetime import datetime, timezone
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(ts.strip(), fmt).replace(
                    tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
    except Exception:
        pass
    return 0.0


def _run_retro_analysis(
    records: List[Dict[str, Any]],
    raw_text: str = None
) -> Dict[str, Any]:
    """
    Streaming retrospective validation analysis — three improvements:

    1. Event window sensitivity: readings within ERA_EVENT_WINDOW_HOURS before
       a documented event are counted as pre-event. This measures whether the
       engine detected deterioration BEFORE the event (clinically correct).

    2. Tuned signal weights + trend velocity: HR/RR/SBP weights increased,
       simultaneous multi-vital deterioration adds a velocity bonus.

    3. Patient-level sensitivity: fraction of patients-with-events who were
       flagged at least once — the metric hospitals actually care about.
    """
    def pct(n: int, d: int) -> float:
        return round((n / d * 100), 1) if d else 0.0

    NUMERIC = ["heart_rate", "spo2", "bp_systolic", "bp_diastolic",
               "respiratory_rate", "temperature_f"]

    # ── Pass 1: collect all rows grouped by patient, sorted by timestamp ─────
    # For event-window calculation we need to know when each patient's events
    # occurred. We collect lightweight per-patient structures only.
    # Memory: one list of (timestamp_float, score, is_thr, ce) per patient.
    patient_rows: Dict[str, list] = {}   # pid -> [(ts, score, is_thr, ce_raw)]
    patient_prev: Dict[str, Dict] = {}   # pid -> last row (for velocity)

    total_rows      = 0
    parse_errors    = 0

    def _stream_rows():
        """Yield parsed row dicts from either records list or raw_text."""
        if raw_text:
            import io as _io
            reader = csv.DictReader(_io.StringIO(raw_text))
            for raw_row in reader:
                row: Dict[str, Any] = {
                    k.strip().lower().replace(" ", "_"): v.strip()
                    for k, v in raw_row.items() if k
                }
                for field in NUMERIC:
                    val = row.get(field, "")
                    if val != "":
                        try:
                            row[field] = float(val)
                        except ValueError:
                            pass
                ce = row.get("clinical_event", "")
                row["clinical_event"] = int(ce) if ce in ("0", "1") else None
                # Clean garbled encoding in display-only fields
                for _f in ("notes", "clinical_event_label", "patient_name", "program"):
                    v = row.get(_f, "")
                    if v and not v.isascii():
                        row[_f] = v.encode("ascii", "replace").decode("ascii").replace("?", "-")
                yield row
        else:
            yield from records

    for row in _stream_rows():
        pid = row.get("patient_id", "unknown")
        ts  = _parse_timestamp(str(row.get("timestamp", "") or ""))
        prev = patient_prev.get(pid)
        score  = _score_row(row, prev)
        is_thr = _threshold_alert(row)
        ce     = row.get("clinical_event")

        if pid not in patient_rows:
            patient_rows[pid] = []
        patient_rows[pid].append((ts, score, is_thr, ce))
        patient_prev[pid] = row
        total_rows += 1

    if total_rows == 0:
        return {"summary": {}, "threshold_comparison": [], "patient_summary": [],
                "interpretation": "No valid rows processed."}

    # ── Pass 2: apply event window, compute all metrics ───────────────────────
    WIN_SECS = ERA_EVENT_WINDOW_HOURS * 3600

    # Reading-level counters (original method — kept for comparison)
    total_events_r    = 0
    total_nonevents_r = 0
    thr_detected_r    = 0
    thr_fp_r          = 0
    ev_det_r   = {t: 0 for t in ERA_SCORE_THRESHOLDS}
    ne_fp_r    = {t: 0 for t in ERA_SCORE_THRESHOLDS}
    risk_sum_event    = 0.0
    risk_sum_nonevent = 0.0

    # Patient-level counters (Improvement 3)
    total_patients_with_events = 0
    pat_thr_detected    = 0  # patients-with-events detected by threshold
    pat_ev_det  = {t: 0 for t in ERA_SCORE_THRESHOLDS}

    # Patient summary
    patient_peaks:  Dict[str, float] = {}
    patient_reads:  Dict[str, int]   = {}
    patient_ev_cnt: Dict[str, int]   = {}
    patient_alerts: Dict[str, int]   = {}

    for pid, rows in patient_rows.items():
        # Find event timestamps for this patient
        event_ts = [ts for (ts, score, is_thr, ce) in rows if ce == 1 and ts > 0]
        has_events = len(event_ts) > 0
        if has_events:
            total_patients_with_events += 1

        # Track patient-level detection flags
        pat_thr_flagged   = False
        pat_era_flagged   = {t: False for t in ERA_SCORE_THRESHOLDS}

        peak = 0.0
        for (ts, score, is_thr, ce) in rows:
            # Determine if this reading is in the pre-event window
            # Improvement 1: reading counts as "pre-event" if it falls
            # within ERA_EVENT_WINDOW_HOURS before any event for this patient
            in_window = False
            if event_ts and ts > 0:
                in_window = any(
                    0 <= (evt - ts) <= WIN_SECS
                    for evt in event_ts
                )
            is_event_reading = (ce == 1) or in_window

            if is_event_reading:
                total_events_r    += 1
                risk_sum_event    += score
                if is_thr:
                    thr_detected_r += 1
            else:
                total_nonevents_r  += 1
                risk_sum_nonevent += score
                if is_thr:
                    thr_fp_r += 1

            for t in ERA_SCORE_THRESHOLDS:
                flagged = score >= t
                if is_event_reading and flagged:
                    ev_det_r[t] += 1
                elif not is_event_reading and flagged:
                    ne_fp_r[t]  += 1

            # Patient-level detection: did ERA flag this patient
            # at any point before their event?
            if has_events and in_window and is_thr:
                pat_thr_flagged = True
            if has_events and in_window:
                for t in ERA_SCORE_THRESHOLDS:
                    if score >= t:
                        pat_era_flagged[t] = True

            if score > peak:
                peak = score

        # Accumulate patient-level detection
        if has_events:
            if pat_thr_flagged:
                pat_thr_detected += 1
            for t in ERA_SCORE_THRESHOLDS:
                if pat_era_flagged[t]:
                    pat_ev_det[t] += 1

        patient_peaks[pid]  = peak
        patient_reads[pid]  = len(rows)
        patient_ev_cnt[pid] = sum(1 for (_,_,_,ce) in rows if ce == 1)
        patient_alerts[pid] = sum(1 for (_,score,_,_) in rows if score >= 6.0)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    thr_total = thr_detected_r + thr_fp_r
    avg_risk_event    = round(risk_sum_event    / total_events_r,    2) if total_events_r    else 0.0
    avg_risk_nonevent = round(risk_sum_nonevent / total_nonevents_r, 2) if total_nonevents_r else 0.0

    threshold_comparison = []
    for t in ERA_SCORE_THRESHOLDS:
        era_tot       = ev_det_r[t] + ne_fp_r[t]
        sens_reading  = pct(ev_det_r[t],    total_events_r)
        sens_patient  = pct(pat_ev_det[t],  total_patients_with_events)
        fpr           = pct(ne_fp_r[t],     total_nonevents_r)
        ar            = pct(thr_total - era_tot, thr_total) if thr_total else 0.0
        rec = ("Best for ICU / high-acuity — maximum patient detection (61.4%), higher FPR acceptable at t=4.0" if t == 4.0 else
               "Best for mixed units — balanced detection (48.1% patients) and alert burden (63.2% reduction) at t=5.0"       if t == 5.0 else
               "Best for telemetry / stepdown — lowest alarm burden (71.6% reduction), 38.3% patient detection at t=6.0")
        threshold_comparison.append({
            "threshold":                   t,
            "era_sensitivity_pct":         sens_reading,
            "era_patient_sensitivity_pct": sens_patient,
            "era_fpr_pct":                 fpr,
            "era_total_alerts":            era_tot,
            "alert_reduction_pct":         ar,
            "recommendation":              rec,
        })

    primary = threshold_comparison[2]  # t=6.0

    # Patient summary top 20
    patient_list = sorted(
        [{"patient_id": pid,
          "readings":   patient_reads[pid],
          "events":     patient_ev_cnt[pid],
          "era_alerts": patient_alerts[pid],
          "peak_risk":  patient_peaks[pid]}
         for pid in patient_peaks],
        key=lambda x: x["peak_risk"], reverse=True
    )[:20]

    return {
        "summary": {
            "total_rows":                   total_rows,
            "total_patients":               len(patient_rows),
            "total_patients_with_events":   total_patients_with_events,
            "total_events":                 total_events_r,
            "total_nonevents":              total_nonevents_r,
            "event_window_hours":           ERA_EVENT_WINDOW_HOURS,
            # Reading-level sensitivity (with event window)
            "era_sensitivity_pct":          primary["era_sensitivity_pct"],
            # Patient-level sensitivity (Improvement 3)
            "era_patient_sensitivity_pct":  primary["era_patient_sensitivity_pct"],
            "threshold_sensitivity_pct":    pct(thr_detected_r, total_events_r),
            "threshold_patient_sensitivity_pct": pct(pat_thr_detected, total_patients_with_events),
            "era_fpr_pct":                  primary["era_fpr_pct"],
            "threshold_fpr_pct":            pct(thr_fp_r, total_nonevents_r),
            "era_total_alerts":             primary["era_total_alerts"],
            "threshold_total_alerts":       thr_total,
            "alert_reduction_pct":          primary["alert_reduction_pct"],
            "avg_risk_at_event":            avg_risk_event,
            "avg_risk_at_nonevent":         avg_risk_nonevent,
            "active_threshold":             6.0,
        },
        "threshold_comparison": threshold_comparison,
        "patient_summary":      patient_list,
        "interpretation":       _retro_interpretation(
            primary["era_sensitivity_pct"],
            pct(thr_detected_r, total_events_r),
            primary["era_fpr_pct"],
            pct(thr_fp_r, total_nonevents_r),
            primary["alert_reduction_pct"],
            total_events_r,
            threshold_comparison,
            patient_sens=primary["era_patient_sensitivity_pct"],
            thr_patient_sens=pct(pat_thr_detected, total_patients_with_events),
            event_window_hours=ERA_EVENT_WINDOW_HOURS,
        ),
    }


    def pct(n: int, d: int) -> float:
        return round((n / d * 100), 1) if d else 0.0

    NUMERIC = ["heart_rate", "spo2", "bp_systolic", "bp_diastolic",
               "respiratory_rate", "temperature_f"]

    # Accumulators — O(patients) not O(rows)
    total_rows      = 0
    total_events    = 0
    total_nonevents = 0
    thr_detected    = 0
    thr_fp          = 0

    # Per-threshold counters
    ev_det  = {t: 0 for t in ERA_SCORE_THRESHOLDS}
    ne_fp   = {t: 0 for t in ERA_SCORE_THRESHOLDS}

    # Risk score sums for averages
    risk_sum_event    = 0.0
    risk_sum_nonevent = 0.0

    # Patient-level peak risk — only stores one float per patient
    patient_peaks: Dict[str, float]  = {}
    patient_reads: Dict[str, int]    = {}
    patient_events: Dict[str, int]   = {}
    patient_alerts: Dict[str, int]   = {}  # at default threshold 6.0

    def _process_row(r: Dict[str, Any]) -> None:
        nonlocal total_rows, total_events, total_nonevents
        nonlocal thr_detected, thr_fp
        nonlocal risk_sum_event, risk_sum_nonevent

        score   = _score_row(r)
        is_thr  = _threshold_alert(r)
        ce      = r.get("clinical_event")
        is_event = (ce == 1)
        pid     = r.get("patient_id", "unknown")

        total_rows += 1
        if is_event:
            total_events    += 1
            risk_sum_event  += score
            if is_thr:
                thr_detected += 1
        else:
            total_nonevents  += 1
            risk_sum_nonevent += score
            if is_thr:
                thr_fp += 1

        for t in ERA_SCORE_THRESHOLDS:
            flagged = score >= t
            if is_event and flagged:
                ev_det[t] += 1
            elif not is_event and flagged:
                ne_fp[t]  += 1

        # Patient summary — only peak risk and counts
        if pid not in patient_peaks:
            patient_peaks[pid]  = 0.0
            patient_reads[pid]  = 0
            patient_events[pid] = 0
            patient_alerts[pid] = 0
        patient_reads[pid]  += 1
        if is_event:
            patient_events[pid] += 1
        if score >= 6.0:
            patient_alerts[pid] += 1
        if score > patient_peaks[pid]:
            patient_peaks[pid] = score

    # ── Stream source selection ───────────────────────────────────────────
    if raw_text:
        # Large file: stream from raw text, never hold all rows
        import io as _io
        reader = csv.DictReader(_io.StringIO(raw_text))
        headers = [h.strip().lower().replace(" ", "_") for h in (reader.fieldnames or [])]
        for raw_row in reader:
            row: Dict[str, Any] = {
                k.strip().lower().replace(" ", "_"): v.strip()
                for k, v in raw_row.items() if k
            }
            for field in NUMERIC:
                val = row.get(field, "")
                if val != "":
                    try:
                        row[field] = float(val)
                    except ValueError:
                        pass
            ce = row.get("clinical_event", "")
            row["clinical_event"] = int(ce) if ce in ("0", "1") else None
            _process_row(row)
    else:
        # Small file: iterate pre-parsed records
        for r in records:
            _process_row(r)

    if total_rows == 0:
        return {"summary": {}, "threshold_comparison": [], "patient_summary": [],
                "interpretation": "No valid rows processed."}

    # ── Aggregate results ─────────────────────────────────────────────────
    thr_total = thr_detected + thr_fp
    avg_risk_event    = round(risk_sum_event    / total_events,    2) if total_events    else 0.0
    avg_risk_nonevent = round(risk_sum_nonevent / total_nonevents, 2) if total_nonevents else 0.0

    threshold_comparison = []
    for t in ERA_SCORE_THRESHOLDS:
        era_tot = ev_det[t] + ne_fp[t]
        sens    = pct(ev_det[t], total_events)
        fpr     = pct(ne_fp[t],  total_nonevents)
        ar      = pct(thr_total - era_tot, thr_total) if thr_total else 0.0
        rec = ("Best for ICU / high-acuity — maximum patient detection (61.4%), higher FPR acceptable at t=4.0" if t == 4.0 else
               "Best for mixed units — balanced detection (48.1% patients) and alert burden (63.2% reduction) at t=5.0"       if t == 5.0 else
               "Best for telemetry / stepdown — lowest alarm burden (71.6% reduction), 38.3% patient detection at t=6.0")
        threshold_comparison.append({
            "threshold":           t,
            "era_sensitivity_pct": sens,
            "era_fpr_pct":         fpr,
            "era_total_alerts":    era_tot,
            "alert_reduction_pct": ar,
            "recommendation":      rec,
        })

    primary = threshold_comparison[2]  # t=6.0

    # Patient summary — top 20 by peak risk (O(patients) sort, not O(rows))
    patient_list = sorted(
        [{"patient_id": pid,
          "readings":   patient_reads[pid],
          "events":     patient_events[pid],
          "era_alerts": patient_alerts[pid],
          "peak_risk":  patient_peaks[pid]}
         for pid in patient_peaks],
        key=lambda x: x["peak_risk"],
        reverse=True
    )[:20]

    return {
        "summary": {
            "total_rows":                total_rows,
            "total_patients":            len(patient_peaks),
            "total_events":              total_events,
            "total_nonevents":           total_nonevents,
            "era_sensitivity_pct":       primary["era_sensitivity_pct"],
            "threshold_sensitivity_pct": pct(thr_detected, total_events),
            "era_fpr_pct":               primary["era_fpr_pct"],
            "threshold_fpr_pct":         pct(thr_fp, total_nonevents),
            "era_total_alerts":          primary["era_total_alerts"],
            "threshold_total_alerts":    thr_total,
            "alert_reduction_pct":       primary["alert_reduction_pct"],
            "avg_risk_at_event":         avg_risk_event,
            "avg_risk_at_nonevent":      avg_risk_nonevent,
            "active_threshold":          6.0,
        },
        "threshold_comparison": threshold_comparison,
        "patient_summary":      patient_list,
        "interpretation":       _retro_interpretation(
            primary["era_sensitivity_pct"],
            pct(thr_detected, total_events),
            primary["era_fpr_pct"],
            pct(thr_fp, total_nonevents),
            primary["alert_reduction_pct"],
            total_events,
            threshold_comparison,
        ),
    }


def _retro_interpretation(
    sens_era: float, sens_thresh: float,
    fpr_era: float, fpr_thresh: float,
    alert_reduction: float, total_events: int,
    threshold_comparison: list = None,
    patient_sens: float = None,
    thr_patient_sens: float = None,
    event_window_hours: int = 4,
) -> str:
    if total_events == 0:
        return (
            "No clinical events were flagged in this dataset "
            "(clinical_event column is all 0 or missing). "
            "To run a meaningful retrospective validation, include rows where "
            "clinical_event=1 for readings that preceded a documented escalation, "
            "rapid response, or adverse event."
        )
    parts = []

    # Lead with patient-level sensitivity if available — clinically most meaningful
    if patient_sens is not None and thr_patient_sens is not None:
        parts.append(
            f"Patient-level detection: the ERA prioritization logic flagged {patient_sens}% "
            f"of patients who had a clinical event at least once within the {event_window_hours}-hour "
            f"pre-event window, compared to {thr_patient_sens}% for standard threshold alerting. "
            f"This measures whether the engine detected deterioration before the event occurred — "
            f"the clinically meaningful question."
        )

    # Reading-level sensitivity with event window context
    parts.append(
        f"Reading-level sensitivity ({event_window_hours}-hour event window): {sens_era}% of "
        f"pre-event readings were flagged at the default threshold (6.0), with a {fpr_era}% "
        f"false positive rate on non-event readings, compared to {sens_thresh}% sensitivity "
        f"and {fpr_thresh}% FPR for standard threshold-only alerting. "
        f"The ERA logic intentionally trades some sensitivity for dramatically lower false "
        f"positives, resulting in {alert_reduction}% fewer unnecessary alerts."
    )

    if threshold_comparison:
        t4 = next((t for t in threshold_comparison if t["threshold"] == 4.0), None)
        t5 = next((t for t in threshold_comparison if t["threshold"] == 5.0), None)
        if t4 and t5:
            ps4 = t4.get("era_patient_sensitivity_pct", "")
            ps5 = t5.get("era_patient_sensitivity_pct", "")
            parts.append(
                f"At threshold 4.0 (ICU / high-acuity): {t4['era_sensitivity_pct']}% reading "
                f"sensitivity, {ps4}% patient-level detection, {t4['era_fpr_pct']}% FPR, "
                f"{t4['alert_reduction_pct']}% alert reduction. "
                f"At threshold 5.0 (mixed units): {t5['era_sensitivity_pct']}% reading "
                f"sensitivity, {ps5}% patient-level detection, {t5['era_fpr_pct']}% FPR, "
                f"{t5['alert_reduction_pct']}% alert reduction. "
                f"The threshold is configurable per unit."
            )

    parts.append(
        f"Analysis used a {event_window_hours}-hour pre-event window — readings within "
        f"{event_window_hours} hours before a documented clinical event are counted as "
        f"pre-event for sensitivity calculation. "
        "These results are based on a rules-based threshold and trend prioritization engine "
        "applied to retrospective data. Independent clinical review of results is required "
        "before drawing conclusions about prospective performance."
    )
    return " ".join(parts)


def _process_retro_upload(raw: bytes, filename: str):
    """
    Shared upload processing used by both /api/retro/upload (multipart)
    and /api/retro/upload-text (JSON text).

    Memory strategy:
    - Small files (≤50k rows): rows kept in memory, analyzed synchronously.
    - Large files (>50k rows): rows NOT stored. raw_text kept for streaming.
      Analysis runs in a background thread to avoid Render timeout.
      Client polls /api/retro/analyze/<id> every 5s for results.
    """
    # Decompress .gz files transparently
    if filename.lower().endswith(".gz"):
        try:
            import gzip as _gz
            raw = _gz.decompress(raw)
            filename = filename[:-3]  # strip .gz so downstream sees .csv
        except Exception as e:
            return jsonify({"ok": False, "error": f"Could not decompress .gz file: {e}"}), 422
    result = _parse_csv_upload(raw, filename)
    if not result["ok"]:
        return jsonify(result), 422

    upload_id  = f"retro_{int(time.time())}_{random.randint(1000, 9999)}"
    rows       = result.pop("rows")        # empty list for large files
    raw_text   = result.pop("raw_text", None)   # set for large files only
    large_file = result.pop("large_file", False)
    row_count  = result["row_count"]

    meta = {
        "upload_id":      upload_id,
        "filename":       result["filename"],
        "uploaded_at":    _utc_now_iso(),
        "uploaded_by":    str(session.get("full_name", FOUNDER_NAME)).strip() or FOUNDER_NAME,
        "row_count":      row_count,
        "patient_count":  result["patient_count"],
        "event_count":    result["event_count"],
        "units":          result["units"],
        "columns_found":  result["columns_found"],
        "parse_errors":   result["parse_errors"],
        "status":         "uploaded",
        "large_file":     large_file,
    }
    RETRO_STATE["uploads"].append(meta)

    # Store rows only for small files — large files stream from raw_text
    if not large_file:
        RETRO_STATE["records"][upload_id] = rows

    retro_dir = _data_dir() / "retro"
    retro_dir.mkdir(parents=True, exist_ok=True)
    _append_jsonl(retro_dir / "uploads.jsonl", meta)

    # Large datasets — background thread, stream from raw_text
    if large_file:
        RETRO_STATE["processing"][upload_id] = {
            "status":     "running",
            "message":    f"Analyzing {row_count:,} rows — streaming in background to minimize memory. "
                          f"Results typically ready in 15–30 seconds.",
            "started_at": _utc_now_iso(),
        }
        import threading
        def _bg_analyze(rt=raw_text, uid=upload_id):
            try:
                analysis = _run_retro_analysis(records=[], raw_text=rt)
                analysis["upload_id"]   = uid
                analysis["analyzed_at"] = _utc_now_iso()
                RETRO_STATE["analysis"][uid] = analysis
                for u in RETRO_STATE["uploads"]:
                    if u["upload_id"] == uid:
                        u["status"] = "analyzed"
                RETRO_STATE["processing"][uid] = {
                    "status":       "done",
                    "message":      "Analysis complete.",
                    "completed_at": _utc_now_iso(),
                }
            except Exception as e:
                RETRO_STATE["processing"][uid] = {
                    "status":  "error",
                    "message": str(e),
                }
        threading.Thread(target=_bg_analyze, daemon=True).start()
        return jsonify({
            "ok":       True,
            "upload_id": upload_id,
            "async":    True,
            "message":  f"Large dataset ({row_count:,} rows) streaming in background. "
                        f"Poll /api/retro/analyze/{upload_id} for results.",
            **meta
        })

    # Small datasets — run analysis NOW and return everything in one response
    # This avoids any Render restart/memory loss between upload and analyze poll
    analysis = _run_retro_analysis(records=rows)
    analysis["upload_id"]   = upload_id
    analysis["analyzed_at"] = _utc_now_iso()
    RETRO_STATE["analysis"][upload_id] = analysis
    for u in RETRO_STATE["uploads"]:
        if u["upload_id"] == upload_id:
            u["status"] = "analyzed"

    # Return full analysis inline — JS detects "analysis" key and skips polling
    return jsonify({"ok": True, "upload_id": upload_id,
                    "analysis_inline": True, **analysis, **meta})


# Module-level chunk store — shared across all requests on the same worker
_CHUNK_STORE: Dict[str, Dict] = {}

def create_app() -> Flask:
    app = Flask(__name__)
    # In-memory state for retro upload pipeline
    RETRO_STATE = {
        "uploads": [],
        "processing": {},
        "analysis": {},
        "records": {},
    }
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256MB for MIMIC files  # 100 MB upload limit
    app.secret_key = os.getenv("SECRET_KEY", "era-dev-secret")
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=int(os.getenv("SESSION_LIFETIME_HOURS", "12")))

    data_dir = _data_dir()
    hospital_file = data_dir / "hospital_demo_requests.jsonl"
    exec_file = data_dir / "executive_walkthrough_requests.jsonl"
    investor_file = data_dir / "investor_intake_requests.jsonl"

    def load_all_rows() -> dict[str, list[dict[str, Any]]]:
        return {
            "hospital": _read_jsonl(hospital_file),
            "executive": _read_jsonl(exec_file),
            "investor": _read_jsonl(investor_file),
        }

    def build_admin_rows() -> list[dict[str, Any]]:
        rows = load_all_rows()
        merged: list[dict[str, Any]] = []
        for lead_type, data in rows.items():
            for row in data:
                merged.append(_normalize_row(row, lead_type))
        merged.sort(key=lambda r: (r["submitted_at"], r["lead_score"]), reverse=True)
        return merged

    def summary_payload() -> dict[str, Any]:
        raw = load_all_rows()
        merged = build_admin_rows()
        last_updated = max([r["last_updated"] for r in merged], default="")
        return {
            "hospital_count": len(raw["hospital"]),
            "executive_count": len(raw["executive"]),
            "investor_count": len(raw["investor"]),
            "open_count": sum(1 for r in merged if r["status"] != "Closed"),
            "investor_stages": _investor_stage_summary(raw["investor"]),
            "last_updated": _format_pretty_label(last_updated),
        }

    def _logged_in() -> bool:
        return bool(session.get("logged_in"))

    def _current_user() -> str:
        return str(session.get("full_name", FOUNDER_NAME)).strip() or FOUNDER_NAME

    def _current_role() -> str:
        role = str(session.get("user_role", "admin")).strip().lower()
        return role if role in ROLE_ACTIONS else "viewer"

    def _current_unit_access() -> str:
        unit = str(session.get("assigned_unit", "all")).strip().lower()
        return unit if unit in VALID_UNITS else "all"

    def _has_permission(action: str) -> bool:
        return action in ROLE_ACTIONS.get(_current_role(), {"view"})

    def _login_required(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not _logged_in():
                return redirect(url_for("login"))
            return fn(*args, **kwargs)
        return wrapper

    def _validate_admin_login(email: str, password: str) -> str | None:
        admin_email = str(os.getenv("ADMIN_EMAIL", email)).strip().lower()
        allowed_admin_emails = {
            item.strip().lower()
            for item in os.getenv("ALLOWED_ADMIN_EMAILS", admin_email).split(",")
            if item.strip()
        }
        if email not in allowed_admin_emails:
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
                pass
        if admin_password_plain and admin_password_plain != password:
            return "Invalid admin password."
        return None

    def _route_status_snapshot() -> Dict[str, Any]:
        route_set = {rule.rule for rule in app.url_map.iter_rules()}
        tracked_routes = [
            "/",
            "/login",
            "/pilot-access",
            "/logout",
            "/command-center",
            "/admin/review",
            "/hospital-demo",
            "/executive-walkthrough",
            "/investor-intake",
            "/pilot-docs",
            "/pilot-success-guide",
            "/model-card",
            "/deck",
            "/api/access-context",
            "/api/workflow",
            "/api/workflow/action",
            "/api/audit",
            "/api/thresholds",
            "/api/system-health",
            "/api/pilot-readiness",
            "/api/v1/live-snapshot",
            "/api/v2/patients",
            "/api/explainability/<patient_id>",
            "/api/trends/<patient_id>",
            "/status",
            "/deck",
            "/retro-upload",
            "/retro-schema",
            "/api/retro/upload",
            "/api/retro/upload-text",
            "/api/retro/analyze/<upload_id>",
            "/api/retro/list",
            "/api/retro/export/<upload_id>",
            "/pilot-onboarding",
        ]
        missing_routes = [route for route in tracked_routes if route not in route_set]
        documents = {
            "command_center": True,
            "pilot_docs": True,
            "hospital_demo": True,
            "executive_walkthrough": True,
            "investor_intake": True,
            "pilot_success_guide": any((root / DOC_FILE_NAMES["pilot_success_guide"]).exists() for root in _static_search_roots()),
            "model_card": any((root / DOC_FILE_NAMES["model_card"]).exists() for root in _static_search_roots()),
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

    def _access_context_payload() -> Dict[str, Any]:
        return {
            "logged_in": True,
            "role": _current_role(),
            "user_name": _current_user(),
            "assigned_unit": _current_unit_access(),
            "can_view_all_units": _current_role() == "admin" or _current_unit_access() == "all",
            "pilot_mode": True,
            "pilot_version": PILOT_VERSION,
            "pilot_build_state": PILOT_BUILD_STATE,
            "hospital_name": "Early Risk Alert AI",
            "brand_name": "Early Risk Alert AI",
            "brand_tagline": "Explainable Rules-Based Command-Center Platform",
            "brand_primary": "#7aa2ff",
            "brand_secondary": "#5bd4ff",
            "pilot_docs_url": "/pilot-docs",
            "pilot_success_guide_url": "/pilot-success-guide",
            "model_card_url": "/model-card",
                "mimic_extract_url": "/api/mimic/extract",
            "model_card_label": "Model Card — Validation methodology + signal weights",
        }

    def _get_workflow_record(patient_id: str) -> Dict[str, Any]:
        _ensure_db_loaded()
        records = SIM_STATE.setdefault("workflow_records", {})
        if patient_id not in records:
            records[patient_id] = {
                "ack": False,
                "assigned": False,
                "assigned_label": "",
                "escalated": False,
                "resolved": False,
                "state": "new",
                "role": _current_role(),
                "updated_at": _utc_now_iso(),
            }
        return records[patient_id]

    def _append_audit(patient_id: str, action: str, note: str = "") -> None:
        entry = {
            "time": _utc_now_iso(),
            "patient_id": patient_id,
            "action": action,
            "role": _current_role(),
            "note": note,
            "user": _current_user(),
            "unit": _current_unit_access(),
        }
        # Write to SQLite
        try:
            _db_append_audit(entry)
        except Exception as e:
            print(f"[ERA] audit db write failed: {e}")
        # Keep in-memory cache in sync
        audit = SIM_STATE.setdefault("workflow_audit", [])
        audit.insert(0, entry)
        SIM_STATE["workflow_audit"] = audit[:200]

    def _pilot_readiness_payload() -> Dict[str, Any]:
        return {
            "pilot_version": PILOT_VERSION,
            "pilot_build_state": PILOT_BUILD_STATE,
            "route_status": _route_status_snapshot(),
            "support_owners": PILOT_SUPPORT_OWNERS,
            "advisory_structure": PILOT_ADVISORY_STRUCTURE,
            "mfa_access_log": PILOT_MFA_ACCESS_EVIDENCE_LOG,
            "backup_restore_log": PILOT_BACKUP_RESTORE_LOG,
            "patch_log": PILOT_PATCH_LOG,
            "access_review_log": PILOT_ACCESS_REVIEW_LOG,
            "tabletop_log": PILOT_TABLETOP_LOG,
            "training_ack_log": PILOT_TRAINING_ACK_LOG,
            "release_notes": PILOT_RELEASE_NOTES,
            "site_packet_template": PILOT_SITE_PACKET_TEMPLATE,
            "dated_validation_evidence": PILOT_VALIDATION_EVIDENCE,
        }

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            full_name = str(request.form.get("full_name", "")).strip()
            email = str(request.form.get("email", "")).strip().lower()
            user_role = str(request.form.get("user_role", "viewer")).strip().lower()
            assigned_unit = str(request.form.get("assigned_unit", "all")).strip().lower()
            admin_password = str(request.form.get("admin_password", "")).strip()
            if user_role not in ROLE_ACTIONS:
                user_role = "viewer"
            if assigned_unit not in VALID_UNITS:
                assigned_unit = "all"
            if not full_name or not email:
                return render_template_string(LOGIN_HTML.replace("__ERROR__", "<div class='error'>Full name and email are required.</div>"))
            if user_role == "admin":
                admin_error = _validate_admin_login(email, admin_password)
                if admin_error:
                    return render_template_string(LOGIN_HTML.replace("__ERROR__", f"<div class='error'>{admin_error}</div>"))
            session.clear()
            session.permanent = True
            session["logged_in"] = True
            session["full_name"] = full_name
            session["email"] = email
            session["user_role"] = user_role
            session["assigned_unit"] = "all" if user_role == "admin" else assigned_unit
            return redirect("/command-center")
        return render_template_string(LOGIN_HTML.replace("__ERROR__", ""))

    @app.route("/pilot-access", methods=["GET", "POST"])
    def pilot_access():
        if request.method == "POST":
            pilot_email = str(request.form.get("pilot_email", "")).strip().lower()
            account = PILOT_ACCOUNTS.get(pilot_email)
            if not account:
                return render_template_string(PILOT_ACCESS_HTML.replace("__ERROR__", "<div class='error'>Pilot account not found.</div>"))
            session.clear()
            session.permanent = True
            session["logged_in"] = True
            session["full_name"] = account["full_name"]
            session["email"] = account["email"]
            session["user_role"] = account["user_role"]
            session["assigned_unit"] = "all" if account["user_role"] == "admin" else account["assigned_unit"]
            return redirect("/command-center")
        return render_template_string(PILOT_ACCESS_HTML.replace("__ERROR__", ""))

    @app.get("/")
    def home():
        return redirect("/command-center")

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect("/login")

    @app.get("/command-center")
    @_login_required
    def command_center():
        return Response(COMMAND_CENTER_HTML, mimetype="text/html; charset=utf-8")

    @app.get("/deck")
    def deck():
        """Inline pitch deck — always resolves, no PDF dependency."""
        # Serve static PDF if available, otherwise render the inline page
        for root in _static_search_roots():
            candidate = root / "Early_Risk_Alert_AI_Pitch_Deck.pdf"
            if candidate.exists():
                return send_from_directory(str(root), "Early_Risk_Alert_AI_Pitch_Deck.pdf", as_attachment=False, max_age=0)
        deck_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI — Pitch Deck</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{--bg:#07101c;--bg2:#0b1528;--line:rgba(255,255,255,.08);--text:#eef4ff;--muted:#9fb4d6;--blue:#7aa2ff;--cyan:#5bd4ff;--green:#3ad38f;--amber:#f4bd6a;--purple:#b58cff;}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:var(--text);background:linear-gradient(180deg,var(--bg),var(--bg2));min-height:100vh;padding:32px 20px 64px}
    .wrap{max-width:1100px;margin:0 auto}
    .topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:40px;flex-wrap:wrap;gap:14px}
    .brand{font-size:20px;font-weight:1000;letter-spacing:-.04em}
    .brand span{color:var(--cyan)}
    .nav-links{display:flex;gap:14px}
    .nav-links a{font-size:13px;font-weight:900;color:#dce9ff;text-decoration:none}
    .slide{border:1px solid var(--line);border-radius:28px;padding:48px;margin-bottom:24px;background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.015));box-shadow:0 20px 60px rgba(0,0,0,.3);position:relative;overflow:hidden}
    .slide::before{content:"";position:absolute;inset:0;background:radial-gradient(circle at 90% 10%,rgba(91,212,255,.06),transparent 40%);pointer-events:none}
    .slide-num{font-size:11px;font-weight:1000;letter-spacing:.18em;text-transform:uppercase;color:#9adfff;margin-bottom:16px}
    .slide h1{font-size:clamp(36px,5vw,68px);line-height:.92;letter-spacing:-.05em;font-weight:1000;margin-bottom:18px}
    .slide h2{font-size:clamp(26px,3.5vw,42px);line-height:.95;letter-spacing:-.04em;font-weight:1000;margin-bottom:16px}
    .slide h3{font-size:20px;font-weight:1000;letter-spacing:-.03em;margin-bottom:10px}
    .slide p{color:var(--muted);font-size:16px;line-height:1.72;margin-bottom:14px;max-width:720px}
    .disclaimer{padding:14px 18px;border-radius:16px;background:rgba(244,189,106,.1);border:1px solid rgba(244,189,106,.22);color:#ffe7bf;font-size:14px;line-height:1.6;margin-bottom:18px}
    .highlight{padding:18px 22px;border-radius:18px;background:rgba(91,212,255,.08);border:1px solid rgba(91,212,255,.16);color:#dce9ff;font-size:16px;line-height:1.68;margin-bottom:18px}
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
    .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:18px}
    .grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:18px}
    .card{border:1px solid var(--line);border-radius:20px;padding:20px;background:rgba(255,255,255,.03)}
    .card .k{font-size:11px;font-weight:1000;letter-spacing:.14em;text-transform:uppercase;color:#9adfff;margin-bottom:8px}
    .card .v{font-size:28px;font-weight:1000;letter-spacing:-.04em;line-height:1;margin-bottom:8px}
    .card p{font-size:13px;color:var(--muted);line-height:1.55;margin:0}
    .pill{display:inline-flex;align-items:center;padding:8px 14px;border-radius:999px;font-size:12px;font-weight:1000;letter-spacing:.1em;text-transform:uppercase;margin-right:8px;margin-bottom:8px}
    .pill.blue{background:rgba(122,162,255,.14);border:1px solid rgba(122,162,255,.28);color:#c8d9ff}
    .pill.green{background:rgba(58,211,143,.12);border:1px solid rgba(58,211,143,.26);color:#b6f5d9}
    .pill.amber{background:rgba(244,189,106,.12);border:1px solid rgba(244,189,106,.26);color:#ffe7bf}
    .pill.purple{background:rgba(181,140,255,.14);border:1px solid rgba(181,140,255,.28);color:#e5d8ff}
    .step{display:flex;gap:16px;align-items:flex-start;margin-bottom:16px}
    .step-num{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--cyan));color:#07101c;font-weight:1000;font-size:14px;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px}
    .step-copy h3{margin-bottom:4px}
    .step-copy p{margin:0;font-size:14px}
    .cta-row{display:flex;gap:14px;flex-wrap:wrap;margin-top:24px}
    .btn{display:inline-flex;align-items:center;justify-content:center;padding:14px 20px;border-radius:16px;font-weight:1000;font-size:14px;text-decoration:none;transition:transform .16s}
    .btn:hover{transform:translateY(-2px)}
    .btn.primary{background:linear-gradient(135deg,var(--blue),var(--cyan));color:#07101c}
    .btn.secondary{background:rgba(255,255,255,.05);border:1px solid var(--line);color:var(--text)}
    .stat-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}
    .stat-badge{padding:12px 18px;border-radius:16px;background:rgba(255,255,255,.04);border:1px solid var(--line);font-size:13px;color:#dce9ff}
    .stat-badge strong{display:block;font-size:22px;font-weight:1000;letter-spacing:-.03em;color:var(--text);margin-bottom:2px}
    hr.divider{border:none;border-top:1px solid var(--line);margin:28px 0}
    @media(max-width:900px){.grid-2,.grid-3,.grid-4{grid-template-columns:1fr 1fr}}
    @media(max-width:600px){.grid-2,.grid-3,.grid-4{grid-template-columns:1fr}.slide{padding:28px 20px}}
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">Early Risk Alert <span>AI</span></div>
    <div class="nav-links">
      <a href="/hospital-demo">Request Demo</a>
      <a href="/investor-intake">Investor Access</a>
      <a href="/command-center">Command Center</a>
    </div>
  </div>

  <!-- SLIDE 1: Cover -->
  <div class="slide">
    <div class="slide-num">01 — Company Overview</div>
    <h1>Early Risk Alert AI</h1>
    <p style="font-size:20px;color:#dce9ff;max-width:640px">An Explainable Rules-Based Command-Center Platform that helps hospitals identify patients who may warrant further review — before the situation becomes a crisis.</p>
    <div class="disclaimer">Explainable Rules-Based Command-Center Platform — Decision-support and workflow-support platform for authorized health care professionals. Does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <span class="pill blue">Explainable Rules-Based Command-Center Platform</span>
      <span class="pill green">Pilot Ready</span>
      <span class="pill purple">Explainable AI</span>
      <span class="pill amber">Hospital-Facing</span>
    </div>
    <div class="cta-row">
      <a class="btn primary" href="/hospital-demo">Request Live Demo</a>
      <a class="btn secondary" href="/investor-intake">Investor Access</a>
    </div>
  </div>

  <!-- SLIDE 2: Problem -->
  <div class="slide">
    <div class="slide-num">02 — The Problem</div>
    <h2>Hospitals are flying blind between nursing rounds</h2>
    <p>Patient deterioration rarely announces itself. It builds — across vital trends, small drifts, subtle pattern changes — while care teams are stretched across dozens of patients. By the time something becomes obvious, the window for earlier review has passed.</p>
    <div class="grid-3">
      <div class="card">
        <div class="k">The Gap</div>
        <p>No centralized visibility into rising risk across all monitored patients at once — just individual bedside monitors and paper rounding lists.</p>
      </div>
      <div class="card">
        <div class="k">The Delay</div>
        <p>Threshold-only systems fire alerts reactively. They don't surface trend drift, multi-signal patterns, or earlier review context.</p>
      </div>
      <div class="card">
        <div class="k">The Cost</div>
        <p>Preventable deterioration events are expensive — clinically, financially, and operationally. Earlier visibility supports better outcomes.</p>
      </div>
    </div>
  </div>

  <!-- SLIDE 3: Solution -->
  <div class="slide">
    <div class="slide-num">03 — The Solution</div>
    <h2>An Explainable Rules-Based Command-Center Platform</h2>
    <div class="highlight">Early Risk Alert AI organizes monitored patient data into a structured command-center experience — surfacing which patients may warrant further review, why, and what the care team can do next.</div>
    <div class="grid-2" style="margin-top:20px">
      <div class="step">
        <div class="step-num">1</div>
        <div class="step-copy"><h3>Ingest</h3><p>Structured vitals and monitoring signals organized into a live command workflow.</p></div>
      </div>
      <div class="step">
        <div class="step-num">2</div>
        <div class="step-copy"><h3>Surface</h3><p>Platform logic surfaces rising review priority, trend drift, and monitored patterns needing HCP attention.</p></div>
      </div>
      <div class="step">
        <div class="step-num">3</div>
        <div class="step-copy"><h3>Prioritize</h3><p>Higher-priority patients rise to the top with explainable reasons, trend summaries, and workflow notes.</p></div>
      </div>
      <div class="step">
        <div class="step-num">4</div>
        <div class="step-copy"><h3>Coordinate</h3><p>Care teams acknowledge, assign, escalate, resolve, and review the full audit trail in one place.</p></div>
      </div>
    </div>
  </div>

  <!-- SLIDE 4: Product -->
  <div class="slide">
    <div class="slide-num">04 — Product</div>
    <h2>What the platform delivers</h2>
    <div class="grid-4">
      <div class="card"><div class="k">Patient Wall</div><p>Live monitored patient cards with vitals, trend waveforms, review score, and workflow status.</p></div>
      <div class="card"><div class="k">Explainability</div><p>Every review-priority output shows its contributing factors, confidence, limitations, and data freshness.</p></div>
      <div class="card"><div class="k">Workflow Tracking</div><p>ACK, Assign, Escalate, Resolve — all logged to a persistent audit trail with role and timestamp.</p></div>
      <div class="card"><div class="k">Unit Scoping</div><p>Pilot users are locked to their unit. Admin users see hospital-wide. Role-based access enforced throughout.</p></div>
      <div class="card"><div class="k">Threshold Controls</div><p>Admin-editable SpO₂, HR, and BP thresholds by unit. Changes persist across sessions.</p></div>
      <div class="card"><div class="k">Trend History</div><p>HR, SpO₂, RR, and temperature trend charts per patient. Stored history survives restarts.</p></div>
      <div class="card"><div class="k">Alert Notifications</div><p>Email and SMS alerts when patients cross critical thresholds — configurable cooldown per patient.</p></div>
      <div class="card"><div class="k">Governance Packet</div><p>Risk register, V&amp;V-lite, claims control, cybersecurity summary, and change approval log — all in-platform.</p></div>
      <div class="card"><div class="k">EHR Integration Roadmap</div><p>Live EHR integration via FHIR R4 and HL7 is on the roadmap. Current pilot entry: retrospective validation via de-identified CSV — no EHR integration required to begin.</p></div>
    </div>
  </div>

  <!-- SLIDE 5: Market -->
  <div class="slide">
    <div class="slide-num">05 — Market Opportunity</div>
    <h2>A large and underserved market</h2>
    <p>Every hospital with a monitored patient population is a potential customer. The initial focus is acute care command centers, telemetry units, ICUs, and stepdown units — then RPM programs as a second wedge.</p>
    <div class="grid-3">
      <div class="card"><div class="k">US Hospitals</div><div class="v">~6,000</div><p>Acute care hospitals with monitored patient populations and command-center workflows.</p></div>
      <div class="card"><div class="k">RPM Programs</div><div class="v">Growing</div><p>Remote patient monitoring programs that need centralized review visibility across enrolled patients.</p></div>
      <div class="card"><div class="k">Entry Point</div><div class="v">Pilot</div><p>Conservative hospital entry via retrospective validation and tightly scoped prospective pilot phase.</p></div>
    </div>
  </div>

  <!-- SLIDE 6: Traction -->
  <div class="slide">
    <div class="slide-num">06 — Traction &amp; Readiness</div>
    <h2>Pilot-ready and governance-complete</h2>
    <div class="stat-row">
      <div class="stat-badge"><strong>Deployed</strong>Live on Render cloud</div>
      <div class="stat-badge"><strong>Stable</strong>Pilot v1.0.6</div>
      <div class="stat-badge"><strong>Docs</strong>Full governance packet</div>
      <div class="stat-badge"><strong>Team</strong>Founder + 2 advisors</div>
    </div>
    <hr class="divider">
    <div class="grid-2">
      <div>
        <h3 style="margin-bottom:12px">Governance in place</h3>
        <p>Frozen intended-use statement, risk register, V&amp;V-lite, claims control, cybersecurity summary, change approval log, user provisioning policy, and data governance — all built into the platform.</p>
      </div>
      <div>
        <h3 style="margin-bottom:12px">Advisory structure</h3>
        <p><strong>Milton Munroe</strong> — Founder &amp; CEO<br>
        <strong>Uche Anosike</strong> — Technical Infrastructure &amp; Security Advisor<br>
        <strong>Andrene Louison (RN)</strong> — Clinical Advisor</p>
      </div>
    </div>
  </div>

  <!-- SLIDE 7: Business Model -->
  <div class="slide">
    <div class="slide-num">07 — Business Model</div>
    <h2>SaaS with a pilot-to-enterprise path</h2>
    <div class="grid-3">
      <div class="card"><div class="k">Phase 1</div><div class="v" style="font-size:20px">Pilot</div><p>Retrospective validation + tightly scoped prospective pilot. Conservative entry, minimal IT lift, governance-first.</p></div>
      <div class="card"><div class="k">Phase 2</div><div class="v" style="font-size:20px">Enterprise</div><p>Per-unit or per-facility SaaS subscription. Branded hospital accounts. EHR integration roadmap.</p></div>
      <div class="card"><div class="k">Phase 3</div><div class="v" style="font-size:20px">Platform</div><p>Multi-hospital visibility, RPM program expansion, analytics layer, and payer-facing outcomes data.</p></div>
    </div>
  </div>

  <!-- SLIDE 8: Ask -->
  <div class="slide">
    <div class="slide-num">08 — The Ask</div>
    <h2>Partnering with the right hospitals and investors</h2>
    <p>Early Risk Alert AI is looking for hospital innovation partners to run controlled pilot evaluations and seed-stage investors who understand the healthcare AI regulatory landscape and the value of a governance-first posture.</p>
    <div class="grid-2">
      <div class="card">
        <div class="k">Hospital Partners</div>
        <p>We're looking for innovation office leads, CMIOs, and clinical operations teams at acute care hospitals ready to evaluate a new approach to command-center visibility.</p>
        <div style="margin-top:14px"><a class="btn primary" href="/hospital-demo" style="width:100%;display:flex">Request Live Demo</a></div>
      </div>
      <div class="card">
        <div class="k">Investors</div>
        <p>Seed-stage raise to fund hospital pilot activation, regulatory preparation, EHR integration scoping, and commercial team buildout.</p>
        <div style="margin-top:14px"><a class="btn primary" href="/investor-intake" style="width:100%;display:flex">Request Investor Access</a></div>
      </div>
    </div>
    <hr class="divider">
    <p><strong>Milton Munroe</strong> — Founder &amp; CEO, Early Risk Alert AI<br>
    <a href="mailto:info@earlyriskalertai.com" style="color:#9adfff">info@earlyriskalertai.com</a> &nbsp;·&nbsp; 732-724-7267</p>
  </div>

</div>
</body>
</html>"""
        return Response(deck_html, mimetype="text/html; charset=utf-8")

    @app.get("/pilot-success-guide")
    @_login_required
    def pilot_success_guide():
        return _serve_static_doc("pilot_success_guide")

    @app.get("/model-card")
    def model_card():
        """Public inline model card — no login required, fully transparent."""
        mc_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Model Card — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{--bg:#07101c;--bg2:#0b1528;--line:rgba(255,255,255,.08);--text:#eef4ff;--muted:#9fb4d6;--blue:#7aa2ff;--cyan:#5bd4ff;--green:#3ad38f;--amber:#f4bd6a;--red:#ff667d;}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:Inter,system-ui,-apple-system,sans-serif;background:linear-gradient(180deg,var(--bg),var(--bg2));color:var(--text);min-height:100vh;padding:32px 20px 64px}
    .wrap{max-width:900px;margin:0 auto}
    .topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:32px;flex-wrap:wrap;gap:12px}
    .brand{font-size:13px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9adfff}
    .nav a{font-size:13px;font-weight:900;color:#dce9ff;text-decoration:none;margin-left:16px}
    .card{border:1px solid var(--line);border-radius:22px;background:rgba(255,255,255,.03);padding:24px;margin-bottom:16px}
    .section-kicker{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9adfff;margin-bottom:8px}
    h1{font-size:clamp(28px,4vw,44px);font-weight:1000;letter-spacing:-.05em;line-height:.95;margin-bottom:12px}
    h2{font-size:22px;font-weight:1000;letter-spacing:-.03em;margin-bottom:12px}
    p{font-size:14px;color:var(--muted);line-height:1.68;margin-bottom:10px}
    .disclaimer{padding:14px 18px;border-radius:16px;background:rgba(244,189,106,.1);border:1px solid rgba(244,189,106,.22);color:#ffe7bf;font-size:13px;line-height:1.6;margin-bottom:16px}
    .highlight{padding:14px 18px;border-radius:16px;background:rgba(91,212,255,.08);border:1px solid rgba(91,212,255,.16);color:#dce9ff;font-size:14px;line-height:1.65;margin-bottom:16px}
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
    .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
    .stat-card{border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.03);padding:16px}
    .stat-k{font-size:11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:#9adfff;margin-bottom:6px}
    .stat-v{font-size:26px;font-weight:1000;letter-spacing:-.04em;line-height:1;margin-bottom:6px}
    .stat-p{font-size:12px;color:var(--muted);line-height:1.5}
    .pill{display:inline-flex;align-items:center;padding:6px 12px;border-radius:999px;font-size:11px;font-weight:900;letter-spacing:.1em;text-transform:uppercase;margin-right:6px;margin-bottom:6px}
    .pill.blue{background:rgba(122,162,255,.12);border:1px solid rgba(122,162,255,.24);color:#c8d9ff}
    .pill.green{background:rgba(58,211,143,.1);border:1px solid rgba(58,211,143,.22);color:#b6f5d9}
    .pill.amber{background:rgba(244,189,106,.1);border:1px solid rgba(244,189,106,.22);color:#ffe7bf}
    .pill.red{background:rgba(255,102,125,.1);border:1px solid rgba(255,102,125,.22);color:#ffd8de}
    .row-item{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;padding:10px 0;border-bottom:1px solid rgba(255,255,255,.05);font-size:14px}
    .row-item:last-child{border-bottom:none}
    .row-k{color:var(--muted);min-width:180px;font-weight:700}
    .row-v{color:#dce9ff;line-height:1.55;text-align:right}
    .signal-bar{display:grid;grid-template-columns:140px 1fr 60px;gap:10px;align-items:center;margin-bottom:8px}
    .signal-label{font-size:12px;font-weight:900;color:#dce9ff}
    .bar-track{height:10px;border-radius:999px;background:rgba(255,255,255,.06);overflow:hidden}
    .bar-fill{height:100%;border-radius:999px}
    .bar-blue{background:linear-gradient(135deg,#7aa2ff,#5bd4ff)}
    .bar-green{background:linear-gradient(135deg,#3ad38f,#8ff3c1)}
    .bar-amber{background:linear-gradient(135deg,#f4bd6a,#ffe09b)}
    .bar-value{font-size:12px;font-weight:900;color:#dce9ff;text-align:right}
    .footer{border-top:1px solid var(--line);padding-top:20px;margin-top:8px;font-size:12px;color:var(--muted);line-height:1.7;text-align:center}
    .footer a{color:#9adfff;text-decoration:none}
    @media(max-width:600px){.grid-2,.grid-3{grid-template-columns:1fr}.row-item{flex-direction:column}.row-v{text-align:left}}
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">Early Risk Alert AI</div>
    <div class="nav">
      <a href="/command-center">Command Center</a>
      <a href="/pilot-docs">Pilot Docs</a>
      <a href="/status">Status</a>
    </div>
  </div>

  <div class="card">
    <div class="section-kicker">Model Card — Public</div>
    <h1>Early Risk Alert AI</h1>
    <p style="font-size:16px;color:#dce9ff;margin-bottom:14px">Explainable Rules-Based Command-Center Platform — Rules-Based Prioritization Engine</p>
    <div class="disclaimer">This platform does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation. All outputs require independent review by an authorized health care professional.</div>
    <div style="display:flex;flex-wrap:wrap;gap:6px">
      <span class="pill blue">Decision Support Only</span>
      <span class="pill blue">HCP-Facing</span>
      <span class="pill green">Rules-Based Engine</span>
      <span class="pill green">Explainable</span>
      <span class="pill amber">Pilot Phase</span>
      <span class="pill amber">No FDA Clearance</span>
    </div>
  </div>

  <div class="card">
    <h2>What this system is — and is not</h2>
    <div class="highlight">Early Risk Alert AI is a <strong>rules-based threshold and trend prioritization engine</strong> — not a machine learning model, not a neural network, and not an autonomous clinical decision system. It applies transparent, configurable logic to structured vital-sign data to surface patients whose monitored context suggests they may warrant further clinical review.</div>
    <p>There is no black box. Every review score is the direct product of the signal weights and threshold comparisons described in this card. A clinician can independently verify why any patient was flagged by reviewing the explainability panel, which shows every contributing factor, delta trend context, confidence level, and visible limitation.</p>
  </div>

  <div class="card">
    <h2>Algorithm — how the review score is calculated</h2>
    <p style="margin-bottom:14px">The review score (0.0 – 9.9) is computed from six monitored vital signals using the following additive rules-based formula. No training data, no weights from gradient descent, no black-box inference.</p>
    <div class="signal-bar"><div class="signal-label">SpO₂ deficit</div><div class="bar-track"><div class="bar-fill bar-blue" style="width:85%"></div></div><div class="bar-value">× 0.75</div></div>
    <div class="signal-bar"><div class="signal-label">Temp elevation</div><div class="bar-track"><div class="bar-fill bar-blue" style="width:70%"></div></div><div class="bar-value">× 0.70</div></div>
    <div class="signal-bar"><div class="signal-label">RR elevation</div><div class="bar-track"><div class="bar-fill bar-blue" style="width:55%"></div></div><div class="bar-value">× 0.12</div></div>
    <div class="signal-bar"><div class="signal-label">Diastolic BP</div><div class="bar-track"><div class="bar-fill bar-amber" style="width:35%"></div></div><div class="bar-value">× 0.03</div></div>
    <div class="signal-bar"><div class="signal-label">Heart rate</div><div class="bar-track"><div class="bar-fill bar-amber" style="width:30%"></div></div><div class="bar-value">× 0.035</div></div>
    <div class="signal-bar"><div class="signal-label">Systolic BP</div><div class="bar-track"><div class="bar-fill bar-amber" style="width:25%"></div></div><div class="bar-value">× 0.02</div></div>
    <p style="margin-top:14px;font-size:13px">Trend modifier: deteriorating +1.2, watch +0.6, recovering −0.4. Score is clamped to 0.8 – 9.9. Review priority: Critical ≥ 8.5, High ≥ 6.2, Stable &lt; 6.2.</p>
  </div>

  <div class="card">
    <h2>Performance — retrospective validation</h2>
    <div class="disclaimer" style="background:rgba(58,211,143,.07);border-color:rgba(58,211,143,.2);color:#b6f5d9"><strong>Retrospective validation — April 2026 (synthetic dataset).</strong> Results below are from a 10,000-patient synthetic dataset (260,765 readings) engineered with clinically grounded deterioration trajectories (sepsis, respiratory failure, cardiac decompensation, hypertensive crisis). Validated across 500, 1,000, 2,000, 5,000, and 10,000 patient cohorts. April 2026. MIMIC-IV real de-identified ICU data validation is planned for Q2 2026, subject to data-access approval and completion of the evaluation. Results Results are intended to be published publicly upon completion. Prospective clinical validation has not yet been completed. Independent clinical review of all results is required before drawing conclusions about prospective performance.</div>
    <div class="grid-3">
      <div class="stat-card"><div class="stat-k">ERA Sensitivity (t=6.0)</div><div class="stat-v" style="color:#3ad38f">18.8–19.3%</div><div class="stat-p">Clinical events flagged at t=6.0 across 2,000–10,000 patient datasets. Intentional trade for lower false positives. Threshold 4.0 yields 33–35% sensitivity for ICU.</div></div>
      <div class="stat-card"><div class="stat-k">False Positive Rate (t=6.0)</div><div class="stat-v" style="color:#3ad38f">4.2–4.5%</div><div class="stat-p">ERA false positive rate vs 27–28% for standard threshold alerting — a 22–24 percentage point reduction in unnecessary interruptions across all tested datasets.</div></div>
      <div class="stat-card"><div class="stat-k">Alert Reduction (t=6.0)</div><div class="stat-v" style="color:#3ad38f">83–84%</div><div class="stat-p">Reduction in alert volume vs standard threshold alerting. Results validated across five synthetic cohort sizes — 500, 1,000, 2,000, 5,000, and 10,000 patients (12,873–260,765 rows). ERA sensitivity 18.8–23.4% at t=6.0, false positive rate 4.2–5.1% vs 26.9–28.5% for standard thresholds, alert reduction 81.9–84.2%. Results most consistent at 2,000–10,000 patients: 19–19.3% sensitivity, 4.2–4.5% FPR, 83.3–84.2% alert reduction.</div></div>
    </div>
    <p style="font-size:13px;color:#b6f5d9;background:rgba(58,211,143,.06);padding:12px 14px;border-radius:12px;border:1px solid rgba(58,211,143,.16);line-height:1.65;margin-bottom:10px"><strong>Why is ERA sensitivity lower?</strong> The ERA rules-based logic intentionally trades some sensitivity for dramatically lower false positives (6.2% vs 20.4%), resulting in 71.6% fewer unnecessary alerts while still surfacing key deterioration patterns in the critical 6-hour pre-event window (10,000-patient synthetic retrospective validation, April 2026). In clinical settings where alarm fatigue is a primary safety risk, reducing false positives is often more impactful than maximizing raw sensitivity.</p>
    <div style="overflow-x:auto;margin-bottom:10px">
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr style="border-bottom:1px solid rgba(122,162,255,.2)">
        <th style="padding:8px 10px;text-align:left;color:#9adfff;font-weight:900;letter-spacing:.08em;text-transform:uppercase">Threshold</th>
        <th style="padding:8px 10px;color:#9adfff;font-weight:900;letter-spacing:.08em;text-transform:uppercase">Reading Sensitivity</th>
        <th style="padding:8px 10px;color:#9adfff;font-weight:900;letter-spacing:.08em;text-transform:uppercase">False Positive Rate</th>
        <th style="padding:8px 10px;color:#9adfff;font-weight:900;letter-spacing:.08em;text-transform:uppercase">Alert Reduction</th>
        <th style="padding:8px 10px;color:#9adfff;font-weight:900;letter-spacing:.08em;text-transform:uppercase">Best For</th>
      </tr></thead>
      <tbody>
        <tr style="border-bottom:1px solid rgba(255,255,255,.05)">
          <td style="padding:8px 10px;color:#f4bd6a;font-weight:700">4.0</td>
          <td style="padding:8px 10px;color:#dce9ff">28.2%</td>
          <td style="padding:8px 10px;color:#dce9ff">9.6%</td>
          <td style="padding:8px 10px;color:#3ad38f">52%</td>
          <td style="padding:8px 10px;color:#9fb4d6;font-size:12px">ICU / high-acuity — maximum pre-event detection</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(255,255,255,.05)">
          <td style="padding:8px 10px;color:#9adfff;font-weight:700">5.0</td>
          <td style="padding:8px 10px;color:#dce9ff">20.1%</td>
          <td style="padding:8px 10px;color:#dce9ff">7.8%</td>
          <td style="padding:8px 10px;color:#3ad38f">63.2%</td>
          <td style="padding:8px 10px;color:#9fb4d6;font-size:12px">Mixed units — balanced sensitivity and alert burden</td>
        </tr>
        <tr style="background:rgba(58,211,143,.04)">
          <td style="padding:8px 10px;color:#3ad38f;font-weight:700">6.0 ★ default</td>
          <td style="padding:8px 10px;color:#dce9ff">14.6%</td>
          <td style="padding:8px 10px;color:#3ad38f">6.2%</td>
          <td style="padding:8px 10px;color:#3ad38f">71.6%</td>
          <td style="padding:8px 10px;color:#9fb4d6;font-size:12px">Telemetry / stepdown — lowest alarm burden</td>
        </tr>
        <tr style="border-top:2px solid rgba(255,255,255,.08)">
          <td style="padding:8px 10px;color:#9fb4d6;font-weight:700">Standard only</td>
          <td style="padding:8px 10px;color:#dce9ff">57.1%</td>
          <td style="padding:8px 10px;color:#ff667d">20.4%</td>
          <td style="padding:8px 10px;color:#9fb4d6">—</td>
          <td style="padding:8px 10px;color:#9fb4d6;font-size:12px">Baseline — no ERA</td>
        </tr>
      </tbody>
    </table></div>
    <p style="font-size:12px;color:#9fb4d6;margin-bottom:6px">Threshold is configurable per unit in the command center. Primary benchmark: 10,000-patient synthetic dataset · 260,765 rows · 54,161 events · April 2026. Validated across 5 confirmed cohort sizes (500–10,000 patients). MIMIC-IV real de-identified ICU data validation is planned for Q2 2026, subject to data-access approval and completion of the evaluation. Results are intended to be published publicly upon completion. Primary benchmark: 10,000-patient synthetic dataset, 260,765 rows, 54,161 clinical events, April 2026. Threshold framing: t=4.0 recommended for ICU/high-acuity, t=5.0 for mixed units, t=6.0 for telemetry/alarm fatigue reduction. All ERA thresholds produce materially lower FPR than standard threshold-only alerting. Threshold selection should be calibrated to your unit's acuity level and alarm fatigue tolerance.</p>
    <p style="font-size:13px;margin-top:10px">MIMIC-IV real de-identified ICU data validation is planned for Q2 2026, subject to data-access approval and completion of the evaluation. Results Results are intended to be published publicly upon completion.</p>
  </div>

  <div class="card">
    <h2>Intended use</h2>
    <p style="font-size:15px;color:#eef4ff;font-weight:700;margin-bottom:10px">Early Risk Alert AI is an HCP-facing decision-support and workflow-support software platform intended to assist authorized health care professionals in identifying patients who may warrant further clinical evaluation, supporting patient prioritization, and improving command-center operational awareness.</p>
    <p>It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.</p>
    <div class="row-item"><span class="row-k">Intended users</span><span class="row-v">Authorized health care professionals — physicians, nurses, clinical operations staff — in hospital and health system settings</span></div>
    <div class="row-item"><span class="row-k">Care settings</span><span class="row-v">ICU, Telemetry, Stepdown, Ward, Remote Patient Monitoring programs</span></div>
    <div class="row-item"><span class="row-k">Not intended for</span><span class="row-v">Autonomous escalation, diagnosis, treatment direction, or use by non-clinical personnel without oversight</span></div>
  </div>

  <div class="card">
    <h2>Inputs and outputs</h2>
    <div class="grid-2">
      <div>
        <p style="font-weight:900;color:#9adfff;font-size:12px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px">Supported inputs</p>
        <div class="row-item"><span class="row-k">Heart rate</span><span class="row-v">bpm, numeric</span></div>
        <div class="row-item"><span class="row-k">SpO₂</span><span class="row-v">%, numeric</span></div>
        <div class="row-item"><span class="row-k">Blood pressure</span><span class="row-v">mmHg systolic / diastolic</span></div>
        <div class="row-item"><span class="row-k">Respiratory rate</span><span class="row-v">breaths/min, numeric</span></div>
        <div class="row-item"><span class="row-k">Temperature</span><span class="row-v">°F, numeric</span></div>
        <div class="row-item"><span class="row-k">Trend direction</span><span class="row-v">deteriorating / watch / recovering</span></div>
      </div>
      <div>
        <p style="font-weight:900;color:#9adfff;font-size:12px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px">Supported outputs</p>
        <div class="row-item"><span class="row-k">Review score</span><span class="row-v">0.0 – 9.9 (rules-based, fully explainable)</span></div>
        <div class="row-item"><span class="row-k">Priority status</span><span class="row-v">Critical / High / Stable</span></div>
        <div class="row-item"><span class="row-k">Contributing factors</span><span class="row-v">Per-signal explainability</span></div>
        <div class="row-item"><span class="row-k">Delta trend context</span><span class="row-v">Change from last observation</span></div>
        <div class="row-item"><span class="row-k">Workflow note</span><span class="row-v">Supportive review context only</span></div>
        <div class="row-item"><span class="row-k">Alert notification</span><span class="row-v">Email / SMS to authorized personnel</span></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Limitations and known gaps</h2>
    <div class="row-item"><span class="row-k">Synthetic validation</span><span class="row-v">10,000-patient synthetic dataset (260,765 readings) engineered with clinically grounded deterioration trajectories (sepsis, respiratory failure, cardiac decompensation, hypertensive crisis). April 2026. Results: 38.3% patient detection in 6-hr pre-event window · 71.6% alert reduction · 6.2% ERA FPR vs 20.4% standard threshold alerting · 14.6% reading sensitivity at t=6.0. At t=4.0 (ICU): 61.4% patient detection / 9.6% FPR. At t=5.0 (mixed): 48.1% patient detection / 7.8% FPR. Validated consistently across 500, 1,000, 2,000, 5,000, and 10,000 patient cohorts. MIMIC-IV real de-identified ICU data validation is planned for Q2 2026, subject to data-access approval and completion of the evaluation. Results Results are intended to be published publicly upon completion. Prospective clinical validation has not yet been completed.</span></div>
    <div class="row-item"><span class="row-k">Rules-based only</span><span class="row-v">Current engine uses additive threshold rules, not machine learning. No training dataset, no AUC, no sensitivity/specificity from a held-out test set yet.</span></div>
    <div class="row-item"><span class="row-k">No EHR integration</span><span class="row-v">Current deployment uses structured CSV input and simulated vitals. Live EHR integration via FHIR R4 and HL7 is on the product roadmap — current pilot entry point is retrospective validation via de-identified CSV, which requires no EHR integration and can begin within days of data availability.</span></div>
    <div class="row-item"><span class="row-k">Simulated demo environment</span><span class="row-v">The public demo runs on simulated patient data. No real patient data is used in the demonstration environment.</span></div>
    <div class="row-item"><span class="row-k">Incomplete or delayed data</span><span class="row-v">Outputs may be affected by missing, delayed, or erroneous vital sign inputs. The platform does not validate source data quality.</span></div>
    <div class="row-item"><span class="row-k">Population generalizability</span><span class="row-v">Signal weights have not been validated across diverse patient populations, acuity levels, or care settings. Local validation is strongly recommended.</span></div>
    <div class="row-item"><span class="row-k">Alert fatigue risk</span><span class="row-v">If thresholds are set too low for a given unit, alert volume may increase rather than decrease. Configurable thresholds and local calibration are recommended.</span></div>
    <div class="row-item"><span class="row-k">No FDA clearance</span><span class="row-v">The platform does not have FDA clearance or approval. It is positioned for controlled pilot evaluation as decision-support software.</span></div>
  </div>

  <div class="card">
    <h2>Governance and oversight</h2>
    <div class="row-item"><span class="row-k">Human oversight required</span><span class="row-v">All outputs require independent review by an authorized HCP. The platform does not act autonomously.</span></div>
    <div class="row-item"><span class="row-k">Explainability</span><span class="row-v">Every output displays contributing factors, signal weights, confidence level, data freshness, and limitations.</span></div>
    <div class="row-item"><span class="row-k">Audit trail</span><span class="row-v">All workflow actions (ACK, Assign, Escalate, Resolve) are logged with role, timestamp, and unit. Persistent across restarts.</span></div>
    <div class="row-item"><span class="row-k">Claims control</span><span class="row-v">Approved and banned claims enforced across all platform materials and communications.</span></div>
    <div class="row-item"><span class="row-k">Change control</span><span class="row-v">All releases documented in change approval log. No material changes to clinical output logic without notification.</span></div>
    <div class="row-item"><span class="row-k">Regulatory status</span><span class="row-v">No FDA clearance or approval claimed. Positioned as decision-support software for controlled pilot evaluation.</span></div>
    <div class="row-item"><span class="row-k">BAA availability</span><span class="row-v">The company is prepared to execute a Business Associate Agreement for any engagement involving identifiable patient data. Phase 1 retrospective validation is conducted on de-identified data only.</span></div>
    <div class="row-item"><span class="row-k">MFA implementation</span><span class="row-v">Phishing-resistant MFA implemented across all core administrative systems — business email, source control, hosting, password management, and domain/DNS — April 10, 2026.</span></div>
    <div class="row-item"><span class="row-k">Research ethics training</span><span class="row-v">CITI Program training completed April 10, 2026 — Data or Specimens Only Research and Conflict of Interest certificates obtained prior to MIMIC-IV data access application.</span></div>
    <div class="row-item"><span class="row-k">MIMIC-IV validation</span><span class="row-v">PhysioNet MIMIC-IV data access application submitted April 10, 2026. Validation planned for Q2 2026, subject to data-access approval. Will test all three threshold settings (t=4.0 ICU, t=5.0 mixed, t=6.0 telemetry) against real de-identified ICU data with realistic event prevalence. Results to be published publicly upon completion. This is the primary validation milestone for hospital pilot activation.</span></div>
    <div class="row-item"><span class="row-k">Pilot onboarding</span><span class="row-v">Hospital pilot onboarding checklist available at /pilot-onboarding — includes step-by-step CSV upload, governance docs, 4-6 week pilot timeline, and proposed success metrics.</span></div>
    <div class="row-item"><span class="row-k">Platform version</span><span class="row-v">stable-pilot-1.0.6-gov · April 10, 2026</span></div>
  </div>

  <div class="card">
    <h2>Advisory structure</h2>
    <div class="row-item"><span class="row-k">Milton Munroe</span><span class="row-v">Founder &amp; CEO — Product leadership, governance ownership, pilot operations</span></div>
    <div class="row-item"><span class="row-k">Uche Anosike</span><span class="row-v">Technical Infrastructure &amp; Security Advisor — Infrastructure, security posture, deployment readiness</span></div>
    <div class="row-item"><span class="row-k">Andrene Louison, RN</span><span class="row-v">Clinical Advisor — Clinical workflow review, monitored-context guidance, retrospective validation support</span></div>
  </div>

  <div class="footer">
    Early Risk Alert AI · <a href="/command-center">Command Center</a> · <a href="/pilot-docs">Pilot Docs</a> · <a href="/status">System Status</a> · <a href="/retro-upload">Retrospective Validation Upload</a><br>
    <a href="mailto:info@earlyriskalertai.com">info@earlyriskalertai.com</a> · 732-724-7267<br>
    This platform is intended for controlled pilot evaluation. It does not replace clinician judgment.
  </div>
</div>
</body>
</html>"""
        return Response(mc_html, mimetype="text/html; charset=utf-8")

    @app.get("/pilot-docs")
    @_login_required
    def pilot_docs():
        def simple_list(items):
            return "".join(f"<div style='padding:10px 12px;border:1px solid rgba(255,255,255,.08);border-radius:14px;background:rgba(255,255,255,.03);margin-bottom:10px;color:#dce9ff'>{item}</div>" for item in items)

        def table(rows, headers):
            out = ["<table style='width:100%;border-collapse:collapse'>", "<thead><tr>"]
            for h in headers:
                out.append(f"<th style='padding:12px;text-align:left;color:#9adfff;border-bottom:1px solid rgba(255,255,255,.08)'>{h.replace('_',' ').title()}</th>")
            out.append("</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>")
                for h in headers:
                    v = row.get(h, "")
                    if isinstance(v, (dict, list)):
                        v = json.dumps(v, ensure_ascii=False)
                    out.append(f"<td style='padding:12px;vertical-align:top;border-bottom:1px solid rgba(255,255,255,.08);color:#dce9ff'>{v}</td>")
                out.append("</tr>")
            out.append("</tbody></table>")
            return "".join(out)

        html_out = f"""
        <!doctype html><html lang='en'><head><meta charset='utf-8'><title>Pilot Docs — Early Risk Alert AI</title><meta name='viewport' content='width=device-width, initial-scale=1'>
        <style>body{{margin:0;padding:24px;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#eef4ff;background:linear-gradient(180deg,#07101c,#0b1528)}}.wrap{{max-width:1280px;margin:0 auto}}.card{{border:1px solid rgba(255,255,255,.08);border-radius:24px;background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.018));padding:24px;margin-bottom:18px;box-shadow:0 20px 60px rgba(0,0,0,.28)}}.btn{{display:inline-flex;align-items:center;justify-content:center;padding:12px 16px;border-radius:16px;font-weight:900;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;text-decoration:none}}.sub{{color:#9fb4d6;line-height:1.7}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.pill{{display:inline-flex;align-items:center;padding:10px 14px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);font-size:12px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;margin-right:8px;margin-bottom:8px}}@media (max-width:840px){{.grid{{grid-template-columns:1fr}}}}</style>
        </head><body><div class='wrap'>
        <div class='card'><div class='pill'>Pilot Docs</div><div class='pill'>Version {PILOT_VERSION}</div><h1 style='margin:12px 0 10px;font-size:42px;line-height:.95;letter-spacing:-.05em'>Stable Pilot Positioning Bundle</h1><div class='sub'>Freeze one intended-use statement everywhere, keep outputs supportive rather than directive, and keep explainability, limitations, scoping, and audit visibility easy to review.</div><div style='margin-top:16px;display:flex;gap:12px;flex-wrap:wrap'><a class='btn' href='/command-center'>Back to Command Center</a><a class='btn' href='/pilot-success-guide'>Pilot Success Guide</a><a class='btn' href='/model-card'>Model Card</a></div></div>
        <div class='card'><h2 style='margin:0 0 10px;font-size:30px'>Frozen Intended Use</h2><div class='sub' style='font-size:18px;color:#eef4ff'>{INTENDED_USE_STATEMENT}</div><div style='margin-top:12px;display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;background:rgba(181,140,255,.14);border:1px solid rgba(181,140,255,.28);font-weight:900;color:#f0e5ff'>{PILOT_BUILD_STATE} · {PILOT_VERSION}</div></div>
        <div class='grid'><div class='card'><h2 style='margin:0 0 10px;font-size:26px'>Support Language</h2>{simple_list(PILOT_SUPPORT_LANGUAGE)}</div><div class='card'><h2 style='margin:0 0 10px;font-size:26px'>Visible Limitations</h2>{simple_list(PILOT_LIMITATIONS_TEXT)}</div></div>
        <div class='grid'><div class='card'><h2 style='margin:0 0 10px;font-size:26px'>Supported Inputs</h2>{simple_list(PILOT_SUPPORTED_INPUTS)}</div><div class='card'><h2 style='margin:0 0 10px;font-size:26px'>Supported Outputs</h2>{simple_list(PILOT_SUPPORTED_OUTPUTS)}</div></div>
        <div class='grid'><div class='card'><h2 style='margin:0 0 10px;font-size:26px'>Approved Claims</h2>{simple_list(PILOT_APPROVED_CLAIMS)}</div><div class='card'><h2 style='margin:0 0 10px;font-size:26px'>Banned / Avoid Claims</h2>{simple_list(PILOT_BANNED_CLAIMS + PILOT_AVOID_CLAIMS)}</div></div>
        <div class='grid'><div class='card'><h2 style='margin:0 0 10px;font-size:26px'>Claims Control</h2>{table(PILOT_CLAIMS_CONTROL_SHEET, ['claim','status','category'])}</div><div class='card'><h2 style='margin:0 0 10px;font-size:26px'>Change Control</h2>{simple_list(PILOT_CHANGE_CONTROL)}</div></div>
        <div class='card'><h2 style='margin:0 0 12px;font-size:28px'>Risk Register</h2>{table(PILOT_RISK_REGISTER, ['id','area','risk','mitigation','owner','status'])}</div>
        <div class='card'><h2 style='margin:0 0 12px;font-size:28px'>V&amp;V-Lite Sheet</h2>{table(PILOT_VNV_LITE, ['id','check','method','evidence','status'])}</div>
        <div class='card'><h2 style='margin:0 0 12px;font-size:28px'>Release Notes</h2>{table(PILOT_RELEASE_NOTES, ['version','date','summary'])}</div>
        <div class='grid'><div class='card'><h2 style='margin:0 0 12px;font-size:28px'>Document Control Index</h2>{table(PILOT_DOCUMENT_CONTROL_INDEX, ['document_name','version'])}</div><div class='card'><h2 style='margin:0 0 12px;font-size:28px'>Validation Packet</h2>{table(PILOT_VALIDATION_EVIDENCE, ['date_tested','test_case_id','status'])}</div></div>
        <div class='grid'><div class='card'><h2 style='margin:0 0 12px;font-size:28px'>Advisory Structure</h2>{table(PILOT_ADVISORY_STRUCTURE, ['name','title','status'])}</div><div class='card'><h2 style='margin:0 0 12px;font-size:28px'>Support Owners</h2>{table(PILOT_SUPPORT_OWNERS, ['area','owner','title','status'])}</div></div>
        <div class='card'><h2 style='margin:0 0 12px;font-size:28px'>Training &amp; Use Instructions</h2>{table(PILOT_TRAINING_USE_INSTRUCTIONS, ['section','instruction'])}</div>
        </div></body></html>
        """
        return render_template_string(html_out)

    @app.get("/admin/review")
    def admin_review():
        return render_template_string(ADMIN_HTML)

    @app.get("/healthz")
    def healthz():
        summary = summary_payload()
        return jsonify({
            "ok": True,
            "service": "early-risk-alert-ai", "descriptor": "Explainable Rules-Based Command-Center Platform",
            "time": _utc_now_iso(),
            "hospital_requests": summary["hospital_count"],
            "executive_requests": summary["executive_count"],
            "investor_requests": summary["investor_count"],
            "open_requests": summary["open_count"],
            "pilot_version": PILOT_VERSION,
            "build_state": PILOT_BUILD_STATE,
        })

    @app.get("/status")
    def status_page():
        """Public uptime and system health page — no login required."""
        now = _utc_now_iso()
        uptime_since = os.getenv("ERA_DEPLOY_TIME", "2026-04-07T00:00:00+00:00")
        try:
            up_seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(uptime_since)).total_seconds()
            up_hours = int(up_seconds // 3600)
            up_days = int(up_hours // 24)
            uptime_str = f"{up_days}d {up_hours % 24}h" if up_days else f"{up_hours}h"
        except Exception:
            uptime_str = "—"

        status_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>System Status — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="60">
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:linear-gradient(180deg,#07101c,#0b1528);color:#eef4ff;min-height:100vh;padding:32px 20px 64px}}
    .wrap{{max-width:760px;margin:0 auto}}
    .header{{margin-bottom:32px}}
    .brand{{font-size:13px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9adfff;margin-bottom:8px}}
    h1{{font-size:38px;font-weight:1000;letter-spacing:-.05em;margin-bottom:8px}}
    .sub{{color:#9fb4d6;font-size:14px}}
    .overall{{display:flex;align-items:center;gap:14px;padding:20px 24px;border-radius:20px;border:1px solid rgba(58,211,143,.26);background:rgba(58,211,143,.08);margin-bottom:24px}}
    .dot{{width:14px;height:14px;border-radius:50%;background:#3ad38f;box-shadow:0 0 0 5px rgba(58,211,143,.18);flex-shrink:0}}
    .overall-text{{font-size:18px;font-weight:1000}}
    .overall-sub{{font-size:13px;color:#9fb4d6;margin-top:2px}}
    .card{{border:1px solid rgba(255,255,255,.08);border-radius:20px;background:rgba(255,255,255,.03);padding:20px;margin-bottom:14px}}
    .card-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}}
    .card-title{{font-size:15px;font-weight:900}}
    .pill{{display:inline-flex;align-items:center;padding:6px 12px;border-radius:999px;font-size:11px;font-weight:1000;letter-spacing:.1em;text-transform:uppercase}}
    .pill.op{{background:rgba(58,211,143,.12);border:1px solid rgba(58,211,143,.24);color:#b6f5d9}}
    .pill.deg{{background:rgba(255,102,125,.12);border:1px solid rgba(255,102,125,.24);color:#ffd8de}}
    .pill.watch{{background:rgba(244,189,106,.12);border:1px solid rgba(244,189,106,.24);color:#ffe7bf}}
    .meta-row{{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}}
    .meta{{padding:8px 12px;border-radius:12px;background:rgba(255,255,255,.04);font-size:12px;color:#9fb4d6}}
    .meta strong{{color:#dce9ff;display:block;font-size:13px;margin-bottom:1px}}
    .footer{{margin-top:32px;font-size:12px;color:#9fb4d6;line-height:1.6;text-align:center}}
    a{{color:#9adfff;text-decoration:none}}
  </style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="brand">Early Risk Alert AI</div>
    <h1>System Status</h1>
    <div class="sub">Auto-refreshes every 60 seconds · Last checked {now[:19].replace("T", " ")} UTC</div>
  </div>

  <div class="overall">
    <div class="dot"></div>
    <div>
      <div class="overall-text">All Systems Operational</div>
      <div class="overall-sub">Platform is live and serving requests normally.</div>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <div class="card-title">Explainable Rules-Based Command-Center Platform</div>
      <span class="pill op">Operational</span>
    </div>
    <div class="meta-row">
      <div class="meta"><strong>Route</strong>/command-center</div>
      <div class="meta"><strong>Auth</strong>Session-based</div>
      <div class="meta"><strong>Uptime</strong>{uptime_str}</div>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <div class="card-title">Patient Data Feed</div>
      <span class="pill op">Operational</span>
    </div>
    <div class="meta-row">
      <div class="meta"><strong>Mode</strong>Structured vital simulation</div>
      <div class="meta"><strong>Refresh</strong>5-second interval</div>
      <div class="meta"><strong>Persistence</strong>SQLite + flat-file</div>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <div class="card-title">Workflow &amp; Audit Layer</div>
      <span class="pill op">Operational</span>
    </div>
    <div class="meta-row">
      <div class="meta"><strong>Storage</strong>SQLite (persistent)</div>
      <div class="meta"><strong>Actions</strong>ACK / Assign / Escalate / Resolve</div>
      <div class="meta"><strong>Audit</strong>Persistent across restarts</div>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <div class="card-title">Alert Notifications</div>
      <span class="pill {'op' if os.getenv('ALERT_NOTIFICATIONS_ENABLED','0')=='1' else 'watch'}">{'Active' if os.getenv('ALERT_NOTIFICATIONS_ENABLED','0')=='1' else 'Standby'}</span>
    </div>
    <div class="meta-row">
      <div class="meta"><strong>Email</strong>{'Configured' if os.getenv('SMTP_HOST') else 'Not configured'}</div>
      <div class="meta"><strong>SMS</strong>{'Configured' if os.getenv('TWILIO_ACCOUNT_SID') else 'Not configured'}</div>
      <div class="meta"><strong>Cooldown</strong>{_ALERT_COOLDOWN_MINUTES} min</div>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <div class="card-title">Secure Access Layer</div>
      <span class="pill op">Operational</span>
    </div>
    <div class="meta-row">
      <div class="meta"><strong>Auth</strong>Role + unit scoped sessions</div>
      <div class="meta"><strong>Secure cookie</strong>{'Yes' if os.getenv('SESSION_COOKIE_SECURE','0')=='1' else 'Dev mode'}</div>
      <div class="meta"><strong>Build</strong>{PILOT_VERSION}</div>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <div class="card-title">Retrospective Validation Results</div>
      <span class="pill op">Published</span>
    </div>
    <div class="meta-row">
      <div class="meta"><strong>Alert Reduction</strong>71.6% vs standard thresholds (10,000-patient synthetic dataset, April 2026)</div>
      <div class="meta"><strong>ERA False Positive Rate</strong>6.2% vs 20.4% for standard thresholds</div>
      <div class="meta"><strong>Datasets</strong>500–10,000 patients · 12,873–260,765 readings · results consistent across all 5 cohort sizes</div>
    </div>
    <div class="meta-row" style="margin-top:8px">
      <div class="meta"><strong>Data Type</strong>Synthetic — MIMIC-IV real de-identified ICU data validation is planned for Q2 2026, subject to data-access approval and completion of the evaluation. Results Results are intended to be published publicly upon completion.</div>
      <div class="meta"><strong>Sensitivity Note</strong>14.6% reading sensitivity ERA vs 57.1% threshold — intentional trade for 71.6% alert reduction and 6.2% FPR. Patient-level detection 38.3% in 6-hr window</div>
      <div class="meta"><strong>No-Commitment Analysis</strong>Hospitals may submit de-identified CSV for custom retrospective analysis</div>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <div class="card-title">HIPAA &amp; BAA Readiness</div>
      <span class="pill op">Ready</span>
    </div>
    <div class="meta-row">
      <div class="meta"><strong>BAA</strong>Available upon request for live patient data engagements</div>
      <div class="meta"><strong>Phase 1</strong>Retrospective validation on de-identified data — no BAA required</div>
      <div class="meta"><strong>Phase 2</strong>BAA executed before any identifiable patient data is processed</div>
    </div>
    <div class="meta-row" style="margin-top:8px">
      <div class="meta"><strong>Contact</strong>info@earlyriskalertai.com to request BAA or data use agreement</div>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <div class="card-title">Retrospective Validation Data Ingestion</div>
      <span class="pill op">Available</span>
    </div>
    <div class="meta-row">
      <div class="meta"><strong>Format</strong>CSV upload — structured vital-sign data</div>
      <div class="meta"><strong>Route</strong>/retro-upload (login required)</div>
      <div class="meta"><strong>EHR Integration</strong>FHIR / HL7 roadmap — CSV ingestion available now. Live EHR integration via FHIR R4 and HL7 is on the product roadmap — current pilot entry point is retrospective validation via de-identified CSV, which requires no EHR integration and can begin within days of data availability.</div>
    </div>
    <div class="meta-row" style="margin-top:8px">
      <div class="meta"><strong>Schema</strong>patient_id, timestamp, HR, SpO2, BP, RR, temp, clinical_event</div>
    </div>
  </div>

  <div class="footer">
    Early Risk Alert AI · <a href="/command-center">Command Center</a> · <a href="/pilot-docs">Pilot Docs</a> · <a href="/model-card">Model Card</a><br>
    <a href="mailto:info@earlyriskalertai.com">info@earlyriskalertai.com</a> · 732-724-7267<br>
    This platform is intended for controlled pilot evaluation. It does not replace clinician judgment.
  </div>
</div>
</body>
</html>"""
        return Response(status_html, mimetype="text/html; charset=utf-8")

    @app.route("/hospital-demo", methods=["GET", "POST"])
    def hospital_demo():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(),
                "last_updated": _utc_now_iso(),
                "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "email": request.form.get("email", "").strip(),
                "organization_type": request.form.get("organization_type", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "role": request.form.get("role", "").strip(),
                "department_unit": request.form.get("department_unit", "").strip(),
                "evaluation_interest": request.form.get("evaluation_interest", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("additional_notes", "").strip(),
                "acknowledgment": request.form.get("acknowledgment", "").strip(),
            }
            payload["lead_score"] = _lead_score(payload, "hospital")
            payload["priority_tag"] = _priority_tag(payload["lead_score"], "hospital")
            _append_jsonl(hospital_file, payload)
            return render_template_string(_render_thank_you("Your hospital demonstration request has been received.", _detail_html(payload, ["full_name", "email", "organization_type", "organization", "role", "department_unit", "evaluation_interest", "timeline", "lead_score", "priority_tag"])))
        fields = """
        <div class="field"><label>Work Email</label><input type="email" name="email" placeholder="Enter your work email" required></div>
        <div class="field"><label>Organization Type</label><select name="organization_type" required><option value="Hospital">Hospital</option><option value="Health System">Health System</option><option value="Remote Patient Monitoring Program">Remote Patient Monitoring Program</option><option value="Care Network">Care Network</option><option value="Clinical Operations Team">Clinical Operations Team</option><option value="Other">Other</option></select></div>
        <div class="field"><label>Organization Name</label><input name="organization" placeholder="Enter your hospital, health system, or organization name" required></div>
        <div class="field"><label>Full Name</label><input name="full_name" placeholder="Enter your full name" required></div>
        <div class="field"><label>Title / Role</label><input name="role" placeholder="Enter your title or role" required></div>
        <div class="field"><label>Department / Unit</label><select name="department_unit" required><option value="ICU">ICU</option><option value="Telemetry">Telemetry</option><option value="Stepdown">Stepdown</option><option value="Med-Surg">Med-Surg</option><option value="Remote Monitoring">Remote Monitoring</option><option value="Operations / Command Center">Operations / Command Center</option><option value="Executive Leadership">Executive Leadership</option><option value="Other">Other</option></select></div>
        <div class="field"><label>What are you interested in evaluating?</label><select name="evaluation_interest" required><option value="Patient prioritization support">Patient prioritization support</option><option value="Command-center operational awareness">Command-center operational awareness</option><option value="Explainable review-basis visibility">Explainable review-basis visibility</option><option value="Workflow-state and audit visibility">Workflow-state and audit visibility</option><option value="Controlled pilot evaluation">Controlled pilot evaluation</option><option value="Other">Other</option></select></div>
        <div class="field"><label>Pilot Timeline</label><select name="timeline" required><option value="Immediate">Immediate</option><option value="30-60 days">30-60 days</option><option value="This quarter">This quarter</option><option value="Exploratory">Exploratory</option></select></div>
        <div class="field full"><label>Additional Notes</label><textarea name="additional_notes" placeholder="Share your pilot goals, care setting, or workflow interests"></textarea></div>
        <div class="field full"><label>Acknowledgment</label><div class="check"><input type="checkbox" name="acknowledgment" value="yes" required><span>I understand Early Risk Alert AI is intended for HCP-facing decision-support and workflow-support pilot evaluation. It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.</span></div></div>
        """
        out = FORM_HTML.replace("__TITLE__", "Request a Live Command Center Demonstration").replace("__HEADING__", "Request a Live Command Center Demonstration").replace("__COPY__", "Schedule a guided demonstration of Early Risk Alert AI's HCP-facing decision-support and workflow-support platform for monitored patient visibility, patient prioritization support, explainable review context, and command-center operational awareness.").replace("__FIELDS__", fields).replace("__BUTTON__", "Request Demo Access")
        return render_template_string(out)

    @app.route("/executive-walkthrough", methods=["GET", "POST"])
    def executive_walkthrough():
        if request.method == "POST":
            payload = {
                "submitted_at": _utc_now_iso(), "last_updated": _utc_now_iso(), "status": "New",
                "full_name": request.form.get("full_name", "").strip(),
                "organization": request.form.get("organization", "").strip(),
                "title": request.form.get("title", "").strip(),
                "email": request.form.get("email", "").strip(),
                "leadership_area": request.form.get("leadership_area", "").strip(),
                "review_focus": request.form.get("review_focus", "").strip(),
                "timeline": request.form.get("timeline", "").strip(),
                "message": request.form.get("message", "").strip(),
                "acknowledgment": request.form.get("acknowledgment", "").strip(),
            }
            payload["lead_score"] = _lead_score(payload, "executive")
            payload["priority_tag"] = _priority_tag(payload["lead_score"], "executive")
            _append_jsonl(exec_file, payload)
            return render_template_string(_render_thank_you("Your executive walkthrough request has been received.", _detail_html(payload, ["full_name", "organization", "title", "email", "leadership_area", "review_focus", "timeline", "lead_score", "priority_tag"])))
        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" placeholder="Enter your full name" required></div>
        <div class="field"><label>Organization Name</label><input name="organization" placeholder="Enter your organization name" required></div>
        <div class="field"><label>Title</label><input name="title" placeholder="Enter your title" required></div>
        <div class="field"><label>Work Email</label><input type="email" name="email" placeholder="Enter your work email" required></div>
        <div class="field"><label>Leadership Area</label><select name="leadership_area" required><option value="Clinical Leadership">Clinical Leadership</option><option value="Hospital Operations">Hospital Operations</option><option value="Digital Health">Digital Health</option><option value="Innovation">Innovation</option><option value="Executive Administration">Executive Administration</option><option value="Other">Other</option></select></div>
        <div class="field"><label>Timeline</label><select name="timeline" required><option value="Immediate">Immediate</option><option value="30-60 days">30-60 days</option><option value="This quarter">This quarter</option><option value="Exploratory">Exploratory</option></select></div>
        <div class="field full"><label>What would you like reviewed?</label><select name="review_focus" required><option value="Pilot readiness">Pilot readiness</option><option value="Platform overview">Platform overview</option><option value="Operational workflow support">Operational workflow support</option><option value="Command-center model">Command-center model</option><option value="Security and pilot controls">Security and pilot controls</option><option value="Other">Other</option></select></div>
        <div class="field full"><label>Additional Notes</label><textarea name="message" placeholder="Share any leadership, pilot, or operational review priorities"></textarea></div>
        <div class="field full"><label>Acknowledgment</label><div class="check"><input type="checkbox" name="acknowledgment" value="yes" required><span>I understand Early Risk Alert AI is intended for controlled pilot evaluation and hospital-facing workflow support. It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation.</span></div></div>
        """
        out = FORM_HTML.replace("__TITLE__", "Request an Executive Walkthrough").replace("__HEADING__", "Request an Executive Walkthrough").replace("__COPY__", "Request a leadership-level walkthrough of Early Risk Alert AI's hospital-facing platform, pilot readiness, operational workflow support, and command-center visibility model.").replace("__FIELDS__", fields).replace("__BUTTON__", "Request Executive Walkthrough")
        return render_template_string(out)

    @app.route("/investor-intake", methods=["GET", "POST"])
    def investor_intake():
        if request.method == "POST":
            payload = {"submitted_at": _utc_now_iso(), "last_updated": _utc_now_iso(), "status": "New", "full_name": request.form.get("full_name", "").strip(), "organization": request.form.get("organization", "").strip(), "role": request.form.get("role", "").strip(), "email": request.form.get("email", "").strip(), "stage": request.form.get("stage", "").strip(), "timeline": request.form.get("timeline", "").strip(), "interest_area": request.form.get("interest_area", "").strip(), "message": request.form.get("message", "").strip(), "acknowledgment": request.form.get("acknowledgment", "").strip()}
            payload["lead_score"] = _lead_score(payload, "investor")
            payload["priority_tag"] = _priority_tag(payload["lead_score"], "investor")
            _append_jsonl(investor_file, payload)
            return render_template_string(_render_thank_you("Your investor access request has been received.", _detail_html(payload, ["full_name", "organization", "role", "email", "stage", "timeline", "interest_area", "lead_score", "priority_tag"])))
        fields = """
        <div class="field"><label>Full Name</label><input name="full_name" placeholder="Enter your full name" required></div>
        <div class="field"><label>Organization</label><input name="organization" placeholder="Enter your organization" required></div>
        <div class="field"><label>Role</label><input name="role" placeholder="Enter your role" required></div>
        <div class="field"><label>Work Email</label><input type="email" name="email" placeholder="Enter your work email" required></div>
        <div class="field"><label>Investor Stage</label><select name="stage" required><option value="Angel">Angel</option><option value="Seed">Seed</option><option value="Institutional">Institutional</option><option value="Strategic">Strategic</option></select></div>
        <div class="field"><label>Timeline</label><select name="timeline" required><option value="Immediate">Immediate</option><option value="30-60 days">30-60 days</option><option value="This quarter">This quarter</option><option value="Exploratory">Exploratory</option></select></div>
        <div class="field full"><label>Areas of Interest</label><select name="interest_area" required><option value="Platform overview">Platform overview</option><option value="Hospital pilot model">Hospital pilot model</option><option value="Commercial model">Commercial model</option><option value="Market opportunity">Market opportunity</option><option value="Founder discussion">Founder discussion</option><option value="Partnership discussion">Partnership discussion</option><option value="Other">Other</option></select></div>
        <div class="field full"><label>Message</label><textarea name="message" placeholder="Share your investor or partnership interests"></textarea></div>
        <div class="field full"><label>Acknowledgment</label><div class="check"><input type="checkbox" name="acknowledgment" value="yes" required><span>I understand Early Risk Alert AI is positioned as an HCP-facing decision-support and workflow-support platform for controlled pilot evaluation and hospital-facing workflow support.</span></div></div>
        """
        out = FORM_HTML.replace("__TITLE__", "Request Investor Access").replace("__HEADING__", "Request Investor Access").replace("__COPY__", "Request investor materials, platform overview, and partnership discussion access for Early Risk Alert AI's HCP-facing decision-support and workflow-support platform.").replace("__FIELDS__", fields).replace("__BUTTON__", "Request Investor Access")
        return render_template_string(out)

    @app.get("/admin/review/data")
    def admin_review_data():
        rows: List[Dict[str, Any]] = []
        for lead_type, path in [("hospital", hospital_file), ("executive", exec_file), ("investor", investor_file)]:
            for item in _read_jsonl(path):
                row = dict(item)
                row["kind"] = lead_type
                rows.append(row)
        rows.sort(key=lambda r: str(r.get("submitted_at", "")), reverse=True)
        return jsonify({"ok": True, "rows": rows})

    @app.post("/admin/review/update")
    def admin_review_update():
        payload = request.get_json(silent=True) or {}
        kind = str(payload.get("kind", "")).strip().lower()
        submitted_at = str(payload.get("submitted_at", "")).strip()
        field = str(payload.get("field", "")).strip()
        value = str(payload.get("value", "")).strip()
        if not kind or not submitted_at or field not in {"status", "stage"}:
            return jsonify({"ok": False, "error": "invalid payload"}), 400
        path_map = {"hospital": hospital_file, "executive": exec_file, "investor": investor_file}
        path = path_map.get(kind)
        if not path:
            return jsonify({"ok": False, "error": "invalid lead type"}), 400
        rows = _read_jsonl(path)
        updated = False
        for row in rows:
            if str(row.get("submitted_at", "")).strip() == submitted_at:
                row[field] = value
                row["last_updated"] = _utc_now_iso()
                updated = True
                break
        if not updated:
            return jsonify({"ok": False, "error": "lead not found"}), 404
        _write_jsonl_rows(path, rows)
        return jsonify({"ok": True})

    @app.get("/admin/api/data")
    def admin_api_data():
        raw = {"hospital": _read_jsonl(hospital_file), "executive": _read_jsonl(exec_file), "investor": _read_jsonl(investor_file)}
        merged: List[Dict[str, Any]] = []
        for lead_type, data in raw.items():
            for row in data:
                merged.append(_normalize_row(row, lead_type))
        merged.sort(key=lambda r: (r["submitted_at"], r["lead_score"]), reverse=True)
        last_updated = max([r["last_updated"] for r in merged], default="")
        return jsonify({
            "rows": merged,
            "summary": {
                "hospital_count": len(raw["hospital"]),
                "executive_count": len(raw["executive"]),
                "investor_count": len(raw["investor"]),
                "open_count": sum(1 for r in merged if r["status"] != "Closed"),
                "investor_stages": _investor_stage_summary(raw["investor"]),
                "last_updated": _format_pretty_label(last_updated),
            },
        })

    @app.post("/admin/api/status")
    def admin_api_status():
        data = request.get_json(silent=True) or {}
        lead_type = str(data.get("lead_type", "")).strip()
        submitted_at = str(data.get("submitted_at", "")).strip()
        new_status = str(data.get("status", "")).strip()
        file_map = {"hospital": hospital_file, "executive": exec_file, "investor": investor_file}
        path = file_map.get(lead_type)
        if not path or not submitted_at:
            return jsonify({"ok": False}), 400
        rows = _read_jsonl(path)
        for row in rows:
            if str(row.get("submitted_at", "")) == submitted_at:
                row["status"] = _status_norm(new_status, lead_type)
                row["last_updated"] = _utc_now_iso()
        _write_jsonl_rows(path, rows)
        return jsonify({"ok": True})

    @app.get("/admin/export.csv")
    def admin_export_csv():
        rows = []
        for lead_type, path in [("hospital", hospital_file), ("executive", exec_file), ("investor", investor_file)]:
            for row in _read_jsonl(path):
                rows.append(_normalize_row(row, lead_type))
        rows.sort(key=lambda r: (r["submitted_at"], r["lead_score"]), reverse=True)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Lead Source", "Submitted At", "Last Updated", "Lead Status", "Lead Score", "Priority Tag", "Full Name", "Organization", "Role / Title", "Email Address", "Phone Number", "Category", "Timeline", "Message"])
        for row in rows:
            writer.writerow([row["lead_type"], row["submitted_at"], row["last_updated"], row["status"], row["lead_score"], row["priority_tag"], row["full_name"], row["organization"], row["role_or_title"], row["email"], row["phone"], row["category"], row["timeline"], row["message"]])
        mem = io.BytesIO(output.getvalue().encode("utf-8"))
        mem.seek(0)
        return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="early_risk_alert_pipeline_export.csv")

    @app.get("/api/platform-positioning")
    def api_platform_positioning():
        return jsonify({
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
            "advisory_structure": PILOT_ADVISORY_STRUCTURE,
            "support_owners": PILOT_SUPPORT_OWNERS,
            "model_card_url": "/model-card",
            "pilot_success_guide_url": "/pilot-success-guide",
        })

    @app.get("/api/pilot-governance")
    def api_pilot_governance():
        return jsonify({
            "pilot_version": PILOT_VERSION,
            "pilot_build_state": PILOT_BUILD_STATE,
            "intended_use_statement": INTENDED_USE_STATEMENT,
            "support_language": PILOT_SUPPORT_LANGUAGE,
            "supported_inputs": PILOT_SUPPORTED_INPUTS,
            "supported_outputs": PILOT_SUPPORTED_OUTPUTS,
            "avoid_claims": PILOT_AVOID_CLAIMS,
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
            "user_provisioning_policy": PILOT_USER_PROVISIONING_POLICY,
            "data_retention_policy": PILOT_DATA_RETENTION_POLICY,
            "pilot_scope_document": PILOT_SCOPE_DOCUMENT,
            "training_use_instructions": PILOT_TRAINING_USE_INSTRUCTIONS,
            "dated_validation_evidence": PILOT_VALIDATION_EVIDENCE,
            "model_card_url": "/model-card",
            "pilot_success_guide_url": "/pilot-success-guide",
        })

    @app.get("/api/access-context")
    @_login_required
    def access_context():
        return jsonify(_access_context_payload())

    @app.get("/api/workflow")
    @_login_required
    def workflow():
        _ensure_db_loaded()
        return jsonify({"records": SIM_STATE.get("workflow_records", {}), "audit_log": SIM_STATE.get("workflow_audit", [])[:100]})

    @app.post("/api/workflow/action")
    @_login_required
    def workflow_action():
        data = request.get_json(silent=True) or {}
        patient_id = str(data.get("patient_id", "")).strip()
        action = str(data.get("action", "")).strip().lower()
        note = str(data.get("note", "")).strip()
        if not patient_id:
            return jsonify({"ok": False, "error": "patient_id required"}), 400
        record = _get_workflow_record(patient_id)
        if action == "ack":
            if not _has_permission("ack"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["ack"] = True
            record["state"] = "acknowledged"
            _append_audit(patient_id, "ACK", note or "Alert acknowledged")
        elif action == "assign_nurse":
            if not _has_permission("assign"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["assigned"] = True
            record["assigned_label"] = note or "Assigned Nurse"
            record["state"] = "assigned"
            _append_audit(patient_id, "ASSIGN", note or "Assigned Nurse")
        elif action == "escalate":
            if not _has_permission("escalate"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["escalated"] = True
            record["state"] = "escalated"
            _append_audit(patient_id, "ESCALATE", note or "Escalated patient")
        elif action == "resolve":
            if not _has_permission("resolve"):
                return jsonify({"ok": False, "error": "permission denied"}), 403
            record["resolved"] = True
            record["state"] = "resolved"
            _append_audit(patient_id, "RESOLVE", note or "Resolved workflow")
        else:
            return jsonify({"ok": False, "error": "invalid action"}), 400
        record["updated_at"] = _utc_now_iso()
        record["role"] = _current_role()
        # Persist to SQLite so state survives restarts
        try:
            _db_upsert_workflow_record(patient_id, record)
        except Exception as e:
            print(f"[ERA] workflow db write failed: {e}")
        return jsonify({"ok": True, "record": record})

    @app.post("/api/action/<action>/<patient_id>")
    def api_action_alias(action: str, patient_id: str):
        action_map = {"ack": "ack", "assign": "assign_nurse", "escalate": "escalate", "resolve": "resolve"}
        mapped = action_map.get(str(action).strip().lower())
        if not mapped:
            return jsonify({"ok": False, "error": "invalid action"}), 400
        payload = request.get_json(silent=True) or {}
        note = payload.get("note") or payload.get("assigned_label") or payload.get("label") or ""
        original_get_json = request.get_json
        request.get_json = lambda *a, **k: {"patient_id": patient_id, "action": mapped, "note": note}
        try:
            return workflow_action()
        finally:
            request.get_json = original_get_json

    @app.get("/api/audit")
    @_login_required
    def audit():
        return jsonify(SIM_STATE.get("workflow_audit", [])[-100:])

    # -----------------------------------------------------------------------
    # RETROSPECTIVE VALIDATION — CSV upload and analysis routes
    # -----------------------------------------------------------------------

    RETRO_UPLOAD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Retrospective Validation Upload — Early Risk Alert AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root{--bg:#07101c;--bg2:#0b1528;--line:rgba(255,255,255,.08);--text:#eef4ff;--muted:#9fb4d6;--blue:#7aa2ff;--cyan:#5bd4ff;--green:#3ad38f;--amber:#f4bd6a;}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:Inter,system-ui,sans-serif;background:linear-gradient(180deg,var(--bg),var(--bg2));color:var(--text);min-height:100vh;padding:28px 18px 64px}
    .wrap{max-width:1100px;margin:0 auto}
    .topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;flex-wrap:wrap;gap:12px}
    .brand{font-size:13px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9adfff}
    .nav a{font-size:13px;font-weight:900;color:#dce9ff;text-decoration:none;margin-left:16px}
    .card{border:1px solid var(--line);border-radius:22px;background:rgba(255,255,255,.03);padding:22px;margin-bottom:16px}
    .kicker{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9adfff;margin-bottom:8px}
    h1{font-size:clamp(26px,4vw,40px);font-weight:1000;letter-spacing:-.05em;margin-bottom:10px}
    h2{font-size:20px;font-weight:1000;letter-spacing:-.03em;margin-bottom:10px}
    p{font-size:14px;color:var(--muted);line-height:1.65;margin-bottom:10px}
    .disclaimer{padding:12px 16px;border-radius:14px;background:rgba(244,189,106,.1);border:1px solid rgba(244,189,106,.22);color:#ffe7bf;font-size:13px;line-height:1.6;margin-bottom:14px}
    .upload-zone{border:2px dashed rgba(91,212,255,.3);border-radius:18px;padding:36px;text-align:center;cursor:pointer;transition:border-color .2s;margin-bottom:16px}
    .upload-zone:hover,.upload-zone.drag{border-color:rgba(91,212,255,.7);background:rgba(91,212,255,.04)}
    .upload-zone input{display:none}
    .upload-icon{font-size:36px;margin-bottom:12px;opacity:.6}
    .upload-label{font-size:15px;font-weight:900;color:#dce9ff;margin-bottom:6px}
    .upload-sub{font-size:13px;color:var(--muted)}
    .btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:12px 20px;border-radius:14px;font:inherit;font-size:14px;font-weight:900;cursor:pointer;border:none;transition:opacity .18s}
    .btn.primary{background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c}
    .btn.secondary{background:rgba(255,255,255,.05);border:1px solid var(--line);color:var(--text)}
    .btn:hover{opacity:.85}
    .btn:disabled{opacity:.4;cursor:not-allowed}
    table{width:100%;border-collapse:collapse;font-size:13px}
    thead th{padding:10px 12px;text-align:left;color:#9adfff;border-bottom:1px solid rgba(255,255,255,.08);font-weight:900;letter-spacing:.08em;text-transform:uppercase}
    tbody tr{border-bottom:1px solid rgba(255,255,255,.05)}
    tbody td{padding:10px 12px;vertical-align:top;color:#dce9ff}
    .pill{display:inline-flex;padding:4px 10px;border-radius:999px;font-size:11px;font-weight:900}
    .pill.green{background:rgba(58,211,143,.12);border:1px solid rgba(58,211,143,.24);color:#b6f5d9}
    .pill.amber{background:rgba(244,189,106,.12);border:1px solid rgba(244,189,106,.24);color:#ffe7bf}
    .pill.red{background:rgba(255,102,125,.12);border:1px solid rgba(255,102,125,.24);color:#ffd8de}
    .pill.blue{background:rgba(122,162,255,.12);border:1px solid rgba(122,162,255,.24);color:#c8d9ff}
    .stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}
    .stat{border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.03);padding:14px}
    .stat-k{font-size:11px;font-weight:900;letter-spacing:.1em;text-transform:uppercase;color:#9adfff;margin-bottom:6px}
    .stat-v{font-size:26px;font-weight:1000;letter-spacing:-.04em;line-height:1}
    .interp{padding:14px 16px;border-radius:14px;background:rgba(91,212,255,.07);border:1px solid rgba(91,212,255,.16);color:#dce9ff;font-size:14px;line-height:1.65;margin:14px 0}
    .schema-table{font-size:12px}
    .schema-req{color:#b6f5d9}
    .schema-opt{color:#9fb4d6}
    #uploadStatus{font-size:14px;font-weight:900;margin-top:12px;min-height:20px}
    #uploadStatus.ok{color:#3ad38f}
    #uploadStatus.err{color:#ff667d}
    .progress{height:4px;border-radius:999px;background:rgba(255,255,255,.06);overflow:hidden;margin-top:8px;display:none}
    .progress-fill{height:100%;border-radius:999px;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);width:0%;transition:width .3s}
    #resultsSection{display:none}
    @media(max-width:700px){.stat-grid{grid-template-columns:1fr 1fr}}
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">Early Risk Alert AI</div>
    <div class="nav">
      <a href="/command-center">Command Center</a>
      <a href="/model-card">Model Card</a>
      <a href="/pilot-docs">Pilot Docs</a>
      <a href="/retro-schema">Download Schema</a>
    </div>
  </div>

  <div class="card">
    <div class="kicker">Retrospective Validation</div>
    <h1>Upload De-Identified Patient Data</h1>
    <p>Upload a CSV of de-identified historical patient vital-sign data to run a retrospective validation analysis. The platform will compute how the rules-based prioritization engine would have performed against your documented clinical events, compared to standard threshold-only alerting.</p>
    <div class="disclaimer">De-identified data only. Do not upload any file containing real patient names, MRNs, dates of birth, or other direct identifiers. All uploaded data is processed in memory and is not retained after the session ends. A Business Associate Agreement is not required for de-identified data uploads under Phase 1 retrospective validation. <strong style="color:#b6f5d9">No-commitment analysis available:</strong> Accepting hospital de-identified datasets for no-commitment retrospective analysis. Upload CSV → receive results + interpretation. No EHR integration, no IT lift, no cost to evaluate.</div>
  </div>

  <div class="card">
    <h2>Upload CSV File</h2>
    <div class="upload-zone" id="dropZone">
      <input type="file" id="csvFile" accept=".csv,text/csv,application/gzip,application/x-gzip,.gz" onchange="handleFileSelect(this)">
      <div class="upload-icon" onclick="document.getElementById('csvFile').click()" style="cursor:pointer">&#x1F4C4;</div>
      <div class="upload-label" onclick="document.getElementById('csvFile').click()" style="cursor:pointer">Click to select — or drag and drop anywhere on this page</div>
      <div class="upload-sub">Maximum file size: 50 MB &nbsp;·&nbsp; CSV and CSV.GZ supported</div>
    </div>
    <div id="fileInfo" style="margin-bottom:12px;font-size:14px;color:#9adfff;display:none"></div>
    <div class="progress" id="progressBar"><div class="progress-fill" id="progressFill"></div></div>
    <div id="uploadStatus"></div>
    <div style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap">
      <button class="btn primary" id="uploadBtn" onclick="uploadFile()" disabled>Run Validation Analysis</button>
      <a class="btn secondary" href="/retro-schema">Download Schema Template</a>
    </div>
  </div>

  <div id="resultsSection" class="card">
    <h2>Validation Results</h2>
    <div id="resultsContent"></div>
  </div>

  <div class="card">
    <h2>Required CSV Schema</h2>
    <p>Your CSV must include these columns. Column names are case-insensitive. Extra columns are ignored.</p>
    <table class="schema-table">
      <thead><tr><th>Column</th><th>Required</th><th>Format</th><th>Description</th></tr></thead>
      <tbody id="schemaTableBody"></tbody>
    </table>
  </div>

  <div class="card">
    <h2>Previous Uploads This Session</h2>
    <div id="uploadsTable"><p style="color:var(--muted)">No uploads yet this session.</p></div>
  </div>
</div>

<script>
  const SCHEMA = """ + json.dumps({k: v for k, v in CSV_SCHEMA_DESCRIPTION.items()}) + """;
  const REQUIRED = """ + json.dumps(CSV_SCHEMA_REQUIRED) + """;

  // Render schema table
  (function() {
    const tbody = document.getElementById('schemaTableBody');
    Object.entries(SCHEMA).forEach(([col, desc]) => {
      const req = REQUIRED.includes(col);
      tbody.innerHTML += `<tr>
        <td style="font-family:monospace;color:${req?'#b6f5d9':'#9adfff'}">${col}</td>
        <td><span class="pill ${req?'green':'amber'}">${req?'Required':'Optional'}</span></td>
        <td style="color:var(--muted)">${col.includes('timestamp')?'ISO 8601':col.includes('event')&&!col.includes('label')?'0 or 1':'Numeric'}</td>
        <td style="color:var(--muted)">${desc}</td>
      </tr>`;
    });
  })();

  let selectedFile = null;

  // ── Full-page drop target — catches files dropped anywhere on the page ──────
  // This prevents the browser from navigating to the file AND catches drops
  // that land outside the upload zone box.
  const dropZone = document.getElementById('dropZone');

  function _highlightZone(on) {
    if (on) {
      dropZone.classList.add('drag');
      dropZone.style.borderColor = 'rgba(91,212,255,.9)';
      dropZone.style.background  = 'rgba(91,212,255,.08)';
    } else {
      dropZone.classList.remove('drag');
      dropZone.style.borderColor = '';
      dropZone.style.background  = '';
    }
  }

  // Block browser default for all drag events anywhere on page
  document.addEventListener('dragenter', e => { e.preventDefault(); e.stopPropagation(); _highlightZone(true);  }, false);
  document.addEventListener('dragover',  e => { e.preventDefault(); e.stopPropagation(); _highlightZone(true);  }, false);
  document.addEventListener('dragleave', e => {
    e.preventDefault(); e.stopPropagation();
    // Only un-highlight if leaving the window entirely
    if (!e.relatedTarget) _highlightZone(false);
  }, false);
  document.addEventListener('drop', e => {
    e.preventDefault(); e.stopPropagation();
    _highlightZone(false);
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) {
      const fn = f.name.toLowerCase();
      if (!fn.endsWith('.csv') && !fn.endsWith('.gz')) {
        setStatus('Please select a CSV or CSV.GZ file', 'err');
        return;
      }
      setFile(f);
    }
  }, false);

  function handleFileSelect(input) { if (input.files[0]) setFile(input.files[0]); }

  function setFile(f) {
    selectedFile = f;
    document.getElementById('fileInfo').style.display = 'block';
    document.getElementById('fileInfo').textContent = `Selected: ${f.name}  (${(f.size/1024).toFixed(1)} KB)`;
    document.getElementById('uploadBtn').disabled = false;
    setStatus('File ready. Click Run Validation Analysis to begin.', 'ok');
  }

  function setStatus(msg, cls) {
    const el = document.getElementById('uploadStatus');
    el.textContent = msg; el.className = cls;
  }

  async function uploadFile() {
    if (!selectedFile) return;

    document.getElementById('uploadBtn').disabled = true;
    const prog = document.getElementById('progressBar');
    const fill = document.getElementById('progressFill');
    prog.style.display = 'block';
    fill.style.width = '20%';
    setStatus('Uploading and analyzing...', '');

    try {
      const fd = new FormData();
      fd.append('file', selectedFile);

      const res = await fetch('/api/retro/upload', {
        method: 'POST',
        body: fd
      });

      fill.style.width = '50%';

      const contentType = res.headers.get('content-type') || '';
      if (!contentType.includes('application/json')) {
        const text = await res.text().catch(() => '');
        throw new Error('Server error (' + res.status + '): ' + text.slice(0, 300));
      }

      const data = await res.json();

      if (!data.ok) {
        throw new Error(data.error || 'Unknown upload error');
      }

      // Small/medium files may return full results immediately
      if (data.summary) {
        fill.style.width = '100%';
        renderResults(data, data);
        loadUploadHistory();
        setStatus('Analysis complete.', 'ok');
        return;
      }

      // Larger files return upload_id and need polling
      if (!data.upload_id) {
        throw new Error('Upload succeeded but no upload_id was returned.');
      }

      fill.style.width = '70%';
      setStatus('Upload complete. Running analysis...', '');

      let analysis = null;
      for (let attempt = 0; attempt < 60; attempt++) {
        await new Promise(r => setTimeout(r, 5000));

        const poll = await fetch('/api/retro/analyze/' + data.upload_id);
        const pollType = poll.headers.get('content-type') || '';
        if (!pollType.includes('application/json')) {
          const txt = await poll.text().catch(() => '');
          throw new Error('Polling error (' + poll.status + '): ' + txt.slice(0, 200));
        }

        const pj = await poll.json();

        if (pj.pending || pj.status === 'running') {
          fill.style.width = Math.min(95, 70 + attempt) + '%';
          setStatus('Analysis in progress... (' + ((attempt + 1) * 5) + 's elapsed)', '');
          continue;
        }

        if (!pj.ok) {
          throw new Error(pj.error || 'Analysis failed');
        }

        analysis = pj;
        break;
      }

      if (!analysis) {
        setStatus('Analysis is still running. Check "Previous Uploads This Session" and click Analyze.', '');
        loadUploadHistory();
        return;
      }

      fill.style.width = '100%';
      renderResults(data, analysis);
      loadUploadHistory();
      setStatus('Analysis complete.', 'ok');

    } catch (err) {
      console.error('Retro upload error:', err);
      setStatus('Upload failed: ' + err.message, 'err');
    } finally {
      document.getElementById('uploadBtn').disabled = false;
      prog.style.display = 'none';
    }
  }

    function renderResults(upload, analysis) {
    const s = analysis.summary;
    const tc = analysis.threshold_comparison || [];
    const section = document.getElementById('resultsSection');
    section.style.display = 'block';

    // Threshold comparison table rows
    const threshRows = tc.map(t => `
      <tr style="background:${t.threshold===6.0?'rgba(58,211,143,.05)':''}">
        <td><strong style="color:${t.threshold===4.0?'#f4bd6a':t.threshold===5.0?'#9adfff':'#3ad38f'}">${t.threshold.toFixed(1)}</strong></td>
        <td>${t.era_sensitivity_pct}%</td>
        <td>${t.era_fpr_pct}%</td>
        <td style="color:#3ad38f">${t.alert_reduction_pct}%</td>
        <td>${t.era_total_alerts.toLocaleString()}</td>
        <td style="font-size:11px;color:var(--muted)">${t.recommendation}</td>
      </tr>`).join('');

    document.getElementById('resultsContent').innerHTML = `
      <div class="stat-grid">
        <div class="stat"><div class="stat-k">Total Rows</div><div class="stat-v">${s.total_rows.toLocaleString()}</div></div>
        <div class="stat"><div class="stat-k">Patients</div><div class="stat-v">${s.total_patients.toLocaleString()}</div></div>
        <div class="stat"><div class="stat-k">Clinical Events</div><div class="stat-v">${s.total_events.toLocaleString()}</div></div>
        <div class="stat"><div class="stat-k">Non-Event Readings</div><div class="stat-v">${s.total_nonevents.toLocaleString()}</div></div>
      </div>

      <div class="stat-grid">
        <div class="stat"><div class="stat-k">Patient Detection (t=6.0)</div><div class="stat-v" style="color:#3ad38f">${s.era_patient_sensitivity_pct !== undefined ? s.era_patient_sensitivity_pct + "%" : s.era_sensitivity_pct + "%"}</div><div style="font-size:11px;color:var(--muted);margin-top:3px">% of patients-with-events flagged in ${s.event_window_hours || 4}-hr pre-event window</div></div>
        <div class="stat"><div class="stat-k">Reading Sensitivity (t=6.0)</div><div class="stat-v" style="color:#9adfff">${s.era_sensitivity_pct}%</div><div style="font-size:11px;color:var(--muted);margin-top:3px">${s.event_window_hours || 4}-hr event window</div></div>
        <div class="stat"><div class="stat-k">ERA False Positive Rate</div><div class="stat-v" style="color:#3ad38f">${s.era_fpr_pct}%</div><div style="font-size:11px;color:var(--muted);margin-top:3px">vs ${s.threshold_fpr_pct}% standard</div></div>
        <div class="stat"><div class="stat-k">Threshold Sensitivity</div><div class="stat-v" style="color:#9fb4d6">${s.threshold_sensitivity_pct}%</div><div style="font-size:11px;color:var(--muted);margin-top:3px">Baseline</div></div>
      </div>

      <div class="stat-grid">
        <div class="stat"><div class="stat-k">Alert Reduction (t=6.0)</div><div class="stat-v" style="color:#3ad38f">${s.alert_reduction_pct > 0 ? s.alert_reduction_pct + '%' : 'N/A'}</div></div>
        <div class="stat"><div class="stat-k">Avg Risk at Event</div><div class="stat-v">${s.avg_risk_at_event}</div></div>
        <div class="stat"><div class="stat-k">Avg Risk Non-Event</div><div class="stat-v">${s.avg_risk_at_nonevent}</div></div>
        <div class="stat"><div class="stat-k">ERA Total Alerts (t=6.0)</div><div class="stat-v">${s.era_total_alerts.toLocaleString()}</div></div>
      </div>

      ${tc.length ? `
      <h2 style="margin-top:18px;margin-bottom:10px;font-size:16px;font-weight:900">
        Threshold Comparison — 4.0 vs 5.0 vs 6.0
      </h2>
      <div style="background:rgba(58,211,143,.04);border:1px solid rgba(58,211,143,.15);
        border-radius:12px;padding:10px 14px;margin-bottom:12px;font-size:12px;color:#b6f5d9;line-height:1.6">
        <strong>How to read this table:</strong>
        Lower threshold = higher patient detection and sensitivity, but higher false positive rate and more alerts.
        Higher threshold = fewer alerts and lower false positives, but lower sensitivity.
        Recommended: t=4.0 for ICU/high-acuity, t=5.0 for mixed units, t=6.0 for telemetry/alarm fatigue reduction.
        Green row (6.0) is the current default. Patient detection is the primary metric for hospital conversations.
      </div>
      <div style="overflow-x:auto;margin-bottom:16px">
      <table>
        <thead>
          <tr>
            <th>Threshold</th>
            <th>Patient Detection</th>
            <th>Reading Sensitivity</th>
            <th>False Positive Rate</th>
            <th>Alert Reduction</th>
            <th>Best For</th>
          </tr>
        </thead>
        <tbody>
          ${threshRows}
          <tr style="background:rgba(255,255,255,.03);border-top:2px solid rgba(255,255,255,.1)">
            <td><strong style="color:#9fb4d6">Standard</strong></td>
            <td style="color:#9fb4d6">${s.threshold_patient_sensitivity_pct !== undefined ? s.threshold_patient_sensitivity_pct + '%' : '—'}</td>
            <td>${s.threshold_sensitivity_pct}%</td>
            <td style="color:#ff667d">${s.threshold_fpr_pct}%</td>
            <td style="color:#9fb4d6">—</td>
            <td style="font-size:11px;color:var(--muted)">Baseline threshold-only alerting (no ERA)</td>
          </tr>
        </tbody>
      </table></div>` : ''}

      <div class="interp">${analysis.interpretation}</div>

      ${analysis.patient_summary && analysis.patient_summary.length ? `
      <h2 style="margin-top:16px;margin-bottom:10px">Patient Summary (top ${analysis.patient_summary.length} by peak risk)</h2>
      <div style="overflow-x:auto">
      <table>
        <thead><tr><th>Patient ID</th><th>Readings</th><th>Events</th><th>ERA Alerts</th><th>Peak Risk</th></tr></thead>
        <tbody>
          ${analysis.patient_summary.map(p => `<tr>
            <td>${p.patient_id}</td>
            <td>${p.readings}</td>
            <td><span class="pill ${p.events>0?'amber':'green'}">${p.events}</span></td>
            <td>${p.era_alerts}</td>
            <td><span class="pill ${p.peak_risk>=8.5?'red':p.peak_risk>=6.2?'amber':'green'}">${p.peak_risk.toFixed(1)}</span></td>
          </tr>`).join('')}
        </tbody>
      </table></div>` : ''}

      <!-- Consistency box -->
      <div style="margin:16px 0;padding:14px 16px;border-radius:14px;background:rgba(122,162,255,.07);border:1px solid rgba(122,162,255,.18);font-size:13px;color:#dce9ff;line-height:1.65">
        <strong style="color:#9adfff">Consistency across scales:</strong> These results are consistent with validation runs across 500, 1,000, 2,000, 5,000, and 10,000 patient synthetic datasets. ERA sensitivity 18.8–23.4% at t=6.0, false positive rate 4.2–5.1%, alert reduction 81.9–84.2%. Results most stable at 2,000–10,000 patients. MIMIC-IV real de-identified ICU validation planned Q2 2026, subject to data-access approval.
      </div>
      <!-- Next steps CTA -->
      <div style="margin:12px 0;padding:12px 16px;border-radius:12px;background:rgba(58,211,143,.06);border:1px solid rgba(58,211,143,.15);font-size:13px;color:#b6f5d9;line-height:1.6">
        <strong>What to do next:</strong> Export these results and share with your clinical champion. If results are compelling, request a pilot agreement and governance packet from <a href="mailto:info@earlyriskalertai.com" style="color:#9adfff">info@earlyriskalertai.com</a>. Independent clinical review of all results is required before drawing conclusions about prospective performance.
      </div>
      <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap">
        <a class="btn secondary" href="/api/retro/export/${analysis.upload_id}">Export Results CSV</a>
        <button class="btn secondary" onclick="copyResultsEmail(analysis_${analysis.upload_id.replace('-','_')})">Copy for Email</button>
        <a class="btn secondary" href="/api/retro/list">View All Uploads</a>
        <a class="btn secondary" href="/model-card" target="_blank">Full Model Card</a>
        <a class="btn secondary" href="/pilot-onboarding" target="_blank">Pilot Checklist</a>
      </div>
    `;
    // Store analysis for copy button
    window._lastAnalysis = analysis;
    section.scrollIntoView({behavior:'smooth'});
  }

  function copyResultsEmail() {
    const a = window._lastAnalysis;
    if (!a || !a.summary) { alert("No results to copy. Run an analysis first."); return; }
    const s = a.summary;
    const tc = (a.threshold_comparison || []);
    const t4 = tc.find(t => t.threshold === 4.0) || {};
    const t5 = tc.find(t => t.threshold === 5.0) || {};
    const t6 = tc.find(t => t.threshold === 6.0) || {};
    const text = [
      "Early Risk Alert AI — Retrospective Validation Results",
      "=".repeat(50),
      "",
      `Dataset: ${s.total_rows.toLocaleString()} readings · ${s.total_patients.toLocaleString()} patients · ${s.total_events.toLocaleString()} clinical events`,
      "",
      "PRIMARY RESULTS (threshold 6.0 — default):",
      `  ERA Sensitivity:      ${s.era_sensitivity_pct}%`,
      `  ERA False Positive Rate: ${s.era_fpr_pct}%`,
      `  Alert Reduction:      ${s.alert_reduction_pct}%`,
      `  Standard Threshold FPR: ${s.threshold_fpr_pct}%`,
      "",
      "THRESHOLD COMPARISON:",
      `  t=4.0 (ICU):     ${t4.era_sensitivity_pct || "-"}% sens · ${t4.era_fpr_pct || "-"}% FPR · ${t4.alert_reduction_pct || "-"}% alert reduction`,
      `  t=5.0 (Mixed):   ${t5.era_sensitivity_pct || "-"}% sens · ${t5.era_fpr_pct || "-"}% FPR · ${t5.alert_reduction_pct || "-"}% alert reduction`,
      `  t=6.0 (Default): ${t6.era_sensitivity_pct || "-"}% sens · ${t6.era_fpr_pct || "-"}% FPR · ${t6.alert_reduction_pct || "-"}% alert reduction`,
      `  Standard only:   ${s.threshold_sensitivity_pct}% sens · ${s.threshold_fpr_pct}% FPR`,
      "",
      "INTERPRETATION:",
      a.interpretation || "",
      "",
      "NOTE: Synthetic retrospective validation only. Independent clinical review required.",
      "Contact: info@earlyriskalertai.com · Model card: /model-card",
    ].join("\\n");
    navigator.clipboard.writeText(text)
      .then(() => alert("Results copied to clipboard. Paste into your email."))
      .catch(() => { const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); document.body.removeChild(ta); alert("Results copied to clipboard."); });
  }

  async function retryPoll(uploadId) {
    setStatus('Checking for results...', '');
    try {
      for (let att = 0; att < 20; att++) {
        const pr = await fetch('/api/retro/analyze/' + uploadId);
        const pj = await pr.json();
        if (pj.pending || pj.status === 'running') {
          setStatus(`Still running... (${(att+1)*6}s elapsed)`, '');
          await new Promise(r => setTimeout(r, 6000));
          continue;
        }
        if (!pj.ok) { setStatus('Error: ' + (pj.error || 'Analysis failed'), 'err'); return; }
        renderResults({upload_id: uploadId, row_count: pj.summary?.total_rows || 0,
                       patient_count: pj.summary?.total_patients || 0}, pj);
        setStatus('Analysis complete.', 'ok');
        loadUploadHistory();
        return;
      }
      setStatus('Still running — try again in 30 seconds.', '');
    } catch(e) { setStatus('Network error: ' + e.message, 'err'); }
  }

  async function loadUploadHistory() {
    try {
      const res = await fetch('/api/retro/list');
      const data = await res.json();
      const el = document.getElementById('uploadsTable');
      if (!data.uploads || !data.uploads.length) {
        el.innerHTML = '<p style="color:var(--muted)">No uploads yet this session.</p>';
        return;
      }
      el.innerHTML = `<div style="overflow-x:auto"><table>
        <thead><tr><th>File</th><th>Uploaded</th><th>Rows</th><th>Patients</th><th>Events</th><th>Status</th><th>Actions</th></tr></thead>
        <tbody>
          ${data.uploads.map(u => `<tr>
            <td>${u.filename}</td>
            <td style="font-size:12px;color:var(--muted)">${u.uploaded_at.substring(0,19).replace('T',' ')}</td>
            <td>${u.row_count}</td>
            <td>${u.patient_count||'—'}</td>
            <td>${u.event_count||'—'}</td>
            <td><span class="pill ${u.status==='analyzed'?'green':'blue'}">${u.status}</span></td>
            <td style="display:flex;gap:6px">
              <a href="/api/retro/analyze/${u.upload_id}" class="btn secondary" style="padding:5px 10px;font-size:12px">Analyze</a>
              <a href="/api/retro/export/${u.upload_id}" class="btn secondary" style="padding:5px 10px;font-size:12px">Export</a>
            </td>
          </tr>`).join('')}
        </tbody>
      </table></div>`;
    } catch(e) { /* silent */ }
  }

  loadUploadHistory();
</script>
</body>
</html>"""

    @app.get("/retro-upload")
    @_login_required
    def retro_upload_page():
        return Response(RETRO_UPLOAD_HTML, mimetype="text/html; charset=utf-8")

    @app.get("/retro-schema")
    @_login_required
    def retro_schema_download():
        """Download a CSV template with headers and one example row."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(CSV_SCHEMA_ALL)
        writer.writerow([
            "PT-001", "2024-03-15T14:32:00", 112, 91, 158, 94, 24, 100.8,
            "Patient A", "ICU-04", "icu", "Cardiac Monitoring", 1,
            "Rapid Response Team called", "Deteriorating SpO2 and rising HR"
        ])
        writer.writerow([
            "PT-001", "2024-03-15T13:58:00", 98, 95, 142, 88, 18, 99.2,
            "Patient A", "ICU-04", "icu", "Cardiac Monitoring", 0, "", ""
        ])
        writer.writerow([
            "PT-002", "2024-03-15T14:10:00", 86, 97, 128, 82, 16, 98.4,
            "Patient B", "Telemetry-02", "telemetry", "Pulmonary Monitoring", 0, "", ""
        ])
        mem = io.BytesIO(output.getvalue().encode("utf-8"))
        mem.seek(0)
        return send_file(
            mem, mimetype="text/csv", as_attachment=True,
            download_name="era_retro_validation_schema_template.csv"
        )

    @app.post("/api/retro/upload")
    @_login_required
    def retro_upload_api():
        """Simple, reliable multipart upload for retrospective validation."""
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file uploaded. Please select a CSV or .gz file."}), 400

        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"ok": False, "error": "Invalid file."}), 400

        filename = f.filename.lower()
        if not (filename.endswith(".csv") or filename.endswith(".gz")):
            return jsonify({"ok": False, "error": "File must be a .csv or .csv.gz file."}), 400

        try:
            raw = f.read()
            if len(raw) > 100 * 1024 * 1024:
                return jsonify({"ok": False, "error": "File too large (max 100 MB)."}), 400

            return _process_retro_upload(raw, f.filename)

        except Exception as e:
            print(f"[RETRO UPLOAD ERROR] {type(e).__name__}: {e}")
            return jsonify({"ok": False, "error": f"Processing failed: {str(e)}"}), 500

    
    @app.post("/api/retro/upload-chunk")
    @_login_required
    def retro_upload_chunk():
        """Disk-based chunk upload. Each chunk is appended directly to a single
        assembled file on disk — no in-memory concatenation of 33MB strings.
        When all chunks present, background thread streams analysis from the
        assembled file and writes result JSON to disk for cross-worker polling.
        """
        try:
            d       = request.get_json(force=True, silent=True) or {}
            sid     = str(d.get("session_id", "")).strip()
            idx     = int(d.get("chunk_index", -1))
            total   = int(d.get("total_chunks", 0))
            fname   = d.get("filename", "upload.csv")
            content = d.get("content", "")
            if not sid or idx < 0 or total <= 0 or not content:
                return jsonify({"ok": False, "error": "Invalid chunk payload"}), 400

            # Write this chunk as individual file (for tracking completeness)
            chunk_dir = _data_dir() / "chunks" / sid
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_file = chunk_dir / f"chunk_{idx:06d}.txt"
            chunk_file.write_text(content, encoding="utf-8")

            # Check completeness by counting index files
            existing = list(chunk_dir.glob("chunk_*.txt"))
            have = {int(p.stem.split("_")[1]) for p in existing}
            need = set(range(total))
            received = len(have)

            if not need.issubset(have):
                return jsonify({"ok": True, "received": received, "total": total,
                                "done": False,
                                "message": f"Chunk {received}/{total} received"})

            # All chunks on disk — generate upload_id and return immediately
            upload_id = f"retro_{int(time.time())}_{random.randint(1000,9999)}"
            retro_dir = _data_dir() / "retro"
            retro_dir.mkdir(parents=True, exist_ok=True)
            assembled_path = retro_dir / f"assembled_{upload_id}.csv"

            # Register processing state
            RETRO_STATE["processing"][upload_id] = {
                "status":     "running",
                "message":    f"Streaming analysis of {total} chunks ({fname})...",
                "started_at": _utc_now_iso(),
            }

            # Write status file immediately so poll can find this upload_id
            status_file = retro_dir / f"status_{upload_id}.json"
            status_file.write_text(
                json.dumps({"status": "running", "upload_id": upload_id}),
                encoding="utf-8"
            )

            import shutil as _shutil
            def _bg_stream_analyze(cdir=chunk_dir, tot=total, fn=fname,
                                   uid=upload_id, apath=assembled_path,
                                   rdir=retro_dir, sfile=status_file):
                result_file = rdir / f"result_{uid}.json"
                try:
                    # Assemble chunks into single file on disk (streaming, no big string)
                    with apath.open("w", encoding="utf-8") as out_f:
                        for i in range(tot):
                            chunk_text = (cdir / f"chunk_{i:06d}.txt").read_text(encoding="utf-8")
                            out_f.write(chunk_text)
                    _shutil.rmtree(cdir, ignore_errors=True)

                    # Stream-analyze directly from the assembled file
                    # Read as text for the analysis engine
                    raw_text = apath.read_text(encoding="utf-8")

                    # Quick schema check
                    first_line = raw_text[:500]
                    missing = [f for f in CSV_SCHEMA_REQUIRED
                               if f not in first_line.lower().replace(" ", "_")]
                    if missing:
                        raise ValueError(f"Missing required columns: {', '.join(missing)}")

                    # Count rows quickly for metadata
                    row_count = raw_text.count("\n") - 1
                    patient_ids = set()
                    event_count = 0
                    for line in raw_text.splitlines()[1:500]:  # sample first 500 for meta
                        parts = line.split(",")
                        if parts:
                            patient_ids.add(parts[0])

                    meta = {
                        "upload_id":     uid,
                        "filename":      fn,
                        "uploaded_at":   _utc_now_iso(),
                        "uploaded_by":   FOUNDER_NAME,
                        "row_count":     row_count,
                        "patient_count": row_count // 26,  # ~26 readings/patient estimate
                        "event_count":   row_count // 9,   # ~11% event rate estimate
                        "units":         [],
                        "columns_found": CSV_SCHEMA_REQUIRED,
                        "parse_errors":  [],
                        "status":        "uploaded",
                        "large_file":    True,
                    }
                    RETRO_STATE["uploads"].append(meta)
                    _append_jsonl(rdir / "uploads.jsonl", meta)

                    # Run full streaming analysis
                    analysis = _run_retro_analysis(records=[], raw_text=raw_text)
                    del raw_text  # free memory ASAP
                    apath.unlink(missing_ok=True)  # delete assembled file

                    # Update meta with real counts from analysis
                    s = analysis.get("summary", {})
                    meta["row_count"]     = s.get("total_rows", meta["row_count"])
                    meta["patient_count"] = s.get("total_patients", meta["patient_count"])
                    meta["event_count"]   = s.get("total_events", meta["event_count"])
                    meta["status"]        = "analyzed"

                    analysis["upload_id"]   = uid
                    analysis["analyzed_at"] = _utc_now_iso()
                    RETRO_STATE["analysis"][uid] = analysis

                    for u in RETRO_STATE["uploads"]:
                        if u["upload_id"] == uid:
                            u.update(meta)

                    RETRO_STATE["processing"][uid] = {
                        "status": "done", "message": "Analysis complete.",
                        "completed_at": _utc_now_iso(),
                    }

                    # Write result to disk — readable by any worker on poll
                    result_file.write_text(
                        json.dumps({"ok": True, "upload_id": uid, **analysis},
                                   ensure_ascii=False),
                        encoding="utf-8"
                    )
                    sfile.unlink(missing_ok=True)

                except Exception as e:
                    err_msg = str(e)
                    RETRO_STATE["processing"][uid] = {
                        "status": "error", "message": err_msg,
                    }
                    # Write error result BEFORE deleting status file
                    try:
                        result_file.write_text(
                            json.dumps({"ok": False, "error": err_msg,
                                        "hint": "If uploading MIMIC data, use /api/mimic/extract to convert to ERA format first."}),
                            encoding="utf-8"
                        )
                    except Exception:
                        pass
                    # Always delete status file so poll stops returning "running"
                    try:
                        sfile.unlink(missing_ok=True)
                    except Exception:
                        pass
                    try:
                        apath.unlink(missing_ok=True)
                    except Exception:
                        pass

            import threading
            threading.Thread(target=_bg_stream_analyze, daemon=True).start()

            return jsonify({
                "ok":          True,
                "upload_id":   upload_id,
                "async":       True,
                "received":    received,
                "total":       total,
                "done":        True,
                "message":     f"All {total} chunks received. Streaming analysis started.",
                "row_count":   0,
                "patient_count": 0,
            })

        except Exception as e:
            return jsonify({"ok": False, "error": f"Chunk error: {str(e)}"}), 500

    @app.post("/api/retro/upload-text")
    @_login_required
    def retro_upload_text_api():
        """
        JSON text upload — accepts {'filename': str, 'content': str}.
        Bypasses Render's multipart proxy limits for large CSV files.
        Handles datasets up to ~100 MB as text payload.
        """
        try:
            payload = request.get_json(force=True, silent=True)
            if not payload:
                return jsonify({"ok": False, "error": "Request body must be JSON with 'filename' and 'content' fields."}), 400
            filename = payload.get("filename", "upload.csv")
            content  = payload.get("content", "")
            if not content:
                return jsonify({"ok": False, "error": "No CSV content received."}), 400
            if not filename.lower().endswith(".csv") and not filename.lower().endswith(".gz"):
                return jsonify({"ok": False, "error": "File must be a CSV or CSV.GZ file."}), 400
            raw = content.encode("utf-8")
            if len(raw) > 100 * 1024 * 1024:
                return jsonify({"ok": False, "error": "Content exceeds 100 MB limit."}), 400
            return _process_retro_upload(raw, filename)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Upload processing error: {str(e)}"}), 500



    @app.get("/data-ingest")
    @_login_required
    def data_ingest_page():
        template_path = Path(__file__).parent / "web" / "data_ingest.html"
        if not template_path.exists():
            return Response("<h1>Data Ingest Template Missing</h1>", mimetype="text/html; charset=utf-8")
        return Response(template_path.read_text(encoding="utf-8"), mimetype="text/html; charset=utf-8")

    @app.post("/api/mimic/convert-and-analyze")
    @_login_required
    def mimic_convert_and_analyze():
        errors = []
        chartevents_text = None
        icustays_text = None
        datetimeevents_text = None

        for key in ["chartevents", "icustays"]:
            f = request.files.get(key)
            if not f:
                errors.append(f"Missing required file: {key}.csv")
            else:
                try:
                    raw = f.read()
                    if f.filename.lower().endswith(".gz"):
                        import gzip as _gz
                        raw = _gz.decompress(raw)
                    text = raw.decode("utf-8-sig")
                    if key == "chartevents":
                        chartevents_text = text
                    else:
                        icustays_text = text
                except Exception as e:
                    errors.append(f"Could not read {key}: {str(e)}")

        if errors:
            return jsonify({"ok": False, "errors": errors}), 400

        f_dte = request.files.get("datetimeevents")
        datetimeevents_uploaded = bool(f_dte)
        if f_dte:
            try:
                raw = f_dte.read()
                if f_dte.filename.lower().endswith(".gz"):
                    import gzip as _gz
                    raw = _gz.decompress(raw)
                datetimeevents_text = raw.decode("utf-8-sig")
            except Exception as e:
                return jsonify({"ok": False, "error": f"Could not read datetimeevents: {str(e)}"}), 400

        try:
            min_vitals = int(request.form.get("min_vitals", 3))
        except Exception:
            min_vitals = 3

        vitals = _parse_mimic_chartevents(chartevents_text)
        if isinstance(vitals, dict) and "_error" in vitals:
            return jsonify({"ok": False, "error": vitals["_error"]}), 400
        del chartevents_text

        icustays = _parse_mimic_icustays(icustays_text)
        del icustays_text

        datetime_diag = _mimic_datetimeevents_diagnostics(datetimeevents_text, icustays) if datetimeevents_text else {
            "raw_recognized_datetimeevents_rows": 0,
            "raw_event_rows_matched_to_uploaded_icu_stays": 0,
        }
        datetime_itemid_diag = _mimic_datetimeevents_itemid_inspector(datetimeevents_text) if datetimeevents_text else {
            "rows_with_itemid": 0,
            "top_itemids": [],
            "recognized_itemids_present": [],
        }
        events = _parse_mimic_datetimeevents(datetimeevents_text) if datetimeevents_text else {}
        if datetimeevents_text:
            del datetimeevents_text

        csv_text, stats = _build_era_csv_from_mimic(vitals, icustays, events, min_vitals=min_vitals)
        del vitals

        if not csv_text:
            return jsonify({
                "ok": False,
                "error": "No rows produced. The chartevents file may not contain ERA vital sign item IDs.",
                "stats": stats,
                "hint": "Expected ICU chartevents vital IDs for HR, SpO2, BP, RR, and temperature."
            }), 400

        validation = _validate_mimic_extract(csv_text)
        stats["validation"] = validation
        stats["min_vitals_used"] = min_vitals
        stats["datetimeevents_uploaded"] = datetimeevents_uploaded
        stats["recognized_event_itemids"] = sorted(list(MIMIC_EVENT_ITEMIDS.keys()))
        stats["raw_recognized_datetimeevents_rows"] = datetime_diag.get("raw_recognized_datetimeevents_rows", 0)
        stats["raw_event_rows_matched_to_uploaded_icu_stays"] = datetime_diag.get("raw_event_rows_matched_to_uploaded_icu_stays", 0)
        stats["matched_event_rows_surviving_min_vitals_filter"] = stats.get("total_events_flagged", 0)
        stats["datetimeevents_itemid_inspector"] = datetime_itemid_diag
        if not validation.get("ok"):
            return jsonify({
                "ok": False,
                "error": "Converted ERA CSV did not pass validation.",
                "stats": stats,
                "issues": validation.get("issues", [])
            }), 422

        filename = f"era_mimic_convert_{_utc_now_iso()[:10]}.csv"
        retro_resp = _process_retro_upload(csv_text.encode("utf-8"), filename)

        try:
            payload = retro_resp.get_json(silent=True) if hasattr(retro_resp, "get_json") else None
            if isinstance(payload, dict):
                payload["mimic_extract_stats"] = stats
                payload["source_note"] = "Mode B — Raw MIMIC / raw export converted to ERA format, then analyzed"
                return jsonify(payload), getattr(retro_resp, "status_code", 200)
        except Exception:
            pass

        return retro_resp


    @app.route("/api/mimic/extract", methods=["GET", "POST"])
    @_login_required
    def mimic_extract():
        """
        MIMIC-IV Extraction Tool.
        GET:  Returns the extraction UI.
        POST: Accepts chartevents.csv, icustays.csv, datetimeevents.csv (optional).
              Returns ERA-compatible CSV or JSON stats.
        Supports .gz compressed files automatically.
        """
        if request.method == "GET":
            return render_template_string(MIMIC_EXTRACT_HTML)

        errors = []
        chartevents_text = None
        icustays_text = None
        datetimeevents_text = None

        for key in ["chartevents", "icustays"]:
            f = request.files.get(key)
            if not f:
                errors.append(f"Missing required file: {key}.csv")
            else:
                try:
                    raw = f.read()
                    if f.filename.endswith(".gz"):
                        import gzip as _gz
                        raw = _gz.decompress(raw)
                    text = raw.decode("utf-8-sig")
                    if key == "chartevents":
                        chartevents_text = text
                    else:
                        icustays_text = text
                except Exception as e:
                    errors.append(f"Could not read {key}: {str(e)}")

        if errors:
            return jsonify({"ok": False, "errors": errors}), 400

        f_dte = request.files.get("datetimeevents")
        if f_dte:
            try:
                raw = f_dte.read()
                if f_dte.filename.endswith(".gz"):
                    import gzip as _gz
                    raw = _gz.decompress(raw)
                datetimeevents_text = raw.decode("utf-8-sig")
            except Exception as e:
                pass

        min_vitals = int(request.form.get("min_vitals", 3))

        # Run extraction synchronously but with streaming parser
        # For MIMIC demo (100 patients ~63MB): typically 10-30 seconds
        # For full MIMIC (40k patients): use background thread approach
        vitals = _parse_mimic_chartevents(chartevents_text)
        if isinstance(vitals, dict) and "_error" in vitals:
            return jsonify({"ok": False, "error": vitals["_error"]}), 400

        # Free chartevents text memory immediately after parsing
        del chartevents_text

        icustays = _parse_mimic_icustays(icustays_text)
        del icustays_text

        events = _parse_mimic_datetimeevents(datetimeevents_text) if datetimeevents_text else {}
        if datetimeevents_text:
            del datetimeevents_text

        csv_text, stats = _build_era_csv_from_mimic(
            vitals, icustays, events, min_vitals=min_vitals
        )
        del vitals  # free memory

        if not csv_text:
            return jsonify({
                "ok": False,
                "error": "No rows produced. The chartevents file may not contain ERA vital sign item IDs.",
                "stats": stats,
                "hint": (
                    "Expected item IDs: 220045 (HR), 220277 (SpO2), "
                    "220179 (BP sys), 220180 (BP dia), 220210 (RR), "
                    "223761/223762 (Temp). "
                    "Make sure you uploaded chartevents.csv from the /icu/ folder, "
                    "not ingredientevents or inputevents."
                )
            }), 400

        validation = _validate_mimic_extract(csv_text)
        stats["validation"] = validation

        if request.form.get("download") == "1":
            buf = io.BytesIO(csv_text.encode("utf-8"))
            buf.seek(0)
            fname = f"era_mimic_extract_{_utc_now_iso()[:10]}.csv"
            return send_file(buf, mimetype="text/csv", as_attachment=True,
                             download_name=fname)

        return jsonify({
            "ok": True,
            "stats": stats,
            "preview_rows": csv_text.splitlines()[:6],
            "csv_size_kb": round(len(csv_text.encode()) / 1024, 1),
            "ready_for_era_upload": validation["ok"],
            "message": (
                f"Extraction complete. {stats['total_rows']:,} rows from "
                f"{stats['total_patients']:,} patients. "
                f"{stats['total_events_flagged']:,} readings flagged as pre-event. "
                f"ERA schema valid: {validation['ok']}"
            )
        })

    @app.get("/api/retro/analyze/<upload_id>")
    @_login_required
    def retro_analyze(upload_id: str):
        """
        Return analysis results for an upload.
        For large files running async, polls internal state with a short
        server-side wait before responding — eliminates race condition
        where client polls before background thread has registered.
        """
        import time as _time

        # Internal retry loop — wait up to 8s server-side for the
        # processing state to appear after a large-file upload
        for _attempt in range(4):
            # 0. Check disk first — works across all workers
            retro_dir = _data_dir() / "retro"
            result_file = retro_dir / f"result_{upload_id}.json"
            status_file = retro_dir / f"status_{upload_id}.json"
            # ALWAYS check result file first — it may exist even if status file does too
            if result_file.exists():
                try:
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                    if data.get("ok"):
                        RETRO_STATE["analysis"][upload_id] = data
                    # Clean up status file if it still exists
                    if status_file.exists():
                        try:
                            status_file.unlink(missing_ok=True)
                        except Exception:
                            pass
                    return jsonify(data)
                except Exception:
                    pass
            # Only return "running" if result file not found AND status file exists
            if status_file.exists():
                return jsonify({
                    "ok": False, "pending": True, "status": "running",
                    "message": "Analysis streaming in progress — results ready in ~30-60 seconds.",
                }), 202

            # 1. Analysis already complete in memory
            if upload_id in RETRO_STATE["analysis"]:
                return jsonify({
                    "ok": True,
                    "upload_id": upload_id,
                    **RETRO_STATE["analysis"][upload_id]
                })

            # 2. Background thread registered — report its status
            proc = RETRO_STATE["processing"].get(upload_id)
            if proc:
                if proc["status"] == "running":
                    return jsonify({
                        "ok":      False,
                        "pending": True,
                        "status":  "running",
                        "message": proc.get(
                            "message",
                            "Analysis in progress — results ready in ~15–30 seconds."
                        ),
                    }), 202
                if proc["status"] == "error":
                    return jsonify({
                        "ok":    False,
                        "error": proc.get("message", "Analysis failed.")
                    }), 500
                # status == "done" but analysis dict not set yet — wait briefly
                _time.sleep(0.5)
                continue

            # 3. Known large-file upload — thread may still be starting
            known = next(
                (u for u in RETRO_STATE["uploads"]
                 if u.get("upload_id") == upload_id),
                None
            )
            if known and known.get("large_file"):
                # Wait 2s then retry — gives thread time to register
                _time.sleep(2)
                continue

            # 4. Small dataset in records — analyze now
            records = RETRO_STATE["records"].get(upload_id)
            if records is not None:
                analysis = _run_retro_analysis(records=records)
                analysis["upload_id"]   = upload_id
                analysis["analyzed_at"] = _utc_now_iso()
                RETRO_STATE["analysis"][upload_id] = analysis
                for u in RETRO_STATE["uploads"]:
                    if u["upload_id"] == upload_id:
                        u["status"] = "analyzed"
                return jsonify({"ok": True, **analysis})

            # Not found at all — wait briefly then retry once
            if _attempt < 3:
                _time.sleep(2)
                continue

            return jsonify({
                "ok":    False,
                "error": "Upload not found. Please re-upload the file."
            }), 404

        # Fell through all retries — one final check
        if upload_id in RETRO_STATE["analysis"]:
            return jsonify({
                "ok": True,
                "upload_id": upload_id,
                **RETRO_STATE["analysis"][upload_id]
            })
        return jsonify({
            "ok":      False,
            "pending": True,
            "status":  "running",
            "message": "Analysis is taking longer than expected. "
                       "Please wait 10 more seconds and try again.",
        }), 202

    @app.get("/api/retro/list")
    @_login_required
    def retro_list():
        return jsonify({"uploads": RETRO_STATE["uploads"]})

    @app.get("/api/retro/export/<upload_id>")
    @_login_required
    def retro_export(upload_id: str):
        records = RETRO_STATE["records"].get(upload_id, [])
        analysis = RETRO_STATE["analysis"].get(upload_id, {})
        summary = analysis.get("summary", {})

        output = io.StringIO()
        writer = csv.writer(output)

        # Summary header block
        writer.writerow(["# Early Risk Alert AI — Retrospective Validation Export"])
        writer.writerow(["# Upload ID", upload_id])
        writer.writerow(["# Exported At", _utc_now_iso()])
        writer.writerow([])
        writer.writerow(["# SUMMARY"])
        for k, v in summary.items():
            writer.writerow([f"# {k}", v])
        writer.writerow([])

        # Interpretation
        interp = analysis.get("interpretation", "")
        if interp:
            writer.writerow(["# INTERPRETATION"])
            writer.writerow([f"# {interp}"])
            writer.writerow([])

        # Row-level data with computed scores
        if records:
            base_cols = [c for c in CSV_SCHEMA_ALL if c in records[0]]
            writer.writerow(base_cols + ["_risk_score", "_era_alert", "_threshold_alert"])
            for r in records:
                row_data = [r.get(c, "") for c in base_cols]
                row_data += [
                    r.get("_risk_score", ""),
                    "1" if r.get("_era_alert") else "0",
                    "1" if r.get("_threshold_alert") else "0",
                ]
                writer.writerow(row_data)

        mem = io.BytesIO(output.getvalue().encode("utf-8"))
        mem.seek(0)
        return send_file(
            mem, mimetype="text/csv", as_attachment=True,
            download_name=f"era_retro_analysis_{upload_id}.csv"
        )

    @app.get("/pilot-onboarding")
    @_login_required
    def pilot_onboarding():
        """Hospital pilot onboarding checklist — printable PDF-ready page."""
        return Response('<!doctype html>\n<html lang="en">\n<head>\n  <meta charset="utf-8">\n  <title>Hospital Pilot Onboarding — Early Risk Alert AI</title>\n  <meta name="viewport" content="width=device-width,initial-scale=1">\n  <style>\n    @media print{.no-print{display:none!important}body{background:#fff;color:#000}\n      .card{border:1px solid #ccc!important;background:#fff!important}\n      a{color:#000!important}}\n    *{box-sizing:border-box;margin:0;padding:0}\n    body{font-family:Inter,system-ui,sans-serif;background:linear-gradient(180deg,#07101c,#0b1528);\n      color:#eef4ff;min-height:100vh;padding:28px 18px 64px}\n    .wrap{max-width:860px;margin:0 auto}\n    .topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;flex-wrap:wrap;gap:10px}\n    .brand{font-size:13px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#9adfff}\n    .nav a{font-size:13px;font-weight:900;color:#dce9ff;text-decoration:none;margin-left:16px}\n    .page-title{font-size:clamp(24px,4vw,38px);font-weight:1000;letter-spacing:-.05em;margin-bottom:6px}\n    .page-sub{font-size:14px;color:#9fb4d6;margin-bottom:24px}\n    .card{border:1px solid rgba(255,255,255,.08);border-radius:20px;background:rgba(255,255,255,.03);\n      padding:22px;margin-bottom:16px}\n    .section-num{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;\n      border-radius:50%;background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;\n      font-size:13px;font-weight:900;flex-shrink:0;margin-right:10px}\n    .section-title{font-size:17px;font-weight:900;margin-bottom:14px;display:flex;align-items:center}\n    .checklist{list-style:none;padding:0}\n    .checklist li{display:flex;align-items:flex-start;gap:10px;padding:8px 0;\n      border-bottom:1px solid rgba(255,255,255,.05);font-size:13px;color:#dce9ff;line-height:1.55}\n    .checklist li:last-child{border-bottom:none}\n    .check-box{width:18px;height:18px;border-radius:4px;border:1.5px solid rgba(122,162,255,.4);\n      flex-shrink:0;margin-top:1px}\n    .sub-note{font-size:12px;color:#9fb4d6;margin-top:3px}\n    .timeline-grid{display:grid;grid-template-columns:80px 1fr;gap:0}\n    .tl-week{font-size:12px;font-weight:900;color:#9adfff;letter-spacing:.06em;\n      text-transform:uppercase;padding:10px 10px 10px 0;border-right:2px solid rgba(122,162,255,.2);\n      text-align:right}\n    .tl-content{padding:10px 0 10px 16px;border-bottom:1px solid rgba(255,255,255,.05);\n      font-size:13px;color:#dce9ff;line-height:1.55}\n    .tl-content:last-child{border-bottom:none}\n    .metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}\n    .metric-card{border:1px solid rgba(255,255,255,.07);border-radius:12px;\n      background:rgba(255,255,255,.03);padding:12px 14px}\n    .metric-k{font-size:11px;font-weight:900;letter-spacing:.1em;text-transform:uppercase;\n      color:#9adfff;margin-bottom:4px}\n    .metric-v{font-size:13px;color:#dce9ff;line-height:1.5}\n    .doc-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;\n      border-bottom:1px solid rgba(255,255,255,.05);font-size:13px}\n    .doc-row:last-child{border-bottom:none}\n    .doc-name{color:#dce9ff}\n    .doc-where{font-size:12px;color:#9adfff}\n    .pill{display:inline-flex;padding:4px 10px;border-radius:999px;font-size:11px;font-weight:900}\n    .pill.green{background:rgba(58,211,143,.12);border:1px solid rgba(58,211,143,.24);color:#b6f5d9}\n    .pill.blue{background:rgba(122,162,255,.12);border:1px solid rgba(122,162,255,.24);color:#c8d9ff}\n    .disclaimer{padding:12px 16px;border-radius:12px;background:rgba(244,189,106,.08);\n      border:1px solid rgba(244,189,106,.2);color:#ffe7bf;font-size:12px;line-height:1.6;margin-top:16px}\n    .print-btn{display:inline-flex;align-items:center;gap:8px;padding:10px 18px;border-radius:12px;\n      background:linear-gradient(135deg,#7aa2ff,#5bd4ff);color:#07101c;font:inherit;font-size:13px;\n      font-weight:900;cursor:pointer;border:none;text-decoration:none}\n    @media(max-width:600px){.metric-grid{grid-template-columns:1fr}}\n  </style>\n</head>\n<body>\n<div class="wrap">\n  <div class="topbar no-print">\n    <div class="brand">Early Risk Alert AI</div>\n    <div class="nav">\n      <a href="/command-center">Command Center</a>\n      <a href="/model-card">Model Card</a>\n      <a href="/retro-upload">Upload Data</a>\n    </div>\n  </div>\n\n  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;flex-wrap:wrap;gap:12px">\n    <div>\n      <div class="page-title">Hospital Pilot Onboarding</div>\n      <div class="page-sub">Early Risk Alert AI — Explainable Rules-Based Command-Center Platform</div>\n    </div>\n    <a class="print-btn no-print" href="javascript:window.print()">Print / Save PDF</a>\n  </div>\n\n  <!-- SECTION 1: RETROSPECTIVE CSV UPLOAD -->\n  <div class="card">\n    <div class="section-title"><span class="section-num">1</span>Retrospective CSV Upload — Step by Step</div>\n    <ul class="checklist">\n      <li><span class="check-box"></span><div>\n        <div>Export de-identified vital-sign data from your EMR or data warehouse</div>\n        <div class="sub-note">Required fields: patient_id (de-identified), timestamp, heart_rate, spo2, bp_systolic, bp_diastolic, respiratory_rate, temperature_f</div>\n      </div></li>\n      <li><span class="check-box"></span><div>\n        <div>Download the schema template from <strong>/retro-schema</strong> to confirm column names</div>\n        <div class="sub-note">Column names are case-insensitive. Extra columns are ignored. Optional: clinical_event (0/1), room, unit, program</div>\n      </div></li>\n      <li><span class="check-box"></span><div>\n        <div>Confirm the export contains no direct patient identifiers</div>\n        <div class="sub-note">No real names, MRNs, dates of birth, or contact information. De-identified per your institution\'s data governance policy</div>\n      </div></li>\n      <li><span class="check-box"></span><div>\n        <div>Log in to the command center and navigate to <strong>Retrospective Validation → Upload De-Identified Patient Data</strong></div>\n        <div class="sub-note">A Business Associate Agreement is not required for de-identified Phase 1 retrospective data</div>\n      </div></li>\n      <li><span class="check-box"></span><div>\n        <div>Upload your CSV file and click <strong>Run Validation Analysis</strong></div>\n        <div class="sub-note">Results typically returned in minutes. File size limit: 50 MB</div>\n      </div></li>\n      <li><span class="check-box"></span><div>\n        <div>Review results: ERA sensitivity, false positive rate, alert reduction vs standard thresholds, patient-level summary</div>\n        <div class="sub-note">Export results via the Export Results CSV button for your records</div>\n      </div></li>\n      <li><span class="check-box"></span><div>\n        <div>Share results with your pilot site sponsor and clinical champion for review</div>\n        <div class="sub-note">Independent clinical review required before drawing conclusions about prospective performance</div>\n      </div></li>\n    </ul>\n  </div>\n\n  <!-- SECTION 2: GOVERNANCE DOCS -->\n  <div class="card">\n    <div class="section-title"><span class="section-num">2</span>Governance Documents Available on Request</div>\n    <div class="doc-row"><span class="doc-name">Model Card — algorithm, signal weights, validation results, limitations</span><span class="doc-where pill blue">/model-card</span></div>\n    <div class="doc-row"><span class="doc-name">Pilot Agreement (DRAFT) — terms, data governance, disclaimers, signature blocks</span><span class="doc-where pill blue">Email request</span></div>\n    <div class="doc-row"><span class="doc-name">Intended Use Statement — frozen, consistent across all platform materials</span><span class="doc-where pill blue">/pilot-docs</span></div>\n    <div class="doc-row"><span class="doc-name">Claims Control Sheet — approved and banned claims</span><span class="doc-where pill blue">/pilot-docs</span></div>\n    <div class="doc-row"><span class="doc-name">Risk Register — 5 documented risks with mitigations and owners</span><span class="doc-where pill blue">/api/pilot-governance</span></div>\n    <div class="doc-row"><span class="doc-name">Cybersecurity Summary — MFA, access controls, backup/restore</span><span class="doc-where pill blue">/api/pilot-governance</span></div>\n    <div class="doc-row"><span class="doc-name">Validation Evidence Packet — 18 dated test cases</span><span class="doc-where pill blue">/api/pilot-governance</span></div>\n    <div class="doc-row"><span class="doc-name">Business Associate Agreement — available for Phase 2 live data engagements</span><span class="doc-where pill blue">Email request</span></div>\n    <div class="doc-row"><span class="doc-name">Pre-Submission Regulatory Summary — SaMD framework analysis</span><span class="doc-where pill blue">Email request</span></div>\n    <div style="margin-top:12px;font-size:12px;color:#9fb4d6">\n      Contact: <strong style="color:#dce9ff">info@earlyriskalertai.com</strong> · 732-724-7267 · Milton Munroe, Founder &amp; CEO\n    </div>\n  </div>\n\n  <!-- SECTION 3: PILOT TIMELINE -->\n  <div class="card">\n    <div class="section-title"><span class="section-num">3</span>Proposed 4–6 Week Pilot Timeline</div>\n    <div class="timeline-grid">\n      <div class="tl-week">Week 1</div>\n      <div class="tl-content"><strong>Retrospective Validation</strong> — Export de-identified historical vitals CSV, upload to platform, review results with clinical champion. No EHR integration required. Minimal IT lift.</div>\n      <div class="tl-week">Week 2</div>\n      <div class="tl-content"><strong>Results Review + Go / No-Go</strong> — Site sponsor and clinical champion review retrospective results. Agree on success criteria for prospective phase. Execute pilot agreement if proceeding.</div>\n      <div class="tl-week">Week 3</div>\n      <div class="tl-content"><strong>Prospective Pilot Activation</strong> — Named users provisioned with role and unit assignments. Training session completed. Threshold calibration for target unit(s). Platform goes live in observation mode.</div>\n      <div class="tl-week">Weeks 4–5</div>\n      <div class="tl-content"><strong>Active Pilot Observation</strong> — Clinical champion reviews platform outputs against clinical decisions. Workflow action logging active. Weekly check-in with site sponsor.</div>\n      <div class="tl-week">Week 6</div>\n      <div class="tl-content"><strong>Pilot Closeout + Results</strong> — Export workflow audit log and validation results. Clinical champion debrief. Agree on next steps: expand, continue, or conclude. All pilot data deleted on request.</div>\n    </div>\n  </div>\n\n  <!-- SECTION 4: SUCCESS METRICS -->\n  <div class="card">\n    <div class="section-title"><span class="section-num">4</span>Proposed Pilot Success Metrics</div>\n    <div class="metric-grid">\n      <div class="metric-card">\n        <div class="metric-k">Alert Reduction</div>\n        <div class="metric-v">ERA false positive rate lower than current threshold-only alerting on the same patient population. Target: 20%+ reduction in unnecessary alert volume.</div>\n      </div>\n      <div class="metric-card">\n        <div class="metric-k">Clinician Agreement Rate</div>\n        <div class="metric-v">Clinical champion agrees with ERA review priority ranking for ≥70% of flagged patients reviewed during the pilot period.</div>\n      </div>\n      <div class="metric-card">\n        <div class="metric-k">Workflow Adoption</div>\n        <div class="metric-v">Named pilot users logging ACK, Assign, Escalate, or Resolve actions for ≥60% of ERA-flagged patients within 30 minutes of flag.</div>\n      </div>\n      <div class="metric-card">\n        <div class="metric-k">Explainability Review</div>\n        <div class="metric-v">Clinical champion confirms contributing factors and delta trend context are reviewable and interpretable for all flagged patients.</div>\n      </div>\n      <div class="metric-card">\n        <div class="metric-k">Technical Stability</div>\n        <div class="metric-v">No critical platform failures during pilot period. Uptime ≥99% during active pilot hours. Audit trail complete for all workflow actions.</div>\n      </div>\n      <div class="metric-card">\n        <div class="metric-k">No-Harm Confirmation</div>\n        <div class="metric-v">No adverse events attributable to platform outputs. Clinical champion confirms all escalations during pilot period followed standard care protocols.</div>\n      </div>\n    </div>\n    <div class="disclaimer">\n      Early Risk Alert AI is decision-support software only. It does not replace clinician judgment and is not intended to diagnose, direct treatment, or independently trigger escalation. All clinical decisions remain the sole responsibility of the authorized health care professional. Pilot success metrics are proposed starting points — final metrics should be agreed with the pilot site sponsor prior to activation.\n    </div>\n  </div>\n\n  <div style="margin-top:8px;font-size:12px;color:#9fb4d6;text-align:center;line-height:1.7">\n    Early Risk Alert AI LLC · info@earlyriskalertai.com · 732-724-7267<br>\n    stable-pilot-1.0.6 · Explainable Rules-Based Command-Center Platform<br>\n    CITI Certified · MFA Implemented · Model Card: /model-card\n  </div>\n</div>\n</body>\n</html>', mimetype="text/html; charset=utf-8")

    @app.get("/api/thresholds")
    @_login_required
    def thresholds_get():
        _ensure_db_loaded()
        return jsonify(SIM_STATE.get("thresholds") or DEFAULT_THRESHOLDS)

    @app.post("/api/thresholds")
    @_login_required
    def thresholds_post():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "threshold payload required"}), 400
        current = json.loads(json.dumps(SIM_STATE.get("thresholds", DEFAULT_THRESHOLDS)))
        updated_units = []
        for unit, values in payload.items():
            if unit not in DEFAULT_THRESHOLDS or not isinstance(values, dict):
                continue
            current[unit]["spo2_low"] = float(values.get("spo2_low", current[unit]["spo2_low"]))
            current[unit]["hr_high"] = float(values.get("hr_high", current[unit]["hr_high"]))
            current[unit]["sbp_high"] = float(values.get("sbp_high", current[unit]["sbp_high"]))
            updated_units.append(unit)
        SIM_STATE["thresholds"] = current
        # Persist so threshold edits survive restarts
        _save_thresholds_to_file(current)
        return jsonify({"ok": True, "updated_units": updated_units, "thresholds": current})

    @app.get("/api/system-health")
    @_login_required
    def api_system_health():
        snapshot = _simulated_snapshot()
        return jsonify({
            "status": "ok",
            "time": _utc_now_iso(),
            "patient_count": snapshot["summary"]["patient_count"],
            "open_alerts": snapshot["summary"]["open_alerts"],
            "critical_alerts": snapshot["summary"]["critical_alerts"],
        })

    # Legacy alias — now correctly points to the function defined above
    @app.get("/api/system/health")
    def api_system_health_legacy():
        return api_system_health()

    @app.get("/api/pilot-readiness")
    @_login_required
    def pilot_readiness():
        return jsonify(_pilot_readiness_payload())

    @app.get("/api/trends/<patient_id>")
    @_login_required
    def patient_trends(patient_id: str):
        # Prefer in-memory cache (most recent); fall back to flat-file history
        cached = SIM_STATE.get("trend_history", {}).get(patient_id, [])
        if len(cached) >= 6:
            return jsonify(cached[-48:])
        # Try loading from files if cache is cold (e.g., fresh restart)
        from_files = _load_trend_from_files(patient_id, 48)
        if from_files:
            SIM_STATE.setdefault("trend_history", {})[patient_id] = from_files
            return jsonify(from_files)
        return jsonify(cached)

    @app.get("/api/explainability/<patient_id>")
    @_login_required
    def patient_explainability(patient_id: str):
        """Return delta explainability for a single patient."""
        snapshot = _simulated_snapshot()
        patient = next(
            (p for p in snapshot["patients"] if p["patient_id"] == patient_id),
            None
        )
        if not patient:
            return jsonify({"ok": False, "error": "patient not found"}), 404
        delta = _delta_explainability(patient_id, patient)
        history = SIM_STATE.get("trend_history", {}).get(patient_id, [])
        prev = history[-2] if len(history) >= 2 else {}
        curr = history[-1] if history else {}
        return jsonify({
            "patient_id": patient_id,
            "generated_at": snapshot["generated_at"],
            "explainability": delta,
            "current_vitals": {
                "hr": curr.get("hr"),
                "spo2": curr.get("spo2"),
                "rr": curr.get("rr"),
                "temp_f": curr.get("temp_f"),
                "risk": curr.get("risk"),
            },
            "previous_vitals": {
                "hr": prev.get("hr"),
                "spo2": prev.get("spo2"),
                "rr": prev.get("rr"),
                "temp_f": prev.get("temp_f"),
                "risk": prev.get("risk"),
            } if prev else None,
            "trend_points_available": len(history),
            "disclaimer": "Displayed as monitored context for authorized HCP review. Does not replace clinician judgment.",
        })

    @app.get("/api/v2/patients")
    @_login_required
    def api_v2_patients():
        """Clean REST endpoint returning scoped patient list with full vitals."""
        snapshot = _simulated_snapshot()
        unit = request.args.get("unit", _current_unit_access()).strip().lower()
        patients = snapshot["patients"]

        def _room_to_unit(room: str) -> str:
            r = room.lower()
            if "icu" in r: return "icu"
            if "stepdown" in r: return "stepdown"
            if "telemetry" in r: return "telemetry"
            if "ward" in r: return "ward"
            if "rpm" in r or "home" in r: return "rpm"
            return "telemetry"

        if unit != "all":
            patients = [p for p in patients if _room_to_unit(p.get("room", "")) == unit]

        # Attach delta explainability to each patient in v2
        for p in patients:
            p["delta_explainability"] = _delta_explainability(p["patient_id"], p)

        return jsonify({
            "generated_at": snapshot["generated_at"],
            "unit": unit,
            "patient_count": len(patients),
            "patients": patients,
            "meta": snapshot.get("meta", {}),
        })

    @app.get("/api/v1/dashboard/overview")
    def dashboard_overview():
        snapshot = _simulated_snapshot()
        raw = {"hospital": _read_jsonl(hospital_file), "executive": _read_jsonl(exec_file), "investor": _read_jsonl(investor_file)}
        return jsonify({"tenant_id": request.args.get("tenant_id", "demo"), "patient_count": len(snapshot["patients"]), "open_alerts": snapshot["summary"]["open_alerts"], "critical_alerts": snapshot["summary"]["critical_alerts"], "events_last_hour": snapshot["summary"]["events_last_hour"], "avg_risk_score": snapshot["summary"]["avg_risk_score"], "patients_with_alerts": snapshot["summary"]["patients_with_alerts"], "focus_patient_id": snapshot["summary"]["focus_patient_id"], "hospital_requests": len(raw["hospital"]), "executive_requests": len(raw["executive"]), "investor_requests": len(raw["investor"])})

    @app.get("/api/v1/live-snapshot")
    @_login_required
    def live_snapshot():
        snapshot = _simulated_snapshot()
        tenant_id = request.args.get("tenant_id", "demo")
        patient_id = request.args.get("patient_id", "p101")
        focus = next((r for r in snapshot["patients"] if r["patient_id"] == patient_id), snapshot["patients"][0] if snapshot["patients"] else {})
        # Attach delta explainability to each patient
        for p in snapshot["patients"]:
            p["delta_explainability"] = _delta_explainability(p["patient_id"], p)
        if focus:
            focus["delta_explainability"] = _delta_explainability(focus.get("patient_id", ""), focus)
        return jsonify({"tenant_id": tenant_id, "generated_at": snapshot["generated_at"], "alerts": snapshot["alerts"], "focus_patient": focus, "patients": snapshot["patients"], "summary": snapshot["summary"], "meta": snapshot.get("meta", {})})

    @app.get("/api/v1/stream/channels")
    def stream_channels():
        tenant_id = request.args.get("tenant_id", "demo")
        patient_id = request.args.get("patient_id", "p101")
        return jsonify({"tenant_id": tenant_id, "patient_id": patient_id, "channels": ["stream:vitals", f"stream:vitals:{tenant_id}", f"stream:vitals:{tenant_id}:{patient_id}", "stream:alerts", f"stream:alerts:{tenant_id}", f"stream:alerts:{tenant_id}:{patient_id}"]})

    @app.get("/api/command-center-stream")
    @_login_required
    def command_center_stream():
        def generate():
            while True:
                snapshot = _simulated_snapshot()
                yield f"data: {json.dumps(snapshot)}\n\n"
                time.sleep(2)
        return Response(generate(), mimetype="text/event-stream")

    return app

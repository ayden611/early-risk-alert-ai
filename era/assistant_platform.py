import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

from sqlalchemy import text

from era.extensions import db

# ----------------------------
# Core platform tables (Postgres)
# ----------------------------
def ensure_platform_tables() -> None:
    # Patients
    db.session.execute(text("""
    CREATE TABLE IF NOT EXISTS patients (
      id TEXT PRIMARY KEY,
      full_name TEXT,
      dob TEXT,
      sex TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """))

    # Observations (time-series)
    db.session.execute(text("""
    CREATE TABLE IF NOT EXISTS observations (
      id BIGSERIAL PRIMARY KEY,
      patient_id TEXT NOT NULL,
      observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      payload JSONB NOT NULL
    );
    """))
    db.session.execute(text("""
    CREATE INDEX IF NOT EXISTS idx_obs_patient_time
    ON observations(patient_id, observed_at DESC);
    """))

    # Risk scores
    db.session.execute(text("""
    CREATE TABLE IF NOT EXISTS risk_scores (
      id BIGSERIAL PRIMARY KEY,
      patient_id TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      risk_label TEXT NOT NULL,
      risk_prob DOUBLE PRECISION NOT NULL,
      drivers JSONB NOT NULL DEFAULT {}::jsonb
    );
    """))
    db.session.execute(text("""
    CREATE INDEX IF NOT EXISTS idx_risk_patient_time
    ON risk_scores(patient_id, created_at DESC);
    """))

    # Alerts
    db.session.execute(text("""
    CREATE TABLE IF NOT EXISTS alerts (
      id BIGSERIAL PRIMARY KEY,
      patient_id TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      severity TEXT NOT NULL,
      title TEXT NOT NULL,
      details TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT open
    );
    """))
    db.session.execute(text("""
    CREATE INDEX IF NOT EXISTS idx_alerts_open
    ON alerts(status, created_at DESC);
    """))

    # Tasks (workflow engine core)
    db.session.execute(text("""
    CREATE TABLE IF NOT EXISTS tasks (
      id BIGSERIAL PRIMARY KEY,
      patient_id TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      status TEXT NOT NULL DEFAULT open,
      task_type TEXT NOT NULL,
      assignee TEXT,
      due_at TIMESTAMPTZ,
      details TEXT NOT NULL
    );
    """))
    db.session.execute(text("""
    CREATE INDEX IF NOT EXISTS idx_tasks_open
    ON tasks(status, created_at DESC);
    """))

    # Notes (AI assistant output)
    db.session.execute(text("""
    CREATE TABLE IF NOT EXISTS notes (
      id BIGSERIAL PRIMARY KEY,
      patient_id TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      note_type TEXT NOT NULL, -- clinician|patient|handoff
      content TEXT NOT NULL,
      meta JSONB NOT NULL DEFAULT {}::jsonb
    );
    """))
    db.session.execute(text("""
    CREATE INDEX IF NOT EXISTS idx_notes_patient_time
    ON notes(patient_id, created_at DESC);
    """))

    db.session.commit()


# ----------------------------
# Helpers
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_patient(patient_id: str, full_name: Optional[str] = None) -> None:
    db.session.execute(text("""
    INSERT INTO patients (id, full_name)
    VALUES (:id, :full_name)
    ON CONFLICT (id) DO UPDATE
      SET full_name = COALESCE(EXCLUDED.full_name, patients.full_name);
    """), {"id": patient_id, "full_name": full_name})
    db.session.commit()


def add_observation(patient_id: str, payload: Dict[str, Any], observed_at: Optional[str] = None) -> int:
    observed_at = observed_at or utc_now_iso()
    r = db.session.execute(text("""
      INSERT INTO observations (patient_id, observed_at, payload)
      VALUES (:patient_id, :observed_at, :payload::jsonb)
      RETURNING id;
    """), {"patient_id": patient_id, "observed_at": observed_at, "payload": json.dumps(payload)})
    db.session.commit()
    return int(r.scalar())


def latest_observations(patient_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    rows = db.session.execute(text("""
      SELECT observed_at, payload
      FROM observations
      WHERE patient_id = :pid
      ORDER BY observed_at DESC
      LIMIT :lim;
    """), {"pid": patient_id, "lim": limit}).fetchall()
    out = []
    for (ts, payload) in rows:
        out.append({"observed_at": str(ts), "payload": payload})
    return out


# ----------------------------
# Risk scoring (plug-in)
# ----------------------------
def compute_simple_risk(payload: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
    """
    Demo-grade triage logic (replace with your ML model outputs).
    Returns: (label, prob, drivers)
    """
    hr = float(payload.get("heart_rate", 0) or 0)
    sys_bp = float(payload.get("systolic_bp", payload.get("sys_bp", 0)) or 0)
    dia_bp = float(payload.get("diastolic_bp", payload.get("dia_bp", 0)) or 0)
    bmi = float(payload.get("bmi", 0) or 0)
    age = float(payload.get("age", 0) or 0)

    score = 0.0
    drivers = {}

    if sys_bp >= 140:
        score += 0.25
        drivers["bp_systolic_high"] = sys_bp
    if dia_bp >= 90:
        score += 0.15
        drivers["bp_diastolic_high"] = dia_bp
    if hr >= 100:
        score += 0.20
        drivers["tachycardia"] = hr
    if bmi >= 30:
        score += 0.15
        drivers["obesity"] = bmi
    if age >= 60:
        score += 0.10
        drivers["age_risk"] = age

    prob = min(0.95, max(0.05, score))
    label = "High" if prob >= 0.45 else "Low"
    return label, float(prob), drivers


def write_risk_score(patient_id: str, label: str, prob: float, drivers: Dict[str, Any]) -> None:
    db.session.execute(text("""
      INSERT INTO risk_scores (patient_id, risk_label, risk_prob, drivers)
      VALUES (:pid, :label, :prob, :drivers::jsonb);
    """), {"pid": patient_id, "label": label, "prob": prob, "drivers": json.dumps(drivers)})
    db.session.commit()


def maybe_create_alerts_and_tasks(patient_id: str, label: str, prob: float, payload: Dict[str, Any]) -> None:
    # Alert thresholds (demo)
    if label != "High":
        return

    title = "High risk patient detected"
    details = f"Risk probability {round(prob*100,1)}%. Latest vitals: {payload}"

    db.session.execute(text("""
      INSERT INTO alerts (patient_id, severity, title, details)
      VALUES (:pid, high, :title, :details);
    """), {"pid": patient_id, "title": title, "details": details})

    db.session.execute(text("""
      INSERT INTO tasks (patient_id, task_type, details)
      VALUES (:pid, review_alert, :details);
    """), {"pid": patient_id, "details": "Review high-risk alert and determine next action."})

    db.session.commit()


# ----------------------------
# AI assistant outputs (no external LLM required)
# ----------------------------
def generate_clinician_note(patient_id: str, label: str, prob: float, drivers: Dict[str, Any], payload: Dict[str, Any]) -> str:
    # “Assistant note draft” format (Assessment/Plan)
    lines = []
    lines.append(f"Patient: {patient_id}")
    lines.append(f"Assessment: {label} risk (prob {round(prob*100,1)}%).")
    if drivers:
        lines.append(f"Drivers: {,

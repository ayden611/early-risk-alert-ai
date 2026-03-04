# era/api/routes.py
import os, time, json, secrets, math, re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import jwt
import numpy as np
import joblib
from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

# ------------------------------------------------------------
# Blueprint (MUST be defined before decorators)
# ------------------------------------------------------------
api_bp = Blueprint("api", __name__)

# ------------------------------------------------------------
# JSON error handler (prevents HTML 500s)
# ------------------------------------------------------------
@api_bp.errorhandler(Exception)
def _json_errors(e):
    current_app.logger.exception("Unhandled error")
    if isinstance(e, HTTPException):
        return jsonify({"error": "http", "code": e.code, "message": e.description}), e.code
    return jsonify({"error": "server", "message": str(e)}), 500

# ------------------------------------------------------------
# Optional DB models (routes must still boot if they don't exist)
# ------------------------------------------------------------
try:
    from era.models import Prediction  # type: ignore
except Exception:
    Prediction = None  # type: ignore

try:
    from era.models import HealthEvent  # type: ignore
except Exception:
    HealthEvent = None  # type: ignore

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _first(payload: dict, keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return default

def _exercise_to_num(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s in ("low", "l", "0"):
        return 0.0
    if s in ("moderate", "mod", "m", "1"):
        return 1.0
    if s in ("high", "h", "2"):
        return 2.0
    try:
        return float(s)
    except Exception:
        return 0.0

def _coerce_inputs(payload: dict) -> Tuple[float, float, float, float, float, float]:
    age = float(payload["age"])
    bmi = float(payload["bmi"])
    ex = _exercise_to_num(payload.get("exercise_level"))
    sbp = float(_first(payload, ["systolic_bp", "sys_bp", "sbp"], 0))
    dbp = float(_first(payload, ["diastolic_bp", "dia_bp", "dbp"], 0))
    hr = float(payload["heart_rate"])
    return age, bmi, ex, sbp, dbp, hr

def _ensure_pipeline_tables() -> None:
    # Creates tables if they don't exist (safe to call repeatedly)
    db.session.execute(text("""
        CREATE TABLE IF NOT EXISTS health_event (
            id BIGSERIAL PRIMARY KEY,
            user_id VARCHAR(64) NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """))
    db.session.execute(text("""
        CREATE TABLE IF NOT EXISTS health_anomaly (
            id BIGSERIAL PRIMARY KEY,
            user_id VARCHAR(64) NOT NULL,
            kind VARCHAR(50) NOT NULL,
            severity INT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """))
    db.session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_health_event_user_time
        ON health_event (user_id, created_at DESC)
    """))
    db.session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_health_anomaly_user_time
        ON health_anomaly (user_id, created_at DESC)
    """))
    db.session.commit()

def _log_event(user_id: str, event_type: str, payload: dict) -> None:
    db.session.execute(
        text("""
            INSERT INTO health_event (user_id, event_type, payload_json, created_at)
            VALUES (:user_id, :event_type, :payload_json, NOW())
        """),
        {"user_id": user_id, "event_type": event_type, "payload_json": json.dumps(payload)},
    )
    db.session.commit()

# ------------------------------------------------------------
# ✅ 20-line anomaly detection drop-in (DB + rule-based)
# Call _detect_anomalies(user_id, age,bmi,ex,sbp,dbp,hr) after parsing
# ------------------------------------------------------------
def _detect_anomalies(user_id: str, age: float, bmi: float, ex: float, sbp: float, dbp: float, hr: float) -> None:
    _ensure_pipeline_tables()
    severity, kind = 0, None
    if sbp >= 180 or dbp >= 120: severity, kind = 3, "hypertensive_crisis"
    elif sbp >= 160 or dbp >= 100: severity, kind = 2, "stage2_hypertension"
    elif sbp >= 140 or dbp >= 90: severity, kind = 1, "stage1_hypertension"
    if hr >= 120: severity, kind = max(severity, 2), (kind or "tachycardia")
    if hr <= 45: severity, kind = max(severity, 2), (kind or "bradycardia")
    if bmi >= 35: severity, kind = max(severity, 1), (kind or "bmi_high")
    if severity <= 0: return
    details = {"age": age, "bmi": bmi, "exercise": ex, "sbp": sbp, "dbp": dbp, "hr": hr}
    db.session.execute(text("""
        INSERT INTO health_anomaly (user_id, kind, severity, details_json, created_at)
        VALUES (:user_id, :kind, :severity, :details_json, NOW())
    """), {"user_id": user_id, "kind": kind, "severity": severity, "details_json": json.dumps(details)})
    db.session.commit()

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@api_bp.get("/healthz")
def healthz():
    try:
        db.session.execute(text("SELECT 1"))
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "degraded", "db_error": str(e)}), 503

@api_bp.post("/demo/intake")
def demo_intake():
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400

    try:
        age, bmi, ex, sbp, dbp, hr = _coerce_inputs(payload)
    except Exception:
        return jsonify({"error": "validation", "message": "Invalid numeric input"}), 400

    # event pipeline (always)
    try:
        _ensure_pipeline_tables()
        _log_event(user_id, "health_intake", payload)
    except Exception:
        current_app.logger.exception("Failed to write health_event (continuing)")

    # anomaly detection (drop-in)
    try:
        _detect_anomalies(user_id, age, bmi, ex, sbp, dbp, hr)
    except Exception:
        current_app.logger.exception("Failed anomaly detection (continuing)")

    # If your ML model exists elsewhere, keep response simple for now:
        return jsonify({"ok": True, "user_id": user_id}), 200
    return jsonify({"ok": True, "user_id": user_id}), 200

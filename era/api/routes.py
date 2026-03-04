# era/api/routes.py
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from sqlalchemy import text
from werkzeug.exceptions import HTTPException

from era.extensions import db

# ------------------------------------------------------------
# Blueprint MUST be defined before decorators
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
# Minimal DB pipeline tables (safe to call repeatedly)
# ------------------------------------------------------------
def _ensure_pipeline_tables() -> None:
    db.session.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS health_event (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(64) NOT NULL,
          event_type VARCHAR(50) NOT NULL,
          payload_json TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_health_event_user_created
          ON health_event(user_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS health_anomaly (
          id BIGSERIAL PRIMARY KEY,
          user_id VARCHAR(64) NOT NULL,
          kind VARCHAR(64) NOT NULL,
          severity INT NOT NULL DEFAULT 0,
          details_json TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS ix_health_anomaly_user_created
          ON health_anomaly(user_id, created_at DESC);
        """
        )
    )
    db.session.commit()


# ------------------------------------------------------------
# Voice intake parser (transcript -> payload)
# Accepts plain text transcript (not raw audio).
# Example transcript: "BP 140 over 90, heart rate 88, bmi 27.5, age 45, exercise moderate"
# ------------------------------------------------------------
_BP_RE = re.compile(r"\b(?:bp|blood pressure)\s*(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", re.I)
_SBP_RE = re.compile(r"\b(?:sbp|systolic)\s*(\d{2,3})\b", re.I)
_DBP_RE = re.compile(r"\b(?:dbp|diastolic)\s*(\d{2,3})\b", re.I)
_HR_RE = re.compile(r"\b(?:hr|heart rate|pulse)\s*(\d{2,3})\b", re.I)
_BMI_RE = re.compile(r"\b(?:bmi)\s*(\d{1,2}(?:\.\d+)?)\b", re.I)
_AGE_RE = re.compile(r"\b(?:age)\s*(\d{1,3})\b", re.I)
_EX_RE = re.compile(r"\b(?:exercise|activity)\s*(low|moderate|high)\b", re.I)


def _parse_voice_transcript(transcript: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    t = (transcript or "").strip()

    m = _BP_RE.search(t)
    if m:
        out["systolic_bp"] = float(m.group(1))
        out["diastolic_bp"] = float(m.group(2))
    else:
        m = _SBP_RE.search(t)
        if m:
            out["systolic_bp"] = float(m.group(1))
        m = _DBP_RE.search(t)
        if m:
            out["diastolic_bp"] = float(m.group(1))

    m = _HR_RE.search(t)
    if m:
        out["heart_rate"] = float(m.group(1))

    m = _BMI_RE.search(t)
    if m:
        out["bmi"] = float(m.group(1))

    m = _AGE_RE.search(t)
    if m:
        out["age"] = float(m.group(1))

    m = _EX_RE.search(t)
    if m:
        out["exercise_level"] = m.group(1).title()

    return out


# ------------------------------------------------------------
# Coercion helpers
# ------------------------------------------------------------
def _first(payload: dict, keys, default=None):
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
    if s in ("low", "l"):
        return 0.0
    if s in ("moderate", "mod", "m", "medium"):
        return 1.0
    if s in ("high", "h", "vigorous"):
        return 2.0
    # fallback: try float
    try:
        return float(s)
    except Exception:
        return 0.0


def _coerce_inputs(payload: dict) -> Tuple[float, float, float, float, float, float]:
    age = float(payload.get("age"))
    bmi = float(payload.get("bmi"))
    ex = _exercise_to_num(payload.get("exercise_level"))
    sbp = float(_first(payload, ["systolic_bp", "sys_bp", "sbp"], 0))
    dbp = float(_first(payload, ["diastolic_bp", "dia_bp", "dbp"], 0))
    hr = float(payload.get("heart_rate"))
    return age, bmi, ex, sbp, dbp, hr


# ------------------------------------------------------------
# AI Health Monitoring Engine (real-time)
# 1) Rule-based anomalies (immediate safety alerts)
# 2) Trend-based alerts (user’s last readings)
# ------------------------------------------------------------
def _detect_anomaly(age: float, bmi: float, sbp: float, dbp: float, hr: float) -> Optional[Dict[str, Any]]:
    severity, kind = 0, None

    # BP categories
    if sbp >= 180 or dbp >= 120:
        severity, kind = 3, "hypertensive_crisis"
    elif sbp >= 160 or dbp >= 100:
        severity, kind = 2, "stage2_hypertension"
    elif sbp >= 140 or dbp >= 90:
        severity, kind = 1, "stage1_hypertension"

    # HR flags
    if hr >= 120:
        severity, kind = max(severity, 2), kind or "tachycardia"
    if hr <= 45:
        severity, kind = max(severity, 2), kind or "bradycardia"

    # BMI flags
    if bmi >= 35:
        severity, kind = max(severity, 1), kind or "bmi_high"

    if severity <= 0 or not kind:
        return None

    return {
        "kind": kind,
        "severity": severity,
        "message": {
            3: "Critical reading detected. Consider seeking urgent medical attention.",
            2: "High-risk reading detected. Consider follow-up and monitoring.",
            1: "Elevated reading detected. Consider lifestyle review and monitoring.",
        }.get(severity, "Abnormal reading detected."),
    }


def _health_trend_alert(user_id: str, sbp: float, dbp: float, hr: float) -> Optional[Dict[str, Any]]:
    # Pull last 5 events and look for fast rise in SBP or HR
    rows = db.session.execute(
        text(
            """
        SELECT payload_json
        FROM health_event
        WHERE user_id = :user_id AND event_type IN ('health_intake','voice_intake')
        ORDER BY created_at DESC
        LIMIT 5
        """
        ),
        {"user_id": user_id},
    ).fetchall()

    if len(rows) < 3:
        return None

    sbps, hrs = [], []
    for (pj,) in rows:
        try:
            p = json.loads(pj)
        except Exception:
            continue
        sbps.append(float(p.get("systolic_bp", 0) or 0))
        hrs.append(float(p.get("heart_rate", 0) or 0))

    if len(sbps) >= 3:
        # compare newest vs oldest in window
        if (sbps[0] - sbps[-1]) >= 20:
            return {
                "alert": "Blood pressure rising rapidly",
                "risk_level": "high",
                "recommendation": "Recheck BP soon and consider medical guidance if symptoms.",
            }

    if len(hrs) >= 3:
        if (hrs[0] - hrs[-1]) >= 20:
            return {
                "alert": "Heart rate rising rapidly",
                "risk_level": "medium",
                "recommendation": "Rest and recheck HR; seek guidance if symptoms occur.",
            }

    return None


def _write_event(user_id: str, event_type: str, payload: dict) -> None:
    db.session.execute(
        text(
            """
        INSERT INTO health_event(user_id, event_type, payload_json, created_at)
        VALUES (:user_id, :event_type, :payload_json, NOW())
        """
        ),
        {"user_id": user_id, "event_type": event_type, "payload_json": json.dumps(payload)},
    )
    db.session.commit()


def _write_anomaly(user_id: str, kind: str, severity: int, details: dict) -> None:
    db.session.execute(
        text(
            """
        INSERT INTO health_anomaly(user_id, kind, severity, details_json, created_at)
        VALUES (:user_id, :kind, :severity, :details_json, NOW())
        """
        ),
        {"user_id": user_id, "kind": kind, "severity": int(severity), "details_json": json.dumps(details)},
    )
    db.session.commit()


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@api_bp.get("/healthz")
def healthz():
    db_ok = True
    err = None
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        err = str(e)
    return jsonify({"status": "ok" if db_ok else "degraded", "db_ok": db_ok, "db_error": err})


@api_bp.post("/api/v1/demo/intake")
def demo_intake():
    _ensure_pipeline_tables()

    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400

    try:
        age, bmi, ex, sbp, dbp, hr = _coerce_inputs(payload)
    except Exception:
        return jsonify({"error": "validation", "message": "Invalid numeric input"}), 400

    # Normalize payload we store
    stored = {
        "user_id": user_id,
        "age": age,
        "bmi": bmi,
        "exercise_level": payload.get("exercise_level", ""),
        "exercise_level_num": ex,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "heart_rate": hr,
        "source": "typed",
    }

    _write_event(user_id, "health_intake", stored)

    anomaly = _detect_anomaly(age=age, bmi=bmi, sbp=sbp, dbp=dbp, hr=hr)
    if anomaly:
        _write_anomaly(
            user_id=user_id,
            kind=anomaly["kind"],
            severity=anomaly["severity"],
            details={"message": anomaly["message"], "reading": {"sbp": sbp, "dbp": dbp, "hr": hr, "bmi": bmi, "age": age}},
        )

    trend = _health_trend_alert(user_id, sbp, dbp, hr)

    return jsonify(
        {
            "ok": True,
            "user_id": user_id,
            "saved": True,
            "anomaly": anomaly,
            "trend_alert": trend,
        }
    )


@api_bp.post("/api/v1/voice/intake")
def voice_intake():
    """
    Voice intake endpoint (text transcript in, structured intake out).
    Send JSON like:
      {"user_id":"123","transcript":"BP 140 over 90 heart rate 88 bmi 27.5 age 45 exercise moderate"}
    """
    _ensure_pipeline_tables()

    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    transcript = str(payload.get("transcript") or "").strip()

    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400
    if not transcript:
        return jsonify({"error": "validation", "message": "transcript is required"}), 400

    parsed = _parse_voice_transcript(transcript)

    # merge transcript-derived fields into payload shape required by the engine
    merged = {
        "user_id": user_id,
        "age": parsed.get("age", payload.get("age", 0)),
        "bmi": parsed.get("bmi", payload.get("bmi", 0)),
        "exercise_level": parsed.get("exercise_level", payload.get("exercise_level", "")),
        "systolic_bp": parsed.get("systolic_bp", payload.get("systolic_bp", 0)),
        "diastolic_bp": parsed.get("diastolic_bp", payload.get("diastolic_bp", 0)),
        "heart_rate": parsed.get("heart_rate", payload.get("heart_rate", 0)),
        "transcript": transcript,
        "source": "voice",
    }

    try:
        age, bmi, ex, sbp, dbp, hr = _coerce_inputs(merged)
    except Exception:
        return jsonify({"error": "validation", "message": "Transcript parsed but numeric input invalid"}), 400

    stored = {
        "user_id": user_id,
        "age": age,
        "bmi": bmi,
        "exercise_level": merged.get("exercise_level", ""),
        "exercise_level_num": ex,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "heart_rate": hr,
        "source": "voice",
        "transcript": transcript,
    }

    _write_event(user_id, "voice_intake", stored)

    anomaly = _detect_anomaly(age=age, bmi=bmi, sbp=sbp, dbp=dbp, hr=hr)
    if anomaly:
        _write_anomaly(
            user_id=user_id,
            kind=anomaly["kind"],
            severity=anomaly["severity"],
            details={"message": anomaly["message"], "reading": {"sbp": sbp, "dbp": dbp, "hr": hr, "bmi": bmi, "age": age}},
        )

    trend = _health_trend_alert(user_id, sbp, dbp, hr)

    return jsonify(
        {
            "ok": True,
            "user_id": user_id,
            "parsed": parsed,  # what we extracted from transcript
            "saved": True,
            "anomaly": anomaly,
            "trend_alert": trend,
        }
    )


# Optional: quick SSE stream to watch latest anomalies (for dashboards)
@api_bp.get("/api/v1/stream/anomalies")
def stream_anomalies():
    _ensure_pipeline_tables()

    user_id = str(request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400

    def gen():
        last_id = 0
        while True:
            rows = db.session.execute(
                text(
                    """
                SELECT id, created_at, kind, severity, details_json
                FROM health_anomaly
                WHERE user_id = :user_id AND id > :last_id
                ORDER BY id ASC
                LIMIT 10
                """
                ),
                {"user_id": user_id, "last_id": last_id},
            ).fetchall()

            for rid, created_at, kind, severity, details_json in rows:
                last_id = max(last_id, int(rid))
                yield f"data: {json.dumps({'id': rid, 'created_at': str(created_at), 'kind': kind, 'severity': severity, 'details': json.loads(details_json)})}\n\n"

            time.sleep(1.0)

    return Response(stream_with_context(gen()), mimetype="text/event-stream")

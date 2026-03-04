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

# Optional ORM model (if you have it)
try:
    from era.models import HealthEvent  # type: ignore
except Exception:
    HealthEvent = None  # type: ignore


api_bp = Blueprint("api", __name__)

# ----------------------------
# Error handler (JSON)
# ----------------------------
@api_bp.errorhandler(Exception)
def _json_errors(e):
    current_app.logger.exception("Unhandled error")
    if isinstance(e, HTTPException):
        return jsonify({"error": "http", "code": e.code, "message": e.description}), e.code
    return jsonify({"error": "server", "message": str(e)}), 500


# ----------------------------
# Helpers
# ----------------------------
_BP_RE = re.compile(r"\b(?:bp|blood\s*pressure)\s*(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", re.I)
_HR_RE = re.compile(r"\b(?:hr|heart\s*rate)\s*(\d{2,3})\b", re.I)
_BMI_RE = re.compile(r"\b(?:bmi)\s*(\d{1,2}(?:\.\d+)?)\b", re.I)
_AGE_RE = re.compile(r"\b(?:age)\s*(\d{1,3})\b", re.I)
_EX_RE = re.compile(r"\b(?:exercise)\s*(low|moderate|high)\b", re.I)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _exercise_to_num(v: Any) -> float:
    s = str(v or "").strip().lower()
    if s in ("0", "low"):
        return 0.0
    if s in ("1", "moderate", "mod"):
        return 1.0
    if s in ("2", "high"):
        return 2.0
    # if user passes numeric
    f = _safe_float(v, None)
    return float(f) if f is not None else 0.0


def _first(payload: dict, keys, default=None):
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return default


def _coerce_inputs(payload: dict) -> Tuple[float, float, float, float, float, float]:
    age = float(_first(payload, ["age"], 0) or 0)
    bmi = float(_first(payload, ["bmi"], 0) or 0)
    ex = _exercise_to_num(_first(payload, ["exercise_level", "exercise"], ""))
    sbp = float(_first(payload, ["systolic_bp", "sys_bp", "sbp"], 0) or 0)
    dbp = float(_first(payload, ["diastolic_bp", "dia_bp", "dbp"], 0) or 0)
    hr = float(_first(payload, ["heart_rate", "hr"], 0) or 0)
    return age, bmi, ex, sbp, dbp, hr


def _parse_transcript(transcript: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    t = (transcript or "").strip()

    m = _BP_RE.search(t)
    if m:
        out["systolic_bp"] = float(m.group(1))
        out["diastolic_bp"] = float(m.group(2))

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
        out["exercise_level"] = m.group(1).capitalize()

    return out


def _ensure_pipeline_tables():
    """Creates lightweight tables if they don't exist (Postgres or SQLite)."""
    dialect = db.engine.dialect.name

    if dialect == "postgresql":
        db.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS health_anomaly (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        )
        db.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS health_event (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        )
    else:
        # sqlite
        db.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS health_anomaly (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
        )
        db.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS health_event (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
        )
    db.session.commit()


def _insert_event(user_id: str, event_type: str, data: Dict[str, Any]) -> None:
    """Save event either via ORM HealthEvent or fallback table."""
    if HealthEvent is not None:
        try:
            ev = HealthEvent(user_id=user_id, event_type=event_type, data=data)  # type: ignore
            db.session.add(ev)
            db.session.commit()
            return
        except Exception:
            db.session.rollback()

    # fallback raw insert
    created_at = _now_utc().isoformat()
    db.session.execute(
        text(
            """
            INSERT INTO health_event (user_id, event_type, data_json, created_at)
            VALUES (:user_id, :event_type, :data_json, :created_at)
            """
        ),
        {
            "user_id": user_id,
            "event_type": event_type,
            "data_json": json.dumps(data),
            "created_at": created_at,
        },
    )
    db.session.commit()


def _linear_slope(xs, ys) -> float:
    # simple least squares slope, small n safe
    n = len(xs)
    if n < 2:
        return 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    return float(num / den) if den else 0.0


def _trend_alert(user_id: str) -> Optional[Tuple[str, int, Dict[str, Any]]]:
    """
    Trend alert if SBP or HR is rising fast over last N events.
    Returns (kind, severity, details) or None.
    """
    # pull last 6 intake events
    rows = db.session.execute(
        text(
            """
            SELECT id, data, created_at
            FROM health_event
            WHERE user_id = :user_id AND event_type IN ('health_intake','voice_intake')
            ORDER BY id DESC
            LIMIT 6
            """
        ),
        {"user_id": user_id},
    ).fetchall()

    if len(rows) < 4:
        return None

    rows = list(reversed(rows))  # oldest -> newest
    xs = list(range(len(rows)))

    sbps, hrs = [], []
    for r in rows:
        try:
            d = json.loads(r[1] or "{}")
        except Exception:
            d = {}
        sbps.append(float(d.get("systolic_bp", 0) or 0))
        hrs.append(float(d.get("heart_rate", 0) or 0))

    sbp_slope = _linear_slope(xs, sbps)  # per-event change
    hr_slope = _linear_slope(xs, hrs)

    # thresholds (tune later)
    if sbp_slope >= 8:  # e.g., +8 mmHg each event
        return (
            "rapid_bp_rise",
            2,
            {"sbp_slope": sbp_slope, "last_sbp": sbps[-1], "n": len(rows)},
        )
    if hr_slope >= 6:
        return (
            "rapid_hr_rise",
            2,
            {"hr_slope": hr_slope, "last_hr": hrs[-1], "n": len(rows)},
        )
    return None


def _rule_anomaly(age: float, bmi: float, sbp: float, dbp: float, hr: float) -> Optional[Tuple[str, int, Dict[str, Any]]]:
    """
    20-ish line real-time anomaly detection engine (rules-based).
    Returns (kind, severity, details) or None.
    """
    severity, kind = 0, None

    if sbp >= 180 or dbp >= 120:
        severity, kind = 3, "hypertensive_crisis"
    elif sbp >= 160 or dbp >= 100:
        severity, kind = 2, "stage2_hypertension"
    elif sbp >= 140 or dbp >= 90:
        severity, kind = 1, "stage1_hypertension"

    if hr >= 120:
        severity, kind = max(severity, 2), kind or "tachycardia"
    if hr <= 45 and hr > 0:
        severity, kind = max(severity, 2), kind or "bradycardia"

    if bmi >= 35:
        severity, kind = max(severity, 1), kind or "bmi_high"

    if severity <= 0 or not kind:
        return None

    details = {"age": age, "bmi": bmi, "sbp": sbp, "dbp": dbp, "hr": hr}
    return kind, severity, details


def _insert_anomaly(user_id: str, kind: str, severity: int, details: Dict[str, Any]) -> int:
    """Insert anomaly and return anomaly id."""
    created_at = _now_utc().isoformat()
    details_json = json.dumps(details)

    # Postgres can RETURNING id; SQLite can't (in same way), so handle both.
    dialect = db.engine.dialect.name
    if dialect == "postgresql":
        row = db.session.execute(
            text(
                """
                INSERT INTO health_anomaly (user_id, kind, severity, details_json, created_at)
                VALUES (:user_id, :kind, :severity, :details_json, NOW())
                RETURNING id
                """
            ),
            {
                "user_id": user_id,
                "kind": kind,
                "severity": severity,
                "details_json": details_json,
            },
        ).fetchone()
        db.session.commit()
        return int(row[0]) if row else 0

    db.session.execute(
        text(
            """
            INSERT INTO health_anomaly (user_id, kind, severity, details_json, created_at)
            VALUES (:user_id, :kind, :severity, :details_json, :created_at)
            """
        ),
        {
            "user_id": user_id,
            "kind": kind,
            "severity": severity,
            "details_json": details_json,
            "created_at": created_at,
        },
    )
    db.session.commit()
    row = db.session.execute(text("SELECT last_insert_rowid()")).fetchone()
    return int(row[0]) if row else 0


def _maybe_create_anomalies(user_id: str, age: float, bmi: float, sbp: float, dbp: float, hr: float) -> None:
    # rule-based anomaly
    r = _rule_anomaly(age, bmi, sbp, dbp, hr)
    if r:
        kind, severity, details = r
        _insert_anomaly(user_id, kind, severity, details)

    # trend-based anomaly (optional)
    t = _trend_alert(user_id)
    if t:
        kind, severity, details = t
        _insert_anomaly(user_id, kind, severity, details)


# ----------------------------
# Routes
# ----------------------------
@api_bp.get("/healthz")
def healthz():
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_ok = False
        return jsonify({"status": "degraded", "db_ok": False, "error": str(e)}), 503
    return jsonify({"status": "ok", "db_ok": db_ok}), 200


@api_bp.post("/demo/intake")
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

    # save event
    _insert_event(
        user_id,
        "health_intake",
        {
            "age": age,
            "bmi": bmi,
            "exercise_level": ex,
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "heart_rate": hr,
        },
    )

    # anomalies
    _maybe_create_anomalies(user_id, age, bmi, sbp, dbp, hr)

    return jsonify({"ok": True, "user_id": user_id}), 200


@api_bp.post("/voice/intake")
def voice_intake():
    _ensure_pipeline_tables()

    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "").strip()
    transcript = str(payload.get("transcript") or "").strip()

    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400
    if not transcript:
        return jsonify({"error": "validation", "message": "transcript is required"}), 400

    extracted = _parse_transcript(transcript)
    # merge: explicit json fields override transcript, transcript fills missing
    merged = dict(extracted)
    for k, v in payload.items():
        if k in ("user_id", "transcript"):
            continue
        if v is not None and v != "":
            merged[k] = v

    try:
        age, bmi, ex, sbp, dbp, hr = _coerce_inputs(merged)
    except Exception:
        return jsonify({"error": "validation", "message": "Invalid numeric input"}), 400

    _insert_event(
        user_id,
        "voice_intake",
        {
            "transcript": transcript,
            "age": age,
            "bmi": bmi,
            "exercise_level": ex,
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "heart_rate": hr,
        },
    )

    _maybe_create_anomalies(user_id, age, bmi, sbp, dbp, hr)

    return jsonify({"ok": True, "user_id": user_id, "parsed": extracted}), 200


@api_bp.get("/anomalies/latest")
def latest_anomaly():
    _ensure_pipeline_tables()

    user_id = str(request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400

    row = db.session.execute(
        text(
            """
            SELECT id, user_id, kind, severity, details_json, created_at
            FROM health_anomaly
            WHERE user_id = :user_id
            ORDER BY id DESC
            LIMIT 1
            """
        ),
        {"user_id": user_id},
    ).fetchone()

    if not row:
        return jsonify({"ok": True, "anomaly": None}), 200

    anomaly = {
        "id": int(row[0]),
        "user_id": row[1],
        "kind": row[2],
        "severity": int(row[3]),
        "details": json.loads(row[4] or "{}"),
        "created_at": str(row[5]),
    }
    return jsonify({"ok": True, "anomaly": anomaly}), 200


@api_bp.get("/stream/anomalies")
def stream_anomalies():
    """
    SSE stream:
      - Only streams NEW anomalies after connect (by using last_id baseline)
      - Sends ping heartbeats
    """
    _ensure_pipeline_tables()

    user_id = str(request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "validation", "message": "user_id is required"}), 400

    # baseline (do NOT replay history)
    row = db.session.execute(
        text("SELECT COALESCE(MAX(id), 0) FROM health_anomaly WHERE user_id = :user_id"),
        {"user_id": user_id},
    ).fetchone()
    last_id = int(row[0] or 0)

    def gen():
        nonlocal last_id
        yield "retry: 2000\n\n"
        last_ping = time.time()

        while True:
            # fetch NEW rows
            rows = db.session.execute(
                text(
                    """
                    SELECT id, kind, severity, details_json, created_at
                    FROM health_anomaly
                    WHERE user_id = :user_id AND id > :last_id
                    ORDER BY id ASC
                    """
                ),
                {"user_id": user_id, "last_id": last_id},
            ).fetchall()

            for r in rows:
                last_id = int(r[0])
                data = {
                    "id": last_id,
                    "user_id": user_id,
                    "kind": r[1],
                    "severity": int(r[2]),
                    "details": json.loads(r[3] or "{}"),
                    "created_at": str(r[4]),
                }
                yield f"event: anomaly\ndata: {json.dumps(data)}\n\n"

            # heartbeat
            if time.time() - last_ping >= 2.0:
                yield f"event: ping\ndata: {json.dumps({'ts': _now_utc().isoformat()})}\n\n"
                last_ping = time.time()

            time.sleep(0.5)

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

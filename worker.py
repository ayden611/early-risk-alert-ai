"""
Background worker: "continuous monitoring" engine
"""
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import text

from era import create_app
from era.extensions import db

try:
    from era.ai.health_reasoning import generate_health_reasoning  # type: ignore
except Exception:
    generate_health_reasoning = None  # type: ignore


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _ensure_pipeline_tables() -> None:
    dialect = db.engine.dialect.name

    if dialect == "postgresql":
        db.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS health_event (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    data_json TEXT
                );
                """
            )
        )
        db.session.execute(text("ALTER TABLE health_event ADD COLUMN IF NOT EXISTS data_json TEXT;"))
    else:
        db.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS health_event (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data_json TEXT
                );
                """
            )
        )
        cols = db.session.execute(text("PRAGMA table_info(health_event);")).fetchall()
        col_names = {c[1] for c in cols}
        if "data_json" not in col_names:
            db.session.execute(text("ALTER TABLE health_event ADD COLUMN data_json TEXT;"))

    db.session.commit()


def _insert_event(user_id: str, event_type: str, payload: Dict[str, Any]) -> int:
    _ensure_pipeline_tables()
    created_at = _now_utc()

    if db.engine.dialect.name == "postgresql":
        row = db.session.execute(
            text(
                """
                INSERT INTO health_event (user_id, event_type, created_at, data_json)
                VALUES (:user_id, :event_type, :created_at, :data_json)
                RETURNING id;
                """
            ),
            {"user_id": user_id, "event_type": event_type, "created_at": created_at, "data_json": json.dumps(payload)},
        ).fetchone()
        db.session.commit()
        return int(row[0])

    db.session.execute(
        text(
            """
            INSERT INTO health_event (user_id, event_type, created_at, data_json)
            VALUES (:user_id, :event_type, :created_at, :data_json);
            """
        ),
        {"user_id": user_id, "event_type": event_type, "created_at": created_at.isoformat(), "data_json": json.dumps(payload)},
    )
    db.session.commit()
    rid = db.session.execute(text("SELECT last_insert_rowid();")).fetchone()
    return int(rid[0])


def _rule_anomalies(details: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    sbp = _safe_float(details.get("systolic_bp") or details.get("sbp"))
    dbp = _safe_float(details.get("diastolic_bp") or details.get("dbp"))
    hr = _safe_float(details.get("heart_rate") or details.get("hr"))

    kind = None
    severity = 0

    if sbp is not None and dbp is not None:
        if sbp >= 180 or dbp >= 120:
            kind = "hypertensive_crisis"
            severity = 3
        elif sbp >= 140 or dbp >= 90:
            kind = "stage1_hypertension"
            severity = 2
        elif sbp >= 130 or dbp >= 80:
            kind = "elevated_bp"
            severity = 1

    if hr is not None:
        if hr >= 130:
            kind = kind or "tachycardia"
            severity = max(severity, 2)
        elif hr <= 45:
            kind = kind or "bradycardia"
            severity = max(severity, 2)

    if not kind:
        return None, {"status": "ok"}

    anomaly = {
        "kind": kind,
        "severity": severity,
        "details": {"sbp": sbp, "dbp": dbp, "hr": hr},
        "ts": _now_utc().isoformat(),
        "source": "worker",
    }
    return anomaly, {"status": "anomaly"}


def _load_json(s: Any) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def run_loop(poll_seconds: float = 2.0) -> None:
    _ensure_pipeline_tables()

    latest = db.session.execute(
        text(
            """
            SELECT COALESCE(MAX(id), 0)
            FROM health_event
            WHERE event_type IN ('health_intake', 'voice_intake');
            """
        )
    ).fetchone()
    last_seen = int(latest[0] or 0)

    processed = set()

    while True:
        rows = db.session.execute(
            text(
                """
                SELECT id, user_id, event_type, created_at, data_json
                FROM health_event
                WHERE id > :last_seen
                  AND event_type IN ('health_intake','voice_intake')
                ORDER BY id ASC
                LIMIT 200;
                """
            ),
            {"last_seen": last_seen},
        ).fetchall()

        if rows:
            for r in rows:
                rid = int(r[0])
                user_id = str(r[1])
                data_json = r[4]
                last_seen = max(last_seen, rid)

                if rid in processed:
                    continue
                processed.add(rid)

                payload = _load_json(data_json)
                anomaly, _ = _rule_anomalies(payload)
                if anomaly:
                    anomaly["intake_event_id"] = rid
                    _insert_event(user_id, "anomaly", anomaly)

                if generate_health_reasoning is not None:
                    try:
                        analysis = generate_health_reasoning(user_id)
                        _insert_event(
                            user_id,
                            "ai_reasoning",
                            {
                                "ts": _now_utc().isoformat(),
                                "user_id": user_id,
                                "intake_event_id": rid,
                                "analysis": analysis,
                            },
                        )
                    except Exception as e:
                        _insert_event(
                            user_id,
                            "ai_reasoning_error",
                            {"ts": _now_utc().isoformat(), "error": str(e)},
                        )

        time.sleep(poll_seconds)


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        print("[worker] starting continuous monitoring loop...")
        run_loop(poll_seconds=float(os.getenv("WORKER_POLL_SECONDS", "2.0")))

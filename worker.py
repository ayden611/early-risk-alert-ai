# worker.py
import os
import json
import time
from datetime import datetime, timezone

from sqlalchemy import text

from era import create_app
from era.extensions import db


POLL_SECONDS = int(os.getenv("WORKER_POLL_SECONDS", "5"))
BATCH_SIZE = int(os.getenv("WORKER_BATCH_SIZE", "25"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_worker_tables():
    # Store generated summaries so the web app/mobile can fetch them
    db.session.execute(text("""
    CREATE TABLE IF NOT EXISTS health_summary (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        window_days INT NOT NULL DEFAULT 30,
        summary_text TEXT NOT NULL,
        source_event_id BIGINT,
        model_version TEXT,
        meta_json TEXT
    );
    """))
    db.session.execute(text("""
    CREATE INDEX IF NOT EXISTS idx_health_summary_user_time
    ON health_summary(user_id, created_at DESC);
    """))
    db.session.commit()


def _generate_summary_from_payload(payload: dict) -> str:
    """
    'ChatGPT-like' but deterministic (safe + reliable).
    We'll get much smarter over time, but this already feels personal.
    """
    user_id = payload.get("user_id", "")
    risk_label = payload.get("risk_label", "Unknown")
    prob = payload.get("probability", None)
    anomaly = payload.get("anomaly", {}) or {}
    baseline = payload.get("baseline", {}) or {}

    parts = []
    parts.append(f"Summary for user {user_id}: {risk_label}.")
    if prob is not None:
        parts.append(f"Model-estimated high-risk probability: {round(float(prob)*100, 1)}%.")

    # Trend baseline hints
    sbp = baseline.get("baseline_avg_sbp")
    dbp_ = baseline.get("baseline_avg_dbp")
    hr = baseline.get("baseline_avg_hr")
    if sbp and dbp_:
        parts.append(f"Recent baseline blood pressure: ~{round(sbp)} / {round(dbp_)}.")
    if hr:
        parts.append(f"Recent baseline heart rate: ~{round(hr)} bpm.")

    # Anomaly callouts
    if anomaly.get("has_anomaly"):
        alerts = anomaly.get("alerts", [])
        parts.append("Notable change detected:")
        for a in alerts[:3]:
            parts.append(f"- {a.get('metric')}: {a.get('current')} vs baseline {a.get('baseline')} ({a.get('reason')}).")
    else:
        parts.append("No major deviations detected compared to your recent baseline.")

    parts.append("Tip: Track readings consistently; trends matter more than one measurement.")
    parts.append("Educational only — not medical advice.")
    return " ".join(parts)


def _mark_delivered(outbox_id: int):
    db.session.execute(
        text("UPDATE event_outbox SET delivered = TRUE WHERE id = :id"),
        {"id": outbox_id},
    )
    db.session.commit()


def _insert_summary(user_id: str, text_summary: str, source_event_id=None, model_version=None, meta=None):
    db.session.execute(
        text("""
        INSERT INTO health_summary (user_id, window_days, summary_text, source_event_id, model_version, meta_json)
        VALUES (:user_id, 30, :summary_text, :source_event_id, :model_version, :meta_json)
        """),
        {
            "user_id": user_id,
            "summary_text": text_summary,
            "source_event_id": source_event_id,
            "model_version": model_version,
            "meta_json": json.dumps(meta or {}),
        },
    )
    db.session.commit()


def run():
    app = create_app()
    with app.app_context():
        _ensure_worker_tables()
        print(f"[worker] started at {_utc_now_iso()} poll={POLL_SECONDS}s batch={BATCH_SIZE}")

        while True:
            try:
                rows = db.session.execute(
                    text("""
                    SELECT id, topic, payload_json, created_at
                    FROM event_outbox
                    WHERE delivered = FALSE
                    ORDER BY created_at ASC
                    LIMIT :n
                    """),
                    {"n": BATCH_SIZE},
                ).fetchall()

                if not rows:
                    time.sleep(POLL_SECONDS)
                    continue

                for (outbox_id, topic, payload_json, created_at) in rows:
                    try:
                        payload = json.loads(payload_json) if payload_json else {}
                    except Exception:
                        payload = {}

                    # We currently summarize ingested readings. Expand topics later.
                    if topic == "health.reading.ingested":
                        user_id = str(payload.get("user_id", "")).strip()
                        if user_id:
                            summary = _generate_summary_from_payload(payload)
                            _insert_summary(
                                user_id=user_id,
                                text_summary=summary,
                                source_event_id=None,
                                model_version=payload.get("model_version"),
                                meta={"topic": topic, "outbox_id": outbox_id},
                            )

                    _mark_delivered(int(outbox_id))

            except Exception as e:
                # Don't crash; back off and keep going
                try:
                    db.session.rollback()
                except Exception:
                    pass
                print("[worker] error:", repr(e))
                time.sleep(max(2, POLL_SECONDS))


if __name__ == "__main__":
    run()

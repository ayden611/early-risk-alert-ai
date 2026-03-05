# worker.py  (Postgres queue worker: batch + SKIP LOCKED)
import os, time, json, math
from sqlalchemy import text
from era import create_app
from era.extensions import db

POLL = float(os.getenv("WORKER_POLL_SECONDS", "0.5"))
BATCH = int(os.getenv("WORKER_BATCH_SIZE", "200"))

CLAIM = text("""
WITH c AS (
  SELECT id FROM risk_jobs
  WHERE status='pending'
  ORDER BY id
  LIMIT :n
  FOR UPDATE SKIP LOCKED
)
UPDATE risk_jobs j
SET status='processing', started_at=NOW()
FROM c
WHERE j.id=c.id
RETURNING j.id, j.tenant_id, j.patient_id, j.payload_json;
""")

DONE = text("UPDATE risk_jobs SET status='done', finished_at=NOW(), result_json=:r WHERE id=:id;")
FAIL = text("UPDATE risk_jobs SET status='failed', finished_at=NOW(), error=:e WHERE id=:id;")

def score(payload: dict) -> dict:
    age = float(payload.get("age", 45))
    bmi = float(payload.get("bmi", 27))
    sbp = float(payload.get("systolic_bp", payload.get("sys_bp", 120)))
    dbp = float(payload.get("diastolic_bp", payload.get("dia_bp", 80)))
    hr  = float(payload.get("heart_rate", 72))
    x = -8.0 + 0.03*age + 0.06*bmi + 0.02*sbp + 0.01*dbp + 0.01*hr
    p = 1.0 / (1.0 + math.exp(-x))
    label = "high" if p >= 0.5 else "low"
    note = "Review meds/lifestyle; consider follow-up" if label=="high" else "Continue monitoring"
    return {"risk": label, "prob": round(p, 4), "note": note}

def main():
    app = create_app()
    with app.app_context():
        while True:
            jobs = db.session.execute(CLAIM, {"n": BATCH}).mappings().all()
            if not jobs:
                time.sleep(POLL); continue
            for j in jobs:
                try:
                    payload = json.loads(j["payload_json"] or "{}")
                    result = score(payload)
                    db.session.execute(DONE, {"id": j["id"], "r": json.dumps(result)})
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    db.session.execute(FAIL, {"id": j["id"], "e": str(e)[:500]})
                    db.session.commit()

if __name__ == "__main__":
    main()

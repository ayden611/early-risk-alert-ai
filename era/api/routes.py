from flask import Blueprint, request, jsonify, current_app
from pydantic import BaseModel, Field, ValidationError

from ..extensions import db
from ..models import Prediction
from ..services.predict import predict_risk

api_bp = Blueprint("api", __name__)


class PredictIn(BaseModel):
    age: float = Field(..., ge=1, le=120)
    bmi: float = Field(..., ge=10, le=80)
    exercise_level: str
    systolic_bp: float = Field(..., ge=70, le=260)
    diastolic_bp: float = Field(..., ge=40, le=160)
    heart_rate: float = Field(..., ge=30, le=220)

    # anonymous identity support (mobile + web can pass a stable id)
    session_id: str | None = None


@api_bp.get("/health")
def health():
    # basic health: app reachable + model loaded
    return {"ok": True}, 200


@api_bp.post("/predict")
def api_predict():
    data = request.get_json(silent=True) or {}

    try:
        payload = PredictIn(**data)
    except ValidationError as e:
        return {"error": "validation", "details": e.errors()}, 400

    label, p, raw_p = predict_risk(
        payload.age,
        payload.bmi,
        payload.exercise_level,
        payload.systolic_bp,
        payload.diastolic_bp,
        payload.heart_rate,
    )

    # Save to DB
    row = Prediction(
        user_id=None,
        session_id=payload.session_id,
        age=payload.age,
        bmi=payload.bmi,
        exercise_level=payload.exercise_level,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        heart_rate=payload.heart_rate,
        risk_label=label,
        probability=p,
    )
    db.session.add(row)
    db.session.commit()

    return jsonify(
        {
            "risk_label": label,
            "probability": round(p * 100, 1),
            "probability_raw": round(raw_p * 100, 1),
            "created_at": row.created_at.isoformat(),
        }
    ), 200


@api_bp.get("/history")
def api_history():
    session_id = request.args.get("session_id")
    risk = request.args.get("risk", "All")
    page = int(request.args.get("page", "1"))
    per_page = min(max(int(request.args.get("per_page", "10")), 5), 50)

    q = Prediction.query

    # anonymous user history
    if session_id:
        q = q.filter(Prediction.session_id == session_id)

    if risk in ("High", "Low"):
        q = q.filter(Prediction.risk_label == risk)

    q = q.order_by(Prediction.created_at.desc())

    total = q.count()
    rows = q.offset((page - 1) * per_page).limit(per_page).all()

    return jsonify(
        {
            "total": total,
            "page": page,
            "per_page": per_page,
            "rows": [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat(),
                    "age": r.age,
                    "bmi": r.bmi,
                    "exercise_level": r.exercise_level,
                    "systolic_bp": r.systolic_bp,
                    "diastolic_bp": r.diastolic_bp,
                    "heart_rate": r.heart_rate,
                    "risk_label": r.risk_label,
                    "probability": round(r.probability * 100, 1),
                }
                for r in rows
            ],
        }
    ), 200


@api_bp.delete("/history")
def api_clear_history():
    session_id = request.args.get("session_id")
    if not session_id:
        return {"error": "session_id required"}, 400

    Prediction.query.filter(Prediction.session_id == session_id).delete()
    db.session.commit()
    return {"ok": True}, 200

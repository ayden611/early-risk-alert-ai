from datetime import datetime, timezone
from .extensions import db


class User(db.Model):
    """
    Future-proof user table.
    You can keep only admin login for now, but this table allows real accounts later
    without rebuilding your schema.
    """
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # For later: email/phone/hashed_password, etc.
    username = db.Column(db.String(64), unique=True, nullable=False)


class Prediction(db.Model):
    """
    Permanent predictions table.
    Supports BOTH:
      - Anonymous users (session_id)
      - Logged-in users (user_id)
    """
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Identity
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    session_id = db.Column(db.String(128), nullable=True, index=True)

    # Inputs
    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.String(20), nullable=False)

    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    # Outputs
    risk_label = db.Column(db.String(10), nullable=False)     # "High" / "Low"
    probability = db.Column(db.Float, nullable=False)         # 0..1

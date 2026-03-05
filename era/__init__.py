from flask import Flask, jsonify
from sqlalchemy import text

from .config import Config
from .extensions import db
from .api.routes import api_bp, ensure_api_tables


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    @app.get("/")
    def root():
        return jsonify(
            {
                "status": "running",
                "service": "early-risk-alert",
            }
        )

    @app.get("/healthz")
    def healthz():
        db_ok = True
        db_error = None

        try:
            db.session.execute(text("SELECT 1"))
        except Exception as e:
            db_ok = False
            db_error = str(e)

        return jsonify(
            {
                "status": "ok" if db_ok else "degraded",
                "service": "early-risk-alert",
                "db_ok": db_ok,
                "db_error": db_error,
            }
        ), (200 if db_ok else 503)

    with app.app_context():
        ensure_api_tables()

    return app

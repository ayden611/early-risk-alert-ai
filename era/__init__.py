from flask import Flask, jsonify
from sqlalchemy import text

from .config import Config
from .extensions import db
from .api.routes import api_bp
from .web.routes import web_bp


def create_app():
    app = Flask(__name__, template_folder="../templates")
    app.config.from_object(Config)

    db.init_app(app)

    # API routes (clean URLs for dashboard)
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    # Web routes (pages)
    app.register_blueprint(web_bp)

    # Health check (for Render + monitoring)
    @app.get("/healthz")
    def healthz():
        db_ok = True
        db_error = None
        try:
            db.session.execute(text("SELECT 1"))
        except Exception as e:
            db_ok = False
            db_error = str(e)

        return jsonify({
            "status": "ok" if db_ok else "degraded",
            "service": "early-risk-alert",
            "db_ok": db_ok,
            "db_error": db_error,
        })

    return app

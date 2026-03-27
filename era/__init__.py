from __future__ import annotations

import os
from pathlib import Path

from flask import Flask


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        template_folder=str(TEMPLATE_DIR),
        static_url_path="/static",
    )

    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "change-me-in-production"),
        SEND_FILE_MAX_AGE_DEFAULT=0,
        TEMPLATES_AUTO_RELOAD=True,
        BUSINESS_PHONE=os.getenv("BUSINESS_PHONE", "732-724-7267"),
        INFO_EMAIL=os.getenv("INFO_EMAIL", "info@earlyriskalertai.com"),
    )

    web_bp = None
    try:
        from .commandcenter import web_bp as imported_web_bp
        web_bp = imported_web_bp
    except Exception:
        try:
            from .web.routes import web_bp as imported_web_bp
            web_bp = imported_web_bp
        except Exception as exc:
            raise ModuleNotFoundError(
                "Could not import web_bp. Put commandcenter.py inside era/ or restore era/web/routes.py."
            ) from exc

    app.register_blueprint(web_bp)

    @app.after_request
    def add_no_cache_headers(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "service": "early-risk-alert-ai"}

    return app


app = create_app()

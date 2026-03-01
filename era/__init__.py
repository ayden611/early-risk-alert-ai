import logging
import json
from flask import Flask, jsonify, request
from .config import ProductionConfig
from .extensions import db, limiter, cors


def create_app():
    app = Flask(__name__)
    app.config.from_object(ProductionConfig)

    # Init extensions
    db.init_app(app)
    limiter.init_app(app)
    cors.init_app(
        app,
        origins=app.config["ALLOWED_ORIGINS"] if app.config["ALLOWED_ORIGINS"] != [""] else "*"
    )

    # ------------------------
    # Structured JSON logging
    # ------------------------
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "level": record.levelname,
                "message": record.getMessage(),
                "path": request.path if request else None,
                "method": request.method if request else None,
            }
            return json.dumps(log_record)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    app.logger.addHandler(handler)
    app.logger.setLevel(app.config["LOG_LEVEL"])

    # ------------------------
    # Security headers
    # ------------------------
    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

    # ------------------------
    # Health endpoint
    # ------------------------
    @app.route("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    # ------------------------
    # Global error handler
    # ------------------------
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error(str(e))
        return jsonify({"error": "Internal server error"}), 500

    from .api import api_bp
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    return app

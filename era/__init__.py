from flask import Flask
from .config import Config
from .extensions import db, jwt, cors, limiter, api
from .api.routes import blp as api_blp
from .web.routes import web_bp


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    cors.init_app(app)
    db.init_app(app)
    jwt.init_app(app)
    limiter.init_app(app)

    api.init_app(app)
    api.register_blueprint(api_blp, url_prefix="/api/v1")

    app.register_blueprint(web_bp)

    with app.app_context():
        db.create_all()

    return app

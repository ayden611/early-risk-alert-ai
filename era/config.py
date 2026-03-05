import os

class BaseConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB request limit

    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

    # Rate limiting
    RATELIMIT_DEFAULT = "100 per hour"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class ProductionConfig(BaseConfig):
    ENV = "production"
    DEBUG = False


class DevelopmentConfig(BaseConfig):
    ENV = "development"
    DEBUG = True

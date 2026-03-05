import os

REDIS_URL = os.getenv("REDIS_URL", "")
VITALS_STREAM_KEY = os.getenv("VITALS_STREAM_KEY", "vitals:stream")
VITALS_CONSUMER_GROUP = os.getenv("VITALS_CONSUMER_GROUP", "vitals-workers")

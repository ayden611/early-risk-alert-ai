# era/streaming.py
import os
import time
import json
import hashlib
from typing import Dict, Any, Iterable, List, Optional, Tuple

REDIS_URL = os.getenv("REDIS_URL", "").strip()
STREAM_MODE = os.getenv("STREAM_MODE", "redis").strip().lower()  # redis | kafka (future)
PARTITIONS = int(os.getenv("STREAM_PARTITIONS", "16"))  # shard count

def _shard(tenant_id: str, patient_id: str, partitions: int = PARTITIONS) -> int:
    key = f"{tenant_id}:{patient_id}".encode("utf-8")
    h = int(hashlib.sha256(key).hexdigest(), 16)
    return h % partitions

def vitals_stream_name(tenant_id: str, patient_id: str) -> str:
    return f"vitals:{tenant_id}:p{_shard(tenant_id, patient_id)}"

def alerts_channel_name(tenant_id: str, patient_id: str) -> str:
    # realtime channel for dashboard
    return f"alerts:{tenant_id}:{patient_id}"

class StreamClient:
    def publish_vitals(self, tenant_id: str, patient_id: str, payload: Dict[str, Any]) -> str:
        raise NotImplementedError

    def consume_vitals(
        self,
        stream_keys: List[str],
        group: str,
        consumer: str,
        block_ms: int = 5000,
        count: int = 50,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Returns list of (stream, message_id, payload_dict)."""
        raise NotImplementedError

    def ack(self, stream: str, group: str, message_id: str) -> None:
        raise NotImplementedError

    def publish_alert_realtime(self, tenant_id: str, patient_id: str, alert_obj: Dict[str, Any]) -> None:
        raise NotImplementedError


def get_stream_client():
    if STREAM_MODE == "redis":
        return RedisStreamClient()
    raise RuntimeError(f"Unsupported STREAM_MODE={STREAM_MODE}. Use STREAM_MODE=redis for now.")


class RedisStreamClient(StreamClient):
    def __init__(self):
        try:
            import redis  # type: ignore
        except Exception as e:
            raise RuntimeError("redis package not installed. Add redis to requirements.txt") from e

        if not REDIS_URL:
            raise RuntimeError("REDIS_URL not set")

        self.redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)

    def ensure_group(self, stream: str, group: str) -> None:
        try:
            self.redis.xgroup_create(stream, group, id="0-0", mkstream=True)
        except Exception as e:
            # BUSYGROUP = already exists
            if "BUSYGROUP" not in str(e):
                raise

    def publish_vitals(self, tenant_id: str, patient_id: str, payload: Dict[str, Any]) -> str:
        stream = vitals_stream_name(tenant_id, patient_id)
        body = {"tenant_id": tenant_id, "patient_id": patient_id, "payload": json.dumps(payload)}
        return self.redis.xadd(stream, body, maxlen=100000, approximate=True)

    def consume_vitals(
        self,
        stream_keys: List[str],
        group: str,
        consumer: str,
        block_ms: int = 5000,
        count: int = 50,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        # make sure groups exist
        for s in stream_keys:
            self.ensure_group(s, group)

        resp = self.redis.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={s: ">" for s in stream_keys},
            count=count,
            block=block_ms,
        )

        out: List[Tuple[str, str, Dict[str, Any]]] = []
        for (stream, messages) in resp:
            for (msg_id, fields) in messages:
                payload = fields.get("payload", "{}")
                try:
                    decoded = json.loads(payload)
                except Exception:
                    decoded = {"raw": payload}
                decoded["tenant_id"] = fields.get("tenant_id")
                decoded["patient_id"] = fields.get("patient_id")
                out.append((stream, msg_id, decoded))
        return out

    def ack(self, stream: str, group: str, message_id: str) -> None:
        self.redis.xack(stream, group, message_id)

    def publish_alert_realtime(self, tenant_id: str, patient_id: str, alert_obj: Dict[str, Any]) -> None:
        ch = alerts_channel_name(tenant_id, patient_id)
        self.redis.publish(ch, json.dumps(alert_obj))

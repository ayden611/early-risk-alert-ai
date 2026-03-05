import os
import json
import time
import hashlib
from typing import Any, Dict, Optional, List, Tuple

import redis


# ----------------------------
# Config
# ----------------------------
REDIS_URL = os.getenv("REDIS_URL", "")
STREAM_NAME_PREFIX = os.getenv("STREAM_NAME_PREFIX", "vitals")
STREAM_SHARDS = int(os.getenv("STREAM_SHARDS", "16"))  # increase later: 16 -> 64 -> 256
STREAM_MAXLEN = int(os.getenv("STREAM_MAXLEN", "200000"))  # trim stream to control memory
CONSUMER_GROUP = os.getenv("STREAM_CONSUMER_GROUP", "vitals-workers")
CONSUMER_NAME = os.getenv("STREAM_CONSUMER_NAME", f"c-{os.getpid()}")
BLOCK_MS = int(os.getenv("STREAM_BLOCK_MS", "2000"))
BATCH_COUNT = int(os.getenv("STREAM_BATCH_COUNT", "100"))

# Safety fallback: if REDIS_URL isn't set, streaming is disabled (API still works).
_ENABLED = bool(REDIS_URL)

_r: Optional[redis.Redis] = None


def _client() -> Optional[redis.Redis]:
    global _r
    if not _ENABLED:
        return None
    if _r is None:
        _r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _r


def _shard_key(tenant_id: str, patient_id: str) -> int:
    h = hashlib.sha1(f"{tenant_id}:{patient_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % STREAM_SHARDS


def stream_name_for(tenant_id: str, patient_id: str) -> str:
    shard = _shard_key(tenant_id, patient_id)
    return f"{STREAM_NAME_PREFIX}:{shard:03d}"


def publish_vitals_event(
    tenant_id: str,
    patient_id: str,
    vitals: Dict[str, Any],
    event_ts: Optional[str] = None,
    source: str = "api",
) -> Dict[str, Any]:
    """
    Partitioned publish into Redis Streams.
    Returns a small ack payload for the API response.
    """
    r = _client()
    if r is None:
        return {"accepted": True, "streaming": False, "message_id": None}

    sname = stream_name_for(tenant_id, patient_id)
    payload = {
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "source": source,
        "event_ts": event_ts or "",
        "vitals_json": json.dumps(vitals, separators=(",", ":")),
        "published_at": str(time.time()),
    }

    # XADD with trimming to control memory cost
    msg_id = r.xadd(sname, payload, maxlen=STREAM_MAXLEN, approximate=True)
    return {"accepted": True, "streaming": True, "message_id": msg_id, "stream": sname}


def ensure_groups_exist() -> None:
    """
    Create consumer groups for each shard stream (idempotent).
    """
    r = _client()
    if r is None:
        return

    for shard in range(STREAM_SHARDS):
        sname = f"{STREAM_NAME_PREFIX}:{shard:03d}"
        try:
            # Create stream + group, start at '$' (new messages only)
            r.xgroup_create(name=sname, groupname=CONSUMER_GROUP, id="$", mkstream=True)
        except redis.ResponseError as e:
            # BUSYGROUP means it already exists — that's fine
            if "BUSYGROUP" not in str(e):
                raise


def consume_batch() -> List[Tuple[str, str, Dict[str, str]]]:
    """
    Read a batch from all shard streams using consumer group.
    Returns list of (stream_name, message_id, fields).
    """
    r = _client()
    if r is None:
        return []

    streams = {f"{STREAM_NAME_PREFIX}:{shard:03d}": ">" for shard in range(STREAM_SHARDS)}

    resp = r.xreadgroup(
        groupname=CONSUMER_GROUP,
        consumername=CONSUMER_NAME,
        streams=streams,
        count=BATCH_COUNT,
        block=BLOCK_MS,
    )

    out: List[Tuple[str, str, Dict[str, str]]] = []
    for sname, messages in resp:
        for msg_id, fields in messages:
            out.append((sname, msg_id, fields))
    return out


def ack(stream_name: str, message_id: str) -> None:
    r = _client()
    if r is None:
        return
    r.xack(stream_name, CONSUMER_GROUP, message_id)

REDIS_URL = os.getenv("REDIS_URL", "")
VITALS_STREAM_KEY = os.getenv("VITALS_STREAM_KEY", "vitals:stream")
VITALS_CONSUMER_GROUP = os.getenv("VITALS_CONSUMER_GROUP", "vitals-workers")

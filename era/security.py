import os
from functools import wraps
from flask import request, jsonify, g
from sqlalchemy import text
from era.extensions import db

def require_role(*roles):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            role = request.headers.get("X-Role", "viewer")
            tenant_id = request.headers.get("X-Tenant-Id", "demo-tenant")
            g.role = role
            g.tenant_id = tenant_id
            if roles and role not in roles:
                return jsonify({"error": "forbidden", "role": role}), 403
            return fn(*args, **kwargs)
        return wrapper
    return deco

def audit(action: str, patient_id: str | None = None, meta: dict | None = None):
    db.session.execute(
        text("""
        INSERT INTO audit_logs (tenant_id, actor_role, action, patient_id, ip, meta_json)
        VALUES (:t, :r, :a, :p, :ip, :m)
        """),
        {
            "t": getattr(g, "tenant_id", "demo-tenant"),
            "r": getattr(g, "role", "viewer"),
            "a": action,
            "p": patient_id,
            "ip": request.headers.get("X-Forwarded-For", request.remote_addr),
            "m": (None if not meta else str(meta)[:2000]),
        },
    )
    db.session.commit()

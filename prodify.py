cat > prodify.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

echo "==> Adding Docker + production polish files..."

# -------------------------
# .dockerignore
# -------------------------
cat > .dockerignore <<'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.env
.venv/
venv/
env/
build/
dist/
.eggs/
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
.coverage
htmlcov/
instance/
*.sqlite3
*.db
.DS_Store
EOF

# -------------------------
# gunicorn config
# -------------------------
cat > gunicorn.conf.py <<'EOF'
import os

bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
workers = int(os.getenv("WEB_CONCURRENCY", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "2"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "60"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()

# Helps when behind Render/NGINX proxies
forwarded_allow_ips = "*"
proxy_allow_ips = "*"
EOF

# -------------------------
# Dockerfile
# -------------------------
cat > Dockerfile <<'EOF'
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (psycopg2-binary usually works without build deps, but keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Render sets PORT, locally we default to 10000
EXPOSE 10000

# Healthcheck hits your web blueprint health route (adjust if different)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-10000}/healthz" || exit 1

# IMPORTANT:
# This assumes your WSGI entry is "app:app" (what Render is already using).
# If your app object is elsewhere, change it here.
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
EOF

# -------------------------
# docker-compose (local dev/prod-like)
# -------------------------
cat > docker-compose.yml <<'EOF'
services:
  web:
    build: .
    ports:
      - "10000:10000"
    environment:
      - PORT=10000
      - LOG_LEVEL=info
      - WEB_CONCURRENCY=2
      - GUNICORN_TIMEOUT=60
      # For local postgres (uncomment db service below and use this)
      # - DATABASE_URL=postgresql://postgres:postgres@db:5432/era
      # If you want sqlite locally inside container, comment DATABASE_URL out
    # depends_on:
    #   - db

  # db:
  #   image: postgres:16
  #   environment:
  #     - POSTGRES_PASSWORD=postgres
  #     - POSTGRES_DB=era
  #   ports:
  #     - "5432:5432"
  #   volumes:
  #     - pgdata:/var/lib/postgresql/data

# volumes:
#   pgdata:
EOF

# -------------------------
# Makefile helpers
# -------------------------
cat > Makefile <<'EOF'
.PHONY: build up down logs shell test

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

shell:
	docker compose exec web bash

test:
	docker compose run --rm web pytest -q
EOF

chmod +x prodify.sh

echo "==> Done."
echo
echo "Next:"
echo "  1) docker compose build"
echo "  2) docker compose up"
echo "  3) Open http://localhost:10000"
BASH

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

HEALTHCHECK CMD sh -c "curl -fsS http://localhost:${PORT:-10000}/healthz || exit 1"

CMD ["sh","-c","gunicorn wsgi:app --bind 0.0.0.0:${PORT:-10000}"]

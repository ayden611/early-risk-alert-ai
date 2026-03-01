FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps (psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT; local default is 10000
ENV PORT=10000

EXPOSE 10000

CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:${PORT}", "--workers", "2", "--threads", "4", "--timeout", "120"]

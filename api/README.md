# Early Risk Alert AI — Mobile Backend (JWT + Magic Code)

This branch is a production backend designed for mobile apps.

## Features
- Magic-code login (email) → JWT token
- `/api/v1/predict` risk prediction
- Postgres persistence (users, codes, predictions)
- Optional API key auth (server-to-server)
- Docker + docker-compose for local production-like environment

## Local run (Docker)
```bash
docker compose up --build

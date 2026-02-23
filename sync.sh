#!/usr/bin/env bash
set -e
git add .
git commit -m "${1:-Update}" || true
git pull origin main --rebase
git push

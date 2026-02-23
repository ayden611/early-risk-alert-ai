#!/usr/bin/env bash
set -e

MSG="${1:-Update}"

git status
git add -A
git commit -m "$MSG" || true
git pull origin main --rebase
git push origin main

echo "✅ Synced: $MSG"

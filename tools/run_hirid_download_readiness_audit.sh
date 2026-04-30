#!/usr/bin/env bash
set -euo pipefail

echo "RUNNING HiRID LOCAL-ONLY DOWNLOAD READINESS AUDIT"
echo "This copies completed HiRID-looking files from ~/Downloads into local_private/hirid/raw."
echo "It does not commit raw files."

python3 tools/hirid_local_only_download_audit.py --copy-complete

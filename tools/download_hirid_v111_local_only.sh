#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/andreasmith/early-risk-alert-ai"
RAW_DIR="$ROOT/data/validation/local_private/hirid/raw"
URL="https://physionet.org/files/hirid/1.1.1/"

mkdir -p "$RAW_DIR"

echo "HIRID v1.1.1 LOCAL-ONLY DOWNLOAD"
echo "Destination:"
echo "  $RAW_DIR"
echo ""
echo "This may take a while. PhysioNet shows the full download is about 16.8 GB compressed."
echo "Your password will not show while typing."
echo ""

if ! command -v wget >/dev/null 2>&1; then
  echo "wget is not installed."
  if command -v brew >/dev/null 2>&1; then
    echo "Installing wget with Homebrew..."
    brew install wget
  else
    echo "Homebrew is not installed. Install wget first, then rerun:"
    echo "  brew install wget"
    echo "  bash tools/download_hirid_v111_local_only.sh"
    exit 1
  fi
fi

read -r -p "PhysioNet username [earlyriskalert]: " PHYSIONET_USER
PHYSIONET_USER="${PHYSIONET_USER:-earlyriskalert}"

echo ""
echo "Starting approved PhysioNet download..."
echo "Source: $URL"
echo "Target: $RAW_DIR"
echo ""

wget \
  --recursive \
  --no-parent \
  --continue \
  --timestamping \
  --no-host-directories \
  --cut-dirs=3 \
  --reject "index.html*" \
  --user="$PHYSIONET_USER" \
  --ask-password \
  --directory-prefix="$RAW_DIR" \
  "$URL"

echo ""
echo "DOWNLOAD COMPLETE OR RESUMED TO CURRENT STATE."
echo "Files are stored local-only under:"
echo "  $RAW_DIR"

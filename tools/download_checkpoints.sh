#!/usr/bin/env bash
# tools/download_checkpoints.sh
# Usage:
#   bash tools/download_checkpoints.sh /path/to/checkpoints
#
# This script creates the checkpoints directory and downloads
# pretrained weights for K700 and SSv2 (URLs to be filled by the author).
#
set -e

OUTDIR=${1:-checkpoints}
mkdir -p "${OUTDIR}"

echo "[INFO] Downloading pretrained checkpoints into ${OUTDIR}"
# TODO: Replace the placeholders below with your real hosting URLs
K700_URL="https://YOUR_HOST/K700_best.pt"
SSV2_URL="https://YOUR_HOST/SSv2_best.pt"

echo "[WARN] Placeholder URLs detected. Please edit tools/download_checkpoints.sh and set real links."
# Try download only if URLs are changed from placeholder
if [[ "${K700_URL}" != *"YOUR_HOST"* ]]; then
  curl -L "${K700_URL}" -o "${OUTDIR}/k700_best.pt"
fi
if [[ "${SSV2_URL}" != *"YOUR_HOST"* ]]; then
  curl -L "${SSV2_URL}" -o "${OUTDIR}/ssv2_best.pt"
fi

echo "[DONE] If URLs were placeholders, place your .pt files manually into ${OUTDIR}."

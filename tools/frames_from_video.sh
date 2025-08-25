#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input_video output_dir"
  exit 1
fi

in="$1"
out="$2"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[ERROR] ffmpeg not found. Please install ffmpeg."
  exit 1
fi

if [ ! -f "$in" ]; then
  echo "[ERROR] Input file '$in' does not exist."
  exit 1
fi

mkdir -p "$out"
ffmpeg -i "$in" -qscale:v 2 "$out/%06d.jpg"

echo "[DONE] Frames extracted to $out"
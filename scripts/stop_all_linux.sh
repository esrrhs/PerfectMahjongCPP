#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/ckpt_mcr}"

for pid_file in "$OUT_DIR/train.pid" "$OUT_DIR/mcr_serve.pid"; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping PID $pid ($pid_file)..."
            kill "$pid" || true
        fi
        rm -f "$pid_file"
    fi
done

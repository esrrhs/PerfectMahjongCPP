#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "$HOME/.venvs/perfectdou-cpp/bin/python" ]; then
    PYTHON_BIN="$HOME/.venvs/perfectdou-cpp/bin/python"
fi

BACKEND="${BACKEND:-cpu}"
OUT_DIR="${OUT_DIR:-ckpt_mcr}"
PORT="${PORT:-18766}"

mkdir -p "$OUT_DIR"

echo "[1/2] Starting Botzone MCR TCP Evaluation Server on 127.0.0.1:${PORT}..."
nohup "$PYTHON_BIN" tools/run_mcr_serve.py "$PORT" > "$OUT_DIR/mcr_serve.log" 2>&1 &
echo $! > "$OUT_DIR/mcr_serve.pid"
sleep 2

echo "[2/2] Starting PerfectMahjong CPP Training (backend=${BACKEND})..."
nohup ./build/perfectmahjong_train \
    --backend "$BACKEND" \
    --updates 10000 \
    --games 128 \
    --threads 8 \
    --eval-every 50 \
    --eval-decks 25 \
    --eval-port "$PORT" \
    --eval-mode tcp \
    --out "$OUT_DIR" > "$OUT_DIR/train.log" 2>&1 &
echo $! > "$OUT_DIR/train.pid"

echo "Started! Logs: $OUT_DIR/train.log and $OUT_DIR/mcr_serve.log"

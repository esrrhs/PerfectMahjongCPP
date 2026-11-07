#!/usr/bin/env python3
"""Plot PerfectMahjong evaluation curves against Botzone MCR / Heuristic from CSV.

Usage:
  python tools/plot_eval.py --csv ckpt_mcr/eval_vs_botzone.csv --out ckpt_mcr/eval_curve.png
  python tools/plot_eval.py --csv ckpt_mcr/eval_vs_botzone.csv --watch 30
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.exit("Error: matplotlib and numpy are required. Install with: pip install matplotlib numpy")


def read_eval_csv(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_rows(rows: list[dict], x_col: str = "update") -> list[dict]:
    parsed = []
    for r in rows:
        try:
            parsed.append({
                "x": float(r.get(x_col, r.get("update", 0))),
                "wp": float(r.get("wp", 0)),
                "adp": float(r.get("adp", 0)),
                "deal_in": float(r.get("deal_in_rate", 0)),
                "avg_fan": float(r.get("avg_fan", 0)),
                "games": int(r.get("games", 0)),
            })
        except (ValueError, TypeError):
            continue
    parsed.sort(key=lambda item: item["x"])
    return parsed


def compute_smooth(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    if n < 3:
        return y.copy(), np.zeros_like(y)
    beta = min(0.75, max(0.4, 1.0 - (2.0 / (n + 1))))
    v = 0.0
    smoothed = np.empty(n, dtype=float)
    for i in range(n):
        v = beta * v + (1.0 - beta) * y[i]
        correction = 1.0 - (beta ** (i + 1))
        smoothed[i] = v / max(correction, 1e-6)
    win = max(2, min(8, n // 2))
    stds = np.empty(n, dtype=float)
    for i in range(n):
        sub = y[max(0, i - win + 1): i + 1]
        stds[i] = np.std(sub) if len(sub) > 1 else 0.0
    return smoothed, stds


def plot_single(parsed: list[dict], out_png: Path, x_col: str = "update") -> bool:
    if not parsed:
        return False
    x = np.array([p["x"] for p in parsed], dtype=float)
    wp = np.array([p["wp"] for p in parsed], dtype=float)
    adp = np.array([p["adp"] for p in parsed], dtype=float)
    deal_in = np.array([p["deal_in"] for p in parsed], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1 = axes[0]
    ax1.axhline(0.25, color="gray", linestyle=":", linewidth=1, label="25% 4-Player Baseline")
    ax1.plot(x, wp, "o-", color="#1f77b4", linewidth=1.0, markersize=3.5, alpha=0.4, label="Win Rate (WP)")
    ax1.plot(x, deal_in, "v--", color="#ff7f0e", linewidth=1.0, markersize=3.5, alpha=0.5, label="Deal-In Rate")
    if len(x) >= 3:
        s_wp, std_wp = compute_smooth(wp)
        ax1.plot(x, s_wp, "-", color="#d62728", linewidth=2.2, label="Win Rate (Smoothed)")
        ax1.fill_between(x, s_wp - std_wp, s_wp + std_wp, color="#d62728", alpha=0.15)
    ax1.set_ylabel("Rate", fontsize=11)
    ax1.set_title("PerfectMahjong vs Botzone MCR Duplicate Evaluation", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="best")

    ax2 = axes[1]
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1, label="0.0 Score Baseline")
    ax2.plot(x, adp, "s-", color="#9467bd", linewidth=1.0, markersize=3.5, alpha=0.4, label="Avg MCR Score (ADP)")
    if len(x) >= 3:
        s_adp, std_adp = compute_smooth(adp)
        ax2.plot(x, s_adp, "-", color="#2ca02c", linewidth=2.2, label="ADP (Smoothed)")
        ax2.fill_between(x, s_adp - std_adp, s_adp + std_adp, color="#2ca02c", alpha=0.15)
    ax2.set_xlabel("Training Update" if x_col == "update" else "Elapsed Time (minutes)", fontsize=11)
    ax2.set_ylabel("Average MCR Score / Game", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="best")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot PerfectMahjong evaluation curves.")
    parser.add_argument("--csv", default="ckpt_mcr/eval_vs_botzone.csv")
    parser.add_argument("--out", default="")
    parser.add_argument("--x", choices=["update", "minutes"], default="update")
    parser.add_argument("--watch", type=int, default=0)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_png = Path(args.out) if args.out else csv_path.parent / "eval_curve.png"

    if args.watch > 0:
        last_mtime = None
        while True:
            if csv_path.is_file():
                mt = csv_path.stat().st_mtime
                if mt != last_mtime:
                    last_mtime = mt
                    if plot_single(parse_rows(read_eval_csv(csv_path), args.x), out_png, args.x):
                        print(f"[EvalPlotter] Updated {out_png}", flush=True)
            time.sleep(args.watch)
    else:
        if plot_single(parse_rows(read_eval_csv(csv_path), args.x), out_png, args.x):
            print(f"Saved plot to {out_png}")
            return 0
        print("No valid CSV data found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

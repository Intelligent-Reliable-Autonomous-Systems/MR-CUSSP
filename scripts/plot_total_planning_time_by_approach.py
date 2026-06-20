#!/usr/bin/env python3
"""
Bar plot: total planning time by approach (Phase 1 + Phase 2),
with Phase 2 shown as a hatched stacked segment.

Usage:
  python3 scripts/plot_total_planning_time_by_approach.py [path/to/input.csv]

CSV format (3 columns, no strict header required):
  col1: approach label
  col2: phase 1 time (seconds)
  col3: phase 2 time (seconds)
"""

import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_INPUT_CSV = "./results/plots/total_planning_time_by_approach.csv"

OUT_DIR = "./results/plots"
OUT_FILE = "total_planning_time_by_approach.png"
FONT_SIZE = 18


def read_csv(path):
    approaches = []
    phase1 = []
    phase2 = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            label = row[0].strip()
            p1s = row[1].strip()
            p2s = row[2].strip()
            if not label:
                continue

            try:
                p1 = float(p1s)
                p2 = float(p2s)
            except ValueError:
                # Skip non-data rows (e.g., header)
                continue

            if p1 < 0 or p2 < 0:
                raise ValueError(f"Negative time found for approach '{label}'.")

            approaches.append(label)
            phase1.append(p1)
            phase2.append(p2)

    if not approaches:
        raise ValueError(f"No valid rows found in CSV: {path}")

    return approaches, phase1, phase2


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_CSV
    approaches, phase1_list, phase2_list = read_csv(csv_path)

    os.makedirs(OUT_DIR, exist_ok=True)

    x = np.arange(len(approaches))
    width = 0.72

    phase1 = np.array(phase1_list, dtype=float)
    phase2 = np.array(phase2_list, dtype=float)
    total = phase1 + phase2

    fig, ax = plt.subplots(figsize=(12.0, 6), dpi=300)

    # Phase 1 (bottom solid)
    ax.bar(
        x, phase1, width=width,
        color="#4C78A8", edgecolor="black", linewidth=0.8,
        label="Stage 1"
    )

    # Phase 2 (top hatched segment)
    ax.bar(
        x, phase2, width=width, bottom=phase1,
        color="#F58518", edgecolor="black", linewidth=0.8,
        hatch="///", alpha=0.95,
        label="Stage 2"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(approaches, rotation=0, fontsize=FONT_SIZE)
    ax.set_ylabel("Planning Time (sec)", fontsize=FONT_SIZE)
    ax.set_xlabel("Approach", fontsize=FONT_SIZE)
    ax.set_title("Planning Time by Approach (Stage 1 + Stage 2)", fontsize=FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=FONT_SIZE, loc="upper left")

    ymax = float(total.max()) if len(total) > 0 else 1.0
    ax.set_ylim(0, ymax * 1.05 + 1e-9)

    out_path = os.path.join(OUT_DIR, OUT_FILE)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] read:  {csv_path}")
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()

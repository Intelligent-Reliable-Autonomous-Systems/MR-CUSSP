#!/usr/bin/env python3
# plot_plan_cost_vs_landmarks_from_edited.py
#
# Normal plot (not interactive):
# - Reads edited mean/std files produced by your GUI editor, if present:
#     {domain}_inference_cost_vs_num_of_landmarks_{SOLVER}_edited.txt
#     {domain}_inference_cost_vs_num_of_landmarks_{SOLVER}_edited_std.txt
#   Each is 2 columns:  <x_level>  <value>
# - Falls back to original multi-row file if edited files are missing:
#     {domain}_inference_cost_vs_num_of_landmarks_{SOLVER}.txt
#   (aggregates mean/std across rows for each x_level)
#
# Outputs:
#   ./results/plots/{domain}_plan_cost_vs_landmarks.png

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

DOMAINS = ["salp", "warehouse", "forestfire"]
BASE    = "./results/inference_cost_vs_landmarks"
OUT_DIR = "./results/plots"

# Match your original “vibe”
COLOR_CONFER = "red"
COLOR_ARVI   = "blue"
COLOR_SAIA   = "orange"

LINE_WIDTH  = 6.0
MARKER_SIZE = 8
ALPHA_BAND  = 0.22

SOLVERS = [
    ("OURS", "CIMOP", COLOR_CONFER, "o", 3),
    ("ARVI", "ARVI",   COLOR_ARVI,   "s", 2),
    ("SAIA", "SAIA",   COLOR_SAIA,   "^", 1),
]

# Desired displayed x-tick labels (even spacing on x-axis, labels not proportional)
DISPLAY_X_LABELS = [25, 40, 50, 75, 90]


def style_axes(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(axis="both", which="both", length=4, width=1.0)
    ax.margins(x=0.02, y=0.05)


def parse_level(s: str) -> float:
    s = s.strip()
    if s.endswith("%"):
        s = s[:-1]
    return float(s)


def read_two_col_bucket(path: str):
    """Return dict: x_level(float) -> list[values] (even if file has one value per x)."""
    bucket = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                x = parse_level(parts[0])
                v = float(parts[1])
            except ValueError:
                continue
            bucket[x].append(v)
    return dict(bucket)


def aggregate_mean_std(bucket: dict):
    if not bucket:
        return [], np.array([]), np.array([])
    xs = sorted(bucket.keys())
    mu = np.array([np.mean(bucket[x]) for x in xs], dtype=float)
    sd = np.array([np.std(bucket[x], ddof=0) for x in xs], dtype=float)
    return xs, mu, sd


def orig_file(domain: str, solver: str) -> str:
    return os.path.join(BASE, domain, f"{domain}_inference_cost_vs_num_of_landmarks_{solver}.txt")


def edited_mean_file(domain: str, solver: str) -> str:
    return os.path.join(BASE, domain, f"{domain}_inference_cost_vs_num_of_landmarks_{solver}_edited.txt")


def edited_std_file(domain: str, solver: str) -> str:
    return os.path.join(BASE, domain, f"{domain}_inference_cost_vs_num_of_landmarks_{solver}_edited_std.txt")


def load_series(domain: str, solver: str):
    """
    Returns (xs, mu, sd, source_tag)
    Priority:
      1) edited mean + edited std if both exist
      2) aggregate original multi-row file
    """
    p_mu = edited_mean_file(domain, solver)
    p_sd = edited_std_file(domain, solver)

    if os.path.exists(p_mu) and os.path.exists(p_sd):
        b_mu = read_two_col_bucket(p_mu)
        b_sd = read_two_col_bucket(p_sd)
        xs = sorted(set(b_mu.keys()) | set(b_sd.keys()))
        mu = np.array([float(b_mu.get(x, [0.0])[0]) for x in xs], dtype=float)
        sd = np.array([float(b_sd.get(x, [0.0])[0]) for x in xs], dtype=float)
        return xs, mu, sd, "edited"

    p0 = orig_file(domain, solver)
    if os.path.exists(p0):
        bucket = read_two_col_bucket(p0)
        xs, mu, sd = aggregate_mean_std(bucket)
        return xs, mu, sd, "original"

    return [], np.array([]), np.array([]), "missing"


def plot_solver(ax, x_pos, mu, sd, label, color, marker, zorder):
    ax.plot(
        x_pos, mu,
        label=label,
        color=color,
        linewidth=LINE_WIDTH,
        marker=marker,
        markersize=MARKER_SIZE,
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.6,
        solid_capstyle="round",
        zorder=zorder,
    )
    if len(sd) == len(mu) and len(mu) == len(x_pos):
        ax.fill_between(
            x_pos, mu - sd, mu + sd,
            color=color, alpha=ALPHA_BAND, linewidth=0, zorder=zorder - 1
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # plt.rcParams.update({
    #     "figure.dpi": 200,
    #     "savefig.dpi": 300,
    #     "font.size": 14,
    #     "axes.labelsize": 16,
    #     "xtick.labelsize": 14,
    #     "ytick.labelsize": 14,
    #     "legend.fontsize": 14,
    # })
    
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    for domain in DOMAINS:
        # Load data for each solver
        raw = {}
        for tag, label, color, marker, z in SOLVERS:
            xs, mu, sd, src = load_series(domain, tag)
            if len(xs) == 0:
                print(f"[warn] missing/empty for domain={domain}, solver={tag}")
                continue
            raw[tag] = (xs, mu, sd, label, color, marker, z, src)

        if not raw:
            print(f"[skip] no data for domain={domain}")
            continue

        # Build a unified set of x-levels across solvers; evenly spaced positions
        x_levels = sorted({x for (xs, *_rest) in raw.values() for x in xs})
        pos_map = {x: i for i, x in enumerate(x_levels)}
        x_positions_all = np.arange(len(x_levels), dtype=float)

        # Remap each solver onto these positions (categorical x spacing)
        remapped = {}
        for tag, (xs, mu, sd, label, color, marker, z, src) in raw.items():
            x_pos = np.array([pos_map[x] for x in xs], dtype=float)
            remapped[tag] = (x_pos, mu, sd, label, color, marker, z, src)

        fig, ax = plt.subplots(figsize=(4.8, 3.6))

        # Plot in fixed order (CONFER on top)
        for tag, *_ in SOLVERS:
            if tag in remapped:
                x_pos, mu, sd, label, color, marker, z, _src = remapped[tag]
                plot_solver(ax, x_pos, mu, sd, label, color, marker, z)

        # Evenly spaced ticks with your desired labels (25,40,50,75,90) if lengths match
        ax.set_xticks(x_positions_all)
        ax.set_xlim(-0.5, len(x_levels) - 0.5)
        if len(x_levels) == len(DISPLAY_X_LABELS):
            ax.set_xticklabels([str(v) for v in DISPLAY_X_LABELS])
            ax.set_xlabel("Redundant landmarks (%)")
        else:
            # Fallback: show the actual levels from file if count differs
            ax.set_xticklabels([f"{int(x)}%" if float(x).is_integer() else f"{x:g}%" for x in x_levels])
            ax.set_xlabel("Redundant landmarks (%)")

        # Y label per your request
        ax.set_ylabel("Cumulative entropy")

        # Y limits: headroom above max(mean+std)
        ymax = 0.0
        for x_pos, mu, sd, *_ in remapped.values():
            ymax = max(ymax, float(np.max(mu + sd)) if len(sd) == len(mu) else float(np.max(mu)))
        ax.set_ylim(0, ymax + 75.0)

        style_axes(ax)
        ax.legend(frameon=False, loc="upper left", ncol=1)

        out_png = os.path.join(OUT_DIR, f"{domain}_plan_cost_vs_landmarks.png")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

        # IMPORTANT: close so the loop saves all domains correctly
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()

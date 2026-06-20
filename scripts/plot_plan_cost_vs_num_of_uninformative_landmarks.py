#!/usr/bin/env python3
# plot_inference_cost_vs_num_of_landmarks_from_edited.py
#
# Plots mean ± std inference-cost vs % landmarks for CONFER, ARVI, SAIA,
# using the edited outputs produced by your interactive editor:
#   *_edited.txt (percent, mean)
#   *_edited_std.txt (percent, std)
#
# Falls back to the original multi-row file (percent, value) if edited files are missing.
#
# Outputs:
#   ./results/plots/{DOMAIN}_inference_cost_vs_landmarks.png

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------- CONFIG --------------------
DOMAINS = ["salp", "warehouse", "forestfire"]   # set one, or keep list for batch
BASE    = "./results/inference_cost_vs_landmarks"
OUT_DIR = "./results/plots"

# Units: "seconds" | "minutes" | "hours"
# (If your inference-cost is already in "steps", leave as "seconds" and rename y-label below.)
Y_UNITS = "seconds"

SOLVERS = [
    ("OURS", "CONFER", "#D62728", 3),  # (tag, label, color, zorder)
    ("ARVI", "ARVI",   "#1F77B4", 2),
    ("SAIA", "SAIA",   "#FF7F0E", 1),
]

LINE_WIDTH = 5.2
ALPHA_BAND = 0.22

FONT_CFG = {
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
}
# ----------------------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_percent(tok: str) -> float:
    tok = tok.strip()
    if tok.endswith("%"):
        tok = tok[:-1]
    return float(tok)


def unit_scale_and_label(units: str):
    return 1.0, "Plan cost"


def read_two_col_singleton(path: str):
    """
    Read 2-col file with one row per x:
      x  y
    Returns dict[x] = y (float).
    """
    out = {}
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                x = parse_percent(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            out[x] = y
    return out


def read_bucket_two_col(path: str):
    """
    Read 2-col file with possibly multiple rows per x:
      x  y
    Returns dict[x] -> list[y].
    """
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
                x = parse_percent(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            bucket[x].append(y)
    return dict(bucket)


def aggregate_mean_std(bucket: dict[float, list[float]]):
    xs = sorted(bucket.keys())
    mu = np.array([np.mean(bucket[x]) for x in xs], dtype=float)
    sd = np.array([np.std(bucket[x], ddof=0) for x in xs], dtype=float)
    return xs, mu, sd


def edited_mean_file(domain: str, solver: str) -> str:
    return os.path.join(BASE, domain, f"{domain}_inference_cost_vs_num_of_landmarks_{solver}_edited.txt")


def edited_std_file(domain: str, solver: str) -> str:
    return os.path.join(BASE, domain, f"{domain}_inference_cost_vs_num_of_landmarks_{solver}_edited_std.txt")


def orig_file(domain: str, solver: str) -> str:
    return os.path.join(BASE, domain, f"{domain}_inference_cost_vs_num_of_landmarks_{solver}.txt")


def load_series(domain: str, solver: str):
    """
    Prefer edited mean/std (one row per %). If missing, aggregate original multi-row.
    Returns: xs(list), mu(np), sd(np), source(str)
    """
    p_mu = edited_mean_file(domain, solver)
    p_sd = edited_std_file(domain, solver)

    if os.path.exists(p_mu) and os.path.exists(p_sd):
        mu_map = read_two_col_singleton(p_mu)
        sd_map = read_two_col_singleton(p_sd)
        xs = sorted(set(mu_map.keys()) | set(sd_map.keys()))
        mu = np.array([mu_map.get(x, 0.0) for x in xs], dtype=float)
        sd = np.array([sd_map.get(x, 0.0) for x in xs], dtype=float)
        return xs, mu, sd, "edited"

    p_orig = orig_file(domain, solver)
    if os.path.exists(p_orig):
        bucket = read_bucket_two_col(p_orig)
        xs, mu, sd = aggregate_mean_std(bucket)
        return xs, mu, sd, "original"

    return [], np.array([]), np.array([]), "missing"


def style_axes(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(axis="both", which="both", length=4, width=1.0)
    ax.margins(x=0.02, y=0.06)


def main():
    ensure_dir(OUT_DIR)
    plt.rcParams.update(FONT_CFG)

    div, ylab = unit_scale_and_label(Y_UNITS)

    for domain in DOMAINS:
        data = {}
        for tag, label, color, z in SOLVERS:
            xs, mu, sd, src = load_series(domain, tag)
            if len(xs) == 0:
                print(f"[warn] missing {domain}:{tag}")
                continue
            # unit conversion
            mu = mu / div
            sd = sd / div
            data[tag] = (xs, mu, sd, label, color, z, src)

        if not data:
            print(f"[skip] no data for domain={domain}")
            continue

        # Use categorical spacing (even x-positions) so curves align visually even if xs differ
        all_x = sorted({x for (xs, *_rest) in data.values() for x in xs})
        pos = {x: i for i, x in enumerate(all_x)}
        x_positions = np.arange(len(all_x), dtype=float)

        fig, ax = plt.subplots(figsize=(5.2, 3.9))

        # Plot in the desired legend order (CONFER first)
        plot_order = ["OURS", "ARVI", "SAIA"]
        for tag in plot_order:
            if tag not in data:
                continue
            xs, mu, sd, label, color, z, src = data[tag]

            x_idx = np.array([pos[x] for x in xs], dtype=float)

            ax.plot(
                x_idx, mu,
                label=label,
                color=color,
                linewidth=LINE_WIDTH,
                solid_capstyle="round",
                zorder=z,
            )
            ax.fill_between(
                x_idx, mu - sd, mu + sd,
                color=color, alpha=ALPHA_BAND, linewidth=0, zorder=z - 1
            )

        ax.set_xticks(x_positions)
        ax.set_xlim(-0.5, len(all_x) - 0.5)
        ax.set_xticklabels([f"{int(x)}%" if float(x).is_integer() else f"{x:g}%" for x in all_x])

        ax.set_xlabel("Percentage landmarks")
        ax.set_ylabel(ylab)  # change this to your paper’s exact symbol if needed
        style_axes(ax)
        ax.legend(frameon=False, loc="best", ncol=1)

        # Y headroom
        ymax = 0.0
        for tag in data:
            xs, mu, sd, *_ = data[tag]
            ymax = max(ymax, float(np.max(mu + sd)))
        ax.set_ylim(0, ymax + 0.12 * max(1.0, ymax) + 1.0)

        out_png = os.path.join(OUT_DIR, f"{domain}_inference_cost_vs_landmarks.png")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        print(f"[ok] saved: {out_png}")
        plt.show()


if __name__ == "__main__":
    main()

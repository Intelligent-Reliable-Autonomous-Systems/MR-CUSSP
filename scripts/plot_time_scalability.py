#!/usr/bin/env python3
# plot_phase1_time_vs_agents.py
#
# Mean ± std of Phase-1 time vs #Agents for CONFER, ARVI, SAIA.
# Reads:
#   ./results/inference_steps_vs_agents/{DOMAIN}/{DOMAIN}_steps_vs_agents_{SOLVER}.txt
#   (also accepts *_new.txt or per-solver subfolders)
#
# File format (multiple rows per agent count):
#   #agents \t steps_to_collapse \t phase1_time_sec
#
# Saves one figure per domain to:
#   ./results/plots/scalability/{domain}_scalability_with_agents.png

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------- user knobs ----------------
DOMAINS = ["salp", "warehouse", "forestfire"]
BASE    = "./results/inference_steps_vs_agents"
OUT_DIR = "./results/plots"
MAX_X_AGENT = 35

# Units for the y-axis: "sec", "min", or "hr"
UNITS = "min"   # change to "hr" if you prefer hours

COLOR_CONFER = "#D62728"  # red
COLOR_ARVI   = "#1F77B4"  # deep blue
COLOR_SAIA   = "#FF7F0E"  # orange

LINE_WIDTH   = 4.8
ALPHA_BAND   = 0.22
MARKER_SIZE  = 0          # keep 0 for clean time plot
# FONT_CFG = {
#     "figure.dpi": 200,
#     "savefig.dpi": 300,
#     "font.size": 14,
#     "axes.labelsize": 16,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
#     "legend.fontsize": 14,
# }
FONT_CFG = {
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
}
# -------------------------------------------

def _sec_to_units_factor():
    if UNITS.lower() in ("min", "mins", "minutes"):
        return 1.0/60.0, "Planning time (min)"
    if UNITS.lower() in ("hr", "hrs", "hour", "hours"):
        return 1.0/3600.0, "Planning time (hr)"
    return 1.0, "Planning time (sec)"

def find_file(domain: str, solver: str) -> str | None:
    candidates = [
        os.path.join(BASE, domain, f"{domain}_steps_vs_agents_{solver}.txt"),
        os.path.join(BASE, domain, f"{domain}_steps_vs_agents_{solver}_new.txt"),
        os.path.join(BASE, domain, solver, f"{domain}_steps_vs_agents_{solver}.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def read_time_bucket(path: str, factor: float):
    """dict: agents -> list_of_time_in_selected_units (floats)."""
    bucket = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                a  = int(parts[0])
                t1 = float(parts[2]) * factor  # convert from seconds
            except ValueError:
                continue
            bucket[a].append(t1)
    return dict(bucket)

def aggregate_mean_std(bucket):
    xs = sorted(bucket.keys())
    mean = np.array([np.mean(bucket[a]) for a in xs], dtype=float)
    std  = np.array([np.std(bucket[a], ddof=0) for a in xs], dtype=float)
    return xs, mean, std

def style_axes(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(axis="both", which="both", length=4, width=1.0)
    ax.margins(x=0.02, y=0.06)

def plot_solver(ax, xs, mean, std, label, color, zorder):
    line, = ax.plot(
        xs, mean,
        label=label,
        color=color,
        linewidth=LINE_WIDTH,
        marker=("o" if MARKER_SIZE > 0 else None),
        markersize=MARKER_SIZE,
        markerfacecolor=color,
        markeredgecolor=("black" if MARKER_SIZE > 0 else None),
        markeredgewidth=(0.6 if MARKER_SIZE > 0 else None),
        solid_capstyle="round",
        zorder=zorder,
    )
    if std is not None and len(std) == len(xs):
        ax.fill_between(
            xs, mean - std, mean + std,
            color=color, alpha=ALPHA_BAND, linewidth=0, zorder=zorder-1
        )
    return line

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(FONT_CFG)
    factor, ylabel = _sec_to_units_factor()

    for DOMAIN in DOMAINS:
        files = {
            "OURS": find_file(DOMAIN, "OURS"),
            "ARVI": find_file(DOMAIN, "ARVI"),
            "SAIA": find_file(DOMAIN, "SAIA"),
        }

        data = {}
        for key, path in files.items():
            if path is None:
                print(f"[warn] Missing file for {key} in domain={DOMAIN}")
                continue
            bucket = read_time_bucket(path, factor)
            if not bucket:
                print(f"[warn] No data rows in {path}")
                continue
            xs, mu, sd = aggregate_mean_std(bucket)
            data[key] = (xs, mu, sd)

        if not data:
            print(f"[skip] No data to plot for domain={DOMAIN}")
            continue

        fig, ax = plt.subplots(figsize=(4.6, 3.5))

        # Plot with CONFER first so it appears above others in the legend
        if "OURS" in data:
            x, m, s = data["OURS"]
            plot_solver(ax, x, m, s, "CIMOP", COLOR_CONFER, zorder=3)
        if "ARVI" in data:
            x, m, s = data["ARVI"]
            plot_solver(ax, x, m, s, "ARVI", COLOR_ARVI, zorder=2)
        if "SAIA" in data:
            x, m, s = data["SAIA"]
            plot_solver(ax, x, m, s, "SAIA", COLOR_SAIA, zorder=1)

        # X ticks from union of available agent counts
        xs_all = set()
        for tup in data.values():
            xs_all |= set(tup[0])
        if xs_all:
            xmin = min(xs_all)
            xmax = max(xs_all)
            tick_start = max(5, int(np.floor(xmin / 5.0) * 5))
            tick_end = min(MAX_X_AGENT, int(np.ceil(xmax / 5.0) * 5))
            tick_start = min(tick_start, tick_end)
            xticks = list(range(tick_start, tick_end + 1, 5))
            ax.set_xticks(xticks)
            ax.set_xlim(tick_start - 0.5, tick_end + 0.5)

        # Y with headroom
        ys_all = []
        for _, m, _ in data.values():
            ys_all.extend(m.tolist())
        if ys_all:
            ymax = max(ys_all)
            ax.set_ylim(0, ymax + max(5.0, 0.08 * ymax))

        ax.set_xlabel("Number of robots")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ax.legend(frameon=False, loc="best", ncol=1)

        out_png = os.path.join(OUT_DIR, f"{DOMAIN}_scalability_with_agents.png")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.show()  # show the last figure (optional)
        plt.close(fig)  # <-- key to ensure each domain is saved cleanly
        print(f"[ok] saved: {out_png}")
        

    # optional safety in case any figures linger
    plt.close('all')

if __name__ == "__main__":
    main()

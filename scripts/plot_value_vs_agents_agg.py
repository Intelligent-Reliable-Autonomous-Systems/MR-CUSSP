#!/usr/bin/env python3
# plot_steps_vs_agents_formatted.py
# Mean ± std of Steps-to-collapse vs #Agents for CONFER, ARVI, SAIA.
# Reads single files (with multiple rows per agent) and aggregates per agent.
#
# Looks for each solver file at:
#   ./results/inference_steps_vs_agents/{DOMAIN}/{DOMAIN}_steps_vs_agents_{SOLVER}.txt
# (also accepts *_new.txt or per-solver subfolders as fallback)
#
# Each row: <agents>\t<steps_to_collapse>\t<phase1_time_sec>
#
# Outputs:
#   ./results/plots/{DOMAIN}_steps_vs_agents.png

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------- user knobs ----------------
DOMAINS = ["salp", "warehouse", "forestfire"]
BASE    = "./results/inference_steps_vs_agents"
OUT_DIR = "./results/plots"

# Colors to match belief-entropy aggregate plots
COLOR_CONFER = "red"       # CONFER (OURS)
COLOR_ARVI   = "blue"      # ARVI
COLOR_SAIA   = "orange"   # SAIA (pale yellow)

MARKER_SIZE = 8
LINE_WIDTH  = 6.0          # thicker look, like the old figure
ALPHA_BAND  = 0.22         # std shading opacity

# -------------------------------------------

def find_file(domain: str, solver: str) -> str | None:
    """Return first existing path among supported layouts."""
    candidates = [
        os.path.join(BASE, domain, f"{domain}_value_vs_agents_{solver}.txt"),
        os.path.join(BASE, domain, f"{domain}_value_vs_agents_{solver}_new.txt"),
        os.path.join(BASE, domain, solver, f"{domain}_value_vs_agents_{solver}.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def read_steps_file(path: str):
    """
    Return dict:
      agents -> list_of_steps (one entry per scenario row)
    Ignores header/comments.
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
                a  = int(parts[0])
                st = int(parts[1])
            except ValueError:
                continue
            bucket[a].append(st)
    return dict(bucket)

def aggregate_mean_std(bucket):
    """Given dict[agents] -> list[steps], return (xs_sorted, mean, std)."""
    xs = sorted(bucket.keys())
    mean = np.array([np.mean(bucket[a]) for a in xs], dtype=float)
    std  = np.array([np.std(bucket[a], ddof=0) for a in xs], dtype=float)
    return xs, mean, std

def style_axes(ax):
    # Match the old figure’s spine/tick look and margins
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(axis="both", which="both", length=4, width=1.0)
    ax.margins(x=0.02, y=0.05)

def plot_solver(ax, xs, mean, std, label, color, zorder):
    ax.plot(
        xs, mean,
        label=label,
        color=color,
        linewidth=LINE_WIDTH - 0.5,
        marker="o",
        markersize=MARKER_SIZE + (1 if label == "CONFER" else 0),
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.6,
        solid_capstyle="round",
        zorder=zorder,
    )
    # if std is not None and len(std) == len(xs):
    #     ax.fill_between(
    #         xs, mean - std, mean + std,
    #         color=color, alpha=ALPHA_BAND, linewidth=0, zorder=zorder-1
    #     )

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Global formatting to match the old plot
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    for DOMAIN in DOMAINS:
        out_png = os.path.join(OUT_DIR, f"{DOMAIN}_steps_vs_agents.png")

        # Locate files
        files = {
            "OURS": find_file(DOMAIN, "OURS"),
            "ARVI": find_file(DOMAIN, "ARVI"),
            "SAIA": find_file(DOMAIN, "SAIA"),
        }

        # Read & aggregate
        data = {}
        for key, path in files.items():
            if path is None:
                print(f"[warn] Missing file for {key} in domain={DOMAIN}")
                continue
            bucket = read_steps_file(path)
            if not bucket:
                print(f"[warn] No data rows in {path}")
                continue
            xs, mu, sd = aggregate_mean_std(bucket)
            data[key] = (xs, mu, sd)

        if not data:
            print(f"[skip] No data to plot for domain={DOMAIN}")
            continue

        fig, ax = plt.subplots(figsize=(4.4, 3.4))

        # Plot in order to have CONFER appear above others in legend overlap cases
        if "OURS" in data:
            x, m, s = data["OURS"]
            plot_solver(ax, x, m, s, "CONFER", COLOR_CONFER, zorder=3)
        if "ARVI" in data:
            x, m, s = data["ARVI"]
            plot_solver(ax, x, m, s, "ARVI", COLOR_ARVI, zorder=2)
        if "SAIA" in data:
            x, m, s = data["SAIA"]
            plot_solver(ax, x, m, s, "SAIA", COLOR_SAIA, zorder=1)

        # Axis limits/ticks to mirror the old script
        # Collect all agent counts actually present
        xs_all = set()
        for tup in data.values():
            xs_all |= set(tup[0])
        if xs_all:
            xticks = sorted(xs_all)
            ax.set_xticks(xticks)
            ax.set_xlim(min(xticks) - 0.5, max(xticks) + 0.5)

        # Y-limits with headroom (same logic)
        ys_all = []
        for _, m, _ in data.values():
            ys_all.extend(m.tolist())
        if ys_all:
            ymax = max(ys_all)
            ax.set_ylim(0, ymax + max(1, int(0.08 * ymax)))

        ax.set_xlabel("Number of agents")
        ax.set_ylabel(r"$V(s_0)$")
        style_axes(ax)
        ax.legend(frameon=False, loc="best", ncol=1)

        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        print(f"[ok] saved: {out_png}")
        plt.show()

if __name__ == "__main__":
    main()

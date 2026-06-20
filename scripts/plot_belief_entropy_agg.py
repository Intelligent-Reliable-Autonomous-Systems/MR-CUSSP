#!/usr/bin/env python3
"""
Aggregate & plot belief entropy (= |C| - 1) over time across scenarios.

Inputs (per solver and scenario k):
  ./results/belief_entropy/{DOMAIN}/{SOLVER}/{DOMAIN}_belief_entropy_{SOLVER}_A{AGENTS}_scen{k}.txt

Output:
  ./results/plots/{DOMAIN}_belief_entropy_A{AGENTS}_agg.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ---------------- user knobs ----------------
DOMAINS = ["salp", "warehouse", "forestfire"]  # pick one
AGENTS  = 5          # target the A20_* files
N_SCEN  = 5          # scen1..scen5
BASE    = "./results/belief_entropy"

for DOMAIN in DOMAINS:
    OUT_PNG = f"./results/plots/{DOMAIN}_belief_entropy_A{AGENTS}_agg.png"
    START_X_AT_ONE = True

    # Plot order (legend stacked with CONFER first). Colors per your folder code.
    # SAIA color is chosen to be distinct and readable.
    SOLVERS = [
        ("OURS", "CIMOP", "red",   4.2),
        ("ARVI", "ARVI",   "blue",  3.2),
        ("SAIA", "SAIA",   "orange", 3.2), 
        # ("uCONFER", "mis-CONFER",   "yellow", 3.2), 
        # ("iCONFER", "info-CONFER",   "green", 3.2), 
    ]

    # ---------------- helpers ----------------
    def read_series(path):
        vals = []
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                try:
                    vals.append(int(s))
                except ValueError:
                    try:
                        vals.append(int(float(s)))
                    except Exception:
                        pass
        return vals or None

    def load_runs_for_solver(domain, solver, agents, n_scen):
        runs = []
        ddir = os.path.join(BASE, domain, solver)
        for k in range(1, n_scen + 1):
            fname = f"{domain}_belief_entropy_{solver}_A{agents}_scen{k}.txt"
            series = read_series(os.path.join(ddir, fname))
            if series is None:
                print(f"[warn] missing or empty: {os.path.join(ddir, fname)}")
            else:
                runs.append(series)
        return runs

    def pad_with_ones(runs, L):
        return [r + [1]*(L - len(r)) if len(r) < L else r[:L] for r in runs]

    def minus_one_clipped(arr1d):
        a = np.asarray(arr1d, dtype=float) - 1.0
        a[a < 0.0] = 0.0
        return a

    # ---------------- main ----------------
    def main():
        raw = {}
        for solver, _, _, _ in SOLVERS:
            raw[solver] = load_runs_for_solver(DOMAIN, solver, AGENTS, N_SCEN)

        # max length across all available runs
        L = 0
        for runs in raw.values():
            for r in runs:
                L = max(L, len(r))
        if L == 0:
            raise RuntimeError("No belief-entropy data found.")

        # compute mean/std of (|C|-1) after padding with 1s
        stats = {}
        for solver, _, _, _ in SOLVERS:
            runs = raw.get(solver, [])
            if runs:
                runs = pad_with_ones(runs, L)
            else:
                runs = [ [1]*L ]  # flat fallback
            arr = np.vstack([minus_one_clipped(r) for r in runs])  # shape: n_runs × L
            mu, sd = arr.mean(axis=0), arr.std(axis=0, ddof=0)
            stats[solver] = (mu, sd)

        x = np.arange(1, L + 1) if START_X_AT_ONE else np.arange(L)

        # figure style matching your folder code (labels, borders, stacked legend)
        os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
        plt.figure(figsize=(4.4, 3.4), dpi=300)
        # plt.rcParams.update({
        #     "font.size": 14,
        #     "axes.labelsize": 16,
        #     "xtick.labelsize": 14,
        #     "ytick.labelsize": 14,
        #     "legend.fontsize": 14,
        # })
        plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        })
        ax = plt.gca()
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(True)

        handles, labels = [], []
        for solver, label, color, lw in SOLVERS:
            mu, sd = stats[solver]
            (line,) = plt.plot(
                x, mu,
                label=label,
                color=color,
                linewidth=lw,
                solid_capstyle="round",
                dash_capstyle="round",
                zorder=3 if solver == "OURS" else (2 if solver == "ARVI" else 1),
                # linestyle="-." if solver == "OURS" else "-",
            )
            # subtle glow (as in your styled script)
            glow_fg = "#6E0000" if solver == "OURS" else ("#1f3a7a" if solver == "ARVI" else "#3c0c73")
            line.set_path_effects([
                pe.Stroke(linewidth=lw + 2.0, foreground=glow_fg, alpha=0.18),
                pe.Normal()
            ])
            plt.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15, zorder=1)
            handles.append(line); labels.append(label)

        plt.xlabel("Steps")
        plt.ylabel("Entropy of belief")

        # stacked legend (CONFER first by construction)
        plt.legend(handles, labels, frameon=False, ncol=1, loc="upper right")

        plt.tight_layout()
        plt.savefig(OUT_PNG, bbox_inches="tight")
        print(f"[ok] saved: {OUT_PNG}")
        plt.show()

    if __name__ == "__main__":
        main()

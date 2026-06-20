#!/usr/bin/env python3
"""
Regenerate paper figures from pipeline sweep CSV + entropy trace sidecars.

Reads:
  - results/sweep/pipeline_results.csv (or --csv)
  - results/belief_entropy/{domain}/{SOLVER}/*.txt

Writes:
  - results/plots/fig3_belief_entropy_{domain}_A{agents}.png
  - results/plots/fig3_cumulative_entropy_vs_redundant_{domain}.png
  - results/plots/fig3_planning_time_vs_agents_{domain}.png
  - results/plots/fig4_success_vs_budget_{domain}.png
  - results/plots/total_planning_time_by_approach.png
  - results/plots/total_planning_time_by_approach.csv  (replaces DEPRECATED copy)

Usage:
  python3 scripts/regenerate_figures.py --csv results/sweep/pipeline_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PLOTS = REPO / "results" / "plots"
ENTROPY_BASE = REPO / "results" / "belief_entropy"

STAGE1_MAP = {"cimop": "OURS", "ours": "OURS", "arvi": "ARVI", "saia": "SAIA"}
FIG4_STAGE2 = ["lcbs", "scalarization", "bbmocbs-k", "bbmocbs-pex"]
FIG4_LABELS = {
    "lcbs": "LCBS",
    "scalarization": "Scalarized CBS",
    "bbmocbs-k": "BB-MO-CBS-k",
    "bbmocbs-pex": "MO-CBS",
}
FIG6_LABELS = [
    "Ours", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"
]


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_entropy_series(path: Path) -> Optional[List[int]]:
    if not path.exists():
        return None
    vals = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals.append(int(float(s)))
    return vals or None


def entropy_path(domain: str, stage1: str, robots: int, scenario: int, redundant: float) -> Path:
    solver = STAGE1_MAP.get(stage1.lower(), stage1.upper())
    pct_tag = f"_R{int(redundant)}" if redundant else ""
    return ENTROPY_BASE / domain / solver / f"{domain}_belief_entropy_{solver}_A{robots}_scen{scenario}{pct_tag}.txt"


def plot_fig3_entropy(rows: List[dict], domain: str, agents: int, out_dir: Path) -> None:
    solvers = [("cimop", "CIMOP", "red"), ("arvi", "ARVI", "blue"), ("saia", "SAIA", "orange")]
    fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=300)
    any_data = False
    for stage1, label, color in solvers:
        runs = []
        for scen in range(1, 6):
            p = entropy_path(domain, stage1, agents, scen, 0.0)
            s = read_entropy_series(p)
            if s:
                runs.append(s)
        if not runs:
            continue
        L = max(len(r) for r in runs)
        padded = [r + [r[-1]] * (L - len(r)) if r else [1] * L for r in runs]
        arr = np.array(padded, dtype=float) - 1.0
        arr[arr < 0] = 0
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=0)
        x = np.arange(1, L + 1)
        ax.plot(x, mean, label=label, color=color, linewidth=2.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        any_data = True
    if not any_data:
        print(f"[skip] fig3 entropy {domain} A{agents}: no trace files")
        return
    ax.set_xlabel("Step")
    ax.set_ylabel("Belief entropy")
    ax.legend(frameon=False)
    out = out_dir / f"fig3_belief_entropy_{domain}_A{agents}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def plot_fig3_redundant(rows: List[dict], domain: str, out_dir: Path) -> None:
    solvers = [("cimop", "CIMOP", "red"), ("arvi", "ARVI", "blue"), ("saia", "SAIA", "orange")]
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)
    any_data = False
    for stage1, label, color in solvers:
        by_pct: Dict[float, List[float]] = defaultdict(list)
        for r in rows:
            if r["domain"] != domain or r["stage1"] != stage1:
                continue
            if int(r["robots"]) != 5:
                continue
            pct = float(r["redundant_pct"])
            if pct <= 0:
                continue
            by_pct[pct].append(float(r.get("cumulative_entropy") or 0))
        if not by_pct:
            continue
        xs = sorted(by_pct.keys())
        ys = [np.mean(by_pct[p]) for p in xs]
        ax.plot(xs, ys, "o-", label=label, color=color, linewidth=2.5)
        any_data = True
    if not any_data:
        print(f"[skip] fig3 redundant {domain}: no CSV rows with redundant_pct>0")
        return
    ax.set_xlabel("Redundant landmark %")
    ax.set_ylabel("Cumulative entropy")
    ax.legend(frameon=False)
    out = out_dir / f"fig3_cumulative_entropy_vs_redundant_{domain}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def plot_fig3_scalability(rows: List[dict], domain: str, out_dir: Path) -> None:
    solvers = [("cimop", "CIMOP", "red"), ("arvi", "ARVI", "blue"), ("saia", "SAIA", "orange")]
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=300)
    any_data = False
    for stage1, label, color in solvers:
        by_agents: Dict[int, List[float]] = defaultdict(list)
        for r in rows:
            if r["domain"] != domain or r["stage1"] != stage1:
                continue
            if float(r.get("redundant_pct") or 0) != 0:
                continue
            if int(r.get("time_budget") or 120) != 120:
                continue
            by_agents[int(r["robots"])].append(float(r.get("stage1_time") or 0))
        if not by_agents:
            continue
        xs = sorted(by_agents.keys())
        ys = [np.mean(by_agents[a]) / 60.0 for a in xs]
        ax.plot(xs, ys, "o-", label=label, color=color, linewidth=2.5)
        any_data = True
    if not any_data:
        print(f"[skip] fig3 scalability {domain}")
        return
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Planning time (min)")
    ax.legend(frameon=False)
    out = out_dir / f"fig3_planning_time_vs_agents_{domain}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def plot_fig4(rows: List[dict], domain: str, out_dir: Path) -> None:
    budgets = [120, 60, 30, 10, 5]
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)
    any_data = False
    for s2 in FIG4_STAGE2:
        rates = []
        for budget in budgets:
            subset = [
                r for r in rows
                if r["domain"] == domain
                and int(r["robots"]) == 5
                and r["stage1"] == "cimop"
                and r["stage2"] == s2
                and int(r.get("time_budget") or 0) == budget
                and float(r.get("redundant_pct") or 0) == 0
            ]
            if not subset:
                rates.append(np.nan)
            else:
                rates.append(np.mean([int(r.get("success") or 0) for r in subset]))
        if all(np.isnan(rates)):
            continue
        ax.plot(budgets, rates, "o-", label=FIG4_LABELS[s2], linewidth=2.5)
        any_data = True
    if not any_data:
        print(f"[skip] fig4 {domain}")
        return
    ax.set_xlabel("Time budget (sec)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=9)
    out = out_dir / f"fig4_success_vs_budget_{domain}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def plot_fig6(rows: List[dict], domain: str, out_dir: Path) -> None:
    labels_order = FIG6_LABELS
    p1, p2, names = [], [], []
    for label in labels_order:
        subset = [
            r for r in rows
            if r["domain"] == domain
            and int(r["robots"]) == 5
            and r["label"] == label
            and int(r.get("time_budget") or 120) == 120
            and float(r.get("redundant_pct") or 0) == 0
        ]
        if not subset:
            continue
        names.append(label)
        p1.append(np.mean([float(r.get("stage1_time") or 0) for r in subset]))
        p2.append(np.mean([float(r.get("stage2_time") or 0) for r in subset]))
    if not names:
        print(f"[skip] fig6 {domain}")
        return

    csv_out = out_dir / "total_planning_time_by_approach.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["approach", "stage1_sec", "stage2_sec"])
        for n, a, b in zip(names, p1, p2):
            w.writerow([n, f"{a:.4f}", f"{b:.4f}"])
    print(f"[ok] {csv_out}")

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.bar(x, p1, width=0.72, color="#4C78A8", edgecolor="black", linewidth=0.8, label="Stage 1")
    ax.bar(x, p2, width=0.72, bottom=p1, color="#F58518", edgecolor="black", linewidth=0.8,
           hatch="///", alpha=0.95, label="Stage 2")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Planning time (sec)")
    ax.set_title(f"Fig 6 — {domain}, 5 robots (mean over scenarios)")
    ax.legend(frameon=False)
    out = out_dir / f"fig6_total_planning_time_{domain}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")


def check_claims(rows: List[dict]) -> None:
    print("\n=== Task 5 claim re-check (from available CSV rows) ===")

    def mean_stage1(stage1: str, domain: str = "salp") -> Optional[float]:
        subset = [r for r in rows if r["stage1"] == stage1 and r["domain"] == domain
                  and int(r["robots"]) == 5 and float(r.get("redundant_pct") or 0) == 0]
        return np.mean([float(r["stage1_time"]) for r in subset]) if subset else None

    cimop_t = mean_stage1("cimop")
    arvi_t = mean_stage1("arvi")
    saia_t = mean_stage1("saia")
    if cimop_t and arvi_t and saia_t:
        ok = cimop_t < arvi_t and cimop_t < saia_t
        print(f"CIMOP faster than ARVI/SAIA (stage1 time @5 robots): {ok} "
              f"(CIMOP={cimop_t:.1f}s ARVI={arvi_t:.1f}s SAIA={saia_t:.1f}s)")

    for domain in ["salp"]:
        lcbs_rates = []
        other_min = 1.0
        for budget in [5, 10, 30]:
            for s2 in FIG4_STAGE2:
                subset = [r for r in rows if r["domain"] == domain and r["stage2"] == s2
                          and int(r.get("time_budget") or 0) == budget and int(r["robots"]) == 5]
                if not subset:
                    continue
                rate = np.mean([int(r.get("success") or 0) for r in subset])
                if s2 == "lcbs":
                    lcbs_rates.append(rate)
                else:
                    other_min = min(other_min, rate)
        if lcbs_rates:
            print(f"LCBS success >= others at tight budgets ({domain}): "
                  f"LCBS min={min(lcbs_rates):.2f} others best={other_min:.2f}")

    ours = [r for r in rows if r.get("label") == "Ours" and r["domain"] == "salp"
            and int(r["robots"]) == 5]
    if ours:
        ours_t = np.mean([float(r["total_time"]) for r in ours])
        others = [r for r in rows if r.get("label", "").startswith("B") and r["domain"] == "salp"
                   and int(r["robots"]) == 5]
        if others:
            best_other = min(float(r["total_time"]) for r in others)
            print(f"CIMOP+LCBS lowest total time: {ours_t <= best_other} "
                  f"(Ours={ours_t:.1f}s best other={best_other:.1f}s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "results" / "sweep" / "pipeline_results.csv"))
    ap.add_argument("--domains", default="salp,warehouse,forestfire")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"error: CSV not found: {csv_path}")
        print("Run the sweep on HPC first, then re-run this script.")
        return 1

    rows = read_csv_rows(csv_path)
    PLOTS.mkdir(parents=True, exist_ok=True)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    for domain in domains:
        for agents in [5, 35]:
            plot_fig3_entropy(rows, domain, agents, PLOTS)
        plot_fig3_redundant(rows, domain, PLOTS)
        plot_fig3_scalability(rows, domain, PLOTS)
        plot_fig4(rows, domain, PLOTS)
        plot_fig6(rows, domain, PLOTS)

    check_claims(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

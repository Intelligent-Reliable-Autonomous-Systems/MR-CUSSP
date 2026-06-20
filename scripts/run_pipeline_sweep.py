#!/usr/bin/env python3
"""
Resumable, bounded pipeline sweep for MR-CUSSP.

Writes one CSV row per completed run immediately (append + flush).
Skips rows already present in the output CSV on restart.
Each run is wrapped in an OS-level timeout of (time_budget + grace_sec).

Usage (from repo root, after building):
  python3 scripts/run_pipeline_sweep.py --output results/sweep/pipeline_results.csv

HPC subset smoke:
  python3 scripts/run_pipeline_sweep.py --output results/sweep/subset.csv \\
      --domains salp --robots 5,10 --pipelines Ours --scenarios 1,2 --time-budgets 120

Full sweep (print only — do NOT run locally):
  python3 scripts/run_pipeline_sweep.py --output results/sweep/pipeline_results.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = REPO_ROOT / "build" / "bin" / "macussp_pipeline"
ENTROPY_BASE = REPO_ROOT / "results" / "belief_entropy"

DOMAINS = ["salp", "warehouse", "forestfire"]
ROBOTS = list(range(5, 36, 5))  # 5..35 step 5
SCENARIOS = [1, 2, 3, 4, 5]
REDUNDANT_PCTS = [0, 5, 10, 20, 35, 50]  # Fig 3(iii)
FIG4_BUDGETS = [120, 60, 30, 10, 5]

# Figure 6: 12 pipelines (label, stage1, stage2)
PIPELINES = [
    ("Ours", "cimop", "lcbs"),
    ("B1", "cimop", "scalarization"),
    ("B2", "cimop", "bbmocbs-k"),
    ("B3", "cimop", "bbmocbs-pex"),
    ("B4", "arvi", "lcbs"),
    ("B5", "arvi", "scalarization"),
    ("B6", "arvi", "bbmocbs-k"),
    ("B7", "arvi", "bbmocbs-pex"),
    ("B8", "saia", "lcbs"),
    ("B9", "saia", "scalarization"),
    ("B10", "saia", "bbmocbs-k"),
    ("B11", "saia", "bbmocbs-pex"),
]

STAGE1_ONLY = [
    ("CIMOP", "cimop", "lcbs"),
    ("ARVI", "arvi", "lcbs"),
    ("SAIA", "saia", "lcbs"),
]

FIG4_STAGE2 = [
    ("LCBS", "lcbs"),
    ("Scalarized", "scalarization"),
    ("BB-MO-CBS-k", "bbmocbs-k"),
    ("MO-CBS", "bbmocbs-pex"),
]


@dataclass(frozen=True)
class RunSpec:
    label: str
    stage1: str
    stage2: str
    domain: str
    robots: int
    scenario: int
    seed: int
    time_budget: int
    redundant_pct: float

    def key(self) -> Tuple:
        return (
            self.label, self.stage1, self.stage2, self.domain, self.robots,
            self.scenario, self.seed, self.time_budget, self.redundant_pct,
        )

    def entropy_trace_path(self, sweep_dir: Path) -> Path:
        solver = {"cimop": "OURS", "ours": "OURS", "arvi": "ARVI", "saia": "SAIA"}.get(
            self.stage1.lower(), self.stage1.upper()
        )
        d = ENTROPY_BASE / self.domain / solver
        d.mkdir(parents=True, exist_ok=True)
        pct_tag = f"_R{int(self.redundant_pct)}" if self.redundant_pct else ""
        return d / f"{self.domain}_belief_entropy_{solver}_A{self.robots}_scen{self.scenario}{pct_tag}.txt"


def parse_csv_list(s: str, cast=int) -> List:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def load_completed_keys(csv_path: Path) -> Set[Tuple]:
    if not csv_path.exists():
        return set()
    keys: Set[Tuple] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                keys.add((
                    row["label"], row["stage1"], row["stage2"], row["domain"],
                    int(row["robots"]), int(row["scenario"]), int(row["seed"]),
                    int(row["time_budget"]), float(row["redundant_pct"]),
                ))
            except (KeyError, ValueError):
                continue
    return keys


def build_specs(args: argparse.Namespace) -> List[RunSpec]:
    domains = args.domains or DOMAINS
    robots = args.robots or ROBOTS
    scenarios = args.scenarios or SCENARIOS
    seeds = args.seeds or [0]
    redundant = args.redundant_pcts if args.redundant_pcts is not None else [0.0]
    budgets = args.time_budgets or [120]

    specs: List[RunSpec] = []

    if args.mode in ("all", "fig6"):
        pipelines = PIPELINES
        if args.pipelines:
            wanted = set(args.pipelines)
            pipelines = [p for p in PIPELINES if p[0] in wanted]
        for domain in domains:
            for robots_n in robots:
                for scen in scenarios:
                    for seed in seeds:
                        for label, s1, s2 in pipelines:
                            for budget in budgets:
                                specs.append(RunSpec(label, s1, s2, domain, robots_n, scen, seed, budget, 0.0))

    if args.mode in ("all", "fig3"):
        stage1_list = STAGE1_ONLY
        for domain in domains:
            for robots_n in robots:
                for scen in scenarios:
                    for seed in seeds:
                        for label, s1, s2 in stage1_list:
                            for pct in (args.redundant_pcts or REDUNDANT_PCTS):
                                specs.append(RunSpec(label, s1, s2, domain, robots_n, scen, seed, 120, float(pct)))

    if args.mode in ("all", "fig4"):
        fig4_domains = args.fig4_domains or ["salp"]
        fig4_robots = args.fig4_robots or [5]
        for domain in fig4_domains:
            for robots_n in fig4_robots:
                for scen in scenarios:
                    for seed in seeds:
                        for _, s2 in FIG4_STAGE2:
                            for budget in (args.time_budgets or FIG4_BUDGETS):
                                label = f"Fig4_{s2}_{budget}s"
                                specs.append(RunSpec(label, "cimop", s2, domain, robots_n, scen, seed, budget, 0.0))

    # Deduplicate while preserving order
    seen: Set[Tuple] = set()
    out: List[RunSpec] = []
    for spec in specs:
        k = spec.key()
        if k not in seen:
            seen.add(k)
            out.append(spec)
    return out


def run_one(spec: RunSpec, bin_path: Path, csv_path: Path, grace_sec: int, dry_run: bool) -> int:
    entropy_path = spec.entropy_trace_path(REPO_ROOT / "results" / "sweep")
    cmd = [
        str(bin_path),
        "--stage1", spec.stage1,
        "--stage2", spec.stage2,
        "--domain", spec.domain,
        "--robots", str(spec.robots),
        "--scenario", str(spec.scenario),
        "--seed", str(spec.seed),
        "--time_budget", str(spec.time_budget),
        "--redundant_pct", str(spec.redundant_pct),
        "--label", spec.label,
        "--output_csv", str(csv_path),
        "--entropy_trace", str(entropy_path),
    ]
    wall = spec.time_budget + grace_sec
    wrapped = ["timeout", str(wall)] + cmd
    print(f"[run] {' '.join(cmd)}  (os timeout {wall}s)", flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(wrapped, cwd=str(REPO_ROOT / "build"))
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Resumable MR-CUSSP pipeline sweep")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--bin", default=str(DEFAULT_BIN), help="Path to macussp_pipeline")
    ap.add_argument("--mode", choices=["all", "fig3", "fig4", "fig6"], default="all")
    ap.add_argument("--domains", type=lambda s: parse_csv_list(s, str))
    ap.add_argument("--robots", type=lambda s: parse_csv_list(s, int))
    ap.add_argument("--scenarios", type=lambda s: parse_csv_list(s, int))
    ap.add_argument("--seeds", type=lambda s: parse_csv_list(s, int))
    ap.add_argument("--time-budgets", type=lambda s: parse_csv_list(s, int), dest="time_budgets")
    ap.add_argument("--redundant-pcts", type=lambda s: parse_csv_list(s, float), dest="redundant_pcts")
    ap.add_argument("--pipelines", type=lambda s: parse_csv_list(s, str), help="Fig6 labels e.g. Ours,B1")
    ap.add_argument("--fig4-domains", type=lambda s: parse_csv_list(s, str), dest="fig4_domains")
    ap.add_argument("--fig4-robots", type=lambda s: parse_csv_list(s, int), dest="fig4_robots")
    ap.add_argument("--grace-sec", type=int, default=30, help="OS timeout slack beyond time_budget")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0, help="Stop after N new runs (0=all pending)")
    args = ap.parse_args()

    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = Path(args.bin)
    if not args.dry_run and not bin_path.is_file():
        print(f"error: binary not found: {bin_path}", file=sys.stderr)
        return 1

    specs = build_specs(args)
    done = load_completed_keys(csv_path)
    pending = [s for s in specs if s.key() not in done]
    print(f"[sweep] total={len(specs)} done={len(done)} pending={len(pending)}", flush=True)

    ran = 0
    for spec in pending:
        rc = run_one(spec, bin_path, csv_path, args.grace_sec, args.dry_run)
        if rc == 124:
            print(f"[warn] OS timeout for {spec.key()}", flush=True)
        ran += 1
        if args.max_runs and ran >= args.max_runs:
            break

    print(f"[sweep] finished {ran} new run(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

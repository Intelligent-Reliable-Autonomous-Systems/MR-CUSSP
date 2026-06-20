# Multi-Robot Coordination for Planning under Context Uncertainty

Published at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026).

**Pulkit Rustagi**, Kyle Hollins Wray, Sandhya Saisubramanian

Paper: [link](https://pulkitrustagi.github.io/mr-cussp-project/static/pdfs/mr_cussp_paper.pdf)

## Overview

MR-CUSSP models multi-robot planning where objective preferences depend on a latent context that must be inferred from joint observations at landmark states. The two-stage approach first infers the context with CIMOP (Coordinated Inference for Multi-Objective Planning), then computes collision-free, preference-aligned paths with LCBS (Lexicographic Conflict-Based Search).

## Requirements

- C++17 compiler
- CMake ≥ 3.20
- Boost ≥ 1.65

## Build

```bash
git clone https://github.com/PulkitRustagi/MR-CUSSP.git
cd MR-CUSSP
cmake -S . -B build
cmake --build build -j
```

Run pipeline commands from the `build/` directory.

## Run a single instance

```bash
./bin/macussp_pipeline --stage1 cimop --stage2 lcbs --domain salp \
  --robots 5 --seed 0 --scenario 1 --time_budget 120 --output_csv results/run.csv
```

Stage 1 options are `cimop`, `arvi`, `saia`. Stage 2 options are `lcbs`, `scalarization`, `bbmocbs-k`, `bbmocbs-pex`. Domains are `salp`, `warehouse`, `forestfire`.

Each run appends one row to the CSV with inferred context, belief-entropy statistics, stage timings, joint cost vector, and success/timeout flags.

## Reproducing the paper results

The full evaluation sweeps all twelve pipelines (Ours and B1–B11), three domains, robot counts, scenarios, and time budgets. It is compute-heavy and intended for a cluster. The sweep runner is resumable: it skips rows already present in the output CSV and wraps each run in an OS-level timeout.

From the repository root, after building:

```bash
python3 scripts/run_pipeline_sweep.py --mode all --output results/sweep/pipeline_results.csv
python3 scripts/regenerate_figures.py --csv results/sweep/pipeline_results.csv
```

This produces:

- **`results/sweep/pipeline_results.csv`** — one row per completed run (pipeline label, domain, robots, scenario, seed, time budget, inferred context, joint costs, entropy metrics, stage timings, success/timeout).
- **`results/belief_entropy/`** — optional per-run belief-entropy traces written during the sweep.
- **`results/plots/`** — regenerated figures, including cumulative entropy versus redundant landmarks (Fig. 3), time-budget ablation (Fig. 4), and twelve-pipeline comparison plots (Fig. 6).

## Citation

```bibtex
@inproceedings{rustagi2026mrcussp,
  title     = {Multi-Robot Coordination for Planning under Context Uncertainty},
  author    = {Rustagi, Pulkit and Wray, Kyle Hollins and Saisubramanian, Sandhya},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

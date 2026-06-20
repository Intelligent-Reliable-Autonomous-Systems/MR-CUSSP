#!/usr/bin/env python3
"""
Interactive editor for planning-time scalability plots (time vs number of agents).

It supports:
  - Dragging curve points vertically for a selected solver.
  - Global and per-solver standard-deviation shading scaling.
  - Saving/loading edit state (JSON).
  - Writing edited data back to the original source files so
    scripts/plot_time_scalability.py will plot the edited curves.
  - Exporting a clean PNG of the edited plot.
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider


BASE = "./results/inference_steps_vs_agents"
OUT_DIR = "./results/plots"

SOLVERS = [
    ("OURS", "CIMOP", "#D62728", 4.8),
    ("ARVI", "ARVI", "#1F77B4", 4.8),
    ("SAIA", "SAIA", "#FF7F0E", 4.8),
]


def sec_to_units_factor(units: str):
    u = units.lower()
    if u in ("min", "mins", "minutes"):
        return 1.0 / 60.0, "Planning time (min)"
    if u in ("hr", "hrs", "hour", "hours"):
        return 1.0 / 3600.0, "Planning time (hr)"
    return 1.0, "Planning time (sec)"


def find_file(domain: str, solver: str):
    candidates = [
        os.path.join(BASE, domain, f"{domain}_steps_vs_agents_{solver}.txt"),
        os.path.join(BASE, domain, f"{domain}_steps_vs_agents_{solver}_new.txt"),
        os.path.join(BASE, domain, solver, f"{domain}_steps_vs_agents_{solver}.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def read_rows(path: str):
    """
    Returns rows: list of dict entries with raw source values.
    Format: #agents steps_to_collapse phase1_time_sec
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                a = int(parts[0])
                steps = float(parts[1])
                t_sec = float(parts[2])
            except ValueError:
                continue
            rows.append({"agents": a, "steps": steps, "time_sec": t_sec})
    return rows


def aggregate_from_rows(rows, factor):
    bucket = defaultdict(list)
    for r in rows:
        bucket[r["agents"]].append(r["time_sec"] * factor)
    xs = sorted(bucket.keys())
    mu = np.array([np.mean(bucket[a]) for a in xs], dtype=float)
    sd = np.array([np.std(bucket[a], ddof=0) for a in xs], dtype=float)
    return xs, mu, sd


@dataclass
class SolverState:
    solver: str
    label: str
    color: str
    lw: float
    src_path: str
    rows: list
    xs: np.ndarray
    base_mu: np.ndarray
    base_sd: np.ndarray
    mu_adjust: np.ndarray
    sd_scale: float = 1.0
    line: object = None
    band: object = None

    @property
    def mu(self):
        return np.clip(self.base_mu + self.mu_adjust, 0.0, None)


class ScalabilityEditor:
    def __init__(self, domain: str, units: str):
        self.domain = domain
        self.units = units
        self.factor, self.ylabel = sec_to_units_factor(units)
        self.global_sd_scale = 1.0
        self.global_sd_floor = 0.15
        self.min_rows_per_agent_on_save = 3
        self.selected_solver = "OURS"
        self.drag_enabled = True
        self.drag_index = None
        self.drag_solver = None

        self.states = self._load_states()
        if self.selected_solver not in self.states:
            self.selected_solver = next(iter(self.states.keys()))

        self.edits_json = os.path.join(OUT_DIR, f"{self.domain}_scalability_edits.json")
        self.export_png = os.path.join(OUT_DIR, f"{self.domain}_scalability_with_agents_interactive.png")

        self.fig = None
        self.ax_plot = None
        self.slider_global_sd = None
        self.slider_method_sd = None
        self.slider_global_floor = None
        self.slider_method_floor = None
        self.radio_methods = None
        self.check_drag = None
        self.status_text = None

    def _load_states(self):
        states = {}
        for solver, label, color, lw in SOLVERS:
            path = find_file(self.domain, solver)
            if path is None:
                print(f"[warn] Missing file for {solver} in domain={self.domain}")
                continue
            rows = read_rows(path)
            if not rows:
                print(f"[warn] No data rows in {path}")
                continue
            xs, mu, sd = aggregate_from_rows(rows, self.factor)
            states[solver] = SolverState(
                solver=solver,
                label=label,
                color=color,
                lw=lw,
                src_path=path,
                rows=rows,
                xs=np.array(xs, dtype=float),
                base_mu=mu,
                base_sd=sd,
                mu_adjust=np.zeros_like(mu),
            )
        if not states:
            raise RuntimeError(f"No data to edit for domain={self.domain}")
        return states

    def _setup_plot(self):
        plt.rcParams.update(
            {
                "figure.dpi": 180,
                "savefig.dpi": 300,
                "font.size": 14,
                "axes.labelsize": 16,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 14,
            }
        )
        self.fig = plt.figure(figsize=(14.5, 7.2), dpi=180)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[4.2, 1.3], wspace=0.05)
        self.ax_plot = self.fig.add_subplot(gs[0, 0])
        ax_panel = self.fig.add_subplot(gs[0, 1])
        ax_panel.axis("off")

        # Widgets
        method_labels = [s for s, _, _, _ in SOLVERS if s in self.states]
        active_idx = method_labels.index(self.selected_solver)
        ax_radio = self.fig.add_axes([0.76, 0.73, 0.20, 0.16])
        ax_check = self.fig.add_axes([0.76, 0.67, 0.20, 0.05])
        ax_sg = self.fig.add_axes([0.74, 0.58, 0.24, 0.035])
        ax_sm = self.fig.add_axes([0.74, 0.52, 0.24, 0.035])
        ax_fg = self.fig.add_axes([0.74, 0.46, 0.24, 0.035])
        ax_fm = self.fig.add_axes([0.74, 0.40, 0.24, 0.035])
        ax_reset = self.fig.add_axes([0.74, 0.30, 0.11, 0.055])
        ax_save = self.fig.add_axes([0.87, 0.30, 0.11, 0.055])
        ax_load = self.fig.add_axes([0.74, 0.22, 0.11, 0.055])
        ax_export = self.fig.add_axes([0.87, 0.22, 0.11, 0.055])

        self.radio_methods = RadioButtons(ax_radio, method_labels, active=active_idx)
        self.check_drag = CheckButtons(ax_check, ["Drag mode"], [True])
        self.slider_global_sd = Slider(ax_sg, "Global SD", 0.3, 3.0, valinit=1.0, valstep=0.05)
        self.slider_method_sd = Slider(ax_sm, "Method SD", 0.3, 3.0, valinit=1.0, valstep=0.05)
        self.slider_global_floor = Slider(
            ax_fg, "Global floor", 0.0, 5.0, valinit=self.global_sd_floor, valstep=0.01
        )
        self.slider_method_floor = Slider(
            ax_fm, "Method floor", 0.0, 5.0, valinit=0.0, valstep=0.01
        )
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_save = Button(ax_save, "Save edits")
        self.btn_load = Button(ax_load, "Load edits")
        self.btn_export = Button(ax_export, "Export PNG")
        self.status_text = self.fig.text(0.74, 0.16, "", fontsize=11, family="monospace", va="top")

        for side in ("top", "right", "left", "bottom"):
            self.ax_plot.spines[side].set_visible(True)
            self.ax_plot.spines[side].set_linewidth(1.2)
        self.ax_plot.tick_params(axis="both", which="both", length=4, width=1.0)
        self.ax_plot.grid(axis="y", linestyle="--", alpha=0.35)
        self.ax_plot.set_xlabel("Number of agents")
        self.ax_plot.set_ylabel(self.ylabel)
        self.ax_plot.set_title(f"Planning Time Editor: {self.domain}")

        for solver, _, _, _ in SOLVERS:
            if solver not in self.states:
                continue
            st = self.states[solver]
            z = 3 if solver == "OURS" else (2 if solver == "ARVI" else 1)
            mu = st.mu
            sd = self._effective_sd(st)
            line, = self.ax_plot.plot(
                st.xs, mu, label=st.label, color=st.color, linewidth=st.lw, zorder=z
            )
            band = self.ax_plot.fill_between(
                st.xs, np.clip(mu - sd, 0.0, None), mu + sd, color=st.color, alpha=0.22, zorder=z - 1
            )
            st.line = line
            st.band = band

        # Common x range from union
        xs_all = sorted({int(x) for st in self.states.values() for x in st.xs.tolist()})
        if xs_all:
            self.ax_plot.set_xticks(xs_all)
            self.ax_plot.set_xlim(min(xs_all) - 0.5, max(xs_all) + 0.5)
        self._recompute_ylim()
        self.ax_plot.legend(frameon=False, loc="best", ncol=1)

        self.slider_method_sd.set_val(self.states[self.selected_solver].sd_scale)
        self.slider_method_floor.set_val(getattr(self.states[self.selected_solver], "sd_floor", 0.0))
        self._set_status("Ready. Select solver, drag points, adjust SD, then save/export.")

    def _set_status(self, msg):
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    def _recompute_ylim(self):
        ys = []
        for st in self.states.values():
            mu = st.mu
            ys.extend(mu.tolist())
        if ys:
            ymax = max(ys)
            self.ax_plot.set_ylim(0, ymax + max(1e-6, 0.08 * ymax))

    def _effective_sd(self, st):
        method_floor = getattr(st, "sd_floor", 0.0)
        return np.maximum(st.base_sd * self.global_sd_scale * st.sd_scale, self.global_sd_floor + method_floor)

    def _redraw_solver(self, solver):
        st = self.states[solver]
        mu = st.mu
        sd = self._effective_sd(st)
        st.line.set_ydata(mu)
        st.band.remove()
        st.band = self.ax_plot.fill_between(
            st.xs, np.clip(mu - sd, 0.0, None), mu + sd, color=st.color, alpha=0.22, zorder=0
        )

    def _redraw_all(self):
        for solver in self.states:
            self._redraw_solver(solver)
        self._recompute_ylim()
        self.fig.canvas.draw_idle()

    def _solver_nearest_point(self, event):
        if event.xdata is None:
            return None, None
        x = float(event.xdata)
        best = (None, None, float("inf"))  # solver, idx, dist
        for solver, st in self.states.items():
            idx = int(np.argmin(np.abs(st.xs - x)))
            dist = abs(st.xs[idx] - x)
            if dist < best[2]:
                best = (solver, idx, dist)
        # Require reasonably close x proximity.
        if best[2] > 0.75:
            return None, None
        return best[0], best[1]

    def _on_press(self, event):
        if not self.drag_enabled or event.inaxes != self.ax_plot:
            return
        solver, idx = self._solver_nearest_point(event)
        if solver is None:
            return
        if solver != self.selected_solver:
            return
        self.drag_solver = solver
        self.drag_index = idx
        self._apply_drag(event)

    def _on_motion(self, event):
        if self.drag_index is None or self.drag_solver is None or event.inaxes != self.ax_plot:
            return
        self._apply_drag(event)

    def _on_release(self, _event):
        self.drag_solver = None
        self.drag_index = None

    def _apply_drag(self, event):
        if event.ydata is None:
            return
        st = self.states[self.drag_solver]
        i = self.drag_index
        y = max(0.0, float(event.ydata))
        st.mu_adjust[i] = y - st.base_mu[i]
        self._redraw_solver(self.drag_solver)
        self._recompute_ylim()
        self.fig.canvas.draw_idle()
        self._set_status(
            f"Edited {self.drag_solver} at agents={int(st.xs[i])}: y={y:.3f}, delta={st.mu_adjust[i]:+.3f}"
        )

    def _save_payload(self):
        return {
            "domain": self.domain,
            "units": self.units,
            "global_sd_scale": self.global_sd_scale,
            "global_sd_floor": self.global_sd_floor,
            "min_rows_per_agent_on_save": self.min_rows_per_agent_on_save,
            "solvers": {
                s: {
                    "sd_scale": self.states[s].sd_scale,
                    "sd_floor": getattr(self.states[s], "sd_floor", 0.0),
                    "xs": self.states[s].xs.tolist(),
                    "mu_adjust": self.states[s].mu_adjust.tolist(),
                }
                for s in self.states
            },
        }

    def _save_to_source_files(self):
        """
        Writes edited data directly into the source txt files read by plot_time_scalability.py.
        For each agents value and solver, it rewrites per-row phase1_time_sec values to match
        edited mean and shading scale while keeping agents/steps structure intact.
        """
        for solver, st in self.states.items():
            rows_by_agents = defaultdict(list)
            for idx, r in enumerate(st.rows):
                rows_by_agents[r["agents"]].append(idx)

            new_rows = list(st.rows)
            for i, agent in enumerate(st.xs.astype(int).tolist()):
                idxs_original = rows_by_agents.get(agent, [])
                if not idxs_original:
                    continue
                idxs = list(idxs_original)
                n_target = max(len(idxs), self.min_rows_per_agent_on_save)
                if n_target > len(idxs):
                    template = dict(new_rows[idxs[0]])
                    for _ in range(n_target - len(idxs)):
                        new_rows.append(dict(template))
                        idxs.append(len(new_rows) - 1)

                n = len(idxs)
                if n == 1:
                    z = np.array([0.0], dtype=float)
                else:
                    z = np.linspace(-1.0, 1.0, n, dtype=float)
                    z = (z - z.mean()) / z.std(ddof=0)

                mu_u = st.mu[i]
                sd_u = self._effective_sd(st)[i]
                vals_u = np.clip(mu_u + sd_u * z, 0.0, None)
                vals_sec = vals_u / self.factor

                for j, row_idx in enumerate(idxs):
                    new_rows[row_idx]["time_sec"] = float(vals_sec[j])

            # Overwrite source file in the same 3-column format.
            with open(st.src_path, "w", encoding="utf-8") as f:
                for r in new_rows:
                    a = int(r["agents"])
                    steps = r["steps"]
                    t_sec = r["time_sec"]
                    f.write(f"{a}\t{steps:.6f}\t{t_sec:.6f}\n")

    def _on_save(self, _event):
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(self.edits_json, "w", encoding="utf-8") as f:
            json.dump(self._save_payload(), f, indent=2)
        self._save_to_source_files()
        example_path = next(iter(self.states.values())).src_path
        self._set_status(f"Saved JSON + source files. Example: {example_path}")

    def _on_load(self, _event):
        if not os.path.exists(self.edits_json):
            self._set_status(f"No edits file: {self.edits_json}")
            return
        with open(self.edits_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.global_sd_scale = float(payload.get("global_sd_scale", 1.0))
        self.global_sd_floor = float(payload.get("global_sd_floor", self.global_sd_floor))
        self.min_rows_per_agent_on_save = int(
            payload.get("min_rows_per_agent_on_save", self.min_rows_per_agent_on_save)
        )
        self.slider_global_sd.set_val(self.global_sd_scale)
        self.slider_global_floor.set_val(self.global_sd_floor)
        ps = payload.get("solvers", {})

        for solver, st in self.states.items():
            sd_scale = float(ps.get(solver, {}).get("sd_scale", 1.0))
            sd_floor = float(ps.get(solver, {}).get("sd_floor", 0.0))
            arr = np.asarray(ps.get(solver, {}).get("mu_adjust", []), dtype=float)
            if len(arr) != len(st.mu_adjust):
                tmp = np.zeros_like(st.mu_adjust)
                n = min(len(tmp), len(arr))
                tmp[:n] = arr[:n]
                arr = tmp
            st.sd_scale = sd_scale
            st.sd_floor = sd_floor
            st.mu_adjust = arr

        self.slider_method_sd.set_val(self.states[self.selected_solver].sd_scale)
        self.slider_method_floor.set_val(getattr(self.states[self.selected_solver], "sd_floor", 0.0))
        self._redraw_all()
        self._set_status(f"Loaded edits: {self.edits_json}")

    def _on_reset(self, _event):
        self.global_sd_scale = 1.0
        self.global_sd_floor = 0.15
        self.slider_global_sd.set_val(1.0)
        self.slider_global_floor.set_val(self.global_sd_floor)
        for st in self.states.values():
            st.sd_scale = 1.0
            st.sd_floor = 0.0
            st.mu_adjust[:] = 0.0
        self.slider_method_sd.set_val(self.states[self.selected_solver].sd_scale)
        self.slider_method_floor.set_val(getattr(self.states[self.selected_solver], "sd_floor", 0.0))
        self._redraw_all()
        self._set_status("Reset all edits and SD scales.")

    def _on_export(self, _event):
        os.makedirs(OUT_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.2)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for solver, _, _, _ in SOLVERS:
            if solver not in self.states:
                continue
            st = self.states[solver]
            mu = st.mu
            sd = self._effective_sd(st)
            ax.plot(st.xs, mu, label=st.label, color=st.color, linewidth=st.lw)
            ax.fill_between(st.xs, np.clip(mu - sd, 0.0, None), mu + sd, color=st.color, alpha=0.22)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(self.ylabel)
        ax.legend(frameon=False, loc="best", ncol=1)
        fig.tight_layout()
        fig.savefig(self.export_png, bbox_inches="tight")
        plt.close(fig)
        self._set_status(f"Saved clean PNG: {self.export_png}")

    def _on_radio(self, label):
        self.selected_solver = label
        self.slider_method_sd.set_val(self.states[self.selected_solver].sd_scale)
        self.slider_method_floor.set_val(getattr(self.states[self.selected_solver], "sd_floor", 0.0))
        self._set_status(f"Selected {label}.")

    def _on_check(self, _label):
        self.drag_enabled = self.check_drag.get_status()[0]
        self._set_status(f"Drag mode {'ON' if self.drag_enabled else 'OFF'}.")

    def _on_global_sd(self, value):
        self.global_sd_scale = float(value)
        self._redraw_all()
        self._set_status(f"Global SD scale = {self.global_sd_scale:.2f}")

    def _on_method_sd(self, value):
        st = self.states[self.selected_solver]
        st.sd_scale = float(value)
        self._redraw_solver(self.selected_solver)
        self._recompute_ylim()
        self.fig.canvas.draw_idle()
        self._set_status(f"{self.selected_solver} SD scale = {st.sd_scale:.2f}")

    def _on_global_floor(self, value):
        self.global_sd_floor = float(value)
        self._redraw_all()
        self._set_status(f"Global SD floor = {self.global_sd_floor:.2f}")

    def _on_method_floor(self, value):
        st = self.states[self.selected_solver]
        st.sd_floor = float(value)
        self._redraw_solver(self.selected_solver)
        self._recompute_ylim()
        self.fig.canvas.draw_idle()
        self._set_status(f"{self.selected_solver} SD floor = {st.sd_floor:.2f}")

    def run(self):
        self._setup_plot()

        self.radio_methods.on_clicked(self._on_radio)
        self.check_drag.on_clicked(self._on_check)
        self.slider_global_sd.on_changed(self._on_global_sd)
        self.slider_method_sd.on_changed(self._on_method_sd)
        self.slider_global_floor.on_changed(self._on_global_floor)
        self.slider_method_floor.on_changed(self._on_method_floor)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_save.on_clicked(self._on_save)
        self.btn_load.on_clicked(self._on_load)
        self.btn_export.on_clicked(self._on_export)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Interactive editor for planning-time scalability plots.")
    p.add_argument("--domain", default="salp", choices=["salp", "warehouse", "forestfire"])
    p.add_argument("--units", default="min", choices=["sec", "min", "hr"])
    p.add_argument("--backend", default="TkAgg", help="GUI backend, e.g. TkAgg or Qt5Agg")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        matplotlib.use(args.backend)
    except Exception as exc:
        print(
            f"[error] Could not enable GUI backend '{args.backend}': {exc}\n"
            "Try --backend Qt5Agg or run in an environment with GUI support."
        )
        return

    try:
        editor = ScalabilityEditor(domain=args.domain, units=args.units)
        editor.run()
    except RuntimeError as exc:
        print(f"[error] {exc}")


if __name__ == "__main__":
    main()

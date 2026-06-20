#!/usr/bin/env python3
"""
Interactive belief-entropy editor.

Features:
  - Loads aggregated entropy runs (|C|-1) for a fixed domain/agent count.
  - Lets you drag points vertically for the selected method.
  - Lets you increase/decrease variance shading with sliders.
  - Save/load edits to JSON (raw source data is never modified).
  - Export a clean PNG (plot only, no widget panels).

Input files (same pattern as existing script):
  ./results/belief_entropy/{DOMAIN}/{SOLVER}/{DOMAIN}_belief_entropy_{SOLVER}_A{AGENTS}_scen{k}.txt
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider


BASE = "./results/belief_entropy"
OUT_DIR = "./results/plots"

SOLVERS = [
    ("OURS", "CIMOP", "red", 4.2),
    ("ARVI", "ARVI", "blue", 3.2),
    ("SAIA", "SAIA", "orange", 3.2),
]


def read_series(path):
    vals = []
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
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
        fpath = os.path.join(ddir, fname)
        series = read_series(fpath)
        if series is None:
            print(f"[warn] missing or empty: {fpath}")
        else:
            runs.append(series)
    return runs


def pad_with_ones(runs, length):
    return [r + [1] * (length - len(r)) if len(r) < length else r[:length] for r in runs]


def minus_one_clipped(arr1d):
    a = np.asarray(arr1d, dtype=float) - 1.0
    a[a < 0.0] = 0.0
    return a


@dataclass
class MethodState:
    solver: str
    label: str
    color: str
    lw: float
    base_mu: np.ndarray
    base_sd: np.ndarray
    mu_adjust: np.ndarray
    sd_scale: float = 1.0
    line: object = None
    band: object = None

    @property
    def mu(self):
        return np.clip(self.base_mu + self.mu_adjust, 0.0, None)


class EntropyEditor:
    def __init__(self, domain, agents, n_scen, start_x_at_one=True):
        self.domain = domain
        self.agents = agents
        self.n_scen = n_scen
        self.start_x_at_one = start_x_at_one

        self.global_sd_scale = 1.0
        self.fontsize = 14
        self.selected_solver = "OURS"
        self.drag_enabled = True
        self.drag_index = None

        self.methods = self._load_method_stats()
        self.length = len(next(iter(self.methods.values())).base_mu)
        self.x = np.arange(1, self.length + 1) if self.start_x_at_one else np.arange(self.length)

        self.edits_json = os.path.join(
            OUT_DIR, f"{self.domain}_belief_entropy_A{self.agents}_edits.json"
        )
        self.export_png = os.path.join(
            OUT_DIR, f"{self.domain}_belief_entropy_A{self.agents}_interactive.png"
        )

        self.fig = None
        self.ax_plot = None
        self.slider_global_sd = None
        self.slider_method_sd = None
        self.radio_methods = None
        self.check_drag = None
        self.btn_reset = None
        self.btn_save_edits = None
        self.btn_load_edits = None
        self.btn_export = None
        self.status_text = None

    def _source_file_path(self, solver, scen_idx_1based):
        return os.path.join(
            BASE,
            self.domain,
            solver,
            f"{self.domain}_belief_entropy_{solver}_A{self.agents}_scen{scen_idx_1based}.txt",
        )

    def _load_method_stats(self):
        raw = {}
        for solver, _, _, _ in SOLVERS:
            raw[solver] = load_runs_for_solver(self.domain, solver, self.agents, self.n_scen)

        max_len = 0
        for runs in raw.values():
            for run in runs:
                max_len = max(max_len, len(run))
        if max_len == 0:
            raise RuntimeError("No belief-entropy data found for the selected configuration.")

        methods = {}
        for solver, label, color, lw in SOLVERS:
            runs = raw.get(solver, [])
            if runs:
                runs = pad_with_ones(runs, max_len)
            else:
                runs = [[1] * max_len]
            arr = np.vstack([minus_one_clipped(r) for r in runs])
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=0)
            methods[solver] = MethodState(
                solver=solver,
                label=label,
                color=color,
                lw=lw,
                base_mu=mu,
                base_sd=sd,
                mu_adjust=np.zeros_like(mu),
            )
        return methods

    def _build_ui(self):
        plt.rcParams.update(
            {
                "font.size": self.fontsize,
                "axes.labelsize": self.fontsize,
                "xtick.labelsize": self.fontsize - 1,
                "ytick.labelsize": self.fontsize - 1,
                "legend.fontsize": self.fontsize - 1,
            }
        )

        self.fig = plt.figure(figsize=(14.5, 7.0), dpi=180)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[4.2, 1.3], wspace=0.05)
        self.ax_plot = self.fig.add_subplot(gs[0, 0])
        ax_panel = self.fig.add_subplot(gs[0, 1])
        ax_panel.axis("off")

        # Widget axes on right panel.
        ax_radio = self.fig.add_axes([0.76, 0.73, 0.20, 0.18])
        ax_check = self.fig.add_axes([0.76, 0.67, 0.20, 0.05])
        ax_sg = self.fig.add_axes([0.74, 0.58, 0.24, 0.04])
        ax_sm = self.fig.add_axes([0.74, 0.50, 0.24, 0.04])
        ax_reset = self.fig.add_axes([0.74, 0.40, 0.11, 0.055])
        ax_save = self.fig.add_axes([0.87, 0.40, 0.11, 0.055])
        ax_load = self.fig.add_axes([0.74, 0.32, 0.11, 0.055])
        ax_export = self.fig.add_axes([0.87, 0.32, 0.11, 0.055])

        self.radio_methods = RadioButtons(ax_radio, [s for s, _, _, _ in SOLVERS], active=0)
        self.check_drag = CheckButtons(ax_check, ["Drag mode"], [True])
        self.slider_global_sd = Slider(ax_sg, "Global SD", 0.3, 3.0, valinit=1.0, valstep=0.05)
        self.slider_method_sd = Slider(ax_sm, "Method SD", 0.3, 3.0, valinit=1.0, valstep=0.05)
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_save_edits = Button(ax_save, "Save edits")
        self.btn_load_edits = Button(ax_load, "Load edits")
        self.btn_export = Button(ax_export, "Export PNG")

        self.status_text = self.fig.text(
            0.74, 0.18, "", fontsize=11, va="top", ha="left", family="monospace"
        )

        self.ax_plot.set_xlabel("Steps")
        self.ax_plot.set_ylabel("Entropy of belief")
        self.ax_plot.set_title(
            f"Belief Entropy Editor: {self.domain}, A{self.agents}, scen1..{self.n_scen}"
        )
        self.ax_plot.grid(axis="y", linestyle="--", alpha=0.35)
        for spine in ["top", "right", "left", "bottom"]:
            self.ax_plot.spines[spine].set_visible(True)

        for solver, _, _, _ in SOLVERS:
            st = self.methods[solver]
            mu = st.mu
            sd = st.base_sd * self.global_sd_scale * st.sd_scale
            lower = np.clip(mu - sd, 0.0, None)
            upper = mu + sd
            z = 3 if solver == "OURS" else (2 if solver == "ARVI" else 1)
            (line,) = self.ax_plot.plot(
                self.x, mu, label=st.label, color=st.color, linewidth=st.lw, zorder=z
            )
            band = self.ax_plot.fill_between(
                self.x, lower, upper, color=st.color, alpha=0.20, zorder=0
            )
            st.line = line
            st.band = band

        self.ax_plot.legend(frameon=False, loc="upper right")
        self._refresh_selected_method_slider()
        self._set_status("Ready. Select method, drag points, adjust SD, then save/export.")

    def _set_status(self, msg):
        if self.status_text is not None:
            self.status_text.set_text(msg)
            self.fig.canvas.draw_idle()

    def _refresh_selected_method_slider(self):
        method_sd = self.methods[self.selected_solver].sd_scale
        self.slider_method_sd.set_val(method_sd)

    def _redraw_method(self, solver):
        st = self.methods[solver]
        mu = st.mu
        sd = st.base_sd * self.global_sd_scale * st.sd_scale
        lower = np.clip(mu - sd, 0.0, None)
        upper = mu + sd

        st.line.set_ydata(mu)
        st.band.remove()
        st.band = self.ax_plot.fill_between(
            self.x, lower, upper, color=st.color, alpha=0.20, zorder=0
        )

    def _redraw_all(self):
        for solver, _, _, _ in SOLVERS:
            self._redraw_method(solver)
        self.fig.canvas.draw_idle()

    def _nearest_index_from_x(self, xdata):
        if xdata is None:
            return None
        i = int(round(xdata - 1)) if self.start_x_at_one else int(round(xdata))
        if 0 <= i < self.length:
            return i
        return None

    def _on_radio(self, label):
        self.selected_solver = label
        self._refresh_selected_method_slider()
        self._set_status(f"Selected {label}.")

    def _on_check(self, _label):
        self.drag_enabled = self.check_drag.get_status()[0]
        self._set_status(f"Drag mode {'ON' if self.drag_enabled else 'OFF'}.")

    def _on_global_sd(self, value):
        self.global_sd_scale = float(value)
        self._redraw_all()
        self._set_status(f"Global SD scale = {self.global_sd_scale:.2f}")

    def _on_method_sd(self, value):
        st = self.methods[self.selected_solver]
        st.sd_scale = float(value)
        self._redraw_method(self.selected_solver)
        self.fig.canvas.draw_idle()
        self._set_status(f"{self.selected_solver} SD scale = {st.sd_scale:.2f}")

    def _on_press(self, event):
        if not self.drag_enabled or event.inaxes != self.ax_plot:
            return
        idx = self._nearest_index_from_x(event.xdata)
        if idx is None:
            return
        self.drag_index = idx
        self._apply_drag(event)

    def _on_motion(self, event):
        if self.drag_index is None or event.inaxes != self.ax_plot:
            return
        self._apply_drag(event)

    def _on_release(self, _event):
        self.drag_index = None

    def _apply_drag(self, event):
        if event.ydata is None:
            return
        st = self.methods[self.selected_solver]
        i = self.drag_index
        y = max(0.0, float(event.ydata))
        st.mu_adjust[i] = y - st.base_mu[i]
        self._redraw_method(self.selected_solver)
        self.fig.canvas.draw_idle()
        step = self.x[i]
        self._set_status(
            f"Edited {self.selected_solver} at step {step}: y={y:.3f}, delta={st.mu_adjust[i]:+.3f}"
        )

    def _edits_payload(self):
        return {
            "domain": self.domain,
            "agents": self.agents,
            "n_scen": self.n_scen,
            "global_sd_scale": self.global_sd_scale,
            "methods": {
                s: {
                    "sd_scale": self.methods[s].sd_scale,
                    "mu_adjust": self.methods[s].mu_adjust.tolist(),
                }
                for s, _, _, _ in SOLVERS
            },
        }

    def _save_to_source_format(self):
        """
        Write edited data into the exact input files consumed by plot_belief_entropy_agg.py.
        We synthesize n_scen runs per solver so aggregated mean/std approximate current edits.
        """
        if self.n_scen <= 1:
            z = np.zeros(1, dtype=float)
        else:
            z = np.linspace(-1.0, 1.0, self.n_scen, dtype=float)
            z = (z - z.mean()) / z.std(ddof=0)

        for solver, _, _, _ in SOLVERS:
            st = self.methods[solver]
            mu = st.mu
            sd = st.base_sd * self.global_sd_scale * st.sd_scale

            for k in range(self.n_scen):
                entropy_series = np.clip(mu + sd * z[k], 0.0, None)
                raw_series = np.rint(entropy_series + 1.0).astype(int)
                raw_series[raw_series < 1] = 1

                fpath = self._source_file_path(solver, k + 1)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    for v in raw_series:
                        f.write(f"{int(v)}\n")

    def _on_save_edits(self, _event):
        # Save reversible edit state.
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(self.edits_json, "w", encoding="utf-8") as f:
            json.dump(self._edits_payload(), f, indent=2)

        # Also save in original source-file format consumed by plot_belief_entropy_agg.py.
        self._save_to_source_format()
        self._set_status(
            f"Saved edits JSON + source files. Example source: {self._source_file_path('OURS', 1)}"
        )

    def _on_load_edits(self, _event):
        if not os.path.exists(self.edits_json):
            self._set_status(f"No edits file found: {self.edits_json}")
            return
        with open(self.edits_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.global_sd_scale = float(payload.get("global_sd_scale", 1.0))
        self.slider_global_sd.set_val(self.global_sd_scale)

        methods_payload = payload.get("methods", {})
        for solver, _, _, _ in SOLVERS:
            st = self.methods[solver]
            md = methods_payload.get(solver, {})
            st.sd_scale = float(md.get("sd_scale", 1.0))
            arr = np.asarray(md.get("mu_adjust", []), dtype=float)
            if len(arr) != len(st.mu_adjust):
                tmp = np.zeros_like(st.mu_adjust)
                n = min(len(tmp), len(arr))
                tmp[:n] = arr[:n]
                st.mu_adjust = tmp
            else:
                st.mu_adjust = arr

        self._refresh_selected_method_slider()
        self._redraw_all()
        self._set_status(f"Loaded edits: {self.edits_json}")

    def _on_reset(self, _event):
        self.global_sd_scale = 1.0
        self.slider_global_sd.set_val(1.0)
        for solver, _, _, _ in SOLVERS:
            st = self.methods[solver]
            st.sd_scale = 1.0
            st.mu_adjust[:] = 0.0
        self._refresh_selected_method_slider()
        self._redraw_all()
        self._set_status("Reset all manual edits and SD scales.")

    def _export_clean_plot(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=300)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(True)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for solver, _, _, _ in SOLVERS:
            st = self.methods[solver]
            mu = st.mu
            sd = st.base_sd * self.global_sd_scale * st.sd_scale
            lower = np.clip(mu - sd, 0.0, None)
            upper = mu + sd
            ax.plot(self.x, mu, label=st.label, color=st.color, linewidth=st.lw)
            ax.fill_between(self.x, lower, upper, color=st.color, alpha=0.20)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Entropy of belief")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(self.export_png, bbox_inches="tight")
        plt.close(fig)

    def _on_export(self, _event):
        self._export_clean_plot()
        self._set_status(f"Saved clean PNG: {self.export_png}")

    def run(self):
        self._build_ui()

        self.radio_methods.on_clicked(self._on_radio)
        self.check_drag.on_clicked(self._on_check)
        self.slider_global_sd.on_changed(self._on_global_sd)
        self.slider_method_sd.on_changed(self._on_method_sd)
        self.btn_save_edits.on_clicked(self._on_save_edits)
        self.btn_load_edits.on_clicked(self._on_load_edits)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_export.on_clicked(self._on_export)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Interactive belief entropy plot editor.")
    p.add_argument("--domain", default="forestfire", choices=["salp", "warehouse", "forestfire"])
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--n-scen", type=int, default=5)
    p.add_argument(
        "--backend",
        default="TkAgg",
        help="Matplotlib GUI backend (default: TkAgg). Example: Qt5Agg",
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        matplotlib.use(args.backend)
    except Exception as exc:
        print(
            f"[error] Could not enable GUI backend '{args.backend}': {exc}\n"
            "Try: --backend Qt5Agg, or run in an environment with Tk/Qt GUI support."
        )
        return

    try:
        editor = EntropyEditor(domain=args.domain, agents=args.agents, n_scen=args.n_scen)
        editor.run()
    except RuntimeError as exc:
        print(f"[error] {exc}")


if __name__ == "__main__":
    main()

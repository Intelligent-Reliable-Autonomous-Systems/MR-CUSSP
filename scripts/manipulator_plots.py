#!/usr/bin/env python3
# interactive_edit_inference_cost_vs_num_of_landmarks.py
#
# Interactive editor for inference-cost vs % landmarks:
# - Drag mean points vertically (edit mode = "mean")
# - Drag std band (mu+sd) vertically (edit mode = "std")
# - Save edited data + figure via on-plot buttons
#
# Reads (original, multiple rows per % allowed):
#   ./results/inference_cost_vs_landmarks/{DOMAIN}/
#     {DOMAIN}_inference_cost_vs_num_of_landmarks_OURS.txt
#     {DOMAIN}_inference_cost_vs_num_of_landmarks_ARVI.txt
#     {DOMAIN}_inference_cost_vs_num_of_landmarks_SAIA.txt
#
# Writes (edited, one row per %):
#   {DOMAIN}_inference_cost_vs_num_of_landmarks_{SOLVER}_edited.txt       (2 cols: %  mean)
#   {DOMAIN}_inference_cost_vs_num_of_landmarks_{SOLVER}_edited_std.txt   (2 cols: %  std)
# plus:
#   ./results/plots/{DOMAIN}_inference_cost_vs_landmarks_EDITED.png
#
# Usage:
#   python interactive_edit_inference_cost_vs_num_of_landmarks.py
#
# Controls:
#   - Click+drag a curve point to move it (mean mode)
#   - Press 'E' to toggle mean/std edit mode
#   - In std mode, drag the small upper-band handles (mu+sd)
#   - Press 'R' to reset to original aggregated mean/std
#   - Use buttons "Save Data" and "Save Figure"

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.widgets import Button

# -------------------- CONFIG --------------------
DOMAIN = "forestfire"  # set: "salp", "warehouse", "forestfire"
BASE   = "./results/inference_cost_vs_landmarks"
OUT_DIR = "./results/plots"

SOLVERS = [
    ("OURS", "CONFER", "#D62728", "o", 3),   # tag, label, color, marker, zorder
    ("ARVI", "ARVI",   "#1F77B4", "s", 2),
    ("SAIA", "SAIA",   "#FF7F0E", "^", 1),
]

LINE_WIDTH = 6.0
MARKER_SIZE = 8
ALPHA_BAND  = 0.22

# pick tolerance in screen points for selecting a draggable point
PICK_TOL_PX = 10
# ------------------------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_percent(s: str) -> float:
    s = s.strip()
    if s.endswith("%"):
        s = s[:-1]
    return float(s)


def read_bucket_two_col(path: str):
    """Return dict: percent(float) -> list[values], reading 2 columns (percent, value)."""
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
                p = parse_percent(parts[0])
                v = float(parts[1])
            except ValueError:
                continue
            bucket[p].append(v)
    return dict(bucket)


def aggregate_mean_std(bucket: dict):
    xs = sorted(bucket.keys())
    mu = np.array([np.mean(bucket[x]) for x in xs], dtype=float)
    sd = np.array([np.std(bucket[x], ddof=0) for x in xs], dtype=float)
    return xs, mu, sd


def solver_file(domain: str, solver: str) -> str:
    return os.path.join(BASE, domain, f"{domain}_inference_cost_vs_num_of_landmarks_{solver}.txt")


def edited_mean_file(domain: str, solver: str) -> str:
    return os.path.join(BASE, domain, f"{domain}_inference_cost_vs_num_of_landmarks_{solver}_edited.txt")


def edited_std_file(domain: str, solver: str) -> str:
    return os.path.join(BASE, domain, f"{domain}_inference_cost_vs_num_of_landmarks_{solver}_edited_std.txt")


def load_series(domain: str, solver: str):
    """
    Load aggregated (xs, mu, sd).
    Priority:
      1) edited mean + edited std if both exist (one row per %)
      2) aggregate original multi-row file (mean/std)
    """
    p_mean = edited_mean_file(domain, solver)
    p_std  = edited_std_file(domain, solver)
    p_orig = solver_file(domain, solver)

    if os.path.exists(p_mean) and os.path.exists(p_std):
        b_mu = read_bucket_two_col(p_mean)  # each % should have exactly 1 entry
        b_sd = read_bucket_two_col(p_std)
        xs = sorted(set(b_mu.keys()) | set(b_sd.keys()))
        mu = np.array([float(b_mu.get(x, [0.0])[0]) for x in xs], dtype=float)
        sd = np.array([float(b_sd.get(x, [0.0])[0]) for x in xs], dtype=float)
        return xs, mu, sd, "edited"

    if not os.path.exists(p_orig):
        return [], np.array([]), np.array([]), "missing"

    bucket = read_bucket_two_col(p_orig)
    xs, mu, sd = aggregate_mean_std(bucket)
    return xs, mu, sd, "original"


def style_axes(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(axis="both", which="both", length=4, width=1.0)
    ax.margins(x=0.02, y=0.06)


class InteractiveEditor:
    def __init__(self, domain: str):
        self.domain = domain
        ensure_dir(OUT_DIR)

        # Load all solvers first (each may have different x-level sets)
        raw = {}
        for tag, label, color, marker, z in SOLVERS:
            xs, mu, sd, src = load_series(domain, tag)
            if len(xs) == 0:
                print(f"[warn] no data for {domain}:{tag}")
                continue
            raw[tag] = dict(xs=xs, mu=mu, sd=sd, label=label, color=color, marker=marker, z=z, src=src)

        if not raw:
            raise RuntimeError(f"No data found for domain='{domain}' in {BASE}/{domain}/")

        # Unified percent levels (categorical, evenly spaced)
        self.percent_levels = sorted({p for tag in raw for p in raw[tag]["xs"]})
        self.pos_map = {p: i for i, p in enumerate(self.percent_levels)}
        self.x_positions = np.arange(len(self.percent_levels), dtype=float)

        # Remap each solver to unified positions; store mutable mu/sd aligned to percent_levels
        self.series = {}
        self.original_backup = {}
        for tag in raw:
            mu_full = np.zeros(len(self.percent_levels), dtype=float)
            sd_full = np.zeros(len(self.percent_levels), dtype=float)

            # fill only for existing xs
            for p, m, s in zip(raw[tag]["xs"], raw[tag]["mu"], raw[tag]["sd"]):
                idx = self.pos_map[p]
                mu_full[idx] = float(m)
                sd_full[idx] = float(s)

            self.series[tag] = dict(
                mu=mu_full,
                sd=sd_full,
                label=raw[tag]["label"],
                color=raw[tag]["color"],
                marker=raw[tag]["marker"],
                z=raw[tag]["z"],
            )
            # backup for reset
            self.original_backup[tag] = (mu_full.copy(), sd_full.copy())

        # edit state
        self.edit_mode = "mean"  # or "std"
        self.active = None  # (tag, idx, kind) where kind in {"mean","std"}
        self._press_event = None

        # Build figure
        plt.rcParams.update({
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        })

        self.fig, self.ax = plt.subplots(figsize=(5.2, 3.9))
        self.fig.subplots_adjust(bottom=0.18)  # room for buttons

        self.lines = {}
        self.bands = {}
        self.std_handles = {}  # handles at mu+sd (only visible in std mode)

        # Plot order: CONFER on top (also legend ordering)
        plot_order = ["OURS", "ARVI", "SAIA"]
        for tag in plot_order:
            if tag not in self.series:
                continue
            self._draw_solver(tag)

        # ticks / labels like your original script
        self.ax.set_xticks(self.x_positions)
        self.ax.set_xlim(-0.5, len(self.percent_levels) - 0.5)
        self.ax.set_xticklabels([f"{int(p)}%" if float(p).is_integer() else f"{p:g}%" for p in self.percent_levels])

        self.ax.set_xlabel("Percentage landmarks")
        self.ax.set_ylabel("Inference cost")
        style_axes(self.ax)
        self.ax.legend(frameon=False, loc="upper right", ncol=1)

        self._autoscale_y()

        # buttons
        self._add_buttons()

        # help text
        self.status = self.ax.text(
            0.01, 1.02, self._status_text(),
            transform=self.ax.transAxes, ha="left", va="bottom"
        )

        # connect events
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self._apply_mode_visibility()

    def _status_text(self):
        return f"Edit mode: {self.edit_mode.upper()}   (press 'E' to toggle, 'R' to reset)"

    def _draw_solver(self, tag: str):
        s = self.series[tag]
        x = self.x_positions
        mu = s["mu"]
        sd = s["sd"]

        # line
        (ln,) = self.ax.plot(
            x, mu,
            label=s["label"],
            color=s["color"],
            linewidth=LINE_WIDTH,
            marker=s["marker"],
            markersize=MARKER_SIZE,
            markerfacecolor=s["color"],
            markeredgecolor="black",
            markeredgewidth=0.6,
            solid_capstyle="round",
            zorder=s["z"],
        )
        self.lines[tag] = ln

        # band
        band = self.ax.fill_between(
            x, mu - sd, mu + sd,
            color=s["color"], alpha=ALPHA_BAND, linewidth=0, zorder=s["z"] - 1
        )
        self.bands[tag] = band

        # std handles at upper band (mu+sd), light outline; only used for picking in std mode
        (h,) = self.ax.plot(
            x, mu + sd,
            linestyle="None",
            marker="D",
            markersize=max(6, MARKER_SIZE - 1),
            markerfacecolor="white",
            markeredgecolor=s["color"],
            markeredgewidth=1.2,
            alpha=0.95,
            zorder=s["z"] + 0.5,
        )
        self.std_handles[tag] = h

    def _redraw_solver(self, tag: str):
        s = self.series[tag]
        x = self.x_positions
        mu = s["mu"]
        sd = s["sd"]

        self.lines[tag].set_ydata(mu)
        self.std_handles[tag].set_ydata(mu + sd)

        # Replace band (simplest robust approach)
        self.bands[tag].remove()
        self.bands[tag] = self.ax.fill_between(
            x, mu - sd, mu + sd,
            color=s["color"], alpha=ALPHA_BAND, linewidth=0, zorder=s["z"] - 1
        )

    def _autoscale_y(self):
        ymax = 0.0
        for tag in self.series:
            mu = self.series[tag]["mu"]
            sd = self.series[tag]["sd"]
            ymax = max(ymax, float(np.max(mu + sd)))
        self.ax.set_ylim(0, ymax + 0.12 * max(1.0, ymax) + 5.0)

    def _add_buttons(self):
        # Save Data
        ax_save = self.fig.add_axes([0.12, 0.04, 0.18, 0.075])
        self.btn_save = Button(ax_save, "Save Data")
        self.btn_save.on_clicked(self.on_save_data)

        # Save Figure
        ax_fig = self.fig.add_axes([0.32, 0.04, 0.18, 0.075])
        self.btn_fig = Button(ax_fig, "Save Figure")
        self.btn_fig.on_clicked(self.on_save_fig)

        # Toggle help text area is in status line; no extra button needed.

    def _apply_mode_visibility(self):
        # In mean mode: show curve markers, hide std handles
        # In std mode: show std handles (mu+sd diamonds) to drag, keep curve markers still visible
        show_std = (self.edit_mode == "std")
        for tag in self.series:
            self.std_handles[tag].set_visible(show_std)

        self.status.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def _pick_nearest(self, event):
        """Return (tag, idx, kind) or None."""
        if event.inaxes != self.ax:
            return None

        # Convert mouse to display coords for distance computations
        x_mouse, y_mouse = event.x, event.y

        best = None
        best_d = float("inf")

        for tag in self.series:
            x = self.x_positions

            if self.edit_mode == "mean":
                y = self.lines[tag].get_ydata()
            else:
                y = self.std_handles[tag].get_ydata()

            # compute screen distances
            xy = self.ax.transData.transform(np.column_stack([x, y]))
            dx = xy[:, 0] - x_mouse
            dy = xy[:, 1] - y_mouse
            d = np.hypot(dx, dy)
            idx = int(np.argmin(d))
            if float(d[idx]) < best_d:
                best_d = float(d[idx])
                kind = "mean" if self.edit_mode == "mean" else "std"
                best = (tag, idx, kind)

        if best is not None and best_d <= PICK_TOL_PX:
            return best
        return None

    def on_press(self, event):
        picked = self._pick_nearest(event)
        if picked is None:
            self.active = None
            return
        self.active = picked
        self._press_event = event

    def on_release(self, event):
        self.active = None
        self._press_event = None

    def on_motion(self, event):
        if self.active is None:
            return
        if event.inaxes != self.ax:
            return
        if event.ydata is None:
            return

        tag, idx, kind = self.active
        y = float(event.ydata)

        if kind == "mean":
            # clamp to >= 0
            self.series[tag]["mu"][idx] = max(0.0, y)
        else:
            # std mode: y corresponds to mu + sd
            mu = float(self.series[tag]["mu"][idx])
            new_sd = max(0.0, y - mu)
            self.series[tag]["sd"][idx] = new_sd

        self._redraw_solver(tag)
        self._autoscale_y()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key is None:
            return
        k = event.key.lower()

        if k == "e":
            self.edit_mode = "std" if self.edit_mode == "mean" else "mean"
            self._apply_mode_visibility()

        elif k == "r":
            # reset all
            for tag in self.series:
                mu0, sd0 = self.original_backup[tag]
                self.series[tag]["mu"][:] = mu0
                self.series[tag]["sd"][:] = sd0
                self._redraw_solver(tag)
            self._autoscale_y()
            self.fig.canvas.draw_idle()

    def on_save_data(self, _event):
        domain_dir = os.path.join(BASE, self.domain)
        ensure_dir(domain_dir)

        # Save per solver: edited mean and std (two-col each), one row per percent
        for tag, _, _, _, _ in SOLVERS:
            if tag not in self.series:
                continue
            mu = self.series[tag]["mu"]
            sd = self.series[tag]["sd"]
            p_mean = edited_mean_file(self.domain, tag)
            p_std  = edited_std_file(self.domain, tag)

            with open(p_mean, "w") as f:
                f.write("#percentage_landmarks\tinference_cost_mean\n")
                for p, m in zip(self.percent_levels, mu):
                    f.write(f"{p:g}\t{m:.6f}\n")

            with open(p_std, "w") as f:
                f.write("#percentage_landmarks\tinference_cost_std\n")
                for p, s in zip(self.percent_levels, sd):
                    f.write(f"{p:g}\t{s:.6f}\n")

            print(f"[ok] saved: {p_mean}")
            print(f"[ok] saved: {p_std}")

        print("[ok] edited data saved (mean + std). Re-running this script will load *_edited*.txt automatically.")

    def on_save_fig(self, _event):
        ensure_dir(OUT_DIR)
        out_png = os.path.join(OUT_DIR, f"{self.domain}_inference_cost_vs_landmarks_EDITED.png")
        self.fig.savefig(out_png, bbox_inches="tight")
        print(f"[ok] saved: {out_png}")


def main():
    editor = InteractiveEditor(DOMAIN)
    plt.show()


if __name__ == "__main__":
    main()

# animate_trajectory_with_belief_display_forestfire.py
# Usage:
#   python animate_trajectory_with_belief_display_forestfire.py <total_contexts> <true_context> <map_name> <saveAnimation(0/1)>

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patheffects as path_effects
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from collections import defaultdict, deque
import math

def _ensure_multi_fire_with_status(frames):
    """
    Normalize to: frames := list[ frame ], frame := list[ (fx, fy, ext) ].
    Accepts legacy [[fx,fy], ...] (single fire) and multi [[[fx,fy(,ext)], ...], ...].
    ext: 0 = burning, 1 = extinguished.
    """
    if frames is None:
        return None

    out = []
    for fr in frames:
        # Legacy single-fire frame is a flat pair/triple
        if isinstance(fr, (list, tuple)) and fr and isinstance(fr[0], (int, float)):
            x = int(fr[0]); y = int(fr[1])
            ext = int(fr[2]) if len(fr) >= 3 else 0
            out.append([(x, y, 0 if ext == 0 else 1)])
            continue

        # Multi-fire frame
        cells = []
        if isinstance(fr, (list, tuple)):
            for c in fr:
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    x = int(c[0]); y = int(c[1])
                    ext = int(c[2]) if len(c) >= 3 else 0
                    cells.append((x, y, 0 if ext == 0 else 1))
        out.append(cells)
    return out

# ---------------- CLI ----------------
total_contexts = int(sys.argv[1])          # e.g., 2
true_context   = int(sys.argv[2])          # e.g., 0
map_name       = sys.argv[3]               # e.g., "forestfire-60-60"
saveAnimation  = bool(int(sys.argv[4]))    # 0/1
writer = PillowWriter(fps=4)

# ---- Fire-path generation knobs (mirror the planner defaults) ----
FIRE_HMAX        = 120
FIRE_BETA        = 1.0
FIRE_SEED        = 42
FIRE_MOVE_PERIOD = 2

FLIP_PERIOD = 1  # mirroring cadence for the flame echo

# Persistent overlays for “encircled fire” rectangles (static-fire mode)
RECT_GROUPS  = {}
RECTS_DRAWN  = {}
STATIC_RECTS = []

# Fire glyphs & state
FIRE_OBJS = []           # list of dicts from draw_fire_icon; each gets 'cell'=(x,y)
FIRE_BY_CELL = {}        # (x,y) -> glyph dict
ASH_BY_CELL  = {}        # (x,y) -> list[patch] for ash icons
FIRE_EXT_CELLS = set()   # cells flagged as extinguished (skip flicker)

# Fire blocks & overlays (static mode only)
FIRE_BLOCKS  = []        # dicts with bbox, cells, corners_ok, etc.
PERSIST_EXTINGUISH = False

# -------- Burn logic: local, reset each repeat --------
burned_cells = set()            # set of (x,y) that have been marked burnt in this cycle
burn_overlays = {}              # (x,y) -> the semi-transparent overlay Rectangle
PREV_FIRES   = []               # used to hide flames that disappear


# --------------- IO helpers ---------------
def load_grid(filename):
    grid = []
    with open(filename, "r") as f:
        # Skip header until "map"
        for line in f:
            if line.strip().lower() == "map":
                break
        for line in f:
            row = list(line.strip())
            if row:
                grid.append(row)
    return grid

def load_cost_grid(filename):
    grid = []
    with open(filename, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            grid.append([int(v) for v in s.split()])
    return grid

def _first_fire_cell_from_map(grid):
    H = len(grid); W = len(grid[0]) if H else 0
    for y in range(H):
        for x in range(W):
            if grid[y][x] == 'F':
                return (x, y)
    return None

def _neighbors4(x, y, W, H):
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H:
            yield nx, ny

def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(z)/len(z)

def generate_fire_path(grid, fuel_grid, Hmax, beta, seed):
    """
    Single-cell fire that moves 4-connected.
    P(next) ∝ exp(beta * fuel_value(next)); ignores obstacles '@'.
    If no valid move, stays in place.
    Returns [(x0,y0), (x1,y1), ...] of length Hmax+1 (includes start).
    """
    H = len(grid); W = len(grid[0]) if H else 0
    start = _first_fire_cell_from_map(grid)
    if start is None:
        return []
    rng = np.random.RandomState(seed)
    path = [start]
    for _ in range(Hmax):
        x, y = path[-1]
        cands, scores = [], []
        for nx, ny in _neighbors4(x, y, W, H):
            if grid[ny][nx] == '@':
                continue
            fv = fuel_grid[ny][nx] if (0 <= ny < len(fuel_grid) and 0 <= nx < len(fuel_grid[ny])) else 0
            cands.append((nx, ny))
            scores.append(beta * float(fv))
        if not cands:
            path.append((x, y))
        else:
            p = _softmax(scores)
            k = rng.choice(len(cands), p=p)
            path.append(cands[k])
    return path

# --------------- Load data ---------------
with open("../results/MAPF_OUTPUT/trajectory.json", "r") as f:
    data = json.load(f)
trajectory = data["trajectory"]

# Optional dynamic fire trajectory (per-frame list of (fx,fy[,ext]))
FIRE_TRAJ_PATH = "../results/MAPF_OUTPUT/fire_trajectory.json"
fire_frames = None
if os.path.exists(FIRE_TRAJ_PATH):
    with open(FIRE_TRAJ_PATH, "r") as f:
        raw = json.load(f)
    fire_frames = _ensure_multi_fire_with_status(raw)

grid_data = load_grid(f"../maps/{map_name}/{map_name}.map")
obj2_grid = load_cost_grid(f"../maps/{map_name}/cost-2.cost")
obj3_grid = load_cost_grid(f"../maps/{map_name}/cost-3.cost")

H = len(grid_data)
W = len(grid_data[0]) if H > 0 else 0

# Choose the fuel grid to bias fire motion. Switch to obj2_grid if desired.
FUEL_GRID = obj3_grid

# --- hard reset if this file is run again in the same interpreter/kernel ---
def _hard_reset():
    global FIRE_OBJS, FIRE_BY_CELL, FIRE_EXT_CELLS, FIRE_BLOCKS
    global STATIC_RECTS, RECTS_DRAWN, RECT_GROUPS, ASH_BY_CELL
    global PREV_FIRES

    FIRE_OBJS      = []
    FIRE_BY_CELL   = {}
    FIRE_EXT_CELLS = set()
    FIRE_BLOCKS    = []
    STATIC_RECTS   = []
    RECTS_DRAWN    = {}
    RECT_GROUPS    = {}
    ASH_BY_CELL    = {}
    PREV_FIRES     = []

plt.close('all')
_hard_reset()               # ensure a fresh state every run

def _find_fire_blocks(grid):
    """Return connected components of 'F' cells, with bbox and cells."""
    H = len(grid); W = len(grid[0]) if H else 0
    seen = [[False]*W for _ in range(H)]
    blocks = []
    for y in range(H):
        for x in range(W):
            if grid[y][x] != 'F' or seen[y][x]:
                continue
            q = deque([(x,y)])
            seen[y][x] = True
            xs, ys, cells = [x], [y], [(x,y)]
            while q:
                cx, cy = q.popleft()
                for nx, ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                    if 0 <= nx < W and 0 <= ny < H and not seen[ny][nx] and grid[ny][nx] == 'F':
                        seen[ny][nx] = True
                        q.append((nx,ny))
                        xs.append(nx); ys.append(ny); cells.append((nx,ny))
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            rect_ok = all(grid[yy][xx] == 'F' for yy in range(ymin, ymax+1) for xx in range(xmin, xmax+1))
            blocks.append({'bbox': (xmin, ymin, xmax, ymax),
                           'cells': [(xx,yy) for yy in range(ymin, ymax+1) for xx in range(xmin, xmax+1)],
                           'rect_ok': rect_ok})
    return blocks

def _guard_corners(bbox, grid, clamp=True):
    """
    Diagonal-outside corners for a bbox. Valid iff in-bounds & not '@'.
    """
    W = len(grid[0]); H = len(grid)
    xmin, ymin, xmax, ymax = bbox
    raw = [(xmin-1, ymin-1), (xmax+1, ymin-1), (xmax+1, ymax+1), (xmin-1, ymax+1)]

    out, ok = [], True
    for x, y in raw:
        if 0 <= x < W and 0 <= y < H and grid[y][x] != '@':
            out.append((x, y))
        else:
            ok = False
            if clamp:
                cx = min(max(x, 0), W-1)
                cy = min(max(y, 0), H-1)
                if grid[cy][cx] != '@':
                    out.append((cx, cy))
                else:
                    out.append((x, y))
    if len(set(out)) != 4:
        ok = False
    return out, ok

# Build FIRE_BLOCKS now (used only in static-fire mode)
FIRE_BLOCKS = []
for blk in _find_fire_blocks(grid_data):
    corners, ok = _guard_corners(blk['bbox'], grid_data)
    blk.update({
        'corners': corners,
        'corners_ok': ok,
        'extinguished': False,
        'rect_patch': None
    })
    FIRE_BLOCKS.append(blk)

goal_list   = data.get("goals", [])
goal_coords = [(g[0], g[1]) for g in goal_list]

# --------------- Figure ---------------
contexts      = [fr"$c_{i}$" for i in range(total_contexts)]
initial_probs = [1/total_contexts for _ in range(total_contexts)]

fig, (ax_grid, ax_bar) = plt.subplots(
    1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]}
)
ax_grid.set_box_aspect(1)
ax_bar.set_box_aspect(1)
ax_grid.set_xlim(0, W)
ax_grid.set_ylim(0, H)
ax_grid.set_aspect('equal')
ax_grid.invert_yaxis()

# --------------- Base grid ---------------
grid_patches = {}
FOREST_BG = "darkseagreen"  # muted green
for y in range(H):
    for x in range(W):
        rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5,
                                 edgecolor='black', facecolor=FOREST_BG)
        ax_grid.add_patch(rect)
        grid_patches[(x, y)] = rect

# Obstacles
for y in range(H):
    for x in range(W):
        if grid_data[y][x] == "@":
            grid_patches[(x, y)].set_facecolor("black")

# Static overlays: fire, landmarks, obj2 (eneryy), obj3 (debris)
for y in range(H):
    for x in range(W):
        t = grid_data[y][x]
        if fire_frames is None and t == "F":
            # static flames only if no dynamic trajectory
            # (static mode has no burn-in; only visual flames)
            pass  # we draw the icon on-demand when needed
        if t == "L":
            star = ax_grid.text(x+0.5, y+0.5, "★", color='yellow', fontsize=10,
                                ha='center', va='center', zorder=6)
            star.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])
        if (y < len(obj2_grid) and x < len(obj2_grid[y]) and obj2_grid[y][x] > 0):
            # wind icon
            # (definition appears below; call after function is defined)
            pass
        if (y < len(obj3_grid) and x < len(obj3_grid[y]) and obj3_grid[y][x] > 0):
            # reserve icon (draw later for simplicity)
            pass

# --------------- Icon drawing (vector) ---------------

def draw_energy_icon(ax, x, y):
    """
    Energy depletion sink:
    - Battery outline with terminal cap
    - Downward chevron/arrow inside
    """
    # Cell-local coordinates
    cx, cy = x + 0.5, y + 0.5
    w, h = 0.66, 0.38
    left = cx - w/2; bottom = cy - h/2

    # Battery body
    body = patches.FancyBboxPatch(
        (left, bottom), w, h,
        boxstyle=patches.BoxStyle.Round(pad=0.02, rounding_size=0.08),
        facecolor="red", edgecolor="red", linewidth=1.6, zorder=5
    )
    ax.add_patch(body)

    # Terminal cap
    cap_w, cap_h = 0.10, 0.18
    cap = patches.FancyBboxPatch(
        (left + w, cy - cap_h/2), cap_w, cap_h,
        boxstyle=patches.BoxStyle.Round(pad=0.0, rounding_size=0.04),
        facecolor="red", edgecolor="yellow", linewidth=1.0, zorder=5.1
    )
    ax.add_patch(cap)

    # Inner “depletion” arrow (downward chevron)
    a_w, a_h = 0.42, 0.26
    tip_y = cy + 0.08
    arrow = patches.Polygon(
        [
            (cx - a_w/2, tip_y - a_h*0.55),
            (cx + a_w/2, tip_y - a_h*0.55),
            (cx,          tip_y + a_h*0.45),
        ],
        closed=True, facecolor="gold", edgecolor="gold", linewidth=1.2, zorder=6
    )
    ax.add_patch(arrow)

    # Optional dim bar to suggest “low energy”
    lvl_w, lvl_h = 0.48, 0.08
    lvl = patches.Rectangle(
        (cx - lvl_w/2, bottom + 0.08), lvl_w*0.22, lvl_h,
        facecolor="red", edgecolor="none", zorder=6
    )
    ax.add_patch(lvl)


def draw_debris_icon(ax, x, y):
    """
    Fire debris / rubble:
    - Irregular charcoal fragments with a few faint ember specks.
    """
    base_z = 5
    rng = np.random.RandomState(hash((x, y)) % 2**32)

    # Three irregular fragments
    for k in range(3):
        # Randomized small polygon around center-bottom of the cell
        cx = x + 0.35 + 0.18*k + 0.03*rng.randn()
        cy = y + 0.66 + 0.02*rng.randn()
        r  = 0.16 + 0.02*rng.randn()
        pts = []
        for i in range(6):
            ang = 2*np.pi*i/6 + 0.2*rng.randn()
            rr  = r * (0.75 + 0.35*rng.rand())
            pts.append((cx + rr*np.cos(ang), cy + 0.6*rr*np.sin(ang)))
        poly = patches.Polygon(
            pts, closed=True,
            facecolor="#3b3b3b", edgecolor="#0e0e0e", linewidth=1.0, zorder=base_z
        )
        ax.add_patch(poly)

    # A few ember specks
    for _ in range(5):
        ex = x + 0.25 + 0.5*rng.rand()
        ey = y + 0.40 + 0.35*rng.rand()
        dot = patches.Circle((ex, ey), 0.015 + 0.007*rng.rand(),
                             facecolor="#ff6a00", edgecolor="none", alpha=0.8, zorder=base_z+0.5)
        ax.add_patch(dot)

    # Light ash haze (very subtle)
    haze = patches.Ellipse((x+0.5, y+0.65), 0.85, 0.38, angle=0,
                           facecolor=(0, 0, 0, 0.10), edgecolor="none", zorder=base_z-0.2)
    ax.add_patch(haze)

def draw_wind_icon(ax, x, y):
    xL, xR = x + 0.12, x + 0.88
    L = xR - xL
    amp = 0.11
    bases = [y + 0.36, y + 0.54, y + 0.72]

    def streamline(y0, lw=1.0):
        xm = xL + 0.5*L
        verts = [
            (xL, y0),
            (xL + 0.20*L, y0 - amp),
            (xm - 0.12*L, y0 + amp),
            (xm, y0),
            (xm + 0.12*L, y0 - 0.9*amp),
            (xR - 0.18*L, y0 + 0.9*amp),
            (xR, y0),
        ]
        codes = [Path.MOVETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4]
        p = Path(verts, codes)
        s1 = PathPatch(p, facecolor='none', edgecolor='#5ca8ff', lw=lw,
                       capstyle='round', joinstyle='round', zorder=5)
        s1.set_path_effects([
            path_effects.Stroke(linewidth=lw+0.9, foreground='black', alpha=0.45),
            path_effects.Normal()
        ])
        ax.add_patch(s1)
        s2 = PathPatch(p, facecolor='none', edgecolor='#a9ccff',
                       lw=max(0.8, lw-0.8), capstyle='round', joinstyle='round',
                       zorder=6, alpha=0.85)
        ax.add_patch(s2)

    for i, y0 in enumerate(bases):
        streamline(y0, lw=2.2 - 0.2*i)

def draw_fire_icon(ax, x, y):
    cx, cy = x + 0.5, y + 0.62

    outer_verts = [
        (cx,      y+0.95),
        (x+0.20,  y+0.85), (x+0.22, y+0.55), (x+0.42, y+0.48),
        (x+0.50,  y+0.28), (x+0.62, y+0.18), (x+0.63, y+0.34),
        (x+0.86,  y+0.52), (x+0.70, y+0.80), (cx,     y+0.95),
    ]
    oc = [Path.MOVETO] + [Path.CURVE4]*(len(outer_verts)-1)
    outer_verts.append(outer_verts[0]); oc.append(Path.CLOSEPOLY)
    outer_path = Path(outer_verts, oc)
    outer = PathPatch(outer_path, facecolor='#ff8c00', edgecolor='#7a1e00', lw=1.3, zorder=6)
    outer.set_path_effects([
        path_effects.Stroke(linewidth=outer.get_linewidth()+0.6, foreground='black', alpha=0.35),
        path_effects.Normal()
    ])

    def scaled_copy(path, s, dy=0.0, face='none', z=7):
        verts = []
        for px, py in path.vertices[:-1]:
            dx, dy0 = px - cx, py - cy
            verts.append((cx + s*dx, cy + s*dy0 + dy))
        codes = [Path.MOVETO] + [Path.CURVE4]*(len(verts)-1)
        verts.append(verts[0]); codes.append(Path.CLOSEPOLY)
        return PathPatch(Path(verts, codes), facecolor=face, edgecolor='none', zorder=z)

    mid   = scaled_copy(outer_path, 0.78, -0.02, face='#ff6d00',  z=7)
    inner = scaled_copy(outer_path, 0.56, -0.06, face='#ffd54d',  z=8)

    ax.add_patch(outer); ax.add_patch(mid); ax.add_patch(inner)

    ghost_colors = ['#ff9b3a', '#ff7f2a', '#ffe088']
    ghosts = []
    for face, z in zip(ghost_colors, [5, 5, 5]):
        g = PathPatch(outer_path, facecolor=face, edgecolor='none', alpha=0.26, zorder=z)
        ax.add_patch(g)
        ghosts.append(g)

    return {
        'patches': [outer, mid, inner],
        'ghosts': ghosts,
        'cx': cx, 'cy': cy,
        'base_colors': [outer.get_facecolor(), mid.get_facecolor(), inner.get_facecolor()],
        'ghost_base_alpha': [g.get_alpha() for g in ghosts],
        'phase':  2*np.pi*np.random.rand(),
        'phase2': 2*np.pi*np.random.rand(),
        'prev_sx': 1.0, 'prev_sy': 1.0,
        'prev_dx': 0.0, 'prev_dy': 0.0,
        'prev_mirror': +1
    }

def draw_extinguished_icon(ax, x, y):
    """
    Extinguished cell marker:
    - Full 1×1 cell rectangle
    - Medium-gray fill with diagonal hatch
    - Drawn on ax_grid above the black burned tile
    Returns a list of patches (for show/hide management).
    """
    patches_out = []

    # Full cell bounds
    rect = patches.Rectangle(
        (x, y), 1.0, 1.0,
        facecolor="gray",     # medium gray so it pops on black
        edgecolor="black",     # darker hatch/edge
        linewidth=1.0,
        hatch="///",             # diagonal hatch
        zorder=8                 # above tiles & flames; below drones (z=9)
    )
    ax.add_patch(rect)
    patches_out.append(rect)

    return patches_out



def display_flame_cell(ax, x, y):
    obj = FIRE_BY_CELL.get((x, y))
    if obj is None:
        obj = draw_fire_icon(ax, x, y)
        obj['cell'] = (x, y)
        FIRE_OBJS.append(obj)
        FIRE_BY_CELL[(x, y)] = obj
    for p in obj['patches'] + obj['ghosts']:
        p.set_visible(True)
    if (x, y) in ASH_BY_CELL:
        for a in ASH_BY_CELL[(x, y)]:
            a.set_visible(False)
    if (x, y) in FIRE_EXT_CELLS:
        FIRE_EXT_CELLS.remove((x, y))

def display_ash_cell(ax, x, y):
    obj = FIRE_BY_CELL.get((x, y))
    if obj:
        for p in obj['patches'] + obj['ghosts']:
            p.set_visible(False)
    if (x, y) not in ASH_BY_CELL:
        ASH_BY_CELL[(x, y)] = draw_extinguished_icon(ax, x, y)
    else:
        for a in ASH_BY_CELL[(x, y)]:
            a.set_visible(True)
    FIRE_EXT_CELLS.add((x, y))

def display_flame_block(ax, blk):
    for (x, y) in blk['cells']:
        display_flame_cell(ax, x, y)
    if blk.get('rect_patch'):
        blk['rect_patch'].set_visible(False)
    blk['extinguished'] = False

def display_extinguished_block(ax, blk):
    if not blk.get('rect_patch'):
        g = blk['corners']
        rect_verts = [(g[0][0] + 0.5, g[0][1] + 0.5),
                      (g[1][0] + 0.5, g[1][1] + 0.5),
                      (g[2][0] + 0.5, g[2][1] + 0.5),
                      (g[3][0] + 0.5, g[3][1] + 0.5)]
        poly = patches.Polygon(rect_verts, closed=True, fill=False,
                               edgecolor="#1b5e20", linewidth=2.6, alpha=0.9, zorder=2)
        try:
            poly.set_path_effects([path_effects.Stroke(linewidth=4.2, foreground=(1,1,1,0.25)),
                                   path_effects.Normal()])
        except Exception:
            pass
        ax.add_patch(poly)
        blk['rect_patch'] = poly
    else:
        blk['rect_patch'].set_visible(True)

    for (x, y) in blk['cells']:
        obj = FIRE_BY_CELL.get((x, y))
        if obj:
            for p in obj['patches'] + obj['ghosts']:
                p.set_visible(False)
        if (x, y) not in ASH_BY_CELL:
            ASH_BY_CELL[(x, y)] = draw_extinguished_icon(ax, x, y)
        else:
            for a in ASH_BY_CELL[(x, y)]:
                a.set_visible(True)
        FIRE_EXT_CELLS.add((x, y))
    blk['extinguished'] = True

def hide_fire_cell(x, y):
    obj = FIRE_BY_CELL.get((x, y))
    if obj:
        for p in obj['patches'] + obj['ghosts']:
            p.set_visible(False)

# Draw energy-sink (obj2) and debris (obj3) icons now that functions exist
for y in range(H):
    for x in range(W):
        if (y < len(obj2_grid) and x < len(obj2_grid[y]) and obj2_grid[y][x] > 0):
            draw_energy_icon(ax_grid, x, y)
        if (y < len(obj3_grid) and x < len(obj3_grid[y]) and obj3_grid[y][x] > 0):
            draw_debris_icon(ax_grid, x, y)

# --------------- Belief bars ---------------
bars = ax_bar.bar(contexts, initial_probs, color=['blue'] * total_contexts)
ax_bar.set_ylim(0, 1.1)
ax_bar.set_ylabel("Probability")
ax_bar.set_xlabel("Possible Contexts")
ax_bar.set_title("Context Belief Probabilities")

# --------------- Agents (dynamic) ---------------
agent_patches = []

# ---------- Drone glyph (agents): FIXED ----------
def draw_drone(ax, cx, cy, scale=0.34, z=8):
    """
    Draw a quadrotor: central body + 4 arms + 4 rotors.
    Yellow outlines improve visibility over black tiles.
    Returns a list of added patches/artists.
    """
    artists = []

    # Central body (rounded square)
    body_w = 0.55 * scale
    body_h = 0.55 * scale
    body_round = patches.BoxStyle.Round(pad=0.01, rounding_size=0.15 * min(body_w, body_h))
    body = patches.FancyBboxPatch(
        (cx - body_w/2, cy - body_h/2), body_w, body_h,
        boxstyle=body_round,
        facecolor="r", edgecolor="r", linewidth=1.6, zorder=z
    )
    ax.add_patch(body); artists.append(body)

    # Arms (thin cross) — rounded rectangles
    arm_len = 1.40 * scale
    arm_w   = 0.14 * scale
    arm_round = patches.BoxStyle.Round(pad=0.0, rounding_size=arm_w/2)

    # horizontal arm
    h_arm = patches.FancyBboxPatch(
        (cx - arm_len/2, cy - arm_w/2), arm_len, arm_w,
        boxstyle=arm_round,
        facecolor="red", edgecolor="gold", linewidth=3, zorder=z
    )
    ax.add_patch(h_arm); artists.append(h_arm)

    # vertical arm
    v_arm = patches.FancyBboxPatch(
        (cx - arm_w/2, cy - arm_len/2), arm_w, arm_len,
        boxstyle=arm_round,
        facecolor="red", edgecolor="gold", linewidth=3, zorder=z
    )
    ax.add_patch(v_arm); artists.append(v_arm)

    # Rotors (4 discs with yellow rim)
    r_R = 0.20 * scale
    off = 0.78 * scale
    rotor_centers = [
        (cx - off, cy - off), (cx + off, cy - off),
        (cx + off, cy + off), (cx - off, cy + off)
    ]
    for (rx, ry) in rotor_centers:
        ring = patches.Circle((rx, ry), r_R*1.15, facecolor="#000000",
                              edgecolor="#ffd400", linewidth=1.3, zorder=z+0.1)
        ax.add_patch(ring); artists.append(ring)
        disc = patches.Circle((rx, ry), r_R, facecolor="#c0c0c0",
                              edgecolor="#303030", linewidth=0.8, zorder=z+0.2)
        ax.add_patch(disc); artists.append(disc)

    # Subtle outline for contrast (optional)
    try:
        body.set_path_effects([
            path_effects.Stroke(linewidth=body.get_linewidth()+0.8, foreground="#000000", alpha=0.35),
            path_effects.Normal()
        ])
        for arm in (h_arm, v_arm):
            arm.set_path_effects([
                path_effects.Stroke(linewidth=arm.get_linewidth()+0.6, foreground="#000000", alpha=0.35),
                path_effects.Normal()
            ])
    except Exception:
        pass

    return artists


def draw_agents(joint):
    """
    Render agents as quadrotor drones. If multiple agents share a cell,
    place them on a small ring to avoid overlap.
    Returns a list of added artists (for later removal).
    """
    patches_list, counts = {}, {}

    # Count agents per cell
    for ag in joint:
        x, y, _ = ag
        counts[(x, y)] = counts.get((x, y), 0) + 1

    for (x, y), k in counts.items():
        cx, cy = x + 0.5, y + 0.5

        if k == 1:
            # Single drone at center
            arts = draw_drone(ax_grid, cx, cy, scale=0.42, z=9)
            patches_list.setdefault((x, y), []).extend(arts)
        else:
            # Place k drones on a small circle around center
            ring_r = 0.22  # cell units
            for i in range(k):
                theta = 2 * math.pi * i / k
                dx = ring_r * math.cos(theta)
                dy = ring_r * math.sin(theta)
                arts = draw_drone(ax_grid, cx + dx, cy + dy, scale=0.36, z=9)
                patches_list.setdefault((x, y), []).extend(arts)

    # Flatten to list for the caller
    flat = []
    for v in patches_list.values():
        flat.extend(v)
    return flat


# -------- Burn logic: local, reset each repeat --------
burned_cells = set()      # <--- LOCAL state; no global
PREV_FIRES   = []         # used to hide flames that disappear

def ensure_burnt(ax, x, y, alpha=0.55):
    """
    Add a semi-transparent black overlay on top of the cell so original content
    (trees, water, icons) remains faintly visible. Idempotent per cycle.
    """
    if (x, y) in burned_cells:
        return

    if (x, y) not in burn_overlays:
        overlay = patches.Rectangle(
            (x, y), 1.0, 1.0,
            facecolor=(0, 0, 0, alpha),  # translucent black
            edgecolor='none',
            # Choose zorder so it sits above base grid (≈1) but below most icons (≥3–6)
            zorder=2.0,
        )
        ax.add_patch(overlay)
        burn_overlays[(x, y)] = overlay

    burned_cells.add((x, y))

# -------- Animation init: called at the start of every cycle --------
def init():
    burned_cells.clear()

    # remove any overlay patches we created last cycle
    for rect in burn_overlays.values():
        try:
            rect.remove()
        except Exception:
            pass
    burn_overlays.clear()

    PREV_FIRES.clear()

    # Reset base tiles
    for y in range(H):
        for x in range(W):
            if grid_data[y][x] == "@":
                grid_patches[(x, y)].set_facecolor("black")
            else:
                grid_patches[(x, y)].set_facecolor(FOREST_BG)

    # Hide all flames, ash, and static rectangles
    for F in FIRE_OBJS:
        for p in F['patches'] + F['ghosts']:
            p.set_visible(False)
        # reset previous transform memory to avoid echo glitch at t=0
        F['prev_sx'], F['prev_sy'] = 1.0, 1.0
        F['prev_dx'], F['prev_dy'] = 0.0, 0.0
        F['prev_mirror'] = +1

    for ashes in ASH_BY_CELL.values():
        for a in ashes:
            a.set_visible(False)

    for blk in FIRE_BLOCKS:
        if blk.get('rect_patch'):
            blk['rect_patch'].set_visible(False)
        blk['extinguished'] = False

    # Clear any previously drawn agents
    for p in list(agent_patches):
        try:
            p.remove()
        except Exception:
            pass
    agent_patches.clear()

    # Set initial bar heights/titles (optional)
    for bar, h in zip(bars, initial_probs):
        bar.set_height(h)
    ax_bar.set_title("Context Belief Probabilities")
    ax_grid.set_title("")

    return []

def update(frame):
    t = frame
    state = trajectory[frame]
    joint = state["joint"]

    # Remove previous agents
    for p in agent_patches:
        p.remove()
    agent_patches[:] = draw_agents(joint)

    # Agent positions this frame (cell coordinates)
    agent_xy = {(a[0], a[1]) for a in joint}

    # --- flame flicker + deterministic left/right mirroring + one-frame echo ---
    for F in FIRE_OBJS:
        if F.get('cell') in FIRE_EXT_CELLS:
            continue
        cx, cy = F['cx'], F['cy']

        s  = 1.0 + 0.07*math.sin(0.35*frame + F['phase']) + 0.03*math.sin(0.77*frame + F['phase2'])
        sx, sy = s, s*1.08

        dx = 0.015*math.sin(0.61*frame + F['phase'])
        dy = 0.010*math.cos(0.49*frame + F['phase2'])

        mirror_sign = -1 if ((frame // FLIP_PERIOD) % 2) else +1

        T = (Affine2D()
             .translate(-cx, -cy)
             .scale(mirror_sign * sx, sy)
             .translate(cx, cy)
             .translate(dx, dy)
             + ax_grid.transData)

        for i, p in enumerate(F['patches']):
            p.set_transform(T)
            base = F['base_colors'][i]
            k = 0.94 + 0.08*math.sin(0.55*frame + F['phase'] + 0.4*i)
            r, g, b, a = base
            p.set_facecolor((min(1, r*k), min(1, g*k), min(1, b*k), a))

        ECHO_SHIFT = 0.02
        Tghost = (Affine2D()
                  .translate(-cx, -cy)
                  .scale(F['prev_mirror'] * F['prev_sx'], F['prev_sy'])
                  .translate(cx, cy)
                  .translate(F['prev_dx'] + ECHO_SHIFT, F['prev_dy'])
                  + ax_grid.transData)

        for i, g in enumerate(F['ghosts']):
            g.set_transform(Tghost)
            base_alpha = F['ghost_base_alpha'][i]
            g.set_alpha(max(0.0, min(1.0, base_alpha * (0.9 + 0.15*math.sin(0.45*frame + F['phase'] + 0.3*i)))))

        F['prev_sx'], F['prev_sy']   = sx, sy
        F['prev_dx'], F['prev_dy']   = dx, dy
        F['prev_mirror']             = mirror_sign

    # ---- dynamic fire position (from C++ dump) ----
    if fire_frames is not None and len(fire_frames) > 0:
        cells = fire_frames[min(frame, len(fire_frames) - 1)]
        if isinstance(cells, (list, tuple)) and cells and isinstance(cells[0], (int, float)):
            cells = [cells[:2]]

        cur_set = set()
        for c in cells:
            fx = int(c[0]); fy = int(c[1])
            ext = int(c[2]) if (isinstance(c, (list, tuple)) and len(c) >= 3) else 0

            # burn **now** (first time this cell is on fire in this cycle)
            ensure_burnt(ax_grid, fx, fy)

            if ext == 1:
                display_ash_cell(ax_grid, fx, fy)
            else:
                display_flame_cell(ax_grid, fx, fy)
                cur_set.add((fx, fy))

        # hide flames that are no longer burning
        for (x, y) in set(PREV_FIRES) - cur_set:
            hide_fire_cell(x, y)
        PREV_FIRES[:] = list(cur_set)

    # For dynamic fire (from file), skip static block logic.
    if fire_frames is None:
        for blk in FIRE_BLOCKS:
            if not blk['corners_ok']:
                display_flame_block(ax_grid, blk)
                continue
            occupied_now = all(c in agent_xy for c in blk['corners'])
            if occupied_now:
                display_extinguished_block(ax_grid, blk)
            else:
                if not PERSIST_EXTINGUISH:
                    display_flame_block(ax_grid, blk)

    # Goal highlighting (recomputed each frame)
    for y in range(H):
        for x in range(W):
            if grid_data[y][x] == "G":
                if state['contexts'] != [true_context]:
                    grid_patches[(x, y)].set_facecolor((0, 1, 0, 0.3))
                else:
                    if (x, y) in goal_coords:
                        grid_patches[(x, y)].set_facecolor('green')
                    else:
                        grid_patches[(x, y)].set_facecolor((0, 1, 0, 0.3))

    # Titles + belief bars
    if len(state['contexts']) > 1:
        ax_grid.set_title(f"Context uncertain (t = {t})")
        target_belief = [0] * total_contexts
        for c in state['contexts']:
            target_belief[c] += 1 / len(state['contexts'])
    else:
        ax_grid.set_title(rf"Context identified as $c_{state['contexts'][0]}$ (t = {t})")
        target_belief = [0] * total_contexts
        target_belief[state['contexts'][0]] = 1.0

    for bar, h in zip(bars, target_belief):
        bar.set_height(h)
    ax_bar.set_title(
        "Beliefs: [" + ", ".join([f"{contexts[i]}: {target_belief[i]:.2f}" for i in range(total_contexts)]) + "]"
    )

    # Return a list of artists if you want to try blitting; we keep blit=False for simplicity.
    return agent_patches + [ax_grid.title]

# --------------- Animate ---------------
ani = FuncAnimation(
    fig, update,
    frames=len(trajectory),
    init_func=init,          # <--- resets per repeat
    interval=250,
    blit=False,
    repeat=True,
    repeat_delay=500         # small pause before restarting (optional)
)

num_agents = len(trajectory[0]['joint'])
if saveAnimation:
    ani.save(f"../animation/forestfire_C{true_context}_A{num_agents}.gif", writer=writer)


# # ---------------- Reliability Plot (1 / |contexts| over time) ----------------
# # Compute reliability for each step based on the size of the current context set.
# reliability_values = []
# for s in trajectory:
#     k = len(s.get('contexts', []))
#     k = max(1, k)  # safety guard
#     reliability_values.append(1.0 / k)

# steps = list(range(len(reliability_values)))

# # Create the reliability figure
# fig_rel = plt.figure()
# ax_rel = fig_rel.add_subplot(111)
# ax_rel.plot(steps, reliability_values, linewidth=2)
# ax_rel.set_xlabel("Step")
# ax_rel.set_ylabel("Confidence of operation (max(belief))")
# ax_rel.set_title("Confidence over trajectory")
# ax_rel.set_ylim(0, 1.05)
# ax_rel.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

# # Save alongside the animation if requested
# try:
#     os.makedirs("../figures", exist_ok=True)
# except Exception:
#     pass

# if saveAnimation:
#     try:
#         fig_rel.savefig(f"../figures/reliability_C{true_context}_A{num_agents}.png", dpi=200, bbox_inches="tight")
#     except Exception:
#         pass
# # ---------------------------------------------------------------------------

plt.show()


#!/usr/bin/env python3
# animate_warehouse.py (updated)
# Usage:
#   python animate_warehouse.py <total_contexts> <true_context> <map_name> <saveAnimation(0/1)>
#
# Changes in this version:
#   • Landmarks 'L' are rendered as shelves (not QR markers).
#   • When an agent steps onto an 'L' cell, that shelf disappears (picked up) and
#     that agent carries a cardboard box overlay for the remainder of the run.
#   • Objective icon overlays (battery/human) are disabled for now (toggle via flags).
#
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# # ---------------- CLI ----------------
# if len(sys.argv) < 5:
#     print("Usage: python animate_warehouse.py <total_contexts> <true_context> <map_name> <saveAnimation(0/1)>")
#     sys.exit(1)

total_contexts = 3
true_context   = 2
map_name       = "warehouse"
saveAnimation  = False
writer = PillowWriter(fps=4)

# Feature flags
SHOW_OBJ2 = False  # battery overlay (energy) — disabled
SHOW_OBJ3 = False  # human overlay — disabled
SHOW_BELIEF = False  # belief bar plot — enabled
first_frame_drawn = False

# ---------------- IO helpers ----------------
def load_grid(filename):
    grid = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip().lower() == "map":
                break
        for line in f:
            row = list(line.strip())
            if row:
                grid.append(row)
    return grid

def load_int_grid(filename):
    grid = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [int(v) for v in line.split()]
            grid.append(row)
    return grid

# ---------------- Data ----------------
with open("../results/MAPF_OUTPUT/trajectory.json", "r") as f:
    data = json.load(f)

trajectory = data["trajectory"]

grid_data = load_grid(f"../maps/{map_name}/{map_name}.map")
# Optional cost layers (kept for future use, but icons disabled by flags)
obj2_grid  = None
obj3_grid  = None
try:
    obj2_grid  = load_int_grid(f"../maps/{map_name}/cost-2.cost")
    obj3_grid  = load_int_grid(f"../maps/{map_name}/cost-3.cost")
except Exception:
    obj2_grid = obj3_grid = None

H = len(grid_data)
W = len(grid_data[0]) if H > 0 else 0

goal_list = data.get("goals", [])
goal_coords = [(g[0], g[1]) for g in goal_list]

# ---------------- Belief ----------------
contexts = [fr"$c_{i}$" for i in range(total_contexts)]
initial_probs = [1/total_contexts for _ in range(total_contexts)]
current_belief = initial_probs.copy()

# ---------------- Figure ----------------
if SHOW_BELIEF:
    fig, (ax_grid, ax_bar) = plt.subplots(1, 2, figsize=(12, 6),
                                           gridspec_kw={'width_ratios': [1, 1]})
    ax_bar.set_box_aspect(1)
else:
    fig, ax_grid = plt.subplots(1, 1, figsize=(6, 6))
    ax_bar = None  # sentinel when belief plot is hidden
ax_grid.set_box_aspect(1)

# Grid setup
ax_grid.set_xlim(0, W)
ax_grid.set_ylim(0, H)
ax_grid.set_aspect('equal')
ax_grid.invert_yaxis()

# ---------------- Drawing helpers ----------------
def draw_shelf_cell(x, y, z=0.5):
    """Render a shelf as a framed unit with slats."""
    base = patches.Rectangle((x, y), 1, 1, facecolor="#B08D57", edgecolor="#5a4320", linewidth=1.2, zorder=z)
    ax_grid.add_patch(base)
    inner = patches.Rectangle((x+0.06, y+0.06), 0.88, 0.88, facecolor="#C9AB77", edgecolor="#5a4320", linewidth=0.6, zorder=z+0.1)
    ax_grid.add_patch(inner)
    post_w = 0.05
    for px in (x+0.06, x+0.89-post_w):
        post = patches.Rectangle((px, y+0.06), post_w, 0.88, facecolor="#8C6A3B", edgecolor="#5a4320", linewidth=0.6, zorder=z+0.2)
        ax_grid.add_patch(post)
    for frac in (0.30, 0.55, 0.80):
        slat = patches.Rectangle((x+0.06, y+frac-0.01), 0.88, 0.02, facecolor="#7A5A2E", edgecolor="#5a4320", linewidth=0.3, zorder=z+0.3)
        ax_grid.add_patch(slat)
    return  # purely drawn

def draw_kiva_robot(cx, cy, scale=1.0, z=3.5):
    patches_added = []
    body_r = 0.28 * scale
    body = patches.Circle((cx, cy), radius=body_r, edgecolor="black", facecolor="#ff8c00", linewidth=1.6, zorder=z)
    ax_grid.add_patch(body); patches_added.append(body)
    ring_r = 0.18 * scale
    ring = patches.Circle((cx, cy), radius=ring_r, edgecolor="black", facecolor="white", linewidth=1.1, zorder=z+0.1)
    ax_grid.add_patch(ring); patches_added.append(ring)
    sens_r = 0.06 * scale
    sensor = patches.Circle((cx, cy), radius=sens_r, edgecolor="black", facecolor="black", linewidth=1.0, zorder=z+0.2)
    ax_grid.add_patch(sensor); patches_added.append(sensor)
    return patches_added

def draw_box_overlay(cx, cy, scale=1.0, z=4.0):
    patches_added = []
    bw = 0.54 * scale
    bh = 0.40 * scale
    box = patches.FancyBboxPatch(
        (cx - bw/2, cy - bh/2), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        edgecolor="#3b2d1f", facecolor="#C49A6C", linewidth=1.2, alpha=0.78, zorder=z
    )
    tape_w = 0.05 * scale
    tape1 = patches.Rectangle((cx - tape_w - 0.02*scale, cy - bh/2 + 0.03*scale),
                              tape_w, bh - 0.06*scale, edgecolor=None, facecolor="#e2d5b5", alpha=0.85, zorder=z+0.05)
    tape2 = patches.Rectangle((cx + 0.02*scale, cy - bh/2 + 0.03*scale),
                              tape_w, bh - 0.06*scale, edgecolor=None, facecolor="#e2d5b5", alpha=0.85, zorder=z+0.05)
    label = patches.Rectangle((cx + 0.08*scale, cy - 0.02*scale), 0.10*scale, 0.06*scale,
                              edgecolor="#3b2d1f", facecolor="#f0e7cf", linewidth=0.6, alpha=0.9, zorder=z+0.1)
    for p in (box, tape1, tape2, label):
        ax_grid.add_patch(p); patches_added.append(p)
    return patches_added

def draw_kiosk_icon(x, y, size=0.84):
    cx = x + 0.5
    cy = y + 0.5
    s  = size
    bw = 0.42 * s; bh = 0.10 * s
    base = patches.FancyBboxPatch((cx - bw/2, cy + 0.20*s), bw, bh,
                                  boxstyle="round,pad=0.01,rounding_size=0.02",
                                  facecolor="#6e6e6e", edgecolor="#333", linewidth=1.0, zorder=2.0)
    ax_grid.add_patch(base)
    pw = 0.06 * s; ph = 0.40 * s
    post = patches.Rectangle((cx - pw/2, cy - ph/2 + 0.12*s), pw, ph,
                             facecolor="#777", edgecolor="#333", linewidth=1.0, zorder=2.1)
    ax_grid.add_patch(post)
    sw = 0.48 * s; sh = 0.28 * s
    screen = patches.FancyBboxPatch((cx - sw/2, cy - 0.35*s), sw, sh,
                                    boxstyle="round,pad=0.015,rounding_size=0.03",
                                    facecolor="#222", edgecolor="#111", linewidth=1.0, zorder=2.2)
    ax_grid.add_patch(screen)
    dw = sw * 0.86; dh = sh * 0.70
    disp = patches.Rectangle((cx - dw/2, cy - 0.35*s + (sh - dh)/2), dw, dh,
                             facecolor="#90caf9", edgecolor="#0d47a1", linewidth=0.8, zorder=2.3)
    ax_grid.add_patch(disp)
    slot = patches.Rectangle((cx - dw*0.25, cy - 0.35*s + sh*0.80), dw*0.50, dh*0.12,
                             facecolor="#e3f2fd", edgecolor="#0d47a1", linewidth=0.6, zorder=2.4)
    ax_grid.add_patch(slot)

# ---------------- Base grid ----------------
grid_patches = {}
for y in range(H):
    for x in range(W):
        rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5, edgecolor='black', facecolor='white', zorder=0.1)
        ax_grid.add_patch(rect)
        grid_patches[(x, y)] = rect

# Shelves '@' and Landmarks 'L' as shelves (collect L patches to remove on pickup)
L_cells = set()
L_shelf_patches = {}  # (x,y) -> list of patches
for y in range(H):
    for x in range(W):
        ch = grid_data[y][x]
        if ch == "@":
            draw_shelf_cell(x, y)
        elif ch == "L":
            L_cells.add((x, y))
            patches_list = []
            # draw 'L' shelves with slightly different tones to be distinguishable
            base = patches.Rectangle((x, y), 1, 1, facecolor="#B08D57", edgecolor="#5a4320", linewidth=1.2, zorder=0.55)
            inner = patches.Rectangle((x+0.06, y+0.06), 0.88, 0.88, facecolor="#D3B685", edgecolor="#5a4320", linewidth=0.6, zorder=0.6)
            ax_grid.add_patch(base); ax_grid.add_patch(inner)
            patches_list.extend([base, inner])
            post_w = 0.05
            for px in (x+0.06, x+0.89-post_w):
                post = patches.Rectangle((px, y+0.06), post_w, 0.88, facecolor="#9C7845", edgecolor="#5a4320", linewidth=0.6, zorder=0.7)
                ax_grid.add_patch(post); patches_list.append(post)
            for frac in (0.30, 0.55, 0.80):
                slat = patches.Rectangle((x+0.06, y+frac-0.01), 0.88, 0.02, facecolor="#866233", edgecolor="#5a4320", linewidth=0.3, zorder=0.8)
                ax_grid.add_patch(slat); patches_list.append(slat)
            L_shelf_patches[(x, y)] = patches_list

# Static overlays (disabled by flags for now)
if SHOW_OBJ2 and obj2_grid is not None:
    for y in range(H):
        for x in range(W):
            if y < len(obj2_grid) and x < len(obj2_grid[y]) and obj2_grid[y][x] == 1:
                grid_patches[(x, y)].set_facecolor("#ffe5e5")

if SHOW_OBJ3 and obj3_grid is not None:
    for y in range(H):
        for x in range(W):
            if y < len(obj3_grid) and x < len(obj3_grid[y]) and obj3_grid[y][x] == 1:
                grid_patches[(x, y)].set_facecolor("#eef5ff")

# ---- Delivery kiosks at goal coordinates (static, visible every frame) ----
for (gx, gy) in goal_coords:
    if 0 <= gy < H and 0 <= gx < W:
        draw_kiosk_icon(gx, gy, size=0.90)

# ---------------- Belief bar plot ----------------
bars = None
if SHOW_BELIEF:
    bars = ax_bar.bar(contexts, initial_probs)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_ylabel("Probability")
    ax_bar.set_xlabel("Possible Contexts")
    ax_bar.set_title("Context Belief Probabilities")

# ---------------- Agent rendering ----------------
agent_patches = []
picked_L = set()  # landmark shelves already picked up (removed)
boxes_active_for = set()  # set of agent indices carrying boxes

def draw_agents(joint, boxes_set):
    patches_list = []
    # group by cell for multi-occupancy layout
    cell_to_indices = {}
    for i, a in enumerate(joint):
        try:
            x, y = a[0], a[1]
        except Exception:
            x, y = a[:2]
        cell_to_indices.setdefault((x, y), []).append(i)

    for (x, y), idxs in cell_to_indices.items():
        cx, cy = x + 0.5, y + 0.5
        cnt = len(idxs)
        if cnt == 1:
            i = idxs[0]
            patches_list += draw_kiva_robot(cx, cy, scale=1.0)
            if i in boxes_set:
                patches_list += draw_box_overlay(cx, cy, scale=1.0)
        else:
            rad = 0.22
            for k, i in enumerate(idxs):
                ang = 2*np.pi * (k / cnt)
                ox = rad * np.cos(ang)
                oy = rad * np.sin(ang)
                scale = 0.72
                patches_list += draw_kiva_robot(cx + ox, cy + oy, scale=scale)
                if i in boxes_set:
                    patches_list += draw_box_overlay(cx + ox, cy + oy, scale=scale)
    return patches_list

# ---------------- Frame update ----------------
def update(frame):
    global agent_patches, picked_L, boxes_active_for, first_frame_drawn
    t = frame
    state = trajectory[frame]
    joint = state["joint"]
    
    # When the animation restarts (frame goes back to 0), reset everything:
    if frame == 0 and first_frame_drawn:
        boxes_active_for.clear()   # no agents carry boxes
        picked_L.clear()           # no landmarks considered picked
        # show all landmark-shelf patches again
        for plist in L_shelf_patches.values():
            for p in plist:
                p.set_visible(True)

    # Landmark pickup logic
    for i, a in enumerate(joint):
        try:
            x, y = a[0], a[1]
        except Exception:
            x, y = a[:2]
        if (x, y) in L_cells and (x, y) not in picked_L:
            # Remove drawn shelf for this landmark cell
            # hide the drawn shelf (so we can restore on loop)
            patches_list = L_shelf_patches.get((x, y), [])
            for p in patches_list:
                p.set_visible(False)
            picked_L.add((x, y))
            boxes_active_for.add(i)

    # Clear previously drawn agents (and overlays)
    for p in agent_patches:
        try:
            p.remove()
        except Exception:
            pass
    agent_patches = []

    # Draw current agents with per-agent boxes
    agent_patches = draw_agents(joint, boxes_active_for)

    # Goals visualization on map 'G' cells (distinct from kiosks)
    for y in range(H):
        for x in range(W):
            if grid_data[y][x] == "G":
                if state.get('contexts', []) != [true_context]:
                    grid_patches[(x, y)].set_facecolor((0, 1, 0, 0.28))
                else:
                    if (x, y) in goal_coords:
                        grid_patches[(x, y)].set_facecolor('green')
                    else:
                        grid_patches[(x, y)].set_facecolor((0, 1, 0, 0.28))

    # Titles and belief bars (unchanged semantics)
    if len(state.get('contexts', [])) > 1:
        ax_grid.set_title(f"Context uncertain (t = {t})")
        target_belief = [0.0] * total_contexts
        for c in state['contexts']:
            if 0 <= c < total_contexts:
                target_belief[c] += 1.0 / len(state['contexts'])
    else:
        c = state.get('contexts', [true_context])[0]
        ax_grid.set_title(rf"Context identified as $c_{c}$ (t = {t})")
        target_belief = [0.0] * total_contexts
        if 0 <= c < total_contexts:
            target_belief[c] = 1.0

    if SHOW_BELIEF and bars is not None:
        for bar, h in zip(bars, target_belief):
            bar.set_height(h)
        ax_bar.set_title("Beliefs: [" + ", ".join(
            [f"{contexts[i]}: {h:.2f}" for i, h in enumerate(target_belief)]
        ) + "]")

    first_frame_drawn = True  # mark that at least one frame has been drawn for animation reset purposes
    
    return agent_patches + [ax_grid.title]

# ---------------- Animate ----------------
ani = FuncAnimation(fig, update, frames=len(trajectory), interval=250, blit=False, repeat=True)

# Save (optional)
num_agents = len(trajectory[0]['joint']) if trajectory and 'joint' in trajectory[0] else 0
if saveAnimation:
    ani.save(f"../animation/warehouse_C{true_context}_A{num_agents}.gif", writer=writer)

plt.show()

# # ---------------- Reliability Plot ----------------
# reliability_values = []
# for s in trajectory:
#     k = len(s.get('contexts', []))
#     k = max(1, k)
#     reliability_values.append(1.0 / k)

# steps = list(range(len(reliability_values)))
# fig_rel = plt.figure()
# ax_rel = fig_rel.add_subplot(111)
# ax_rel.plot(steps, reliability_values, linewidth=2)
# ax_rel.set_xlabel("Step")
# ax_rel.set_ylabel("Confidence of operation (max(belief))")
# ax_rel.set_title("Confidence over trajectory")
# ax_rel.set_ylim(0, 1.05)
# ax_rel.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

# try:
#     os.makedirs("../figures", exist_ok=True)
# except Exception:
#     pass

# if saveAnimation:
#     try:
#         fig_rel.savefig(f"../figures/reliability_C{true_context}_A{num_agents}.png", dpi=200, bbox_inches="tight")
#     except Exception:
#         pass

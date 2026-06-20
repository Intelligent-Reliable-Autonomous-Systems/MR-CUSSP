#!/usr/bin/env python3
# animate_warehouse.py
# Usage:
#   python animate_warehouse.py <total_contexts> <true_context> <map_name> <saveAnimation(0/1)>
#
# Visual + behavior summary:
#   - '@' cells rendered as warehouse shelves.
#   - Agents are orange (Kiva-style top view).
#   - Objective-2 (energy depletion == 1) cells: pale-red tint + RED battery icon.
#   - Objective-3 (human == 1) cells: subtle blue tint + human pictogram.
#   - Landmarks 'L': rendered as QR-like markers (denser pattern).
#   - Belief collapse to the true context: ALL agents carry a cargo box from that frame to the end.
#   - Goal coordinates (goal_coords): rendered as delivery kiosks (static overlays visible every frame).
#
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patheffects as path_effects

# ---------------- CLI ----------------
if len(sys.argv) < 5:
    print("Usage: python animate_warehouse.py <total_contexts> <true_context> <map_name> <saveAnimation(0/1)>")
    sys.exit(1)

total_contexts = int(sys.argv[1])
true_context   = int(sys.argv[2])
map_name       = sys.argv[3]
saveAnimation  = bool(int(sys.argv[4]))
writer = PillowWriter(fps=4)

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
obj2_grid  = load_int_grid(f"../maps/{map_name}/cost-2.cost")   # energy depletion mask
obj3_grid  = load_int_grid(f"../maps/{map_name}/cost-3.cost")   # human presence mask

H = len(grid_data)
W = len(grid_data[0]) if H > 0 else 0

goal_list = data.get("goals", [])
goal_coords = [(g[0], g[1]) for g in goal_list]

# ---------------- Belief ----------------
contexts = [fr"$c_{i}$" for i in range(total_contexts)]
initial_probs = [1/total_contexts for _ in range(total_contexts)]
current_belief = initial_probs.copy()

# ---------------- Figure ----------------
fig, (ax_grid, ax_bar) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
ax_grid.set_box_aspect(1)
ax_bar.set_box_aspect(1)

# Grid setup
ax_grid.set_xlim(0, W)
ax_grid.set_ylim(0, H)
ax_grid.set_aspect('equal')
ax_grid.invert_yaxis()

# ---------------- Drawing helpers ----------------
def draw_shelf_cell(x, y):
    """Render a '@' obstacle as a warehouse shelf with frame and slats."""
    base = patches.Rectangle((x, y), 1, 1, facecolor="#B08D57", edgecolor="#5a4320", linewidth=1.2, zorder=0.5)
    ax_grid.add_patch(base)
    inner = patches.Rectangle((x+0.06, y+0.06), 0.88, 0.88, facecolor="#C9AB77", edgecolor="#5a4320", linewidth=0.6, zorder=0.6)
    ax_grid.add_patch(inner)
    post_w = 0.05
    for px in (x+0.06, x+0.89-post_w):
        post = patches.Rectangle((px, y+0.06), post_w, 0.88, facecolor="#8C6A3B", edgecolor="#5a4320", linewidth=0.6, zorder=0.7)
        ax_grid.add_patch(post)
    for frac in (0.30, 0.55, 0.80):
        slat = patches.Rectangle((x+0.06, y+frac-0.01), 0.88, 0.02, facecolor="#7A5A2E", edgecolor="#5a4320", linewidth=0.3, zorder=0.8)
        ax_grid.add_patch(slat)

def draw_battery_icon(cx, cy, scale=0.9):
    """Red battery icon centered at cell (cx, cy)."""
    px = cx + 0.5
    py = cy + 0.5
    bw = 0.58 * scale
    bh = 0.34 * scale
    body = patches.FancyBboxPatch(
        (px - bw/2, py - bh/2), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        edgecolor="black", facecolor="#E53935", linewidth=1.2, zorder=1.6
    )
    tw = 0.10 * scale
    th = 0.12 * scale
    term = patches.Rectangle((px + bw/2 - 0.015, py - th/2), tw, th,
                             edgecolor="black", facecolor="#B71C1C", linewidth=1.0, zorder=1.6)
    ind = patches.Rectangle((px - bw/2 + 0.05*scale, py - 0.10*scale), 0.10*scale, 0.20*scale,
                            edgecolor="black", facecolor="white", linewidth=0.8, zorder=1.7)
    for p in (body, term, ind):
        ax_grid.add_patch(p)
    return [body, term, ind]

def draw_human_icon(cx, cy, scale=0.95):
    """Human pictogram centered at cell (cx, cy)."""
    px = cx + 0.5
    py = cy + 0.5
    head = patches.Circle((px, py - 0.20*scale), 0.10*scale, edgecolor="black", facecolor="#2E2E2E", linewidth=1.2, zorder=1.6)
    torso = patches.FancyBboxPatch((px - 0.22*scale/2, py - 0.05*scale), 0.22*scale, 0.34*scale,
                                   boxstyle="round,pad=0.02,rounding_size=0.03",
                                   edgecolor="black", facecolor="#444", linewidth=1.2, zorder=1.6)
    arm_l = patches.FancyArrow(px - 0.02, py + 0.02*scale, -0.24*scale, 0, width=0.001,
                               edgecolor="black", facecolor="#222", zorder=1.6, length_includes_head=True)
    arm_r = patches.FancyArrow(px + 0.02, py + 0.02*scale,  0.24*scale, 0, width=0.001,
                               edgecolor="black", facecolor="#222", zorder=1.6, length_includes_head=True)
    leg_l = patches.FancyArrow(px - 0.06, py + 0.16*scale, -0.06*scale, 0.28*scale, width=0.001,
                               edgecolor="black", facecolor="#222", zorder=1.6, length_includes_head=True)
    leg_r = patches.FancyArrow(px + 0.06, py + 0.16*scale,  0.06*scale, 0.28*scale, width=0.001,
                               edgecolor="black", facecolor="#222", zorder=1.6, length_includes_head=True)
    for p in (head, torso, arm_l, arm_r, leg_l, leg_r):
        ax_grid.add_patch(p)
    return [head, torso, arm_l, arm_r, leg_l, leg_r]

def draw_kiva_robot(cx, cy, scale=1.0):
    """Orange Kiva-style robot at (cx, cy)."""
    patches_added = []
    body_r = 0.28 * scale
    body = patches.Circle((cx, cy), radius=body_r, edgecolor="black", facecolor="#ff8c00", linewidth=1.6, zorder=3.5)
    ax_grid.add_patch(body); patches_added.append(body)
    ring_r = 0.18 * scale
    ring = patches.Circle((cx, cy), radius=ring_r, edgecolor="black", facecolor="white", linewidth=1.1, zorder=3.6)
    ax_grid.add_patch(ring); patches_added.append(ring)
    sens_r = 0.06 * scale
    sensor = patches.Circle((cx, cy), radius=sens_r, edgecolor="black", facecolor="black", linewidth=1.0, zorder=3.7)
    ax_grid.add_patch(sensor); patches_added.append(sensor)
    return patches_added

def draw_box_overlay(cx, cy, scale=1.0):
    """Semi-transparent cardboard cargo box centered at (cx, cy)."""
    patches_added = []
    bw = 0.54 * scale
    bh = 0.40 * scale
    box = patches.FancyBboxPatch(
        (cx - bw/2, cy - bh/2), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        edgecolor="#3b2d1f", facecolor="#C49A6C", linewidth=1.2, alpha=0.78, zorder=4.0
    )
    tape_w = 0.05 * scale
    tape1 = patches.Rectangle((cx - tape_w - 0.02*scale, cy - bh/2 + 0.03*scale),
                              tape_w, bh - 0.06*scale, edgecolor=None, facecolor="#e2d5b5", alpha=0.85, zorder=4.1)
    tape2 = patches.Rectangle((cx + 0.02*scale, cy - bh/2 + 0.03*scale),
                              tape_w, bh - 0.06*scale, edgecolor=None, facecolor="#e2d5b5", alpha=0.85, zorder=4.1)
    label = patches.Rectangle((cx + 0.08*scale, cy - 0.02*scale), 0.10*scale, 0.06*scale,
                              edgecolor="#3b2d1f", facecolor="#f0e7cf", linewidth=0.6, alpha=0.9, zorder=4.2)
    for p in (box, tape1, tape2, label):
        ax_grid.add_patch(p); patches_added.append(p)
    return patches_added

def draw_qr_marker(x, y, size=0.86, modules_n=9, fill_levels=(0,1,2,4,6)):
    """
    Denser QR-like marker centered in cell (x, y).
      - Three main finder patterns (TL, TR, BL).
      - Small alignment finder near BR.
      - Higher-resolution data area (modules_n × modules_n).
    """
    cx = x + 0.5
    cy = y + 0.5
    s = size
    left = cx - s/2
    top  = cy - s/2

    # background
    bg = patches.Rectangle((left, top), s, s, facecolor="white",
                           edgecolor="black", linewidth=1.4, zorder=2.2)
    ax_grid.add_patch(bg)

    def finder(px, py, fsize):
        outer = patches.Rectangle((px, py), fsize, fsize, facecolor="black",
                                  edgecolor="black", linewidth=0.0, zorder=2.3)
        m = fsize * 0.65
        mid_off = (fsize - m)/2
        middle = patches.Rectangle((px + mid_off, py + mid_off), m, m,
                                   facecolor="white", edgecolor="white",
                                   linewidth=0.0, zorder=2.4)
        c = m * 0.55
        ctr_off = (m - c)/2
        center = patches.Rectangle((px + mid_off + ctr_off, py + mid_off + ctr_off), c, c,
                                   facecolor="black", edgecolor="black",
                                   linewidth=0.0, zorder=2.5)
        for p in (outer, middle, center):
            ax_grid.add_patch(p)

    # main finders
    fsize = s * 0.26
    pad   = s * 0.05
    finder(left + pad,             top + pad,             fsize)  # TL
    finder(left + s - pad - fsize, top + pad,             fsize)  # TR
    finder(left + pad,             top + s - pad - fsize, fsize)  # BL

    # small alignment finder (BR)
    a = s * 0.14
    ax0 = left + s - pad - a - 0.02
    ay0 = top  + s - pad - a - 0.02
    align_outer  = patches.Rectangle((ax0, ay0), a, a, facecolor="black",
                                     edgecolor="black", linewidth=0.0, zorder=2.3)
    align_middle = patches.Rectangle((ax0 + a*0.2, ay0 + a*0.2), a*0.6, a*0.6,
                                     facecolor="white", edgecolor="white",
                                     linewidth=0.0, zorder=2.4)
    align_center = patches.Rectangle((ax0 + a*0.35, ay0 + a*0.35), a*0.3, a*0.3,
                                     facecolor="black", edgecolor="black",
                                     linewidth=0.0, zorder=2.5)
    for p in (align_outer, align_middle, align_center):
        ax_grid.add_patch(p)

    # data area bounds between finders
    grid_left   = left + fsize + pad*1.05
    grid_top    = top  + fsize + pad*1.05
    grid_right  = left + s - pad*1.05 - fsize
    grid_bottom = top  + s - pad*1.05 - fsize
    gw = max(0.0, grid_right - grid_left)
    gh = max(0.0, grid_bottom - grid_top)

    # dense module grid
    mod_n = int(max(5, modules_n))
    if mod_n <= 0 or gw == 0.0 or gh == 0.0:
        return
    mod_s = min(gw / mod_n, gh / mod_n)

    for gx in range(mod_n):
        for gy in range(mod_n):
            if ((3*gx + 5*gy) % 7) in (0,1,2,4,6):
                rx = grid_left + gx * mod_s + mod_s*0.12
                ry = grid_top  + gy * mod_s + mod_s*0.12
                dot = patches.Rectangle((rx, ry), mod_s*0.76, mod_s*0.76,
                                        facecolor="black", edgecolor="black",
                                        linewidth=0.0, zorder=2.6)
                ax_grid.add_patch(dot)

def draw_kiosk_icon(x, y, size=0.84):
    """
    Delivery kiosk icon centered in cell (x, y):
      - pedestal base + vertical post
      - tablet/screen at the top with bluish display
    """
    cx = x + 0.5
    cy = y + 0.5
    s  = size

    # base plate
    bw = 0.42 * s
    bh = 0.10 * s
    base = patches.FancyBboxPatch((cx - bw/2, cy + 0.20*s), bw, bh,
                                  boxstyle="round,pad=0.01,rounding_size=0.02",
                                  facecolor="#6e6e6e", edgecolor="#333", linewidth=1.0, zorder=2.0)
    ax_grid.add_patch(base)

    # vertical post
    pw = 0.06 * s
    ph = 0.40 * s
    post = patches.Rectangle((cx - pw/2, cy - ph/2 + 0.12*s), pw, ph,
                             facecolor="#777", edgecolor="#333", linewidth=1.0, zorder=2.1)
    ax_grid.add_patch(post)

    # screen housing
    sw = 0.48 * s
    sh = 0.28 * s
    screen = patches.FancyBboxPatch((cx - sw/2, cy - 0.35*s), sw, sh,
                                    boxstyle="round,pad=0.015,rounding_size=0.03",
                                    facecolor="#222", edgecolor="#111", linewidth=1.0, zorder=2.2)
    ax_grid.add_patch(screen)

    # display content
    dw = sw * 0.86
    dh = sh * 0.70
    disp = patches.Rectangle((cx - dw/2, cy - 0.35*s + (sh - dh)/2), dw, dh,
                             facecolor="#90caf9", edgecolor="#0d47a1", linewidth=0.8, zorder=2.3)
    ax_grid.add_patch(disp)

    # small "scan slot" hint
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

# Shelves for '@'
for y in range(H):
    for x in range(W):
        if grid_data[y][x] == "@":
            draw_shelf_cell(x, y)

# Landmarks (QR-like markers)
for y in range(H):
    for x in range(W):
        if grid_data[y][x] == "L":
            draw_qr_marker(x, y, size=0.86, modules_n=9)

# Static overlays: Obj2 red battery + red tint; Obj3 human + blue tint
static_icon_patches = []
for y in range(H):
    for x in range(W):
        if y < len(obj2_grid) and x < len(obj2_grid[y]) and obj2_grid[y][x] == 1:
            grid_patches[(x, y)].set_facecolor("#ffe5e5")
            static_icon_patches += draw_battery_icon(x, y, scale=0.95)
        if y < len(obj3_grid) and x < len(obj3_grid[y]) and obj3_grid[y][x] == 1:
            grid_patches[(x, y)].set_facecolor("#eef5ff")
            static_icon_patches += draw_human_icon(x, y, scale=0.95)

# ---- Delivery kiosks at goal coordinates (static, visible every frame) ----
for (gx, gy) in goal_coords:
    if 0 <= gy < H and 0 <= gx < W:
        draw_kiosk_icon(gx, gy, size=0.90)

# ---------------- Belief bar plot ----------------
bars = ax_bar.bar(contexts, initial_probs)
ax_bar.set_ylim(0, 1.1)
ax_bar.set_ylabel("Probability")
ax_bar.set_xlabel("Possible Contexts")
ax_bar.set_title("Context Belief Probabilities")

# ---------------- Agent rendering ----------------
agent_patches = []
boxes_active = False  # becomes True at first frame where contexts == [true_context]

def draw_agents(joint, with_boxes=False):
    # cell -> list of agent indices present
    cell_to_indices = {}
    for i, a in enumerate(joint):
        try:
            x, y = a[0], a[1]
        except Exception:
            x, y = a[:2]
        cell_to_indices.setdefault((x, y), []).append(i)

    patches_list = []
    for (x, y), idxs in cell_to_indices.items():
        cx, cy = x + 0.5, y + 0.5
        cnt = len(idxs)
        if cnt == 1:
            patches_list += draw_kiva_robot(cx, cy, scale=1.0)
            if with_boxes:
                patches_list += draw_box_overlay(cx, cy, scale=1.0)
        else:
            rad = 0.22
            for k, i in enumerate(idxs):
                ang = 2*np.pi * (k / cnt)
                ox = rad * np.cos(ang)
                oy = rad * np.sin(ang)
                scale = 0.72
                patches_list += draw_kiva_robot(cx + ox, cy + oy, scale=scale)
                if with_boxes:
                    patches_list += draw_box_overlay(cx + ox, cy + oy, scale=scale)
    return patches_list

# ---------------- Frame update ----------------
def update(frame):
    global current_belief, agent_patches, boxes_active
    t = frame
    state = trajectory[frame]
    joint = state["joint"]

    # Activate once when belief collapses to the true context
    if (not boxes_active) and len(state.get('contexts', [])) == 1 and state['contexts'][0] == true_context:
        boxes_active = True

    # Clear previously drawn agents (and their overlays)
    for p in agent_patches:
        try:
            p.remove()
        except Exception:
            pass
    agent_patches = []

    # Draw current agents (+ box overlays for all agents if active)
    agent_patches = draw_agents(joint, with_boxes=boxes_active)

    # Goals visualization on map 'G' cells (separate from kiosk icons)
    for y in range(H):
        for x in range(W):
            if grid_data[y][x] == "G":
                if state['contexts'] != [true_context]:
                    grid_patches[(x, y)].set_facecolor((0, 1, 0, 0.28))
                else:
                    if (x, y) in goal_coords:
                        grid_patches[(x, y)].set_facecolor('green')
                    else:
                        grid_patches[(x, y)].set_facecolor((0, 1, 0, 0.28))

    # Titles and belief bars
    if len(state['contexts']) > 1:
        ax_grid.set_title(f"Context uncertain (t = {t})")
        target_belief = [0.0] * total_contexts
        for c in state['contexts']:
            if 0 <= c < total_contexts:
                target_belief[c] += 1.0 / len(state['contexts'])
    else:
        c = state['contexts'][0]
        ax_grid.set_title(rf"Context identified as $c_{c}$ (t = {t})")
        target_belief = [0.0] * total_contexts
        if 0 <= c < total_contexts:
            target_belief[c] = 1.0

    for bar, h in zip(bars, target_belief):
        bar.set_height(h)
    ax_bar.set_title("Beliefs: [" + ", ".join([f"{contexts[i]}: {h:.2f}" for i, h in enumerate(target_belief)]) + "]")

    return agent_patches + [ax_grid.title]

# ---------------- Animate ----------------
ani = FuncAnimation(fig, update, frames=len(trajectory), interval=250, blit=False, repeat=True)

# Save (optional)
num_agents = len(trajectory[0]['joint']) if trajectory and 'joint' in trajectory[0] else 0
if saveAnimation:
    ani.save(f"../animation/warehouse_C{true_context}_A{num_agents}.gif", writer=writer)

plt.show()


# ---------------- Reliability Plot (1 / |contexts| over time) ----------------
# Compute reliability for each step based on the size of the current context set.
reliability_values = []
for s in trajectory:
    k = len(s.get('contexts', []))
    k = max(1, k)  # safety guard
    reliability_values.append(1.0 / k)

steps = list(range(len(reliability_values)))

# Create the reliability figure
fig_rel = plt.figure()
ax_rel = fig_rel.add_subplot(111)
ax_rel.plot(steps, reliability_values, linewidth=2)
ax_rel.set_xlabel("Step")
ax_rel.set_ylabel("Confidence of operation (max(belief))")
ax_rel.set_title("Confidence over trajectory")
ax_rel.set_ylim(0, 1.05)
ax_rel.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

# Save alongside the animation if requested
try:
    os.makedirs("../figures", exist_ok=True)
except Exception:
    pass

if saveAnimation:
    try:
        fig_rel.savefig(f"../figures/reliability_C{true_context}_A{num_agents}.png", dpi=200, bbox_inches="tight")
    except Exception:
        pass
# ---------------------------------------------------------------------------


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

# plt.show()

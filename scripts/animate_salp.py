# animate_salp_3d.py
import numpy as np
import json, sys
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # already imported at top

matplotlib.rcParams['axes.facecolor'] = '#0f1216'

# ---------------- CLI ----------------
# Usage: python animate_salp_3d.py TOTAL_CONTEXTS TRUE_CONTEXT MAP_NAME SAVE_ANIM(0/1)
# total_contexts = int(sys.argv[1])
# true_context   = int(sys.argv[2])
map_name       = "salp"
saveAnimation  = True
writer = PillowWriter(fps=40)
SUBSTEPS = 8  # 4–12 works well; higher = smoother but heavier

# ---------------- I/O ----------------
def load_grid(filename):
    grid = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip().lower() == "map":
                break
        for line in f:
            row = list(line.rstrip("\n"))
            if row:
                grid.append(row)
    return grid

with open("../results/MAPF_OUTPUT/trajectory.json", "r") as f:
    data = json.load(f)
trajectory = data["trajectory"]
belief_y = np.array([max(0, len(s.get("contexts", [])) - 1) for s in trajectory], dtype=int)

grid_data = load_grid(f"../maps/{map_name}/{map_name}.map")
H = len(grid_data)
W = len(grid_data[0]) if H > 0 else 0

# ---------------- Water-slab extents (depth enhanced) ----------------
# Thickness scales with map size to give a clear 3D volume.
WATER_Z0 = 0.00
WATER_THICKNESS = max(0.6, 0.08 * min(W, H))     # depth for pool vibe
WATER_Z1 = WATER_Z0 + WATER_THICKNESS
EPS_Z = 1e-3

# Agent altitude and look
AGENT_Z = WATER_Z0 + 0.06
JELLY_COLOR = "purple"#"#5e2ca5"    # dark purple

# ---------------- Helpers ----------------
def format_ctx_set(ctxs):
    return "{" + ", ".join(fr"$c_{{{c}}}$" for c in sorted(set(ctxs))) + "}"


def set_axes_water(ax):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(WATER_Z0 - EPS_Z, WATER_Z1 + EPS_Z)
    ax.set_box_aspect((W, H, WATER_THICKNESS))
    ax.set_proj_type('persp')      # perspective for depth
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    # zoom to reduce empty borders
    try:
        ax.dist = 7.5              # smaller => closer; default ~10
    except Exception:
        pass

def connected_components_at(grid):
    H = len(grid)
    W = len(grid[0]) if H else 0
    vis = [[False]*W for _ in range(H)]
    comps = []
    for y in range(H):
        for x in range(W):
            if grid[y][x] == "@" and not vis[y][x]:
                q = deque([(x, y)])
                vis[y][x] = True
                cells = []
                while q:
                    cx, cy = q.popleft()
                    cells.append((cx, cy))
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < W and 0 <= ny < H and not vis[ny][nx] and grid[ny][nx] == "@":
                            vis[ny][nx] = True
                            q.append((nx, ny))
                comps.append(cells)
    return comps

def component_centroid_and_extent(cells):
    xs = np.array([c[0]+0.5 for c in cells], dtype=float)
    ys = np.array([c[1]+0.5 for c in cells], dtype=float)
    cx, cy = xs.mean(), ys.mean()
    if len(cells) == 1:
        ex, ey = 0.5, 0.5
    else:
        ex = max(0.6, (xs.max() - xs.min())/2.0 + 0.3)
        ey = max(0.6, (ys.max() - ys.min())/2.0 + 0.3)
    return cx, cy, ex, ey

def make_ellipsoid_mesh(cx, cy, cz, ax_len, ay_len, az_len, n_u=28, n_v=18, wobble=0.0):
    u = np.linspace(0, 2*np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)
    rmod = 1.0 + wobble * 0.12*np.sin(5*uu) * np.sin(3*vv)
    X = cx + ax_len * np.cos(uu)*np.sin(vv) * rmod
    Y = cy + ay_len * np.sin(uu)*np.sin(vv) * rmod
    Z = cz + az_len * np.cos(vv)
    return X, Y, Z

def add_upright_star(ax, cx, cy, z_mid,
                     R=0.55, r_ratio=0.45,
                     thickness=0.12,
                     face_color=(1.00, 0.84, 0.00),
                     edge_color=(0.75, 0.60, 0.00),
                     alpha=0.95, n_points=5,
                     orient='x'):
    """
    orient:
      'x' -> face in (y,z), extrude along x  (your current behavior)
      'y' -> face in (x,z), extrude along y  (what you want)
      'z' -> face in (x,y), extrude along z
    """
    m = 2 * n_points
    ang0 = np.pi / 2.0  # tip up
    R_out = R
    R_in  = R * r_ratio
    radii  = np.array([R_out if k % 2 == 0 else R_in for k in range(m)])
    angles = ang0 + np.arange(m) * (np.pi / n_points)

    faces = []
    if orient == 'x':
        ys = cy + radii * np.cos(angles)
        zs = z_mid + radii * np.sin(angles)
        x1, x2 = cx - thickness/2.0, cx + thickness/2.0
        faces.append([(x2, ys[k], zs[k]) for k in range(m)])  # front
        faces.append([(x1, ys[k], zs[k]) for k in range(m)])  # back
        for k in range(m):
            k2 = (k + 1) % m
            faces.append([(x1, ys[k],  zs[k]),
                          (x2, ys[k],  zs[k]),
                          (x2, ys[k2], zs[k2]),
                          (x1, ys[k2], zs[k2])])

    elif orient == 'y':
        xs = cx + radii * np.cos(angles)
        zs = z_mid + radii * np.sin(angles)
        y1, y2 = cy - thickness/2.0, cy + thickness/2.0
        faces.append([(xs[k], y2, zs[k]) for k in range(m)])  # front
        faces.append([(xs[k], y1, zs[k]) for k in range(m)])  # back
        for k in range(m):
            k2 = (k + 1) % m
            faces.append([(xs[k],  y1, zs[k]),
                          (xs[k],  y2, zs[k]),
                          (xs[k2], y2, zs[k2]),
                          (xs[k2], y1, zs[k2])])

    else:  # orient == 'z'
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)
        z1, z2 = z_mid - thickness/2.0, z_mid + thickness/2.0
        faces.append([(xs[k], ys[k], z2) for k in range(m)])  # front
        faces.append([(xs[k], ys[k], z1) for k in range(m)])  # back
        for k in range(m):
            k2 = (k + 1) % m
            faces.append([(xs[k],  ys[k],  z1),
                          (xs[k],  ys[k],  z2),
                          (xs[k2], ys[k2], z2),
                          (xs[k2], ys[k2], z1)])

    star = Poly3DCollection(faces, facecolors=face_color, edgecolors=edge_color,
                            linewidths=0.8, alpha=alpha)
    ax.add_collection3d(star)
    return star

# ---------------- Figure / 3D Axis ----------------
# fig = plt.figure(figsize=(14, 7))
# gs  = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.35], wspace=0.12)
fig = plt.figure(figsize=(14, 7))
gs  = fig.add_gridspec(
    1, 2,
    width_ratios=[3.15, 1.35],   # keeps the left panel dominant
    left=-0.15, right=0.985,      # <<< trims outer whitespace
    bottom=0.06, top=0.96,       # reasonable top/bottom margins
    wspace=-0.25                 # small gap between the two panes
)
ax  = fig.add_subplot(gs[0, 0], projection='3d')   # 3D on the left
# Figure-anchored title positioned above the left 3D axes
bbox = ax.get_position()
ctx_title_text = fig.text(bbox.x0 + bbox.width/2, bbox.y1 + 0.012, "",
                          ha="center", va="bottom", color="w", fontsize=12, fontweight="bold")
set_axes_water(ax)
ax.set_proj_type('persp')
ax.view_init(elev=40, azim=-90)   # screen-aligned edges
# ax.dist = 2.0
ax.set_title("", color='w', pad=6)

ax_belief = fig.add_subplot(gs[0, 1])              # belief on the right
line_belief, = ax_belief.plot([], [], lw=2.5, drawstyle='steps-post', color='gold')
ax_belief.set_xlim(0, len(trajectory) - 1)
ax_belief.set_ylim(-0.01, int(belief_y.max()) + 0.5)
ax_belief.set_xlabel("Time steps")
ax_belief.set_ylabel("Belief distance from oracle")
ax_belief.set_title("Belief Over Time", pad=8)
ax_belief.grid(alpha=0.25)

# ---------------- Static geometry inside the slab ----------------
# Grids for floor and water surfaces
nx, ny = max(40, 2*W), max(40, 2*H)
xg = np.linspace(0, W, nx)
yg = np.linspace(0, H, ny)
Xg, Yg = np.meshgrid(xg, yg)

def water_surface(t=0.0):
    amp = 0.05 * WATER_THICKNESS
    return WATER_Z1 + amp*np.sin(0.6*Xg/W + 0.5*t) * np.cos(0.5*Yg/H - 0.4*t)

# Floor (light sand) at WATER_Z0
Z_floor = np.full_like(Xg, WATER_Z0) + 0.01*WATER_THICKNESS*np.sin(0.6*Xg/W)*np.sin(0.5*Yg/H)
floor_surf = ax.plot_surface(
    Xg, Yg, Z_floor, rstride=1, cstride=1, linewidth=0, antialiased=True,
    shade=True, alpha=0.25, color=(0.78, 0.74, 0.60)
)
floor_surf.set_zsort('min')
floor_surf.set_rasterized(True)

# Pool side-walls to convey volume (thin translucent planes)
def wall(x0, x1, y0, y1):
    Xp = np.array([[x0, x1],[x0, x1]], dtype=float)
    Yp = np.array([[y0, y1],[y0, y1]], dtype=float)
    Zp = np.array([[WATER_Z0, WATER_Z0],[WATER_Z1, WATER_Z1]], dtype=float)
    s = ax.plot_surface(Xp, Yp, Zp, shade=False, alpha=0.05, color=(0.40, 0.60, 0.90), linewidth=0)
    s.set_zsort('min')

wall(0, 0, 0, H)
wall(W, W, 0, H)
wall(0, W, 0, 0)
wall(0, W, H, H)

# Volumetric "water haze" — stack a few faint horizontal layers (static)
HAZE_LAYERS = 6
haze_alpha = 0.02
for k in range(HAZE_LAYERS):
    zk = WATER_Z0 + (k+1)/(HAZE_LAYERS+1) * WATER_THICKNESS
    Zk = np.full_like(Xg, zk)
    haze = ax.plot_surface(
        Xg, Yg, Zk, rstride=1, cstride=1, linewidth=0, antialiased=True,
        shade=False, alpha=haze_alpha, color=(0.38, 0.60, 0.92)
    )
    haze.set_zsort('min')

# Initial dynamic water surface (updates each frame)
water_surf = ax.plot_surface(Xg, Yg, water_surface(0.0),
    rstride=1, cstride=1, linewidth=0, antialiased=True,
    shade=True, alpha=0.05, color=(0.75, 0.88, 0.98))
water_surf.set_zsort('min')
water_surf.set_rasterized(True)
# Zb = WATER_Z0 + 0.65 * WATER_THICKNESS
# Boulders from '@' components (sit on floor, below surface)
for comp in connected_components_at(grid_data):
    cx, cy, ex, ey = component_centroid_and_extent(comp)
    ax_len = 0.70*ex
    ay_len = 0.70*ey
    az_len = min(0.30*max(ex, ey), 0.40*WATER_THICKNESS)
    Xb, Yb, Z_boulder = make_ellipsoid_mesh(
        cx, cy, cz=WATER_Z0 + az_len*0.98,
        ax_len=ax_len, ay_len=ay_len, az_len=az_len,
        wobble=0.35, n_u=30, n_v=20
    )
    rock = ax.plot_surface(Xb, Yb, Z_boulder, rstride=1, cstride=1, linewidth=0,
                        antialiased=True, shade=True, alpha=0.95, color=(0.45, 0.40, 0.35))
    # rock = ax.plot_surface(
    #     Xb, Yb, Zb, rstride=1, cstride=1, linewidth=0, antialiased=True,
    #     shade=True, alpha=0.95, color=(0.45, 0.40, 0.35)
    # )
    rock.set_zsort('min')

# Landmarks 'L' as upright gold stars
STAR_Z = WATER_Z0 + 0.3 * WATER_THICKNESS 
for y in range(H):
    for x in range(W):
        if grid_data[y][x] == 'L':
            add_upright_star(ax, x+0.5, y+0.5, STAR_Z,
                             R=0.55, r_ratio=0.45, thickness=0.12,
                             face_color=(1.00, 0.84, 0.00), edge_color=(0.75, 0.60, 0.00),
                             alpha=0.96, orient='y')

# ---------------- Jellyfish agents ----------------
class Jelly:
    def __init__(self, ax, R=0.42, H=0.32, color=JELLY_COLOR, alpha=0.70,
                 n_u=26, n_v=16, n_tent=6, seed=None):
        self.ax = ax
        self.R = R
        self.H = H
        self.color = color
        self.alpha = alpha
        self.n_u = n_u
        self.n_v = n_v
        self.n_tent = n_tent
        self.rng = np.random.default_rng(seed)
        self.artists = []
        self.phi = self.rng.uniform(0, 2*np.pi, size=n_tent)
        self.sign = self.rng.choice([-1, 1], size=n_tent)

    def clear(self):
        for a in self.artists:
            try: a.remove()
            except Exception: pass
        self.artists = []

    def _bell_mesh(self, cx, cy, cz, t):
        pulse = 1.0 + 0.12*np.sin(2*np.pi*(0.7*t) + 1.3)
        R = self.R * pulse
        H = self.H * (2.0 - pulse)
        u = np.linspace(0, 2*np.pi, self.n_u)
        v = np.linspace(0, 1, self.n_v)
        uu, vv = np.meshgrid(u, v)
        edge_ruffle = 1.0 + 0.12*np.sin(6*uu + 1.2*np.cos(2*np.pi*0.35*t))
        rr = R * (1 - vv**2) * edge_ruffle
        zz = cz + H*(0.15 - 1.15*vv**1.2)
        X = cx + rr * np.cos(uu)
        Y = cy + rr * np.sin(uu)
        Z = zz
        return X, Y, Z

    def _tentacles(self, cx, cy, cz, t):
        curves = []
        base_r = 0.65*self.R
        base_z = cz - 0.02
        T = 28
        for k in range(self.n_tent):
            ang = 2*np.pi * k / self.n_tent
            bx = cx + base_r * np.cos(ang)
            by = cy + base_r * np.sin(ang)
            tt = np.linspace(0, 1, T)
            A = 0.22*self.R
            w = 2.2
            sway = A*np.sin(2*np.pi*(w*t) + self.phi[k]) * (tt**1.1)
            x = bx + sway * np.cos(ang + self.sign[k]*np.pi/2)
            y = by + sway * np.sin(ang + self.sign[k]*np.pi/2)
            z = base_z - 0.9*self.H*tt - 0.15*self.H*np.sin(1.5*np.pi*tt)
            curves.append((x, y, z))
        return curves

    def draw(self, cx, cy, cz, t):
        self.clear()
        X, Y, Z = self._bell_mesh(cx, cy, cz, t)
        surf = self.ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, linewidth=0,
            antialiased=True, shade=True, alpha=self.alpha, color=self.color
        )
        surf.set_zsort('max')  # keep above surfaces
        self.artists.append(surf)
        for (xt, yt, zt) in self._tentacles(cx, cy, cz, t):
            line, = self.ax.plot3D(xt, yt, zt, lw=1.4, alpha=0.95, color=self.color)
            self.artists.append(line)
        return self.artists

num_agents = len(trajectory[0]['joint'])
jellies = [Jelly(ax, seed=i) for i in range(num_agents)]

# ---------------- Animation ----------------
# def update(frame):
#     global water_surf
#     # update dynamic water surface (background)
#     try:
#         if water_surf is not None:
#             water_surf.remove()
#     except (ValueError, RuntimeError):
#         pass
#     zsurf = water_surface(t=0.08*frame)
#     water_surf = ax.plot_surface(
#         Xg, Yg, zsurf,
#         rstride=1, cstride=1, linewidth=0,
#         antialiased=True, shade=True, alpha=0.20, color=(0.40, 0.60, 0.90)
#     )
#     water_surf.set_zsort('min')

#     # agents
#     state = trajectory[frame]
#     ctx_title_text.set_text(f"Belief context set: {format_ctx_set(state.get('contexts', []))}")

#     joint = state["joint"]
#     for i, agent in enumerate(joint):
#         x, y, _ = agent
#         cx, cy = x + 0.5, y + 0.5
#         jellies[i].draw(cx, cy, AGENT_Z, t=0.05*frame)
        
#     # live belief line up to current frame
#     line_belief.set_data(np.arange(frame + 1), belief_y[:frame + 1])
    
#     # set_axes_water(ax)
#     ctx_title = format_ctx_set(state.get("contexts", []))
#     # print(len(state.get('contexts', [])))
#     # print(format_ctx_set(state.get('contexts', [])))
#     if (len(state.get('contexts', [])) > 1):
#         # print title in bold white if multiple contexts
#         ax.set_title(f"Current belief over contexts: {format_ctx_set(state.get('contexts', []))}",
#                      color="w", y=1, pad=2, fontweight='bold')
#     else:
#         # print title in green color if only one context
#         ax.set_title(f"Current belief over contexts: {format_ctx_set(state.get('contexts', []))}",
#                      color="limegreen", y=1, pad=2, fontweight='bold')
#     # (Optional) to include the step too:
#     # ax.set_title(f"t = {frame}   |   Belief context set: {ctx_title}", color="w", pad=6)

#     artists = [water_surf, line_belief]
#     for j in jellies:
#         artists.extend(j.artists)
#     return artists
def update(frame):
    global water_surf

    # ----- continuous time within each discrete step -----
    K = len(trajectory)
    if K < 2:
        return []

    # map animation frame -> (discrete index k, fractional tau in [0,1])
    if frame == (K - 1) * SUBSTEPS:
        k, tau = K - 2, 1.0        # exact last frame
    else:
        k   = frame // SUBSTEPS
        tau = (frame % SUBSTEPS) / SUBSTEPS

    # (optional) easing to soften starts/stops (cubic smoothstep)
    def ease(u): return u*u*(3 - 2*u)
    tau_e = ease(tau)

    state0 = trajectory[k]
    state1 = trajectory[k + 1]

    # ----- dynamic water surface uses continuous time -----
    try:
        if water_surf is not None:
            water_surf.remove()
    except (ValueError, RuntimeError):
        pass
    zsurf = water_surface(t=0.08 * (k + tau_e))
    water_surf = ax.plot_surface(
        Xg, Yg, zsurf, rstride=1, cstride=1, linewidth=0,
        antialiased=True, shade=True, alpha=0.20, color=(0.40, 0.60, 0.90)
    )
    water_surf.set_zsort('min')

    # ----- agents: linear interpolation of positions -----
    joint0 = state0["joint"]
    joint1 = state1["joint"]
    for i in range(len(jellies)):
        x0, y0, _ = joint0[i]
        x1, y1, _ = joint1[i]
        x = x0 + tau_e * (x1 - x0)
        y = y0 + tau_e * (y1 - y0)
        cx, cy = x + 0.5, y + 0.5
        # pass continuous time to jelly shader so tentacles look fluid
        jellies[i].draw(cx, cy, AGENT_Z, t=0.05 * (k + tau_e))

    # ----- titles & belief plot use the discrete index (no flicker) -----
    ctx_now = state0.get('contexts', [])
    ctx_title_text.set_text(f"Belief context set: {format_ctx_set(ctx_now)}")

    line_belief.set_data(np.arange(k + 1), belief_y[:k + 1])

    if len(ctx_now) > 1:
        ax.set_title(f"Current belief over contexts: {format_ctx_set(ctx_now)}",
                     color="w", y=1, pad=2, fontweight='bold')
    else:
        ax.set_title(f"Current belief over contexts: {format_ctx_set(ctx_now)}",
                     color="limegreen", y=1, pad=2, fontweight='bold')

    artists = [water_surf, line_belief]
    for j in jellies:
        artists.extend(j.artists)
    return artists

# ani = FuncAnimation(fig, update, frames=range(0, len(trajectory), 2), interval=25, blit=False, repeat=True)
# ani = FuncAnimation(fig, update, frames=range(len(trajectory)), interval=25, blit=False, repeat=True)
# total frames now include substeps between each discrete step
total_frames = (len(trajectory) - 1) * SUBSTEPS + 1
ani = FuncAnimation(fig, update, frames=range(total_frames),
                    interval=25 / SUBSTEPS,  # keeps overall runtime ~same
                    blit=False, repeat=True)
# ---------------- Save animation ----------------
if saveAnimation:
    out_name = f"../animation/salp3d_{map_name}.gif"
    if len(trajectory) == 0:
        raise RuntimeError("No frames to render: trajectory is empty.")
    ani.save(out_name, writer=writer)
    print(f"Saved: {out_name}")

plt.show()

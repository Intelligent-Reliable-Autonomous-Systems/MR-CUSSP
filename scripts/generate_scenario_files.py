#!/usr/bin/env python3
from math import sqrt
import random
from pathlib import Path

# -------------------- CONFIG (edit these) --------------------
MAP_NAME = "salp"                                               # folder under ./maps/
MAP_PATH = Path(f"./maps/{MAP_NAME}/{MAP_NAME}.map")            # full path to the .map
MAP_NAME_FOR_SCEN = f"{MAP_NAME}.map"                           # how it appears inside .scen lines
SCEN_TEMPLATE = Path(f"./maps/{MAP_NAME}/scenario_files/{MAP_NAME}{{i}}.scen")
SCEN_COUNT = 5                                                  # how many .scen files to make
NUM_INSTANCES = 65                                              # start-goal pairs per file
BASE_SEED = 42                                                  # base RNG seed (files use BASE_SEED+i)
FREE_CHARS = set(".SGLFC")                                      # passable tiles in your domains
MIN_MANHATTAN_DIST = 3                                          # avoid trivial cases
# -------------------------------------------------------------

def parse_map(path: Path):
    """Parse MovingAI-style map and return (W, H, grid_lines)."""
    with path.open("r") as f:
        header = [next(f).rstrip("\n") for _ in range(4)]
        try:
            H = int(header[1].split()[1])  # "height H"
            W = int(header[2].split()[1])  # "width W"
        except Exception as e:
            raise RuntimeError(f"Failed to parse width/height from header: {header}") from e

        grid = []
        for _ in range(H):
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading map grid.")
            line = line.rstrip("\n")
            if len(line) != W:
                raise RuntimeError(f"Map line has length {len(line)} but width is {W}: '{line}'")
            grid.append(line)
    return W, H, grid

def collect_free_cells(W, H, grid):
    free = []
    for y in range(H):
        for x in range(W):
            if grid[y][x] in FREE_CHARS:
                free.append((x, y))
    if not free:
        raise RuntimeError("No free cells found in map for the provided FREE_CHARS set.")
    return free

def octile_distance(ax, ay, bx, by):
    dx, dy = abs(ax - bx), abs(ay - by)
    dmin = min(dx, dy)
    dmax = max(dx, dy)
    return dmin * sqrt(2.0) + (dmax - dmin) * 1.0

def sample_pairs(free_cells, n_pairs, rng: random.Random):
    """
    Sample n_pairs (start, goal) pairs.
    Ensures start != goal and a minimum Manhattan distance.
    Ensures no duplicate (start,goal) pairs within a single scenario file.
    """
    pairs = []
    seen = set()
    attempts = 0
    max_attempts = n_pairs * 1000  # safety
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        sx, sy = rng.choice(free_cells)
        gx, gy = rng.choice(free_cells)
        if (sx, sy) == (gx, gy):
            continue
        if abs(sx - gx) + abs(sy - gy) < MIN_MANHATTAN_DIST:
            continue
        key = (sx, sy, gx, gy)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(((sx, sy), (gx, gy)))
    if len(pairs) < n_pairs:
        raise RuntimeError(f"Could only sample {len(pairs)} pairs (requested {n_pairs}). "
                           f"Try lowering MIN_MANHATTAN_DIST or check FREE_CHARS.")
    return pairs

def write_scen(path: Path, map_name: str, W: int, H: int, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("version 1\n")
        for i, ((sx, sy), (gx, gy)) in enumerate(pairs):
            cost = octile_distance(sx, sy, gx, gy)
            # Your parser expects: y1 x1 y2 x2
            f.write(f"{i}\t{map_name}\t{W}\t{H}\t{sy}\t{sx}\t{gy}\t{gx}\t{cost:.8f}\n")

def main():
    W, H, grid = parse_map(MAP_PATH)
    free_cells = collect_free_cells(W, H, grid)

    for i in range(1, SCEN_COUNT + 1):
        rng = random.Random(BASE_SEED + i)  # independent seed per file
        pairs = sample_pairs(free_cells, NUM_INSTANCES, rng)
        out_path = Path(str(SCEN_TEMPLATE).format(i=i))  # e.g., salp1.scen, salp2.scen, ...
        write_scen(out_path, MAP_NAME_FOR_SCEN, W, H, pairs)
        print(f"[ok] wrote {len(pairs)} instances to {out_path} (map {MAP_NAME_FOR_SCEN}, {W}x{H}, seed={BASE_SEED + i}).")

if __name__ == "__main__":
    main()

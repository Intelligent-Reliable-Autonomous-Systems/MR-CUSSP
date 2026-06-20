import json, sys
import matplotlib.pyplot as plt

# --- data inlined from C++ ---
data = {"OURS":[5,5,5,5,5,3,3,3,1],"ARVI":[5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1]}

out_png = r"../results/plots/belief_distance.png"
solvers = ["OURS", "ARVI"]
if not data:
    print("[plot] No data to plot."); sys.exit(0)

plt.figure(figsize=(8, 4.5), dpi=150)
for solver_idx, ys in data.items():
    xs = list(range(len(ys)))
    if not xs: 
        continue
    plt.plot([int(i) for i in xs], [int(j)-1 for j in ys], linewidth=2, label=solver_idx)

plt.title("Belief Deviation from Oracle")
plt.xlabel("Time steps")
plt.ylabel("Belief Distance")
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc="best")
plt.tight_layout()
plt.savefig(out_png)
print(f"[plot] Saved figure to {out_png}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage[T2A]{fontenc}\usepackage[utf8]{inputenc}\usepackage[russian]{babel}"
    })
    LATEX_ENABLED = True
except RuntimeError:
    LATEX_ENABLED = False

data = pd.read_csv(r'./data/6_wt_data/6_wt_35_3_statistics.csv')

px_to_mm = 0.02
framerate = 25000
base_scale = 10
min_radius = 0.1
threshhold = 0.155

max_radius = data.loc[data.groupby("id")["eq_radius"].idxmax(), ['id', 'frame', 'eq_radius']].reset_index(drop=True)

max_radius['eq_radius'] = max_radius['eq_radius'] * px_to_mm
max_radius['frame'] = max_radius['frame'] / framerate
bubbles = max_radius[max_radius['eq_radius'] > threshhold]

scales = bubbles['eq_radius'] * base_scale ** 2

fig, ax = plt.subplots(figsize=(7, 4))

ax.scatter(bubbles['frame'], bubbles['eq_radius'], s=scales, facecolors='none', edgecolors='blue')
ax.hlines(0.15, 0.13, 0.45, color='green', linestyles='-', label=r'Detection limit', linewidth=0.8)
ax.hlines(0.3, 0.13, 0.45, color='black', linestyles='--', label=r'$R_f$', linewidth=1.8)
ax.hlines(0.6, 0.13, 0.45, color='red', linestyle=':', label=r'$2R_f$', linewidth=1.8)



ax.set_ylim(0.01, 1.8)
ax.set_xlim(0.13, 0.45)

plt.tick_params(axis='both', direction='in', length=5, width=0.5)

ax.set_ylabel(r'Maximum bubble radius, mm', fontsize=14)
ax.set_xlabel(r'Time, s', fontsize=14)
ax.legend(loc='upper right')

plt.grid(True, linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig("./pictures/rad_time_6_3.png", dpi=600, bbox_inches='tight')
#plt.savefig("./pictures/rad_time_6.pdf", dpi=600, bbox_inches='tight')

plt.show()

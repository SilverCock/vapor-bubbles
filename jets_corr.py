import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats

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

DIR_PATH = r'/home/john/Documents/projects/vapor-bubbles/data/jets/'

all_x = []
all_y = []

px_to_mm = 0.02

files = ['2_wt', '4_wt', '5_wt', '6_wt', '8_wt']
markers = ['o', 's', '^', 'D', '*']
labels = ['2 W', '4 W', '5 W', '6 W', '8 W']
colors = []

plt.figure(figsize=(7, 4))

i = 0

for file in glob.glob(os.path.join(DIR_PATH, "*.csv")):
    df = pd.read_csv(file)
    if df.shape[1] >= 3:
        x_data = df.iloc[:, 1] * px_to_mm
        y_data = df.iloc[:, 2] * px_to_mm

        plt.scatter(x_data, y_data, marker=markers[i], label=labels[i])

        all_x.extend(x_data)
        all_y.extend(y_data)
    else:
        print(f"No such file: {file}")
    i += 1

x_array = np.array(all_x)
y_array = np.array(all_y)

slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)
line = slope * x_array + intercept

plt.plot(x_array, line, color='blue', linewidth=1, label=f'Linear fit')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.tick_params(axis='both', direction='in')

plt.xlim(0.61, 1.39)
plt.ylim(0.5, 3)

plt.xlabel(r'Maximum bubble radius, mm', fontsize=14)
plt.ylabel(r'Jet penetration depth, mm', fontsize=14)

plt.tight_layout()

plt.grid(True, linestyle=":", alpha=0.7)

plt.savefig("./pictures/jets_corr.pdf", dpi=600, bbox_inches='tight')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline

# --- Настройка LaTeX ---
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

# --- Папки ---
data_dirs = {
    2: Path("/home/john/Downloads/statistics/35_t/2_wt_stat"),
    4: Path("/home/john/Downloads/statistics/35_t/4_wt_stat"),
    # 5: Path("G:/EXPERIMENT/result_stats/35_t/5_wt_stat"),
    6: Path("/home/john/Downloads/statistics/35_t/6_wt_stat"),
    8: Path("/home/john/Downloads/statistics/35_t/8_wt_stat"),
}

frame_rate = 25000
thresholds = [30, 37.5, 45]

# Стили
styles = {
    30: {"color": "blue",  "linestyle": "-",  "hatch": "//", "label": r"$R_{\max} > 2 R_{f}$"},
    37.5: {"color": "red",   "linestyle": "--", "hatch": "..",  "label": r"$R_{\max} > 2.5 R_{f}$"},
    45: {"color": "green", "linestyle": "-.", "hatch": "--", "label": r"$R_{\max} > 3 R_{f} $"},
}

results = {thr: {"powers": [], "means": [], "stds": []} for thr in thresholds}

for power, data_dir in data_dirs.items():
    csv_files = list(data_dir.glob("*.csv"))
    values_by_thr = {thr: [] for thr in thresholds}

    for file in csv_files:
        data = pd.read_csv(file)
        if not {"id", "frame", "eq_radius"}.issubset(data.columns):
            continue

        max_radius = data.loc[data.groupby("id")["eq_radius"].idxmax(), ['id', 'frame', 'eq_radius']].reset_index(drop=True)
        first_frame, last_frame = data["frame"].min(), data["frame"].max()
        duration_frames = last_frame - first_frame + 1
        if duration_frames <= 0:
            continue
        duration_seconds = duration_frames / frame_rate

        for thr in thresholds:
            large_bubbles = max_radius[max_radius["eq_radius"] >= thr]
            norm_count = len(large_bubbles['id']) / duration_seconds
            values_by_thr[thr].append(norm_count)

    for thr in thresholds:
        vals = values_by_thr[thr]
        if vals:
            results[thr]["powers"].append(power)
            results[thr]["means"].append(np.mean(vals))
            results[thr]["stds"].append(np.std(vals))

# --- Построение ---
plt.figure(figsize=(5.5, 5))

for thr in thresholds:
    powers = np.array(results[thr]["powers"])
    means = np.array(results[thr]["means"])
    stds = np.array(results[thr]["stds"])

    st = styles[thr]

    # Сплайн по среднему
    spline_mean = make_interp_spline(powers, means, k=3)
    xs = np.linspace(powers.min(), powers.max(), 300)
    ys_mean = spline_mean(xs)

    # Сплайн по нижней и верхней границе
    spline_lower = make_interp_spline(powers, means - stds, k=3)
    spline_upper = make_interp_spline(powers, means + stds, k=3)
    ys_lower = spline_lower(xs)
    ys_upper = spline_upper(xs)

    # Линия
    plt.plot(xs, ys_mean, color=st["color"], linestyle=st["linestyle"], label=st["label"])

    # Область доверительного интервала
    plt.fill_between(xs, ys_lower, ys_upper,
                     facecolor=st['color'], alpha=0.2,
                     #hatch=st["hatch"],
                     #edgecolor=st["color"]
                     )

plt.xlim(1.9, 8.1)
plt.ylim(bottom=0)
plt.tick_params(axis='both', direction='in', length=5, width=0.5)

# plt.yscale("log")
plt.xlabel(r'Laser power, W', fontsize=14)
plt.ylabel(r'Number of bubbles per second, s$^{-1}$', fontsize=14)
#plt.title(r'The bubble nucleation rate')


plt.grid(True, linestyle=":", alpha=0.7, which="both")
#plt.grid(False)
plt.legend()

plt.tight_layout()
#plt.savefig("./pictures/large_bubbles.png", bbox_inches="tight")
plt.savefig("./pictures/large_bubbles.pdf", dpi=600, bbox_inches="tight")
plt.show()

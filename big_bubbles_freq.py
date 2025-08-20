import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

# --- Папки с данными ---
data_dirs = {
    2: Path("G:/EXPERIMENT/result_stats/35_t/2_wt_stat"),
    4: Path("G:/EXPERIMENT/result_stats/35_t/4_wt_stat"),
    5: Path("G:/EXPERIMENT/result_stats/35_t/5_wt_stat"),
    6: Path("G:/EXPERIMENT/result_stats/35_t/6_wt_stat"),
    8: Path("G:/EXPERIMENT/result_stats/35_t/8_wt_stat"),
}

frame_rate = 25000  # кадров в секунду
thresholds = [30, 45, 60]  # пороги радиуса

# Словарь для стилей
styles = {
    30: {"color": "blue",  "linestyle": "-",  "marker": "o", "label": r"$R_{\max} > 30$ px"},
    45: {"color": "red",   "linestyle": "--", "marker": "s", "label": r"$R_{\max} > 45$ px"},
    60: {"color": "green", "linestyle": "-.", "marker": "^", "label": r"$R_{\max} > 60$ px"},
}

# --- Подсчёт ---
results = {thr: {"powers": [], "means": [], "stds": []} for thr in thresholds}

for power, data_dir in data_dirs.items():
    csv_files = list(data_dir.glob("*.csv"))
    values_by_thr = {thr: [] for thr in thresholds}

    for file in csv_files:
        df = pd.read_csv(file)
        if not {"id", "frame", "eq_radius"}.issubset(df.columns):
            continue

        # максимальный радиус по каждому пузырю
        max_radius_df = df.groupby("id")["eq_radius"].max().reset_index()

        first_frame = df["frame"].min()
        last_frame = df["frame"].max()
        duration_frames = last_frame - first_frame + 1
        if duration_frames <= 0:
            continue
        duration_seconds = duration_frames / frame_rate

        for thr in thresholds:
            large_bubbles = max_radius_df[max_radius_df["eq_radius"] > thr]
            norm_count = len(large_bubbles) / duration_seconds
            values_by_thr[thr].append(norm_count)

    for thr in thresholds:
        vals = values_by_thr[thr]
        if vals:
            results[thr]["powers"].append(power)
            results[thr]["means"].append(np.mean(vals))
            results[thr]["stds"].append(np.std(vals))

# --- Построение графика ---
plt.figure(figsize=(8, 6))

for thr in thresholds:
    powers = np.array(results[thr]["powers"])
    means = np.array(results[thr]["means"])
    stds = np.array(results[thr]["stds"])

    st = styles[thr]
    #plt.errorbar(
    #    powers, means, yerr=stds,
    #    fmt=st["marker"] + st["linestyle"], color=st["color"],
    #    capsize=5, elinewidth=1.2, ecolor="gray",
    #    label=st["label"]
    #)

# Настройки осей
plt.xlim(0, 10)
plt.ylim(bottom=1e-1)
plt.yscale("log")                  

if LATEX_ENABLED:
    plt.xlabel(r'Laser power, W')
    plt.ylabel(r'Bubble rate per second')
    plt.title(r'Large bubble formation vs. laser power')
else:
    plt.xlabel("Laser power (W)")
    plt.ylabel("Bubbles per second")
    plt.title("Large bubble formation vs laser power")

plt.grid(True, linestyle=":", alpha=0.7)
plt.legend()

plt.tight_layout()

# Сохранение
# plt.savefig("large_bubbles_thresholds.png", dpi=300, bbox_inches="tight")

plt.show()

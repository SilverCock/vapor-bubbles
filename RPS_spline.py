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

# --- Параметры ---
bin_width = 2
bin_edges = np.arange(0, 81, bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

cutoff_x = 7.5
fiber_radius = 15
frame_rate = 25000  # fps

# --- Папки с данными ---
data_dirs = {
    "2 W": Path("G:/EXPERIMENT/result_stats/35_t/2_wt_stat"),
    "5 W": Path("G:/EXPERIMENT/result_stats/35_t/5_wt_stat"),
    "8 W": Path("G:/EXPERIMENT/result_stats/35_t/8_wt_stat"),
}

# --- Стили линий ---
styles = {
    "2 W": {"color": "green", "linestyle": "--"},
    "5 W": {"color": "blue", "linestyle": "-"},
    "8 W": {"color": "brown", "linestyle": "-."},
}

plt.figure(figsize=(10, 6))

for label, data_dir in data_dirs.items():
    csv_files = list(data_dir.glob("*.csv"))
    distributions = []
    avg_rates = []  # список общих скоростей пузырей (в секунду) по каждому файлу

    for file in csv_files:
        df = pd.read_csv(file)
        if not {"id", "frame", "eq_radius"}.issubset(df.columns):
            continue

        max_radius_df = df.groupby("id")["eq_radius"].max().reset_index()
        total_bubbles = df["id"].max()
        if total_bubbles <= 0:
            continue

        # считаем гистограмму
        hist, _ = np.histogram(max_radius_df["eq_radius"], bins=bin_edges)

        # длительность съёмки (в секундах)
        first_frame = df["frame"].min()
        last_frame = df["frame"].max()
        duration_frames = last_frame - first_frame + 1
        if duration_frames <= 0:
            continue
        duration_seconds = duration_frames / frame_rate

        # нормируем гистограмму
        rate = hist / duration_seconds
        distributions.append(rate)

        # считаем общее число пузырей в секунду для этого файла
        avg_rates.append(total_bubbles / duration_seconds)

    if not distributions:
        print(f"No data in {data_dir}")
        continue

    dist_array = np.vstack(distributions)
    mean_dist = dist_array.mean(axis=0)

    # --- Обрезка по cutoff_x ---
    interp_mean = np.interp(cutoff_x, bin_centers, mean_dist)
    mask = bin_centers > cutoff_x
    plot_x = np.concatenate(([cutoff_x], bin_centers[mask]))
    plot_y = np.concatenate(([interp_mean], mean_dist[mask]))

    # --- Сплайн ---
    spline = make_interp_spline(plot_x, plot_y, k=3)
    xs = np.linspace(plot_x.min(), plot_x.max(), 400)
    ys = spline(xs)

    # --- Среднее число пузырей в секунду по всем файлам ---
    overall_avg = np.mean(avg_rates) if avg_rates else 0
    label_with_avg = f"{label} (avg ~ {overall_avg:.1f}/s)"

    st = styles[label]
    plt.plot(xs, ys, color=st["color"], linestyle=st["linestyle"], label=label_with_avg)

# --- Область неопределенности ---
plt.axvspan(0, cutoff_x, color="grey", alpha=0.3)
plt.axvline(x=cutoff_x, color="grey", linestyle="--", linewidth=1.2)
ymin, ymax = plt.ylim()
plt.text(
    cutoff_x / 2, (ymin + ymax) / 2,
    "Unresolved region",
    color="black", ha="center", va="center",
    rotation=90, fontsize=20, weight="bold"
)

# --- Радиус оптоволокна ---
plt.axvline(x=fiber_radius, color="black", linestyle="--", linewidth=1.2)
plt.text(fiber_radius, -0.0075, r"$R_f$", ha="center", va="top", fontsize=12)

# --- Настройки ---
plt.xlim(0, bin_edges[-1])
plt.ylim(bottom=0)

if LATEX_ENABLED:
    plt.xlabel(r'Equivalent maximum bubble radius, pixel')
    plt.ylabel(r'Bubble count per second')
    plt.title(r'Cubic spline-smoothed bubble radius distributions (rate-normalized)')
else:
    plt.xlabel("Эквивалентный максимальный радиус пузыря (пиксели)")
    plt.ylabel("Количество пузырей в секунду")
    plt.title("Bubble radius distribution function (rate-normalized)")

plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("PDF_spline_rate_avg.png", dpi=300, bbox_inches="tight")
plt.show()

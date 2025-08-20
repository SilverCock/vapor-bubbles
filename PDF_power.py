import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Настройка LaTeX для рендеринга текста в графиках ---
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage[T2A]{fontenc}\usepackage[utf8]{inputenc}\usepackage[russian]{babel}"
    })
    LATEX_ENABLED = True
except RuntimeError:
    print("cannot find LaTeX")
    LATEX_ENABLED = False

# --- Параметры графика ---
bin_width = 2
bin_edges = np.arange(0, 81, bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

cutoff_x = 7.5   # область неопределенности
fiber_radius = 15

# --- Папки с данными ---
data_dirs = {
    "2 W": Path("G:/EXPERIMENT/result_stats/35_t/2_wt_stat"),
    "5 W": Path("G:/EXPERIMENT/result_stats/35_t/5_wt_stat"),
    "8 W": Path("G:/EXPERIMENT/result_stats/35_t/8_wt_stat"),
}

# --- Стили отрисовки ---
styles = {
    "2 W": {"color": "green", "linestyle": "--", "hatch": "..."},
    "5 W": {"color": "blue", "linestyle": "-",  "hatch": "//"},
    "8 W": {"color": "brown", "linestyle": "-.", "hatch": "---"},
}

plt.figure(figsize=(10, 6))

for label, data_dir in data_dirs.items():
    csv_files = list(data_dir.glob("*.csv"))
    distributions = []

    for file in csv_files:
        df = pd.read_csv(file)
        if not {"id", "frame", "eq_radius"}.issubset(df.columns):
            continue
        max_radius_df = df.groupby("id")["eq_radius"].max().reset_index()
        total_bubbles = df["id"].max()
        if total_bubbles <= 0:
            continue
        hist, _ = np.histogram(max_radius_df["eq_radius"], bins=bin_edges)
        pdf = hist / (total_bubbles * bin_width)
        distributions.append(pdf)

    if not distributions:
        print(f"No data in {data_dir}")
        continue

    dist_array = np.vstack(distributions)
    mean_dist = dist_array.mean(axis=0)
    std_dist = dist_array.std(axis=0)
    lower_bound = mean_dist - std_dist
    upper_bound = mean_dist + std_dist

    # --- Обрезка данных по cutoff_x ---
    interp_mean = np.interp(cutoff_x, bin_centers, mean_dist)
    interp_lower = np.interp(cutoff_x, bin_centers, lower_bound)
    interp_upper = np.interp(cutoff_x, bin_centers, upper_bound)

    mask = bin_centers > cutoff_x
    plot_x = np.concatenate(([cutoff_x], bin_centers[mask]))
    plot_mean = np.concatenate(([interp_mean], mean_dist[mask]))
    plot_lower = np.concatenate(([interp_lower], lower_bound[mask]))
    plot_upper = np.concatenate(([interp_upper], upper_bound[mask]))

    # --- Рисуем линии и доверительные интервалы ---
    st = styles[label]
    plt.plot(plot_x, plot_mean, color=st["color"], linestyle=st["linestyle"], label=label)
    plt.fill_between(
        plot_x, plot_lower, plot_upper,
        edgecolor=st["color"], hatch=st["hatch"], facecolor="none", alpha=0.3
    )

# --- Область неопределенности ---
plt.axvspan(0, cutoff_x, color="grey", alpha=0.3)
plt.axvline(x=cutoff_x, color="grey", linestyle="--", linewidth=1.2)

# подпись внутри зоны — вертикальная, черная, крупная
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

# --- Настройка отображения ---
plt.xlim(0, bin_edges[-1])
plt.ylim(bottom=0)

if LATEX_ENABLED:
    plt.xlabel(r'Equivalent maximum bubble radius, pixel')
    plt.ylabel(r'Probability density, $1/\mathrm{pixel}$')
    plt.title(r'Bubble radius distribution functions at different laser powers')
else:
    plt.xlabel("Эквивалентный максимальный радиус пузыря (пиксели)")
    plt.ylabel("Плотность вероятности (1/пиксель)")
    plt.title("Bubble radius distribution function")

plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("PDF_exp.png", dpi=300, bbox_inches="tight")
plt.show()

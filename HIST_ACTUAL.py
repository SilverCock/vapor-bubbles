import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- LaTeX ---
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
bin_width_px = 2
bin_edges_px = np.arange(0, 81, bin_width_px)
bin_centers_px = (bin_edges_px[:-1] + bin_edges_px[1:]) / 2

px_to_mm = 0.02

bin_width = bin_width_px * px_to_mm

frame_rate = 25000
cutoff_x = 7.5 * px_to_mm
fiber_radius = 15 * px_to_mm

bin_edges = bin_edges_px * px_to_mm
bin_centers = bin_centers_px * px_to_mm

# --- Папка ---
data_dir = Path("/home/john/Downloads/statistics/35_t/5_wt_stat")
csv_files = list(data_dir.glob("*.csv"))

all_rates = []

for file in csv_files:
    df = pd.read_csv(file)
    if not {"id", "frame", "eq_radius"}.issubset(df.columns):
        continue

    max_radius_df = df.groupby("id")["eq_radius"].max().reset_index()
    max_radius_df["eq_radius"] *= px_to_mm
    hist, _ = np.histogram(max_radius_df["eq_radius"], bins=bin_edges)

    first_frame = df["frame"].min()
    last_frame = df["frame"].max()
    duration_frames = last_frame - first_frame + 1
    if duration_frames <= 0:
        continue
    duration_seconds = duration_frames / frame_rate

    rate = hist / duration_seconds
    all_rates.append(rate)

if not all_rates:
    print("No valid data.")
else:
    dist_array = np.vstack(all_rates)
    mean_rate = dist_array.mean(axis=0)
    std_rate = dist_array.std(axis=0)

    # маска только для x > cutoff_x
    mask = bin_centers > cutoff_x
    plot_x = bin_centers[mask]
    plot_mean = mean_rate[mask]
    plot_std = std_rate[mask]

    plt.figure(figsize=(7, 4))

    # серые бины (среднее)
    plt.bar(plot_x, plot_mean, width=bin_width, color="gray", alpha=0.5,
            label="Mean count rate per bin")

    # прямоугольники ± std
    for x, m, s in zip(plot_x, plot_mean, plot_std):
        plt.bar(x, 2*s, bottom=m-s, width=bin_width, color="blue", alpha=0.2)

    # cutoff
    #plt.axvspan(0, cutoff_x, hatch="//", color="grey", alpha=0.3)
    #plt.axvline(cutoff_x, color="grey", linestyle="--", linewidth=1.2)
     # plt.text(cutoff_x/2, plt.ylim()[1]*0.5, "Unresolved region",
            # color="black", ha="center", va="center", rotation=90, fontsize=20)

    # радиус оптоволокна
    plt.axvline(fiber_radius, color="black", linestyle="--", linewidth=1.0)
    plt.text(fiber_radius, -6, r"$R_f$", ha="center", va="top")

    plt.xlim(cutoff_x+0.01, 1.4)
    plt.ylim(bottom=0)


    plt.xlabel(r'Maximum bubble radius, mm', fontsize=14)
    plt.ylabel(r'Number of bubbles per second, s$^{-1}$', fontsize=14)
    #plt.title(r'Bubble radius histogram at 5 W')
    plt.tick_params(axis='both', direction='in', length=5, width=0.5)

    plt.grid(True
             , linestyle=":", alpha=0.7
             )
    #plt.legend()
    plt.tight_layout()
    #plt.savefig('./pictures/5w_hist.png', dpi=300, bbox_inches='tight')
    plt.savefig('./pictures/5w_hist.pdf', dpi=600, bbox_inches='tight')
    plt.show()

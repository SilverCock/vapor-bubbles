import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Параметры гистограммы
bin_edges = np.linspace(0, 80, 20)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
histograms = []

# Папка с файлами CSV
data_dir = Path("G:/experement/result_stats/5_wT_stat")
csv_files = list(data_dir.glob("*.csv"))

for file in csv_files:
    df = pd.read_csv(file)
    if not {"id", "frame", "eq_radius"}.issubset(df.columns):
        continue

    max_radius_df = df.groupby("id")["eq_radius"].max().reset_index()

    first_frame = df["frame"].min()
    last_frame = df["frame"].max()
    total_frames = last_frame - first_frame
    if total_frames <= 0:
        continue

    scale = 25000 / total_frames

    hist, _ = np.histogram(max_radius_df["eq_radius"], bins=bin_edges)
    hist_scaled = hist * scale
    histograms.append(hist_scaled)

# Построение графика
if not histograms:
    print("[!] No valid histograms found.")
else:
    hist_array = np.vstack(histograms)
    mean_counts = hist_array.mean(axis=0)
    std_counts = hist_array.std(axis=0)

    # Исключаем область радиусов до 10 пикселей
    valid_mask = bin_centers > 0 ####
    bin_centers_valid = bin_centers[valid_mask]
    mean_counts_valid = mean_counts[valid_mask]
    lower = (mean_counts - std_counts)[valid_mask]
    upper = (mean_counts + std_counts)[valid_mask]

    # Построение
    plt.figure(figsize=(10, 6))

    # Доверительная полоса
    plt.fill_between(bin_centers_valid, lower, upper, color='blue', alpha=0.2, label="±1 σ")
    plt.plot(bin_centers_valid, lower, linestyle='-', linewidth=0.5, color='blue', alpha=0.5)
    plt.plot(bin_centers_valid, upper, linestyle='-', linewidth=0.5, color='blue', alpha=0.5)

    # Основная линия
    plt.plot(bin_centers_valid, mean_counts_valid, color='blue', label='Средняя частота')

    # Заштрихованная область неопределённости
    # plt.axvspan(0, 10, facecolor='grey', alpha=0.2, hatch='//', edgecolor='grey', linestyle='--', linewidth=0.8)

    plt.xlabel("Эквивалентный максимальный радиус пузыря (пиксели)")
    plt.ylabel("Частота")
    plt.title("Распределение пузырей по радиусу, 5 ватт")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

import cv2
import numpy as np
import os

# === Папка с кадрами тренировочного датасета ===
train_frames_dir = "H:\\dataset"  # путь к папке с кадрами
output_hist_file = "H:\\dataset\\YOLODataset\\average_histogram.txt"  # куда сохраняем усреднённую гистограмму

# Пустая гистограмма
hist_sum = np.zeros((256,), dtype=np.float64)
count = 0

for filename in os.listdir(train_frames_dir):
    if filename.lower().endswith(('.png')):
        img_path = os.path.join(train_frames_dir, filename)

        # Загружаем в grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Cannot open {filename}")
            continue

        # Гистограмма по яркости
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

        # Нормализуем гистограмму, чтобы сумма = 1
        hist /= hist.sum()

        hist_sum += hist
        count += 1

if count > 0:
    avg_hist = hist_sum / count
    # Сохраняем в текстовый файл
    np.savetxt(output_hist_file, avg_hist, fmt="%.8f")
    print(f"Histogramm saved to {output_hist_file}")
else:
    print("Havent find any frames")


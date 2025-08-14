import cv2
import numpy as np
import os
from pathlib import Path

# === Параметры фильтров ===
median_ksize = 3
gaussian_ksize = (3, 3)
gaussian_sigma = 1.0
sharpen_strength = 1.5

# === Папки ===
input_folder = Path("G:\\experement\\35_t\\videoset")   # исходные изображения
output_folder = Path("G:\\experement\\35_t\\dataset") # куда сохранить обработанные

# Создаём выходную папку, если её нет
output_folder.mkdir(parents=True, exist_ok=True)

# === Обработка всех PNG (или других форматов) ===
for img_path in sorted(input_folder.glob("*.*")):
    # Пропускаем, если это не изображение
    if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        continue

    # Загружаем
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"CAnnot open file: {img_path}")
        continue

    # Медианный фильтр
    median_filtered = cv2.medianBlur(img, median_ksize)

    # Гауссов фильтр
    gaussian_filtered = cv2.GaussianBlur(median_filtered, gaussian_ksize, gaussian_sigma)

    # Усиление контуров
    kernel_sharpening = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32) * sharpen_strength
    sharpened = cv2.filter2D(gaussian_filtered, -1, kernel_sharpening)

    # Сохраняем в выходную папку
    out_path = output_folder / img_path.name
    cv2.imwrite(str(out_path), sharpened)

print("Images saved to:", output_folder)

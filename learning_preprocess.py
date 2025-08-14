import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# === Параметры фильтров ===
median_ksize = 3
gaussian_ksize = (3, 3)
gaussian_sigma = 0
sharpen_strength = 1.1

# === Настройки ===
hist_file = "H:\\dataset\\YOLODataset\\average_histogram.txt"  # файл с эталонной гистограммой
input_frame_path = "G:\\EXPERIMENT\\35_t\\videoset\\videoset_4_000085.png" # кадр, который хотим подогнать

# === 1. Загружаем эталонную гистограмму ===
ref_hist = np.loadtxt(hist_file, dtype=np.float64)
if ref_hist.shape[0] != 256:
    raise ValueError("Histogram must have 256 values")

# === 2. Вычисляем CDF эталонной гистограммы ===
ref_cdf = np.cumsum(ref_hist)
ref_cdf /= ref_cdf[-1]  # нормализация в диапазон [0,1]

# === 3. Загружаем входной кадр ===
img = cv.imread(input_frame_path, cv.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Cannot open {input_frame_path}")

# === 4. Гистограмма и CDF входного кадра ===
src_hist = cv.calcHist([img], [0], None, [256], [0, 256]).flatten()
src_hist /= src_hist.sum()
src_cdf = np.cumsum(src_hist)
src_cdf /= src_cdf[-1]

# === 5. Строим LUT для преобразования ===
lut = np.zeros(256, dtype=np.uint8)
for src_val in range(256):
    diff = np.abs(ref_cdf - src_cdf[src_val])
    lut[src_val] = np.argmin(diff)

# === 6. Применяем LUT к изображению ===
matched_img = cv.LUT(img, lut)

frame_f = cv.medianBlur(matched_img, median_ksize)
frame_f = cv.GaussianBlur(frame_f, gaussian_ksize, gaussian_sigma)
kernel_sharpening = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]], dtype=np.float32) * sharpen_strength
frame_f = cv.filter2D(frame_f, -1, kernel_sharpening)

plt.imshow(frame_f, cmap='gray')
plt.show()
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# === Параметры фильтров ===
median_ksize = 3
gaussian_ksize = (3, 3)
gaussian_sigma = 0
sharpen_strength = 1.1

# === Параметры LUT ===
gamma = 2.0
contrast = 0.35

# === LUT-функция (PFV4-подобная) ===
def build_lut_pfv(gamma_val: float, contrast_val: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    y = np.power(x, gamma_val)
    y = 0.5 + (y - 0.5) * (1.0 + contrast_val)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)

lut = build_lut_pfv(gamma, contrast)

# === Путь к видео и модели ===
video_path = Path("input_video.avi")
model_dir = Path("path/to/model")  # здесь папка с best.pt

# === Загрузка модели ===
model = YOLO(model_dir / "best.pt")

# === Открываем видео ===
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open file {video_path}")

# Параметры исходного видео
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Инициализация сохранения результата ===
output_path = video_path.with_name(video_path.stem + "_tracked.avi")
fourcc = cv2.VideoWriter_fourcc(*"avi")
out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

# === Обработка кадров ===
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Приводим к 8-битам, если нужно ---
    if frame.dtype == np.uint16:
        frame = cv2.convertScaleAbs(frame, alpha=255.0/65535.0)

    # --- Фильтры ---
    frame_f = cv2.medianBlur(frame, median_ksize)
    frame_f = cv2.GaussianBlur(frame_f, gaussian_ksize, gaussian_sigma)
    kernel_sharpening = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]], dtype=np.float32) * sharpen_strength
    frame_f = cv2.filter2D(frame_f, -1, kernel_sharpening)

    # --- LUT ---
    frame_f = cv2.LUT(frame_f, lut)

    # --- YOLO tracking ---
    results = model.track(frame_f, tracker="botsort.yaml", persist=True, verbose=False)

    if results and len(results) > 0:
        res = results[0]
        # Рисуем сегментации, bbox и ID
        if res.masks is not None:
            for seg, box, tid in zip(res.masks.xy, res.boxes.xyxy, res.boxes.id):
                # Маска
                mask_img = np.zeros_like(frame_f, dtype=np.uint8)
                pts = np.int32([seg])
                cv2.fillPoly(mask_img, pts, (0, 255, 0))
                frame_f = cv2.addWeighted(frame_f, 1, mask_img, 0.4, 0)

                # BBox
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_f, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ID
                if tid is not None:
                    cv2.putText(frame_f, f"ID {int(tid)}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Показываем и записываем
    cv2.imshow("Tracking", frame_f)
    out_writer.write(frame_f)

    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
        break

cap.release()
out_writer.release()
cv2.destroyAllWindows()

print(f"Video saved to: {output_path}")

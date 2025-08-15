import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# === Параметры фильтров ===
median_ksize = 3
gaussian_ksize = (3, 3)
gaussian_sigma = 0
sharpen_strength = 1.1

# === Пути ===
video_dir = Path(r"D:\\videos")        # Папка с входными AVI
output_dir = Path(r"D:\\output_stats") # Папка для сохранения CSV
model_path = Path(r"D:\\best.pt")      # Модель YOLO

output_dir.mkdir(parents=True, exist_ok=True)

# === Загрузка модели ===
model = YOLO(model_path)

# === Обработка всех видео ===
for video_path in video_dir.glob("*.avi"):
    print(f"Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        continue

    # Постобработка ID
    last_seen_frame = {}
    temp_ids = {}   # временные ID {orig_id: (start_frame, tentative_id)}
    id_map = {}     # постоянные ID {orig_id: new_id}
    next_new_id = 1

    frame_idx = 0
    data_records = []  # список строк для CSV

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Фильтры ---
        frame_f = cv2.medianBlur(frame, median_ksize)
        frame_f = cv2.GaussianBlur(frame_f, gaussian_ksize, gaussian_sigma)
        kernel_sharpening = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]], dtype=np.float32) * sharpen_strength
        frame_f = cv2.filter2D(frame_f, -1, kernel_sharpening)

        # --- YOLO tracking ---
        results = model.track(frame_f, tracker="botsort.yaml", persist=True, verbose=False)

        if results and len(results) > 0:
            res = results[0]
            if res.masks is not None and res.boxes.id is not None:
                for seg, tid in zip(res.masks.xy, res.boxes.id):
                    if tid is None:
                        continue

                    orig_id = int(tid)

                    # === No-resurrection + фильтр 2 кадра ===
                    if (orig_id not in last_seen_frame) or (frame_idx - last_seen_frame[orig_id] > 1):
                        temp_ids[orig_id] = (frame_idx, next_new_id)
                    else:
                        if orig_id in temp_ids:
                            start_frame, tentative_id = temp_ids[orig_id]
                            if frame_idx - start_frame == 1:
                                id_map[orig_id] = tentative_id
                                next_new_id += 1
                                temp_ids.pop(orig_id, None)  # перенос в постоянные
                            elif frame_idx - start_frame > 1 and orig_id not in id_map:
                                temp_ids.pop(orig_id, None)  # выкидываем

                    last_seen_frame[orig_id] = frame_idx

                    # Только постоянные ID идут в CSV
                    if orig_id not in id_map:
                        continue

                    new_id = id_map[orig_id]

                    # === Площадь маски и эквивалентный радиус ===
                    mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                    pts = np.int32([seg])
                    cv2.fillPoly(mask_img, pts, 255)

                    area = cv2.countNonZero(mask_img)  # пиксели исходного кадра
                    eq_radius = np.sqrt(area / np.pi)  # эквивалентный радиус

                    data_records.append((new_id, frame_idx, eq_radius))

        frame_idx += 1

    cap.release()

    # === Сохраняем CSV ===
    df = pd.DataFrame(data_records, columns=["id", "frame", "eq_radius"])
    csv_path = output_dir / f"{video_path.stem}_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

print("Processing complete.")

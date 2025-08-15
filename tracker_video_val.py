import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# === Параметры фильтров ===
median_ksize = 3
gaussian_ksize = (3, 3)
gaussian_sigma = 0
sharpen_strength = 1.1

# === Путь к видео и модели ===
video_path = Path("G:\\EXPERIMENT\\filtered\\5_wt\\5_wt_35_1.avi")
model_dir = Path("C:\\Users\\knja3\\runs\\segment\\train9\\weights\\best.pt")  # здесь папка с best.pt

# === Загрузка модели ===
model = YOLO(model_dir)

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
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out_writer = cv2.VideoWriter(str(output_path), fourcc, 10, (width, height))

# === Словари для постобработки ID ===
last_seen_frame = {}       # {orig_id: frame_idx}
temp_ids = {}              # {orig_id: (frame_idx, tentative_id)}
id_map = {}                # {orig_id: new_id}
next_new_id = 1            # наш следующий ID

# === Обработка кадров ===
frame_idx = 0
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
        # Рисуем сегментации, bbox и ID
        if res.masks is not None and res.boxes.id is not None:
            for seg, box, tid in zip(res.masks.xy, res.boxes.xyxy, res.boxes.id):
                if tid is None:
                    continue

                orig_id = int(tid)
                
                # --- No-resurrection ---
                if (orig_id not in last_seen_frame) or (frame_idx - last_seen_frame[orig_id] > 1):
                    # Новый объект — сначала помещаем в temp_ids
                    temp_ids[orig_id] = (frame_idx, next_new_id)

                else:
                    # Объект продолжается
                    if orig_id in temp_ids:
                        start_frame, tentative_id = temp_ids[orig_id]
                        if frame_idx - start_frame == 1:
                            # Живёт минимум 2 кадра — делаем постоянным
                            id_map[orig_id] = tentative_id
                            next_new_id += 1
                        elif frame_idx - start_frame > 1 and orig_id not in id_map:
                            # Пропал до второго кадра — забываем
                            temp_ids.pop(orig_id, None)

                last_seen_frame[orig_id] = frame_idx

                # Отображаем только объекты с постоянным ID
                if orig_id not in id_map:
                    continue

                new_id = id_map[orig_id]


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
                    cv2.putText(frame_f, f"ID {new_id} (orig {orig_id})", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_writer.write(frame_f)

    frame_idx += 1

cap.release()
out_writer.release()

print(f"Video saved to: {output_path}")

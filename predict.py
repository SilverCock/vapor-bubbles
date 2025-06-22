import os
import cv2
import numpy as np
from ultralytics import YOLO

# Параметры
model_path = 'C:\\Users\\knja3\\runs\\segment\\train6\\weights\\best.pt'  # путь к модели
source_dir = 'G:\\experement\\filtered\\frames'                             # папка с изображениями
output_dir = 'C:\\Users\\knja3\\runs\\valraw'                            # папка для сохранения результатов

os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)
results = model.predict(source=source_dir, save=False, imgsz=200, device=0, verbose=False, stream=True)

for r in results:
    im_path = r.path
    im = cv2.imread(im_path)
    im_h, im_w = im.shape[:2]

    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        for i, mask in enumerate(masks):
            # Маска -> бинарная -> масштабируем до размера изображения
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_uint8, (im_w, im_h))

            # Цветная маска: синий
            mask_color = np.zeros_like(im)
            mask_color[:, :, 0] = mask_resized  # BGR: [blue, green, red]

            # Альфа-наложение
            im = cv2.addWeighted(im, 1.0, mask_color, 0.4, 0)

            # Нарисовать bbox
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Сохраняем изображение
    filename = os.path.basename(im_path)
    cv2.imwrite(os.path.join(output_dir, filename), im)

print(f"Визуализация завершена. Файлы сохранены в: {output_dir}")

#Это что из себя выдавил GPT, надо разобраться с его работой, может что-то подправить
import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
from glob import glob
import logging

# === ПАРАМЕТРЫ ===
FRAME_DIR = 'frames'
OUTPUT_DIR = 'output'
MIN_AREA = 10
BUBBLE_ECC_MAX = 0.7
JET_ECC_MIN = 0.85
D_MAX = 15

# === НАСТРОЙКА ЛОГОВ ===
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_frames(frame_dir):
    frames = sorted(glob(os.path.join(frame_dir, "*.jpg")))
    if not frames:
        logging.warning(f"Не найдены кадры в {frame_dir}")
    else:
        logging.info(f"Загружено {len(frames)} кадров.")
    return frames

def preprocess_frame(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.warning(f"Кадр не загружен: {path}")
        return np.zeros((1, 1), dtype=np.uint8)

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Если есть маска: "mask.png" в том же каталоге
    mask_path = os.path.join(FRAME_DIR, 'mask.png')
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            binary[mask > 0] = 0

    return binary

def extract_objects(binary):
    regions = [r for r in regionprops(label(binary)) if r.area > MIN_AREA]
    detections = []

    for r in regions:
        ecc = r.eccentricity
        if np.isnan(ecc): continue  # защита от нечисел

        cy, cx = r.centroid
        d_eq = r.equivalent_diameter

        if ecc < BUBBLE_ECC_MAX:
            obj_class = 'bubble'
        elif ecc > JET_ECC_MIN:
            obj_class = 'jet'
        else:
            continue

        detections.append({
            'x': cx, 'y': cy,
            'radius': d_eq / 2,
            'class': obj_class,
            'matched': False
        })

    return detections

def match_objects(active_objects, detections, frame_idx, tracks, objects_summary, next_id):
    updated_objects = []

    for obj in active_objects:
        found = False
        candidates = [d for d in detections if d['class'] == obj['class'] and not d['matched']]

        if candidates:
            tree = cKDTree([(d['x'], d['y']) for d in candidates])
            dist, idx = tree.query([obj['x'], obj['y']], distance_upper_bound=D_MAX)

            if dist < D_MAX and idx < len(candidates):
                d = candidates[idx]
                d['matched'] = True
                d['id'] = obj['id']
                d['start'] = obj['start']
                updated_objects.append(d)

                if obj['class'] == 'bubble':
                    tracks.append([d['id'], frame_idx, d['x'], d['y'], d['radius']])

                found = True

        if not found:
            objects_summary.append([
                obj['id'], obj['class'], obj['start'],
                frame_idx - 1, frame_idx - obj['start']
            ])

    for d in detections:
        if not d['matched']:
            d['id'] = next_id
            d['start'] = frame_idx
            updated_objects.append(d)

            if d['class'] == 'bubble':
                tracks.append([d['id'], frame_idx, d['x'], d['y'], d['radius']])

            next_id += 1

    return updated_objects, next_id

def finalize_remaining_objects(active_objects, last_frame_idx, objects_summary):
    for obj in active_objects:
        objects_summary.append([
            obj['id'], obj['class'], obj['start'],
            last_frame_idx, last_frame_idx - obj['start'] + 1
        ])

def save_results(objects_summary, tracks, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df_summary = pd.DataFrame(objects_summary, columns=[
        'id', 'class', 'start_frame', 'end_frame', 'lifetime'
    ]).sort_values(by=['id'])

    df_tracks = pd.DataFrame(tracks, columns=[
        'id', 'frame', 'x', 'y', 'radius'
    ]).sort_values(by=['id', 'frame'])

    df_summary.to_csv(os.path.join(output_dir, 'objects_summary.csv'), index=False)
    df_tracks.to_csv(os.path.join(output_dir, 'bubble_tracks.csv'), index=False)

    logging.info(f"Сохранены таблицы: {output_dir}/objects_summary.csv и bubble_tracks.csv")

def main():
    frame_paths = load_frames(FRAME_DIR)
    active_objects = []
    objects_summary = []
    tracks = []
    next_id = 0
    last_frame_idx = -1

    for frame_idx, path in enumerate(frame_paths):
        binary = preprocess_frame(path)
        detections = extract_objects(binary)
        active_objects, next_id = match_objects(
            active_objects, detections, frame_idx,
            tracks, objects_summary, next_id
        )
        last_frame_idx = frame_idx

        if frame_idx % 100 == 0:
            logging.info(f"Обработано кадров: {frame_idx + 1}")

    finalize_remaining_objects(active_objects, last_frame_idx, objects_summary)
    save_results(objects_summary, tracks, OUTPUT_DIR)

    logging.info("Обработка завершена.")

if __name__ == "__main__":
    main()

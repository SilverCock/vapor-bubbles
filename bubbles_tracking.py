import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Parameters
MODEL_DIR = "best.pt"
VIDEO_DIR = "2_wT_1.avi" 
OUTPUT_DIR = "2_wT_1_stat.csv"

model = YOLO(MODEL_DIR)

#create output file
try:
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    print(f"Output directory verified: {os.path.dirname(OUTPUT_DIR)}")
except Exception as e:
    print(f"Failed to verify output directory: {e}")

# load video
cap = cv2.VideoCapture(VIDEO_DIR)
if not cap.isOpened():
    print(f"Error opening video file: {VIDEO_DIR}")
    exit()

# get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video loaded successfully. Total frames: {total_frames}")


frame_idx = 0
records = []

if not cap.isOpened():
    print("Error opening video file")
    exit()
    
#main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    try:
        results = model.track(source=frame, persist=True, imgsz=224, device=0, verbose=False, tracker="botsort.yaml", show=True)

        if results is not None and len(results) > 0:
            r = results[0]
            
            if r.boxes is not None and r.boxes.id is not None:
 
                #working with bbox -- get ID, class, center coordinates
                for i in range(len(r.boxes.id)):
                    obj_id = int(r.boxes.id[i])
                    cls = int(r.boxes.cls[i])
                    x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                #working with mask -- get area an radius
                    if r.masks is not None and r.masks.data is not None:
                        mask = r.masks.data[i].cpu().numpy()
                        area = np.sum(mask)
                        radius = np.sqrt(area / np.pi)
                    else:
                        area, radius = None, None

                #udpating records array 
                    records.append({"id": obj_id, "frame": frame_idx, "cx": cx, "cy": cy, "area": area, "eq_radius": radius})

    except Exception as e:
        print(f"Error on frame {frame_idx}: {e}")
        exit()

    #progress log
    if total_frames > 0 and frame_idx % max(1, total_frames // 10) == 0:
        percent = int(100 * frame_idx / total_frames)
        print(f"{percent}% of frames processed")

    frame_idx += 1

cap.release()

if records:
    try:
        df = pd.DataFrame(records)
        df.to_csv(OUTPUT_DIR, index=False)
        print(f"Tracking results saved successfully to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")
        exit()
else:
    print("No tracking data collected — CSV file was not created")
    exit()
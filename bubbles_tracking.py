import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Parameters
MODEL_DIR = "\\path\\to\\seg\\model.pt"
VIDEO_DIR = "\\path\\to\\source\\video.avi" #must load 200x200 video of fiber, not raw 1024x200 video
OUTPUT_DIR = "\\path\\to\\output\\file.csv"

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

# Preprocess
def preprocess(frame):
    #someone did not commit his preprocess function
    return frame
    
#main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    try:
        filtered_frame = preprocess(frame)

        results = model.track(source=filtered_frame, persist=True, imgsz=200, device=0, stream=True, verbose=False)
    
        if results and results[0].boxes is not None:
            r = results[0]

            for i in range(len(r.boxes)):

                #working with bbox -- get ID, class, center coordinates
                if r.boxes and r.boxes.data is not None:
                    obj_id = int(r.boxes.id[i])
                    cls = int(r.boxes.cls[i])
                    x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                else:
                    obj_id, cls, cx, cy = None, None, None, None

                #working with mask -- get area an radius
                if r.masks and r.masks.data is not None:
                    mask = r.masks.data[i].cpu().numpy()
                    area = np.sum(mask)
                    radius = np.sqrt(area / np.pi)
                else:
                    area, radius = None, None

                #udpating records array 
                records.append({"id": obj_id, "frame": frame_idx, "cx": cx, "cy": cy, "area": area, "eq_radius": radius})

    except Exception as e:
        print(f"Error on frame {frame_idx}: {e}")

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
else:
    print("No tracking data collected — CSV file was not created")
import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Parameters
MODEL_DIR = "\\path\\to\\seg\\model.yaml"
VIDEO_DIR = "\\path\\to\\source\\video.avi" #must load 200x200 video of fiber, not raw 1024x200 video
OUTPUT_DIR = "\\path\\to\\output\\file.csv"

model = YOLO(MODEL_DIR)

#create output file
os.mkdir(OUTPUT_DIR)

# load video
cap = cv2.VideoCapture(VIDEO_DIR)
frame_idx = 0
records = []

if not cap.isOpened():
    print("Error opening video file")

# Preprocess
def preprocess(frame):
    #someone did not commit his preprocess function
    return filtered_frame
    
#main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
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

    frame_idx += 1

cap.release()

df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR)
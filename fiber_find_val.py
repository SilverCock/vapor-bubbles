import cv2 as cv
import numpy as np
from ultralytics import YOLO

MODEL_DIR = "C:\\Users\\knja3\\runs\\detect\\train4\\weights\\best.pt" #fiber detecrion model
VIDEO_DIR = "G:\\experement\\45_t\\4_wt_45_25000\\4_wt_45_25000_04_08_2025_1_C001H001S0001.avi" #video direction

model_fiber = YOLO(MODEL_DIR)

cap = cv.VideoCapture(VIDEO_DIR)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, first_frame = cap.read()

if not ret:
    print("Cannot read first frame")
    cap.release()
    exit()

result_first_frame = model_fiber.predict(source=first_frame, show=True, device=0, verbose=False, conf=0.7)

x1_fiber = None

if result_first_frame and result_first_frame[0].boxes:
    for box in result_first_frame[0].boxes.xyxy:
        x1, y1, x2, y2 = box.cpu().numpy()
        x1_fiber = int(x1)

cap.set(cv.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    if x1_fiber is not None:
        crop_start = x1_fiber - 200
        crop_end = x1_fiber + 100
        croped_frame = frame[:, crop_start:crop_end]
    else:
        print("Cropping went wrong")
        break

    cv.imshow('frame', croped_frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
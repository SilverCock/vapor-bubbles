# Надо переписать эту хуйню ей-богу

Automatic detection and tracking of laser-induced vapor bubbles and submerged jets in high-speed video capture (up to 100 000 fps).
Process frames, segment objects, track them over time, and save the results in a CSV file.

---

## Сapabilities 

- Object detection/classification on each frame (using pre-trained YOLO-seg model):
  - **Bubbles**
  - **Submerged jets**
  - **Fiber** (ignored)
- Tracking objects over frames
- Collecting statistics (lifetime, coordinates, radius)
- Saving results in a CSV file
- Creating graphs
- (Optional) visualizing tracks on frames

---

## Data example

- Frames in '.jpg' format
- Resolution '1280 x 120'
- Frame rate up to 100 000 fps

---

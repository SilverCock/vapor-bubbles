import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Parameters
MODEL_DIR = "\\path\\to\\seg\\model"
VIDEO_DIR = "\\path\\to\\source\\video"

model = YOLO(MODEL_DIR)

#create output csv file

# Loading video
def load_video(VIDEO_DIR):
    print('Loading video')
    # Only load video from path

# Preprocess
def preprocess():
    print('delet me')
    
#while true read video frame by frame

    #preprocess frame
    #model.track

    #working with mask
        #obj_id
        #center
        #area
        #eq_radius
         
    #record to csv file

    #frame_idx+=1

    #maybe some logging

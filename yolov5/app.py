import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/runs/train/exp2/weights/best.pt', force_reload=True)

img = 'image_name.png'

results = model(img)
results.print()
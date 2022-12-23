from time import time
import tkinter as tk
from tkinter import font
import os
from tkinter import filedialog
import torch
import numpy as np
import cv2
import deskew as ds
from subprocess import call
from pathlib import Path


root = tk.Tk()

root.title("AutoVIS")
root.geometry('500x150')
root.maxsize(500, 150)
root.minsize(500, 150)

blue = "#346beb"

robo = font.Font(family="Roboto", size=14, weight="bold")

model = torch.hub.load('ultralytics/yolov5', 'custom', './weights/best.pt')
classes = model.names


def score_frame(frame):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


def class_to_label(x):
    return classes[int(x)]


def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] *
                                                                               x_shape), int(row[3] * y_shape)
            bgr = (0, 255, 0)

            #print("X1: ", x1, "X2: ", x2, "Y1: ", y1, "Y2: ", y2)

            # Crop image
            if class_to_label(labels[i]) == "car":
                if x1 > 410 and x1 < 420 and x2 > 730 and x2 < 750:
                    vehiclecrop = frame[y1:y2, x1:x2]
                    cv2.imwrite('frame.png', vehiclecrop)
                    call(["python", "detect-lp.py"])

            elif class_to_label(labels[i]) == "van":
                if x1 > 300 and x1 < 310 and x2 > 570 and x2 < 580:
                    vehiclecrop = frame[y1:y2, x1:x2]
                    cv2.imwrite('frame.png', vehiclecrop)
                    call(["python", "detect-lp.py"])

            ##############################
            ## Define for other vehicles##

            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame


# Browse for video file
def browseVideo():
    # Video frame
    video_path = filedialog.askopenfilename(initialdir=os.path.normpath(
        "C://"), title="Browse video", filetypes=(("MP4", "*.mp4"), ("AVI", "*.avi"), ("All Files", "*.*")))
    if (video_path):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()

        while True:

            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (920, 600))

            start_time = time()
            results = score_frame(frame)
            frame = plot_boxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            #print(f"Frames Per Second : {fps}")

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Detection', frame)
            keyCode = cv2.waitKey(1)
            if cv2.getWindowProperty('YOLOv5 Detection', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()


# Open reports file explorer
def openReports():
    downloads_path = str(Path.home() / "Downloads")
    explorer_path = (downloads_path + "\\AutoVIS Reports")
    os.startfile(explorer_path)


# Header title
titleLabel = tk.Label(
    text="Auto Vehicle Identification System",
    foreground="white",
    background=blue,
    font=robo,
    width=500,
    height=2
)
titleLabel.pack()


# Browse video button
browseBtn = tk.Button(
    text="Browse video",
    width=15,
    height=2,
    bg=blue,
    fg="white",
    command=browseVideo
)
browseBtn.pack()
browseBtn.place(x=100, y=70)


# Reports button
reportsBtn = tk.Button(
    text="Open Reports",
    width=15,
    height=2,
    bg=blue,
    fg="white",
    command=openReports
)
reportsBtn.pack()
reportsBtn.place(x=240, y=70)

root.mainloop()

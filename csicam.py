import cv2
import numpy as np
import time
from elements.detect import OBJ_DETECTION

Object_classes = [ 'Bus', 'Car', 'Motorbike', 'Van', 'Truck', 'Threewheel' ]

Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('weights/best.pt', Object_classes)

def gstreamer_pipeline(
    capture_width=1366,
    capture_height=768,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


# flip video
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow("AutoVIS", cv2.WINDOW_AUTOSIZE)
    font=cv2.FONT_HERSHEY_SIMPLEX
    # FPS init
    timeMark=time.time()
    fpsFilter=0
    # Window
    while cv2.getWindowProperty("AutoVIS", 0) >= 0:
        ret, frame = cap.read()
        # calculate fps
        item=''
        dt=time.time()-timeMark
        fps=1/dt
        fpsFilter=.95*fpsFilter +.05*fps
        timeMark=time.time()
        if ret:
            # detection process
            objs = Object_detector.detect(frame)

            # show bounding boxes
            for obj in objs:
                # print(obj)
                label = obj['label']
                score = obj['score']
                [(xmin,ymin),(xmax,ymax)] = obj['bbox']
                color = Object_colors[Object_classes.index(label)]
                frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
                frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)

        # show FPS        
        frame = cv2.putText(frame,str(round(fpsFilter,1))+' fps '+item,(0,30),font,1,(0,0,255),2)
        cv2.imshow("AutoVIS", frame)
        keyCode = cv2.waitKey(30)
        if keyCode == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")

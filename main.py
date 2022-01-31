#import jetson.inference
#import jetson.utils
import cv2
import numpy as np
import time
width=1200
height=1024
dispW=width
dispH=height
flip=2
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance  contrast=1.5 brightness=-.3 saturation=1.2 ! appsink  '
cam=cv2.VideoCapture(camSet)
#cam1=cv2.VideoCapture('/dev/video1')
#cam1.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
#cam1.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
#net=jetson.inference.imageNet('alexnet')
font=cv2.FONT_HERSHEY_SIMPLEX
timeMark=time.time()
fpsFilter=0
while True:
    _,frame=cam.read()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    #img=jetson.utils.cudaFromNumpy(img)
    #classID, confidence =net.Classify(img, width, height)
    item=''
    #item =net.GetClassDesc(classID)
    dt=time.time()-timeMark
    fps=1/dt
    fpsFilter=.95*fpsFilter +.05*fps
    timeMark=time.time()
    cv2.putText(frame,str(round(fpsFilter,1))+' fps '+item,(0,30),font,1,(0,0,255),2)
    cv2.imshow('recCam',frame)
    cv2.moveWindow('recCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
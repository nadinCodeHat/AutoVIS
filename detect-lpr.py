import cv2

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('./samples/video.mp4')
#cap = cv2.resize(cap, (frameWidth, frameHeight))
# Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=80)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

already_saved = False
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #height, width, _ = frame.shape
    #print(height, width)

    if ret == True:

        # Extract region of interest
        roi = frame[400: 700, 500: 1400]

        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 100:
                # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 1)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)

                detections.append([x, y, w, h])
                if detections and already_saved == False:
                    already_saved = True
                    #cv2.imwrite("frame.png", frame)
                    cv2.imwrite('C:\\Users\\LENOVO\\Documents\\Projects\\AutoVIS\\croppedimages\\' +
                                'frame.png', roi)
                    print("Hello")
                else:
                    print("Detections are empty or frame already saved")

        print(detections)
        # Display the resulting frame
        # cv2.resize(frame, (frameWidth, frameHeight)))
        cv2.imshow('Video Frame', frame)
        cv2.imshow('roi', roi)
        # cv2.resize(mask, (frameWidth, frameHeight)))
        cv2.imshow('Mask', mask)
    else:
        print('no video')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    # Press Q on keyboard to  exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

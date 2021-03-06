import cv2
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",	help="OpenCV object tracker type")
args = vars(ap.parse_args())

OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"mil": cv2.TrackerMIL_create}

	

cap = cv2.VideoCapture(1)
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
tracker = cv2.TrackerKCF_create()
initBB = None




while True:
    ret, frame = cap.read()
    if ret == True:
        H, W = frame.shape[:2]
        if initBB is not None:
            (success, box) = tracker.update(frame)

            if success:
                x, y, w, h = [int (v) for v in box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255, 0), 2)

        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            tracker.init(frame, initBB)
        elif key == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()



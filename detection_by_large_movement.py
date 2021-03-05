import cv2
import numpy as np

# for my computer, the webcam is 1;
# the number could be 0 or 2 for others
cap = cv2.VideoCapture(1)

array_template = np.zeros((480, 640, 2))
kernel = np.ones((3,3),np.uint8)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # remove salt and peper noise
    gray = cv2.medianBlur(gray,5)
    gray = np.expand_dims(gray,axis=2)
    array_template = np.concatenate((array_template, gray), axis=2)
    array_template = np.delete(array_template, 0, axis=2)
    first_frame, last_frame = array_template[:,:,0], array_template[:,:,1]
    diff = abs(first_frame - last_frame).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (5,5), 0)
    # do a thresholding to the difference between two frames
    _,thresh_on_diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    # opening on the thresholding result
    dilated = cv2.dilate(thresh_on_diff, kernel, iterations=5)
    # opening = cv2.morphologyEx(thresh_on_diff, cv2.MORPH_OPEN, kernel, iterations=5)
    # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow('frame', frame)
    
    cv2.imshow('dilated', dilated)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
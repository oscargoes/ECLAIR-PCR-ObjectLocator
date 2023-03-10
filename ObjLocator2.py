import cv2 as cv
import numpy as np
import imutils

# use "1" if no built in webcam
videoCapture = cv.VideoCapture(0, cv.CAP_DSHOW)

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    gray = cv.GaussianBlur(gray, (11, 11), 0)

    mask = cv.inRange(gray, greenLower, greenUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        contour = max(cnts, key=cv.contourArea)
        ((x, y), r) = cv.minEnclosingCircle(contour)
        M = cv.moments(contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if r > 10:
            cv.circle(frame, (int(x), int(y)), int(r), (0, 255, 255), 2)
            cv.circle(frame, center, 2, (0, 255, 255), 2)

            print("x: " + str(int(x)) + ", y: " + str(int(y)))


    cv.imshow("circles", frame)
    cv.imshow("grayscale", gray)
    cv.imshow("mask", mask)
    # cv.imshow("circles", frame)

    if cv.waitKey(1) == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()
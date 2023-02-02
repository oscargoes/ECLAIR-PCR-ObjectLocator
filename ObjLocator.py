import cv2 as cv
import numpy as np

videoCapture = cv.VideoCapture(0)
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)


while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (17,17), 0)

    # mask = cv.inRange(frame, greenLower, greenUpper)
    # mask = cv.erode(mask, None, iterations=2)
    # mask = cv.dilate(mask, None, iterations=2)
    # result = cv.bitwise_and(gray, gray, mask= mask)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 100, 
                              param1 = 100, param2 = 30, minRadius = 50, maxRadius = 500)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i
        cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
        cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255, 0 , 255), 3)
        
        # print("x: " + str(chosen[0]) + ", y: " + str(chosen[1]))
        prevCircle = chosen

    cv.imshow("circles", frame)
    # cv.imshow("mask", result)
    cv.imshow("grayscale", gray)

    if cv.waitKey(1) == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()
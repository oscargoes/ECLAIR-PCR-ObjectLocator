import cv2 as cv
import numpy as np

videoCapture = cv.VideoCapture(1, cv.CAP_DSHOW)
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)


while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # figuring out what types of blurs are needed. This one seems to work the best?
    gray = cv.GaussianBlur(gray, (9,9), 0)
    # gray = cv.medianBlur(gray, 13)
    # gray = cv.bilateralFilter(gray,d=10,sigmaColor=200,sigmaSpace=200)


    # mask = cv.inRange(frame, greenLower, greenUpper)
    # mask = cv.erode(mask, None, iterations=2)
    # mask = cv.dilate(mask, None, iterations=2)
    # result = cv.bitwise_and(gray, gray, mask= mask)
    
    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=10, 
                              param1 = 60,
                              param2 = 75,
                              minRadius = 1, maxRadius = 500)

    # check out morphologyEx (can bypass 2-3 steps)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i
            cv.circle(frame, (i[0], i[1]), 1, (0, 100, 100), 3)
            cv.circle(frame, (i[0], i[1]), i[2], (255, 0 , 255), 3)
        
        # print("x: " + str(chosen[0]) + ", y: " + str(chosen[1]))
        prevCircle = chosen

    cv.imshow("circles", frame)
    # cv.imshow("mask", result)
    cv.imshow("grayscale", gray)

    if cv.waitKey(1) == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()
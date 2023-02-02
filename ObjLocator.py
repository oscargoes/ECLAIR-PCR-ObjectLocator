import cv2 as cv
import numpy as np

videoCapture = cv.VideoCapture(1)
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, 
                              param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for (x, y, r) in circles[0, :]:
            # if chosen is None:
            #     chosen = i
            # if prevCircle is not None:
            #     if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
            #         chosen = i
            cv.circle(frame, (x, y), 1, (0, 100, 100), 3)
            cv.circle(frame, (x, y), r, (255, 0 , 255), 3)
        
        # prevCircle = chosen

    cv.imshow("circles", frame)

    if cv.waitKey(1) == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()
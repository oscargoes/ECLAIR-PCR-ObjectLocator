import cv2 as cv
import numpy as np

# Load image, grayscale, median blur, Otsus threshold
videoCapture = cv.VideoCapture(1, cv.CAP_DSHOW)


while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 17)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    # Morph open 
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=3)

    # Find contours and filter using contour area and aspect ratio
    cnts = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.04 * peri, True)
        area = cv.contourArea(c)
        if len(approx) > 5 and area > 1000 and area < 500000:
            ((x, y), r) = cv.minEnclosingCircle(c)
            cv.circle(frame, (int(x), int(y)), int(r), (36, 255, 12), 2)

    cv.imshow('thresh', thresh)
    cv.imshow('opening', opening)
    cv.imshow('framez', frame)

    if cv.waitKey(1) == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()
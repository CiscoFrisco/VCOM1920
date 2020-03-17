import cv2
import numpy as np


def circleDetection(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(grayscale, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 200,
                               param1=50, param2=30, minRadius=2, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    return len(circles) != 0


def lineDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength, maxLineGap)

    if lines is not None:
        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

import cv2
import numpy as np


def colorDetection(image):
    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    redMask = cv2.inRange(hsvImg, lower_red, upper_red)

    # Bitwise-AND mask and original image
    redRes = cv2.bitwise_and(image, image, mask=redMask)

    # Threshold the HSV image to get only blue colors
    blueMask = cv2.inRange(hsvImg, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    blueRes = cv2.bitwise_and(image, image, mask=blueMask)

    cv2.imshow('frame', image)
    cv2.imshow('mask', redMask)
    cv2.imshow('res', redRes)
    cv2.imshow('mask2', blueMask)
    cv2.imshow('res2', blueRes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

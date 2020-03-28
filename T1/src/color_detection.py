import cv2
import numpy as np


def colorDetection(image):
    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    lower_red = np.array([0, 191, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(hsvImg, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([160, 191, 50])
    upper_red = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsvImg, lower_red, upper_red)

    # join my masks
    redMask = mask0+mask1

    # Bitwise-AND mask and original image
    redRes = cv2.bitwise_and(image, image, mask=redMask)

    # Threshold the HSV image to get only blue colors
    blueMask = cv2.inRange(hsvImg, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    blueRes = cv2.bitwise_and(image, image, mask=blueMask)

    return blueRes, redRes

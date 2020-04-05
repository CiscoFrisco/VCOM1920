import cv2
import numpy as np

lighting = {
    'regular_lower_blue' : np.array([100, 160, 50]),
    'regular_upper_blue' : np.array([135, 255, 255]),
    'regular_lower_red1' : np.array([0,180,120]),
    'regular_upper_red1' : np.array([10,255,255]),
    'regular_lower_red2' : np.array([170,180,120]),
    'regular_upper_red2' : np.array([180,255,255]),
    'bad_lower_blue' : np.array([100, 150, 50]),
    'bad_upper_blue' : np.array([140, 255, 255]),
    'bad_lower_red1' : np.array([0,70,70]),
    'bad_upper_red1' : np.array([10,255,255]),
    'bad_lower_red2' : np.array([170,70,70]),
    'bad_upper_red2' : np.array([180,255,255])
}


def fixBadContrast(image):
    img = image.copy()
   
    # _, _, gray = cv2.split(image)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl1 = clahe.apply(gray)

    # cv2.imshow("contrast", cl1)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab[...,0] = clahe.apply(lab[...,0])
    
    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return contrast


def fixBadLighting(image):

    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    lighting = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return lighting


def colorDetection(image, type):
    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = lighting[type + '_lower_blue']
    upper_blue = lighting[type + '_upper_blue']

    lower_red = lighting[type + '_lower_red1']
    upper_red = lighting[type + '_upper_red1']

    mask1 = cv2.inRange(hsvImg, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = lighting[type + '_lower_red2']
    upper_red = lighting[type + '_upper_red2']

    mask2 = cv2.inRange(hsvImg, lower_red, upper_red)

    # join my masks
    redMask = mask1+mask2

    # Threshold the HSV image to get only blue colors
    blueMask = cv2.inRange(hsvImg, lower_blue, upper_blue)

    redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    redMask = cv2.morphologyEx(redMask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

    # Bitwise-AND mask and original image
    redRes = cv2.bitwise_and(image, image, mask=redMask)

    # Bitwise-AND mask and original image
    blueRes = cv2.bitwise_and(image, image, mask=blueMask)

    mask = blueMask + redMask

    res = cv2.bitwise_and(image, image, mask=mask)

    return res, blueRes, redRes

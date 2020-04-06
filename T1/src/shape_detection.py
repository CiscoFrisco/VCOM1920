import cv2
import numpy as np
import imutils


def writeText(img, text, size, x, y):
    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 2)[0]
    # get coords based on boundary
    textX = int((x - (textsize[0] / 2)))
    cv2.putText(img, text, (textX, y),
                cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 2)


def detectShape(c):
    # Compute perimeter of contour and perform contour approximation
    shape = ""
    peri = cv2.arcLength(c, True)
    # approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    approx = cv2.approxPolyDP(c, 0.011 * peri, True)

    if len(approx) > 4 and len(approx) < 8:
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # Triangle
    if len(approx) == 3:
        shape = "triangle"

    # Square or rectangle
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # A square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    #Stop
    elif len(approx) == 8:
        shape = "octagonal"

    # Otherwise assume as circle or oval
    else:
        shape = "circle"

    return shape


def shapeDetection(image, colorRes, redRes, blueRes):
    img = image.copy()
    _, _, v = cv2.split(colorRes)
    _, _, redV = cv2.split(redRes)
    _, _, blueV = cv2.split(blueRes)

    # Find contours and detect shape
    cnts = cv2.findContours(v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    blueCnts = cv2.findContours(
        blueV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blueCnts = blueCnts[0] if len(blueCnts) == 2 else blueCnts[1]

    redCnts = cv2.findContours(
        redV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    redCnts = redCnts[0] if len(redCnts) == 2 else redCnts[1]

    centers = []
    for c in cnts:
        area = cv2.contourArea(c)

        # Ignore very small areas
        if area > 100:        # Identify shape
            shape = detectShape(c)

            if shape != "":
                # Find centroid and label shape name
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append([cX, cY])
                writeText(img, shape, 0.5, cX, (cY + 10))
                cv2.drawContours(img, [c], 0, (0, 255, 0), 6)

    writeColor(img, blueCnts, centers, "blue")
    writeColor(img, redCnts, centers, "red")

    return img

# Label the sign's color
def writeColor(img, cnts, centers, color):
    for c in cnts:
        area = cv2.contourArea(c)

        if area > 100:        # Identify shape
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if [cX, cY] in centers:
                writeText(img, color, 0.5, cX, (cY - 10))

import cv2
import numpy as np
import imutils


def circleDetection(image, res, color):
    h, s, v = cv2.split(res)

    cimg = image.copy()

    circles = cv2.HoughCircles(v, cv2.HOUGH_GRADIENT, 1, 120,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
            # get boundary of this text
            writeText(cimg, color + " circle", 0.6, i[0], i[1])

    cv2.imshow("circle", cimg)

    return cimg


def writeText(img, text, size, x, y):
    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 2)[0]
    # get coords based on boundary
    textX = int((x - (textsize[0] / 2)))
    cv2.putText(img, text, (textX, y),
                cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 2)


def detect_shape(c):
    # Compute perimeter of contour and perform contour approximation
    shape = ""
    peri = cv2.arcLength(c, True)
    # approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    approx = cv2.approxPolyDP(c, 0.011 * peri, True)

    if len(approx) > 4 and len(approx) < 8:
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    print(len(approx))

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

        if area > 100:        # Identify shape
            # circleDetection(image, colorRes, "red")
            shape = detect_shape(c)

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

def writeColor(img, cnts, centers, color):
    for c in cnts:
        area = cv2.contourArea(c)

        if area > 100:        # Identify shape
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if [cX, cY] in centers:
                writeText(img, color, 0.5, cX, (cY - 10))


def triangleDetection(image, res, color):
    img = image.copy()
    h, s, v = cv2.split(res)

    canny = cv2.Canny(v, 50, 200)
    contours, hier = cv2.findContours(canny, 1, 2)
    tri = []

    if contours is not None and len(contours) != 0:
        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 100:
                approx = cv2.approxPolyDP(
                    cnt, 0.01*cv2.arcLength(cnt, True), True)
                if len(approx) == 3:
                    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
                    tri = approx

                if len(tri) != 0:
                    x, y = triangleCenter(tri[0], tri[1], tri[2])
                    writeText(img, color + " triangle", 0.6, x, y)
                    for vertex in tri:
                        cv2.circle(
                            img, (vertex[0][0], vertex[0][1]), 5, 255, -1)

    return img


def triangleCenter(vertex1, vertex2, vertex3):
    return [int((vertex1[0][0] + vertex2[0][0] + vertex3[0][0])/3), int((vertex1[0][1] + vertex2[0][1] + vertex3[0][1])/3)]


def rectangleDetection(image, res, color):
    img = image.copy()
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)

    # Find contours
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and draw rectangles around contours
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 and h > 30:
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            x, y = rectangleCenter(x, y, x+w, y + h)
            writeText(img, color + " rectangle", 0.6, x, y)

    return img


def rectangleCenter(x1, y1, x2, y2):
    return [int((x1 + x2)/2), int((y1 + y2)/2)]

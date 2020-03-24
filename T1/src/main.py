import argparse
import numpy as np
import cv2
import color_detection
import shape_detection


def readImageFromFile(fileName):
    return cv2.imread(fileName, cv2.IMREAD_COLOR)


def readImageFromCamera():
    cap = cv2.VideoCapture(0)
    if not (cap.isOpened()):
        print("Could not open video device")
        return None
    ret, frame = cap.read()
    cap.release()
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a road image.')
    parser.add_argument(
        '-f', '--file', help="Read the image from a file.", type=str, required=False)
    args = parser.parse_args()

    image = readImageFromFile(
        args.file) if args.file else readImageFromCamera()

    if image is None:
        quit()

    blueRes, redRes = color_detection.colorDetection(image)
    rectangleBlue = shape_detection.rectangleDetection(image, blueRes)
    rectangleRed = shape_detection.rectangleDetection(image, redRes)
    triangleBlue = shape_detection.triangleDetection(image, blueRes)
    triangleRed = shape_detection.triangleDetection(image, redRes)
    circleBlue = shape_detection.circleDetection(image, blueRes)
    circleRed = shape_detection.circleDetection(image, redRes)
    cv2.imshow('image', image)
    cv2.imshow('circleBlue', circleBlue)
    cv2.imshow('circleRed', circleRed)
    cv2.imshow('rectangleRed', rectangleRed)
    cv2.imshow('rectangleBlue', rectangleBlue)
    cv2.imshow('triangleBlue', triangleBlue)
    cv2.imshow('triangleRed', triangleRed)
    cv2.imshow('blueRes', blueRes)
    cv2.imshow('redRes', redRes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

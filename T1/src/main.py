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
        print("Image not found!")
        quit()

    colorRes, blueRes, redRes = color_detection.colorDetection(image)
    signs = shape_detection.shapeDetection(image, colorRes, redRes, blueRes)

    cv2.imshow('Signs', signs)
    # cv2.imshow('blueRes', blueRes)
    # cv2.imshow('redRes', redRes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

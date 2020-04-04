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

    # Color detection in both regular and bad lighting
    regColorRes, regBlueRes, regRedRes = color_detection.colorDetection(image, "regular")
    badColorRes, badBlueRes, badRedRes = color_detection.colorDetection(image, "bad")
    
    #Shape detection in both regular and bad lighting
    regular_signs = shape_detection.shapeDetection(image, regColorRes, regRedRes, regBlueRes)
    bad_signs = shape_detection.shapeDetection(image, badColorRes, badRedRes, badBlueRes)

    # Print Results
    cv2.imshow('Regular Lighting', regular_signs)
    cv2.imshow('Bad Lighting', bad_signs)
    
    cv2.imshow('Regular Blue', regBlueRes)
    cv2.imshow('Bad Blue', badBlueRes)

    cv2.imshow('Regular Red', regRedRes)
    cv2.imshow('Bad Red', badRedRes)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

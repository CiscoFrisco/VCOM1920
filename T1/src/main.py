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


    #Color correction
    contrast_image = color_detection.fixBadContrast(image)
    lighting_image = color_detection.fixBadLighting(image)
    

    # Color detection in both regular and bad lighting
    regColorRes, regBlueRes, regRedRes = color_detection.colorDetection(image, "regular")
    contrastColorRes, contrastBlueRes, contrastRedRes = color_detection.colorDetection(contrast_image, "bad")
    lightingColorRes, lightingBlueRes, lightingRedRes = color_detection.colorDetection(lighting_image, "bad")
    
    #Shape detection in both regular and bad lighting
    
    regular_signs = shape_detection.shapeDetection(image, regColorRes, regRedRes, regBlueRes)
    contrast_regular_signs = shape_detection.shapeDetection(contrast_image, contrastColorRes, contrastRedRes, contrastBlueRes)
    lighting_regular_signs = shape_detection.shapeDetection(lighting_image, lightingColorRes, lightingRedRes, lightingBlueRes)

    # bad_signs = shape_detection.shapeDetection(image, badColorRes, badRedRes, badBlueRes)

    # Print Results
    cv2.imshow('Regular', regular_signs)
    cv2.imshow('Fixed Contrast', contrast_regular_signs)
    cv2.imshow('Fixed Lighting', lighting_regular_signs)

    
    # cv2.imshow('Regular Blue', regBlueRes)
    # cv2.imshow('Bad Blue', badBlueRes)

    # cv2.imshow('Regular Red', regRedRes)
    # cv2.imshow('Bad Red', badRedRes)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

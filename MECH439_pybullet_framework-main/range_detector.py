# reference
# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils

from utils.Toolbox import *
from utils.Camera.realsense import RealSense, D455_DEFAULT_COLOR, D455_DEFAULT_DEPTH, L515_DEFAULT_DEPTH, L515_DEFAULT_COLOR

REALSENSE_SERIAL = "141322251060"
# REALSENSE_SERIAL = "f1371347"


def nothing(x):
    pass

if __name__ == '__main__':

    cam = RealSense(serial=REALSENSE_SERIAL)
    cam.initialize(resolution_color=D455_DEFAULT_COLOR, resolution_depth=D455_DEFAULT_DEPTH)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('HMin', 'RealSense', 0, 179, nothing)  # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin', 'RealSense', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'RealSense', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'RealSense', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'RealSense', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'RealSense', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'RealSense', 179)
    cv2.setTrackbarPos('SMax', 'RealSense', 255)
    cv2.setTrackbarPos('VMax', 'RealSense', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while True:

        img_rgb = cv2.resize(cam.get_color('cv'), dsize=(640, 400), interpolation=cv2.INTER_AREA)

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'RealSense')
        sMin = cv2.getTrackbarPos('SMin', 'RealSense')
        vMin = cv2.getTrackbarPos('VMin', 'RealSense')

        hMax = cv2.getTrackbarPos('HMax', 'RealSense')
        sMax = cv2.getTrackbarPos('SMax', 'RealSense')
        vMax = cv2.getTrackbarPos('VMax', 'RealSense')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        # Print if there is a change in HSV value
        if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
            hMin, sMin, vMin, hMax, sMax, vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('RealSense', output)


        # cv2.imshow('RealSense', img_rgb)

        # Keyboard input
        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # ESC
            cv2.destroyAllWindows()
            break
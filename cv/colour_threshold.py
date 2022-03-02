# code to determine hsv colour range of objects 
# input image in filepath to use it

import cv2
import sys
import numpy as np

def nothing(x):
    pass

# Load in image
image = cv2.imread('qr_codes/dot4.jpg')
# Optionally undistort the image
from unfisheye import undistort
image = undistort(image)

SHOW_MASK = False # False => show entire image after threshold, True => just show mask

# Set initial values
hMin = 140; sMin = 90; vMin = 101; hMax = 166; sMax = 255; vMax = 255

# Create a window, scale it to fit screen
win = cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('image', 600, 600)

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for trackbars
cv2.setTrackbarPos('HMax', 'image', hMax)
cv2.setTrackbarPos('SMax', 'image', sMax)
cv2.setTrackbarPos('VMax', 'image', vMax)
cv2.setTrackbarPos('HMin', 'image', hMin)
cv2.setTrackbarPos('SMin', 'image', sMin)
cv2.setTrackbarPos('VMin', 'image', vMin)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

output = image
wait_time = 33

while(1):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image,image, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    if SHOW_MASK:
        cv2.imshow('image',mask)
    else:
        cv2.imshow('image',output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
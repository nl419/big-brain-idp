# code to determine hsv colour range of objects 
# input image in filepath to use it

import cv2
import sys
import numpy as np
from unfisheye import undistort

def nothing(x):
    pass

invert_hue = False
do_crop_board = True
do_crop_pickup = False
do_blur = False
do_remove_shadow = False
do_kmeans = False
do_erode = False

# Load in image
# image = cv2.imread('new_board/1.jpg')
# image = cv2.imread('dots/dot3.jpg')
image = cv2.imread('dots/dot7.jpg')
# image = cv2.imread('checkerboard2/3.jpg')

# Preprocess
from unfisheye import undistort
from crop_board import crop_board, remove_shadow, kmeans, get_pickup_corners
from find_coords import get_shift_invmat_mat
image = undistort(image)
if do_crop_board or do_crop_pickup:
    image2 = image.copy()
    image2 = remove_shadow(image2)
    shift, invmat, _ = get_shift_invmat_mat(image2)
    if do_crop_pickup:
        image = crop_board(image, shift, invmat, get_pickup_corners(shift, invmat))
    else:
        image = crop_board(image, shift, invmat)
if do_blur:
    image = cv2.blur(image, (5,5))
if do_remove_shadow:
    image = remove_shadow(image, 101)
temp = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
cv2.imshow("image", temp)
cv2.waitKey(0)
if do_kmeans:
    image = kmeans(image, 4)
    cv2.imshow("image", image)
    cv2.waitKey(0)



SHOW_MASK = False # False => show entire image after threshold, True => just show mask

# Set initial values
hMin = 0; sMin = 100; vMin = 80; hMax = 179; sMax = 255; vMax = 255

# Create a window, scale it to fit screen
# win = cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_NORMAL)
win = cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 50, 50)

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

    if invert_hue:
        # Choose all parts of the image EXCEPT the selected hue
        # (also limit by sat and val)
        lower0 = np.array([0, sMin, vMin])
        upper0 = np.array([hMin, sMax, vMax])

        lower1 = np.array([hMax, sMin, vMin])
        upper1 = np.array([179, sMax, vMax])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask0 = cv2.inRange(hsv, lower0, upper0)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask = cv2.bitwise_or(mask0, mask1)
    else:
        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
    
    if do_erode:
        cv2.erode(mask, (5,5), iterations=5)
        cv2.dilate(mask, (5,5), iterations=5)
    output = cv2.bitwise_and(image,image, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        print("hMin = %d; sMin = %d; vMin = %d; hMax = %d; sMax = %d; vMax = %d" % (hMin , sMin , vMin, hMax, sMax , vMax))
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
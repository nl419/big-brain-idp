import cv2
from cv2 import COLOR_BGR2HSV
import numpy as np

# Use camera height + position to project coordinates on the ground
# up to the height of the robot
# (literally just scale the coordinates to the centre a bit)

def find_block(image: np.ndarray):
    # Look at the bottom left corner
    # Mask the pickup area
    # Reduce the number of colours to 4 with k-means
        # (theoretically: black, white, red, blue)
    # Mask off any parts which aren't coloured
    # 

    hMin = 18; sMin = 84; vMin = 131; hMax = 153; sMax = 255; vMax = 255
    hsv = cv2.cvtColor(image, COLOR_BGR2HSV)
    
    lower0 = np.array([0, sMin, vMin])
    upper0 = np.array([hMin, sMax, vMax])
    lower1 = np.array([hMax, sMin, vMin])
    upper1 = np.array([179, sMax, vMax])

    mask0 = cv2.inRange(hsv, lower0, upper0)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask = cv2.bitwise_or(mask0, mask1)


"""Show all the corners found with cv2.goodFeaturesToTrack.
Sensitive to noise, so """

import cv2
from laggy_video import VideoCapture
import numpy as np
from unfisheye import undistort

cap = VideoCapture('http://localhost:8081/stream/video.mjpeg')
USE_HARRIS = False # whether to use Harris corner detection

CORNER_CLOSENESS = 5 # Minimum distance between corners
CORNER_QUALITY = 0.1
CORNER_NUMBER = 100
BLUR_RAD = 5 # Blur radius for dealing with noise (only used if USE_HARRIS == False)

while True:
    frame = cap.read()
    frame = undistort(frame, 0.4)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if USE_HARRIS:
      # Create binary image
      gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                      cv2.THRESH_BINARY,401,2)
      # Get corner image (brighter == more corner-y)
      corners = cv2.cornerHarris(gray, 5,3,0.04)
      corners = cv2.dilate(corners, None)
      # Set frame to red at corners
      frame[corners>0.01*corners.max()] = [0,0,255]
    else:
      # Blur image to filter out noise
      blurred = cv2.blur(gray, (BLUR_RAD,BLUR_RAD))
      # Get corner locations
      corners = cv2.goodFeaturesToTrack(blurred, CORNER_NUMBER, CORNER_QUALITY, CORNER_CLOSENESS)
      # Draw corners
      corners = np.int0(corners)
      for corner in corners:
        # print(corner)
        cv2.circle(frame, corner[0], 5, (0,0,255), 2)
      # print("===")
    
    cv2.imshow("frame", frame)
    if chr(cv2.waitKey(1)&255) == 'q':
        break
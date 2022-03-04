from collections import deque
from imutils.video import VideoStream
from laggy_video import VideoCapture
import numpy as np
import math
import argparse
import cv2
import imutils
import time
from unfisheye import undistort

def draw_angled_rec(x0, y0, width, height, angle, img):

    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, (255, 255, 255), 3)
    cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    cv2.line(img, pt2, pt3, (255, 255, 255), 3)
    cv2.line(img, pt3, pt0, (255, 255, 255), 3)

#colour detection with colour detection program
'''# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,help="max buffer size")
args = vars(ap.parse_args())

#capturing laggy video
cap = VideoCapture('http://localhost:8081/stream/video.mjpeg')


# define the lower and upper boundaries of the "white"
# ball in the HSV color space, then initialize the
# list of tracked points

redLower = (164, 46, 166)
redUpper = (179, 255, 244)


pts = deque(maxlen=args["buffer"])

while True:
    # grab the current frame
    frame = cap.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
    #undistort the fisheye lens on the camera
    frame = undistort(frame)
    # resize the frame, blur it, and convert it to the HSV
    # color space
    #frame = imutils.resize(frame, width=1806)   

    #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "white", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
# find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to draw minimum enclosing rectange
        area = cv2.contourArea
        c = max(cnts, key=area)
        ((x,y), (w,h) , a) = cv2.minAreaRect(c)
        moments = cv2.moments(c)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

        #code to encircile the 2 objects that seem the most like the red blocks
        #block is approx 8x8 pixels 
        #if (w*h) < 70:
            #draw_angled_rec(x0, y0, width, height, angle, img)
        print('centre = ', x, y)
        print(w*h)
        draw_angled_rec(x,y,w,h,a,frame)
            

    # update the points queue
    pts.appendleft(center)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break'''

#template matching with colour detection first and then edge detection mask
#capturing laggy video
cap = VideoCapture('http://localhost:8081/stream/video.mjpeg')
while True:
    # grab the current frame
    frame = cap.read()
    frame = undistort(frame)
    if frame is None:
        break

    #use colour dection with hsv limits
    lower1 = np.array([120, 76, 89])
    upper1 = np.array([179, 255, 245])
    lower2 = np.array([0,0, 0])
    upper2 = np.array([20,0,0])
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    #need second mask to find all red values
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    output = cv2.bitwise_and(frame,frame, mask= mask)

    #use edge detction
    img_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (11,11), 0)
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    output2 = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

    #now find the shape of the block expecting - using template matching? or finding the shape?


    cv2.imshow('Frame', output2)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close all windows
cv2.destroyAllWindows()
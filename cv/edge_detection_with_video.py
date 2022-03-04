import cv2
import numpy as np
from unfisheye import undistort


print ("starting grab")
cap = cv2.VideoCapture('http://localhost:8081/stream/video.mjpeg')
while True:
    #capture image
    ret, frame = cap.read()
    frame = undistort(frame)

    #edge detection
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    frame = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    #cv2.imshow('Canny Edge Detection', edges)
    #cv2.waitKey(0)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    #print ("after release")
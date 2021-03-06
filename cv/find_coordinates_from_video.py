"code to send over ramp"
import cv2
import sys
from unfisheye import undistort

'''cap = cv2.VideoCapture('http://localhost:8081/stream/video.mjpeg')
ret, frame = cap.read()
cv2.imshow('Frame', frame)'''

img = cv2.imread('cv/block.jpg')
frame = img
cv2.imshow('image', frame)
frame = undistort(frame, 0.4)

def mouseHandler(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)


cv2.setMouseCallback('image', mouseHandler)

while(True):

    # Capture frame-by-frame
    #ret, frame = cap.read()
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('image', frame)

#cap.release()
cv2.destroyAllWindows()
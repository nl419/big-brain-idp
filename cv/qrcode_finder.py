import cv2
import numpy as np
from find_qr import drawMarkers,getQRShape,getQRData
from laggy_video import VideoCapture
from unfisheye import undistort
cap = VideoCapture('http://localhost:8081/stream/video.mjpeg')
detector = cv2.QRCodeDetector()
while True:
    image = cap.read()
    image = undistort(image, 0.4)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_HITMISS,kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area,mindotp = getQRShape(box) 
        if area>1000 and area<5000:
            cv2.drawContours(image,[box],0,(0,191,255),2)
            #print(f"{box = }")
            #found = getQRData(image,box,detector)
            print(box)
            #frame = image[box[0]:box[1],box[2]:box[3]]
            #found, _ = detector.detect(image, np.array([box]))
            #print(found)
    #cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


# You should replace these 3 lines with the output in calibration step
import numpy as np
import cv2
import sys
import imutils


def undistort(img):
    DIM=(1806, 1440)
    K=np.array([[1000.0817683510912, 0.0, 898.0145797792809], [0.0, 1003.730509803676, 795.6302878125562], [0.0, 0.0, 1.0]])
    D=np.array([[-0.15481953360923378], [0.46847268935941083], [-0.6939986960654416], [0.3509367759414608]])
    #h,w = img.shape[:2]
    h = 1860
    w = 1440
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return undistorted_img
#img = cv2.imread('1.jpg')
#undistort(img)


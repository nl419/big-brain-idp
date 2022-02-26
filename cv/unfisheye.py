# You should replace these 3 lines with the output in calibration step
import numpy as np
import cv2
import sys
import imutils

# MAC checkerboard
# DIM=(1806, 1440)
# K=np.array([[1000.0817683510912, 0.0, 898.0145797792809], [0.0, 1003.730509803676, 795.6302878125562], [0.0, 0.0, 1.0]])
# D=np.array([[-0.15481953360923378], [0.46847268935941083], [-0.6939986960654416], [0.3509367759414608]])
# pic_name = "cv/1.jpg"

# Windows checkerboard
DIM=(903, 720)
K=np.array([[497.25261908023276, 0.0, 451.0843444150783], [0.0, 495.86040011896534, 385.5926770729965], [0.0, 0.0, 1.0]])
D=np.array([[-0.09268730609892956], [0.2504965645950943], [-0.3533145752831843], [0.16524591721022114]])
pic_name = "cv/1.jpg"

# Windows checkerboard2
# DIM=(1016, 760)
# K=np.array([[617.046363722009, 0.0, 508.1742575190148], [0.0, 614.3819211597321, 382.7568542470966], [0.0, 0.0, 1.0]])
# D=np.array([[-0.23284420692502958], [0.7725973070172576], [-1.2587721414504571], [0.7234483916379288]])
# pic_name = "checkerboard2/3.jpg"

debug = __name__ == "__main__"

def undistort(img):
    img = cv2.resize(img, DIM)
    if debug:
        cv2.namedWindow("input", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("input", DIM[0], DIM[1])
        cv2.imshow("input", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if debug:
        cv2.namedWindow("undistorted", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("undistorted", DIM[0], DIM[1])
        cv2.rectangle(undistorted_img, (1,1), (DIM[0] - 2, DIM[1] - 2), (255,0,0), 1)
        cv2.imshow("undistorted", undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return undistorted_img
if debug:
    img = cv2.imread(pic_name)
    undistort(img)


import cv2
import numpy as np
from unfisheye import undistort
import timeit

CHECKERBOARD = (2,2)
RESOLUTION = np.array([1016, 760])

start = timeit.timeit()

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

end = timeit.timeit()

cap = cv2.VideoCapture('http://localhost:8081/stream/video.mjpeg')
while True:
    ret, frame = cap.read()
    if frame is not None:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, RESOLUTION.tolist())
            # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #     cv2.THRESH_BINARY,21,2)
    ret, corners = cv2.findChessboardCorners(frame, CHECKERBOARD)
    if ret == True:
        frame = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, True)
        objpoints.append(objp)
        cv2.cornerSubPix(frame,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
    frame = undistort(frame)
    cv2.imshow("hehexd", frame)
    key = cv2.waitKey(1) & 0xFF

    # # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
         break
cv2.destroyAllWindows()

"""Calibrate unfisheye coefficients using images containing a known checkerboard pattern.
Paste the coefficients into unfisheye.py to see the results."""

import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
import imutils

### VARIABLES

CHECKERBOARD = (6,9)                            # Number of internal corners in checkerboard
folder_name = "checkerboard2"                   # Folder containing checkerboard images
# RESOLUTION = np.array([1806, 1440]) // 2      # For Checkerboard
RESOLUTION = np.array([1016, 760])              # For Checkerboard2
DEBUG = True                                    # Whether to see intermediate images

# Flags for calibration - uncomment as appropriate
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
# FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE

### END OF VARIABLES

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
DEBUG = DEBUG and __name__ == "__main__"
#images = glob.glob('*.jpg')
def load_images_from_folder(folder):
    count = 0
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            count += 1
            img = cv2.resize(img, RESOLUTION.tolist())
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,21,2)
            # Denoising
            N = 2
            its = 1
            kernel = np.ones((N,N),np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=its)
            if DEBUG:
                cv2.imshow("debug calibration:", img)
                cv2.waitKey(0)
            images.append(img)
    cv2.destroyAllWindows()
    print("Found", count, "images in folder", folder)
    return images

images = load_images_from_folder(folder_name)

for img in images:
    #img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = img
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, FLAGS)
    # If found, add object points, image points (after refining them)
    if ret == True:
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, True)
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
    if DEBUG:
        cv2.imshow("hehexd", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
N_OK = len(objpoints)
print("Found " + str(N_OK) + " valid images for calibration")
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")
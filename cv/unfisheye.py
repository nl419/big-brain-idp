"""Remove fisheye effects from a single image, given the calibration coefficients found in calibrate.py"""

import numpy as np
import cv2

# MAC checkerboard
# DIM=(1806, 1440)
# K=np.array([[1000.0817683510912, 0.0, 898.0145797792809], [0.0, 1003.730509803676, 795.6302878125562], [0.0, 0.0, 1.0]])
# D=np.array([[-0.15481953360923378], [0.46847268935941083], [-0.6939986960654416], [0.3509367759414608]])
# pic_name = "cv/1.jpg"

# Windows checkerboard
# DIM=(903, 720)
# K=np.array([[497.25261908023276, 0.0, 451.0843444150783], [0.0, 495.86040011896534, 385.5926770729965], [0.0, 0.0, 1.0]])
# D=np.array([[-0.09268730609892956], [0.2504965645950943], [-0.3533145752831843], [0.16524591721022114]])
# pic_name = "cv/1.jpg"

# Windows checkerboard2
DIM=(1016, 760)
K=np.array([[617.046363722009, 0.0, 508.1742575190148], [0.0, 614.3819211597321, 382.7568542470966], [0.0, 0.0, 1.0]])
D=np.array([[-0.23284420692502958], [0.7725973070172576], [-1.2587721414504571], [0.7234483916379288]])
pic_name = "checkerboard2/3.jpg"

debug = __name__ == "__main__"

def undistort(img: np.ndarray , balance=0.0, dim2=None, dim3=None) -> np.ndarray:
    """Returns the image without fisheye distortion.

    Uses calibration coefficients found using calibrate.py with checkerboards.
    Allows retrieval of some of the cropped pixels with the help of "balance".
    See link for more info:
    https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f

    Parameters
    ----------
    img : np.ndarray
        The image with distortion
    balance : float
        How much of the undistorted image to keep: 1 = keep all, 0 = keep only inner portion.
    dim2 : list
        Dimensions (x,y) of the cropped image (default same as img)
    dim3 : list
        Dimensions (x,y) of the output image (default same as img)

    Returns
    -------
    undistorted_img : np.ndarray
        The image without distortion
    """
    # img = cv2.resize(img, DIM)
    if debug:
        cv2.imshow("input", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. 
    # OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, 
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if debug:
        cv2.imshow("undistorted", undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return undistorted_img
if debug:
    img = cv2.imread(pic_name)
    undistort(img, 0.4)


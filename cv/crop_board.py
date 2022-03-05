import numpy as np
import cv2
from find_coords import untransform_board

_DEBUG = __name__ == "__main__"

BOARD_CORNERS_T = np.array((
    (-0.04257, -2.39275),
    (-2.36891, -0.01291),
    (-0.06849, 2.59533),
    (2.55669, -0.03823)
))

PICKUP_CORNERS_T = np.array((
    (0.22048, 1.67371),
    (-0.31464, 1.64469),
    (-0.32059, 2.16882),
    (0.20810, 2.20481)
))

def get_board_corners(shift, invmat):
    return np.array((
        untransform_board(shift, invmat, BOARD_CORNERS_T[0]),
        untransform_board(shift, invmat, BOARD_CORNERS_T[1]),
        untransform_board(shift, invmat, BOARD_CORNERS_T[2]),
        untransform_board(shift, invmat, BOARD_CORNERS_T[3])
    ))

def get_pickup_corners(shift, invmat):
    return np.array((
        untransform_board(shift, invmat, PICKUP_CORNERS_T[0]),
        untransform_board(shift, invmat, PICKUP_CORNERS_T[1]),
        untransform_board(shift, invmat, PICKUP_CORNERS_T[2]),
        untransform_board(shift, invmat, PICKUP_CORNERS_T[3])
    ))
    

def crop_board(image: np.ndarray, shift: np.ndarray, invmat: np.ndarray,
               corners: np.ndarray = None):
    """Remove everything in the image except for the area inside the corners

    Parameters
    ----------
    image : np.ndarray
        An undistorted image
    shift : np.ndarray
        The shift vector found with get_shift_invmat_mat()
    invmat : np.ndarray
        The inverse matrix found with get_shift_invmat_mat()
    corners : np.ndarray or None
        Corners to crop into, default board outer corners

    Returns
    -------
    np.ndarray
        The cropped image
    """
    if corners is None:
        corners = get_board_corners(shift, invmat)
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int0(corners), 255)
    cropped = cv2.bitwise_and(image, image, mask=mask)
    if __name__ == "__main__":
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.imshow("cropped", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return cropped

def remove_shadow(image: np.ndarray):
    """Remove the lightspots / shadows from an image of the board

    Parameters
    ----------
    image : np.ndarray
        Image to remove lightspots / shadows from

    Returns
    -------
    np.ndarray
        Image without lightspots / shadows
    """
    inverted = cv2.bitwise_not(image)
    rgb_planes = cv2.split(inverted)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 101)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    if _DEBUG:
        cv2.imshow("result_norm", cv2.bitwise_not(result_norm))
        cv2.waitKey(0)

    return cv2.bitwise_not(result_norm)

def _test_kmeans():
    from find_coords import get_shift_invmat_mat
    from unfisheye import undistort
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html

    # Load, undistort, crop, remove shadow
    image = cv2.imread('checkerboard2/3.jpg')
    image = undistort(image)
    shift, invmat, mat = get_shift_invmat_mat(image)
    image = crop_board(image, shift, invmat)
    image = remove_shadow(image)

    K = 8
    Z = image.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    cv2.imshow('res2',res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _test_crop():
    """Test cropping into the board and into the pickup area"""
    from find_coords import get_shift_invmat_mat
    from unfisheye import undistort
    image = cv2.imread("dots/dot2.jpg")
    image = undistort(image)
    shift, invmat, _ = get_shift_invmat_mat(image)
    crop_board(image, shift, invmat)
    cv2.destroyAllWindows()
    crop_board(image, shift, invmat, get_pickup_corners(shift, invmat))


if __name__ == "__main__":
    # _test_kmeans()
    _test_crop()
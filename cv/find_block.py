import cv2
import numpy as np
from unfisheye import undistort
from find_coords import get_shift_invmat_mat
from crop_board import get_pickup_corners, crop_board, remove_shadow

_DEBUG = __name__ == "__main__"

def find_block(image: np.ndarray, shift: np.ndarray, invmat: np.ndarray, angle_offset: float = 0):
    """Find a single block in the pickup area. 
    Will return None, None if no blocks or multiple blocks detected.

    Parameters
    ----------
    image : np.ndarray
        The undistorted image to search
    shift : np.ndarray
        Shift vector
    invmat : np.ndarray
        Inverse matrix
    angle_offset : float
        Angle offset for orientation, +ve ccw, in degrees, default 0

    Returns
    -------
    centre : np.ndarray or None
        Centre of the block, if found, else None
    forward : np.ndarray or None
        Forward direction of the block, if found, else None
    """
    
    # Mask the pickup area
    cropped = crop_board(image.copy(), shift, invmat, get_pickup_corners(shift, invmat))
    
    # Mask the coloured areas
    hMin = 0; sMin = 100; vMin = 80; hMax = 179; sMax = 255; vMax = 255
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    mask_coloured = cv2.inRange(hsv, lower, upper)

    # Get contours, remove ones with small area
    cnts, _ = cv2.findContours(mask_coloured, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    areas = []
    block = None
    for c in cnts:
        M = cv2.moments(c)
        area = M['m00']
        _DEBUG and print(area)
        if area > 50:
            centre = (M['m10']/M['m00'], M['m01']/M['m00'])
            centre = np.array(centre)
            centres.append(centre)
            areas.append(area)
            block = c

    # If no block found, or more than one block found
    if block is None or len(centres) > 1:
        return None, None

    # Get orientation
    mid,_,angle = cv2.minAreaRect(block)
    angle = (angle - angle_offset) % 90
    forward = np.array((np.sin(np.radians(angle)), -np.cos(np.radians(angle))))
    _DEBUG and print(f"forward {forward}")
    # so forward now points in the direction that the robot should come from
    
    return mid, forward
    
def _test_rectangle_orientation():
    # A very basic example demonstrating orientation detection
    # for a block with some perspective offset.
    # Try varying offset, and see what happens.

    offset = 100

    points = np.array((
        (0,0),
        (100,0),
        (100 + offset, offset),
        (100 + offset, 100 + offset),
        (offset, 100 + offset),
        (0,100)
    ), dtype = np.int0)

    blank_im = np.zeros((300,300), np.uint8)

    cv2.fillConvexPoly(blank_im, points, 255)
    output = cv2.minAreaRect(points)
    print(output)
    _,_,angle = cv2.minAreaRect(points)
    print(angle)
    cv2.imshow("im", blank_im)
    cv2.waitKey(0)
    
    skew = 100
    angle_offset = 30

    points = np.array((
        (skew,0),
        (skew + 100,skew),
        (100,100 + skew),
        (0,100)
    ), dtype = np.int0)

    blank_im = np.zeros((300,300), np.uint8)

    cv2.fillConvexPoly(blank_im, points, 255)
    output = cv2.minAreaRect(points)
    print(f"skew: {output}")
    centre,_,angle = cv2.minAreaRect(points)
    print(f"angle1: {angle}")
    angle = (angle - angle_offset) % 90
    front = np.array((np.sin(np.radians(angle)), -np.cos(np.radians(angle)))) * 100 + centre
    front.astype(np.int0)
    print(f"angle2: {angle}")
    print(f"front: {front}")
    cv2.circle(blank_im, np.int0(centre), 5, 127, -1)
    cv2.circle(blank_im, np.int0(front), 5, 127, -1)
    cv2.imshow("im", blank_im)
    cv2.waitKey(0)

def _test_find_block():
    # Find blocks in the pickup area of still images

    # image = cv2.imread("checkerboard2/3.jpg")
    # image = cv2.imread("block_ims/2.jpg")
    image = cv2.imread("new_board/1.jpg")
    image = undistort(image)
    im2 = remove_shadow(image.copy())
    shift, invmat, mat = get_shift_invmat_mat(im2)
    image = crop_board(image, shift, invmat)
    image = remove_shadow(image)
    cv2.imshow("before", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mid, forward = find_block(image, shift, invmat, 0)

    if mid is None:
        cv2.imshow("no block!", image)
        cv2.waitKey(0)
        return

    front = mid + forward * 80

    cv2.drawMarker(image, np.int0(mid), (0,0,255), cv2.MARKER_CROSS,
                   40, 2)
    cv2.line(image, np.int0(mid), np.int0(front), (0,255,0), 4)

    print(mid)
    print(forward)

    cv2.imshow("block found", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    # _test_rectangle_orientation()
    _test_find_block()


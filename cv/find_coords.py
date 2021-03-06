# importing the module
import cv2
import numpy as np

_DEBUG = __name__ == "__main__"
_DRAW_MASKS = True


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    image = params
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        coords = np.array((x,y))

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.drawMarker(image, np.int0(coords), (0,0,255), cv2.MARKER_CROSS, 10, 2)
        cv2.imshow('image', image)

    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = image[y, x, 0]
        g = image[y, x, 1]
        r = image[y, x, 2]
        cv2.putText(image, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', image)

# function to display the coordinates of
# of the points clicked on the image and normalise
# them based on reference points
def click_event_normalised(event, x, y, flags, params):
    image = params[0]
    shift = params[1]
    mat = params[2]
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        coords = np.array((x,y))
        trans_coords = np.matmul(mat, coords - shift)
        # displaying the coordinates
        # on the Shell
        # print(f"real coords {coords[0]}, {coords[1]}   transformed coords {trans_coords[0]:.5f}, {trans_coords[1]:.5f}")
        print(f"({trans_coords[0]:.5f}, {trans_coords[1]:.5f})")

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.drawMarker(image, np.int0(coords), (0,0,255), cv2.MARKER_CROSS, 10, 2)
        cv2.imshow('image', image)

def barrier_centres(image: np.ndarray):
    """Find the centrepoints of the yellow barriers in the image

    Parameters
    ----------
    image : np.ndarray
        The image to search (works best if undistorted)

    Returns
    -------
    np.ndarray
        The two centrepoints of the barriers
    """
    # Only pickup the yellow parts
    # Filter out small areas
    # Filter out areas with small length
    # Pick the two contours which are closest to the middle
    # hMin = 17; sMin = 81; vMin = 97; hMax = 49; sMax = 255; vMax = 255
    hMin = 20; sMin = 20; vMin = 20; hMax = 50; sMax = 255; vMax = 255
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Find contours of correct area
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    middle = np.array((image.shape[1], image.shape[0])) / 2
    if _DEBUG:
        image_cnt = image.copy()
        cv2.drawContours(image_cnt, cnts, -1, (255,255,0), 2)
        cv2.circle(image_cnt, np.int0(middle), 10, (0,255,0), 3)
        cv2.imshow("image_cnt", image_cnt)
        cv2.waitKey(0)

        image_cnt_thresh = image.copy()
    centres = []
    areas = []
    for c in cnts:
        M = cv2.moments(c)
        area = M['m00']
        _,radius = cv2.minEnclosingCircle(c)
        _DEBUG and print(area, radius)
        if area > 800 and radius > 80:
            centre = (M['m10']/M['m00'], M['m01']/M['m00'])
            centre = np.array(centre) - middle
            centres.append(centre)
            areas.append(area)
            if _DEBUG:
                cv2.drawContours(image_cnt_thresh, [c], -1, (255,255,0), 2)
    if _DEBUG:
        cv2.imshow("image_cnt_thresh", image_cnt_thresh)
        cv2.waitKey(0)
        cv2.destroyWindow("image_cnt_thresh")
        cv2.destroyWindow("image_cnt")

    # Grab the two closest contours to the middle
    mags = np.linalg.norm(centres, axis=1)
    args = mags.argsort()
    centres = np.take(centres, args[:2], axis=0)

    return centres + middle

def dropoff_boxes(img: np.ndarray, shift=None, invmat=None, improve_dropoff=True):
    """Find the dropoff boxes more accurately. Supply shift and invmat to
    make the processing faster.

    Parameters
    ----------
    img : np.ndarray
        The undistorted image to search
    shift : np.ndarray
        The shift vector found with get_shift_invmat_mat(), default None
    invmat : np.ndarray
        The inverse matrix found with get_shift_invmat_mat(), default None
    improve_dropoff : bool
        Whether to improve the dropoff locations

    Returns
    -------
    blues : list of np.ndarray
        The coordinates of the blue dropoff boxes
    reds : list of np.ndarray
        The coordinates of the red dropoff boxes
    """
    # Find the dropoff boxes
    
    # Preprocess
    from unfisheye import undistort
    from crop_board import crop_board, remove_shadow, kmeans, untransform_board
    from find_coords import get_shift_invmat_mat
    image = img.copy()
    if shift is None or invmat is None:
        image2 = image.copy()
        image2 = remove_shadow(image2)
        shift, invmat, _ = get_shift_invmat_mat(image2)
    dropoff_area = (
        untransform_board(shift, invmat, np.array((-0.77739, -0.55168))),
        untransform_board(shift, invmat, np.array((-0.77179, -1.26696))),
        untransform_board(shift, invmat, np.array((0.72577, -1.32124))),
        untransform_board(shift, invmat, np.array((0.74001, -0.60672)))
    )
    if _DEBUG:
        cv2.imshow("before crop", image)
        cv2.waitKey(0)
        cv2.destroyWindow("before crop")
    # image = crop_board(image, shift, invmat)
    image = remove_shadow(image, 81)
    image = crop_board(image, shift, invmat, dropoff_area)
    if _DRAW_MASKS:
        cv2.imshow("after crop", image)
        cv2.waitKey(0)
        cv2.destroyWindow("after crop")
    image = kmeans(image, 4)


    if _DRAW_MASKS:
        cv2.imshow("Kmeans", image)
        cv2.waitKey(0)
        cv2.destroyWindow("Kmeans")

    # Threshold for white
    hMin = 0; sMin = 0; vMin = 120; hMax = 179; sMax = 40; vMax = 255
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    if _DRAW_MASKS:
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.destroyWindow("mask")

    # mask = cv2.erode(mask, (8,8), iterations=1)
    # mask = cv2.dilate(mask, (8,8), iterations=1)

    # if _DEBUG:
    #     cv2.imshow("after opening", mask)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow("after opening")

    # Filter out contours of wrong size
    # Calculate centres for the remaining contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if _DRAW_MASKS:
        image = img.copy()
        cv2.drawContours(image, cnts, -1, (255,255,0), 2)
        cv2.imshow("contours", image)
        cv2.waitKey(0)
        cv2.destroyWindow("contours")
        image = img.copy()

    centres = []
    areas = []
    for c in cnts:
        M = cv2.moments(c)
        area = M['m00']
        print(area)
        if area > 100 and area < 200:
            centre = (M['m10']/M['m00'], M['m01']/M['m00'])
            centre = np.array(centre)
            centres.append(centre)
            areas.append(area)
            if _DRAW_MASKS: cv2.drawContours(image, [c], -1, (255,255,0), 2)
    
    if _DRAW_MASKS:
        print(centres)
        cv2.imshow("filtered", image)
        cv2.waitKey(0)
        cv2.destroyWindow("filtered")

    from waypoints import BLUE_DROPOFFS_T, RED_DROPOFFS_T, untransform_board
    shift, invmat, mat = get_shift_invmat_mat(img)
    if not improve_dropoff:
        return (untransform_board(shift, invmat, BLUE_DROPOFFS_T[0]),
                untransform_board(shift, invmat, BLUE_DROPOFFS_T[1])),\
               (untransform_board(shift, invmat, RED_DROPOFFS_T[0]),
                untransform_board(shift, invmat, RED_DROPOFFS_T[1]))

    # Use the estimated locations from waypoints.py
    # Get the nearest contours to the estimates
    blues = []
    reds = []
    fail_i = -1
    for i,t in enumerate(np.append(BLUE_DROPOFFS_T, RED_DROPOFFS_T, axis=0)):
        # Yes this is inefficient, but it's only ever run once or twice.
        est = untransform_board(shift, invmat, t)
        vecs = centres - est
        mags = np.linalg.norm(vecs, axis=1)
        j = np.argmin(mags)
        centre = centres[j]
        if mags[j] > 40:
            print(f"Dropoff location {i} was not found. Skipping...")
            assert fail_i == -1, "Too many boxes couldn't be found."
            fail_i = i
            centre = None
        if i < 2:
            blues.append(centre)
        else:
            reds.append(centre)
    
    if fail_i != -1:
        # Estimate failed box location based on other boxes
        if fail_i < 2:
            blues[fail_i] = blues[1-fail_i] + (reds[fail_i] - reds[1-fail_i])
        else:
            fail_i -= 2
            reds[fail_i] = reds[1-fail_i] + (blues[fail_i] - blues[1-fail_i])

    return blues, reds

def get_shift_invmat_mat(image: np.ndarray):
    """Get the shift vector, inverse matrix, and forward matrix for
    transformation between camera coords and normalised board coords.

    Parameters
    ----------
    image : np.ndarray
        The image to process (without distortion)

    Returns
    -------
    shift, invmat, mat : np.ndarray
        Shift, inverse matrix, and forward matrix for transformation.
    """
    centres = barrier_centres(image)
    shift = (centres[0] + centres[1]) / 2
    # c = bottom right barrier centroid - shift
    c = centres[0] - shift
    c = c if c[0] > 0 else centres[1] - shift
    # Map to normalised coordinates
    mat = np.array(((c[0], c[1]), (-c[1], c[0]))) / (c[0]**2 + c[1]**2)
    # Map from normalised coordinates
    invmat = np.array(((c[0], -c[1]), (c[1], c[0])))
    if __name__ == "__main__":
        print(centres)
        cv2.circle(image, np.int0(centres[0]), 5, (255,0,0), -1)
        cv2.circle(image, np.int0(centres[1]), 5, (255,0,0), -1)
    return shift, invmat, mat

def transform_board(shift: np.ndarray, mat: np.ndarray, x: np.ndarray):
    """Return the normalised board coordinates from camera coordinates

    Parameters
    ----------
    shift : np.ndarray
        The shift vector found with get_shift_invmat_mat()
    mat : np.ndarray
        The matrix found with get_shift_invmat_mat()
    x : np.ndarray
        Camera coordinates to transform

    Returns
    -------
    np.ndarray
        Transformed coordinates
    """
    return np.matmul(mat, x - shift)

def untransform_board(shift: np.ndarray, invmat: np.ndarray, transformed :np.ndarray):
    """Return the camera coordinates from normalised board coordinates

    Parameters
    ----------
    shift : np.ndarray
        The shift vector found with get_shift_invmat_mat()
    invmat : np.ndarray
        The inverse matrix found with get_shift_invmat_mat()
    transformed : np.ndarray
        Normalised board coordinates to untransform

    Returns
    -------
    np.ndarray
        Camera coords corresponding to the transformed coords
    """
    return np.matmul(invmat, transformed) + shift

def _pick_points():
    from unfisheye import undistort
    # reading the image
    image = cv2.imread('checkerboard2/3.jpg', 1)
    image = undistort(image)

    # displaying the image
    cv2.imshow('image', image)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event, image)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

def _pick_points_normalised():
    from unfisheye import undistort
    from crop_board import crop_board, remove_shadow

    # Return the points normalised by the barrier positions

    # image = cv2.imread('dots/dot17.jpg')
    image = cv2.imread('nav-test-fails/1.jpg')
    # image = cv2.imread('checkerboard2/3.jpg')

    image = undistort(image, 0.4)
    image2 = image.copy()
    # Shadow removal on an uncropped image results in many artefacts,
    # but is helpful for reliable shift / mat calculation.
    # So we keep a copy handy, before any artefacts.
    
    # Preprocessing
    image2 = remove_shadow(image2, 101)
    shift, invmat, mat = get_shift_invmat_mat(image2)
    image = crop_board(image, shift, invmat)

    # displaying the image
    cv2.imshow('image', remove_shadow(image))

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event_normalised, (image, shift, mat))

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

def _test_dropoff_boxes():
    from unfisheye import undistort

    # image = cv2.imread('new_board/1.jpg')
    image = cv2.imread('nav-test-fails/1.jpg')
    # image = cv2.imread('dropoff-boxes/5.jpg')
    # image = cv2.imread('checkerboard2/3.jpg')

    image = undistort(image)
    
    for i,group in enumerate(dropoff_boxes(image)):
        for p in group:
            cv2.drawMarker(image, np.int0(p), (255,0,0) if i == 0 else (0,0,255),
                           cv2.MARKER_CROSS, 40, 1)

    cv2.imshow('image', image)
    cv2.waitKey(0)

if __name__=="__main__":
    # _pick_points()
    # _pick_points_normalised()
    _test_dropoff_boxes()
import cv2
import numpy as np
from unfisheye import undistort
from laggy_video import VideoCapture
from crop_board import crop_board
from robot_properties import *

# Flag to print debug information
_DEBUG = __name__ == "__main__"
_DRAW_MASKS = True and _DEBUG

def drawMarkers(img: np.ndarray, points: "list[int]", lineColour: "list[int]", removeParallax: bool = True):
    """Draws markers on the image for a four point bounding box.
    Also draws the forward-facing line. Removes effects of parallax
    by default.

    Parameters
    ----------
    img : np.ndarray
        The image to draw onto
    points : list[int]
        List of (x,y) coordinates of the bounding box vertices
    lineColour : list[int]
        Colour (B,G,R) of lines joining the vertices
    removeParallax : bool, optional
        Whether to remove parallax, default True
    Returns
    -------
    centre : np.ndarray 
        Centre of bounding box, to sub-pixel accuracay
    front : np.ndarray
        Front of bounding box, to sub-pixel accuracy
    """
    # Draw bounding box
    if removeParallax: 
        unparallaxed_points = []
        for j,p in enumerate(points):
            p1 = np.int0(undo_parallax(p))
            p2 = np.int0(undo_parallax(points[(j+1) % len(points)]))
            unparallaxed_points.append(undo_parallax(p))
            cv2.line(img, p1, p2, lineColour, 3)
        points = np.array(unparallaxed_points)
    else:
        for j,p in enumerate(points):
            p1 = np.int0(p)
            p2 = np.int0(points[(j+1) % len(points)])
            cv2.line(img, p1, p2, lineColour, 3)

    # Find salient coordinates
    centre = np.mean(points, axis=0)
    top_midpoint = np.mean(points[0:2], axis=0)
    marker_radius = 5
    cv2.circle(img, np.int0(centre), radius=marker_radius,
               color=(0, 0, 255), thickness=-1)
    cv2.circle(img, np.int0(top_midpoint), radius=marker_radius,
               color=(0, 0, 255), thickness=-1)
    cv2.line(img, np.int0(centre), np.int0(top_midpoint), color=(0, 0, 255), thickness=3)
    return centre, top_midpoint

def getDotBbox(centres):
    """Given a list of centres of contours (of the correct area), returns
    bbox around the triangular dot pattern (if found)

    Parameters
    ----------
    centres : np.ndarray
        A list of coordinates (x,y) of centres of contours (of the correct area)

    Returns
    -------
    found : bool
        Whether a valid set of centres was determined
    bbox : np.ndarray or None
        The bounding box, if determined.
    """
    minLength = 20
    maxLength = 50
    if len(centres) < 3:
        return False, None
    # Find a centre & corresponding pair of distance vectors which are perpendicular
    for c in centres:
        vecs = centres - c
        lengths = np.linalg.norm(vecs, axis=1)
        mask = (lengths > minLength) & (lengths < maxLength)
        if _DEBUG:
            print(f"c: {c}")
            print(f"vecs: {vecs}")
            print(f"lengths: {lengths}")
            print(f"mask: {mask}")

        vecs = vecs[mask]
        lengths = lengths[mask]
        for i, v in enumerate(vecs):
            j = (i+1) % len(vecs)
            cross = np.cross(v, vecs[j]) / lengths[i] / lengths[j]
            _DEBUG and print(f"cross: {cross}")
            if abs(cross) > 0.98: # <16 deg away from perp.
                break # Found.
        else:
            continue # No perp vectors found for this centre
        break
    else:
        _DEBUG and print("Invalid centres!")
        return False, None  # No perp vectors found for any centre
    if cross < 0: # Depending on ordering of centres
        vec1 = vecs[j]
        vec2 = vecs[i]
    else:
        vec1 = vecs[i]
        vec2 = vecs[j]
    if DOT_PATTERN_DIR == -1: # TODO make this a rotation, not just "forward" or "backward"
        bbox = [c + vec1 + vec2, c + vec2, c, c + vec1]
    else: 
        bbox = [c, c + vec1, c + vec1 + vec2, c + vec2]
    return True, np.int0(bbox)

def getDots(image: np.ndarray):
    """Return all the magenta dots of correct area within the image

    Parameters
    ----------
    image : np.ndarray
        The image to scan

    Returns
    -------
    list of np.ndarray
        Coordinates (x,y) of the centres of the dots
    """
    # Threshold hsv
    # hMin = 140; sMin = 90; vMin = 101; hMax = 166; sMax = 255; vMax = 255
    hMin = 140; sMin = 20; vMin = 160; hMax = 170; sMax = 255; vMax = 255
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    if _DRAW_MASKS:
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        contim = image.copy()
        filt_contim = image.copy()
    # Find contours of correct area
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    areas = []
    for c in cnts:
        M = cv2.moments(c)
        area = M['m00']
        _DEBUG and print(area)
        _DRAW_MASKS and cv2.drawContours(contim, [c], -1, (0,255,0), 2)
        # if area > 200 and area < 750: # A3 dots
        if area > 100 and area < 350: # A4 dots
            centre = (M['m10']/M['m00'], M['m01']/M['m00'])
            centre = np.array(centre)
            centres.append(centre)
            areas.append(area)
            _DRAW_MASKS and cv2.drawContours(filt_contim, [c], -1, (255,0,0), 2)
    if _DRAW_MASKS:
        cv2.imshow("Before filter", contim)
        cv2.imshow("Filtered", filt_contim)
        cv2.waitKey(0)

    return centres

def transform_coords(x: np.ndarray, centre: np.ndarray, front: np.ndarray):
    """Transform coordinates into the local coordinate system of the robot

    Examples:
    All inputs should be numpy arrays, but are written as tuples for convenience
    >>> transform_coords((1,0), (0,0), (1,0))
    (1, 0)
    >>> transform_coords((1,0), (0,0), (1,1))
    (0.5, -0.5)

    Parameters
    ----------
    x : np.ndarray
        The coordinates to transform
    centre : np.ndarray
        Coordinates (x,y) of the centre of the robot
    front : np.ndarray
        Coordinates (x,y) of the front of the robot

    Returns
    -------
    np.ndarray
        The transformed coordinates
    """
    
    dx = x - centre
    df = front - centre
    # Transformation matrix of df to unit length x axis
    mat = np.array(((df[0], df[1]), (-df[1], df[0]))) / np.linalg.norm(df)**2
    return np.matmul(mat, dx)

def untransform_coords(x: np.ndarray, centre: np.ndarray, front: np.ndarray):
    """Transform coordinates out of the local coordinate system of the robot

    Examples:
    All inputs should be numpy arrays, but are written as tuples for convenience
    >>> transform_coords((1,0), (0,0), (1,0))
    (1, 0)
    >>> transform_coords((1,0), (0,0), (1,1))
    (1, 1)

    Parameters
    ----------
    x : np.ndarray
        The coordinates to untransform
    centre : np.ndarray
        Coordinates (x,y) of the centre of the robot
    front : np.ndarray
        Coordinates (x,y) of the front of the robot

    Returns
    -------
    np.ndarray
        The transformed coordinates
    """
    
    df = front - centre
    # Transformation matrix of df to unit length x axis
    mat = np.array(((df[0], -df[1]), (df[1], df[0])))
    return np.matmul(mat, x) + centre

def get_CofR(centre: np.ndarray, front: np.ndarray):
    return untransform_coords(COFR_OFFSET, centre, front)

def get_true_front(centre: np.ndarray, front:np.ndarray):
    return untransform_coords(TRUE_FRONT_OFFSET, centre, front)

def _test_transform():
    x = np.array([1,0])
    c = np.array([2,1])
    f = np.array([3,1])
    result = transform_coords(x,c,f)
    print(f"transformed {result}")
    unresult = untransform_coords(result, c,f)
    print(f"untransformed {unresult}")

def _test_image():
    # load image
    # image = cv2.imread('checkerboard2/3.jpg') # No dots - shouldn't find any
    image = cv2.imread('dots/smol1.jpg') # Dots - should find them
    # image = cv2.imread('nav-test-fails/3.jpg') # Dots - should find them

    # process image
    image = undistort(image, 0.4)
    centres = getDots(image)
    print(centres)
    found, bbox = getDotBbox(centres)
    if found:
        drawMarkers(image, bbox, (255,0,0), True)
    
    # show image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _test_video():
    #video = VideoCapture('http://localhost:8081/stream/video.mjpeg')
    video = DotPatternVideo('http://localhost:8081/stream/video.mjpeg')

    while(1):
        image, found,centre,front = video.find()
        if found:
            cv2.circle(image, np.int0(centre), 4, (255,255,0), -1)
            cv2.circle(image, np.int0(front), 4, (255,255,0), -1)
            cv2.circle(image, np.int0(get_CofR(centre, front)), 4, (255,255,0), -1)
            cv2.circle(image, np.int0(untransform_coords(GATE_OFFSET, centre, front)), 4, (0,255,0), -1)
        # show image
        cv2.imshow('image', image)
        key = cv2.waitKey(1) & 0xFF
        # Exit if q pressed
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    
# return 2D vector perpendicular to input
def perp(a: np.ndarray):
    """Return a 2D vector which is perpendicular to the input

    Parameters
    ----------
    a : np.ndarray
        Input vector

    Returns
    -------
    np.ndarray
        A vector perpendicular to the input, in the cw direction
        if in a left-handed coordinate system (i.e. with OpenCV)
    """
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def angle(vec1: np.ndarray, vec2: np.ndarray):
    """Return the signed angle between two vectors between -pi and pi,
    in left-handed coordinates (which is what OpenCV uses)

    Parameters
    ----------
    vec1 : np.ndarray
        The starting vector
    vec2 : np.ndarray
        The final vector

    Returns
    -------
    float
        Angle of rotation of vec2 from vec1 in ccw direction 
        (i.e. +ve => vec2 is ccw from vec1)
    """
    # Angle between two vectors (in left handed coordinates)
    # +ve = vec2 is cw from vec1, -ve = vec2 is ccw from vec1
    mag_vec1 = np.linalg.norm(vec1)
    mag_vec2 = np.linalg.norm(vec2)
    dot = np.dot(vec1, vec2) / mag_vec1 / mag_vec2
    cross = np.cross(vec1, vec2)
    angle = np.arccos(np.clip(dot, -1, 1))
    if cross < 0:
        angle = -angle
    return angle

def undo_parallax(coord: np.ndarray, height=0.11):
    # The dot pattern on the robot will appear to be further away
    # from the centre than it really is, due to parallax.
    # Similar triangles: 
    # 10 cm dot pattern height, 1.5 m camera height
    # smaller triangle = 1.63 m * 'a' m, bigger triangle = (1.5 + 0.1) m * 'a' * (1.6/1.5) m
    # Simply scale about the centre of the image to undo parallax
    CAMERA_HEIGHT = 1.8 # 1.63
    parallax_scale = (CAMERA_HEIGHT - height) / CAMERA_HEIGHT
    parallax_shift = np.array((1016,760)) / 2
    shifted = coord - parallax_shift
    scaled = shifted * parallax_scale
    undone = scaled + parallax_shift
    return undone


class DotPatternVideo:
    filename: str
    video: VideoCapture

    balance = 0
    track_fail_count = 0
    
    def __init__(self, filename: str, balance: float = 0):
        """Dot pattern tracker class, designed to work on video streams

        Parameters
        ----------
        filename : str
            The filename (or http address) of the video stream
        balance : float
            How much of the black areas to include in the image (due to fisheye effects)
        """

        self.filename = filename
        self.video = VideoCapture(filename)
        self.balance = balance
    
    def find(self, annotate = True, shift: np.ndarray = None, invmat: np.ndarray = None):
        """Find the dot pattern in the latest video frame, and draw it on if found.
        Will also subtract any parallax from the coordinates.
        If shift and invmat are specified, will also crop to the board.

        Parameters
        ----------
        annotate : bool
            Whether to add annotations, default True
        shift : np.ndarray or None
            Shift found with get_shift_invmat_mat(), default None
        invmat : np.ndarray or None
            Invmat found with get_shift_invmat_mat(), default None

        Returns
        -------
        frame : np.ndarray
            The latest video frame after undistortion / cropping with FPS and pattern annotations
        found : bool
            Whether a valid pattern was found
        centre : np.ndarray or None
            Coordinates of the centre of the pattern, if found
        front : np.ndarray or None
            Coordinates of the front of the pattern, if found
        """

        timer = cv2.getTickCount()
        # Read a new frame
        frame = self.video.read()
        frame = undistort(frame, self.balance)
        if shift is not None and invmat is not None:
            frame = crop_board(frame, shift, invmat)
        found, bbox = getDotBbox(getDots(frame))
        centre = None
        front = None
        if found:
            centre, front = drawMarkers(frame, bbox, (255, 0, 0))
            # centre, front = undo_parallax(centre), undo_parallax(front)
        else: # not found
            annotate and cv2.putText(frame, "Dot pattern not detected", (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            self.track_fail_count += 1

        if annotate:
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            # Display fails on frame
            cv2.putText(frame, "fails : " + str(int(self.track_fail_count)), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        return frame, found, centre, front


if _DEBUG:
    # _test_transform()
    # _test_video()
    _test_image()
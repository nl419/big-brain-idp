import cv2
import numpy as np
from find_qr import drawMarkers
from unfisheye import undistort
from laggy_video import VideoCapture

# Flag to print debug information
_DEBUG = __name__ == "__main__"

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
    minLength = 35
    maxLength = 65
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
            if abs(cross) > 0.99: # <8 deg away from perp.
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
    hMin = 140; sMin = 42; vMin = 101; hMax = 166; sMax = 255; vMax = 255
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # Find contours of correct area
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    areas = []
    for c in cnts:
        M = cv2.moments(c)
        area = M['m00']
        _DEBUG and print(area)
        if area > 300 and area < 650:
            centre = (M['m10']/M['m00'], M['m01']/M['m00'])
            centre = np.array(centre)
            centres.append(centre)
            areas.append(area)
    return centres

def _testImage():
    # load image
    # image = cv2.imread('checkerboard2/3.jpg') # No dots - shouldn't find any
    image = cv2.imread('dots/dot12.jpg') # Dots - should find them

    # process image
    image = undistort(image)
    centres = getDots(image)
    print(centres)
    found, bbox = getDotBbox(centres)
    if found:
        drawMarkers(image, bbox, (255,0,0))
    
    # show image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _testVideo():
    #video = VideoCapture('http://localhost:8081/stream/video.mjpeg')
    video = DotPatternVideo('http://localhost:8081/stream/video.mjpeg')

    while(1):
        image, _,_,_ = video.find()
        # show image
        cv2.imshow('image', image)
        key = cv2.waitKey(1) & 0xFF
        # Exit if q pressed
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    

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
    
    def find(self):
        """Find the dot pattern in the latest video frame, and draw it on if found.

        Returns
        -------
        frame : np.ndarray
            The latest video frame with FPS and pattern annotations
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
        found, bbox = getDotBbox(getDots(frame))
        centre = None
        front = None
        isValid = False
        if found:
            centre, front = drawMarkers(frame, bbox, (255, 0, 0))
        else: # not found
            cv2.putText(frame, "Dot pattern not detected", (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            self.track_fail_count += 1

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Display fails on frame
        cv2.putText(frame, "fails : " + str(int(self.track_fail_count)), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        return frame, found, centre, front


if _DEBUG:
    # _testVideo()
    _testImage()
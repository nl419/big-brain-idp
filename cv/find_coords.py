# importing the module
import cv2
import numpy as np

_DEBUG = False and __name__ == "__main__"

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    image = params
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
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
        cv2.imshow('image', image)

def barrier_centres(image: np.ndarray):
    # Only pickup the yellow parts
    # Filter out small areas
    # Pick the two contours which are closest to the middle
    hMin = 17; sMin = 50; vMin = 160; hMax = 49; sMax = 255; vMax = 255
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Find contours of correct area
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, cnts, -1, (255,255,0), 3)
    middle = np.array((image.shape[1], image.shape[0])) / 2
    cv2.circle(image, np.int0(middle), 10, (0,255,0), 3)
    _DEBUG and cv2.imshow("image", image)
    _DEBUG and cv2.waitKey(0)
    centres = []
    areas = []
    for c in cnts:
        M = cv2.moments(c)
        area = M['m00']
        _DEBUG and print(area)
        if area > 800:
            centre = (M['m10']/M['m00'], M['m01']/M['m00'])
            centre = np.array(centre) - middle
            centres.append(centre)
            areas.append(area)
    # Grab the two closest contours to the middle
    mags = np.linalg.norm(centres, axis=1)
    args = mags.argsort()
    centres = np.take(centres, args[:2], axis=0)

    return centres + middle

from unfisheye import undistort

def _pick_points():
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
    # Return the points normalised by the barrier positions
    # image = cv2.imread('dots/dot4.jpg', 1)
    image = cv2.imread('checkerboard2/3.jpg', 1)
    image = undistort(image)

    centres = barrier_centres(image)
    shift = (centres[0] + centres[1]) / 2
    # c = bottom right barrier centroid - shift
    c = centres[0] - shift
    c = c if c[0] > 0 else centres[1] - shift
    mat = np.array(((c[0], c[1]), (-c[1], c[0]))) / (c[0]**2 + c[1]**2)
    print(centres)

    cv2.circle(image, np.int0(centres[0]), 5, (255,0,0), -1)
    cv2.circle(image, np.int0(centres[1]), 5, (255,0,0), -1)

    # displaying the image
    cv2.imshow('image', image)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event_normalised, (image, shift, mat))

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

    # dot4.jpg
    # real coords 712, 353   transformed coords 0.59870, -0.75853
    # real coords 602, 101   transformed coords -0.63577, -1.18904
    # real coords 344, 510   transformed coords -0.04266, 1.03243

    # checkerboard2/3.jpg
    # real coords 713, 354   transformed coords 0.60051, -0.75271
    # real coords 603, 101   transformed coords -0.63574, -1.18360
    # real coords 346, 509   transformed coords -0.04045, 1.02590


# driver function
if __name__=="__main__":
    # _pick_points()
    _pick_points_normalised()
"""Utility functions for finding QR codes"""

import cv2
import numpy as np

# Generate QR codes with https://barcode.tec-it.com/en/QRCode,
# or with the extension in InkScape (Extensions > Render > Barcode > QR Code...)

DECODE = False  # Whether to additionally decode the QR code
img = cv2.imread("qr_codes/small/rotated.png")
# img = cv2.imread("qr_codes/small/original.png")


def getQRData(img: np.ndarray, bbox: list, decoder: cv2.QRCodeDetector) -> str or None:
    """Return the data within a QR code, if valid.

    This function can be used to check that the correct QR code was found, although
    this runs much slower than getQRShape, so should only be used for more stringent
    checks.
    Occasionally, cv2 will detect a QR code that isn't there, or slightly miss the
    true corners of the QR code, meaning the data cannot be decoded properly.

    Args
    -------
    img : np.ndarray
        The image in NumPy array form
    bbox : list
        A list of four pairs of (x,y) coordinates of the corners of the QR code
    decoder : cv2.QRCodeDetector
        Any QRCodeDetector object

    Returns
    -------
    data : str or None
        The data contained in the QR code if found, else None
    """
    data, _ = decoder.decode(img, bbox)
    return data


def getQRShape(points: list):
    """Returns the shape properties of a 4 point bounding box.

    Parameters
    ----------
    points : list
        List of (x,y) coordinates of the bounding box vertices

    Returns
    -------
    area : float
        The area enclosed by the bounding box
    minDotp : float
        The minimum dot product between opposite sides
    """

    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

    points = np.insert(points, 4, points[0], axis=0)
    lines = np.diff(points, axis=0)                 # Lines joining each point
    # Magnitudes of each of the lines
    mags = np.linalg.norm(lines, axis=1)

    dotps = np.zeros(2)
    for i in range(2):
        dotps[i] = np.abs(np.dot(lines[i], lines[(i+2) % 4])
                          ) / mags[i] / mags[(i+2) % 4]
    minDotp = np.min(dotps)
    return area, minDotp


def drawMarkers(img: np.ndarray, points: "list[int]", lineColour: "list[int]"):
    """Draws markers on the image for a QR code bounding box.
    Also draws the forward-facing line

    Parameters
    ----------
    img : np.ndarray
        The image to draw onto
    points : list[int]
        List of (x,y) coordinates of the bounding box vertices
    lineColour : list[int]
        Colour (B,G,R) of lines joining the vertices
    Returns
    -------
    centre : np.ndarray 
        Centre of bounding box
    front : np.ndarray
        Front of bounding box
    """
    x = points.astype(int)
    n = len(x)
    for j in range(n):
        p1 = tuple(x[j])
        p2 = tuple(x[(j+1) % n])
        cv2.line(img, p1, p2, lineColour, 3)

    centre = np.mean(x, axis=0).astype(int)
    top_midpoint = np.mean(x[0:2], axis=0).astype(int)
    marker_radius = 5
    cv2.circle(img, centre, radius=marker_radius,
               color=(0, 0, 255), thickness=-1)
    cv2.circle(img, top_midpoint, radius=marker_radius,
               color=(0, 0, 255), thickness=-1)
    cv2.line(img, centre, top_midpoint, color=(0, 0, 255), thickness=3)
    return centre, top_midpoint


if __name__ == "__main__":
    qrDecoder = cv2.QRCodeDetector()
    if DECODE:
        # Detect and decode the qrcode
        data, bbox, rectifiedImage = qrDecoder.detectAndDecode(img)
        if len(data) > 0:
            print("Decoded Data: {}".format(data))
            drawMarkers(img, bbox, (255, 0, 0))
        else:
            print("QR Code not detected")
        cv2.imshow("Results", img)
    else:
        # Detect the qrcode
        found, bbox = qrDecoder.detect(img)
        if found:
            print("Found QR code")
            print(bbox)
            drawMarkers(img, bbox, (255, 0, 0))
        else:
            print("QR Code not detected")
        cv2.imshow("Results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""Utility functions for finding QR codes"""

import cv2
import numpy as np

# Generate QR codes with https://barcode.tec-it.com/en/QRCode,
# or with the extension in InkScape (Extensions > Render > Barcode > QR Code...)


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


from laggy_video import VideoCapture
from unfisheye import undistort

class QRVideo:
    filename: str
    video: VideoCapture
    decoder = cv2.QRCodeDetector()
    global_scale: float     # How much to scale the frame globally. 2 seems to work best.
    crop_scale: float       # How much to additionally scale the frame after cropping. 1 seems to work best.
    crop_radius: int

    lastCentre = np.zeros(2).astype(int)
    front = np.array((100,100)).astype(int)
    track_timeout = 5  # How many frames of consecutive tracking failure before resetting lastCentre
    track_fail_count = track_timeout # Always reset on first frame
    target = lastCentre
    centre = lastCentre

    balance = 0
    
    def __init__(self, filename: str, balance: float = 0, global_scale: float = 2, crop_scale: float = 1, crop_radius: int = 150):
        """QR Tracker class, designed to work on a video stream

        Parameters
        ----------
        filename : str
            The filename (or http address) of the video stream
        balance : float
            How much of the black areas to include in the image (due to fisheye effects)
        global_scale : float, optional
            How much to scale the image before finding the QR code, by default 2
        crop_scale : float, optional
            How much to scale the image after cropping, before finding the QR code, by default 1
        crop_radius : int, optional
            Radius of the cropping bounding box, by default 150
        """

        self.filename = filename
        self.video = VideoCapture(filename)
        self.global_scale = global_scale
        self.crop_scale = crop_scale
        self.crop_radius = crop_radius
        self.balance = balance
    
    def find(self, use_crop: bool = True):
        """Find the QR code in the latest video frame

        Parameters
        ----------
        use_crop : bool, optional
            Whether to crop into the last known location of the QR code, by default True

        Returns
        -------
        frame : np.ndarray
            The latest video frame with FPS and QR code annotations
        found : bool
            Whether a valid QR code was found
        centre : np.ndarray or None
            Coordinates of the centre of the QR code, if found
        front : np.ndarray or None
            Coordinates of the front of the QR code, if found
        """

        timer = cv2.getTickCount()
        # Read a new frame
        frame = self.video.read()
        OLDDIM = np.array([frame.shape[1], frame.shape[0]]).astype(int)
        DIM = np.int0(OLDDIM * self.global_scale)
        frame = cv2.resize(frame, DIM)
        frame = undistort(frame, self.balance)

        # Return foundBool, centre, front, all in the unscaled coordinates
        transform = [0,0]
        scale = 1
        if use_crop:
            CROP_RADIUS = self.crop_radius * self.global_scale # Crop radius in global coordinates
            # Crop and track QR code
            if self.track_fail_count < self.track_timeout:
                x_clipped = np.array([self.lastCentre[0] - CROP_RADIUS, self.lastCentre[0] + CROP_RADIUS]).astype(int)
                y_clipped = np.array([self.lastCentre[1] - CROP_RADIUS, self.lastCentre[1] + CROP_RADIUS]).astype(int)
                # Limit maximum x and y
                x_clipped = np.clip(x_clipped, 0, DIM[0] - 1)
                y_clipped = np.clip(y_clipped, 0, DIM[1] - 1)

                # Crop the image
                qrframe = frame[y_clipped[0]:y_clipped[1],
                                x_clipped[0]:x_clipped[1]]

                # Scale the cropped image
                crop_dim = np.array([x_clipped[1] - x_clipped[0], y_clipped[1] - y_clipped[0]]) * self.crop_scale
                crop_dim = np.int0(crop_dim)
                qrframe = cv2.resize(qrframe, crop_dim)

                # Update data for going from cropped coords to global coords
                transform = [x_clipped[0], y_clipped[0]]
                scale = self.crop_scale
            else:
                qrframe = frame
            # Search for QR code in (potentially cropped) frame
            found, bbox = self.decoder.detect(qrframe)
        else: # No cropping
            found, bbox = self.decoder.detect(frame)
        
        centre = None
        front = None
        isValid = False
        if found:
            bbox = bbox[0]  # bbox is always a unit length list, so just grab the first element
            # now bbox is a list of 4 vertices relative to cropped coords, so transform them
            for i in range(len(bbox)):
                bbox[i] /= scale
                bbox[i,0] += transform[0]
                bbox[i,1] += transform[1]
            
            # check validity
            shape_data = getQRShape(bbox)
            max_area = 5000 * (self.global_scale)**2
            min_area = 2500 * (self.global_scale)**2
            isValid = shape_data[0] < max_area and shape_data[0] > min_area and shape_data[1] > 0.98
            # text_data = getQRData(frame, bbox, qrDecoder)
            # isValid = text_data == "bit.ly/3tbqjqL"

            # Draw blue border if valid, green otherwise.
            if isValid:
                centre, front = drawMarkers(frame, bbox, (255, 0, 0))
                centre = centre / self.global_scale
                front = front / self.global_scale
            else:
                drawMarkers(frame, bbox, (0, 255, 0))

            # update position estimate and failure counter
            if use_crop:
                if isValid:
                    self.lastCentre = np.mean(bbox, axis=0).astype(int)
                    self.track_fail_count = 0
                else:
                    self.track_fail_count += 1
        else: # not found
            cv2.putText(frame, "QR code not detected", (100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            self.track_fail_count += 1

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Display fails on frame
        cv2.putText(frame, "fails : " + str(int(self.track_fail_count)), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        frame = cv2.resize(frame, OLDDIM)

        # return qrframe, found and isValid, centre, front
        return frame, found and isValid, centre, front

def QR_finder_better(frame: np.ndarray):
    # It's not actually better...
    k = (7,7)
    thresh = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, k)
    thresh = cv2.dilate(thresh, k, iterations=10)
    thresh = cv2.blur(thresh, k)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (10,10), iterations=5)
    cv2.imshow("tawoyiaw", thresh)
    cv2.waitKey(0)

def _find_decode_test():
    """Test the finding and/or decoding of a QR code
    """
    DECODE = False  # Whether to additionally decode the QR code
    img = cv2.imread("qr_codes/small/rotated.png")
    # img = cv2.imread("qr_codes/small/original.png")

    # Find / decode a single QR code
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

def _QRVideo_test():
    finder = QRVideo('http://localhost:8081/stream/video.mjpeg', 0, 2, 1)
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    while 1:
        frame, _, _, _ = finder.find()
        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        # Exit if q pressed
        if key == ord('q'):
            break

if __name__ == "__main__":
    # _find_decode_test()
    # _QRVideo_test()
    frame = cv2.imread("qr_codes/1.jpg")
    QR_finder_better(frame)

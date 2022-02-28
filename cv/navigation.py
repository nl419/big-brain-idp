"""Generate motor commands to move from point to point.
Reads video stream from idpcam2, finds robot location & orientation,
calculates motor commands to arrive at target location.

Reset target to robot centre with ENTER key."""

import numpy as np
import cv2


FORWARD = np.array((255,255))
BACKWARD = -FORWARD
LEFT = np.array((-100, 100))
RIGHT = -LEFT

def go_to_coord (start: np.ndarray, end: np.ndarray, front: np.ndarray,
                 smallTurnThresh: float = 0.2, largeTurnThresh: float = 0.5):
    """Generate the next motor commands for the robot given the start coords, end coords,
    and the coords of the front of the robot.

    Examples:
    >>> go_to_coord((0,0), (5,0), (1,0))
    np.array((255,255))  # Forward
    >>> go_to_coord((0,0), (5,0), (0,1))
    np.array((-100,100)) # Stationary left turn

    Parameters
    ----------
    start : np.ndarray
        Starting coordinates (x,y) of the centre of the robot
    end : np.ndarray
        Ending coordinates (x,y) of the centre of the robot
    front : np.ndarray
        Coordinates (x,y) of the front of the robot (used for orientation)
    smallTurnThresh : float 
        Threshold 0->1 for turning while moving
    largeTurnThresh : float
        Threshold 0->1 for turning on the spot. Should be larger than smallTurnThresh.
    """

    orientation = front - start
    displacement = end - start
    orient_mag = np.linalg.norm(orientation)
    disp_mag = np.linalg.norm(displacement)

    # Rotation
    crossp = np.cross(orientation, displacement) / disp_mag / orient_mag
    doTurn = abs(crossp) > smallTurnThresh
    rotation = LEFT if crossp < 0 else RIGHT

    # Translation
    smallMoveThresh = 10
    fullMoveThresh = 50
    
    doMove = abs(crossp) < largeTurnThresh and disp_mag > smallMoveThresh
    if np.dot(orientation, displacement) > 0:
        translation = FORWARD
    else:
        translation = BACKWARD
        rotation = -rotation
    if disp_mag < fullMoveThresh: 
        doMove = doMove and (not doTurn)
        translation = translation / 2

    # Generate command
    command = np.zeros(2)
    ### DEBUG: don't attempt to turn around if near point
    if not doMove: return command.astype(int)
    if doTurn: command += rotation
    if doMove: command += translation
    return np.clip(command, -255, 255).astype(int)


start = np.array((0,0))
end = np.array((20,0))
front = np.array((0.95,0.2))


import urllib.request
ip = "192.168.urmom.urdad"
command = go_to_coord(start, end, front)
getString = ip + "/TRIGGER/" + str(command[0]) + "/" + str(command[1]) + "///"
print(getString)


from laggy_video import VideoCapture
from unfisheye import undistort
from find_qr import *

DIM = (np.array((1016, 760))).astype(int)
USE_CROP = False
CROP_RADIUS = 400 # radius around last known point for cropping, used when TRACKER = False

if __name__ == "__main__":
    video = VideoCapture('http://localhost:8081/stream/video.mjpeg')
    qrDecoder = cv2.QRCodeDetector()
    lastCentre = np.zeros(2).astype(int)
    track_timeout = 5  # How many frames of consecutive tracking failure before resetting lastCentre
    track_fail_count = track_timeout # Always reset on first frame

    target = DIM // 2
    centre = DIM // 2
    front = np.ndarray((0,0)).astype(int)
    getString = ip + "/"
    lastString = ""
    while True:
        # Read a new frame
        frame = video.read()
        frame = undistort(frame, 0.4)
        # Start timer
        timer = cv2.getTickCount()

        transform = [0,0] # Transformation from cropped coords to real coords
        if USE_CROP:
            # Crop and track QR code
            if track_fail_count < track_timeout:
                qrframe = frame[lastCentre[1] - CROP_RADIUS:lastCentre[1] + CROP_RADIUS,
                                lastCentre[0] - CROP_RADIUS:lastCentre[0] + CROP_RADIUS]
                transform = [lastCentre[0] - CROP_RADIUS, lastCentre[1] - CROP_RADIUS]
            else:
                qrframe = frame
            # Search for QR code in (potentially cropped) frame
            found, bbox = qrDecoder.detect(qrframe)
        else: # No cropping
            found, bbox = qrDecoder.detect(frame)

        if found:
            bbox = bbox[0]  # bbox is always a unit length list, so just grab the first element
            # now bbox is a list of 4 vertices relative to cropped coords, so transform them
            for i in range(len(bbox)):
                bbox[i,0] += transform[0]
                bbox[i,1] += transform[1]
            
            # check validity
            shape_data = getQRShape(bbox)
            isValid = shape_data[0] < 100**2 and shape_data[0] > 20**2 and shape_data[1] > 0.98
            # text_data = getQRData(frame, bbox, qrDecoder)
            # isValid = text_data == "bit.ly/3tbqjqL"

            # Draw blue border if valid, green otherwise.
            if isValid:
                centre, front = drawMarkers(frame, bbox, (255, 0, 0))
            else:
                drawMarkers(frame, bbox, (0, 255, 0))

            # update position estimate and failure counter
            if USE_CROP:
                if isValid:
                    lastCentre = np.mean(bbox, axis=0).astype(int)
                    track_fail_count = 0
                else:
                    track_fail_count += 1
        else: # not found
            cv2.putText(frame, "QR code not detected", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            track_fail_count += 1
        
        # Get motor commands
        command = go_to_coord(centre, target, front)
        getString = ip + "/TRIGGER/" + str(command[0]) + "/" + str(command[1]) + "///"

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Display fails on frame
        cv2.putText(frame, "fails : " + str(int(track_fail_count)), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Display motor command and target
        cv2.putText(frame, getString, (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.circle(frame, target, 5, (255,255,0), -1)

        # Send command
        if lastString != getString:
            urllib.request.urlopen(getString)
            print("sending new command")
            lastString = getString
        
        # cv2.rectangle(frame, (1,1), (DIM[0] - 2, DIM[1] - 2), (0,0,255), 2)
        cv2.imshow("Tracking", frame)
        # cv2.imshow("Tracking", qrframe)

        key = cv2.waitKey(1) & 0xFF
        # Exit if q pressed
        if key == ord('q'):
            break
        # If ENTER pressed, reset target
        if key == 13:
            target = centre
            
    cv2.destroyAllWindows()

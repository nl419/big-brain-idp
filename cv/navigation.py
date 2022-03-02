"""Generate motor commands to move from point to point.
Reads video stream from idpcam2, finds robot location & orientation,
calculates motor commands to arrive at target location.

Reset target to robot centre with ENTER key."""

import numpy as np
import cv2


FORWARD = np.array((-255,-255))
BACKWARD = -FORWARD
LEFT = np.array((100, -100))
RIGHT = -LEFT

# 100 = 1, 255 = 4 => 50 ~= 0
STALL_SPEED = 50        # Maximum motor speed command which produces zero rotation
MOVEMENT_SPEED = 44     # Forward/Backward movement speed in px/s on unscaled image
ROTATION_SPEED = 2 * np.pi / 18.83      # Radians per second

def get_precise_translation(distance: float, scale: float = 1):
    """Get a motor command and duration in order to execute a precise translation

    Parameters
    ----------
    distance : float
        The distance to travel in the forward direction in px
    scale : float, optional
        How much the video has been scaled from (1016,760), by default 1

    Returns
    -------
    commands : np.ndarray
        The motor commands to be sent
    time : float
        The duration of time for the command to be run in seconds
    """
    
    distance /= scale
    translation = FORWARD
    if distance < 0:
        translation = BACKWARD
    
    return translation, distance / MOVEMENT_SPEED

def get_precise_rotation (orientation: np.ndarray, target_orientation: np.ndarray, backwardOk: bool = True):
    """Get a motor command and duration in order to execute a precise rotation

    Parameters
    ----------
    orientation : np.ndarray
        A vector pointing in the forward direction of the robot
    target_orientation : np.ndarray
        A target vector to be parallel with
    backwardOk : bool, optional
        Whether being aligned with the backward direction is ok, by default True

    Returns
    -------
    commands : np.ndarray
        The motor commands to be sent
    time : float
        The duration of time for the command to be run in seconds
    """

    cross = np.cross(orientation, target_orientation)
    dot = np.dot(orientation, target_orientation) / np.linalg.norm(orientation) / np.linalg.norm(target_orientation)

    rotation = LEFT if cross < 0 else RIGHT
    if backwardOk and dot < 0:
        rotation = RIGHT if cross < 0 else LEFT
        angle = np.arccos(-dot)
    else:
        angle = np.arccos(dot)
    
    return rotation, angle / ROTATION_SPEED
    


def go_to_coord (start: np.ndarray, end: np.ndarray, front: np.ndarray,
                 pixelScale: float = 1,
                 smallTurnThresh: float = 0.2, largeTurnThresh: float = 0.5,
                 smallMoveThresh: int = 10, largeMoveThresh: int = 50):
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
    pixelScale : float
        Global scaling factor for pixel-based thresholds and speeds
    smallTurnThresh : float 
        Threshold 0->1 for turning while moving
    largeTurnThresh : float
        Threshold 0->1 for turning on the spot. Should be larger than smallTurnThresh.
    smallMoveThresh : int
        Threshold in px of distance between start and end for a small movement to be made
    largeMoveThresh : int
        Threshold in px of distance between start and end for a large movement to be made
    """

    smallMoveThresh *= pixelScale
    largeMoveThresh *= pixelScale

    orientation = front - start
    displacement = end - start
    orient_mag = np.linalg.norm(orientation)
    disp_mag = np.linalg.norm(displacement)

    # Rotation
    crossp = np.cross(orientation, displacement) / disp_mag / orient_mag
    doTurn = abs(crossp) > smallTurnThresh
    rotation = LEFT if crossp < 0 else RIGHT
    
    doMove = abs(crossp) < largeTurnThresh and disp_mag > smallMoveThresh
    if np.dot(orientation, displacement) > 0:
        translation = FORWARD
    else:
        translation = BACKWARD
        rotation = -rotation
    if disp_mag < largeMoveThresh: 
        doMove = doMove and (not doTurn)
        translation = translation / 2

    # Generate command
    command = np.zeros(2)
    ### DEBUG: don't attempt to turn around if near point
    if disp_mag < smallMoveThresh: return command.astype(int)
    if doTurn: command += rotation
    if doMove: command += translation
    return np.clip(command, -255, 255).astype(int)


start = np.array((0,0))
end = np.array((20,0))
front = np.array((0.95,0.2))


import urllib.request
ip = "http://192.168.137.28"
command = go_to_coord(start, end, front)
getString = ip + "/TRIGGER/" + str(command[0]) + "/" + str(command[1]) + "///"
print(getString)
# urllib.request.urlopen(getString)


from laggy_video import VideoCapture
from unfisheye import undistort
from find_qr import *
GLOBAL_SCALE = 2
DIM = (np.array((1016, 760)) * GLOBAL_SCALE).astype(int)
USE_CROP = True
CROP_RADIUS = 200 # radius around last known point for cropping, used when TRACKER = False
CROP_SCALE = 1.5

if __name__ == "__main__":
    video = VideoCapture('http://localhost:8081/stream/video.mjpeg')
    qrDecoder = cv2.QRCodeDetector()
    lastCentre = np.zeros(2).astype(int)
    track_timeout = 5  # How many frames of consecutive tracking failure before resetting lastCentre
    track_fail_count = track_timeout # Always reset on first frame

    target = DIM // 2
    centre = DIM // 2
    front = np.array((100,100)).astype(int)
    print(front)
    getString = ip + "/"
    lastString = ""
    while True:
        # Read a new frame
        frame = video.read()
        frame = cv2.resize(frame, DIM)
        frame = undistort(frame, 0.4)
        win = cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

        # print(frame.shape)
        # Start timer
        timer = cv2.getTickCount()

        transform = [0,0] # Transformation from cropped coords to real coords
        scale = 1
        if USE_CROP:
            # Crop and track QR code
            if track_fail_count < track_timeout:
                x_clipped = np.array([lastCentre[0] - CROP_RADIUS, lastCentre[0] + CROP_RADIUS]).astype(int)
                y_clipped = np.array([lastCentre[1] - CROP_RADIUS, lastCentre[1] + CROP_RADIUS]).astype(int)
                # Limit maximum x and y
                x_clipped = np.clip(x_clipped, 0, DIM[0] - 1)
                y_clipped = np.clip(y_clipped, 0, DIM[1] - 1)

                qrframe = frame[y_clipped[0]:y_clipped[1],
                                x_clipped[0]:x_clipped[1]]
                transform = [x_clipped[0], y_clipped[0]]
                crop_dim = np.array([x_clipped[1] - x_clipped[0], y_clipped[1] - y_clipped[0]]) * CROP_SCALE
                crop_dim = np.int0(crop_dim)
                scale = CROP_SCALE
                qrframe = cv2.resize(qrframe, crop_dim)
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
                bbox[i] /= scale
                bbox[i,0] += transform[0]
                bbox[i,1] += transform[1]
            
            # check validity
            shape_data = getQRShape(bbox)
            # print(shape_data)
            isValid = shape_data[0] < 20000 and shape_data[0] > 10000 and shape_data[1] > 0.98
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
        # print(centre)
        # print(front)
        command = go_to_coord(centre, target, front, scale * GLOBAL_SCALE)
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
            # urllib.request.urlopen(getString)
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
            print(target)
            
    cv2.destroyAllWindows()

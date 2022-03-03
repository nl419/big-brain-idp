"""Generate motor commands to move from point to point.
Reads video stream from idpcam2, finds robot location & orientation,
calculates motor commands to arrive at target location.

Reset target to robot centre with ENTER key."""

import numpy as np
import cv2


FORWARD = np.array((-250,-255))
BACKWARD = -FORWARD
LEFT = np.array((100, -100))
RIGHT = -LEFT

# 100 = 1, 255 = 4 => 50 ~= 0
STALL_SPEED = 50        # Maximum motor speed command which produces zero rotation
MOVEMENT_SPEED = 44     # Forward/Backward movement speed in px/s on unscaled image
ROTATION_SPEED = 2 * np.pi / 18      # Radians per second

# Hardcoded points:
# TODO: recalibrate points automatically 
# (maybe find four easy points, and find mapping to those points)
PICKUP_BBOX = np.array(((131, 629), (200, 713), (283, 642), (211, 556)))
RED_DROPOFFS = np.array(((712, 353), (786, 285)))
BLUE_DROPOFFS = np.array(((603, 101), (532, 165)))
BRIDGE_POINTS = np.array(((344, 509), (657, 221)))

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
    
    return translation, abs(distance) / MOVEMENT_SPEED

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
    
def go_to_coord_timed (start: np.ndarray, end: np.ndarray, front: np.ndarray):
    turnThresh = 5 * np.pi / 180 # 5 degrees
    moveThresh = 5 # in pixels

    orientation = front - start
    displacement = end - start
    orient_mag = np.linalg.norm(orientation)
    disp_mag = np.linalg.norm(displacement)
    cross = np.cross(orientation, displacement) / orient_mag / disp_mag
    dot = np.dot(orientation, displacement) / orient_mag
    print("Inside:")
    print(cross)
    print(dot)
    print()
    if disp_mag > moveThresh:
        if abs(cross) > turnThresh:
            return get_precise_rotation(orientation, displacement)
        else:
            return get_precise_translation(dot)
    return (0,0),0


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
ip = "http://192.168.137.188"
command = go_to_coord(start, end, front)
getString = ip + "/TRIGGER/" + str(command[0]) + "/" + str(command[1]) + "///"
print(getString)
# urllib.request.urlopen(getString)


from laggy_video import VideoCapture
from unfisheye import undistort
from find_qr import *
from find_dots import DotPatternVideo
GLOBAL_SCALE = 2
DIM = (np.array((1016, 760)) * GLOBAL_SCALE).astype(int)
USE_CROP = True
CROP_RADIUS = 200 # radius around last known point for cropping, used when TRACKER = False
CROP_SCALE = 1.5

SEND_COMMANDS = False # whether to attempt to send commands
MIN_COMMAND_INTERVAL = 500

import time

if __name__ == "__main__":
    #video = QRVideo('http://localhost:8081/stream/video.mjpeg', 0, 2.5)
    video = DotPatternVideo('http://localhost:8081/stream/video.mjpeg')
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    old_centre = np.array([100,100])
    old_front = np.array([200,200])
    TOL_STATIONARY = 1
    target = old_centre

    getString = ip + "/"
    lastString = ""
    next_command_time = 0
    while True:
        frame, found, centre, front = video.find()
        if found:
            # Translate the centre back a little bit
            orientation = front - centre
            centre = centre - orientation

            distance = np.linalg.norm(old_centre - centre) # How far the bot has moved
            distance = max(distance, np.linalg.norm(front - old_front)) # How much has the bot rotated
            old_centre = centre
            old_front = front
            current_time = round(time.time() * 1000)
            if distance < TOL_STATIONARY and current_time > next_command_time: # if it isn't moving
                print("Getting new command")
                command, duration = go_to_coord_timed(centre, target, front)
                duration *= 1000 # Turn s into ms
                duration = int(duration)
                getString = ip + "/TRIGGER/" + str(command[0]) + "/" + str(command[1]) + "//" + str(duration) + "/"

            if lastString != getString:
                SEND_COMMANDS and urllib.request.urlopen(getString)
                print("sending new command")
                print(getString)
                lastString = getString
                next_command_time = current_time + MIN_COMMAND_INTERVAL
        cv2.circle(frame, target, 5, (255,255,0), -1)
        cv2.imshow("Tracking", frame)
        # cv2.imshow("Tracking", qrframe)

        key = cv2.waitKey(1) & 0xFF
        # Exit if q pressed
        if key == ord('q'):
            break
        # If ENTER pressed, reset target
        if key == 13:
            target = np.int0(old_centre)
            print(target)
                
    cv2.destroyAllWindows()
"""Generate motor commands to move from point to point.
Reads video stream from idpcam2, finds robot location & orientation,
calculates motor commands to arrive at target location.

With _test_go_to_target(), press ENTER to reset target to current location.
With _test_go_loop(), watch it as it goes back and forth over the bridge!"""

import numpy as np
import cv2
# For the example cases at bottom
import urllib.request
from find_qr import *
from find_dots import DotPatternVideo
import time
import math


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
PICKUP_BBOX = np.array(((131, 629), (200, 713), (283, 642), (211, 556))).astype(int)
RED_DROPOFFS = np.array(((712, 353), (786, 285))).astype(int)
BLUE_DROPOFFS = np.array(((603, 101), (532, 165))).astype(int)
BRIDGE_POINTS = np.array(((344, 509), (657, 221))).astype(int)
HOME = np.array((782, 108))

def draw_waypoints(image: np.ndarray):
    for p in PICKUP_BBOX:
        cv2.circle(image, p, 3, (0,128,128), thickness=2)
    for p in RED_DROPOFFS:
        cv2.circle(image, p, 3, (0,0,128), thickness=2)
    for p in BLUE_DROPOFFS:
        cv2.circle(image, p, 3, (128,0,0), thickness=2)
    for p in BRIDGE_POINTS:
        cv2.circle(image, p, 3, (128,128,0), thickness=2)
    return image

def get_precise_translation(distance: float, thresh: float = 1, duration_thresh: float = 3, scale: float = 1):
    """Get a motor command and duration in order to execute a precise translation

    Parameters
    ----------
    distance : float
        The distance to travel in the forward direction in px
    thresh : float
        Threshold for a movement to be commanded in px
    duration_thresh : float
        Duration threshold in seconds. If duration > duration_thresh, duration *= 0.8
    scale : float, optional
        How much the video has been scaled from (1016,760), by default 1

    Returns
    -------
    commands : np.ndarray or None
        The motor commands to be sent if needed, None otherwise
    time : float
        The duration of time for the command to be run in seconds
    """
    
    distance /= scale
    if abs(distance) < thresh:
        return None, 0
    translation = FORWARD
    if distance < 0:
        translation = BACKWARD
    
    duration = abs(distance) / MOVEMENT_SPEED
    if duration > duration_thresh:
        duration *= 0.8

    return translation, duration

def get_precise_rotation (orientation: np.ndarray, target_orientation: np.ndarray, backwardOk: bool = True,
                          angle_thresh: float = 5, dist_thresh: float = 3):
    """Get a motor command and duration in order to execute a precise rotation

    Parameters
    ----------
    orientation : np.ndarray
        A vector pointing in the forward direction of the robot
    target_orientation : np.ndarray
        A target vector to be parallel with
    backwardOk : bool, optional
        Whether being aligned with the backward direction is ok, by default True
    angle_thresh : float
        Threshold for a rotation to be commanded in degrees
    dist_thresh : float
        Minimum size of target orientation in px to cause a rotation

    Returns
    -------
    commands : np.ndarray or None
        The motor commands to be sent if needed, None otherwise
    time : float
        The duration of time for the command to be run in seconds
    """
    thresh_dot = np.cos(np.radians(angle_thresh))

    cross = np.cross(orientation, target_orientation)
    dot = np.dot(orientation, target_orientation) / np.linalg.norm(orientation) / np.linalg.norm(target_orientation)
    dot = np.clip(dot, -1, 1)

    if np.linalg.norm(target_orientation) < dist_thresh: # Too close
        return None, 0

    if backwardOk and abs(dot) > thresh_dot or dot > thresh_dot: # Already aligned
        return None, 0

    rotation = LEFT if cross < 0 else RIGHT
    if backwardOk and dot < 0:
        rotation = RIGHT if cross < 0 else LEFT
        angle = np.arccos(-dot)
    else:
        angle = np.arccos(dot)
    
    return rotation, angle / ROTATION_SPEED

def go_to_coord_timed (start: np.ndarray, end: np.ndarray, front: np.ndarray,
                       backwardOk: bool = True, dist_thresh: float = 5, 
                       angle_thresh: float = 5):
    """Return the command and duration to go to end

    Parameters
    ----------
    start : np.ndarray
        Coordinates (x,y) of the start point
    end : np.ndarray
        Coordinates (x,y) of the end point
    front : np.ndarray
        Coordinates (x,y) of the front of the robot (orientation = front - start)
    backwardOk : bool, optional
        Whether reversing is ok, by default True
    dist_thresh : float, optional
        Distance threshold for being "at" end, by default 5
    angle_thresh : float, optional
        Angle threshold for being "at" target orientation in degrees, by default 5

    Returns
    -------
    commands : np.ndarray or None
        The motor commands to be sent if needed, None otherwise
    time : float
        The duration of time for the command to be run in seconds
    """

    orientation = front - start
    displacement = end - start
    orient_mag = np.linalg.norm(orientation)
    dot = np.dot(orientation, displacement) / orient_mag
    # Try to rotate first, then try translation, then give up.
    command, duration = get_precise_rotation(orientation, displacement, backwardOk, angle_thresh)
    if command is None:
        command, duration = get_precise_translation(dot, dist_thresh)
    return command, duration

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

def centre_of_rotation (a0: np.ndarray, b0: np.ndarray, a1: np.ndarray, b1: np.ndarray):
    """Returns the centre of rotation for a pair of initial and final coords
    for two points
    
    Examples
    --------
    All inputs should be numpy arrays, but are written as tuples for convenience
    >>> centre_of_rotation((0,0), (1,0), (0,0), (0,1))
    (0,0)   # Centre of rotation is the origin

    Parameters
    ----------
    a0 : np.ndarray
        Coordinates (x,y) of the 1st point before the motion
    b0 : np.ndarray
        Coordinates (x,y) of the 2nd point before the motion
    a1 : np.ndarray
        Coordinates (x,y) of the 1st point after the motion
    b1 : np.ndarray
        Coordinates (x,y) of the 2nd point after the motion

    Returns
    -------
    CofR : np.ndarray
        Coordinates (x,y) of the centre of rotation
    """
    
    # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    # return 2D vector perpendicular to input
    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    # return intersections of line segments a and b, each defined with 2 points
    def seg_intersect(w1,w2, v1,v2) :
        dw = w2-w1
        dv = v2-v1
        dp = w1-v1
        dwp = perp(dw)
        denom = np.dot(dwp, dv)
        num = np.dot(dwp, dp)
        return (num / denom.astype(float))*dv + b1
    
    # Tolerance for a "zero" length vector
    LENGTH_TOL = 1e-2

    da = a1 - a0
    db = b1 - b0

    # Get points defining the line perpendicular to each of da and db
    ahalf = (a0 + a1) / 2
    aperp = ahalf + perp(da)
    bhalf = (b0 + b1) / 2
    bperp = bhalf + perp(db)

    # handle edge cases of zero length
    if np.linalg.norm(da) < LENGTH_TOL:
        c = ahalf
    elif np.linalg.norm(db) < LENGTH_TOL:
        c = bhalf
    else: # all vectors are well behaved. find the centre.
        c = seg_intersect(ahalf, aperp, bhalf, bperp)
    
    return c

# If I know the centre of rotation for a few different moves,
# Then I can plot a curved course, and reach a particular destination & orientation simultaneously.
# So I need to measure a few centres of rotation for different moves
# Then lerp between them.
# Could do the same measurements for movement speed

# To plot curved course: 
    # Get current location (of robot's own centre of rotation), 
    # current orientation, target location.
    # lines = (cloc - corient, cloc + corient), (cloc, tloc)
    # c = get centre of rotation for lines
    # vecs = (middle(line0) - c, middle(line1) - c)
    # angle = signed angle between vecs

# This is much too complicated, for only minor improvements.
# The main issue, is if the robot wants to get to a certain place & orientation,
# But it misses the place slightly, then it will make a big turn to get there,
# then have to turn around again to match the desired orientation.

# Instead, should match orientation, reverse a little bit, 
# make a smaller turn to the place, go to place,
# then make a small turn to correct orientation.

def go_to_coord_orient (start: np.ndarray, end: np.ndarray, front: np.ndarray, 
                        target_orient : np.ndarray, at_thresh: float = 3,
                        near_thresh: float = 20, angle_thresh: float = 3):
    """Return the command and duration to go to end and align with target_orient

    Parameters
    ----------
    start : np.ndarray
        Coordinates (x,y) of the start point
    end : np.ndarray
        Coordinates (x,y) of the end point
    front : np.ndarray
        Coordinates (x,y) of the front of the robot (orientation = front - start)
    target_orient : np.ndarray
        Target orientation to be aligned to
    at_thresh : float, optional
        Distance threshold for "at" end, by default 3
    near_thresh : float, optional
        Distance threshold for "near" end, by default 20
    angle_thresh : float, optional
        Angle threshold for "at" target orientation in degrees, by default 3

    Returns
    -------
    commands : np.ndarray or None
        The motor commands to be sent if needed, None otherwise
    time : float
        The duration of time for the command to be run in seconds
    """

    # If at end
        # Align with target_orient
    # If not quite at the end (but fairly close),
        # align to target_orient, then reverse for 0.5s
    # If far from end, align / drive to end

    disp = end - start
    orient = front - start
    mag_orient = np.linalg.norm(orient)
    assert mag_orient > 4, "Front must be at least 4px away from start."
    mag_disp = np.linalg.norm(disp)

    if mag_disp < at_thresh:
        return get_precise_rotation(orient, target_orient, False, angle_thresh)
    if mag_disp < near_thresh:
        command, duration = get_precise_rotation(orient, target_orient, False, angle_thresh)
        if command is None:
            return BACKWARD, 0.5
        return command, duration
    command, duration = get_precise_rotation(orient, disp, False, angle_thresh)
    if command is None:
        return get_precise_translation(mag_disp)
    return command, duration

ip = "http://192.168.137.43"

SEND_COMMANDS = True # whether to attempt to send commands to the ip address
MIN_COMMAND_INTERVAL = 500 # in ms

def _test_go_to_target():
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

def cross_bridge(centre: np.ndarray, front: np.ndarray, go_to_dropoff_side: bool = True):
    """Return the command and duration to go over the bridge

    Parameters
    ----------
    centre : np.ndarray
        Coordinates (x,y) of the centre of the robot
    front : np.ndarray
        Coordinates (x,y) of the front of the robot
    go_to_dropoff_side : bool, optional
        If true, go to dropoff, else go to pickup, by default True

    Returns
    -------
    commands : np.ndarray or None
        The motor commands to be sent if needed, None otherwise
    time : float
        The duration of time for the command to be run in seconds
    """
    
    # Return command, duration
    TARGET_THRESH = 30
    NOT_TARGET_THRESH = 10
    # get frame
    # get target side
    if go_to_dropoff_side:
        target = BRIDGE_POINTS[1]
        not_target = BRIDGE_POINTS[0]
    else:
        target = BRIDGE_POINTS[0]
        not_target = BRIDGE_POINTS[1]
    print(target)

    # If at target, return nothing
    # If closer to target side or at non-target side, align / go to target side
    # Else align / go to non-target side
    target_disp = target - centre
    not_target_disp = not_target - centre
    target_dist = np.linalg.norm(target_disp)
    not_target_dist = np.linalg.norm(not_target_disp)
    if target_dist < TARGET_THRESH:
        print("We're here")
        return None, 0
    if target_dist < not_target_dist:
        return go_to_coord_timed(centre, target, front)
    if not_target_dist < NOT_TARGET_THRESH:
        return go_to_coord_timed(centre, target, front, False, TARGET_THRESH, 2)
    return go_to_coord_orient(centre, not_target, front, target_disp, NOT_TARGET_THRESH, angle_thresh=2)

# Commands to try if the robot gets stuck
STUCK_COMMANDS = (
    (FORWARD, 0.5),
    (BACKWARD, 0.5),
    (LEFT, 2),
    (FORWARD, 0.5),
    (BACKWARD, 0.5),
    (RIGHT, 2),
    (FORWARD, 0.5),
    (BACKWARD, 0.5),
    (RIGHT, 2)
)

STUCK_TIMEOUT = 2000

# Keep crossing the bridge in a loop
def _test_go_loop():
    #video = QRVideo('http://localhost:8081/stream/video.mjpeg', 0, 2.5)
    video = DotPatternVideo('http://localhost:8081/stream/video.mjpeg')
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    old_centre = np.array([100,100])
    old_front = np.array([200,200])
    TOL_STATIONARY = 1

    getString = ip + "/"
    lastString = ""
    next_command_time = 0
    go_to_dropoff_side = True
    
    stuck_time = 10000
    stuck_counter = 0

    while True:
        frame, found, centre, front = video.find()
        frame = draw_waypoints(frame)
        if found:
            back = centre - (front - centre)
            # Make sure commands aren't sent while the bot is moving
            distance = np.linalg.norm(old_centre - centre)
            distance = max(distance, np.linalg.norm(front - old_front))
            old_centre = centre
            old_front = front

            # Make sure commands aren't sent too rapidly
            current_time = round(time.time() * 1000)

            if distance > TOL_STATIONARY:
                stuck_time = current_time + STUCK_TIMEOUT
            elif current_time > next_command_time:
                # Get command, duration
                if current_time > stuck_time:
                    print("Stuck detected.")
                    command, duration = STUCK_COMMANDS[stuck_counter]
                    stuck_counter = (stuck_counter + 1) % len(STUCK_COMMANDS)
                else:
                    command, duration = cross_bridge(back, front, go_to_dropoff_side)
                    if command is None:
                        go_to_dropoff_side = not go_to_dropoff_side
                        command, duration = cross_bridge(centre, front, go_to_dropoff_side)
                if abs(command[0]) < STALL_SPEED and abs(command[1]) < STALL_SPEED:
                    print("Attempted to send a stall command.")
                    stuck_time = current_time + 1e6
                
                # Turn that into a getString
                duration *= 1000 # Turn s into ms
                if math.isnan(duration): 
                    # If nan, just give up, and pray the next frame is ok.
                    # This should never happen!
                    print("==============================")
                    print("DURATION WAS NAN, MUST FIX NOW")
                    print("==============================")
                    print(f"command: {command}")
                    time.sleep(5)
                    continue
                duration = int(duration)
                getString = ip + "/TRIGGER/" + str(command[0]) + "/" + str(command[1]) + "//" + str(duration) + "/"

            # Don't send duplicate commands
            if lastString != getString:
                SEND_COMMANDS and urllib.request.urlopen(getString)
                print("sending new command")
                print(getString)
                lastString = getString
                next_command_time = current_time + MIN_COMMAND_INTERVAL
                stuck_time = current_time + STUCK_TIMEOUT
            cv2.circle(frame, np.int0(back), 5, (255,255,0),-1)
        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        # Exit if q pressed
        if key == ord('q'):
            break
     
    cv2.destroyAllWindows()

# Calibrate the forward direction and the centre of rotation
def _test_calibrate():
    return 0

if __name__ == "__main__":
    # _test_go_to_target()
    _test_go_loop()
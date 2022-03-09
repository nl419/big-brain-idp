"""Generate motor commands to move from point to point.
Reads video stream from idpcam2, finds robot location & orientation,
calculates motor commands to arrive at target location."""

from cv2 import createBackgroundSubtractorKNN
import numpy as np
import cv2
# For the example cases at bottom
import urllib.request
from crop_board import PICKUP_CENTRE_T
from find_coords import untransform_board
from find_coords import dropoff_boxes, get_shift_invmat_mat, transform_board
from waypoints import *
from find_block import find_block
from find_qr import *
from find_dots import *
import time
import math
from robot_properties import *
ip = "http://192.168.137.169"

SEND_COMMANDS = False # whether to attempt to send commands to the ip address
MIN_COMMAND_INTERVAL = 1000 # in ms
DEBUG_WAYPOINTS = True

BLOCK_HEIGHT = 0.03 # in m, for unparallaxing

# Commands to try if the robot gets stuck
STUCK_COMMANDS = (
    (FORWARD, 0.5),
    (BACKWARD, 0.5),
    (LEFT, 2),
    (FORWARD, 0.5),
    (BACKWARD, 0.5),
    (RIGHT, 2),
    # (FORWARD, 0.5),
    # (BACKWARD, 0.5),
    # (RIGHT, 2)
)

CALIBRATION_COMMANDS = (
    (LEFT, 5),
    (FORWARD, 2)
)

# How to parse the return from the Arduino
SENSOR_DATA_TRIGGER = "TRIGGER/"
END_SENSOR_DATA_TRIGGER = "/"
SENSOR_DELTA_THRESH = 102 # 1024 is 5v
STATIONARY_THRESH = 2
STUCK_TIME_THRESH = 5000 # ms, maybe there was a huge lag spike

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

    # return intersections of line segments a and b, each defined with 2 points
    def seg_intersect(w1,w2, v1,v2) :
        dw = w2-w1
        dv = v2-v1
        dp = w1-v1
        dwp = perp(dw)
        denom = np.dot(dwp, dv)
        num = np.dot(dwp, dp)
        return (num / denom.astype(float))*dv + v1
    
    # Tolerance for a "zero" length vector
    LENGTH_TOL = 1e-2

    da = a1 - a0
    db = b1 - b0

    print(f"da {da}")
    print(f"db {db}")

    # Get points defining the line perpendicular to each of da and db
    ahalf = (a0 + a1) / 2
    aperp = ahalf + perp(da)
    bhalf = (b0 + b1) / 2
    bperp = bhalf + perp(db)

    print(f"ahalf {ahalf}")
    print(f"bhalf {bhalf}")
    print(f"aperp {aperp}")
    print(f"bperp {bperp}")

    # handle edge cases of zero length
    if np.linalg.norm(da) < LENGTH_TOL:
        c = ahalf
    elif np.linalg.norm(db) < LENGTH_TOL:
        c = bhalf
    # If the vectors are very nearly parallel
    elif np.cross(perp(da), perp(db)) / np.linalg.norm(da) / np.linalg.norm(db) < 0.01:
        c = seg_intersect(a0, b0, a1, b1)
    else: # all vectors are well behaved. find the centre.
        c = seg_intersect(ahalf, aperp, bhalf, bperp)
    
    return c


# Calibrate the forward direction and the centre of rotation
# Command 1: LEFT, 5
    # Find CofR from how the front + centre moved.
# Command 2: FORWARD, 2
    # Find forward direction from how the CofR moved
def _test_calibrate():
    #video = QRVideo('http://localhost:8081/stream/video.mjpeg', 0, 2.5)
    video = DotPatternVideo('http://localhost:8082/stream/video.mjpeg')
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    old_centre = np.array([100,100])
    old_front = np.array([200,200])
    start_centre = old_centre
    start_front = old_front
    CofR = old_centre
    TOL_STATIONARY = 2

    getString = ip + "/"
    lastString = ip + "/TRIGGER/0/0///"
    current_time = round(time.time() * 1000)
    next_command_time = round(time.time() * 1000) + 3000
    duration = 0

    calibration_mode = 0 # 0 = rotation, 1 = translation
    calibration = {
        # In the local coordinate system of the robot,
        # where is the CofR, and where is the true front?
        # (robot moves forward in the "true_front" - "centre" direction)
        "CofR": (0,0),      # [-0.87290985, -0.34059074] [-0.74257382 -0.13944251] [-0.74846031 -0.11834294]
        "true_front": (1,0) # [3.31095469 0.02314122] [3.299921   0.00386202]
    }

    while True:
        # TODO actually use the calibration
        frame, found, centre, front = video.find()
        if found:
            # Make sure commands aren't sent while the bot is moving
            distance = np.linalg.norm(old_centre - centre)
            distance = max(distance, np.linalg.norm(front - old_front))
            old_centre = centre
            old_front = front

            # Make sure commands aren't sent too rapidly
            current_time = round(time.time() * 1000)

            if distance < TOL_STATIONARY and current_time > next_command_time:
                if calibration_mode == 2: # Just finished a forward calibration
                    forward = centre - start_centre
                    print(f"forward {forward}")
                    true_front = transform_coords(centre, start_centre, start_front)
                    print(f"true_front {true_front}")
                    cv2.line(frame, np.int0(start_centre), np.int0(centre), (0,0,255), 3)
                    cv2.imshow("Tracking", frame)
                    cv2.waitKey(0)
                    break
                command, duration = CALIBRATION_COMMANDS[calibration_mode]
                if calibration_mode == 1: # Just finished a rotation calibration
                    CofR = centre_of_rotation(start_centre, start_front, centre, front)
                    CofR_trans = transform_coords(CofR, centre, front)
                    calibration["CofR"] = CofR_trans
                    print(f"CofR_trans {CofR_trans}")
                    cv2.circle(frame, np.int0(CofR), 5, (255,255,0), -1)
                    cv2.line(frame, np.int0(centre), np.int0(start_centre), (0,0,255), 3)
                    cv2.line(frame, np.int0(front), np.int0(start_front), (0,255,255), 3)
                    cv2.imshow("Tracking", frame)
                    cv2.waitKey(0)

                calibration_mode += 1
                start_centre = centre
                start_front = front
                # Turn that into a getString
                duration *= 1000 # Turn s into ms
                duration = int(duration)
                getString = ip + "/TRIGGER/" + str(command[0]) + "/" + str(command[1]) + "//" + str(duration) + "/"

                SEND_COMMANDS and urllib.request.urlopen(getString)
                print("sending new command")
                print(getString)
                next_command_time = round(time.time() * 1000) + duration + MIN_COMMAND_INTERVAL * 4
        cv2.circle(frame, np.int0(start_centre), 5, (0,255,0), -1)
        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        # Exit if q pressed
        if key == ord('q'):
            break
     
    cv2.destroyAllWindows()

class Navigator:
    _video: DotPatternVideo
    _frame: np.ndarray
    _found: bool = False
    _shift: np.ndarray
    _invmat: np.ndarray
    _mat: np.ndarray

    _got_block: bool = False
    _block_blue: bool = False
    _last_reading: int = 0
    _wps_up_to_date: bool = False

    # Robot locations
    _centre: np.ndarray = np.array((100,100)) # Dummy initial values
    _front: np.ndarray = np.array((200,200))
    _delta: float = 0 # How much the robot moved in the last frame
    # TODO normalise delta with framerate

    _not_stuck_thresh: float = -1 # if _delta > _not_stuck_thresh, then not stuck.
    _stuck_since: int or None = None # ms, when the robot got stuck
    _stuck_counter: int # Counter to iterate through the stuck commands

    # Locations of dropoff boxes, each length 2
    _blues: "tuple[np.ndarray]"
    _reds: "tuple[np.ndarray]"

    # First box = 0, second box = 1, second box reversed = 2
    _blue_dropoff_wps: "tuple[tuple[Waypoint or tuple]]"
    _red_dropoff_wps: "tuple[tuple[Waypoint or tuple]]"

    _blue_count: int = 0
    _red_count: int = 0

    # Towards pickup = 0, Towards dropoff = 1
    _blue_corner_wps: "tuple[tuple[Waypoint]]"
    _red_corner_wps: "tuple[tuple[Waypoint]]"

    # Generated automatically on state change
    _current_wps: "list[Waypoint or tuple]" = []
    _wp_counter: int = 0

    # When placing the last block, place it backwards to the usual orientation
    # so that then you can just go straight home, without knocking the blocks
    _home_wp: "Waypoint"

    # The string that was returned by the Arduino
    _ret_string: str = ""
    _command_timeout: int = 0

    def __init__(self, videostream_url: str):
        self._video = DotPatternVideo(videostream_url)
        frame, _,_,_ = self._video.find(annotate=False)
        self._shift, self._invmat, self._mat = get_shift_invmat_mat(frame)
        self._blues, self._reds = dropoff_boxes(frame, self._shift, self._invmat)

        self._home_wp = Waypoint(target_pos=untransform_board(self._shift, self._invmat, HOME_T), 
            target_orient=untransform_board(self._shift, self._invmat, np.array((0,0))) - untransform_board(self._shift, self._invmat, HOME_T)
        )
        self._blue_corner_wps = (
            (
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[0]),
                    target_orient=untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[1]) - \
                    untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[0]),
                    robot_offset=COFR_OFFSET, pos_tol=40
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[1]),
                    robot_offset=COFR_OFFSET, pos_tol=5
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[2]),
                    robot_offset=COFR_OFFSET, pos_tol=5
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[3]),
                    robot_offset=COFR_OFFSET, pos_tol=40
                )
            ),
            (
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[3]),
                    target_orient=untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[2]) - \
                    untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[3]),
                    robot_offset=COFR_OFFSET, pos_tol=40
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[2]),
                    robot_offset=COFR_OFFSET, pos_tol=5
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[1]),
                    robot_offset=COFR_OFFSET, pos_tol=5
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, BLUE_CORNER_TS[0]),
                    robot_offset=COFR_OFFSET, pos_tol=40
                )
            )
        )

        self._red_corner_wps = (
            (
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, RED_CORNER_TS[0]),
                    target_orient=untransform_board(self._shift, self._invmat, RED_CORNER_TS[1]) - \
                    untransform_board(self._shift, self._invmat, RED_CORNER_TS[0]),
                    robot_offset=COFR_OFFSET, pos_tol=40
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, RED_CORNER_TS[1]),
                    robot_offset=COFR_OFFSET, pos_tol=5
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, RED_CORNER_TS[2]),
                    robot_offset=COFR_OFFSET, pos_tol=5
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, RED_CORNER_TS[3]),
                    robot_offset=COFR_OFFSET, pos_tol=40
                )
            ),
            (
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, RED_CORNER_TS[3]),
                    target_orient=untransform_board(self._shift, self._invmat, RED_CORNER_TS[2]) - \
                    untransform_board(self._shift, self._invmat, RED_CORNER_TS[3]),
                    robot_offset=COFR_OFFSET, pos_tol=40
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, RED_CORNER_TS[2]),
                    robot_offset=COFR_OFFSET, pos_tol=5
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, RED_CORNER_TS[1]),
                    robot_offset=COFR_OFFSET, pos_tol=5
                ),
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, RED_CORNER_TS[0]),
                    robot_offset=COFR_OFFSET, pos_tol=40
                )
            )
        )

        dropoff_sequence = \
        [
            (
                (0,0),
                GATE_UP,
                1 # In seconds
            ),
            (
                BACKWARD,
                GATE_UP,
                1
            ),
            (
                (0,0),
                GATE_DOWN,
                0.5
            )
        ]

        b_d_wps = [
            [
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, BLUE_POINT_BOX1_T),
                    target_orient=self._blues[0] - untransform_board(self._shift, self._invmat, BLUE_POINT_BOX1_T),
                    pos_tol=10, orient_tol=3, robot_offset=GATE_OFFSET
                ),
                Waypoint(
                    target_pos=self._blues[0],
                    pos_tol=3, orient_tol=3, robot_offset=GATE_OFFSET
                )
            ],
            [
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, BLUE_POINT_BOX2_T),
                    target_orient=self._blues[1] - untransform_board(self._shift, self._invmat, BLUE_POINT_BOX2_T),
                    pos_tol=10, orient_tol=3, robot_offset=GATE_OFFSET
                ),
                Waypoint(
                    target_pos=self._blues[1],
                    pos_tol=3, orient_tol=3, robot_offset=GATE_OFFSET
                )
            ],
            [
                Waypoint(
                    target_pos=self._blues[1] + np.array((100,0)),
                    target_orient=np.array((-100,0)),
                    pos_tol=10, orient_tol=3, robot_offset=GATE_OFFSET
                ),
                Waypoint(
                    target_pos=self._blues[1],
                    pos_tol=3, orient_tol=3, robot_offset=GATE_OFFSET
                )
            ]
        ]

        r_d_wps = [
            [
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, RED_POINT_BOX1_T),
                    target_orient=self._reds[0] - untransform_board(self._shift, self._invmat, RED_POINT_BOX1_T),
                    pos_tol=10, orient_tol=3, robot_offset=GATE_OFFSET
                ),
                Waypoint(
                    target_pos=self._reds[0],
                    pos_tol=3, orient_tol=3, robot_offset=GATE_OFFSET
                )
            ],
            [
                Waypoint(
                    target_pos=untransform_board(self._shift, self._invmat, RED_POINT_BOX2_T),
                    target_orient=self._reds[1] - untransform_board(self._shift, self._invmat, RED_POINT_BOX2_T),
                    pos_tol=10, orient_tol=3, robot_offset=GATE_OFFSET
                ),
                Waypoint(
                    target_pos=self._reds[1],
                    pos_tol=3, orient_tol=3, robot_offset=GATE_OFFSET
                )
            ],
            [
                Waypoint(
                    target_pos=self._reds[1] + np.array((0,-100)),
                    target_orient=np.array((0,100)),
                    pos_tol=10, orient_tol=3, robot_offset=GATE_OFFSET
                ),
                Waypoint(
                    target_pos=self._reds[1],
                    pos_tol=3, orient_tol=3, robot_offset=GATE_OFFSET
                )
            ]
        ]

        self._blue_dropoff_wps = [i + dropoff_sequence for i in b_d_wps]
        self._red_dropoff_wps = [i + dropoff_sequence for i in r_d_wps]

    def __repr__(self) -> str:
        result = (\
      f"Navigator:\n"
      f" =Robot State=\n"
      f"  got_block: {self._got_block}\n"
      f"  block_blue: {self._block_blue}\n"
      f"  stuck_since: {self._stuck_since}\n"
      f"  stuck_counter: {self._stuck_counter}\n"
      f"  not_stuck_thresh: {self._not_stuck_thresh}\n"
      f"  wps_up_to_date: {self._wps_up_to_date}\n"
      f" =Coordinates=\n"
      f"  centre: {self._centre}\n"
      f"  front: {self._front}\n"
      f"  delta: {self._delta}\n"
      f"  centre: {self._centre}\n"
      f" =Waypoints=\n"
      f"  len(current_wps): {len(self._current_wps)}\n"
      f"  wp_counter: {self._wp_counter}\n"
        )
        
        for i,wp in enumerate(self._current_wps):
            result += f"  {i:2}: {wp}\n"
        return result

    def _draw_wp(self, wp):
        if type(wp) is Waypoint:
            wp.draw(self._frame)

    def _run_wps(self):
        # Returns False if the end of the wps were reached
        # Relies on self._centre, self._front being up to date
        # If wps[i] isn't a Waypoint, should contain 
        # (motor commands, gate_pos, duration: s)
        # None triggers a block colour check

        # Check for stuck, don't send commands while moving / before timeout.
        now = round(time.time() * 1000)
        if self._delta > self._not_stuck_thresh: self._stuck_since = None
        elif self._stuck_since is None: self._stuck_since = now

        if self._delta > STATIONARY_THRESH or self._command_timeout > now:
            return True

        while True:
            # Handle being stuck
            gate_pos = None
            if self._stuck_since is not None and self._stuck_since > now + STUCK_TIME_THRESH:
                print("Stuck detected. Executing stuck command")
                command, duration = STUCK_COMMANDS[self._stuck_counter]
                self._stuck_counter = (self._stuck_counter + 1) % len(STUCK_COMMANDS)
                break
            if self._wp_counter >= len(self._current_wps):
                # Reached the end of the waypoints
                self._wp_counter = 0
                return False
            wp = self._current_wps[self._wp_counter]
            if type(wp) is Waypoint:
                print("Executing waypoint")
                command, duration = wp.get_command(self._centre, self._front)
            elif wp is None:
                # Check the colour of the block by sending a blank command
                get_string = ip
                if SEND_COMMANDS: self._ret_string = urllib.request.urlopen(get_string)
                print("Reading sensor data...")
                if SEND_COMMANDS:
                    i = self._ret_string.find(SENSOR_DATA_TRIGGER)
                    assert i != -1, "Sensor data was not found."
                    i += len(SENSOR_DATA_TRIGGER)
                    j = self._ret_string[i:].find(END_SENSOR_DATA_TRIGGER)
                    assert j != -1, "End of sensor data was not found."
                    data = int(self._ret_string[i:i+j])
                else:
                    data = 200 # dummy value
                print(f"data {data}")
                if self._last_reading == 0:
                    print("Background levels measured.")
                    self._last_reading = data
                elif data - self._last_reading > SENSOR_DELTA_THRESH:
                    print("Red block found.")
                    self._block_blue = False
                    self._last_reading = 0
                else:
                    print("Blue block found.")
                    if DEBUG_WAYPOINTS: print("OVERRIDE: not updating block colour.")
                    else: self._block_blue = True
                    self._last_reading = 0
                # Don't send commands too quickly
                time.sleep(MIN_COMMAND_INTERVAL / 1000)
                # Stand still for 1 second.
                command, duration = (0,0), 1
                self._wp_counter += 1
            else:
                print("Executing hardcoded command")
                command, gate_pos, duration = wp
                self._wp_counter += 1
            if command is None:
                # Waypoint completed, get the next one
                self._wp_counter += 1
            else:
                # We have a valid command, send it.
                break
        # Turn that into a getString
        duration *= 1000 # Turn s into ms
        if math.isnan(duration): 
            # If nan, just give up, and pray the next frame is ok.
            # This should never happen!
            print("==============================")
            print("DURATION WAS NAN, MUST FIX NOW")
            print("==============================")
            print(f"command: {command}")
            print(repr(self))
            time.sleep(5)
            return True
        duration = int(duration)
        get_string = ip + "/TRIGGER/" + str(command[0]) + "/" + str(command[1]) + \
                    "/" + (str(gate_pos) if gate_pos is not None else "") + "/" + str(duration) + "/"

        if SEND_COMMANDS: self._ret_string = urllib.request.urlopen(get_string)
        print(f"sending command {get_string}")
        self._command_timeout = now + duration + MIN_COMMAND_INTERVAL
        
        # Update not stuck threshold based on how much the robot should be moving
        if command[0] != 0 and duration < 300: # A short duration command was given
            self._not_stuck_thresh = 2
        elif abs(command[0]) > 150: # A forward / backward command was given
            self._not_stuck_thresh = 5
        elif command[0] != 0: # A turn command was given
            self._not_stuck_thresh = 2
        else: # No movement was ordered.
            self._not_stuck_thresh = -1 
        return True

    def _update_state(self, reuse_frame: bool = False):
        # Gets the next frame, and updates the waypoints if necessary
        # Returns False if the state was not properly updated
        if not reuse_frame: 
            self._frame, self._found, new_c, new_f = self._video.find(shift=self._shift, invmat=self._invmat)
            if not self._found:
                if DEBUG_WAYPOINTS:
                    new_c = np.array((800,200))
                    new_f = np.array((700,300))
                else: return False
            self._delta = max(np.linalg.norm(new_c - self._centre), np.linalg.norm(new_f - self._front))
            self._centre = new_c
            self._front = new_f
        if not self._found and not DEBUG_WAYPOINTS:
            return False

        if not self._wps_up_to_date:
            self._current_wps = []
            if self._got_block:
                # Go to dropoff
                self._current_wps.extend(self._blue_corner_wps[1] if self._block_blue else self._red_corner_wps[1])
                i = self._blue_count if self._block_blue else self._red_count
                assert i < 2, "Too many " + "blue" if self._block_blue else "red" + " blocks have been found"
                # Put the very last block in backwards, so that home can still be driven to easily
                if (self._red_count == 2 or self._blue_count == 2) and i == 1:
                    i = 2
                self._current_wps.extend(self._blue_dropoff_wps[i] if self._block_blue else self._red_dropoff_wps[i])
                if self._block_blue:
                    self._blue_count += 1
                else:
                    self._red_count += 1
            else:
                # Summary:
                # Find a block
                # Approach the pickup area
                # Move to take baseline colour reading
                # Move to take block colour reading
                # Open gate
                # Move over block
                # Close gate
                # Reverse a little

                # If all the blocks have been delivered, go home.
                if self._blue_count == 2 and self._red_count == 2:
                    self._current_wps = [self._home_wp]
                    return True
                # Make a route to pickup a block
                b_c, _ = find_block(self._frame, self._shift, self._invmat)
                if b_c is None:
                    if DEBUG_WAYPOINTS: b_c = np.array((300,600))
                    else: return False
                b_c = undo_parallax(b_c, BLOCK_HEIGHT)
                # Make a waypoint at the block
                b_c_T = transform_board(self._shift, self._mat, b_c)
                block_wp = Waypoint(
                    target_pos=b_c, pos_tol=5, robot_offset=PICKUP_OFFSET,
                    orient_backward_ok=False, move_backward_ok=True
                )
                corner_wps =  self._blue_corner_wps[0] if self._block_blue else self._red_corner_wps[0]
                self._current_wps.extend(corner_wps)

                # Avoid hitting the walls if the block is near the corner
                route_wp = None
                if b_c_T[1] > PICKUP_CENTRE_T[1]:
                    if b_c_T[0] > 0:
                        # Block is in the bottom right half of the pickup area
                        # So approach from the top
                        coord = b_c + np.array((0, -80))
                    else:
                        coord = b_c + np.array((80, 0))
                    route_wp = Waypoint(
                        target_pos=coord, target_orient=coord-b_c,
                        robot_offset=PICKUP_OFFSET
                    )
                    self._current_wps.append(route_wp)

                # Read background lighting conds
                self._current_wps.append(Waypoint(
                    target_pos=b_c, pos_tol=5, robot_offset=SENSOR_OFFSET_NO_DETECT,
                    orient_backward_ok=False, move_backward_ok=True
                ))
                self._current_wps.append(
                    None
                )
                # Read block lighting conds
                self._current_wps.append(Waypoint(
                    target_pos=b_c, pos_tol=5, robot_offset=SENSOR_OFFSET_DETECT,
                    orient_backward_ok=False, move_backward_ok=True
                ))
                self._current_wps.append(
                    None
                )
                # Pickup sequence
                self._current_wps.append((
                    (0,0),
                    GATE_UP,
                    1
                ))
                self._current_wps.append(block_wp)
                self._current_wps.append((
                    (0,0),
                    GATE_DOWN,
                    1
                ))
                self._current_wps.append((
                    BACKWARD,
                    GATE_DOWN,
                    0.5
                ))
        return True

    def run(self) -> "tuple[np.ndarray, bool]":
        """Run the navigation code:
        - Detect block
        - Pickup block
        - Dropoff block
        etc.

        Also shows the latest frame with annotations.


        Returns
        -------
        frame : np.ndarray
            The latest video frame with annotations
        ok : bool
            Whether the navigation was successful. 
            - True: 
                - Command was sent
                - Waiting for command timeout
                - Waiting for robot to be stationary
            - False: 
                - Robot not found
                - Block not found
                - Waypoint calculation failed
        """
        # Get the next frame, process it, send commands to robot.
        
        ok = self._update_state() # Grab next frame, update state
        if len(self._current_wps) > 0 and self._wp_counter < len(self._current_wps):
            self._draw_wp(self._current_wps[self._wp_counter])
        if not ok: return self._frame, False
        self._wps_up_to_date = True # If we got here, the waypoints must be ok.

        if not self._run_wps():
            # reached end of wps
            self._got_block = not self._got_block
            self._wps_up_to_date = False
            self._wps_up_to_date = self._update_state(reuse_frame=True) # Get next waypoints
            # If waypoints couldn't be calculated, wait until next frame
            if not self._wps_up_to_date: return self._frame, False 
            # New waypoints found, so execute the next one.
            self._run_wps()
        return self._frame, True

def _test_navigator():
    nav = Navigator('http://localhost:8082/stream/video.mjpeg')
    write_count = 0
    while True:
        frame, ok = nav.run()
        cv2.imshow("nav", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            nav._wp_counter += 1
        elif key == ord('b'):
            nav._block_blue = True
        elif key == ord('r'):
            nav._block_blue = False
        elif key == ord('w'):
            cv2.imwrite("nav-test-fails/" + write_count + ".jpg", frame)
            write_count += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # _test_calibrate()
    _test_navigator()
import numpy as np
from robot_properties import *
from find_dots import untransform_coords, get_CofR, get_true_front, perp, angle
from find_coords import untransform_board
import cv2

_DEBUG = __name__ == "__main__"
MAX_COMMAND_NUM = 2


class Waypoint:
    _target_pos: np.ndarray
    _target_orient: np.ndarray # which way the true forward dir should point, always unit length
    _pos_tol: float # pixels
    _near_tol: float # pixels
    _orient_tol: float # radians
    _orient_backward_ok: bool
    _move_backward_ok: bool
    _do_fine: bool

    # The position in the robot's coordinate system which we want to get to 
    # target_pos
    _robot_offset: np.ndarray

    def __init__(self, target_pos: np.ndarray, target_orient: np.ndarray = None, 
                 pos_tol: float = 10, orient_tol: float = 5, robot_offset: np.ndarray = np.array((0,0)), 
                 orient_backward_ok: bool = False, move_backward_ok: bool = True, near_tol: float = -1, do_fine: bool = True):
        """A waypoint on the board. Run [Waypoint object].get_command(centre, front) 
        to get the next command & duration

        Parameters
        ----------
        target_pos : np.ndarray
            Coordinates (x,y) of the target
        target_orient : np.ndarray or None
            Vector (x,y) to align with the true front direction of the robot, default None
        pos_tol : float
            Tolerance for being "at" the target position, default 10
        orient_tol : float
            Tolerance for being "parallel" to the target orientation in degrees, default 5
        robot_offset : np.ndarray
            The point on the robot to move to the target (in robot's coord system), default (0,0)
        orient_backward_ok : bool, optional
            Whether aligning to the reverse orientation is ok, by default False
        move_backward_ok : bool, optional
            Whether moving backward for large distances is ok, by default True.
        near_tol : float, optional
            Threshold for reversing a little bit instead of doing large turns, default -1
        do_fine : float, optional
            Whether swapping to fine movements are allowed, default True
        """
        self._target_pos = target_pos
        if target_orient is None:
            self._target_orient = None
        else:
            self._target_orient = target_orient / np.linalg.norm(target_orient)
        self._pos_tol = pos_tol
        self._orient_tol = np.radians(orient_tol)
        self._robot_offset = robot_offset
        self._orient_backward_ok = orient_backward_ok
        self._move_backward_ok = move_backward_ok
        self._near_tol = near_tol
        self._do_fine = do_fine

    def _get_rotation_noalign(self, centre, front):
        ### Assuming mag_tm is larger than mag_om_
        # Returns positive ccw, should be positive cw
        m = get_CofR(centre, front)
        forward = get_true_front(centre, front) - centre
        right = perp(forward)
        o = untransform_coords(self._robot_offset, centre, front) 
        om = o - m
        om_ = np.dot(om, right) * right / np.linalg.norm(right)**2
        t = self._target_pos
        tm = t - m
        mag_tm = np.linalg.norm(tm)
        mag_om_ = np.linalg.norm(om_)

        if abs(mag_om_ < 3): # o is in front of CofR
            theta = -angle(tm, forward)
            if self._move_backward_ok:
                # Limit rotations to between +-90 deg (+- pi/2 rad)
                return (theta + np.pi/2) % np.pi - np.pi/2
            # Limit rotations to between +-180 deg (+- pi rad)
            return (theta + np.pi) % (np.pi * 2) - np.pi

        theta = -angle(tm, om_)
        # If we can reverse into the target, align with the reverse direction.
        direction = -1
        if theta < 0 or not self._move_backward_ok: # closer to the forward direction
            theta = -theta
            direction = 1
        alpha = (np.arccos(mag_om_ / mag_tm) - theta) * direction
        # Limit rotations to between +-180 deg (+- pi rad)
        return (alpha + np.pi) % (np.pi * 2) - np.pi

    def _get_m_new(self, centre, front):
        # Find change in angle, and corresponding matrix
        # TODO: this is a bit inefficient, should really turn this into
        # a bunch of np.matmul()'s instead of sin / cos
        f_true = get_true_front(centre, front) - centre
        alpha = angle(f_true, self._target_orient)
        sin = np.sin(alpha)
        cos = np.cos(alpha)
        mat = np.array(((cos, -sin), (sin, cos)))

        # Find new front & centre
        oc = untransform_coords(self._robot_offset, centre, front) - centre
        oc_new = np.matmul(mat, oc)
        c_new = self._target_pos - oc_new
        fc = front - centre
        fc_new = np.matmul(mat, fc)
        f_new = fc_new + c_new

        return get_CofR(c_new, f_new)

    def _get_rotation_align_target(self, centre, front):
        f_true = get_true_front(centre, front) - centre
        alpha = angle(f_true, self._target_orient)

        if self._orient_backward_ok:
            # Limit rotation to +-90 deg (+- pi/2 rad)
            return (alpha + np.pi/2) % np.pi - np.pi/2
        return alpha

    def _get_rotation_align_CofR(self, centre, front):
        f_true = get_true_front(centre, front) - centre

        # Find CofR's
        m = get_CofR(centre, front)
        m_new = self._get_m_new(centre, front)
        dm = m_new - m
        
        alpha = angle(f_true, dm)
        if self._move_backward_ok:
            # Limit rotation to +-90 deg (+- pi/2 rad)
            return (alpha + np.pi/2) % np.pi - np.pi/2
        return alpha


        return 

    def _get_translation_noalign(self, centre, front):
        o = untransform_coords(self._robot_offset, centre, front)
        t = self._target_pos

        f_true = get_true_front(centre, front) - centre
        f_unit = f_true / np.linalg.norm(f_true)

        return np.dot(t - o, f_unit)
    
    def _get_translation_align(self, centre, front, return_abs = False):
        m = get_CofR(centre, front)
        m_new = self._get_m_new(centre, front)

        f_true = get_true_front(centre, front) - centre
        f_unit = f_true / np.linalg.norm(f_true)

        delta = m_new - m

        return np.linalg.norm(delta) if return_abs else np.dot(delta, f_unit)

    def _get_abs_distance(self, centre, front):
        o = untransform_coords(self._robot_offset, centre, front)
        t = self._target_pos
        return np.linalg.norm(t - o)

    def get_command(self, centre: np.ndarray, front: np.ndarray, rotation_only: bool = False):
        """Get a motor command pair and duration in order to execute this waypoint

        Parameters
        ----------
        centre : np.ndarray
            Coordinates (x,y) of the centre of the robot
        front : np.ndarray
            Coordinates (x,y) of the front of the robot

        Returns
        -------
        command : np.ndarray or None
            Motor speeds to send to the robot, if needed
        duration : float
            Required duration of the command
        """
        # Summary:
        # Get a rotation first,
        # If it's large, execute it
        # Else get a translation
        # If it's large, execute it
        # Else return None, 0

        # Should align to a target orientation
        if self._target_orient is not None:
            # Make sure robot is near the target point
            dist = self._get_translation_align(centre, front, True)
            if dist > self._pos_tol:
                rot = self._get_rotation_align_CofR(centre, front)
                # Tolerance for the intermediate rotation is fixed at 5 deg
                if abs(rot) > np.radians(3):
                    # Reverse a bit if too close
                    if dist < self._near_tol:
                        if rotation_only: return None, 0
                        return BACKWARD, self._near_tol / MOVEMENT_SPEED
                    duration = abs(rot) / ROTATION_SPEED
                    if not self._do_fine or duration > FINE_THRESH:
                        command = RIGHT if rot > 0 else LEFT
                        return command, duration
                    command = RIGHT_FINE if rot > 0 else LEFT_FINE
                    return command, abs(rot) / ROTATION_SPEED_FINE
                if rotation_only: return None, 0

                # If we got here, then no turning was needed.
                # Should now just head to the point.
                trans = self._get_translation_align(centre, front)
                duration = abs(trans) / MOVEMENT_SPEED
                if not self._do_fine or duration > FINE_THRESH:
                    command = FORWARD if trans > 0 else BACKWARD
                    return command, duration
                command = FORWARD_FINE if trans > 0 else BACKWARD_FINE
                return command, abs(trans) / MOVEMENT_SPEED_FINE

            # Make sure the robot is facing the right way
            rot = self._get_rotation_align_target(centre, front)
            if abs(rot) > self._orient_tol:
                duration = abs(rot) / ROTATION_SPEED
                if not self._do_fine or duration > FINE_THRESH:
                    command = RIGHT if rot > 0 else LEFT
                    return command, duration
                command = RIGHT_FINE if rot > 0 else LEFT_FINE
                return command, abs(rot) / ROTATION_SPEED_FINE

            # No actions needed.
            return None, 0
        
        # Should not align to a target orientation.
        rot = self._get_rotation_noalign(centre, front)
        trans = self._get_translation_noalign(centre, front)
        abs_dist = self._get_abs_distance(centre, front)
        # If already close enough, return None
        if abs_dist <= self._pos_tol: 
            return None, 0
        # If near, but not quite there, reverse a little
        if abs_dist < self._near_tol and abs(rot) > np.radians(30):
            # Reverse a little
            if rotation_only: return None, 0
            return BACKWARD, self._near_tol / MOVEMENT_SPEED

        # Rotate first, then move.
        if abs(rot) > self._orient_tol:
            duration = abs(rot) / ROTATION_SPEED
            if not self._do_fine or duration > FINE_THRESH:
                command = RIGHT if rot > 0 else LEFT
                return command, duration
            command = RIGHT_FINE if rot > 0 else LEFT_FINE
            return command, abs(rot) / ROTATION_SPEED_FINE
        if abs(trans) > self._pos_tol:
            if rotation_only: return None, 0
            duration = abs(trans) / MOVEMENT_SPEED
            if not self._do_fine or duration > FINE_THRESH:
                command = FORWARD if trans > 0 else BACKWARD
                return command, duration
            command = FORWARD_FINE if trans > 0 else BACKWARD_FINE
            return command, abs(trans) / MOVEMENT_SPEED_FINE

        return None, 0
    
    def draw(self, image: np.ndarray, colour: tuple = (0,255,0)):
        """Draw the marker onto the image, showing position of waypoint,
        orientation of waypoint (if specified), and tolerances for each.

        Parameters
        ----------
        image : np.ndarray
            Image to mark
        colour : tuple, optional
            Colour of the markers, by default (0,255,0)
        """
        # Draw target_pos and pos_tol
        start = np.int0(self._target_pos)
        cv2.drawMarker(image, start, colour, cv2.MARKER_CROSS, 30, 2)
        cv2.circle(image, start, self._pos_tol, colour, 2)

        forward = np.array((1,0))
        if self._target_orient is not None:
            # Draw target orientation
            end = np.int0(self._target_pos + self._target_orient * 50)
            cv2.arrowedLine(image, start, end, colour, 4)
            forward = self._target_orient
        # Draw orientation tolerance
        sin = np.sin(self._orient_tol)
        cos = np.cos(self._orient_tol)
        rot_mat = np.array(((cos, sin), (-sin, cos)))
        inv_rot_mat = np.array(((cos, -sin), (sin, cos)))
        orient_rot = np.matmul(rot_mat, forward)
        orient_inv_rot = np.matmul(inv_rot_mat, forward)
        end = np.int0(self._target_pos + orient_rot * 50)
        cv2.line(image, start, end, colour, 2)
        end = np.int0(self._target_pos + orient_inv_rot * 50)
        cv2.line(image, start, end, colour, 2)

def get_rot_mat(alpha: float):
    """Returns the rotation matrix corresponding to `alpha`
    radians clockwise.

    Parameters
    ----------
    alpha : float
        The angle to rotate by, +ve clockwise, in radians

    Returns
    -------
    np.ndarray
        Matrix corresponding to the rotation
    """
    cos = np.cos(alpha)
    sin = np.sin(alpha)
    return np.array(((cos, -sin), (sin, cos)))

def get_bbox_from_centre_front(centre: np.ndarray, front: np.ndarray) -> "tuple[np.ndarray]":
    """Returns the bounding box coordinates corresponding to
    given centre and front coords

    Parameters
    ----------
    centre : np.ndarray
        Centre of the robot
    front : np.ndarray
        Front of the robot

    Returns
    -------
    tuple of np.ndarray
        List of bounding box points
    """
    forward = front - centre
    right = perp(forward)
    return centre + forward - right, centre + forward + right,\
    centre - forward + right, centre - forward - right

def predict_centre_front(centre: np.ndarray, front: np.ndarray,\
    command: list or np.ndarray, duration: float) -> "tuple[np.ndarray, np.ndarray]":
    """Get the predicted positions of the centre and front
    of the robot

    Parameters
    ----------
    centre : np.ndarray
        Centre of the robot
    front : np.ndarray
        Front of the robot
    command : list or np.ndarray
        The motor commands to run
    duration : float
        Duration of command in seconds

    Returns
    -------
    new_centre: np.ndarray
        Location of centre of the robot after command
    new_front: np.ndarray
        Location of front of the robot after command
    """
    if type(command) is np.ndarray:
        command = command.tolist()
    if (command == FORWARD).all() or (command == BACKWARD).all()\
    or (command == FORWARD_FINE).all() or (command == BACKWARD_FINE).all():
        direction = 1
        if (command == BACKWARD).all() or (command == BACKWARD_FINE).all():
            direction = -1
        speed = MOVEMENT_SPEED
        if (command == FORWARD_FINE).all() or (command == BACKWARD_FINE).all():
            speed = MOVEMENT_SPEED_FINE
        # Get unit forward dir
        f_t = get_true_front(centre, front) - centre
        f_t = f_t / np.linalg.norm(f_t)
        delta = f_t * duration * direction * speed
        return centre + delta, front + delta
    elif (command == LEFT).all() or (command == LEFT_FINE).all()\
    or (command == RIGHT).all() or (command == RIGHT_FINE).all():
        direction = 1 # +ve cw
        if (command == LEFT).all() or (command == LEFT_FINE).all():
            direction = -1
        speed = ROTATION_SPEED
        if (command == LEFT_FINE).all() or (command == RIGHT_FINE).all():
            speed = ROTATION_SPEED_FINE
        alpha = speed * duration * direction
        mat = get_rot_mat(alpha)
        m = get_CofR(centre, front)
        f_new = np.matmul(mat, front - m) + m
        c_new = np.matmul(mat, centre - m) + m
        return c_new, f_new
    elif (command == CORNER_LEFT).all() or (command == CORNER_RIGHT).all():
        direction = 1 # +ve cw
        if (command == CORNER_LEFT).all():
            direction = -1
        speed = CORNER_SPEED
        if (command == LEFT_FINE).all() or (command == RIGHT_FINE).all():
            speed = ROTATION_SPEED_FINE
        alpha = speed * duration * direction * np.pi / 2
        mat = get_rot_mat(alpha)
        m = untransform_coords(CORNER_LEFT_OFFSET if direction == -1 else CORNER_RIGHT_OFFSET, centre, front)
        f_new = np.matmul(mat, front - m) + m
        c_new = np.matmul(mat, centre - m) + m
        return c_new, f_new
    else: return centre, front



# Generalises the idea of a list of actions to carry out
# Returns all the commands to be run at once.
class Subroutine:
    _actions: list or tuple
    _skip_checks: bool
    _just_once: "tuple[bool]"
    _has_run: "list[bool]"

    def __init__(self, actions: list or tuple, skip_checks: bool = True, just_once: "tuple[bool]" = None) -> None:
        self._actions = actions
        self._skip_checks = skip_checks
        self._just_once = just_once if just_once is not None else tuple([False] * len(actions))
        self._has_run = [False] * len(actions)

    def draw(self, image: np.ndarray, colour: tuple = (0,255,0), centre=None, front=None):
        # Draw all the waypoints on
        for a in self._actions:
            if type(a) is Waypoint:
                a.draw(image, colour)
                if centre is not None and front is not None:
                    offset = untransform_coords(a._robot_offset, centre, front)
                    cv2.drawMarker(image, np.int0(offset), colour, cv2.MARKER_CROSS, 10, 1)


    def get_command_list(self, centre, front, colour_thresh=-1):
        # Return all the commands to be sent
        # Colour thresh of None means "read the sensor, but don't light LEDs"
        # Colour thresh of -1 means "don't read sensor"
        commands, durations, servo_poss, colour_threshs = [], [], [], []
        c_new = centre
        f_new = front
        done = False
        force_get_next = False
        for i,a in enumerate(self._actions):
            while not done:
                # Don't send too many commands at once
                if len(commands) >= MAX_COMMAND_NUM: 
                    done = True
                    break
                if self._has_run[i] and self._just_once[i]:
                    break # Just get the next command
                if type(a) is Waypoint:
                    # Try to get a rotation
                    command, duration = a.get_command(c_new, f_new, True)
                    if command is not None and duration > LARGE_ROTATION_THRESH:
                        done = True
                    if command is None:
                        # Must translate
                        command, duration = a.get_command(c_new, f_new, False)
                        # If translation commanded,
                        if command is not None: done = True
                    servo_pos, check_colour = None, False
                else: # 'a' is tuple of hardcoded commands
                    command, servo_pos, duration, check_colour = a
                    self._has_run[i] = True
                    force_get_next = True
                if command is None:
                    self._has_run[i] = True
                    break # Waypoint complete
                commands.append(command)
                durations.append(duration)
                servo_poss.append(servo_pos)
                colour_threshs.append(colour_thresh if check_colour else -1)
                if not self._skip_checks:
                    done = True
                c_new, f_new = predict_centre_front(c_new, f_new, command, duration)
                if force_get_next:
                    force_get_next = False
                    break # Hardcoded command complete
            else:
                break # Done was true, so stop getting commands
        
        if len(commands) == 0:
            for i in self._has_run:
                i = False
        return commands, servo_poss, durations, colour_threshs

# _T means transformed coordinates (relative to the yellow barriers)

BLUE_BARRIER_T = np.array((-1.57657, 0.00544))
RED_BARRIER_T = np.array((1.63555, -0.00785))

BLUE_CORNER_T = np.array((-1.90484, 0)) # 0.02536
RED_CORNER_T = np.array((1.90484, 0)) # -0.00796 2.25079
# Ordered towards pickup, first & last are routing waypoints
BLUE_CORNER_FORWARD_TS = np.array((
    (-1.77714, -0.16240), # (-1.79356, -0.18146) (-1.80997, -0.20051)
    BLUE_CORNER_T,
    (-1.11507, 0.82523)
))
BLUE_CORNER_BACKWARD_TS = np.array((
    (-1.77395, 0.17771),
    BLUE_CORNER_T,
    (-1.28200, -0.67462),
))
RED_CORNER_FORWARD_TS = np.array((
    (1.90121, -0.22039),
    RED_CORNER_T,
    (1.23903, 0.74794)
))
RED_CORNER_BACKWARD_TS = np.array((
    (1.94007, 0.21116),
    RED_CORNER_T,
    (1.37965, -0.58829),
))
PICKUP_CROSS_T = np.array((-0.04367, 1.02938))
DROPOFF_CROSS_T = np.array((-0.04001, -0.98237))

BRIDGE_MIDDLE_T = (PICKUP_CROSS_T + DROPOFF_CROSS_T) / 2

BLUE_DROPOFFS_T = np.array((
    (-0.63225, -1.18038),
    (-0.63766, -0.72680)
))
RED_DROPOFFS_T = np.array((
    (0.59841, -1.22614),
    (0.59703, -0.75593)
))

HOME_T = np.array((-0.02170, -1.78574))

## Intermediate waypoints
# Near corners on pickup / dropoff side
BLUE_POINT_PICKUP_T = np.array((-1.45095, 0.37588)) 
BLUE_POINT_DROPOFF_T = np.array((-1.43618, -0.39134))
# Near dropoff boxes
BLUE_POINT_BOX1_T = np.array((-0.82942, -0.98781))
BLUE_POINT_BOX2_T = np.array((-0.84395, -0.51035))

RED_POINT_PICKUP_T = np.array((1.48792, 0.51021))
RED_POINT_DROPOFF_T = np.array((1.56643, -0.41359))
RED_POINT_BOX1_T = np.array((0.79716, -1.04265))
RED_POINT_BOX2_T = np.array((0.80650, -0.55608))


# =====================================================================
# =====================================================================

def _test_waypoint():
    from unfisheye import undistort
    from crop_board import remove_shadow, crop_board
    from find_coords import get_shift_invmat_mat
    from find_dots import getDots, getDotBbox, get_true_front, get_CofR, drawMarkers
    # image = cv2.imread("dots/dot17.jpg")
    image = cv2.imread("dots/smol1.jpg")
    image = undistort(image)
    image2 = image.copy()
    
    # Preprocess
    image2 = remove_shadow(image2)
    shift, invmat, mat = get_shift_invmat_mat(image2)
    image = crop_board(image, shift, invmat)

    untransform_board(shift, invmat, RED_CORNER_T)
    dots = getDots(image)
    found, bbox = getDotBbox(dots)
    if not found:
        assert False, "Dots not found!"
    centre, front = drawMarkers(image, bbox, (255,0,0))
    c_new, f_new = centre.copy(), front.copy()
    print(centre, front)
    original = image.copy()

    target_pos = np.array((100,100))
    offset = COFR_OFFSET
    orient = np.array((1,0))

    def click_event(event, x, y, flags, params):
        nonlocal target_pos, offset
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            target_pos = np.array((x,y))
            offset = COFR_OFFSET
            redraw()

    def redraw():
        nonlocal target_pos, centre, front, c_new, f_new, offset, orient
        image = original.copy()
        cv2.drawMarker(image, np.int0(centre), (255,0,0), cv2.MARKER_CROSS, 30, 2)
        cv2.drawMarker(image, np.int0(front), (255,0,0), cv2.MARKER_CROSS, 30, 2)
        wp = Waypoint(target_pos=target_pos, target_orient=orient, 
                  pos_tol=20, orient_tol=5, robot_offset=offset,
                  move_backward_ok=False)
        command, duration = wp.get_command(centre, front)
        print(f"command: {command}")
        print(f"duration: {duration} s")
        print(c_new, f_new)
        c_new, f_new = predict_centre_front(centre, front, command, duration)
        print(c_new, f_new)
        bbox = get_bbox_from_centre_front(c_new, f_new)
        drawMarkers(image, bbox, (0,255,0), False)
        wp.draw(image)
        cv2.imshow("image", image)
    
    cv2.namedWindow("image")
    cv2.setMouseCallback('image', click_event)
    while True:
        redraw()
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            centre, front = c_new, f_new
        elif key == ord('q'):
            break
        # Testing cornering
        elif key == ord('i'):
            target_pos = untransform_board(shift, invmat, BLUE_BARRIER_T)
            offset = CORNER_LEFT_OFFSET
            orient = np.array((-1, 0))
        elif key == ord('o'):
            target_pos = untransform_board(shift, invmat, BLUE_BARRIER_T)
            offset = CORNER_RIGHT_OFFSET
            orient = np.array((0, -1))
        elif key == ord('k'):
            target_pos = untransform_board(shift, invmat, RED_BARRIER_T)
            offset = CORNER_LEFT_OFFSET
            orient = np.array((1, 0))
        elif key == ord('l'):
            target_pos = untransform_board(shift, invmat, RED_BARRIER_T)
            offset = CORNER_RIGHT_OFFSET
            orient = np.array((0, 1))

def _test_waypoint_list():
    from unfisheye import undistort
    from crop_board import remove_shadow, crop_board
    from find_coords import get_shift_invmat_mat
    from find_dots import getDots, getDotBbox, get_true_front, get_CofR, drawMarkers, get_outer_bbox
    # image = cv2.imread("dots/dot17.jpg")
    image = cv2.imread("dots/smol1.jpg")
    image = undistort(image)
    shift, invmat, _ = get_shift_invmat_mat(image)
    waypoints = []
    for w in np.concatenate((BLUE_CORNER_FORWARD_TS, BLUE_CORNER_BACKWARD_TS,
        BLUE_DROPOFFS_T), axis = 0):
        waypoints.append(untransform_board(shift, invmat, w))

    waypoint_index = 0

    image2 = image.copy()
    
    # Preprocess
    image2 = remove_shadow(image2)
    shift, invmat, mat = get_shift_invmat_mat(image2)
    image = crop_board(image, shift, invmat)

    untransform_board(shift, invmat, RED_CORNER_T)
    dots = getDots(image)
    found, bbox = getDotBbox(dots)
    if not found:
        assert False, "Dots not found!"
    centre, front = drawMarkers(image, bbox, (255,0,0))
    c_new, f_new = centre.copy(), front.copy()
    print(centre, front)
    original = image.copy()

    target_pos = np.array((100,100))

    def event(event, x, y, flags, params):
        nonlocal target_pos
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            target_pos = np.array((x,y))
            redraw()

    def trackbar_event(event):
        nonlocal waypoint_index, target_pos
        print("hello!")
        waypoint_index = cv2.getTrackbarPos("Waypoint index", "image")
        target_pos = waypoints[waypoint_index]
        redraw()

    def redraw():
        nonlocal target_pos, centre, front, c_new, f_new
        image = original.copy()
        cv2.drawMarker(image, np.int0(centre), (255,0,0), cv2.MARKER_CROSS, 30, 2)
        cv2.drawMarker(image, np.int0(front), (255,0,0), cv2.MARKER_CROSS, 30, 2)
        wp = Waypoint(target_pos=target_pos, target_orient=None, 
                  pos_tol=20, orient_tol=5, robot_offset=COFR_OFFSET,
                  move_backward_ok=False)
        command, duration = wp.get_command(centre, front)
        print(f"command: {command}")
        print(f"duration: {duration} s")
        print(c_new, f_new)
        c_new, f_new = predict_centre_front(centre, front, command, duration)
        print(c_new, f_new)
        bbox = get_bbox_from_centre_front(c_new, f_new)
        drawMarkers(image, bbox, (0,255,0), False)
        drawMarkers(image, get_outer_bbox(c_new, f_new), (0,0,255), False)
        wp.draw(image)
        cv2.imshow("image", image)
    
    cv2.namedWindow("image")
    cv2.createTrackbar('Waypoint index','image',0,len(waypoints)-1,trackbar_event)
    cv2.setMouseCallback('image', event)
    while True:
        redraw()
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            centre, front = c_new, f_new
        elif key == ord('q'):
            break


def _test_subroutine():
    from unfisheye import undistort
    from crop_board import remove_shadow, crop_board
    from find_coords import get_shift_invmat_mat
    from find_dots import getDots, getDotBbox, get_true_front, get_CofR, drawMarkers
    # image = cv2.imread("dots/dot17.jpg")
    image = cv2.imread("dots/smol1.jpg")
    image = undistort(image)
    image2 = image.copy()
    
    # Preprocess
    image2 = remove_shadow(image2)
    shift, invmat, mat = get_shift_invmat_mat(image2)
    image = crop_board(image, shift, invmat)

    untransform_board(shift, invmat, RED_CORNER_T)
    dots = getDots(image)
    found, bbox = getDotBbox(dots)
    if not found:
        assert False, "Dots not found!"
    centre, front = drawMarkers(image, bbox, (255,0,0))
    c_new, f_new = centre.copy(), front.copy()
    print(centre, front)
    original = image.copy()

    target_pos = np.array((100,100))
    offset = COFR_OFFSET
    orient = np.array((1,0))
    srt: Subroutine
    MAX_COMMAND_NUM = 2

    def click_event(event, x, y, flags, params):
        nonlocal target_pos, offset
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            target_pos = np.array((x,y))
            offset = COFR_OFFSET
            reset_srt()
            redraw()

    def reset_srt():
        nonlocal srt, target_pos, orient, offset
        srt = Subroutine([
            Waypoint(target_pos=target_pos, target_orient=orient, 
                  pos_tol=20, orient_tol=5, robot_offset=offset,
                  move_backward_ok=False),
            (
                CORNER_RIGHT if (offset==CORNER_RIGHT_OFFSET).all() else CORNER_LEFT,
                None,
                1/CORNER_SPEED,
                False
            )
        ], just_once=(True, True))


    def redraw():
        nonlocal target_pos, centre, front, c_new, f_new, offset, orient
        image = original.copy()
        cv2.drawMarker(image, np.int0(centre), (255,0,0), cv2.MARKER_CROSS, 30, 2)
        cv2.drawMarker(image, np.int0(front), (255,0,0), cv2.MARKER_CROSS, 30, 2)
        commands, _, durations, _ = srt.get_command_list(centre, front)
        print(f"commands: {commands}")
        print(f"durations: {durations}")
        c_new, f_new = centre, front
        for i, (command, duration) in enumerate(zip(commands, durations)):
            if i >= MAX_COMMAND_NUM: break
            c_new, f_new = predict_centre_front(c_new, f_new, command, duration)
        print(c_new, f_new)
        bbox = get_bbox_from_centre_front(c_new, f_new)
        drawMarkers(image, bbox, (0,255,0), False)
        srt.draw(image)
        cv2.imshow("image", image)
    
    cv2.namedWindow("image")
    cv2.setMouseCallback('image', click_event)
    reset_srt()
    while True:
        redraw()
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            centre, front = c_new, f_new
        elif key == ord('q'):
            break
        # Testing cornering
        elif key == ord('i'):
            target_pos = untransform_board(shift, invmat, BLUE_BARRIER_T)
            offset = CORNER_LEFT_OFFSET
            orient = np.array((-1, 0))
            reset_srt()
        elif key == ord('o'):
            target_pos = untransform_board(shift, invmat, BLUE_BARRIER_T)
            offset = CORNER_RIGHT_OFFSET
            orient = np.array((0, -1))
            reset_srt()
        elif key == ord('k'):
            target_pos = untransform_board(shift, invmat, RED_BARRIER_T)
            offset = CORNER_LEFT_OFFSET
            orient = np.array((1, 0))
            reset_srt()
        elif key == ord('l'):
            target_pos = untransform_board(shift, invmat, RED_BARRIER_T)
            offset = CORNER_RIGHT_OFFSET
            orient = np.array((0, 1))
            reset_srt()
if __name__ == "__main__":
    # _test_waypoint()
    # _test_waypoint_list()
    _test_subroutine()

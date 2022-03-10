import numpy as np
from robot_properties import *
from find_dots import untransform_coords, get_CofR, get_true_front, perp, angle
from find_coords import untransform_board
import cv2

_DEBUG = __name__ == "__main__"

class Waypoint:
    _target_pos: np.ndarray
    _target_orient: np.ndarray # which way the true forward dir should point, always unit length
    _pos_tol: float # pixels
    _near_tol: float # pixels
    _orient_tol: float # radians
    _orient_backward_ok: bool
    _move_backward_ok: bool

    # The position in the robot's coordinate system which we want to get to 
    # target_pos
    _robot_offset: np.ndarray

    def __init__(self, target_pos: np.ndarray, target_orient: np.ndarray = None, 
                 pos_tol: float = 10, orient_tol: float = 5, robot_offset: np.ndarray = np.array((0,0)), 
                 orient_backward_ok: bool = False, move_backward_ok: bool = True, near_tol: float = -1):
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
        near_tol : float
            Threshold for reversing a little bit instead of doing large turns, default -1
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

    def get_command(self, centre: np.ndarray, front: np.ndarray):
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
                    command = RIGHT if rot > 0 else LEFT
                    return command, abs(rot) / ROTATION_SPEED

                # If we got here, then no reversing / turning was needed.
                # Should now just head to the point.
                trans = self._get_translation_align(centre, front)
                command = FORWARD if trans > 0 else BACKWARD
                return command, abs(trans) / MOVEMENT_SPEED

            # Make sure the robot is facing the right way
            rot = self._get_rotation_align_target(centre, front)
            # If the rotation required is large, and we're near the target
            abs_dist = self._get_abs_distance(centre, front)
            if rot > np.radians(30) and abs_dist > self._pos_tol and abs_dist < self._near_tol:
                # Reverse a little
                return BACKWARD, self._near_tol / MOVEMENT_SPEED

            if abs(rot) > self._orient_tol:
                command = RIGHT if rot > 0 else LEFT
                return command, abs(rot) / ROTATION_SPEED

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
            return BACKWARD, self._near_tol / MOVEMENT_SPEED

        # Rotate first, then move.
        if abs(rot) > self._orient_tol:
            command = RIGHT if rot > 0 else LEFT
            return command, abs(rot) / ROTATION_SPEED
        if abs(trans) > self._pos_tol:
            command = FORWARD if trans > 0 else BACKWARD
            return command, abs(trans) / MOVEMENT_SPEED

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
    or (command == FORWARD_SLEW).all() or (command == BACKWARD_SLEW).all():
        direction = 1
        if (command == BACKWARD).all() or (command == BACKWARD_SLEW).all():
            direction = -1
        speed = MOVEMENT_SPEED
        if (command == FORWARD_SLEW).all() or (command == BACKWARD_SLEW).all():
            speed = MOVEMENT_SPEED_SLEW
        # Get unit forward dir
        f_t = get_true_front(centre, front) - centre
        f_t = f_t / np.linalg.norm(f_t)
        delta = f_t * duration * direction * speed
        return centre + delta, front + delta
    elif (command == LEFT).all() or (command == LEFT_SLEW).all()\
    or (command == RIGHT).all() or (command == RIGHT_SLEW).all():
        direction = 1 # +ve cw
        if (command == LEFT).all() or (command == LEFT_SLEW).all():
            direction = -1
        speed = ROTATION_SPEED
        if (command == LEFT_SLEW).all() or (command == RIGHT_SLEW).all():
            speed = ROTATION_SPEED_SLEW
        alpha = speed * duration * direction
        mat = get_rot_mat(alpha)
        m = get_CofR(centre, front)
        f_new = np.matmul(mat, front - m) + m
        c_new = np.matmul(mat, centre - m) + m
        return c_new, f_new
    else: return centre, front


# Generalises the idea of a list of actions to carry out
# Returns all the commands to be run at once.
class Subroutine:
    def __init__(self) -> None:
        # Grab a list of things to run
        # Store them
        pass

    def _draw(self):
        # Draw all the waypoints on
        pass

    def run(self):
        # Return all the commands to be sent
        pass

# Generalises multiple subroutines being grouped together
# Runs each of the routines one by one.
class Routine:
    def __init__(self) -> None:
        pass

# _T means transformed coordinates (relative to the yellow barriers)

BLUE_CORNER_T = np.array((-2.18769, 0.01154))
# Ordered towards pickup, first & last are routing waypoints
BLUE_CORNER_TS = np.array((
    (-1.28200, -0.67462),
    (-1.79988, -0.21200),
    (-1.80700, 0.15932),
    (-1.11507, 0.82523)
))
RED_CORNER_T = np.array((2.25079, -0.00796))
RED_CORNER_TS = np.array(( 
    (1.37965, -0.58829),
    (1.90121, -0.22039),
    (1.94007, 0.21116),
    (1.23903, 0.74794)
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
BLUE_POINT_PICKUP_T = np.array((-1.83500, 0.20022)) 
BLUE_POINT_DROPOFF_T = np.array((-1.83830, -0.23002))
# Near dropoff boxes
BLUE_POINT_BOX1_T = np.array((-0.82942, -0.98781))
BLUE_POINT_BOX2_T = np.array((-0.84395, -0.51035))

RED_POINT_PICKUP_T = np.array((1.89850, 0.24917))
RED_POINT_DROPOFF_T = np.array((1.88261, -0.25010))
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

    def click_event(event, x, y, flags, params):
        nonlocal target_pos, centre, front, c_new, f_new
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            target_pos = np.array((x,y))
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
        wp.draw(image)
        cv2.imshow("image", image)
    
    cv2.namedWindow("image")
    cv2.setMouseCallback('image', click_event)
    while True:
        redraw()
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            centre, front = c_new, f_new

if __name__ == "__main__":
    _test_waypoint()
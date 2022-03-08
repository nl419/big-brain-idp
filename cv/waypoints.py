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
    _orient_tol: float # radians
    _orient_backward_ok: bool
    _move_backward_ok: bool

    # The position in the robot's coordinate system which we want to get to 
    # target_pos
    _robot_offset: np.ndarray

    def __init__(self, target_pos: np.ndarray, target_orient: np.ndarray = None, 
                 pos_tol: float = 10, orient_tol: float = 5, robot_offset: np.ndarray = np.array((0,0)), 
                 orient_backward_ok: bool = False, move_backward_ok: bool = True):
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
            This parameter is only used if target_orient is None
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
            theta = angle(tm, forward)
            if self._move_backward_ok:
                # Limit rotations to between +-90 deg (+- pi/2 rad)
                return (theta + np.pi/2) % np.pi - np.pi/2
            # Limit rotations to between +-180 deg (+- pi rad)
            return (theta + np.pi) % (np.pi * 2) - np.pi

        theta = angle(tm, om_)
        # If we can reverse into the target, align with the reverse direction.
        direction = 1
        if theta < 0 or not self._move_backward_ok: # closer to the forward direction
            theta = -theta
            direction = -1
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
                # If nearly (but not quite) there, just reverse a bit.
                if dist < 20:
                    return BACKWARD, dist * 3 / MOVEMENT_SPEED

                rot = self._get_rotation_align_CofR(centre, front)
                # Tolerance for the intermediate rotation is fixed at 5 deg
                if abs(rot) > np.radians(5):
                    command = RIGHT if rot > 0 else LEFT
                    return command, abs(rot) / ROTATION_SPEED

                # If we got here, then no reversing / turning was needed.
                # Should now just head to the point.
                trans = self._get_translation_align(centre, front)
                command = FORWARD if trans > 0 else BACKWARD
                return command, abs(trans) / MOVEMENT_SPEED

            # Make sure the robot is facing the right way
            rot = self._get_rotation_align_target(centre, front)
            if abs(rot) > self._orient_tol:
                command = RIGHT if rot > 0 else LEFT
                return command, abs(rot) / ROTATION_SPEED

            # No actions needed.
            return None, 0
        
        # Should not align to a target orientation.
        rot = self._get_rotation_noalign(centre, front)
        if abs(rot) > self._orient_tol:
            command = RIGHT if rot > 0 else LEFT
            return command, abs(rot) / ROTATION_SPEED

        trans = self._get_translation_noalign(centre, front)
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


# _T means transformed coordinates (relative to the yellow barriers)

BLUE_CORNER_T = np.array((-2.18769, 0.01154))
# Ordered towards pickup, first & last are routing waypoints
BLUE_CORNER_TS = np.array((
    (-1.28200, -0.67462),
    (-1.80111, -0.14153),
    (-1.80187, 0.18293),
    (-1.11507, 0.82523)
))
RED_CORNER_T = np.array((2.25079, -0.00796))
RED_CORNER_TS = np.array(( 
    (1.37965, -0.58829),
    (1.89786, -0.19772),
    (1.93512, 0.16831),
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

# Generate waypoints to go to a blue dropoff location
def generate_blue_waypoints(image, shift, invmat, count):
    return 1

# Generate waypoints to go to a red dropoff location
def generate_red_waypoints(image, shift, invmat, count):
    # BROKEN DO NOT USE
    from find_coords import dropoff_boxes
    blues, reds = dropoff_boxes(image)
    red = reds[count]

    corner = untransform_board(shift, invmat, RED_CORNER_T)
    corner_pickup = untransform_board(shift, invmat, RED_POINT_PICKUP_T)
    corner_dropoff = untransform_board(shift, invmat, RED_POINT_DROPOFF_T)

    pickup = untransform_board(shift, invmat, PICKUP_CROSS_T)

    # Corner, dropoff, corner, pickup

    return (
        (
            Waypoint(red)
        )
    )



def generate_waypoints(shift, invmat):

    BLUE_CORNER = untransform_board(shift, invmat, BLUE_CORNER_T)
    RED_CORNER = untransform_board(shift, invmat, RED_CORNER_T)
    PICKUP_CROSS = untransform_board(shift, invmat, PICKUP_CROSS_T)
    DROPOFF_CROSS = untransform_board(shift, invmat, DROPOFF_CROSS_T)

    BRIDGE_MIDDLE = (PICKUP_CROSS + DROPOFF_CROSS) / 2

    BLUE_DROPOFFS = np.array((
        untransform_board(shift, invmat, BLUE_DROPOFFS_T[0]),
        untransform_board(shift, invmat, BLUE_DROPOFFS_T[1])
    ))
    RED_DROPOFFS = np.array((
        untransform_board(shift, invmat, RED_DROPOFFS_T[0]),
        untransform_board(shift, invmat, RED_DROPOFFS_T[1])
    ))

    # Intermediate waypoints
    BLUE_POINT_PICKUP = untransform_board(shift, invmat, BLUE_POINT_PICKUP_T)
    BLUE_POINT_DROPOFF = untransform_board(shift, invmat, BLUE_POINT_DROPOFF_T)
    BLUE_POINT_BOX1 = untransform_board(shift, invmat, BLUE_POINT_BOX1_T)
    BLUE_POINT_BOX2 = untransform_board(shift, invmat, BLUE_POINT_BOX2_T)

    RED_POINT_PICKUP = untransform_board(shift, invmat, RED_POINT_PICKUP_T)
    RED_POINT_DROPOFF = untransform_board(shift, invmat, RED_POINT_DROPOFF_T)
    RED_POINT_BOX1 = untransform_board(shift, invmat, RED_POINT_BOX1_T)
    RED_POINT_BOX2 = untransform_board(shift, invmat, RED_POINT_BOX2_T)

    HOME = untransform_board(shift, invmat, HOME_T)

    ZEROS = np.array((0,0))
    MECHANISM = np.array((1,0)) # Location of mechanism relative to robot pov

    # Lists of waypoints to execute after a block is obtained
    BLUE_WAYPOINTS = (
        (
            Waypoint(target_pos=BLUE_POINT_PICKUP, 
                    target_orient=(BLUE_CORNER-BLUE_POINT_PICKUP),
                    pos_tol=25, orient_tol=15, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=BLUE_CORNER, 
                    pos_tol=5, orient_tol=5, robot_offset=ZEROS, 
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=BLUE_POINT_DROPOFF, 
                    pos_tol=25, orient_tol=15, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=BLUE_POINT_BOX1,
                    target_orient=(BLUE_DROPOFFS[0]-BLUE_POINT_BOX1),
                    pos_tol=5, orient_tol=5, robot_offset=MECHANISM,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=BLUE_DROPOFFS[0],
                    pos_tol=2, orient_tol=2, robot_offset=MECHANISM,
                    orient_backward_ok=False, move_backward_ok=False),
        ),
        (
            Waypoint(target_pos=BLUE_POINT_PICKUP, 
                    target_orient=(BLUE_CORNER-BLUE_POINT_PICKUP),
                    pos_tol=25, orient_tol=15, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=BLUE_CORNER, 
                    pos_tol=5, orient_tol=5, robot_offset=ZEROS, 
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=BLUE_POINT_DROPOFF, 
                    pos_tol=25, orient_tol=15, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=BLUE_POINT_BOX2,
                    target_orient=(BLUE_DROPOFFS[1]-BLUE_POINT_BOX2),
                    pos_tol=5, orient_tol=5, robot_offset=MECHANISM,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=BLUE_DROPOFFS[1],
                    pos_tol=2, orient_tol=2, robot_offset=MECHANISM,
                    orient_backward_ok=False, move_backward_ok=False),
        )
    )

    RED_WAYPOINTS = (
        (
            Waypoint(target_pos=RED_POINT_PICKUP, 
                    target_orient=(RED_CORNER-RED_POINT_PICKUP),
                    pos_tol=25, orient_tol=15, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=RED_CORNER, 
                    pos_tol=5, orient_tol=5, robot_offset=ZEROS, 
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=RED_POINT_DROPOFF, 
                    pos_tol=25, orient_tol=15, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=RED_POINT_BOX1,
                    target_orient=(RED_DROPOFFS[0]-RED_POINT_BOX1),
                    pos_tol=5, orient_tol=5, robot_offset=MECHANISM,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=RED_DROPOFFS[0],
                    pos_tol=2, orient_tol=2, robot_offset=MECHANISM,
                    orient_backward_ok=False, move_backward_ok=False),
        ),
        (
            Waypoint(target_pos=RED_POINT_PICKUP, 
                    target_orient=(RED_CORNER-RED_POINT_PICKUP),
                    pos_tol=25, orient_tol=15, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=RED_CORNER, 
                    pos_tol=5, orient_tol=5, robot_offset=ZEROS, 
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=RED_POINT_DROPOFF, 
                    pos_tol=25, orient_tol=15, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=RED_POINT_BOX2,
                    target_orient=(RED_DROPOFFS[1]-RED_POINT_BOX2),
                    pos_tol=5, orient_tol=5, robot_offset=MECHANISM,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=RED_DROPOFFS[1],
                    pos_tol=2, orient_tol=2, robot_offset=MECHANISM,
                    orient_backward_ok=False, move_backward_ok=False),
        )
    )

    # 0 = go over bridge to pickup, 1 = go over bridge to dropoff
    BRIDGE_WAYPOINTS = (
        (
            Waypoint(target_pos=DROPOFF_CROSS,
                    target_orient=(BRIDGE_MIDDLE - DROPOFF_CROSS),
                    pos_tol=10, orient_tol=5, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=BRIDGE_MIDDLE,
                    target_orient=None,
                    pos_tol=40, orient_tol=5, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=PICKUP_CROSS,
                    target_orient=None,
                    pos_tol=40, orient_tol=5, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
        ),
        (
            Waypoint(target_pos=PICKUP_CROSS,
                    target_orient=(BRIDGE_MIDDLE - PICKUP_CROSS),
                    pos_tol=10, orient_tol=5, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=BRIDGE_MIDDLE,
                    target_orient=None,
                    pos_tol=40, orient_tol=5, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
            Waypoint(target_pos=DROPOFF_CROSS,
                    target_orient=None,
                    pos_tol=40, orient_tol=5, robot_offset=ZEROS,
                    orient_backward_ok=False, move_backward_ok=False),
        )
    )

    HOME_WAYPOINTS = (Waypoint(target_pos=HOME,
                            target_orient=(DROPOFF_CROSS - HOME),
                            pos_tol=10, orient_tol=5, robot_offset=ZEROS,
                            orient_backward_ok=False, move_backward_ok=True))
    
    return BLUE_WAYPOINTS, RED_WAYPOINTS, BRIDGE_WAYPOINTS, HOME_WAYPOINTS

def draw_points(image, shift, invmat):

    BLUE_CORNER = untransform_board(shift, invmat, BLUE_CORNER_T)
    RED_CORNER = untransform_board(shift, invmat, RED_CORNER_T)
    PICKUP_CROSS = untransform_board(shift, invmat, PICKUP_CROSS_T)
    DROPOFF_CROSS = untransform_board(shift, invmat, DROPOFF_CROSS_T)

    BRIDGE_MIDDLE = (PICKUP_CROSS + DROPOFF_CROSS) / 2

    BLUE_DROPOFFS = np.array((
        untransform_board(shift, invmat, BLUE_DROPOFFS_T[0]),
        untransform_board(shift, invmat, BLUE_DROPOFFS_T[1])
    ))
    RED_DROPOFFS = np.array((
        untransform_board(shift, invmat, RED_DROPOFFS_T[0]),
        untransform_board(shift, invmat, RED_DROPOFFS_T[1])
    ))

    # Intermediate waypoints
    BLUE_POINT_PICKUP = untransform_board(shift, invmat, BLUE_POINT_PICKUP_T)
    BLUE_POINT_DROPOFF = untransform_board(shift, invmat, BLUE_POINT_DROPOFF_T)
    BLUE_POINT_BOX1 = untransform_board(shift, invmat, BLUE_POINT_BOX1_T)
    BLUE_POINT_BOX2 = untransform_board(shift, invmat, BLUE_POINT_BOX2_T)

    RED_POINT_PICKUP = untransform_board(shift, invmat, RED_POINT_PICKUP_T)
    RED_POINT_DROPOFF = untransform_board(shift, invmat, RED_POINT_DROPOFF_T)
    RED_POINT_BOX1 = untransform_board(shift, invmat, RED_POINT_BOX1_T)
    RED_POINT_BOX2 = untransform_board(shift, invmat, RED_POINT_BOX2_T)

    HOME = untransform_board(shift, invmat, HOME_T)

    BLUES = [BLUE_CORNER, BLUE_DROPOFFS[0], BLUE_DROPOFFS[1],
             BLUE_POINT_PICKUP, BLUE_POINT_DROPOFF,
             BLUE_POINT_BOX1, BLUE_POINT_BOX2]
    REDS =  [RED_CORNER, RED_DROPOFFS[0], RED_DROPOFFS[1],
             RED_POINT_PICKUP, RED_POINT_DROPOFF,
             RED_POINT_BOX1, RED_POINT_BOX2]
    MISC =  [PICKUP_CROSS, DROPOFF_CROSS, BRIDGE_MIDDLE, HOME]

    for p in BLUES:
        cv2.drawMarker(image, np.int0(p), (255,0,0), cv2.MARKER_CROSS, 10, 2)
    for p in REDS:
        cv2.drawMarker(image, np.int0(p), (0,0,255), cv2.MARKER_CROSS, 10, 2)
    for p in MISC:
        cv2.drawMarker(image, np.int0(p), (0,255,0), cv2.MARKER_CROSS, 10, 2)
    return image

def _test_waypoint():
    from unfisheye import undistort
    from crop_board import remove_shadow, crop_board
    from find_coords import get_shift_invmat_mat
    from find_dots import getDots, getDotBbox, get_true_front, get_CofR
    from find_qr import drawMarkers
    image = cv2.imread("dots/dot3.jpg")
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
    print(centre, front)
    original = image.copy()

    target_pos = np.array((100,100))

    def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            new_pos = np.array((x,y))

            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            redraw(new_pos)

    def redraw(target_pos):
        image = original.copy()
        wp = Waypoint(target_pos=target_pos, target_orient=np.array((0,1)), 
                  pos_tol=5, orient_tol=5, robot_offset=np.array((0,1)),
                  move_backward_ok=True)
        command, duration = wp.get_command(centre, front)
        print(f"command: {command}")
        print(f"duration: {duration} s")
        if command[0] == FORWARD[0] and command[1] == FORWARD[1]\
        or command[0] == BACKWARD[0] and command[1] == BACKWARD[1]:
            dist = duration * MOVEMENT_SPEED
            if command[0] == BACKWARD[0]:
                dist = -dist
            f_true = get_true_front(centre, front) - centre
            end = centre + dist * (f_true / np.linalg.norm(f_true)) 
            print(f"corresponding distance: {dist}")
            cv2.line(image, np.int0(centre), np.int0(end), color=(0,0,255), thickness=2)
        else:
            alpha = duration * ROTATION_SPEED
            if command[0] == LEFT[0]:
                alpha = -alpha
            # Draw on the new centre / front position, as well as CofR
            m = get_CofR(centre, front)
            sin = np.sin(alpha)
            cos = np.cos(alpha)
            mat = np.array(((cos, -sin), (sin, cos)))
            cm = np.matmul(mat, centre - m)
            fm = np.matmul(mat, front - m)
            c_new = cm + m
            f_new = fm + m
            cv2.line(image, np.int0(c_new), np.int0(f_new), (0,0,255), 2)
            cv2.circle(image, np.int0(c_new), 5, (0,0,255), -1)
            cv2.circle(image, np.int0(f_new), 5, (0,0,255), -1)
            cv2.circle(image, np.int0(m), 5, (255,255,0), -1)
            print(f"corresponding angle: {np.degrees(alpha)}")
        wp.draw(image)
        cv2.imshow("image", image)
    
    redraw(target_pos)
    cv2.imshow("image", image)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)

def _test_points():
    from unfisheye import undistort
    from crop_board import remove_shadow, crop_board
    from find_coords import get_shift_invmat_mat
    image = cv2.imread("new_board/1.jpg")
    # image = cv2.imread("dots/dot3.jpg")
    # image = cv2.imread("checkerboard2/3.jpg")
    image = undistort(image)
    image2 = image.copy()
    
    # Preprocess
    image2 = remove_shadow(image2)
    shift, invmat, mat = get_shift_invmat_mat(image2)
    image = crop_board(image, shift, invmat)
    image = remove_shadow(image)


    draw_points(image, shift, invmat)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    _test_waypoint()
    # _test_points()
import numpy as np
from navigation import FORWARD, BACKWARD, LEFT, RIGHT, ROTATION_SPEED, MOVEMENT_SPEED
from find_dots import untransform_coords, get_CofR, get_true_front, perp, angle
from find_coords import untransform_board

class Waypoint:
    _target_pos: np.ndarray
    _target_orient: np.ndarray # which way the true forward dir should point, always unit length
    _pos_tol: float
    _orient_tol: float
    _orient_backward_ok: bool
    _move_backward_ok: bool

    # The position in the robot's coordinate system which we want to get to 
    # target_pos
    _robot_offset: np.ndarray

    def __init__(self, target_pos: np.ndarray, target_orient: np.ndarray or None, 
                 pos_tol: float, orient_tol: float, robot_offset: np.ndarray, 
                 orient_backward_ok: bool = False, move_backward_ok: bool = True):
        """A waypoint on the board. Run [Waypoint object].get_command(centre, front) 
        to get the next command & duration

        Parameters
        ----------
        target_pos : np.ndarray
            Coordinates (x,y) of the target
        target_orient : np.ndarray | None
            Vector (x,y) to align with the true front direction of the robot
        pos_tol : float
            Tolerance for being "at" the target position
        orient_tol : float
            Tolerance for being "parallel" to the target orientation
        robot_offset : np.ndarray
            The point on the robot to move to the target (in robot's coord system)
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
        self._orient_tol = orient_tol
        self._robot_offset = robot_offset
        self._orient_backward_ok = orient_backward_ok
        self._move_backward_ok = move_backward_ok

    def _get_rotation_noalign(self, centre, front):
        ### Assuming mag_tm is larger than mag_om_
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

        if np.isclose(mag_om_, 0): # o is in front of CofR
            theta = angle(tm, forward)
            if self._move_backward_ok:
                # Limit rotations to between +-90 deg (+- pi/2 rad)
                return (theta + np.pi/2) % np.pi - np.pi/2
            # Limit rotations to between +-180 deg (+- pi rad)
            return (theta + np.pi) % (np.pi * 2) - np.pi

        theta = angle(tm, om_)
        # If we can reverse into the target, align with the reverse direction.
        direction = 1 if theta > 0 and self._move_backward_ok else -1
        alpha = direction * np.arccos(mag_om_ / mag_tm) - theta
        # Limit rotations to between +-180 deg (+- pi rad)
        return (alpha + np.pi) % (np.pi * 2) - np.pi

    def _get_m_new(self, centre, front):
        # Find change in angle, and corresponding matrix
        # TODO: this is a bit inefficient, should really turn this into
        # a bunch of np.matmul()'s
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
        f_true = get_true_front(centre, front) - front
        alpha = angle(f_true, self._target_orient)

        if self._orient_backward_ok:
            # Limit rotation to +-90 deg (+- pi/2 rad)
            return (alpha + np.pi/2) % np.pi - np.pi/2
        return alpha

    def _get_rotation_align_CofR(self, centre, front):
        f_true = get_true_front(centre, front) - front

        # Find CofR's
        m = get_CofR(centre, front)
        m_new = self._get_m_new(centre, front)
        dm = m_new - m

        return angle(f_true, dm)

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
                    command = LEFT if rot > 0 else RIGHT
                    return command, abs(rot) / ROTATION_SPEED

                # If we got here, then no reversing / turning was needed.
                # Should now just head to the point.
                trans = self._get_translation_align(centre, front)
                command = FORWARD if trans > 0 else BACKWARD
                return command, abs(trans) / MOVEMENT_SPEED

            # Make sure the robot is facing the right way
            rot = self._get_rotation_align_target(centre, front)
            if abs(rot) > self._orient_tol:
                command = LEFT if rot > 0 else RIGHT
                return command, abs(rot) / ROTATION_SPEED

            # No actions needed.
            return None, 0
        
        # Should not align to a target orientation.
        rot = self._get_rotation_noalign(centre, front)
        if abs(rot) > self._orient_tol:
            command = LEFT if rot > 0 else RIGHT
            return command, abs(rot) / ROTATION_SPEED

        trans = self._get_translation_noalign(centre, front)
        if abs(trans) > self._pos_tol:
            command = FORWARD if trans > 0 else BACKWARD
            return command, abs(trans) / MOVEMENT_SPEED

        return None, 0

BLUE_CORNER_T = np.array((-1.88769, 0.01154))
RED_CORNER_T = np.array((2.05079, -0.00796))
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

# Intermediate waypoints
BLUE_POINT_PICKUP_T = np.array((-1.43523, 0.49387))
BLUE_POINT_DROPOFF_T = np.array((-1.44936, -0.44925))
BLUE_POINT_BOX1_T = np.array((-0.82942, -0.98781))
BLUE_POINT_BOX2_T = np.array((-0.84395, -0.51035))

RED_POINT_PICKUP_T = np.array((1.47270, 0.48527))
RED_POINT_DROPOFF_T = np.array((1.46287, -0.51847))
RED_POINT_BOX1_T = np.array((0.79716, -1.04265))
RED_POINT_BOX2_T = np.array((0.80650, -0.55608))

HOME_T = np.array((-0.02170, -1.78574))

# =====================================================================
# =====================================================================



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


if __name__ == "__main__":
    wp = Waypoint(target_pos=np.array((100,100)), target_orient=None, 
                  pos_tol=1, orient_tol=1, robot_offset=np.array((1,0)),
                  move_backward_ok=False)
    centre = np.array((0,0))
    front = np.array((1,1))
    print(wp.get_command(centre, front))
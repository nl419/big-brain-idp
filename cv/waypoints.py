import numpy as np
from navigation import FORWARD, BACKWARD, LEFT, RIGHT, ROTATION_SPEED, MOVEMENT_SPEED
from find_dots import untransform_coords, get_CofR, get_true_front, perp, angle

# Realigning halfway will be done by making another waypoint halfway there
# with very sloppy position tolerance

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

    def __init__(self, target_pos: np.ndarray, target_orient: np.ndarray, 
                 pos_tol: float, orient_tol: float, robot_offset: np.ndarray, 
                 orient_backward_ok: bool = False, move_backward_ok: bool = True):
        """A waypoint on the board. Run [Waypoint object].get_command(centre, front) 
        to get the next command & duration

        Parameters
        ----------
        target_pos : np.ndarray
            Coordinates (x,y) of the target
        target_orient : np.ndarray
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
            Whether moving backward for large distances is ok, by default True
        """
        self._target_pos = target_pos
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


if __name__ == "__main__":
    wp = Waypoint(target_pos=np.array((0.95,1.05)), target_orient=None, 
                  pos_tol=1, orient_tol=1, robot_offset=np.array((1,0)),
                  move_backward_ok=False)
import numpy as np

FORWARD = np.array((-250,-255))
BACKWARD = -FORWARD
LEFT = np.array((100, -100))
RIGHT = -LEFT

GATE_UP = 105
GATE_DOWN = 45
GATE_OFFSET = np.array((0.8,0))

MOVEMENT_SPEED = 44     # Forward/Backward movement speed in px/s on unscaled image
ROTATION_SPEED = 2 * np.pi / 18      # Radians per second

PICKUP_OFFSET = np.array((1,0))
COFR_OFFSET = np.array((-1.32223476e+00, -1.94289029e-15))
TRUE_FRONT_OFFSET = np.array([3.42170111, 0.20508744])
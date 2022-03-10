import numpy as np

DOT_PATTERN_DIR = -1 # -1 = forward in the 

# Old motors:
# FORWARD = np.array((-255,-250))
# BACKWARD = -FORWARD
# LEFT = np.array((100, -100))
# RIGHT = -LEFT

# MOVEMENT_SPEED = 50     # Forward/Backward movement speed in px/s on unscaled image
# ROTATION_SPEED = 2 * np.pi / 14      # Radians per second

# New motors:
FORWARD_SLEW = np.array((-255,-255))
BACKWARD_SLEW = -FORWARD_SLEW
FORWARD = np.array((-150,-150))
BACKWARD = -FORWARD
LEFT_SLEW = np.array((255,-255))
RIGHT_SLEW = -LEFT_SLEW
LEFT = np.array((100, -100))
RIGHT = -LEFT

MOVEMENT_SPEED_SLEW = 150     # Forward/Backward movement speed in px/s on unscaled image
ROTATION_SPEED_SLEW = 2 * np.pi / 5      # Radians per second
MOVEMENT_SPEED = 50     # Forward/Backward movement speed in px/s on unscaled image
ROTATION_SPEED = 2 * np.pi / 15      # Radians per second

GATE_UP = 135
GATE_DOWN = 45
GATE_OFFSET = np.array((1.5,0))
MIDDLE_OFFSET = np.array((0,0))

SENSOR_OFFSET_DETECT = np.array((2.5,0.7))
SENSOR_OFFSET_NO_DETECT = np.array((3.5,0.7))

# PICKUP_OFFSET = np.array((1,0))
COFR_OFFSET = np.array([-1.23, -0.03]) # [-1.2981606  -0.05502733], [-1.24198103 -0.00497366], [-1.14363318  0.0148905 ], 
TRUE_FRONT_OFFSET = np.array([ 5.71991755, 0.078])
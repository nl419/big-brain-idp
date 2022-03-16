import numpy as np

DOT_PATTERN_DIR = -1 # -1 = forward in the 

#------------------

# # Old motors:
# FORWARD = np.array((-255,-250))
# BACKWARD = -FORWARD
# LEFT = np.array((255, -255))
# RIGHT = -LEFT

# # Move forward and left/right, designed for going around the corner.
# CORNER_LEFT = np.array((-100, -255))
# CORNER_RIGHT = np.array((-255, -100))

# MOVEMENT_SPEED = 50     # Forward/Backward movement speed in px/s on unscaled image
# ROTATION_SPEED = 2 * np.pi / 4      # Radians per second
# CORNER_SPEED = 1 / 3.5 # Corners per second

# # NOT USED

# FORWARD_FINE = np.array((-150,-150))
# BACKWARD_FINE = -FORWARD_FINE
# LEFT_FINE = np.array((100,-100))
# RIGHT_FINE = -LEFT_FINE
# MOVEMENT_SPEED_FINE = 50     # Forward/Backward movement speed in px/s on unscaled image
# ROTATION_SPEED_FINE = 2 * np.pi / 15      # Radians per second


# COFR_OFFSET = np.array([-1.23, -0.03]) # [-1.2981606  -0.05502733], [-1.24198103 -0.00497366], [-1.14363318  0.0148905 ], 
# TRUE_FRONT_OFFSET = np.array([5.48090966, 0.25370332])
# # Centres of rotation for cornering
# CORNER_LEFT_OFFSET = np.array([-1.22412327, -3.50819671])
# CORNER_RIGHT_OFFSET = np.array([-1.23345301,  3.69989283])


#------------------

# New motors:
FINE_THRESH = 0.7 # in seconds
FORWARD_FINE = np.array((155,150))
BACKWARD_FINE = -FORWARD_FINE
FORWARD = np.array((255,255))
BACKWARD = -FORWARD
LEFT_FINE = np.array((-150,150))
RIGHT_FINE = -LEFT_FINE
LEFT = np.array((-255,255))
RIGHT = -LEFT

# Move forward and left/right, designed for going around the corner.
CORNER_LEFT = np.array((100,255))
CORNER_RIGHT = np.array((255,100))

# was 25
MOVEMENT_SPEED_FINE = 30     # Forward/Backward movement speed in px/s on unscaled image
ROTATION_SPEED_FINE = 2 * np.pi / 6.85      # Radians per second
MOVEMENT_SPEED = 75     # Forward/Backward movement speed in px/s on unscaled image
ROTATION_SPEED = 2 * np.pi / 2.5      # Radians per second
CORNER_SPEED = 1 / 2.1 # Corners per second

LARGE_ROTATION_THRESH = 0.5

COFR_OFFSET = np.array([-1.1, 0]) # [-1.07271384  0.12336127], [-1.16548528 -0.327872  ]
TRUE_FRONT_OFFSET = np.array([9.12422037, 0.1491684 ])
CORNER_LEFT_OFFSET = np.array([-0.75782368, -3.76264503]) # 
CORNER_RIGHT_OFFSET = np.array([-1.0455142,  3.9454316]) #

#------------------

GATE_UP = 135
GATE_DOWN = 45
GATE_OFFSET = np.array((1.25,0))
MIDDLE_OFFSET = np.array((0,0))

SENSOR_OFFSET_DETECT = np.array((2.3,1))
SENSOR_OFFSET_NO_DETECT = np.array((3.5,1))

# Purely for debugging purposes
_BOT_SIZE = 2
OUTER_BBOX = np.array((
    (-_BOT_SIZE, -_BOT_SIZE),
    (-_BOT_SIZE, _BOT_SIZE),
    (_BOT_SIZE, _BOT_SIZE),
    (_BOT_SIZE, -_BOT_SIZE)
))
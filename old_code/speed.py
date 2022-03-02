"""Calculate speed in px/s given start, end, and time taken"""

import numpy as np


# Speed data:
# [1613  201] -> [1030  147] in 6.5s    # 90px/s
# [1594  242] -> [869 162] in 8.5s      # 85.8px/s
# [1595  284] -> [897 163] in 8s        # 88.6px/s
# Avg: 88.1px/s (global scale 2)

# point1 = np.array([1613, 201])
# point2 = np.array([1030, 147])
# time = 6.5

# point1 = np.array([1594, 242])
# point2 = np.array([869, 162])
# time = 8.5

point1 = np.array([1595, 284])
point2 = np.array([897, 163])
time = 8



print(np.linalg.norm(point2 - point1) / time)
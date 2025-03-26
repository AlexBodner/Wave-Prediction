import numpy as np
import cv2

# Load depth image (already in meters)}
im_path = "raw_depths/depth_raw_1741892282067347514.png"
depth = cv2.imread(im_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

# Load camera intrinsics
K = np.array([
    [393.2049865722656, 0.0, 328.3466796875],
    [0.0, 392.77886962890625, 244.46273803710938],
    [0.0, 0.0, 1.0]
])

height, width = depth.shape
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# Compute point cloud
points = []
for v in range(height):
    for u in range(width):
        z = depth[v, u]
        if z <= 0: continue  # skip invalid
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points.append([x, y, z])

points = np.array(points, dtype=np.float32)
np.savetxt("points.xyz", points)  # WASS accepts .xyz format

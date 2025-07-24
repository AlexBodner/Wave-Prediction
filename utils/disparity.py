import os
import numpy as np
import cv2

def depth_to_points(depth_compressed_msg, camera_info):
    # Deserialize the message
    msg = depth_compressed_msg
    data_bytes = bytes(msg.data)

    # Decode PNG with 12-byte prefix skip
    if len(data_bytes) <= 12:
        raise ValueError("Data too short for prefix skip")
    np_arr = np.frombuffer(data_bytes[12:], dtype=np.uint8)
    depth_array = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if depth_array is None:
        raise ValueError("Failed to decode depth image")


    # Convert from mm to meters 
    z = depth_array.astype(np.float32) / 1000.0
    # Get intrinsic matrix from camera_info
    K = np.array(camera_info.k).reshape(3, 3)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]


    height, width = depth_array.shape
    i, j = np.indices((height, width))

    # Calculate 3D points
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    mask = np.isfinite(points).all(axis=1) & (points[:, 2] <= 6.0) & (points[:, 2] > 0)
    points = points[mask]
    return points

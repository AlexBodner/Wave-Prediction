import os
import numpy as np
import cv2

import rclpy
from rosbags.highlevel import AnyReader
from pathlib import Path
import tf2_ros
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
from tf_transformations import quaternion_matrix, quaternion_from_matrix, translation_from_matrix
from rclpy.clock import ClockType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def process_bag(bag_path, fit_plane = True):

    tf_buffer = tf2_ros.Buffer(cache_time=rclpy.time.Duration(seconds=500000000))
    camera_info_msg = None

    imu_T_imu_optical = TransformStamped()

    with AnyReader([bag_path]) as reader:
        # Iterate over messages in the bag
        for connection, timestamp, rawdata in reader.messages():

            if connection.topic == '/tf' or connection.topic == '/tf_static':
                try:
                    # Deserialize the message
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    #print(msg.__dict__)

                    for transform_msg in msg.transforms:
                        transform = TransformStamped()

                        # rosbags is not the official rosbag library and returns messages as dictionaries.
                        if not isinstance(transform_msg.header.stamp, Time):
                            # If it's a dict or has sec/nanosec attributes
                            transform_msg.header.stamp = Time(
                                sec=getattr(transform_msg.header.stamp, 'sec', 0),
                                nanosec=getattr(transform_msg.header.stamp, 'nanosec', 0))
                            
                        transform.header.stamp = transform_msg.header.stamp                        #transform.header.stamp = transform_msg.header.stamp
                        transform.header.frame_id = transform_msg.header.frame_id
                        transform.child_frame_id = transform_msg.child_frame_id

                        transform.transform.translation.x = transform_msg.transform.translation.x
                        transform.transform.translation.y = transform_msg.transform.translation.y
                        transform.transform.translation.z = transform_msg.transform.translation.z
                        transform.transform.rotation.x = transform_msg.transform.rotation.x
                        transform.transform.rotation.y = transform_msg.transform.rotation.y
                        transform.transform.rotation.z = transform_msg.transform.rotation.z
                        transform.transform.rotation.w = transform_msg.transform.rotation.w

                        
                        if connection.topic == '/tf_static':
                            tf_buffer.set_transform_static(transform, 'sim_rosbag')
                        if connection.topic == '/tf':
                            tf_buffer.set_transform(transform, 'sim_rosbag')

                        if transform.header.frame_id == 'camera_imu_frame' and transform.child_frame_id == 'camera_imu_optical_frame':
                            imu_T_imu_optical = transform

                    #print(f"TF Message at {timestamp}: {msg.transforms}")
                except Exception as e:
                    print(f"Error processing TF message: {e}")

            if not camera_info_msg and connection.topic == '/camera/camera/depth/camera_info':
                try:
                    # Deserialize the camera info message
                    camera_info_msg = reader.deserialize(rawdata, connection.msgtype)
                    print(f"Camera Info at {timestamp}: {camera_info_msg}")
                except Exception as e:
                    print(f"Error processing camera info message: {e}")
                    raise e

            if camera_info_msg and connection.topic == '/camera/camera/depth/image_rect_raw/compressedDepth':

                try:
                    # Deserialize the message
                    depth_image_msg = reader.deserialize(rawdata, connection.msgtype)

                    try:
                        imu_T_depth_optical = tf_buffer.lookup_transform(
                                                target_frame='camera_imu_frame',
                                                source_frame='camera_depth_optical_frame',
                                                time=tf2_ros.Time()) #depth_image_msg.header.stamp)
                    except Exception as e:
                        print(f"Error looking up transform: {e}")
                        continue

                    try:
                        enu_T_imu_optical = tf_buffer.lookup_transform(
                                                target_frame='odom_enu',
                                                source_frame='camera_imu_optical_frame',
                                                time=tf2_ros.Time()) #depth_image_msg.header.stamp)
                    except Exception as e:
                        print(f"Error looking up transform: {e}")
                        continue
                    
                    # Concatenate transforms to get ENU to depth optical frame (transforms points from depth optical to ENU)
                    enu_T_depth_optical = transform_to_matrix(enu_T_imu_optical) @ np.linalg.inv(transform_to_matrix(imu_T_imu_optical)) @ transform_to_matrix(imu_T_depth_optical)

                    # Convert depth image to 3D points
                    points = depth_to_points(depth_image_msg, camera_info_msg)
                    visualize_points(points, coord_system = "camara")

                    # Stack points to a homogeneous coordinates matrix as columns of 4 elements (x,y,z,1)
                    points_hom = np.vstack([points.T, np.ones((1, points.T.shape[1]))]) # 4xN matrix
                    points_in_enu = (enu_T_depth_optical @ points_hom)[:3, :].T # Convert back to Nx3
                    visualize_points(points_in_enu, coord_system = "ENU")


                except Exception as e:
                    print(f"Error processing compressedDepth message: {e}")
                    raise e

def transform_to_matrix(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    translation = np.array([t.x, t.y, t.z])
    quaternion = np.array([q.x, q.y, q.z, q.w])
    matrix = quaternion_matrix(quaternion)
    matrix[0:3, 3] = translation
    return matrix

def matrix_to_transform(matrix, header):
    t = translation_from_matrix(matrix)
    q = quaternion_from_matrix(matrix)
    ts = TransformStamped()
    ts.header = header
    ts.transform.translation.x = t[0]
    ts.transform.translation.y = t[1]
    ts.transform.translation.z = t[2]
    ts.transform.rotation.x = q[0]
    ts.transform.rotation.y = q[1]
    ts.transform.rotation.z = q[2]
    ts.transform.rotation.w = q[3]
    return ts

def visualize_points(points, coord_system):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')


    # Plot a subsample if points are too many
    subsample = points[::max(1, len(points)//10000)]


    ax.scatter(subsample[:, 0], subsample[:, 1], subsample[:, 2],
                s=0.5, c=subsample[:, 2], cmap='jet', marker='.')

    #plt.axis('equal')
    plt.axis('auto')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Pointcloud in {coord_system} Coordinates')
    plt.show()

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


    # Convert from mm to meters if needed
    z = depth_array.astype(np.float32) / 1000.0
    z[z>5] = 2
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
    #print(points)
    return points

if __name__ == '__main__':
    bag_path = Path('datasets/mesa_desde_lejos/')
    process_bag(bag_path, fit_plane=True)
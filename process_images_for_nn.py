import os
import numpy as np

import rclpy
from rosbags.highlevel import AnyReader
from pathlib import Path
import tf2_ros
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time

from mpl_toolkits.mplot3d import Axes3D

from utils.disparity import depth_to_points
from utils.ros_rotations import *
from utils.compute_grid import generate_weighted_height_grid
import utils.fit_mean_plane as fit_mean_plane
from utils.visualize import visualize_points


def float_to_builtin_time(t: float) -> Time:
    secs = int(t)
    nanosecs = int((t - secs) * 1e9)
    return Time(sec=secs, nanosec=nanosecs)
def process_bag(bag_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    tf_buffer = tf2_ros.Buffer(cache_time=rclpy.time.Duration(seconds=500000000))
    camera_info_msg = None

    imu_T_imu_optical = TransformStamped()

    all_points = []
    depth_msgs = []
    timestamps = []

    with AnyReader([bag_path]) as reader:
        # Iterate over messages in the bag
        for connection, timestamp, rawdata in reader.messages():

            if connection.topic == '/tf' or connection.topic == '/tf_static':
                try:
                    # Deserialize the message
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    #print(msg.__dict__)

                    for transform_msg in msg.transforms:
                        transform = transform_msg_to_transformedStamp(transform_msg)
                        
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
                    ros_time = float_to_builtin_time(timestamp)

                    try:
                        imu_T_depth_optical = tf_buffer.lookup_transform(
                                                target_frame='camera_imu_frame',
                                                source_frame='camera_depth_optical_frame',
                                                time=ros_time) #depth_image_msg.header.stamp)
                    except Exception as e:
                        print(f"Error looking up transform: {e}")
                        continue

                    try:
                        enu_T_imu_optical = tf_buffer.lookup_transform(
                                                target_frame='odom_enu',
                                                source_frame='camera_imu_optical_frame',
                                                time=ros_time) #depth_image_msg.header.stamp)
                    except Exception as e:
                        print(f"Error looking up transform: {e}")
                        continue
                    
                    # Concatenate transforms to get ENU to depth optical frame (transforms points from depth optical to ENU)
                    enu_T_depth_optical = transform_to_matrix(enu_T_imu_optical) @ np.linalg.inv(transform_to_matrix(imu_T_imu_optical)) @ transform_to_matrix(imu_T_depth_optical)

                    # Convert depth image to 3D points
                    points = depth_to_points(depth_image_msg, camera_info_msg)
                    #visualize_points(points, coord_system = "Cam")

                    # Stack points to a homogeneous coordinates matrix as columns of 4 elements (x,y,z,1)
                    points_hom = np.vstack([points.T, np.ones((1, points.T.shape[1]))]) # 4xN matrix
                    points_in_enu = (enu_T_depth_optical @ points_hom)[:3, :].T # Convert back to Nx3
                    #visualize_points(points_in_enu, coord_system = "ENU")
                    all_points.append(points_in_enu)
                    depth_msgs.append(depth_image_msg)
                    timestamps.append(timestamp)
                except Exception as e:
                    print(f"Error processing compressedDepth message: {e}")
                    continue

    # Concatenar solo los primeros 10 frames para calcular el plano medio
    n_mean_plane = 10
    all_points_concat = np.concatenate(all_points[:n_mean_plane], axis=0)
    centroid, R, normal, u, v = fit_mean_plane.compute_plane_transform_ransac(all_points_concat,thresh=0.01,max_iterations=100)

    # Calcular min y max en cada coordenada (x, y, z)
    mins = np.min(all_points_concat, axis=0)
    maxs = np.max(all_points_concat, axis=0)
    print("Minimos (x, y, z):", mins)
    print("Maximos (x, y, z):", maxs)

    # Guardar el plano medio
    np.savez(os.path.join(output_folder, "mean_plane.npz"),
             centroid=centroid, R=R, normal=normal, u=u, v=v, mins=mins, maxs=maxs)

    # Para cada frame, transformar los puntos al sistema del plano medio y guardar el grid
    for points_in_enu, depth_image_msg, timestamp in zip(all_points, depth_msgs, timestamps):
        # POR AHORA USO POINT IN ENU PARA VISUALIZAR BIEN EL PLANO MEDIO
        #points_plane = fit_mean_plane.transform_to_plane_coords(points_in_enu, centroid, R)
        grid = generate_weighted_height_grid(points_in_enu, grid_size=(100,100), origin=(-7,-6), max_coords=(3,3),
                                             radius=0.2)
        #compute_grid(points, grid_size =(50,50), grid_origin= (-7,-6), grid_extent =(3,2),) #
        print("grid shape:", grid.shape)
        np.save(os.path.join(output_folder, f"grid_{timestamp}.npy"), grid)


if __name__ == '__main__':
    bag_path = Path('datasets/mesa_desde_lejos/')
    process_bag(bag_path, output_folder= 'numpy_grids')
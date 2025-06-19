import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image, CameraInfo, Imu
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from tf2_ros import TransformException
import rclpy
from tf2_ros.buffer_interface import TransformStamped
from mpl_toolkits.mplot3d import Axes3D


import matplotlib.pyplot as plt
import fit_mean_plane
import pyvista as pv


def visualize_points(points, coord_system):
   fig = plt.figure(figsize=(10, 7))
   ax = fig.add_subplot(111, projection='3d')


   # Plot a subsample if points are too many
   subsample = points[::max(1, len(points)//10000)]


   ax.scatter(subsample[:, 0], subsample[:, 1], subsample[:, 2],
              s=0.5, c=subsample[:, 2], cmap='jet', marker='.')


   ax.set_xlabel('X (m)')
   ax.set_ylabel('Y (m)')
   ax.set_zlabel('Z (m)')
   ax.set_title(f'Pointcloud in {coord_system} Coordinates')
   plt.show()
def read_rosbag(bag_path):
   storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
   converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
   reader = SequentialReader()
   reader.open(storage_options, converter_options)


   type_map = {}
   messages = {'/camera/camera/depth/image_rect_raw/compressedDepth': [],
               '/camera/camera/color/image_raw/compressed': [],  # <-- Add RGB compressed topic here
               '/camera/camera/depth/camera_info': [],
               '/imu/data': [],
               '/tf': [],
               '/tf_static': []}


   while reader.has_next():
       topic, data, t = reader.read_next()
       if topic not in messages:
           continue
       if topic not in type_map:
           type_map[topic] = get_message(reader.get_all_topics_and_types()[[t.name for t in reader.get_all_topics_and_types()].index(topic)].type)
       msg_type = type_map[topic]
       msg = deserialize_message(data, msg_type)
       messages[topic].append((t, msg))


   return messages


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


def get_transform(tf_messages, parent_frame, child_frame, timestamp):
   """
   Find the latest transform before the given timestamp for the parent->child frames.
   Returns the TransformStamped or None if not found.
   """
   latest_transform = None
   latest_time = None


   for t, msg in reversed(tf_messages):
       if t > timestamp:
           continue  # Ignore transforms after requested time
       for transform in msg.transforms:
           if transform.header.frame_id == parent_frame and transform.child_frame_id == child_frame:
               if latest_time is None or t > latest_time:
                   latest_transform = transform
                   latest_time = t
   return latest_transform
def apply_transform(points, transform):
   # Apply translation and rotation to the points
   translation = np.array([transform.transform.translation.x,
                           transform.transform.translation.y,
                           transform.transform.translation.z])
   rotation = R.from_quat([transform.transform.rotation.x,
                           transform.transform.rotation.y,
                           transform.transform.rotation.z,
                           transform.transform.rotation.w])
   rotated_points = rotation.apply(points)
   transformed_points = rotated_points + translation
   return transformed_points


def process_bag(bag_path, fit_plane = True):
   messages = read_rosbag(bag_path)
   depth_images = messages['/camera/camera/depth/image_rect_raw/compressedDepth']
   camera_infos = messages['/camera/camera/depth/camera_info']
   imu_data = messages['/imu/data']
   tf_messages = messages['/tf'] + messages['/tf_static']


   # Fixed transform chain
   chain = [
       'camera_depth_optical_frame',
       'camera_depth_frame',
       'camera_link',
       'camera_gyro_frame',
       'camera_imu_frame',
       'camera_imu_optical_frame'
   ]


   # Compute composite transform once
   try:
       transform_cam_to_imu = get_transform_along_chain(chain, tf_messages)
   except ValueError as e:
       print(f"Failed to compute transform chain: {e}")
       return


   camera_info = camera_infos[0][1]
   i = 0
   for t, depth_image in depth_images:
       i+=1
       if i<3:
           continue
       # Convert depth image to 3D points
       points = depth_to_points(depth_image, camera_info)
       visualize_points(points, coord_system = "camara")


       # Apply transform from camera to imu optical frame
       points_in_imu = apply_transform(points, transform_cam_to_imu)
       visualize_points(points_in_imu, coord_system = "imu")


       # Get closest IMU orientation
       imu_orientation = None
       for imu_t, imu in imu_data:
           if imu_t >= t:
               imu_orientation = imu.orientation
               break
       if imu_orientation is None:
           continue


       # Apply IMU orientation to transform points to world frame
       print(imu_orientation)
       rotation = R.from_quat([imu_orientation.x,
                               imu_orientation.y,
                               imu_orientation.z,
                               imu_orientation.w])
       points_in_world = rotation.apply(points_in_imu)
       visualize_points(points_in_world, coord_system = "world")


       if fit_plane:
           # Crear grillado uniforme
           grid_points, mesh = fit_mean_plane.create_uniform_grid(
               points_in_world, grid_size=(50,50), spacing=0.1)
          
           # Calcular transformación al plano
           centroid, Rot, normal, u, v = fit_mean_plane.compute_plane_transform(grid_points)
          
           # Transformar puntos al sistema del plano
           points_plane = fit_mean_plane.transform_to_plane_coords(points_in_world, centroid, Rot)
          
           grid_plane = fit_mean_plane.transform_to_plane_coords(grid_points.points, centroid, Rot)
          
           # Normalizar grid
           grid_normalized, (mins, maxs) = fit_mean_plane.normalize_grid(
               grid_plane)
          
           # Extraer la matriz de puntos (n,3)
           raw = grid_points.points  # ndarray (n,3)




           # Visualización
           p = pv.Plotter()
           p.add_mesh(pv.PolyData(points_in_world), color="lightgray", opacity=0.3, label="Points in world")
           #p.add_mesh(grid, scalars=grid_normalized[...,2].ravel(), cmap="viridis",
          
           p.add_legend()
           p.show()
       visualize_points(points_in_world, coord_system = "world")
       print(f"Processed {len(points_in_world)} points at timestamp {t}")
       print(points_in_world)
       break
def build_tf_tree(tf_messages):
   tf_tree = {}
   for t, msg in tf_messages:
       for transform in msg.transforms:
           parent = transform.header.frame_id.strip('/')
           child = transform.child_frame_id.strip('/')
           tf_tree.setdefault(parent, {})[child] = transform
           tf_tree.setdefault(child, {})[parent] = invert_transform(transform)
   return tf_tree


def invert_transform(transform):
   inverse = TransformStamped()
   inverse.header.stamp = transform.header.stamp
   inverse.header.frame_id = transform.child_frame_id
   inverse.child_frame_id = transform.header.frame_id


   quat = [transform.transform.rotation.x,
           transform.transform.rotation.y,
           transform.transform.rotation.z,
           transform.transform.rotation.w]
   rot = R.from_quat(quat)
   inv_rot = rot.inv()


   trans = np.array([transform.transform.translation.x,
                     transform.transform.translation.y,
                     transform.transform.translation.z])
   inv_trans = -inv_rot.apply(trans)


   inverse.transform.translation.x = inv_trans[0]
   inverse.transform.translation.y = inv_trans[1]
   inverse.transform.translation.z = inv_trans[2]
   inv_quat = inv_rot.as_quat()
   inverse.transform.rotation.x = inv_quat[0]
   inverse.transform.rotation.y = inv_quat[1]
   inverse.transform.rotation.z = inv_quat[2]
   inverse.transform.rotation.w = inv_quat[3]
   return inverse


def get_transform_along_chain(chain, tf_messages):
   tf_tree = build_tf_tree(tf_messages)
   composed = np.eye(4)


   for i in range(len(chain) - 1):
       from_frame = chain[i]
       to_frame = chain[i + 1]


       if to_frame in tf_tree.get(from_frame, {}):
           transform = tf_tree[from_frame][to_frame]
       elif from_frame in tf_tree.get(to_frame, {}):
           transform = invert_transform(tf_tree[to_frame][from_frame])
       else:
           raise ValueError(f"Transform not found between {from_frame} and {to_frame}")


       t = np.array([transform.transform.translation.x,
                     transform.transform.translation.y,
                     transform.transform.translation.z])
       q = [transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w]
       T = np.eye(4)
       T[:3, :3] = R.from_quat(q).as_matrix()
       T[:3, 3] = t
       composed = composed @ T


   final = TransformStamped()
   final.header.frame_id = chain[0]
   final.child_frame_id = chain[-1]
   final.transform.translation.x = composed[0, 3]
   final.transform.translation.y = composed[1, 3]
   final.transform.translation.z = composed[2, 3]
   q_final = R.from_matrix(composed[:3, :3]).as_quat()
   final.transform.rotation.x = q_final[0]
   final.transform.rotation.y = q_final[1]
   final.transform.rotation.z = q_final[2]
   final.transform.rotation.w = q_final[3]
   return final


# Example usage
if __name__ == '__main__':
   bag_path = 'rosbag2_2025_06_19-15_23_07'
   process_bag(bag_path, fit_plane=True)








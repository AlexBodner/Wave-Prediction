from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
from tf_transformations import quaternion_matrix, quaternion_from_matrix, translation_from_matrix
import numpy as np
from builtin_interfaces.msg import Time
from rclpy.clock import ClockType
from builtin_interfaces.msg import Time as BuiltinTime

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
def transform_msg_to_transformedStamp(transform_msg):
    transform = TransformStamped()

    # rosbags is not the official rosbag library and returns messages as dictionaries.
    if not isinstance(transform_msg.header.stamp, Time):
        #print('sec=',getattr(transform_msg.header.stamp, 'sec', 0),'nanosec=',getattr(transform_msg.header.stamp, 'nanosec', 0))
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

                        
    return transform
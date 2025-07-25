from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
from transformations import quaternion_matrix, quaternion_from_matrix, translation_from_matrix
import numpy as np
from rclpy.clock import ClockType
from rclpy.time import Time
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

    # Extract or rebuild the stamp
    raw_stamp = transform_msg.header.stamp

    # Ensure it's a builtin_interfaces.msg.Time instance
    if not isinstance(raw_stamp, BuiltinTime):
        stamp = BuiltinTime()
        stamp.sec = getattr(raw_stamp, 'sec', 0)
        stamp.nanosec = getattr(raw_stamp, 'nanosec', 0)
    else:
        stamp = raw_stamp

    # Convert to rclpy.Time with clock_type for consistency
    ros_time = Time.from_msg(stamp, clock_type=ClockType.ROS_TIME)

    # Now set fields on the TransformStamped
    transform.header.stamp = ros_time.to_msg()  # Convert back to BuiltinTime
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
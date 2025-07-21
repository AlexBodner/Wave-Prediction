import os
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# Path to your ROS 2 bag directory
bag_path = 'bag_imu'

# Initialize the reader
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
reader.open(storage_options, converter_options)

# Retrieve topic types
topic_types = reader.get_all_topics_and_types()
type_map = {topic.name: topic.type for topic in topic_types}

# Specify the topic you want to read
target_topic = '/imu/data'
msg_type_str = type_map.get(target_topic)

if msg_type_str is None:
    print(f"Topic '{target_topic}' not found in the bag.")
    exit(1)

# Get the message type
msg_type = get_message(msg_type_str)

# Read messages
while reader.has_next():
    (topic, data, t) = reader.read_next()
    if topic == target_topic:
        msg = deserialize_message(data, msg_type)
        print(f"Timestamp: {t}")
        print(f"Orientation: x={msg.orientation.x}, y={msg.orientation.y}, z={msg.orientation.z}, w={msg.orientation.w}")
        print(f"Angular Velocity: x={msg.angular_velocity.x}, y={msg.angular_velocity.y}, z={msg.angular_velocity.z}")
        print(f"Linear Acceleration: x={msg.linear_acceleration.x}, y={msg.linear_acceleration.y}, z={msg.linear_acceleration.z}")
        print('---')

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

def read_tfs_from_rosbag(bag_path):
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {}
    all_tfs = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in type_map:
            type_map[topic] = get_message(reader.get_all_topics_and_types()[[t.name for t in reader.get_all_topics_and_types()].index(topic)].type)

        msg_type = type_map[topic]
        msg = deserialize_message(data, msg_type)   

        if topic in ['/tf', '/tf_static']:

            for transform in msg.transforms:
                all_tfs.append(transform)

    return all_tfs
import datetime
# Example usage
if __name__ == '__main__':
    bag_path = "datasets/barco_nordelta_realsense/realsense_records/rosbag2_2025_11_01-11_56_58"  # <- path to your .db3 folder
    tfs = read_tfs_from_rosbag(bag_path)
    
    for tf in tfs[:22]:  # Print first 5 for brevity
        sec = tf.header.stamp.sec
        nanosec = tf.header.stamp.nanosec
        timestamp = sec + nanosec * 1e-9
        readable_time = datetime.datetime.fromtimestamp(timestamp)

        print(f"[{readable_time}] TF from '{tf.header.frame_id}' to '{tf.child_frame_id}'")
        print(f"  Translation: x={tf.transform.translation.x:.6f}, "
            f"y={tf.transform.translation.y:.6f}, z={tf.transform.translation.z:.6f}")
        print(f"  Rotation:    x={tf.transform.rotation.x:.6f}, "
        f"y={tf.transform.rotation.y:.6f}, z={tf.transform.rotation.z:.6f}, "
        f"w={tf.transform.rotation.w:.6f}\n")
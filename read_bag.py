import sqlite3
import os
import cv2
import numpy as np
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
import numpy as np
import zlib

def connect(sqlite_file):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    return conn, c

def close(conn):
    conn.close()

def countRows(cursor, table_name, print_out=False):
    cursor.execute('SELECT COUNT(*) FROM {}'.format(table_name))
    count = cursor.fetchall()
    if print_out:
        print('\nTotal rows: {}'.format(count[0][0]))
    return count[0][0]

def getHeaders(cursor, table_name, print_out=False):
    cursor.execute('PRAGMA TABLE_INFO({})'.format(table_name))
    info = cursor.fetchall()
    if print_out:
        print("\nColumn Info:\nID, Name, Type, NotNull, DefaultVal, PrimaryKey")
        for col in info:
            print(col)
    return info

def getAllElements(cursor, table_name, print_out=False):
    cursor.execute('SELECT * from({})'.format(table_name))
    records = cursor.fetchall()
    if print_out:
        print("\nAll elements:")
        for row in records:
            print(row)
    return records

def isTopic(cursor, topic_name, print_out=False):
    boolIsTopic = False
    topicFound = []
    records = getAllElements(cursor, 'topics', print_out=False)
    for row in records:
        if row[1] == topic_name:
            boolIsTopic = True
            topicFound = row
    if print_out:
        if boolIsTopic:
            print('\nTopic named', topicFound[1], ' exists at id ', topicFound[0], '\n')
        else:
            print('\nTopic', topic_name, 'could not be found. \n')
    return topicFound

def getAllMessagesInTopic(cursor, topic_name, print_out=False):
    count = 0
    timestamps = []
    messages = []
    topicFound = isTopic(cursor, topic_name, print_out=False)
    if not topicFound:
        print('Topic', topic_name, 'could not be found. \n')
    else:
        records = getAllElements(cursor, 'messages', print_out=False)
        for row in records:
            if row[1] == topicFound[0]:
                count += 1
                timestamps.append(row[2])
                messages.append(row[3])
        if print_out:
            print('\nThere are ', count, 'messages in ', topicFound[1])
    return timestamps, messages

def getAllTopicsNames(cursor, print_out=False):
    topicNames = []
    records = getAllElements(cursor, 'topics', print_out=False)
    for row in records:
        topicNames.append(row[1])
    if print_out:
        print('\nTopics names are:')
        print(topicNames)
    return topicNames

def getAllMsgsTypes(cursor, print_out=False):
    msgsTypes = []
    records = getAllElements(cursor, 'topics', print_out=False)
    for row in records:
        msgsTypes.append(row[2])
    if print_out:
        print('\nMessages types are:')
        print(msgsTypes)
    return msgsTypes

def getMsgType(cursor, topic_name, print_out=False):
    msg_type = []
    topic_names = getAllTopicsNames(cursor, print_out=False)
    msgs_types = getAllMsgsTypes(cursor, print_out=False)
    for index, element in enumerate(topic_names):
        if element == topic_name:
            msg_type = msgs_types[index]
    if print_out:
        print('\nMessage type in', topic_name, 'is', msg_type)
    return msg_type
if __name__ == "__main__":
    # Path to the bag file
    bag_file = 'datasets/primera_captura_13_03/rosbag2_2025_03_13-15_57_34_0.db3'

    # Topic name for compressed images
    topic_name = '/camera/camera/depth/image_rect_raw/compressedDepth'

    # Output directory for saving images
    output_dir = 'extracted_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### Connect to the database
    conn, c = connect(bag_file)

    ### Get all topics names and types
    topic_names = getAllTopicsNames(c, print_out=False)
    print("Available topics:", topic_names)
    topic_types = getAllMsgsTypes(c, print_out=False)
    print("Topic types:", topic_types)

    # Create a map for quicker lookup
    type_map = {topic_names[i]: topic_types[i] for i in range(len(topic_types))}
    print("Type map:", type_map)

    # Verify the topic exists and its type
    if topic_name not in type_map:
        print(f"Error: Topic {topic_name} not found in bag file.")
        close(conn)
        exit(1)
    expected_type = 'sensor_msgs/msg/CompressedImage'
    if type_map[topic_name] != expected_type:
        print(f"Error: Topic {topic_name} has type {type_map[topic_name]}, expected {expected_type}")
        close(conn)
        exit(1)
    print(f"Message type for {topic_name}: {type_map[topic_name]}")

    ### Get all timestamps and messages
    t, msgs = getAllMessagesInTopic(c, topic_name, print_out=True)

    # Deserialize the message (CompressedImage type)
    msg_type = get_message(type_map[topic_name])

    # Process and save each image
    for timestamp, message in zip(t, msgs):
        try:
            # Print raw message details
            print(f"\nTimestamp: {timestamp}")
            print(f"Raw message length: {len(message)}")
            print(f"Raw message (first 50 bytes): {message[:50]}")

            # Deserialize the message
            deserialized_msg = deserialize_message(message, msg_type)

            # Debugging: Check the deserialized message
            print(f"Header frame_id: {deserialized_msg.header.frame_id}")
            print(f"Format: {deserialized_msg.format}")
            print(f"Data length: {len(deserialized_msg.data)}")
            if len(deserialized_msg.data) > 0:
                print(f"Data (first 10 bytes): {list(deserialized_msg.data[:10])}")
            else:
                print("Data is empty")

            # Skip if data is empty
            if not deserialized_msg.data:
                print(f"Skipping empty message at timestamp: {timestamp}")
                continue

            # Decompress the zlib-compressed data
            decompressed_data = zlib.decompress(deserialized_msg.data)
            print(f"Decompressed data length: {len(decompressed_data)}")

            # Convert to NumPy array (assuming 640x480, 16-bit depth)
            width, height = 640, 480  # Adjust if resolution differs
            expected_size = width * height * 2  # 2 bytes per pixel for uint16
            if len(decompressed_data) != expected_size:
                print(f"Warning: Decompressed size {len(decompressed_data)} doesn’t match {expected_size}")
                continue

            image = np.frombuffer(decompressed_data, dtype=np.uint16).reshape(height, width)

            # Save the image as 16-bit PNG
            output_path = os.path.join(output_dir, f"depth_image_{timestamp}.png")
            cv2.imwrite(output_path, image)
            print(f"Saved image: {output_path}")

        except Exception as e:
            print(f"Error processing message at timestamp {timestamp}: {e}")

    ### Close connection to the database
    close(conn)
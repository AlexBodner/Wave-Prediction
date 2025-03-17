import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path

# Path to your ROS 2 bag directory
bag_path = Path('bag_alex')

# Initialize the AnyReader
with AnyReader([bag_path]) as reader:
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/camera/camera/color/image_raw/compressed':
            try:
                print(f"\nTimestamp: {timestamp}")
                print("Raw data length:", len(rawdata))
                msg = reader.deserialize(rawdata, connection.msgtype)
                print(f"Message: {msg}")
                print(f"Data length: {len(msg.data)}")
                print(f"Data (first 10 bytes): {list(msg.data[:10])}")

                if not len(msg.data):
                    raise ValueError("Received an empty message.")

                # Convert msg.data to bytes
                data_bytes = bytes(msg.data)

                # Decode JPEG image (no need to skip bytes)
                np_arr = np.frombuffer(data_bytes, dtype=np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("JPEG decoding failed")

                print("Successfully decoded JPEG color image")
                print(f"Image shape: {image.shape}, dtype: {image.dtype}")

                # Save the color image
                output_path = f'extracted_images/color_image_{timestamp}.png'
                success = cv2.imwrite(output_path, image)
                if success:
                    print(f"Saved color image: {output_path}")
                else:
                    raise ValueError("Failed to save color image")

            except Exception as e:
                print(f"Error processing message on topic {connection.topic} at timestamp {timestamp}: {e}")
                print(f"Data (first 50 bytes): {list(data_bytes[:50])}")

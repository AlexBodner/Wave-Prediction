import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path

# Path to your ROS 2 bag directory
bag_path = Path('datasets/primera_captura_13_03')

# Initialize the AnyReader
with AnyReader([bag_path]) as reader:
    # Iterate over messages in the bag
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/camera/camera/depth/image_rect_raw/compressedDepth':
            try:
                # Deserialize the message
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

                # Decode PNG with 12-byte prefix skip
                if len(data_bytes) > 12:
                    np_arr = np.frombuffer(data_bytes[12:], dtype=np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise ValueError("PNG decoding failed after 12-byte prefix skip")
                    print("Successfully decoded as PNG (12-byte prefix skipped)")
                else:
                    raise ValueError("Data too short for prefix skip")

                # Check image properties
                print(f"Image shape: {image.shape}, dtype: {image.dtype}")
                print(f"Pixel stats - Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.2f}")

                # Save original 16-bit PNG
                #output_path_16bit = f'depth_image_{timestamp}_16bit.png'
                #success = cv2.imwrite(output_path_16bit, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                #if success:
                #    print(f"Saved 16-bit image: {output_path_16bit}")
                #else:
                #    raise ValueError("Failed to save 16-bit image")

                # Normalize to 8-bit for visualization
                if image.max() > 0:  # Avoid division by zero
                    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else:
                    normalized_image = np.zeros_like(image, dtype=np.uint8)
                    print("Warning: All pixel values are zero, normalized image will be black")
                
                # Save normalized 8-bit PNG
                output_path_8bit = f'extracted_images/depth_image_{timestamp}_8bit.png'
                success = cv2.imwrite(output_path_8bit, normalized_image)
                if success:
                    print(f"Saved 8-bit normalized image: {output_path_8bit}")
                else:
                    raise ValueError("Failed to save 8-bit image")

            except Exception as e:
                print(f"Error processing message on topic {connection.topic} at timestamp {timestamp}: {e}")
                print(f"Data (first 50 bytes): {list(data_bytes[:50])}")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path
import os

# Create output directory if it doesn't exist
output_dir = Path('extracted_depths')
output_dir.mkdir(exist_ok=True)

# Path to your ROS 2 bag directory
bag_path = Path('test_bag')

# Initialize the AnyReader
with AnyReader([bag_path]) as reader:
    # Iterate over messages in the bag
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/camera/camera/depth/image_rect_raw/compressedDepth':
            try:
                # Deserialize the message
                msg = reader.deserialize(rawdata, connection.msgtype)
                data_bytes = bytes(msg.data)

                # Decode PNG with 12-byte prefix skip
                if len(data_bytes) <= 12:
                    raise ValueError("Data too short for prefix skip")
                    
                np_arr = np.frombuffer(data_bytes[12:], dtype=np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                
                if image is None:
                    raise ValueError("PNG decoding failed")

                # Convert from mm to meters and handle invalid values (0)
                depth_meters = image.astype(np.float32) / 1000.0
                depth_meters[image == 0] = np.nan  # Mark invalid pixels as NaN

                # Create heatmap plot
                plt.figure(figsize=(10, 6))
                heatmap = plt.imshow(depth_meters, cmap='jet')
                plt.title(f"Depth Heatmap - {timestamp}")
                
                # Add colorbar with legend
                cbar = plt.colorbar(heatmap, shrink=0.8)
                cbar.set_label('Distance (meters)', rotation=270, labelpad=15)
                
                # Remove axes ticks
                plt.xticks([])
                plt.yticks([])
                
                # Save the figure
                output_path = output_dir / f'depth_heatmap_{timestamp}.png'
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                print(f"Saved heatmap with legend: {output_path}")

                # Optional: Save original 16-bit data
                raw_output_path = output_dir / f'depth_raw_{timestamp}.png'
                #cv2.imwrite(str(raw_output_path), image)
                #print(f"Saved raw 16-bit image: {raw_output_path}")

            except Exception as e:
                print(f"Error processing message: {e}")

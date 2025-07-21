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
bag_path = Path('datasets/mesa_desde_lejos')

def read_compressed_image_from_bag(bag_path, topic):
    """
    Reads compressed images from a ROS2 bag for a given topic.
    Returns a list of (timestamp, image) tuples.
    """
    images = []
    with AnyReader([bag_path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == topic:
                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    data_bytes = bytes(msg.data)
                    if len(data_bytes) <= 12:
                        continue
                    np_arr = np.frombuffer(data_bytes[12:], dtype=np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    if image is not None:
                        images.append((timestamp, image))
                except Exception as e:
                    print(f"Error reading image: {e}")
    return images

def find_closest_image_by_timestamp(images, target_timestamp):
    """
    Given a list of (timestamp, image) and a target timestamp,
    prints all timestamps and their difference in seconds,
    and returns (closest_timestamp, closest_image, min_diff_seconds).
    """
    if not images:
        print("No images to search.")
        return None, None, None
    diffs = []
    for ts, img in images:
        diff_sec = abs(ts - target_timestamp) / 1e9
        #print(f"Image timestamp: {ts}, diff to target: {diff_sec:.6f} seconds")
        diffs.append((diff_sec, ts, img))
    min_diff, closest_ts, closest_img = min(diffs, key=lambda x: x[0])
    print(f"Closest timestamp: {closest_ts}, diff: {min_diff:.6f} seconds")
    return closest_ts, closest_img, min_diff

if __name__ == "__main__":
    # Example usage for reading depth images
    depth_images = read_compressed_image_from_bag(bag_path, '/camera/camera/depth/image_rect_raw/compressedDepth')

    for timestamp, image in depth_images:
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
        
        #print(f"Saved heatmap with legend: {output_path}")

        # Optional: Save original 16-bit data
        raw_output_path = output_dir / f'depth_raw_{timestamp}.png'
        cv2.imwrite(str(raw_output_path), image)
        #print(f"Saved raw 16-bit image: {raw_output_path}")

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path
import gc
import os

# Create output directory if it doesn't exist
output_dir = Path('extracted_depths')
output_dir.mkdir(exist_ok=True)

# Path to your ROS 2 bag directory
bag_path = Path('/home/alex/Documents/Wave-Prediction/datasets/barco_nordelta_realsense/realsense_records/rosbag2_2025_11_01-11_56_58')
topic = '/camera/camera/depth/image_rect_raw/compressedDepth'

def save_colormap(depth_image, out_path):
    # depth_image is uint16 (mm). convert to meters float, mask zeros.
    depth = depth_image.astype(np.float32)
    mask = (depth == 0)
    depth[mask] = np.nan
    # compute robust min/max (ignore NaN)
    try:
        minv = np.nanpercentile(depth, 1)
        maxv = np.nanpercentile(depth, 99)
    except Exception:
        minv = np.nanmin(depth) if not np.isnan(np.nanmin(depth)) else 0.0
        maxv = np.nanmax(depth) if not np.isnan(np.nanmax(depth)) else 1.0
    if minv == maxv or np.isnan(minv) or np.isnan(maxv):
        # fallback grayscale empty
        color = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        norm = (depth - minv) / (maxv - minv)
        norm = np.clip(norm, 0.0, 1.0)
        norm_uint8 = ( (1.0 - norm) * 255 ).astype(np.uint8)  # invert so closer=hotter if desired
        color = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
        color[mask] = (0, 0, 0)
    cv2.imwrite(str(out_path), color)

def process_bag_stream(bag_path, topic, max_frames=None, skip=1):
    written = 0
    with AnyReader([bag_path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic != topic:
                continue
            if max_frames is not None and written >= max_frames:
                break
            try:
                msg = reader.deserialize(rawdata, connection.msgtype)
                data_bytes = bytes(msg.data)
                if len(data_bytes) <= 12:
                    continue
                np_arr = np.frombuffer(data_bytes[12:], dtype=np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)  # expected uint16
                if image is None:
                    continue
                # optional skipping
                if (written % skip) != 0 and max_frames is None:
                    written += 1
                    continue
                ts_str = str(timestamp)
                raw_out = output_dir / f"depth_raw_{ts_str}.png"
                cv2.imwrite(str(raw_out), image)  # saves 16-bit PNG

                color_out = output_dir / f"depth_colormap_{ts_str}.png"
                save_colormap(image, color_out)

                written += 1

                # free and hint GC
                del image
                gc.collect()
            except Exception as e:
                print("Read/process error:", e)
    return written
bag_path = Path('datasets/pileta')

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
    # limit frames when testing to avoid OOM
    total = process_bag_stream(bag_path, topic, max_frames=500, skip=1)
    print(f"Saved {total} frames to {output_dir}")

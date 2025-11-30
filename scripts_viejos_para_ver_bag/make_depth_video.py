import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path
import os
import shutil
import gc
import sys

# --- Configuration ---
bag_path = Path("/home/alex/Documents/Wave-Prediction/datasets/barco_nordelta_realsense/realsense_records/rosbag2_2025_11_01-11_56_58")
topic = "/camera/camera/depth/image_rect_raw/compressedDepth"

skip_frames = 5             # Only process 1 out of every N frames (fast forward)
fps = 30                    # Output video frame rate
output_video = "depth_fastforward.mp4"

# REMOVED: max_frames_to_read = 500 (Processing all frames now)
frames_dir = Path("tmp_frames_for_video") # Directory for ffmpeg fallback frames

# --- Environment Variables ---
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_OPENGL_SUPPORT"] = "0"

# ----------------------------------------------------------------------

def depth_to_color_frame(depth_image):
    """
    Converts 16-bit depth (mm) to a BGR frame using robust percentile normalization.
    Returns a BGR frame (np.uint8, 3-channel).
    """
    # 1. Prepare depth data and mask zeros
    depth = depth_image.astype(np.float32)
    mask = (depth == 0)
    depth[mask] = np.nan
    
    # 2. Compute robust min/max using percentiles
    try:
        minv = np.nanpercentile(depth, 1)
        maxv = np.nanpercentile(depth, 99)
    except Exception:
        minv = np.nanmin(depth) if not np.isnan(np.nanmin(depth)) else 0.0
        maxv = np.nanmax(depth) if not np.isnan(np.nanmax(depth)) else 1.0

    # 3. Handle invalid range
    if minv == maxv or np.isnan(minv) or np.isnan(maxv):
        color = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        # 4. Normalize (0.0 to 1.0)
        norm = (depth - minv) / (maxv - minv)
        norm = np.clip(norm, 0.0, 1.0)
        
        # 5. Convert to uint8 (0-255) and apply colormap (closer=hotter)
        norm_uint8 = ( (1.0 - norm) * 255 ).astype(np.uint8) 
        color = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
        
        # 6. Apply mask: ensure invalid/zero depth is black
        color[mask] = (0, 0, 0)
        
    return color

# ----------------------------------------------------------------------

def process_and_save_frames(bag_path, topic, frames_dir, skip_frames):
    """
    Reads frames one-by-one, converts them, and saves them directly to disk.
    This is the memory-optimized approach.
    """
    frames_dir.mkdir(exist_ok=True)
    written_count = 0
    read_count = 0
    
    print(f"Starting memory-optimized stream processing of frames...")
    
    with AnyReader([bag_path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic != topic:
                continue
            
            # --- Frame Skipping Check ---
            if read_count % skip_frames == 0:
                try:
                    # 1. Deserialize and Decode
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    data_bytes = bytes(msg.data)
                    if len(data_bytes) <= 12: 
                        read_count += 1
                        continue
                        
                    np_arr = np.frombuffer(data_bytes[12:], dtype=np.uint8)
                    depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED) 
                    
                    if depth_image is not None:
                        # 2. Convert to Color Frame
                        color_frame = depth_to_color_frame(depth_image)
                        
                        # 3. Write Directly to Disk
                        out_path = frames_dir / f"frame_{written_count:05d}.png"
                        cv2.imwrite(str(out_path), color_frame)
                        written_count += 1
                        
                        # Print progress
                        sys.stdout.write(f"Processed {read_count} frames, Saved {written_count} to disk. \r")
                        sys.stdout.flush()

                    # 4. Clean up memory for the current frame
                    del depth_image, color_frame
                    gc.collect()
                    
                except Exception as e:
                    print(f"\nRead/process error on frame {read_count}: {e}")
                    
            read_count += 1 # Increment total frames read (regardless of skip)

    print(f"\n✅ Finished processing stream. Total frames read: {read_count}, Frames saved for video: {written_count}")
    return written_count

# ----------------------------------------------------------------------

def finalize_video_with_ffmpeg(fps, output_video, frames_dir, written_count):
    """Creates the MP4 video using ffmpeg from the saved PNG files."""
    if written_count == 0:
        raise RuntimeError("No frames saved for ffmpeg creation.")
    
    print(f"Running ffmpeg to stitch {written_count} frames into: {output_video}")
    # Command to create the MP4 video
    cmd = f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {output_video}"
    
    print("Running command:", cmd)
    rc = os.system(cmd)
    
    if rc != 0:
        raise RuntimeError("ffmpeg failed to create video (return code != 0). Check the command output and ensure ffmpeg is installed.")
    
    print(f"✅ Created video via ffmpeg: {Path(output_video).resolve()}")
    
    # Cleanup
    try:
        shutil.rmtree(frames_dir)
        print("Cleaned up temporary frames directory.")
    except OSError as e:
        print(f"Warning: Error cleaning up temporary directory: {e}")

# ----------------------------------------------------------------------

if __name__ == "__main__":
    if not bag_path.exists():
        print(f"❌ Error: Bag path not found at {bag_path.resolve()}")
        exit()
        
    try:
        # 1. Process frames in a memory-efficient stream
        total_frames_written = process_and_save_frames(bag_path, topic, frames_dir, skip_frames)

        # 2. Finalize video creation using the saved frames
        finalize_video_with_ffmpeg(fps, output_video, frames_dir, total_frames_written)
        
        print("🎉 Video creation complete!")
        
    except RuntimeError as e:
        print(f"FATAL ERROR: {e}")
        # Clean up frames_dir on failure if it exists
        if frames_dir.exists():
             shutil.rmtree(frames_dir)
             print("Cleaned up temporary directory due to failure.")
        exit(1)
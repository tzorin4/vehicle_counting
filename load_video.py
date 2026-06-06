#!/usr/bin/env python3
"""
Extract frames from a YouTube video (or local video file) with time range and frame skipping.

Usage examples:
    # Download YouTube video and extract all frames from 00:01:30 to 00:02:45
    python yt_to_frames.py --url https://youtu.be/... --start 00:01:30 --end 00:02:45 --output frames

    # Extract every 5th frame only
    python yt_to_frames.py --url https://youtu.be/... --step 5 --output frames

    # Use a local video file instead of downloading
    python yt_to_frames.py --local video.mp4 --start 10 --end 20 --step 2
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
from tqdm import tqdm

def time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS or MM:SS or seconds string to float seconds."""
    parts = time_str.strip().split(':')
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def download_youtube_video(url: str, output_path: str) -> str:
    """Download YouTube video to a file using yt-dlp. Returns the file path."""
    print(f"Downloading video from {url} ...")
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]",
        "-o", output_path,
        url
    ]
    subprocess.run(cmd, check=True)
    print(f"Downloaded to {output_path}")
    return output_path

def extract_frames(video_path: str, output_dir: str, start_sec: float, end_sec: float,
                   step: int, max_frames: int = None):
    """Extract frames from video using OpenCV, respecting time range and step."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Determine frame indices
    start_frame = int(start_sec * fps) if start_sec > 0 else 0
    end_frame = int(end_sec * fps) if end_sec > 0 and end_sec < duration else total_frames - 1

    if start_frame >= total_frames:
        raise ValueError(f"Start time {start_sec}s exceeds video duration {duration:.2f}s")
    if end_frame > total_frames - 1:
        end_frame = total_frames - 1
        print(f"End time adjusted to video end (frame {end_frame})")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Prepare output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    saved_count = 0
    pbar = tqdm(total=end_frame - start_frame + 1, desc="Frames processed")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = start_frame + frame_count
        if current_frame > end_frame:
            break

        # Save only every `step`-th frame
        if frame_count % step == 0:
            filename = out_path / f"frame_{current_frame:08d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1
            if max_frames is not None and saved_count >= max_frames:
                break

        frame_count += 1
        pbar.update(1)
        if max_frames is not None and saved_count >= max_frames:
            break

    pbar.close()
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir} (step={step}, range={start_sec}s-{end_sec}s)")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from YouTube or local video with time range and step.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="YouTube video URL")
    group.add_argument("--local", help="Local video file path (skip download)")
    parser.add_argument("--start", default="0", help="Start timestamp (HH:MM:SS, MM:SS, or seconds). Default: 0")
    parser.add_argument("--end", default=None, help="End timestamp (HH:MM:SS, MM:SS, or seconds). Default: end of video")
    parser.add_argument("--step", type=int, default=1, help="Extract every N-th frame (default: 1 = all frames)")
    parser.add_argument("--output", required=True, help="Output directory for extracted frames (will be created)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to extract (useful for sampling)")
    parser.add_argument("--temp-dir", default=tempfile.gettempdir(), help="Directory for temporary downloaded video (default: system temp)")

    args = parser.parse_args()

    try:
        start_sec = time_to_seconds(args.start)
        end_sec = time_to_seconds(args.end) if args.end else None
    except ValueError as e:
        print(f"Time format error: {e}")
        sys.exit(1)

    # Obtain video file path
    if args.url:
        # Create a temporary filename
        temp_video = os.path.join(args.temp_dir, "yt_download_temp.mp4")
        try:
            video_path = download_youtube_video(args.url, temp_video)
        except subprocess.CalledProcessError:
            print("yt-dlp download failed. Ensure yt-dlp is installed (pip install yt-dlp).")
            sys.exit(1)
    else:
        video_path = args.local
        if not os.path.exists(video_path):
            print(f"Local video not found: {video_path}")
            sys.exit(1)

    # If end time not provided, we will process until the end (set end_sec = 0 meaning ignore)
    end_sec_for_extract = end_sec if end_sec is not None else 0

    try:
        extract_frames(video_path, args.output, start_sec, end_sec_for_extract,
                       args.step, args.max_frames)
    except Exception as e:
        print(f"Error during frame extraction: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary file if we downloaded
        if args.url and os.path.exists(temp_video):
            os.remove(temp_video)
            print("Cleaned up temporary video file.")

if __name__ == "__main__":
    main()
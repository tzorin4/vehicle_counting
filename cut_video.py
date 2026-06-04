#!/usr/bin/env python3
"""
Video cutter based on timestamps.
Usage:
    python cut_video.py --input input.mp4 --start 00:01:30 --end 00:02:45 --output output.mp4
    python cut_video.py -i video.mp4 -s 1:30 -e 2:45 -o clip.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path

def time_to_seconds(time_str: str) -> float:
    """
    Convert timestamp string (HH:MM:SS or MM:SS or SS) to seconds.
    Examples: '1:30:45', '02:15', '125', '1:02:30.5'
    """
    parts = time_str.strip().split(':')
    if len(parts) == 1:          # seconds only
        return float(parts[0])
    elif len(parts) == 2:        # minutes:seconds
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:        # hours:minutes:seconds
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS, MM:SS, or SS.")

def cut_video(input_path: str, start_time: str, end_time: str, output_path: str, reencode: bool = False):
    """
    Cut video from start_time to end_time using ffmpeg.
    If reencode is False, it uses fast copy (no re-encoding, but may have keyframe inaccuracies).
    If reencode is True, it re-encodes for precise cutting (slower).
    """
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    start_sec = time_to_seconds(start_time)
    end_sec = time_to_seconds(end_time)
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"End time ({end_time}) must be after start time ({start_time})")

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(input_file),
        "-ss", str(start_sec),   # seek to start position
        "-t", str(duration),     # duration to copy
        "-avoid_negative_ts", "make_zero",
    ]

    if reencode:
        # Re-encode to ensure accurate cut at non-keyframes (slower)
        cmd += ["-c:v", "libx264", "-c:a", "aac", "-preset", "fast"]
    else:
        # Copy codec (fast, but cut may start at previous keyframe)
        cmd += ["-c", "copy"]

    cmd.append(str(output_path))

    # Overwrite output if exists
    if Path(output_path).exists():
        cmd.insert(1, "-y")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr)
        sys.exit(1)
    else:
        print(f"Successfully cut video to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Cut a video using start and end timestamps.")
    parser.add_argument("-i", "--input", required=True, help="Path to input video file")
    parser.add_argument("-s", "--start", required=True, help="Start timestamp (e.g., 00:01:30, 1:30, 90)")
    parser.add_argument("-e", "--end", required=True, help="End timestamp (e.g., 00:02:45, 2:45, 165)")
    parser.add_argument("-o", "--output", required=True, help="Output video file path")
    parser.add_argument("--reencode", action="store_true",
                        help="Re-encode video for precise cutting (slower but accurate at any frame)")
    args = parser.parse_args()

    cut_video(args.input, args.start, args.end, args.output, args.reencode)

if __name__ == "__main__":
    main()
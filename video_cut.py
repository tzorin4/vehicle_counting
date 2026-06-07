#!/usr/bin/env python3
"""
video_cut.py — Download a video from a URL and cut a section from it.

Usage:
    python video_cut.py -o <output_file> -u <url> -s <start> -e <end>

Timestamps can be in mm:ss or hh:mm:ss format.

Examples:
    python video_cut.py -o clip.mp4 -u https://youtu.be/dQw4w9WgXcQ -s 1:23 -e 2:45
    python video_cut.py -o clip.mp4 -u https://youtu.be/dQw4w9WgXcQ -s 0:01:23 -e 0:02:45

Dependencies:
    pip install yt-dlp imageio-ffmpeg
"""

import sys
import os
import argparse
import tempfile
import subprocess
import yt_dlp


def parse_timestamp(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid timestamp '{ts}'. Use mm:ss or hh:mm:ss.")


def find_ffmpeg() -> str:
    """Find ffmpeg: prefer imageio_ffmpeg's bundled binary, fall back to system PATH."""
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.isfile(path):
            return path
    except ImportError:
        pass
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise RuntimeError("ffmpeg not found. Run: pip install imageio-ffmpeg")


def download_video(url: str, output_path: str) -> str:
    """
    Download a single pre-muxed file — no merging, no ffmpeg needed by yt-dlp.
    'best[ext=mp4]' picks the highest-quality single-file mp4 stream.
    """
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": output_path,
        "overwrites": True,
        "nopart": True,
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        actual = ydl.prepare_filename(info)

    if not os.path.isfile(actual):
        base = os.path.splitext(actual)[0]
        for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
            candidate = base + ext
            if os.path.isfile(candidate):
                return candidate

    return actual


def cut_video(ffmpeg_exe: str, input_path: str, output_path: str,
              start: float, end: float) -> None:
    duration = end - start
    cmd = [
        ffmpeg_exe, "-y",
        "-ss", f"{start:.3f}",
        "-i", input_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "18",
        output_path,
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg cut failed — see output above.")


def main():
    parser = argparse.ArgumentParser(
        description="Download a video and cut a section from it.")
    parser.add_argument("-o", "--output", required=True, help="Output file (e.g. clip.mp4)")
    parser.add_argument("-u", "--url",    required=True, help="Video URL")
    parser.add_argument("-s", "--start",  required=True, help="Start timestamp mm:ss or hh:mm:ss")
    parser.add_argument("-e", "--end",    required=True, help="End timestamp mm:ss or hh:mm:ss")
    parser.add_argument("--ffmpeg", help="Optional: full path to ffmpeg binary")
    args = parser.parse_args()

    try:
        start_sec = parse_timestamp(args.start)
        end_sec   = parse_timestamp(args.end)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    if start_sec >= end_sec:
        sys.exit(f"Error: start ({args.start}) must be before end ({args.end}).")

    print(f"Start  : {args.start} ({start_sec:.1f}s)")
    print(f"End    : {args.end} ({end_sec:.1f}s)")
    print(f"Output : {args.output}\n")

    if args.ffmpeg:
        ffmpeg = args.ffmpeg
        if not os.path.isfile(ffmpeg):
            sys.exit(f"Error: --ffmpeg path not found: {ffmpeg}")
    else:
        try:
            ffmpeg = find_ffmpeg()
        except RuntimeError as e:
            sys.exit(f"Error: {e}")

    print(f"ffmpeg : {ffmpeg}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = os.path.join(tmpdir, "downloaded.mp4")

        print("==> Downloading video (single-file format, no merging)...")
        try:
            actual_path = download_video(args.url, tmp_video)
        except yt_dlp.utils.DownloadError as e:
            sys.exit(f"\nDownload failed: {e}")

        if not os.path.isfile(actual_path):
            sys.exit(f"Error: downloaded file not found at '{actual_path}'.")

        size_mb = os.path.getsize(actual_path) / 1024 / 1024
        print(f"\nDownloaded: {actual_path} ({size_mb:.1f} MB)")

        if size_mb == 0:
            sys.exit("Error: file is 0 bytes. Try: pip install -U yt-dlp")

        print(f"\n==> Cutting {args.start} -> {args.end} ...")
        try:
            cut_video(ffmpeg, actual_path, args.output, start_sec, end_sec)
        except RuntimeError as e:
            sys.exit(f"Error: {e}")

    print(f"\nDone! Saved to: {args.output}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
YOLO inference benchmark script.
Times how long it takes to analyze a set of images with a YOLO model.

Usage:
    python yolo_benchmark.py --model <model_path> --input <image_dir> [--device cpu]
"""

import argparse
import time
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO inference time over a set of images."
    )
    parser.add_argument(
        "--model", required=True, help="Path to the YOLO model file (e.g. best.pt)"
    )
    parser.add_argument(
        "--input", required=True, help="Path to a directory of images or a single image file"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run inference on: 'cpu', '0', '0,1', etc. (default: cpu)"
    )
    return parser.parse_args()


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
    elif input_path.is_dir():
        images = sorted(
            p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        return images
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


def format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    args = parse_args()

    # --- Load model ---
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is not installed. Install it with: pip install ultralytics"
        )

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\n{'='*60}")
    print("  YOLO Inference Benchmark")
    print(f"{'='*60}")
    print(f"  Model  : {model_path}")
    print(f"  Device : {args.device}")

    print(f"\nLoading model...", end=" ", flush=True)
    load_start = time.perf_counter()
    model = YOLO(str(model_path))
    load_time = time.perf_counter() - load_start
    print(f"done ({load_time:.3f}s)")

    # --- Collect images ---
    input_path = Path(args.input)
    images = collect_images(input_path)

    if not images:
        print("\nNo supported images found. Supported formats:", ", ".join(SUPPORTED_EXTENSIONS))
        return

    total_size = sum(img.stat().st_size for img in images)

    print(f"\n{'─'*60}")
    print(f"  Images found : {len(images)}")
    print(f"  Total size   : {format_size(total_size)}")
    print(f"  Avg size     : {format_size(total_size // len(images))}")
    print(f"{'─'*60}")

    # Print per-image info
    print(f"\n  {'#':<6} {'Filename':<40} {'Size':>10}")
    print(f"  {'─'*6} {'─'*40} {'─'*10}")
    for i, img in enumerate(images, 1):
        size_str = format_size(img.stat().st_size)
        name = img.name if len(img.name) <= 40 else "…" + img.name[-39:]
        print(f"  {i:<6} {name:<40} {size_str:>10}")

    # --- Run inference ---
    print(f"\n{'─'*60}")
    print("  Running inference...\n")

    times = []
    for i, img in enumerate(images, 1):
        t0 = time.perf_counter()
        model.predict(str(img), device=args.device, verbose=False)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  [{i:>{len(str(len(images)))}}/{len(images)}] {img.name:<40} {elapsed*1000:>8.1f} ms")

    # --- Summary ---
    total_time = sum(times)
    avg_time = total_time / len(times)
    min_time = min(times)
    max_time = max(times)
    throughput = len(images) / total_time

    print(f"\n{'='*60}")
    print("  Benchmark Results")
    print(f"{'─'*60}")
    print(f"  Images processed : {len(images)}")
    print(f"  Total time       : {total_time:.3f} s")
    print(f"  Avg per image    : {avg_time*1000:.1f} ms")
    print(f"  Fastest image    : {min_time*1000:.1f} ms")
    print(f"  Slowest image    : {max_time*1000:.1f} ms")
    print(f"  Throughput       : {throughput:.2f} images/s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
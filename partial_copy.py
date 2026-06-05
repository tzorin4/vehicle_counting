#!/usr/bin/env python3
"""
Select first N image-label pairs (naturally sorted) and copy them to an output directory.

Usage:
    python select_pairs.py --images /path/to/images --labels /path/to/labels --output /path/to/output --num 500
    python select_pairs.py -i ./images -l ./labels -o ./selected -n 100
"""

import os
import re
import shutil
import argparse
from pathlib import Path

def natural_sort_key(path: str):
    """Return a sort key that orders numbers naturally (e.g., img2.jpg before img10.jpg)."""
    basename = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', basename)]

def find_matching_pairs(images_dir: str, labels_dir: str):
    """Find all image files (.jpg, .png) and matching label files (.txt, .xml) by basename."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # Collect all images
    images = list(images_dir.glob('**/*.jpg')) + list(images_dir.glob('**/*.png'))
    # Collect all labels
    labels = list(labels_dir.glob('**/*.txt')) + list(labels_dir.glob('**/*.xml'))

    # Build dictionaries keyed by basename (without extension)
    image_dict = {img.stem: img for img in images}
    label_dict = {lbl.stem: lbl for lbl in labels}

    # Find common basenames
    common = set(image_dict.keys()) & set(label_dict.keys())

    # Create list of (image_path, label_path) tuples
    pairs = [(image_dict[name], label_dict[name]) for name in common]

    # Sort by image path using natural sorting
    pairs.sort(key=lambda p: natural_sort_key(str(p[0])))
    return pairs

def copy_pairs(pairs, output_dir: str, num: int = -1):
    """Copy the first `num` pairs into output_dir/images and output_dir/labels."""
    out_dir = Path(output_dir)
    images_out = out_dir / 'images'
    labels_out = out_dir / 'labels'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    if num == -1:
        num = len(pairs)
    else:
        num = min(num, len(pairs))

    for i, (img_path, lbl_path) in enumerate(pairs[:num]):
        dest_img = images_out / img_path.name
        dest_lbl = labels_out / lbl_path.name
        shutil.copy2(img_path, dest_img)
        shutil.copy2(lbl_path, dest_lbl)
        if (i + 1) % 100 == 0:
            print(f"Copied {i+1}/{num} pairs")

    print(f"\nDone. Copied {num} image-label pairs to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Select first N naturally sorted image-label pairs and copy them.")
    parser.add_argument('-i', '--images', required=True, help='Directory containing images (.jpg, .png)')
    parser.add_argument('-l', '--labels', required=True, help='Directory containing labels (.txt, .xml)')
    parser.add_argument('-o', '--output', required=True, help='Output directory (creates subfolders images/ and labels/)')
    parser.add_argument('-n', '--num', type=int, default=-1,
                        help='Number of pairs to copy (default: all matching pairs)')
    args = parser.parse_args()

    if not os.path.isdir(args.images):
        print(f"Error: Images directory not found: {args.images}")
        return 1
    if not os.path.isdir(args.labels):
        print(f"Error: Labels directory not found: {args.labels}")
        return 1

    print("Finding matching image-label pairs...")
    pairs = find_matching_pairs(args.images, args.labels)
    if not pairs:
        print("No matching pairs found (image and label must share the same base name).")
        return 1

    print(f"Found {len(pairs)} matching pairs. Sorting naturally by image name...")
    copy_pairs(pairs, args.output, args.num)
    return 0

if __name__ == '__main__':
    exit(main())
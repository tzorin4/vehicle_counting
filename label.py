#!/usr/bin/env python3
"""
Run YOLO inference on images and save annotated frames + YOLO label files.
Usage: python label.py --input images/ --output out/ --model yolo26s.pt --conf 0.5
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO
from tqdm import tqdm
from test import natural_sort_key

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input folder with images")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument("--model", "-m", default="yolo26s.pt", help="Model weights")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", "-d", default="cpu", help="cpu or cuda:0")
    args = parser.parse_args()

    # Setup
    out_dir = Path(args.output)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    device = int(args.device) if args.device.isdigit() else args.device
    model.to(device)

    # Process images
    image_paths = sorted(Path(args.input).glob("*.jpg"), key=lambda p: natural_sort_key(str(p)))
    for img_path in tqdm(image_paths, desc="Processing"):
        # Inference
        results = model(img_path, conf=args.conf)[0]
        # Save annotated image (with boxes drawn)
        annotated = results.plot()
        out_img = out_dir / "images" / img_path.name
        cv2.imwrite(str(out_img), annotated)

        # Save label file (YOLO format)
        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            h, w = results.orig_shape
            label_path = out_dir / "labels" / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                for xyxy, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    x1, y1, x2, y2 = xyxy
                    xc = (x1 + x2) / 2 / w
                    yc = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")

if __name__ == "__main__":
    main()
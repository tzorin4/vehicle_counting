"""
YOLO Line-Crossing Object Counter — with per-class counts
Usage: python count_vehicles.py --video input.mp4 --model best.pt --output ./output

Improvements over original:
  - Flicker suppression: a crossing is only counted after an object has remained
    on the new side of the line for CONFIRM_FRAMES consecutive frames.
  - Proximity deduplication: new detections within MIN_PIXEL_DIST pixels of an
    existing track are merged into that track instead of spawning a new one.
  - Per-class crossing counts for classes 0-3 (car, motorbike, truck, bus),
    shown in the overlay and printed in the final summary.
"""

import argparse
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

# ── tuneable constants ────────────────────────────────────────────────────────
CONFIRM_FRAMES = 10   # frames object must stay on new side before crossing counts
MIN_PIXEL_DIST =  8   # detections closer than this (px) are merged, not spawned

# Classes to track individually (id -> display name)
CLASS_NAMES = {
    0: "car",
    1: "motorbike",
    2: "truck",
    3: "bus",
}
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO line-crossing counter")
    parser.add_argument("--video",   required=True, help="Path to input .mp4 file")
    parser.add_argument("--model",   required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--output",  required=True, help="Output directory for labels, images, and video")
    parser.add_argument("--line", nargs=4, type=float, metavar=("X1","Y1","X2","Y2"),
                        help="Counting line coords (default: horizontal 2/3-line)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames as images")
    parser.add_argument("--confirm-frames", type=int, default=CONFIRM_FRAMES,
                        help=f"Frames to confirm a crossing (default: {CONFIRM_FRAMES})")
    parser.add_argument("--min-pixel-dist", type=int, default=MIN_PIXEL_DIST,
                        help=f"Min px distance to merge detections (default: {MIN_PIXEL_DIST})")
    return parser.parse_args()


# ── geometry helpers ──────────────────────────────────────────────────────────

def side_of_line(point, a, b):
    """Which side of line AB is the point on? Returns +1 or -1."""
    return int(np.sign((b[0]-a[0])*(point[1]-a[1]) - (b[1]-a[1])*(point[0]-a[0])))


def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# ── tracker ───────────────────────────────────────────────────────────────────

class SimpleTracker:
    """
    Lightweight centroid tracker.
    Stores the most-recently-seen class ID alongside each track so that
    CrossingState can attribute confirmed crossings to the right class.
    """
    def __init__(self, max_dist=80, max_missing=10, min_spawn_dist=8):
        self.next_id        = 0
        self.objects        = {}   # id -> {centroid, cls_id, missing}
        self.max_dist       = max_dist
        self.max_missing    = max_missing
        self.min_spawn_dist = min_spawn_dist

    def update(self, detections):
        """
        detections: list of (centroid, cls_id) tuples
        Returns dict: id -> {centroid, cls_id, missing}
        """
        centroids = [d[0] for d in detections]
        cls_ids   = [d[1] for d in detections]

        if not self.objects:
            for c, cls in zip(centroids, cls_ids):
                self.objects[self.next_id] = {"centroid": c, "cls_id": cls, "missing": 0}
                self.next_id += 1
            return dict(self.objects)

        ids        = list(self.objects.keys())
        prev_cents = [self.objects[i]["centroid"] for i in ids]

        used_new = set()
        for i, pc in zip(ids, prev_cents):
            best_d, best_j = float("inf"), -1
            for j, nc in enumerate(centroids):
                if j in used_new:
                    continue
                d = np.hypot(nc[0]-pc[0], nc[1]-pc[1])
                if d < best_d:
                    best_d, best_j = d, j
            if best_d < self.max_dist and best_j >= 0:
                self.objects[i]["centroid"] = centroids[best_j]
                self.objects[i]["cls_id"]   = cls_ids[best_j]   # update class
                self.objects[i]["missing"]  = 0
                used_new.add(best_j)
            else:
                self.objects[i]["missing"] += 1

        for j, (nc, cls) in enumerate(zip(centroids, cls_ids)):
            if j in used_new:
                continue
            # proximity deduplication — absorb near-duplicate detections
            nearest_d = min(
                (np.hypot(nc[0]-v["centroid"][0], nc[1]-v["centroid"][1])
                 for v in self.objects.values()),
                default=float("inf")
            )
            if nearest_d < self.min_spawn_dist:
                continue
            self.objects[self.next_id] = {"centroid": nc, "cls_id": cls, "missing": 0}
            self.next_id += 1

        self.objects = {i: v for i, v in self.objects.items()
                        if v["missing"] <= self.max_missing}
        return dict(self.objects)


# ── crossing state machine ────────────────────────────────────────────────────

class CrossingState:
    """
    Per-track flicker-suppressed crossing detector.
    Returns (direction, cls_id) on confirmation, else None.
    """
    def __init__(self, confirm_frames=10):
        self.confirm_frames = confirm_frames
        self._state = {}
        # id -> {last_side, candidate_dir, frames_held, pending_cls}

    def _init_id(self, obj_id, side, cls_id):
        self._state[obj_id] = {
            "last_side":     side,
            "candidate_dir": None,
            "frames_held":   0,
            "pending_cls":   cls_id,
        }

    def update(self, obj_id, cent, cls_id, line_a, line_b):
        """
        Returns ("in"|"out", cls_id) when a crossing is confirmed, else None.
        """
        side = side_of_line(cent, line_a, line_b)
        if side == 0:
            return None

        if obj_id not in self._state:
            self._init_id(obj_id, side, cls_id)
            return None

        st = self._state[obj_id]
        # always keep cls_id fresh (handles brief misclassification frames)
        st["pending_cls"] = cls_id

        if st["candidate_dir"] is None:
            if side != st["last_side"]:
                st["candidate_dir"] = st["last_side"]
                st["frames_held"]   = 1
            return None

        if side == st["last_side"]:
            # bounced back → flicker, discard
            st["candidate_dir"] = None
            st["frames_held"]   = 0
            return None

        st["frames_held"] += 1

        if st["frames_held"] >= self.confirm_frames:
            came_from         = st["candidate_dir"]
            direction         = "in" if came_from == 1 else "out"
            confirmed_cls     = st["pending_cls"]
            st["last_side"]   = side
            st["candidate_dir"] = None
            st["frames_held"] = 0
            return direction, confirmed_cls

        return None

    def purge(self, active_ids):
        for dead_id in set(self._state) - set(active_ids):
            del self._state[dead_id]


# ── drawing helpers ───────────────────────────────────────────────────────────

def draw_line(frame, a, b):
    cv2.line(frame, a, b, (255, 180, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, a, 5, (255, 180, 0), -1)
    cv2.circle(frame, b, 5, (255, 180, 0), -1)
    cv2.putText(frame, "A", (a[0]+6, a[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, "B", (b[0]+6, b[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


def draw_overlay(frame, total, count_in, count_out, class_counts):
    """
    Overlay box showing total/in/out plus a per-class breakdown.
    class_counts: dict of cls_id -> {"in": n, "out": n}
    """
    # count how many class rows we'll actually draw (only classes seen so far)
    active_classes = [cid for cid in sorted(CLASS_NAMES) if sum(class_counts[cid].values()) > 0]
    n_rows  = 3 + len(active_classes)          # total / in / out + one per class
    box_h   = 16 + n_rows * 20
    box_w   = 190

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y = 28
    cv2.putText(frame, f"total : {total}",     (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1); y += 20
    cv2.putText(frame, f"in    : {count_in}",  (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,220,100), 1); y += 20
    cv2.putText(frame, f"out   : {count_out}", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,160,255), 1); y += 20

    for cid in sorted(CLASS_NAMES):
        cc = class_counts[cid]
        if cc["in"] + cc["out"] == 0:
            continue
        name = CLASS_NAMES[cid]
        text = f"  {name}: {cc['in']+cc['out']} (i{cc['in']} o{cc['out']})"
        cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        y += 20


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    out_dir    = Path(args.output)
    labels_dir = out_dir / "labels"
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    if args.save_frames:
        images_dir.mkdir(exist_ok=True)

    model = YOLO(args.model)
    cap   = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.line:
        line_a = (int(args.line[0]*W), int(args.line[1]*H))
        line_b = (int(args.line[2]*W), int(args.line[3]*H))
    else:
        line_a = (0, (H // 3) * 2)
        line_b = (W, (H // 3) * 2)
        print(f"No --line given, using horizontal 2/3-line: {line_a} -> {line_b}")

    video_out_path = out_dir / "output.mp4"
    fourcc         = cv2.VideoWriter_fourcc(*"mp4v")
    writer         = cv2.VideoWriter(str(video_out_path), fourcc, fps, (W, H))

    tracker        = SimpleTracker(min_spawn_dist=args.min_pixel_dist)
    crossing_state = CrossingState(confirm_frames=args.confirm_frames)

    count_total = count_in = count_out = 0
    # per-class counters: cls_id -> {"in": n, "out": n}
    class_counts = defaultdict(lambda: {"in": 0, "out": 0})

    frame_idx = 0
    print(f"Processing {total_frames} frames  "
          f"(confirm_frames={args.confirm_frames}, min_pixel_dist={args.min_pixel_dist}) ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]
        boxes   = results.boxes

        detections = []   # list of (centroid, cls_id)
        yolo_lines = []

        for box in boxes:
            xyxy   = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            c      = centroid(xyxy)
            detections.append((c, cls_id))

            cx_n = (xyxy[0] + xyxy[2]) / 2 / W
            cy_n = (xyxy[1] + xyxy[3]) / 2 / H
            bw_n = (xyxy[2] - xyxy[0]) / W
            bh_n = (xyxy[3] - xyxy[1]) / H
            yolo_lines.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (55, 138, 221), 2)
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.putText(frame, label, (xyxy[0], xyxy[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.circle(frame, c, 3, (55, 138, 221), -1)

        tracked = tracker.update(detections)

        # ── flicker-suppressed, class-aware crossing detection ────────────────
        for obj_id, obj in tracked.items():
            result = crossing_state.update(
                obj_id, obj["centroid"], obj["cls_id"], line_a, line_b
            )
            if result is not None:
                direction, cls_id = result
                count_total += 1
                if direction == "in":
                    count_in += 1
                    class_counts[cls_id]["in"] += 1
                else:
                    count_out += 1
                    class_counts[cls_id]["out"] += 1

        crossing_state.purge(tracked.keys())
        # ─────────────────────────────────────────────────────────────────────

        label_file = labels_dir / f"frame_{frame_idx:06d}.txt"
        label_file.write_text("\n".join(yolo_lines))

        draw_line(frame, line_a, line_b)
        draw_overlay(frame, count_total, count_in, count_out, class_counts)

        if args.save_frames:
            cv2.imwrite(str(images_dir / f"frame_{frame_idx:06d}.jpg"), frame)

        writer.write(frame)

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = frame_idx / total_frames * 100 if total_frames else 0
            print(f"  frame {frame_idx}/{total_frames} ({pct:.1f}%) "
                  f"| crossings: {count_total} (in:{count_in} out:{count_out})")

    cap.release()
    writer.release()

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\nDone.")
    print(f"  Total crossings : {count_total}")
    print(f"  In              : {count_in}")
    print(f"  Out             : {count_out}")
    print()
    print("  Per-class breakdown:")
    for cid, name in CLASS_NAMES.items():
        cc = class_counts[cid]
        total_cls = cc["in"] + cc["out"]
        print(f"    [{cid}] {name:<12}  total={total_cls}  in={cc['in']}  out={cc['out']}")
    print()
    print(f"  Output video    : {video_out_path}")
    print(f"  Labels          : {labels_dir}")
    if args.save_frames:
        print(f"  Images          : {images_dir}")


if __name__ == "__main__":
    main()
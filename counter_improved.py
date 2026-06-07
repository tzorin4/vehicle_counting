"""
YOLO Line-Crossing Object Counter
Usage: python count_vehicles.py --video input.mp4 --model best.pt --output ./output

Improvements over original:
  - Flicker suppression: a crossing is only counted after an object has remained
    on the new side of the line for CONFIRM_FRAMES consecutive frames.
  - Proximity deduplication: new detections within MIN_PIXEL_DIST pixels of an
    existing track are merged into that track instead of spawning a new one.
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── tuneable constants ────────────────────────────────────────────────────────
CONFIRM_FRAMES  = 10   # frames object must stay on new side before crossing counts
MIN_PIXEL_DIST  =  8   # detections closer than this (px) are merged, not spawned
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO line-crossing counter")
    parser.add_argument("--video", required=True, help="Path to input .mp4 file")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--output", required=True, help="Output directory for labels, images, and video")
    parser.add_argument("--line", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"),
                        help="Counting line coords (default: horizontal midline)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames as images")
    parser.add_argument("--confirm-frames", type=int, default=CONFIRM_FRAMES,
                        help=f"Frames to confirm a crossing (default: {CONFIRM_FRAMES})")
    parser.add_argument("--min-pixel-dist", type=int, default=MIN_PIXEL_DIST,
                        help=f"Min px distance to merge detections (default: {MIN_PIXEL_DIST})")
    return parser.parse_args()


def side_of_line(point, a, b):
    """Which side of line AB is the point on? Returns +1 or -1."""
    return int(np.sign((b[0]-a[0])*(point[1]-a[1]) - (b[1]-a[1])*(point[0]-a[0])))


def segments_cross(p1, p2, p3, p4):
    """Do segments p1-p2 and p3-p4 intersect?"""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


class SimpleTracker:
    """
    Lightweight centroid tracker.

    Two additions vs. the original:
      • min_spawn_dist  – a new detection is merged into the nearest existing
                          track when the distance is below this threshold,
                          preventing duplicate IDs for nearly-stationary objects.
      • (max_dist / max_missing are unchanged)
    """
    def __init__(self, max_dist=80, max_missing=10, min_spawn_dist=8):
        self.next_id        = 0
        self.objects        = {}   # id -> {centroid, missing}
        self.max_dist       = max_dist
        self.max_missing    = max_missing
        self.min_spawn_dist = min_spawn_dist

    def update(self, centroids):
        if not self.objects:
            for c in centroids:
                self.objects[self.next_id] = {"centroid": c, "missing": 0}
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
                self.objects[i]["missing"]  = 0
                used_new.add(best_j)
            else:
                self.objects[i]["missing"] += 1

        for j, nc in enumerate(centroids):
            if j in used_new:
                continue
            # ── proximity deduplication ──────────────────────────────────────
            # If this unmatched detection is very close to any existing track,
            # absorb it rather than creating a brand-new ID.
            nearest_d = min(
                (np.hypot(nc[0]-v["centroid"][0], nc[1]-v["centroid"][1])
                 for v in self.objects.values()),
                default=float("inf")
            )
            if nearest_d < self.min_spawn_dist:
                continue   # too close — skip, don't spawn
            # ────────────────────────────────────────────────────────────────
            self.objects[self.next_id] = {"centroid": nc, "missing": 0}
            self.next_id += 1

        self.objects = {i: v for i, v in self.objects.items()
                        if v["missing"] <= self.max_missing}
        return dict(self.objects)


class CrossingState:
    """
    Per-track state machine that suppresses flicker.

    Logic:
      1. When an object first crosses the line, record the candidate direction
         and start a confirmation counter.
      2. Each subsequent frame the object stays on the new side increments the
         counter.  If it flips back before CONFIRM_FRAMES, the candidate is
         discarded (flicker).
      3. Only when the counter reaches CONFIRM_FRAMES is the crossing committed
         to the running totals.
      4. After committing, the last_side is updated so a second real crossing
         in the opposite direction can be detected normally.
    """
    def __init__(self, confirm_frames=10):
        self.confirm_frames = confirm_frames
        # id -> {"last_side": int, "candidate_dir": int|None, "frames_held": int, "counted": bool}
        self._state = {}

    def _init_id(self, obj_id, side):
        self._state[obj_id] = {
            "last_side":     side,
            "candidate_dir": None,
            "frames_held":   0,
            "counted":       False,   # True while we're in 'pending' state
        }

    def update(self, obj_id, centroid, line_a, line_b):
        """
        Call once per frame for each tracked object.
        Returns ("in"|"out"|None) when a crossing is confirmed, else None.
        """
        side = side_of_line(centroid, line_a, line_b)
        if side == 0:
            return None   # exactly on the line — ignore

        if obj_id not in self._state:
            self._init_id(obj_id, side)
            return None

        st = self._state[obj_id]

        if st["candidate_dir"] is None:
            # No pending crossing — check if object just moved to a new side
            if side != st["last_side"]:
                st["candidate_dir"] = st["last_side"]  # remember which side it came FROM
                st["frames_held"]   = 1
            return None

        # There IS a pending crossing
        if side == st["last_side"]:
            # Object bounced back — flicker, discard candidate
            st["candidate_dir"] = None
            st["frames_held"]   = 0
            return None

        # Object is still on the new side
        st["frames_held"] += 1

        if st["frames_held"] >= self.confirm_frames:
            # ── confirmed! ──────────────────────────────────────────────────
            came_from   = st["candidate_dir"]   # +1 or -1
            direction   = "in" if came_from == 1 else "out"
            st["last_side"]     = side
            st["candidate_dir"] = None
            st["frames_held"]   = 0
            return direction

        return None

    def purge(self, active_ids):
        """Remove state for objects that are no longer tracked."""
        for dead_id in set(self._state) - set(active_ids):
            del self._state[dead_id]


def draw_line(frame, a, b):
    cv2.line(frame, a, b, (255, 180, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, a, 5, (255, 180, 0), -1)
    cv2.circle(frame, b, 5, (255, 180, 0), -1)
    cv2.putText(frame, "A", (a[0]+6, a[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, "B", (b[0]+6, b[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


def draw_overlay(frame, total, count_in, count_out):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (160, 72), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"total : {total}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    cv2.putText(frame, f"in    : {count_in}", (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,220,100), 1)
    cv2.putText(frame, f"out   : {count_out}", (14, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,160,255), 1)


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
        line_a = (args.line[0], args.line[1])
        line_b = (args.line[2], args.line[3])
    else:
        line_a = (0, (H // 3) * 2)
        line_b = (W, (H // 3) * 2)
        print(f"No --line given, using horizontal midline: {line_a} -> {line_b}")

    video_out_path = out_dir / "output.mp4"
    fourcc         = cv2.VideoWriter_fourcc(*"mp4v")
    writer         = cv2.VideoWriter(str(video_out_path), fourcc, fps, (W, H))

    tracker        = SimpleTracker(min_spawn_dist=args.min_pixel_dist)
    crossing_state = CrossingState(confirm_frames=args.confirm_frames)

    count_total = count_in = count_out = 0
    frame_idx   = 0

    print(f"Processing {total_frames} frames  "
          f"(confirm_frames={args.confirm_frames}, min_pixel_dist={args.min_pixel_dist}) ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]
        boxes   = results.boxes

        current_cents = []
        yolo_lines    = []

        for box in boxes:
            xyxy   = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            c      = centroid(xyxy)
            current_cents.append(c)

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

        tracked = tracker.update(current_cents)

        # ── flicker-suppressed crossing detection ────────────────────────────
        for obj_id, obj in tracked.items():
            result = crossing_state.update(obj_id, obj["centroid"], line_a, line_b)
            if result == "in":
                count_in    += 1
                count_total += 1
            elif result == "out":
                count_out   += 1
                count_total += 1

        crossing_state.purge(tracked.keys())
        # ─────────────────────────────────────────────────────────────────────

        label_file = labels_dir / f"frame_{frame_idx:06d}.txt"
        label_file.write_text("\n".join(yolo_lines))

        draw_line(frame, line_a, line_b)
        draw_overlay(frame, count_total, count_in, count_out)

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

    print(f"\nDone.")
    print(f"  Total crossings : {count_total}")
    print(f"  In              : {count_in}")
    print(f"  Out             : {count_out}")
    print(f"  Output video    : {video_out_path}")
    print(f"  Labels          : {labels_dir}")
    if args.save_frames:
        print(f"  Images          : {images_dir}")


if __name__ == "__main__":
    main()
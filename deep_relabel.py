#!/usr/bin/env python3
"""
Dataset relabeling tool for UA-DETRAC / YOLO labels.
Automatically skips images that need no manual review.
Navigation: ±1, ±30 images. Saves current progress on jump.
"""

import os
import sys
import glob
import shutil
import random
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import xml.etree.ElementTree as ET

# --- Configuration -------------------------------------------------
# Target classes (YOLO format)
TARGET_CLASSES = {0: "Car", 1: "Motorcycle", 2: "Truck", 3: "Bus"}

# UA-DETRAC string -> temporary numeric ID (before mapping)
UA_DETRAC_MAP = {'bus': 3, 'car': 1, 'van': 2, 'others': 4}

# Auto‑mapping: old_id -> target_id (from TARGET_CLASSES)
# Here, car(1) and van(2) become Car(0), bus(3) stays Bus(3).
# 'others' (4) will need manual review.
AUTO_MAP_RULES = {
    1: 0,   # car -> Car
    2: 0,   # van -> Car
    3: 3,   # bus -> Bus
    # 4 (others) remains unmapped -> needs review
}


class RelabelApp:
    def __init__(self, root, output_dir):
        self.root = root
        self.root.title("Dataset Relabel & Split Tool")
        self.root.geometry("1000x800")

        # State
        self.images_list = []
        self.current_idx = -1          # index in images_list
        self.current_boxes = []        # list of dicts: {id, bbox, needs_review}
        self.img_path = None
        self.img_w = self.img_h = 0
        self.cv_img = None

        # Directories
        self.img_dir = ""
        self.lbl_dir = ""
        self.output_dir = output_dir

        # Splitting
        self.train_split = 0.8
        self.max_images = 1000

        # GUI setup
        self.setup_ui()

    # -----------------------------------------------------------------
    def setup_ui(self):
        # Control panel (top)
        ctrl_frame = tk.Frame(self.root, pady=10)
        ctrl_frame.pack(fill=tk.X)

        tk.Button(ctrl_frame, text="Select Images Dir", command=self.load_img_dir).grid(row=0, column=0, padx=5)
        self.lbl_img_dir = tk.Label(ctrl_frame, text="No folder selected", width=30, anchor="w")
        self.lbl_img_dir.grid(row=0, column=1, padx=5)

        tk.Button(ctrl_frame, text="Select Labels Dir (XML or TXT)", command=self.load_lbl_dir).grid(row=0, column=2, padx=5)
        self.lbl_lbl_dir = tk.Label(ctrl_frame, text="No folder selected", width=30, anchor="w")
        self.lbl_lbl_dir.grid(row=0, column=3, padx=5)

        tk.Label(ctrl_frame, text="Train/Test Split (% Train):").grid(row=1, column=0, pady=5)
        self.entry_split = tk.Entry(ctrl_frame, width=5)
        self.entry_split.insert(0, "80")
        self.entry_split.grid(row=1, column=1, sticky="w")

        tk.Label(ctrl_frame, text="Shorten Dataset (Max Images):").grid(row=1, column=2, pady=5)
        self.entry_limit = tk.Entry(ctrl_frame, width=10)
        self.entry_limit.insert(0, "1000")
        self.entry_limit.grid(row=1, column=3, sticky="w")

        tk.Button(ctrl_frame, text="START PROCESSING", bg="green", fg="white",
                  font=("Arial", 10, "bold"), command=self.start_processing).grid(row=2, column=0, columnspan=4, pady=10)

        self.lbl_status = tk.Label(ctrl_frame, text="", fg="blue")
        self.lbl_status.grid(row=3, column=0, columnspan=4)

        # Image display canvas
        self.canvas = tk.Canvas(self.root, bg="black", width=800, height=500)
        self.canvas.pack(pady=10)
        self.tk_img = None

        # Navigation buttons
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(fill=tk.X, pady=5)
        tk.Button(nav_frame, text="<< Prev 30", width=12, command=lambda: self.jump(-30)).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Prev 1", width=12, command=lambda: self.jump(-1)).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next 1", width=12, command=lambda: self.jump(1)).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next 30 >>", width=12, command=lambda: self.jump(30)).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Save & Stay", width=12, command=self.save_current_image).pack(side=tk.RIGHT, padx=5)

        # Class assignment buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, pady=10)
        tk.Button(btn_frame, text="1. Set as CAR", bg="lightblue", width=15,
                  command=lambda: self.set_label(0)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="2. Set as MOTORCYCLE", bg="lightgreen", width=15,
                  command=lambda: self.set_label(1)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="3. Set as TRUCK", bg="lightyellow", width=15,
                  command=lambda: self.set_label(2)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="4. Set as BUS", bg="orange", width=15,
                  command=lambda: self.set_label(3)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="DELETE BOX (X)", bg="red", fg="white", width=15,
                  command=self.delete_box).pack(side=tk.RIGHT, padx=10)

        # Keyboard bindings
        self.root.bind('<KeyPress>', self.handle_keypress)

    # -----------------------------------------------------------------
    def handle_keypress(self, event):
        if event.char == '1':
            self.set_label(0)
        elif event.char == '2':
            self.set_label(1)
        elif event.char == '3':
            self.set_label(2)
        elif event.char == '4':
            self.set_label(3)
        elif event.char.lower() == 'x':
            self.delete_box()

    # -----------------------------------------------------------------
    def load_img_dir(self):
        self.img_dir = filedialog.askdirectory(title="Select Images Directory")
        self.lbl_img_dir.config(text=self.img_dir[-30:] if len(self.img_dir) > 30 else self.img_dir)

    def load_lbl_dir(self):
        self.lbl_dir = filedialog.askdirectory(title="Select Labels Directory")
        self.lbl_lbl_dir.config(text=self.lbl_dir[-30:] if len(self.lbl_dir) > 30 else self.lbl_dir)

    # -----------------------------------------------------------------
    def start_processing(self):
        if not self.img_dir or not self.lbl_dir:
            messagebox.showerror("Error", "Please select both image and label directories.")
            return

        # Create output folders
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)

        # Collect all images
        found = (glob.glob(os.path.join(self.img_dir, '**', '*.jpg'), recursive=True) +
                 glob.glob(os.path.join(self.img_dir, '**', '*.png'), recursive=True))
        if not found:
            messagebox.showerror("Error", "No images found.")
            return

        random.shuffle(found)
        try:
            self.max_images = int(self.entry_limit.get())
        except ValueError:
            self.max_images = 1000
        self.images_list = found[:self.max_images]

        try:
            self.train_split = float(self.entry_split.get()) / 100.0
        except ValueError:
            self.train_split = 0.8

        # Start from first image
        self.current_idx = 0
        self.load_current_image()

    # -----------------------------------------------------------------
    def load_current_image(self):
        """Load image at self.current_idx, apply auto‑mapping, and either
        skip automatically (if no review needed) or show for manual review."""
        if self.current_idx >= len(self.images_list):
            messagebox.showinfo("Done", f"Finished! Output in {self.output_dir}")
            self.root.quit()
            return

        self.img_path = self.images_list[self.current_idx]
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        self.lbl_status.config(text=f"Image {self.current_idx+1}/{len(self.images_list)}: {base_name}")

        # Read image for size
        self.cv_img = cv2.imread(self.img_path)
        if self.cv_img is None:
            print(f"Warning: cannot read {self.img_path}, skipping.")
            self.current_idx += 1
            self.load_current_image()
            return
        self.img_h, self.img_w = self.cv_img.shape[:2]

        # Parse labels (XML or TXT) and apply auto‑mapping
        self.current_boxes = self.parse_and_auto_map(base_name)

        # Check if any box still needs review
        needs_manual = any(box['needs_review'] for box in self.current_boxes)

        if not needs_manual:
            # Auto‑save and move to next image (skip display)
            self.save_current_image()
            self.current_idx += 1
            self.load_current_image()
        else:
            # Show for manual review
            self.current_box_idx = 0
            self.show_next_review_box()

    # -----------------------------------------------------------------
    def parse_and_auto_map(self, base_name):
        """Find label file (XML or TXT), parse boxes, apply AUTO_MAP_RULES.
        Returns list of dicts with keys: id, bbox, needs_review."""
        boxes = []

        # Search recursively for label file
        txt_path = next(iter(glob.glob(os.path.join(self.lbl_dir, '**', f"{base_name}.txt"), recursive=True)), None)
        xml_path = next(iter(glob.glob(os.path.join(self.lbl_dir, '**', f"{base_name}.xml"), recursive=True)), None)

        if xml_path:
            # UA-DETRAC XML parsing
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for target in root.findall('.//target'):
                box_elem = target.find('box')
                attr = target.find('attribute')
                if box_elem is not None and attr is not None:
                    try:
                        left = float(box_elem.get('left'))
                        top = float(box_elem.get('top'))
                        width = float(box_elem.get('width'))
                        height = float(box_elem.get('height'))
                        v_type = attr.get('vehicle_type', '').lower()

                        # Convert to YOLO format
                        x_c = (left + width/2) / self.img_w
                        y_c = (top + height/2) / self.img_h
                        w_norm = width / self.img_w
                        h_norm = height / self.img_h

                        old_id = UA_DETRAC_MAP.get(v_type, 4)  # 4 = 'others'
                        new_id = AUTO_MAP_RULES.get(old_id, old_id)
                        needs_review = (new_id == old_id and old_id not in AUTO_MAP_RULES) or (old_id == 4)
                        boxes.append({'id': new_id, 'bbox': [x_c, y_c, w_norm, h_norm], 'needs_review': needs_review})
                    except Exception:
                        pass

        elif txt_path:
            # YOLO TXT parsing
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        old_id = int(parts[0])
                        bbox = [float(p) for p in parts[1:]]
                        new_id = AUTO_MAP_RULES.get(old_id, old_id)
                        needs_review = (new_id == old_id and old_id not in AUTO_MAP_RULES)
                        boxes.append({'id': new_id, 'bbox': bbox, 'needs_review': needs_review})

        return boxes

    # -----------------------------------------------------------------
    def show_next_review_box(self):
        """Find the next box that still needs review and draw it."""
        while self.current_box_idx < len(self.current_boxes):
            if self.current_boxes[self.current_box_idx]['needs_review']:
                break
            self.current_box_idx += 1

        self.draw_canvas()

        if self.current_box_idx >= len(self.current_boxes):
            # All boxes reviewed -> save and move to next image
            self.save_current_image()
            self.current_idx += 1
            self.load_current_image()

    # -----------------------------------------------------------------
    def draw_canvas(self):
        """Draw the current image with all boxes. Active box in red."""
        display = self.cv_img.copy()
        for i, box in enumerate(self.current_boxes):
            xc, yc, w, h = box['bbox']
            x1 = int((xc - w/2) * self.img_w)
            y1 = int((yc - h/2) * self.img_h)
            x2 = int((xc + w/2) * self.img_w)
            y2 = int((yc + h/2) * self.img_h)

            if i == self.current_box_idx:
                # Active box – red
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(display, "???", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif not box['needs_review']:
                # Already reviewed – green
                label = TARGET_CLASSES.get(box['id'], str(box['id']))
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Boxes that still need review but are not active – could draw differently? Not needed.

        # Convert to RGB and fit canvas
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w = display.shape[:2]
        canvas_w, canvas_h = 800, 500
        scale = min(canvas_w / w, canvas_h / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            display = cv2.resize(display, (new_w, new_h))
        else:
            new_w, new_h = w, h

        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(display))
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=self.tk_img)

    # -----------------------------------------------------------------
    def set_label(self, class_id):
        """Assign the current box to a target class and move to next."""
        if self.current_box_idx < len(self.current_boxes):
            self.current_boxes[self.current_box_idx]['id'] = class_id
            self.current_boxes[self.current_box_idx]['needs_review'] = False
            self.current_box_idx += 1
            self.show_next_review_box()

    def delete_box(self):
        """Remove the current box entirely."""
        if self.current_box_idx < len(self.current_boxes):
            self.current_boxes.pop(self.current_box_idx)
            # Do not increment index – next box shifts into same index
            self.show_next_review_box()

    # -----------------------------------------------------------------
    def save_current_image(self):
        """Save the current image and its labels (YOLO format) to train/val folder."""
        # Determine split
        split = 'train' if random.random() <= self.train_split else 'val'

        base_name = os.path.basename(self.img_path)
        dest_img = os.path.join(self.output_dir, split, 'images', base_name)
        dest_lbl = os.path.join(self.output_dir, split, 'labels',
                                os.path.splitext(base_name)[0] + '.txt')

        # Copy image
        shutil.copy2(self.img_path, dest_img)

        # Write labels (only non‑deleted boxes)
        with open(dest_lbl, 'w') as f:
            for box in self.current_boxes:
                f.write(f"{box['id']} {' '.join(map(str, box['bbox']))}\n")

    # -----------------------------------------------------------------
    def jump(self, delta):
        """Jump delta images (±1 or ±30). Saves current image before moving."""
        if self.current_idx < 0:
            return
        # Save current progress (if any)
        self.save_current_image()
        new_idx = self.current_idx + delta
        if new_idx < 0:
            new_idx = 0
        if new_idx >= len(self.images_list):
            new_idx = len(self.images_list) - 1
        self.current_idx = new_idx
        self.load_current_image()


# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Vehicle dataset relabeling tool with auto‑skip.")
    parser.add_argument("--output", type=str, default="dataset_output",
                        help="Output directory for train/val split (default: dataset_output)")
    args = parser.parse_args()

    root = tk.Tk()
    app = RelabelApp(root, args.output)
    root.mainloop()


if __name__ == "__main__":
    main()
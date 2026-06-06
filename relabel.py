import os
import glob
import shutil
import random
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET
import re
from test import load_img_and_lbl, find_and_parse_labels, natural_sort_key

# --- TARGET CLASSES (YOLO MODEL) ---
# 0: Car, 1: Motorcycle, 2: Truck, 3: Bus
TARGET_CLASSES = {0: "Car", 1: "Motorcycle", 2: "Truck", 3: "Bus"}

# Map string labels from UA-Detrac to temporary numerical IDs for processing
UA_DETRAC_MAP = {'bus': 3, 'car': 1, 'van': 2, 'others': 4}

# Auto-skip mappings based on instructions (Map Car and Van -> target Car)
# It leaves 2 (Bus) and 3 (Others) for manual GUI review
# AUTO_MAP_RULES = {
#     1: 0, # Previous 1 (Car) -> Target 0 (Car)
#     2: 0, # van -> car
#     3: 3, #bus -> bus
# }

class RelabelApp:
    def __init__(self, root, output_dir="dataset_output_test"):
        self.root = root
        self.root.title("Dataset Relabel & Split Tool (YOLO & UA-Detrac)")
        self.root.geometry("1000x800")

        # State
        self.skip_var = tk.BooleanVar(value=True)
        self.images_list = []
        self.current_img_idx = -1
        self.images_processed = 0
        self.current_bboxes = []
        self.current_box_idx = -1
        self.default_skip = 30

        self.img_dir = ""
        self.lbl_dir = ""
        self.output_dir = output_dir
        self.setup_ui()

    def setup_ui(self):
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

        skip_frame = tk.Frame(ctrl_frame)
        skip_frame.grid(row=3, column=0, columnspan=4, pady=5)

        skip_check = tk.Checkbutton(skip_frame, text="Skip auto-mapped images", variable=self.skip_var)
        skip_check.pack(side=tk.LEFT, padx=5)

        tk.Label(skip_frame, text="Skip amount:").pack(side=tk.LEFT, padx=5)
        self.skip_amount_entry = tk.Entry(skip_frame, width=5)
        self.skip_amount_entry.insert(0, "30")
        self.skip_amount_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(ctrl_frame, text="START PROCESSING", bg="green", fg="white", font=("Arial", 10, "bold"),
                  command=self.start_processing).grid(row=2, column=0, columnspan=4, pady=10)
        
        self.lbl_status = tk.Label(ctrl_frame, text="", fg="blue")
        self.lbl_status.grid(row=4, column=0, columnspan=4)

        self.canvas = tk.Canvas(self.root, bg="black", width=800, height=500)
        self.canvas.pack(pady=10)
        self.img_on_canvas = None

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, pady=10)

        tk.Button(btn_frame, text="1. Set as CAR", bg="lightblue", width=15, command=lambda: self.set_label(0)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="2. Set as MOTORCYCLE", bg="lightgreen", width=15, command=lambda: self.set_label(1)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="3. Set as TRUCK", bg="lightyellow", width=15, command=lambda: self.set_label(2)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="4. Set as BUS", bg="orange", width=15, command=lambda: self.set_label(3)).pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="DELETE BOX (X)", bg="red", fg="white", width=15, command=self.delete_box).pack(side=tk.RIGHT, padx=10)
        # tk.Button(btn_frame, text="SKIP (Space)", bg="gray", fg="white", width=15, command=self.skip_box).pack(side=tk.RIGHT, padx=10) #TODO remove the initialised

        self.root.bind('<KeyPress>', self.handle_keypress)

    def handle_keypress(self, event):
        if event.char == '1': self.set_label(0)
        elif event.char == '2': self.set_label(1)
        elif event.char == '3': self.set_label(2)
        elif event.char == '4': self.set_label(3)
        elif event.char.lower() == 'x': self.delete_box()
        # elif event.space: self.skip_box()
    
    def load_img_dir(self):
        self.img_dir = filedialog.askdirectory(title="Select Images Directory")
        self.lbl_img_dir.config(text=self.img_dir[-30:])
    
    def load_lbl_dir(self):
        self.lbl_dir = filedialog.askdirectory(title="Select Labels Directory")
        self.lbl_lbl_dir.config(text=self.lbl_dir[-30:])
    

    

    def start_processing(self):
        if not self.img_dir or not self.lbl_dir:
            messagebox.showerror("Error", "Please select both image and label directories.")
            return

        self.images_list = load_img_and_lbl(self.img_dir, self.lbl_dir)
        self.create_output_dirs()

        if len(self.images_list) == 0:
            messagebox.showerror("Error", "No images found in the selected directory.")
            return
        
        try:
            self.train_split = float(self.entry_split.get()) / 100.0
        except ValueError:
            self.train_split = 0.8

        try:
            self.max_images = int(self.entry_limit.get())
        except ValueError:
            self.max_images = len(self.images_list)
        self.current_img_idx = 0
        self.process_current_image()
        
        
    def process_current_image(self):
        
        self.keep_processing = True
        self.image_loaded = False
        while not self.image_loaded and self.current_img_idx < len(self.images_list) and self.keep_processing:
            # print("here 1")
            self.load_image()
            

    def load_image(self): # TODO : check method
        if self.current_img_idx >= len(self.images_list) or self.current_img_idx < 0:
            messagebox.showinfo("Done", f"Finished processing! Files saved to {self.output_dir}")
            return

        self.img_path = self.images_list[self.current_img_idx][0]
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]

        self.lbl_path = self.images_list[self.current_img_idx][1]
        self.current_bboxes = find_and_parse_labels(self.lbl_path)
        self.images_processed += 1
        left_to_check = False
        self.lbl_status.config(text=f"img:{self.images_processed}/{self.max_images}, Processing: {base_name} ({self.current_img_idx+1}/{len(self.images_list)})")
        for box in self.current_bboxes:
            old_id = box['id']
            if old_id in AUTO_MAP_RULES:
                box['id'] = AUTO_MAP_RULES[old_id]
                box['needs_review'] = False # Skip car and van
            else:
                box['needs_review'] = True
                left_to_check = True
        # print("here 0")
        if not left_to_check and self.skip_var.get():
            # print("here 2")
            self.save_and_next()
            # print("here 3")
            self.next_image(skip_count=int(self.skip_amount_entry.get()))
            return
        
        self.image_loaded = True

        self.cv_img = cv2.imread(self.img_path)
        if self.cv_img is None:
            self.next_image()
            return

        self.img_h, self.img_w, _ = self.cv_img.shape
        
        self.current_box_idx = 0
        self.show_next_review_box()
 
    def save_and_next(self):
        split = 'train' if random.random() <= self.train_split else 'val'

        base_name = os.path.basename(self.img_path)
        dest_img_path = os.path.join(self.output_dir, split, 'images', base_name)
        dest_lbl_path = os.path.join(self.output_dir, split, 'labels', os.path.splitext(base_name)[0] + '.txt')
       
        shutil.copy2(self.img_path, dest_img_path)

        with open(dest_lbl_path, 'w') as f:
            for box in self.current_bboxes:
                f.write(f"{box['id']} {' '.join(map(str, box['bbox']))}\n")

        self.next_image(skip_count=self.default_skip)

    def next_image(self, skip_count=1):
        if self.current_img_idx + skip_count >= len(self.images_list) or self.current_img_idx + skip_count < 0:
            self.keep_processing = False
            messagebox.showinfo("Info", "No more images to process.")
            return
        self.current_img_idx += self.default_skip

    def delete_box(self):
        pass

    def show_next_review_box(self):
        while self.current_box_idx < len(self.current_bboxes):
            if self.current_bboxes[self.current_box_idx]['needs_review']:
                break
            self.current_box_idx += 1
        # print("here 4")
        self.draw_canvas()

        if self.current_box_idx >= len(self.current_bboxes):
            # print("here 5")
            self.save_and_next()
            self.process_current_image()

    def draw_canvas(self):
        display_img = self.cv_img.copy()

        # Draw boxes
        for i, box in enumerate(self.current_bboxes):
            x_c, y_c, bw, bh = box['bbox']
            x1 = int((x_c - bw / 2) * self.img_w)
            y1 = int((y_c - bh / 2) * self.img_h)
            x2 = int((x_c + bw / 2) * self.img_w)
            y2 = int((y_c + bh / 2) * self.img_h)

            if i == self.current_box_idx:
                # Active box (RED)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(display_img, "???", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif not box['needs_review']:
                # Finished box (GREEN)
                label = TARGET_CLASSES.get(box['id'], str(box['id']))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # OpenCV to PIL
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        # Resize to fit canvas
        h, w = display_img.shape[:2]
        canvas_w, canvas_h = 800, 500
        scale = min(canvas_w/w, canvas_h/h)
        if scale < 1:
            display_img = cv2.resize(display_img, (int(w*scale), int(h*scale)))

        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(display_img))
        # print("here 6")
        self.canvas.create_image(canvas_w//2, canvas_h//2, anchor=tk.CENTER, image=self.tk_img)

    def create_output_dirs(self):
        """Create train/val image and label directories."""
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)
    
    def set_label(self, class_id):
        if self.current_box_idx < len(self.current_bboxes):
            self.current_bboxes[self.current_box_idx]['id'] = class_id
            self.current_bboxes[self.current_box_idx]['needs_review'] = False
            self.show_next_review_box()

if __name__ == "__main__":
    root = tk.Tk()
    app = RelabelApp(root)
    root.mainloop()
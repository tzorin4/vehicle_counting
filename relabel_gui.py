import os
import glob
import shutil
import random
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET

# --- TARGET CLASSES (YOLO MODEL) ---
# 0: Car, 1: Motorcycle, 2: Truck, 3: Bus
TARGET_CLASSES = {0: "Car", 1: "Motorcycle", 2: "Truck", 3: "Bus"}

# Map string labels from UA-Detrac to temporary numerical IDs for processing
UA_DETRAC_MAP = {'car': 0, 'van': 1, 'bus': 2, 'others': 3}

# Auto-skip mappings based on instructions (Map Car and Van -> target Car)
# It leaves 2 (Bus) and 3 (Others) for manual GUI review
AUTO_MAP_RULES = {
    0: 0, # Previous 0 (Car) -> Target 0 (Car)
    1: 0, # Previous 1 (Van) -> Target 0 (Car)
}

class RelabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Relabel & Split Tool (YOLO & UA-Detrac)")
        self.root.geometry("1000x800")
        
        # State
        self.images_list = []
        self.current_img_idx = -1
        self.current_bboxes = []
        self.current_box_idx = -1
        
        self.img_dir = ""
        self.lbl_dir = ""
        self.output_dir = "dataset_output"
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- Control Panel ---
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
        
        tk.Button(ctrl_frame, text="START PROCESSING", bg="green", fg="white", font=("Arial", 10, "bold"), 
                  command=self.start_processing).grid(row=2, column=0, columnspan=4, pady=10)
        
        self.lbl_status = tk.Label(ctrl_frame, text="", fg="blue")
        self.lbl_status.grid(row=3, column=0, columnspan=4)

        # --- Image Display ---
        self.canvas = tk.Canvas(self.root, bg="black", width=800, height=500)
        self.canvas.pack(pady=10)
        self.img_on_canvas = None
        
        # --- Buttons Panel ---
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="1. Set as CAR", bg="lightblue", width=15, command=lambda: self.set_label(0)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="2. Set as MOTORCYCLE", bg="lightgreen", width=15, command=lambda: self.set_label(1)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="3. Set as TRUCK", bg="lightyellow", width=15, command=lambda: self.set_label(2)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="4. Set as BUS", bg="orange", width=15, command=lambda: self.set_label(3)).pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="DELETE BOX (X)", bg="red", fg="white", width=15, command=self.delete_box).pack(side=tk.RIGHT, padx=10)
        
        # Keyboard bindings
        self.root.bind('<KeyPress>', self.handle_keypress)
        
    def handle_keypress(self, event):
        if event.char == '1': self.set_label(0)
        elif event.char == '2': self.set_label(1)
        elif event.char == '3': self.set_label(2)
        elif event.char == '4': self.set_label(3)
        elif event.char.lower() == 'x': self.delete_box()

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

        # Prepare output directories
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)

        found_images = glob.glob(os.path.join(self.img_dir, '**', '*.jpg'), recursive=True) + \
                       glob.glob(os.path.join(self.img_dir, '**', '*.png'), recursive=True)
        
        if not found_images:
            messagebox.showerror("Error", "No images found in the selected directory.")
            return
            
        random.shuffle(found_images)
        
        try:
            limit = int(self.entry_limit.get())
            self.images_list = found_images[:limit]
        except ValueError:
            self.images_list = found_images
            
        try:
            self.train_split = float(self.entry_split.get()) / 100.0
        except ValueError:
            self.train_split = 0.8
            
        self.current_img_idx = 0
        self.load_image()

    def load_image(self):
        if self.current_img_idx >= len(self.images_list):
            messagebox.showinfo("Done", f"Finished processing! Files saved to {self.output_dir}")
            return
            
        self.img_path = self.images_list[self.current_img_idx]
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        
        self.lbl_status.config(text=f"Processing {self.current_img_idx+1}/{len(self.images_list)}: {base_name}")
        
        self.cv_img = cv2.imread(self.img_path)
        if self.cv_img is None:
            self.next_image()
            return
            
        self.img_h, self.img_w, _ = self.cv_img.shape
        self.current_bboxes = self.find_and_parse_labels(base_name)
        
        # Apply Auto-maper
        for box in self.current_bboxes:
            old_id = box['id']
            if old_id in AUTO_MAP_RULES:
                box['id'] = AUTO_MAP_RULES[old_id]
                box['needs_review'] = False # Skip car and van
                
        self.current_box_idx = 0
        self.show_next_review_box()

    def find_and_parse_labels(self, base_name):
        # Look for txt (YOLO) or xml (UA-DETRAC)
        # Search recursively in label dir
        txt_path = next(iter(glob.glob(os.path.join(self.lbl_dir, '**', f"{base_name}.txt"), recursive=True)), None)
        xml_path = next(iter(glob.glob(os.path.join(self.lbl_dir, '**', f"{base_name}.xml"), recursive=True)), None)
        
        boxes = []
        if xml_path:
            # UA-DETRAC XML Parser
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for target in root.findall('.//target'):
                box = target.find('box')
                attr = target.find('attribute')
                if box is not None and attr is not None:
                    try:
                        left = float(box.get('left'))
                        top = float(box.get('top'))
                        width = float(box.get('width'))
                        height = float(box.get('height'))
                        v_type = attr.get('vehicle_type').lower()
                        
                        # YOLO format conversion
                        x_c = (left + width/2) / self.img_w
                        y_c = (top + height/2) / self.img_h
                        w = width / self.img_w
                        h = height / self.img_h
                        
                        cls_id = UA_DETRAC_MAP.get(v_type, 3) # default to others
                        boxes.append({'id': cls_id, 'bbox': [x_c, y_c, w, h], 'needs_review': True})
                    except:
                        pass
        elif txt_path:
            # YOLO TXT Parser
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        bbox = [float(p) for p in parts[1:]]
                        # All boxes need review initially, unless caught by auto-map later
                        boxes.append({'id': cls_id, 'bbox': bbox, 'needs_review': True})
                        
        return boxes

    def show_next_review_box(self):
        # Find next box that needs review
        while self.current_box_idx < len(self.current_bboxes):
            if self.current_bboxes[self.current_box_idx]['needs_review']:
                break
            self.current_box_idx += 1
            
        self.draw_canvas()
        
        if self.current_box_idx >= len(self.current_bboxes):
            self.save_and_next()

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
        self.canvas.create_image(canvas_w//2, canvas_h//2, anchor=tk.CENTER, image=self.tk_img)

    def set_label(self, class_id):
        if self.current_box_idx < len(self.current_bboxes):
            self.current_bboxes[self.current_box_idx]['id'] = class_id
            self.current_bboxes[self.current_box_idx]['needs_review'] = False
            self.show_next_review_box()

    def delete_box(self):
        if self.current_box_idx < len(self.current_bboxes):
            # Mark it for removal later by setting a special flag, or just remove from list
            self.current_bboxes.pop(self.current_box_idx)
            # Do NOT increment index because we just popped it, next item is at same index
            self.show_next_review_box()

    def save_and_next(self):
        # Do Train / Val split
        split = 'train' if random.random() <= self.train_split else 'val'
        
        base_name = os.path.basename(self.img_path)
        dest_img_path = os.path.join(self.output_dir, split, 'images', base_name)
        dest_lbl_path = os.path.join(self.output_dir, split, 'labels', os.path.splitext(base_name)[0] + '.txt')
        
        # Copy Image
        shutil.copy2(self.img_path, dest_img_path)
        
        # Write YOLO label
        with open(dest_lbl_path, 'w') as f:
            for box in self.current_bboxes:
                f.write(f"{box['id']} {' '.join(map(str, box['bbox']))}\n")
                
        self.next_image()

    def next_image(self):
        self.current_img_idx += 1
        self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = RelabelApp(root)
    root.mainloop()
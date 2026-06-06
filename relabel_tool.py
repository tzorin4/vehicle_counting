"""
YOLO Image Labeling Tool
========================
A desktop GUI tool for labeling images in YOLO format.

SETUP:
    pip install pillow

USAGE:
    python yolo_labeler.py

CLASSES (edit this dict — key = class name, value = class index):
"""

# ─────────────────────────────────────────────
#  CONFIGURE YOUR CLASSES HERE
#  Format: "class_name": index
# ─────────────────────────────────────────────
CLASSES = {
    "car":    0,
    "bus":    1,
    "truck": 2,
    "motorbike":    3,
}
# ─────────────────────────────────────────────

import os
import sys
import shutil
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    print("ERROR: Pillow is required. Install with:  pip install pillow")
    sys.exit(1)


# ── Reverse lookup: index → name ──────────────────────────────────────────────
INDEX_TO_CLASS = {v: k for k, v in CLASSES.items()}
CLASS_NAMES    = list(CLASSES.keys())          # ordered list for the UI

# ── Colour palette for boxes ──────────────────────────────────────────────────
PALETTE = [
    "#FF4444", "#44AAFF", "#44FF88", "#FFB800", "#CC44FF",
    "#FF8844", "#00CCCC", "#FF44AA", "#88FF44", "#8844FF",
]

def class_color(class_idx: int) -> str:
    return PALETTE[class_idx % len(PALETTE)]


# ══════════════════════════════════════════════════════════════════════════════
class BBox:
    """One labelled bounding box (pixel coords + class index)."""
    def __init__(self, x1, y1, x2, y2, class_idx):
        self.x1, self.y1 = min(x1, x2), min(y1, y2)
        self.x2, self.y2 = max(x1, x2), max(y1, y2)
        self.class_idx   = class_idx

    def to_yolo(self, img_w, img_h) -> str:
        cx = ((self.x1 + self.x2) / 2) / img_w
        cy = ((self.y1 + self.y2) / 2) / img_h
        bw = (self.x2 - self.x1) / img_w
        bh = (self.y2 - self.y1) / img_h
        return f"{self.class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    @classmethod
    def from_yolo(cls, line: str, img_w, img_h):
        parts = line.strip().split()
        ci, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        return cls(x1, y1, x2, y2, ci)


# ══════════════════════════════════════════════════════════════════════════════
class LabelerApp(tk.Tk):

    CANVAS_W = 900
    CANVAS_H = 650

    def __init__(self):
        super().__init__()
        self.title("YOLO Labeler")
        self.resizable(True, True)
        self.configure(bg="#1a1a2e")

        # ── state ──────────────────────────────────────────────────────────
        self.image_paths:  list[Path] = []
        self.current_idx:  int        = -1
        self.boxes:        list[BBox] = []
        self.output_dir:   Path | None = None

        self.pil_image:    Image.Image | None = None
        self.tk_image:     ImageTk.PhotoImage | None = None
        self.scale:        float = 1.0
        self.offset_x:    int   = 0
        self.offset_y:    int   = 0

        # drawing state
        self.drawing       = False
        self.drag_start    = None
        self.drag_rect_id  = None
        self.selected_box  = None   # index into self.boxes

        # active class
        self.active_class_var = tk.StringVar(value=CLASS_NAMES[0] if CLASS_NAMES else "")

        self._build_ui()
        self._bind_keys()

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self, bg="#16213e", pady=6)
        toolbar.pack(fill=tk.X)

        btn_cfg = dict(bg="#0f3460", fg="white", relief="flat",
                       font=("Courier", 10, "bold"), padx=12, pady=4,
                       activebackground="#e94560", activeforeground="white",
                       cursor="hand2")

        tk.Button(toolbar, text="📂  Open Images",   command=self.open_images,  **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="📁  Set Output Dir", command=self.set_output,   **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="💾  Save Labels",    command=self.save_current, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="✅  Export Dataset",  command=self.export_all,   **btn_cfg).pack(side=tk.LEFT, padx=4)

        self.output_label = tk.Label(toolbar, text="No output dir set",
                                     bg="#16213e", fg="#888", font=("Courier", 9))
        self.output_label.pack(side=tk.RIGHT, padx=12)

        # Main area: canvas + side panel
        main = tk.Frame(self, bg="#1a1a2e")
        main.pack(fill=tk.BOTH, expand=True)

        # Canvas
        self.canvas = tk.Canvas(main, width=self.CANVAS_W, height=self.CANVAS_H,
                                 bg="#0d0d1a", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Side panel
        side = tk.Frame(main, bg="#16213e", width=220)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        side.pack_propagate(False)

        # ── Image list ─────────────────────────────────────────────────────
        tk.Label(side, text="IMAGES", bg="#16213e", fg="#e94560",
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=8, pady=(8,2))

        list_frame = tk.Frame(side, bg="#16213e")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=6)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                         bg="#0d0d1a", fg="#ccc", selectbackground="#e94560",
                                         font=("Courier", 9), relief="flat",
                                         activestyle="none", borderwidth=0)
        self.image_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # Navigation
        nav = tk.Frame(side, bg="#16213e")
        nav.pack(fill=tk.X, padx=6, pady=4)
        nb = dict(bg="#0f3460", fg="white", relief="flat",
                  font=("Courier", 10, "bold"), padx=8, pady=3,
                  activebackground="#e94560", activeforeground="white", cursor="hand2")
        tk.Button(nav, text="◀ Prev", command=self.prev_image, **nb).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        tk.Button(nav, text="Next ▶", command=self.next_image, **nb).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2,0))

        self.progress_label = tk.Label(side, text="0 / 0", bg="#16213e", fg="#888",
                                        font=("Courier", 9))
        self.progress_label.pack(pady=2)

        # ── Class selector ─────────────────────────────────────────────────
        tk.Label(side, text="ACTIVE CLASS", bg="#16213e", fg="#e94560",
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=8, pady=(10,2))

        self.class_frame = tk.Frame(side, bg="#16213e")
        self.class_frame.pack(fill=tk.X, padx=6)
        self._build_class_buttons()

        # ── Box list ───────────────────────────────────────────────────────
        tk.Label(side, text="BOXES  (click to select)", bg="#16213e", fg="#e94560",
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=8, pady=(10,2))

        box_list_frame = tk.Frame(side, bg="#16213e")
        box_list_frame.pack(fill=tk.BOTH, padx=6, pady=(0,4))

        bscroll = tk.Scrollbar(box_list_frame)
        bscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.box_listbox = tk.Listbox(box_list_frame, yscrollcommand=bscroll.set,
                                       bg="#0d0d1a", fg="#ccc", selectbackground="#e94560",
                                       font=("Courier", 8), height=8, relief="flat",
                                       activestyle="none", borderwidth=0)
        self.box_listbox.pack(fill=tk.BOTH, expand=True)
        bscroll.config(command=self.box_listbox.yview)
        self.box_listbox.bind("<<ListboxSelect>>", self._on_box_select)

        db = dict(bg="#3a0a0a", fg="#ff6666", relief="flat",
                  font=("Courier", 9, "bold"), pady=3, cursor="hand2",
                  activebackground="#e94560", activeforeground="white")
        tk.Button(side, text="🗑  Delete Selected Box", command=self.delete_selected_box, **db).pack(fill=tk.X, padx=6, pady=2)

        # ── Hint ───────────────────────────────────────────────────────────
        hint_text = "Draw: left-drag\nDelete: Del key\nNav: ← →"
        tk.Label(side, text=hint_text, bg="#16213e", fg="#555",
                 font=("Courier", 8), justify="left").pack(anchor="w", padx=8, pady=6)

        # Status bar
        self.status_var = tk.StringVar(value="Open a folder of images to begin.")
        status = tk.Label(self, textvariable=self.status_var,
                          bg="#0f3460", fg="#aaa", font=("Courier", 9),
                          anchor="w", padx=8, pady=3)
        status.pack(fill=tk.X, side=tk.BOTTOM)

        # Canvas events
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self.canvas.bind("<B1-Motion>",        self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._on_mouse_release)
        self.canvas.bind("<Configure>",        lambda e: self._redraw())

    def _build_class_buttons(self):
        for w in self.class_frame.winfo_children():
            w.destroy()
        for name in CLASS_NAMES:
            idx = CLASSES[name]
            color = class_color(idx)
            is_active = (name == self.active_class_var.get())
            relief = "solid" if is_active else "flat"
            bd     = 2 if is_active else 0
            btn = tk.Button(self.class_frame, text=f"[{idx}] {name}",
                            bg=color if is_active else "#0d0d1a",
                            fg="black" if is_active else color,
                            font=("Courier", 9, "bold"), relief=relief, bd=bd,
                            padx=6, pady=2, cursor="hand2", anchor="w",
                            command=lambda n=name: self._select_class(n))
            btn.pack(fill=tk.X, pady=1)

    def _bind_keys(self):
        self.bind("<Left>",  lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        self.bind("<Delete>", lambda e: self.delete_selected_box())
        self.bind("<BackSpace>", lambda e: self.delete_selected_box())
        for i, name in enumerate(CLASS_NAMES[:10]):
            self.bind(str(i), lambda e, n=name: self._select_class(n))

    # ── Class selection ────────────────────────────────────────────────────
    def _select_class(self, name: str):
        self.active_class_var.set(name)
        self._build_class_buttons()
        self.status(f"Active class: [{CLASSES[name]}] {name}")

    def active_class_idx(self) -> int:
        return CLASSES.get(self.active_class_var.get(), 0)

    # ── File operations ────────────────────────────────────────────────────
    def open_images(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        self.image_paths = sorted(
            [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]
        )
        self.image_listbox.delete(0, tk.END)
        for p in self.image_paths:
            self.image_listbox.insert(tk.END, p.name)

        self.current_idx = -1
        self.status(f"Loaded {len(self.image_paths)} images from {folder}")
        if self.image_paths:
            self.load_image(0)

    def set_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if not folder:
            return
        self.output_dir = Path(folder)
        self.output_label.config(text=f"→ {self.output_dir.name}", fg="#44ff88")
        self.status(f"Output directory: {folder}")

    # ── Image loading ──────────────────────────────────────────────────────
    def load_image(self, idx: int):
        if not self.image_paths or idx < 0 or idx >= len(self.image_paths):
            return

        # Save previous
        if self.current_idx >= 0:
            self._auto_save()

        self.current_idx = idx
        path = self.image_paths[idx]

        self.pil_image = Image.open(path).convert("RGB")
        self.boxes = self._load_existing_labels(path)
        self.selected_box = None

        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(idx)
        self.image_listbox.see(idx)
        self.progress_label.config(text=f"{idx+1} / {len(self.image_paths)}")

        self._redraw()
        self._refresh_box_list()
        self.status(f"Image: {path.name}  ({self.pil_image.width}×{self.pil_image.height})  |  {len(self.boxes)} box(es)")

    def _load_existing_labels(self, image_path: Path) -> list[BBox]:
        label_path = image_path.with_suffix(".txt")
        boxes = []
        if label_path.exists():
            w, h = Image.open(image_path).size
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            boxes.append(BBox.from_yolo(line, w, h))
                        except Exception:
                            pass
        return boxes

    # ── Canvas drawing ─────────────────────────────────────────────────────
    def _redraw(self):
        self.canvas.delete("all")
        if self.pil_image is None:
            self._draw_placeholder()
            return

        cw = self.canvas.winfo_width()  or self.CANVAS_W
        ch = self.canvas.winfo_height() or self.CANVAS_H

        iw, ih = self.pil_image.size
        scale = min(cw / iw, ch / ih)
        self.scale = scale

        nw, nh = int(iw * scale), int(ih * scale)
        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2

        resized = self.pil_image.resize((nw, nh), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)

        for i, box in enumerate(self.boxes):
            self._draw_box(i, box)

    def _draw_placeholder(self):
        cw = self.canvas.winfo_width()  or self.CANVAS_W
        ch = self.canvas.winfo_height() or self.CANVAS_H
        self.canvas.create_text(cw//2, ch//2, text="Open images to begin",
                                 fill="#333", font=("Courier", 16))

    def _draw_box(self, idx: int, box: BBox):
        color  = class_color(box.class_idx)
        name   = INDEX_TO_CLASS.get(box.class_idx, str(box.class_idx))
        x1, y1 = self._img_to_canvas(box.x1, box.y1)
        x2, y2 = self._img_to_canvas(box.x2, box.y2)
        width  = 3 if idx == self.selected_box else 2
        dash   = (4, 2) if idx == self.selected_box else None

        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color,
                                      width=width, dash=dash,
                                      tags=f"box_{idx}")
        # label pill
        lbl = f" [{box.class_idx}] {name} "
        tx, ty = x1 + 2, y1 - 14
        self.canvas.create_rectangle(tx - 1, ty, tx + len(lbl) * 6 + 1, ty + 13,
                                      fill=color, outline="")
        self.canvas.create_text(tx + 3, ty + 7, text=lbl, fill="black",
                                 font=("Courier", 8, "bold"), anchor="w")

    # ── Coordinate helpers ─────────────────────────────────────────────────
    def _img_to_canvas(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def _canvas_to_img(self, cx, cy):
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    # ── Mouse events ───────────────────────────────────────────────────────
    def _on_mouse_press(self, event):
        if self.pil_image is None:
            return
        # Check click on existing box first
        clicked = self._hit_test(event.x, event.y)
        if clicked is not None:
            self.selected_box = clicked
            self._refresh_box_list()
            self._redraw()
            return
        # Start drawing
        self.drawing   = True
        self.drag_start = (event.x, event.y)
        self.selected_box = None

    def _on_mouse_drag(self, event):
        if not self.drawing or self.drag_start is None:
            return
        if self.drag_rect_id:
            self.canvas.delete(self.drag_rect_id)
        x0, y0 = self.drag_start
        color = class_color(self.active_class_idx())
        self.drag_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y, outline=color, width=2, dash=(4, 2))

    def _on_mouse_release(self, event):
        if not self.drawing or self.drag_start is None:
            return
        self.drawing = False
        if self.drag_rect_id:
            self.canvas.delete(self.drag_rect_id)
            self.drag_rect_id = None

        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        # Ignore tiny boxes
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            self.drag_start = None
            return

        ix0, iy0 = self._canvas_to_img(x0, y0)
        ix1, iy1 = self._canvas_to_img(x1, y1)

        # Clamp to image bounds
        iw, ih = self.pil_image.size
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(iw, ix1), min(ih, iy1)

        box = BBox(ix0, iy0, ix1, iy1, self.active_class_idx())
        self.boxes.append(box)
        self.selected_box = len(self.boxes) - 1
        self.drag_start   = None

        self._refresh_box_list()
        self._redraw()
        self.status(f"Added box [{box.class_idx}] {INDEX_TO_CLASS.get(box.class_idx, '?')}")

    def _hit_test(self, cx, cy) -> int | None:
        """Return index of box under cursor, or None."""
        ix, iy = self._canvas_to_img(cx, cy)
        for i in range(len(self.boxes) - 1, -1, -1):
            b = self.boxes[i]
            if b.x1 <= ix <= b.x2 and b.y1 <= iy <= b.y2:
                return i
        return None

    # ── Box list ───────────────────────────────────────────────────────────
    def _refresh_box_list(self):
        self.box_listbox.delete(0, tk.END)
        for i, b in enumerate(self.boxes):
            name = INDEX_TO_CLASS.get(b.class_idx, str(b.class_idx))
            w  = int(b.x2 - b.x1)
            h  = int(b.y2 - b.y1)
            mark = "►" if i == self.selected_box else " "
            self.box_listbox.insert(tk.END, f"{mark} [{b.class_idx}] {name}  {w}×{h}")
        if self.selected_box is not None and self.selected_box < self.box_listbox.size():
            self.box_listbox.selection_clear(0, tk.END)
            self.box_listbox.selection_set(self.selected_box)
            self.box_listbox.see(self.selected_box)

    def _on_box_select(self, event):
        sel = self.box_listbox.curselection()
        if sel:
            self.selected_box = sel[0]
            self._redraw()

    def _on_list_select(self, event):
        sel = self.image_listbox.curselection()
        if sel:
            self.load_image(sel[0])

    # ── Delete box ─────────────────────────────────────────────────────────
    def delete_selected_box(self):
        if self.selected_box is None or not self.boxes:
            return
        self.boxes.pop(self.selected_box)
        self.selected_box = None
        self._refresh_box_list()
        self._redraw()

    # ── Navigation ─────────────────────────────────────────────────────────
    def prev_image(self):
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.load_image(self.current_idx + 1)

    # ── Saving ─────────────────────────────────────────────────────────────
    def _auto_save(self):
        """Save labels next to the source image (working copy)."""
        if self.current_idx < 0 or not self.image_paths:
            return
        path = self.image_paths[self.current_idx]
        label_path = path.with_suffix(".txt")
        with open(label_path, "w") as f:
            iw, ih = self.pil_image.size
            for box in self.boxes:
                f.write(box.to_yolo(iw, ih) + "\n")

    def save_current(self):
        self._auto_save()
        if self.current_idx >= 0:
            self.status(f"Saved labels for {self.image_paths[self.current_idx].name}")

    # ── Export dataset ─────────────────────────────────────────────────────
    def export_all(self):
        if not self.output_dir:
            messagebox.showerror("No Output Dir", "Please set an output directory first.")
            return
        if not self.image_paths:
            messagebox.showerror("No Images", "No images loaded.")
            return

        # Save current image's labels first
        self._auto_save()

        train_ratio = 0.8
        result = self._ask_train_ratio()
        if result is None:
            return
        train_ratio = result

        # Build file list (only images that have a label file)
        labeled = []
        for p in self.image_paths:
            lp = p.with_suffix(".txt")
            if lp.exists():
                labeled.append(p)

        if not labeled:
            messagebox.showwarning("No Labels", "No images have been labeled yet.")
            return

        random.shuffle(labeled)
        split     = int(len(labeled) * train_ratio)
        train_set = labeled[:split]
        val_set   = labeled[split:]

        for split_name, files in [("train", train_set), ("val", val_set)]:
            img_dir   = self.output_dir / split_name / "images"
            label_dir = self.output_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            for img_path in files:
                shutil.copy2(img_path, img_dir / img_path.name)
                lp = img_path.with_suffix(".txt")
                shutil.copy2(lp, label_dir / lp.name)

        # Write data.yaml
        yaml_lines = [
            f"path: {self.output_dir}",
            "train: train/images",
            "val:   val/images",
            "",
            f"nc: {len(CLASSES)}",
            f"names: {CLASS_NAMES}",
            "",
            "# Class map:",
        ]
        for name, idx in sorted(CLASSES.items(), key=lambda x: x[1]):
            yaml_lines.append(f"#   {idx}: {name}")

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write("\n".join(yaml_lines) + "\n")

        msg = (
            f"Dataset exported!\n\n"
            f"  Train: {len(train_set)} images\n"
            f"  Val:   {len(val_set)} images\n\n"
            f"  Output: {self.output_dir}\n"
            f"  Config: data.yaml"
        )
        messagebox.showinfo("Export Complete", msg)
        self.status(f"Exported {len(labeled)} labeled images → {self.output_dir}")

    def _ask_train_ratio(self) -> float | None:
        """Show a dialog to pick the train/val split."""
        dialog = tk.Toplevel(self)
        dialog.title("Train / Val Split")
        dialog.configure(bg="#1a1a2e")
        dialog.resizable(False, False)
        dialog.grab_set()

        result = {"value": None}

        tk.Label(dialog, text="Train split ratio  (e.g. 0.8 = 80% train, 20% val)",
                 bg="#1a1a2e", fg="white", font=("Courier", 10), padx=16, pady=12).pack()

        var = tk.DoubleVar(value=0.8)
        scale = tk.Scale(dialog, from_=0.5, to=1.0, resolution=0.05,
                         orient=tk.HORIZONTAL, variable=var, length=280,
                         bg="#1a1a2e", fg="white", troughcolor="#0f3460",
                         highlightthickness=0, font=("Courier", 9))
        scale.pack(padx=16)

        def confirm():
            result["value"] = var.get()
            dialog.destroy()

        def cancel():
            dialog.destroy()

        btn_f = tk.Frame(dialog, bg="#1a1a2e")
        btn_f.pack(pady=10)
        tk.Button(btn_f, text="Export", command=confirm,
                  bg="#e94560", fg="white", font=("Courier", 10, "bold"),
                  relief="flat", padx=16, pady=4).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_f, text="Cancel", command=cancel,
                  bg="#333", fg="white", font=("Courier", 10),
                  relief="flat", padx=16, pady=4).pack(side=tk.LEFT, padx=6)

        dialog.wait_window()
        return result["value"]

    # ── Status bar ─────────────────────────────────────────────────────────
    def status(self, msg: str):
        self.status_var.set(msg)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = LabelerApp()
    app.mainloop()
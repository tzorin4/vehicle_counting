"""
YOLO Image Labeling Tool  —  with AI-Assist (Iterative Labeling)
=================================================================
SETUP:
    pip install pillow

    For AI-assist (optional):
    pip install ultralytics        # YOLOv8/v11
    # or:
    pip install torch torchvision  # plain PyTorch for custom .pt models

USAGE:
    python yolo_labeler.py

HOW AI-ASSIST WORKS:
    1. Train your model on an initial set of labels.
    2. Export to a .pt file (YOLOv8: `model.save("best.pt")`).
    3. In the tool, click "🤖 Load Model" and pick the .pt file.
    4. Click "▶ Run AI" on any image — predictions appear as dashed
       "ghost" boxes in a distinct colour.
    5. Accept individual predictions (click → "✔ Accept") or
       accept all at once with "✔ Accept All Predictions".
    6. Rejected / unwanted ghosts vanish; accepted ones become
       normal confirmed boxes you can still edit.
    7. Add more manual boxes as needed, export, retrain, repeat.

CLASSES — edit the dict below (key = class name, value = YOLO index):
"""

# ─────────────────────────────────────────────
#  CONFIGURE YOUR CLASSES HERE
# ─────────────────────────────────────────────
CLASSES = {
    "car":    0,
    "motorbike":    1,
    "truck": 2,
    "bus":    3,
}
# ─────────────────────────────────────────────

import os, sys, shutil, random, threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    print("ERROR: Pillow is required.  pip install pillow")
    sys.exit(1)

# ── reverse maps ──────────────────────────────────────────────────────────────
INDEX_TO_CLASS = {v: k for k, v in CLASSES.items()}
CLASS_NAMES    = list(CLASSES.keys())

# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = [
    "#FF4444", "#44AAFF", "#44FF88", "#FFB800", "#CC44FF",
    "#FF8844", "#00CCCC", "#FF44AA", "#88FF44", "#8844FF",
]
GHOST_OUTLINE = "#FFFFFF"          # prediction ghost colour
GHOST_FILL_ALPHA = ""              # canvas has no alpha, we use dash pattern

def class_color(class_idx: int) -> str:
    return PALETTE[class_idx % len(PALETTE)]


# ══════════════════════════════════════════════════════════════════════════════
class BBox:
    """One bounding box — either confirmed (human) or a prediction ghost."""

    def __init__(self, x1, y1, x2, y2, class_idx, *, ghost=False, conf=None):
        self.x1, self.y1 = min(x1, x2), min(y1, y2)
        self.x2, self.y2 = max(x1, x2), max(y1, y2)
        self.class_idx   = class_idx
        self.ghost       = ghost     # True = AI prediction, not yet confirmed
        self.conf        = conf      # confidence score (float | None)

    def to_yolo(self, img_w, img_h) -> str:
        cx = ((self.x1 + self.x2) / 2) / img_w
        cy = ((self.y1 + self.y2) / 2) / img_h
        bw = (self.x2 - self.x1) / img_w
        bh = (self.y2 - self.y1) / img_h
        return f"{self.class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    @classmethod
    def from_yolo(cls, line: str, img_w, img_h):
        parts = line.strip().split()
        ci    = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        return cls(x1, y1, x2, y2, ci)


# ══════════════════════════════════════════════════════════════════════════════
class ModelWrapper:
    """
    Thin wrapper that tries Ultralytics YOLOv8/v11 first, then plain
    torch.hub / custom forward pass as a fallback.
    Returns a list of BBox (ghost=True) for a PIL image.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model     = None
        self._kind      = None          # "ultralytics" | "torch"
        self._load()

    def _load(self):
        path = self.model_path
        # ── Try Ultralytics first ──────────────────────────────────────────
        try:
            from ultralytics import YOLO
            self._model = YOLO(path)
            self._kind  = "ultralytics"
            return
        except Exception:
            pass
        # ── Fallback: raw torch ────────────────────────────────────────────
        try:
            import torch
            self._model = torch.load(path, map_location="cpu")
            if hasattr(self._model, "eval"):
                self._model.eval()
            self._kind = "torch"
            return
        except Exception as e:
            raise RuntimeError(
                f"Could not load model from {path}.\n"
                "Install ultralytics (pip install ultralytics) for YOLOv8/v11 support.\n"
                f"Original error: {e}"
            )

    def predict(self, pil_img: Image.Image, conf_thresh: float) -> list[BBox]:
        if self._kind == "ultralytics":
            return self._predict_ultralytics(pil_img, conf_thresh)
        elif self._kind == "torch":
            return self._predict_torch(pil_img, conf_thresh)
        return []

    def _predict_ultralytics(self, pil_img, conf_thresh):
        results = self._model.predict(pil_img, conf=conf_thresh, verbose=False)
        boxes = []
        iw, ih = pil_img.size
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                ci   = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                # remap model class index → our CLASSES index if possible
                # (if the model was trained on our dataset they should match)
                boxes.append(BBox(x1, y1, x2, y2, ci, ghost=True, conf=conf))
        return boxes

    def _predict_torch(self, pil_img, conf_thresh):
        """
        Generic torch path — expects the model to be a YOLOv5-style hub model
        that accepts a PIL image and returns .pandas().xyxy[0].
        """
        import torch
        results = self._model(pil_img)
        df      = results.pandas().xyxy[0]
        boxes   = []
        for _, row in df.iterrows():
            if row["confidence"] < conf_thresh:
                continue
            boxes.append(BBox(
                row["xmin"], row["ymin"], row["xmax"], row["ymax"],
                int(row["class"]), ghost=True, conf=float(row["confidence"])
            ))
        return boxes

    @property
    def name(self) -> str:
        return Path(self.model_path).name


# ══════════════════════════════════════════════════════════════════════════════
class LabelerApp(tk.Tk):

    CANVAS_W = 920
    CANVAS_H = 660

    def __init__(self):
        super().__init__()
        self.title("YOLO Labeler  +  AI Assist")
        self.resizable(True, True)
        self.configure(bg="#1a1a2e")

        # ── core state ─────────────────────────────────────────────────────
        self.image_paths:  list[Path]      = []
        self.image_entries: list[dict]     = []   # {img, lbl, split}
        self.current_idx:  int             = -1
        self.boxes:        list[BBox]      = []   # confirmed + ghost mixed
        self.output_dir:   Path | None     = None
        self.model:        ModelWrapper | None = None

        self.pil_image:    Image.Image | None       = None
        self.tk_image:     ImageTk.PhotoImage | None = None
        self.scale:        float = 1.0
        self.offset_x:     int   = 0
        self.offset_y:     int   = 0

        # drawing / interaction state
        self.drawing       = False
        self.drag_start    = None
        self.drag_rect_id  = None
        self.selected_box: int | None = None

        # AI state
        self.ai_running    = False
        self.conf_thresh   = tk.DoubleVar(value=0.40)

        self.active_class_var = tk.StringVar(value=CLASS_NAMES[0] if CLASS_NAMES else "")

        self._build_ui()
        self._bind_keys()

    # =========================================================================
    # UI construction
    # =========================================================================
    def _build_ui(self):
        # ── Top toolbar ───────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg="#16213e", pady=6)
        toolbar.pack(fill=tk.X)

        B = dict(bg="#0f3460", fg="white", relief="flat",
                 font=("Courier", 10, "bold"), padx=10, pady=4,
                 activebackground="#e94560", activeforeground="white",
                 cursor="hand2")

        tk.Button(toolbar, text="📂 Open Images",    command=self.open_images,  **B).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="🏷 Open + Labels", command=self.open_with_labels, **B).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="📁 Output Dir",     command=self.set_output,   **B).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="💾 Save",           command=self.save_current, **B).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="✅ Export Dataset", command=self.export_all,   **B).pack(side=tk.LEFT, padx=3)
        tk.Button(toolbar, text="🔀 Remap Classes",  command=self.open_remap_dialog, **B).pack(side=tk.LEFT, padx=3)

        # separator
        tk.Frame(toolbar, bg="#333", width=2).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        # AI section in toolbar
        AI = dict(bg="#1a3a2a", fg="#44ff88", relief="flat",
                  font=("Courier", 10, "bold"), padx=10, pady=4,
                  activebackground="#44ff88", activeforeground="black",
                  cursor="hand2")

        tk.Button(toolbar, text="🤖 Load Model", command=self.load_model, **AI).pack(side=tk.LEFT, padx=3)
        self.run_btn = tk.Button(toolbar, text="▶ Run AI", command=self.run_ai,
                                  state=tk.DISABLED, **AI)
        self.run_btn.pack(side=tk.LEFT, padx=3)

        tk.Button(toolbar, text="✔ Accept All", command=self.accept_all_ghosts,
                  bg="#0a2a1a", fg="#44ff88", relief="flat",
                  font=("Courier", 10, "bold"), padx=10, pady=4,
                  activebackground="#44ff88", activeforeground="black",
                  cursor="hand2").pack(side=tk.LEFT, padx=3)

        tk.Button(toolbar, text="✘ Reject All", command=self.reject_all_ghosts,
                  bg="#2a0a0a", fg="#ff6666", relief="flat",
                  font=("Courier", 10, "bold"), padx=10, pady=4,
                  activebackground="#ff6666", activeforeground="black",
                  cursor="hand2").pack(side=tk.LEFT, padx=3)

        self.model_label = tk.Label(toolbar, text="No model loaded",
                                     bg="#16213e", fg="#555", font=("Courier", 9))
        self.model_label.pack(side=tk.LEFT, padx=8)

        self.output_label = tk.Label(toolbar, text="No output dir",
                                      bg="#16213e", fg="#888", font=("Courier", 9))
        self.output_label.pack(side=tk.RIGHT, padx=12)

        # ── Second toolbar row: confidence slider ─────────────────────────
        conf_row = tk.Frame(self, bg="#12122a", pady=3)
        conf_row.pack(fill=tk.X)

        tk.Label(conf_row, text="AI conf threshold:", bg="#12122a", fg="#888",
                 font=("Courier", 9)).pack(side=tk.LEFT, padx=(10, 4))

        self.conf_slider = tk.Scale(conf_row, from_=0.05, to=0.95, resolution=0.05,
                                     orient=tk.HORIZONTAL, variable=self.conf_thresh,
                                     length=220, bg="#12122a", fg="white",
                                     troughcolor="#1a3a2a", highlightthickness=0,
                                     font=("Courier", 8), sliderlength=14)
        self.conf_slider.pack(side=tk.LEFT)

        self.conf_val_label = tk.Label(conf_row, textvariable=self.conf_thresh,
                                        bg="#12122a", fg="#44ff88", font=("Courier", 9, "bold"), width=4)
        self.conf_val_label.pack(side=tk.LEFT, padx=4)

        tk.Label(conf_row, text="  |  ghost boxes shown in white dashes  |  click ghost → accept/reject",
                 bg="#12122a", fg="#444", font=("Courier", 8)).pack(side=tk.LEFT, padx=8)

        # ── Main area ─────────────────────────────────────────────────────
        main = tk.Frame(self, bg="#1a1a2e")
        main.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main, width=self.CANVAS_W, height=self.CANVAS_H,
                                 bg="#0d0d1a", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── Side panel ────────────────────────────────────────────────────
        side = tk.Frame(main, bg="#16213e", width=230)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        side.pack_propagate(False)

        # Image list
        tk.Label(side, text="IMAGES", bg="#16213e", fg="#e94560",
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=8, pady=(8, 2))

        lf = tk.Frame(side, bg="#16213e")
        lf.pack(fill=tk.BOTH, expand=True, padx=6)
        sb = tk.Scrollbar(lf); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox = tk.Listbox(lf, yscrollcommand=sb.set,
                                         bg="#0d0d1a", fg="#ccc", selectbackground="#e94560",
                                         font=("Courier", 9), relief="flat",
                                         activestyle="none", borderwidth=0)
        self.image_listbox.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.image_listbox.yview)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        nav = tk.Frame(side, bg="#16213e")
        nav.pack(fill=tk.X, padx=6, pady=4)
        NB = dict(bg="#0f3460", fg="white", relief="flat",
                  font=("Courier", 10, "bold"), padx=8, pady=3,
                  activebackground="#e94560", activeforeground="white", cursor="hand2")
        tk.Button(nav, text="◀ Prev", command=self.prev_image, **NB).pack(side=tk.LEFT,  fill=tk.X, expand=True, padx=(0,2))
        tk.Button(nav, text="Next ▶", command=self.next_image, **NB).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2,0))

        self.progress_label = tk.Label(side, text="0 / 0", bg="#16213e", fg="#888",
                                        font=("Courier", 9))
        self.progress_label.pack(pady=2)

        # Class buttons
        tk.Label(side, text="ACTIVE CLASS  (0-9 hotkeys)", bg="#16213e", fg="#e94560",
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=8, pady=(10, 2))
        self.class_frame = tk.Frame(side, bg="#16213e")
        self.class_frame.pack(fill=tk.X, padx=6)
        self._build_class_buttons()

        # Box list
        tk.Label(side, text="BOXES  (click to select)", bg="#16213e", fg="#e94560",
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=8, pady=(10, 2))

        bf = tk.Frame(side, bg="#16213e")
        bf.pack(fill=tk.X, padx=6, pady=(0, 2))
        bsb = tk.Scrollbar(bf); bsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.box_listbox = tk.Listbox(bf, yscrollcommand=bsb.set,
                                       bg="#0d0d1a", fg="#ccc", selectbackground="#e94560",
                                       font=("Courier", 8), height=8, relief="flat",
                                       activestyle="none", borderwidth=0)
        self.box_listbox.pack(fill=tk.BOTH, expand=True)
        bsb.config(command=self.box_listbox.yview)
        self.box_listbox.bind("<<ListboxSelect>>", self._on_box_select)

        # Per-box action buttons (only relevant when a ghost is selected)
        box_actions = tk.Frame(side, bg="#16213e")
        box_actions.pack(fill=tk.X, padx=6, pady=2)

        self.accept_btn = tk.Button(box_actions, text="✔ Accept",
                                     command=self.accept_selected_ghost,
                                     bg="#0a2a1a", fg="#44ff88", relief="flat",
                                     font=("Courier", 9, "bold"), pady=3, cursor="hand2",
                                     activebackground="#44ff88", activeforeground="black")
        self.accept_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))

        tk.Button(box_actions, text="🗑 Delete",
                  command=self.delete_selected_box,
                  bg="#3a0a0a", fg="#ff6666", relief="flat",
                  font=("Courier", 9, "bold"), pady=3, cursor="hand2",
                  activebackground="#e94560", activeforeground="white"
                  ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2,0))

        # Hints
        tk.Label(side,
                 text="Draw: left-drag  |  Del: delete\n"
                      "← → navigate  |  L-click = select\n"
                      "R-click box = reassign class\n"
                      "🔀 Remap = bulk class remapping",
                 bg="#16213e", fg="#444", font=("Courier", 8), justify="left"
                 ).pack(anchor="w", padx=8, pady=6)

        # Status bar
        self.status_var = tk.StringVar(value="Open a folder of images to begin.")
        tk.Label(self, textvariable=self.status_var,
                 bg="#0f3460", fg="#aaa", font=("Courier", 9),
                 anchor="w", padx=8, pady=3).pack(fill=tk.X, side=tk.BOTTOM)

        # Canvas events
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self.canvas.bind("<B1-Motion>",        self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._on_mouse_release)
        self.canvas.bind("<ButtonPress-3>",    self._on_right_click)
        self.canvas.bind("<Configure>",        lambda e: self._redraw())

    # =========================================================================
    # Class UI helpers
    # =========================================================================
    def _build_class_buttons(self):
        for w in self.class_frame.winfo_children():
            w.destroy()
        for name in CLASS_NAMES:
            idx      = CLASSES[name]
            color    = class_color(idx)
            active   = (name == self.active_class_var.get())
            tk.Button(self.class_frame,
                      text=f"[{idx}] {name}",
                      bg=color if active else "#0d0d1a",
                      fg="black" if active else color,
                      font=("Courier", 9, "bold"),
                      relief="solid" if active else "flat",
                      bd=2 if active else 0,
                      padx=6, pady=2, cursor="hand2", anchor="w",
                      command=lambda n=name: self._select_class(n)
                      ).pack(fill=tk.X, pady=1)

    def _bind_keys(self):
        self.bind("<Left>",      lambda e: self.prev_image())
        self.bind("<Right>",     lambda e: self.next_image())
        self.bind("<Delete>",    lambda e: self.delete_selected_box())
        self.bind("<BackSpace>", lambda e: self.delete_selected_box())
        self.bind("<Return>",    lambda e: self.accept_selected_ghost())
        for i, name in enumerate(CLASS_NAMES[:10]):
            self.bind(str(i), lambda e, n=name: self._select_class(n))

    def _select_class(self, name: str):
        self.active_class_var.set(name)
        self._build_class_buttons()
        self.status(f"Active class → [{CLASSES[name]}] {name}")

    def active_class_idx(self) -> int:
        return CLASSES.get(self.active_class_var.get(), 0)

    # =========================================================================
    # File / folder operations
    # =========================================================================

    # --- internal image registry -----------------------------------------
    # Each entry: {"img": Path, "lbl": Path | None, "split": str | None}
    # "split" is "train" / "val" / None (flat folder mode)
    # We keep a parallel list so the listbox index always matches.

    def _register_images(self, entries: list[dict]):
        """Populate image_paths + entries list and refresh the listbox."""
        self.image_entries = entries                          # NEW
        self.image_paths   = [e["img"] for e in entries]     # kept for compat

        self.image_listbox.delete(0, tk.END)
        for e in entries:
            split_tag = f"[{e['split']}] " if e["split"] else ""
            has_lbl   = "✔ " if e["lbl"] and e["lbl"].exists() else "  "
            self.image_listbox.insert(tk.END, f"{has_lbl}{split_tag}{e['img'].name}")
            # colour: labelled = bright, unlabelled = dim
            row = self.image_listbox.size() - 1
            color = "#88ccff" if (e["lbl"] and e["lbl"].exists()) else "#555"
            self.image_listbox.itemconfig(row, fg=color)

        self.current_idx = -1
        if self.image_paths:
            self.load_image(0)

    def open_images(self):
        """Open a flat folder of images (labels are .txt siblings)."""
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        exts    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        entries = []
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in exts:
                lbl = p.with_suffix(".txt")
                entries.append({"img": p, "lbl": lbl, "split": None})

        self._register_images(entries)
        self.status(f"Flat folder: {len(entries)} images from {folder}")

    def open_with_labels(self):
        """
        Ask for an images folder, then a labels folder.
        Pairs every image with <labels_dir>/<stem>.txt automatically.
        Any image whose .txt does not exist yet is treated as unlabeled.
        """
        img_folder = filedialog.askdirectory(title="Step 1 — Select IMAGES folder")
        if not img_folder:
            return
        lbl_folder = filedialog.askdirectory(title="Step 2 — Select LABELS folder  (where the .txt files are)")
        if not lbl_folder:
            return

        img_dir = Path(img_folder)
        lbl_dir = Path(lbl_folder)
        exts    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        entries = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() in exts:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                entries.append({"img": img_path, "lbl": lbl_path, "split": None})

        if not entries:
            messagebox.showwarning("No Images", f"No images found in:\n{img_dir}")
            return

        self._register_images(entries)
        labeled   = sum(1 for e in entries if e["lbl"].exists())
        unlabeled = len(entries) - labeled
        self.status(
            f"Loaded {len(entries)} images from {img_dir.name}  |  "
            f"Labels from {lbl_dir.name}  |  "
            f"{labeled} already labeled, {unlabeled} unlabeled"
        )

    def set_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if not folder:
            return
        self.output_dir = Path(folder)
        self.output_label.config(text=f"→ {self.output_dir.name}", fg="#44ff88")
        self.status(f"Output: {folder}")

    # =========================================================================
    # AI model
    # =========================================================================
    def load_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO .pt model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.model = ModelWrapper(path)
            self.model_label.config(text=f"🤖 {self.model.name}", fg="#44ff88")
            self.run_btn.config(state=tk.NORMAL)
            self.status(f"Model loaded: {self.model.name}")
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            self.model = None

    def run_ai(self):
        if self.model is None or self.pil_image is None or self.ai_running:
            return
        self.ai_running = True
        self.run_btn.config(state=tk.DISABLED, text="⏳ Running…")
        self.status("Running inference…")

        img_copy   = self.pil_image.copy()
        conf       = self.conf_thresh.get()

        def worker():
            try:
                predictions = self.model.predict(img_copy, conf)
                self.after(0, lambda: self._on_ai_done(predictions))
            except Exception as e:
                self.after(0, lambda: self._on_ai_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_ai_done(self, predictions: list[BBox]):
        self.ai_running = False
        self.run_btn.config(state=tk.NORMAL, text="▶ Run AI")

        # Remove any old ghosts, keep confirmed boxes
        self.boxes = [b for b in self.boxes if not b.ghost]
        self.boxes.extend(predictions)

        self._refresh_box_list()
        self._redraw()
        n = len(predictions)
        self.status(f"AI found {n} prediction{'s' if n != 1 else ''}  —  review ghost boxes (white dashes)")

    def _on_ai_error(self, msg: str):
        self.ai_running = False
        self.run_btn.config(state=tk.NORMAL, text="▶ Run AI")
        messagebox.showerror("Inference Error", msg)
        self.status("Inference failed — see error dialog.")

    # ── Accept / reject ghosts ────────────────────────────────────────────
    def accept_selected_ghost(self):
        if self.selected_box is None:
            return
        box = self.boxes[self.selected_box]
        if box.ghost:
            box.ghost = False
        self._refresh_box_list()
        self._redraw()

    def accept_all_ghosts(self):
        for b in self.boxes:
            b.ghost = False
        self._refresh_box_list()
        self._redraw()
        self.status("All predictions accepted.")

    def reject_all_ghosts(self):
        self.boxes = [b for b in self.boxes if not b.ghost]
        self.selected_box = None
        self._refresh_box_list()
        self._redraw()
        self.status("All predictions rejected.")

    def reassign_selected_class(self, class_name: str):
        """Reassign the selected box to class_name."""
        if self.selected_box is None or class_name not in CLASSES:
            return
        self.boxes[self.selected_box].class_idx = CLASSES[class_name]
        self._refresh_box_list()
        self._redraw()
        self.status(f"Box reassigned → [{CLASSES[class_name]}] {class_name}")

    # ── Right-click context menu ───────────────────────────────────────────
    def _on_right_click(self, event):
        """Right-click on canvas: if over a box, pop up a class-reassign menu."""
        if self.pil_image is None:
            return
        hit = self._hit_test(event.x, event.y)
        if hit is None:
            return
        self.selected_box = hit
        self._refresh_box_list()
        self._redraw()

        menu = tk.Menu(self, tearoff=0, bg="#0d0d1a", fg="white",
                       activebackground="#e94560", activeforeground="white",
                       font=("Courier", 9), relief="flat", bd=0)
        box  = self.boxes[hit]
        menu.add_command(label=f"  Box [{box.class_idx}]  →  reassign to:",
                         state="disabled")
        menu.add_separator()
        for name in CLASS_NAMES:
            idx   = CLASSES[name]
            check = "● " if idx == box.class_idx else "  "
            color = class_color(idx)
            menu.add_command(
                label=f"{check}[{idx}] {name}",
                foreground=color,
                command=lambda n=name: self.reassign_selected_class(n)
            )
        menu.add_separator()
        menu.add_command(label="  🗑  Delete this box",
                         foreground="#ff6666",
                         command=self.delete_selected_box)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    # ── Bulk remap dialog ──────────────────────────────────────────────────
    def open_remap_dialog(self):
        """
        Remap dialog.

        SOURCE column: every class index actually present in loaded labels
                       PLUS every index defined in CLASSES — shown as
                       "idx  name-if-known".
        TARGET column: a Spinbox (integer entry). Type any integer you want,
                       including indices that don't exist in your CLASSES dict.
                       This lets you collapse a fine-grained model (e.g. 15
                       classes) down to a coarser one (e.g. 5 classes).

        Multiple rows can be changed at once.
        Scope: current image  or  all loaded label files.
        """
        # ── Collect all class indices actually present in loaded labels ──
        present_indices: set[int] = set()
        for entry in self.image_entries:
            lbl = entry["lbl"]
            if lbl and lbl.exists():
                with open(lbl) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                present_indices.add(int(parts[0]))
                            except ValueError:
                                pass
        # Also add currently drawn boxes
        for b in self.boxes:
            present_indices.add(b.class_idx)
        # Merge with CLASSES-defined indices
        all_indices = sorted(present_indices | set(CLASSES.values()))

        def idx_label(idx: int) -> str:
            name = INDEX_TO_CLASS.get(idx)
            return f"{idx}  ({name})" if name else f"{idx}  (unknown)"

        # ── Dialog window ────────────────────────────────────────────────
        dlg = tk.Toplevel(self)
        dlg.title("Remap Classes")
        dlg.configure(bg="#1a1a2e")
        dlg.resizable(True, False)
        dlg.grab_set()

        tk.Label(dlg, text="BULK CLASS REMAPPING",
                 bg="#1a1a2e", fg="#e94560",
                 font=("Courier", 12, "bold"), pady=8).pack()

        tk.Label(dlg,
                 text="Source = class index found in your labels.\n"
                      "Target = any integer you want (type it or spin).\n"
                      "Rows where source = target are skipped.",
                 bg="#1a1a2e", fg="#888", font=("Courier", 9)).pack(pady=(0, 6))

        # ── Scrollable rows area ─────────────────────────────────────────
        outer = tk.Frame(dlg, bg="#1a1a2e")
        outer.pack(fill=tk.BOTH, expand=True, padx=12)

        canvas_scroll = tk.Canvas(outer, bg="#1a1a2e", highlightthickness=0,
                                   height=min(40 * len(all_indices) + 10, 480))
        vsb = tk.Scrollbar(outer, orient="vertical", command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        rows_frame = tk.Frame(canvas_scroll, bg="#1a1a2e")
        canvas_scroll.create_window((0, 0), window=rows_frame, anchor="nw")

        def on_frame_configure(e):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        rows_frame.bind("<Configure>", on_frame_configure)

        # Header
        hdr_bg = "#0f3460"
        for col, (txt, w) in enumerate([("Source index / name", 24),
                                          ("→", 3),
                                          ("Target index", 14),
                                          ("", 6)]):
            tk.Label(rows_frame, text=txt, bg=hdr_bg, fg="white",
                     font=("Courier", 9, "bold"), padx=8, pady=4,
                     width=w, anchor="w").grid(row=0, column=col, sticky="ew", padx=1)

        # One row per index
        target_vars: dict[int, tk.StringVar] = {}   # src_idx -> StringVar(str(int))

        for i, src_idx in enumerate(all_indices):
            row = i + 1
            bg  = "#0d0d1a" if i % 2 == 0 else "#111125"
            color = class_color(src_idx)

            # Source label
            tk.Label(rows_frame, text=idx_label(src_idx),
                     bg=bg, fg=color,
                     font=("Courier", 10, "bold"), padx=10, pady=4,
                     anchor="w", width=24
                     ).grid(row=row, column=0, sticky="ew")

            tk.Label(rows_frame, text="→", bg=bg, fg="#555",
                     font=("Courier", 12)
                     ).grid(row=row, column=1, padx=6)

            var = tk.StringVar(value=str(src_idx))
            target_vars[src_idx] = var

            # Spinbox: accepts any integer; user can also type freely
            sb = tk.Spinbox(rows_frame, from_=0, to=9999,
                             textvariable=var, width=12,
                             bg=bg, fg="white",
                             insertbackground="white",
                             buttonbackground="#0f3460",
                             font=("Courier", 10, "bold"),
                             relief="flat", bd=0)
            sb.grid(row=row, column=2, sticky="ew", padx=4, pady=2)

            # Quick-set buttons: one per known class
            btn_row = tk.Frame(rows_frame, bg=bg)
            btn_row.grid(row=row, column=3, sticky="w", padx=4)
            for tgt_name in CLASS_NAMES:
                tgt_idx = CLASSES[tgt_name]
                c = class_color(tgt_idx)
                tk.Button(btn_row, text=str(tgt_idx),
                           bg="#1a1a2e", fg=c,
                           font=("Courier", 8, "bold"),
                           relief="flat", padx=3, pady=1,
                           cursor="hand2",
                           activebackground=c, activeforeground="black",
                           command=lambda v=var, t=str(tgt_idx): v.set(t)
                           ).pack(side=tk.LEFT, padx=1)

        # ── Preview ──────────────────────────────────────────────────────
        preview_var = tk.StringVar(value="No changes pending.")
        tk.Label(dlg, textvariable=preview_var,
                 bg="#1a1a2e", fg="#44ff88", font=("Courier", 8),
                 wraplength=560, justify="left", pady=4).pack(padx=12)

        def update_preview(*_):
            changes = []
            for si, v in target_vars.items():
                try:
                    ti = int(v.get())
                except ValueError:
                    continue
                if ti != si:
                    tname = INDEX_TO_CLASS.get(ti, "?")
                    sname = INDEX_TO_CLASS.get(si, "?")
                    changes.append(f"{si}({sname})→{ti}({tname})")
            preview_var.set("Pending: " + "  |  ".join(changes) if changes else "No changes pending.")

        for v in target_vars.values():
            v.trace_add("write", update_preview)
        update_preview()

        # ── Scope ─────────────────────────────────────────────────────────
        scope_frame = tk.Frame(dlg, bg="#1a1a2e")
        scope_frame.pack(fill=tk.X, padx=12, pady=(4, 2))
        scope_var = tk.StringVar(value="current")
        tk.Label(scope_frame, text="Apply to:", bg="#1a1a2e", fg="#aaa",
                 font=("Courier", 9, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        for val, lbl_text in [("current", "Current image only"),
                                ("all",     "ALL loaded images (rewrites label files)")]:
            tk.Radiobutton(scope_frame, text=lbl_text,
                           variable=scope_var, value=val,
                           bg="#1a1a2e", fg="#ccc", selectcolor="#0f3460",
                           activebackground="#1a1a2e", activeforeground="white",
                           font=("Courier", 9)).pack(side=tk.LEFT, padx=6)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_frame = tk.Frame(dlg, bg="#1a1a2e")
        btn_frame.pack(pady=8)

        def apply():
            # Build remap: src_int -> tgt_int, skip unchanged
            remap: dict[int, int] = {}
            bad = []
            for si, v in target_vars.items():
                raw = v.get().strip()
                try:
                    ti = int(raw)
                except ValueError:
                    bad.append(f"\"{raw}\" for source {si}")
                    continue
                if ti != si:
                    remap[si] = ti

            if bad:
                messagebox.showerror("Invalid input",
                    "These target values are not integers:\n" + "\n".join(bad))
                return
            if not remap:
                messagebox.showinfo("Nothing to do", "All targets equal their sources.")
                return

            scope = scope_var.get()
            if scope == "current":
                self._apply_remap_to_boxes(self.boxes, remap)
                self._auto_save()
                self._refresh_box_list()
                self._redraw()
                self.status(f"Remapped {len(remap)} class(es) on current image.")
            else:
                self._auto_save()
                self._apply_remap_to_boxes(self.boxes, remap)
                self._refresh_box_list()
                self._redraw()
                changed_files = 0
                iw_cache: dict = {}
                for idx_e, entry in enumerate(self.image_entries):
                    lbl = entry["lbl"]
                    if lbl is None or not lbl.exists():
                        continue
                    img_path = entry["img"]
                    if img_path not in iw_cache:
                        iw, ih = Image.open(img_path).size
                        iw_cache[img_path] = (iw, ih)
                    iw, ih = iw_cache[img_path]
                    boxes = self._load_existing_labels(img_path, lbl)
                    self._apply_remap_to_boxes(boxes, remap)
                    lbl.parent.mkdir(parents=True, exist_ok=True)
                    with open(lbl, "w") as f:
                        for b in boxes:
                            f.write(b.to_yolo(iw, ih) + "\n")
                    # Mirror to output_dir if set
                    if self.output_dir:
                        out_lbl = self.output_dir / "labels" / (img_path.stem + ".txt")
                        out_lbl.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_lbl, "w") as f:
                            for b in boxes:
                                f.write(b.to_yolo(iw, ih) + "\n")
                    changed_files += 1
                self.status(
                    f"Remapped {len(remap)} class(es) across "
                    f"{changed_files} file(s)."
                    + (f"  Mirrored to {self.output_dir / 'labels'}"
                       if self.output_dir else "")
                )
            dlg.destroy()

        tk.Button(btn_frame, text="Apply Remap", command=apply,
                  bg="#e94560", fg="white", font=("Courier", 10, "bold"),
                  relief="flat", padx=20, pady=5, cursor="hand2",
                  activebackground="#ff6688", activeforeground="white"
                  ).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_frame, text="Reset all", command=lambda: [
                      v.set(str(si)) for si, v in target_vars.items()],
                  bg="#222", fg="#aaa", font=("Courier", 9),
                  relief="flat", padx=12, pady=5, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_frame, text="Cancel", command=dlg.destroy,
                  bg="#333", fg="white", font=("Courier", 10),
                  relief="flat", padx=20, pady=5, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=6)


    @staticmethod
    def _apply_remap_to_boxes(boxes: list, remap: dict):
        """In-place: change class_idx of each box according to remap dict."""
        for b in boxes:
            if b.class_idx in remap:
                b.class_idx = remap[b.class_idx]

    # =========================================================================
    # Image loading
    # =========================================================================
    def load_image(self, idx: int):
        if not self.image_paths or idx < 0 or idx >= len(self.image_paths):
            return
        if self.current_idx >= 0:
            self._auto_save()

        self.current_idx  = idx
        entry             = self.image_entries[idx]
        img_path          = entry["img"]
        self.pil_image    = Image.open(img_path).convert("RGB")
        self.boxes        = self._load_existing_labels(img_path, entry["lbl"])
        self.selected_box = None

        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(idx)
        self.image_listbox.see(idx)
        self.progress_label.config(text=f"{idx+1} / {len(self.image_paths)}")
        self._redraw()
        self._refresh_box_list()

        split_info = f"[{entry['split']}]  " if entry["split"] else ""
        n_conf  = sum(not b.ghost for b in self.boxes)
        n_ghost = sum(b.ghost     for b in self.boxes)
        self.status(
            f"{split_info}{img_path.name}  "
            f"({self.pil_image.width}x{self.pil_image.height})"
            f"  |  {n_conf} box(es)"
            + (f"  {n_ghost} ghost(s)" if n_ghost else "")
        )
        self._refresh_listbox_row(idx)

    def _load_existing_labels(self, image_path: Path,
                               label_path=None) -> list:
        """Load YOLO labels. label_path overrides the default sibling .txt."""
        if label_path is None:
            label_path = image_path.with_suffix(".txt")
        boxes = []
        if label_path and label_path.exists():
            iw, ih = Image.open(image_path).size
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            boxes.append(BBox.from_yolo(line, iw, ih))
                        except Exception:
                            pass
        return boxes

    def _refresh_listbox_row(self, idx: int):
        """Update colour and checkmark of a single listbox row after save."""
        if idx < 0 or idx >= self.image_listbox.size():
            return
        entry    = self.image_entries[idx]
        lbl      = entry["lbl"]
        labeled  = lbl is not None and lbl.exists() and lbl.stat().st_size > 0
        split_tag = f"[{entry['split']}] " if entry["split"] else ""
        has_lbl   = "\u2714 " if labeled else "  "
        self.image_listbox.delete(idx)
        self.image_listbox.insert(idx, f"{has_lbl}{split_tag}{entry['img'].name}")
        color = "#88ccff" if labeled else "#555"
        self.image_listbox.itemconfig(idx, fg=color)
        # Restore selection if this row is current
        if idx == self.current_idx:
            self.image_listbox.selection_set(idx)

    # =========================================================================
    # Canvas rendering
    # =========================================================================
    def _redraw(self):
        self.canvas.delete("all")
        if self.pil_image is None:
            self._draw_placeholder()
            return

        cw = self.canvas.winfo_width()  or self.CANVAS_W
        ch = self.canvas.winfo_height() or self.CANVAS_H
        iw, ih = self.pil_image.size
        self.scale    = min(cw / iw, ch / ih)
        nw, nh        = int(iw * self.scale), int(ih * self.scale)
        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2

        resized       = self.pil_image.resize((nw, nh), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y,
                                  anchor="nw", image=self.tk_image)

        for i, box in enumerate(self.boxes):
            self._draw_box(i, box)

    def _draw_placeholder(self):
        cw = self.canvas.winfo_width()  or self.CANVAS_W
        ch = self.canvas.winfo_height() or self.CANVAS_H
        self.canvas.create_text(cw // 2, ch // 2,
                                 text="Open images to begin",
                                 fill="#333", font=("Courier", 16))

    def _draw_box(self, idx: int, box: BBox):
        x1, y1 = self._img_to_canvas(box.x1, box.y1)
        x2, y2 = self._img_to_canvas(box.x2, box.y2)
        selected = (idx == self.selected_box)

        if box.ghost:
            # White dashed outline for predictions
            outline = "#ffffff"
            dash    = (6, 3)
            width   = 2 if not selected else 3
            # semi-transparent tint: draw a thin coloured inner rect
            inner_color = class_color(box.class_idx)
            self.canvas.create_rectangle(x1+2, y1+2, x2-2, y2-2,
                                          outline=inner_color, width=1, dash=(2,4))
        else:
            outline = class_color(box.class_idx)
            dash    = (4, 2) if selected else None
            width   = 3 if selected else 2

        self.canvas.create_rectangle(x1, y1, x2, y2,
                                      outline=outline, width=width,
                                      dash=dash, tags=f"box_{idx}")

        # Label pill
        name  = INDEX_TO_CLASS.get(box.class_idx, str(box.class_idx))
        conf_str = f" {box.conf:.0%}" if box.conf is not None else ""
        ghost_str = " ?" if box.ghost else ""
        lbl   = f" [{box.class_idx}] {name}{conf_str}{ghost_str} "
        tx, ty = x1 + 2, y1 - 15
        pill_bg = "#444444" if box.ghost else class_color(box.class_idx)
        pill_fg = "white"   if box.ghost else "black"
        char_w  = 6
        self.canvas.create_rectangle(tx - 1, ty,
                                      tx + len(lbl) * char_w + 1, ty + 13,
                                      fill=pill_bg, outline="")
        self.canvas.create_text(tx + 3, ty + 7, text=lbl,
                                 fill=pill_fg,
                                 font=("Courier", 8, "bold"), anchor="w")

    # =========================================================================
    # Coordinate helpers
    # =========================================================================
    def _img_to_canvas(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def _canvas_to_img(self, cx, cy):
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    # =========================================================================
    # Mouse events
    # =========================================================================
    def _on_mouse_press(self, event):
        if self.pil_image is None:
            return
        clicked = self._hit_test(event.x, event.y)
        if clicked is not None:
            self.selected_box = clicked
            self._refresh_box_list()
            self._redraw()
            return
        self.drawing      = True
        self.drag_start   = (event.x, event.y)
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
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            self.drag_start = None
            return

        ix0, iy0 = self._canvas_to_img(x0, y0)
        ix1, iy1 = self._canvas_to_img(x1, y1)
        iw, ih   = self.pil_image.size
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(iw, ix1), min(ih, iy1)

        box = BBox(ix0, iy0, ix1, iy1, self.active_class_idx())
        self.boxes.append(box)
        self.selected_box = len(self.boxes) - 1
        self.drag_start   = None
        self._refresh_box_list()
        self._redraw()
        self.status(f"Added [{box.class_idx}] {INDEX_TO_CLASS.get(box.class_idx, '?')}")

    def _hit_test(self, cx, cy) -> "int | None":
        ix, iy = self._canvas_to_img(cx, cy)
        for i in range(len(self.boxes) - 1, -1, -1):
            b = self.boxes[i]
            if b.x1 <= ix <= b.x2 and b.y1 <= iy <= b.y2:
                return i
        return None

    # =========================================================================
    # Box list panel
    # =========================================================================
    def _refresh_box_list(self):
        self.box_listbox.delete(0, tk.END)
        for i, b in enumerate(self.boxes):
            name      = INDEX_TO_CLASS.get(b.class_idx, str(b.class_idx))
            w, h      = int(b.x2 - b.x1), int(b.y2 - b.y1)
            mark      = "►" if i == self.selected_box else " "
            kind      = "?" if b.ghost else "✔"
            conf_str  = f" {b.conf:.0%}" if b.conf is not None else ""
            self.box_listbox.insert(tk.END,
                f"{mark}{kind} [{b.class_idx}] {name}{conf_str}  {w}×{h}")
            # colour ghost rows differently
            if b.ghost:
                self.box_listbox.itemconfig(i, fg="#888888")
            else:
                self.box_listbox.itemconfig(i, fg=class_color(b.class_idx))

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

    # =========================================================================
    # Box operations
    # =========================================================================
    def delete_selected_box(self):
        if self.selected_box is None or not self.boxes:
            return
        self.boxes.pop(self.selected_box)
        self.selected_box = None
        self._refresh_box_list()
        self._redraw()

    def prev_image(self):
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.load_image(self.current_idx + 1)

    # =========================================================================
    # Saving
    # =========================================================================
    def _auto_save(self):
        """Persist only confirmed (non-ghost) boxes.
        Always writes to the entry label path (source).
        If an output_dir is set, ALSO writes a mirror copy there under
        output_dir/labels/<stem>.txt so the output folder stays up to date."""
        if self.current_idx < 0 or not self.image_paths or self.pil_image is None:
            return
        entry      = self.image_entries[self.current_idx]
        label_path = entry["lbl"]
        if label_path is None:
            label_path = entry["img"].with_suffix(".txt")
            entry["lbl"] = label_path

        iw, ih    = self.pil_image.size
        confirmed = [b for b in self.boxes if not b.ghost]
        lines     = [b.to_yolo(iw, ih) for b in confirmed]

        # Write to source location
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        # Mirror to output_dir/labels/ if set
        if self.output_dir:
            out_lbl_dir = self.output_dir / "labels"
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_path = out_lbl_dir / (entry["img"].stem + ".txt")
            with open(out_lbl_path, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            # Also mirror the image once (skip if already there)
            out_img_dir  = self.output_dir / "images"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_img_path = out_img_dir / entry["img"].name
            if not out_img_path.exists():
                shutil.copy2(entry["img"], out_img_path)

    def save_current(self):
        self._auto_save()
        if self.current_idx >= 0:
            entry = self.image_entries[self.current_idx]
            n     = sum(not b.ghost for b in self.boxes)
            self.status(f"Saved {n} box(es) for {entry['img'].name}")
            self._refresh_listbox_row(self.current_idx)

    # =========================================================================
    # Export dataset
    # =========================================================================
    def export_all(self):
        if not self.output_dir:
            messagebox.showerror("No Output Dir", "Set an output directory first.")
            return
        if not self.image_paths:
            messagebox.showerror("No Images", "No images loaded.")
            return

        self._auto_save()

        result = self._ask_train_ratio()
        if result is None:
            return
        train_ratio = result

        labeled = [e["img"] for e in self.image_entries
                   if e["lbl"] and e["lbl"].exists() and e["lbl"].stat().st_size > 0]
        if not labeled:
            messagebox.showwarning("No Labels", "No images have been labeled yet.")
            return

        random.shuffle(labeled)
        split     = int(len(labeled) * train_ratio)
        train_set = labeled[:split]
        val_set   = labeled[split:]

        for split_name, files in [("train", train_set), ("val", val_set)]:
            img_dir   = self.output_dir / split_name / "images"
            lbl_dir   = self.output_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for p in files:
                # find matching entry to get its label path
                entry = next((e for e in self.image_entries if e["img"] == p), None)
                lp    = entry["lbl"] if entry else p.with_suffix(".txt")
                shutil.copy2(p, img_dir / p.name)
                if lp and lp.exists():
                    shutil.copy2(lp, lbl_dir / lp.name)

        yaml_content = "\n".join([
            f"path: {self.output_dir}",
            "train: train/images",
            "val:   val/images",
            "",
            f"nc: {len(CLASSES)}",
            f"names: {CLASS_NAMES}",
            "",
            "# Class map:",
            *[f"#   {idx}: {name}"
              for name, idx in sorted(CLASSES.items(), key=lambda x: x[1])],
        ])
        with open(self.output_dir / "data.yaml", "w") as f:
            f.write(yaml_content + "\n")

        messagebox.showinfo("Export Complete",
            f"Dataset exported!\n\n"
            f"  Train: {len(train_set)} images\n"
            f"  Val:   {len(val_set)} images\n\n"
            f"  Output: {self.output_dir}")
        self.status(f"Exported {len(labeled)} images → {self.output_dir}")

    def _ask_train_ratio(self) -> "float | None":
        dialog = tk.Toplevel(self)
        dialog.title("Train / Val Split")
        dialog.configure(bg="#1a1a2e")
        dialog.resizable(False, False)
        dialog.grab_set()
        result = {"value": None}

        tk.Label(dialog, text="Train ratio  (0.8 = 80% train, 20% val)",
                 bg="#1a1a2e", fg="white", font=("Courier", 10), padx=16, pady=12).pack()
        var = tk.DoubleVar(value=0.8)
        tk.Scale(dialog, from_=0.5, to=1.0, resolution=0.05,
                 orient=tk.HORIZONTAL, variable=var, length=280,
                 bg="#1a1a2e", fg="white", troughcolor="#0f3460",
                 highlightthickness=0, font=("Courier", 9)).pack(padx=16)

        def confirm():
            result["value"] = var.get(); dialog.destroy()
        def cancel():
            dialog.destroy()

        bf = tk.Frame(dialog, bg="#1a1a2e"); bf.pack(pady=10)
        tk.Button(bf, text="Export", command=confirm,
                  bg="#e94560", fg="white", font=("Courier", 10, "bold"),
                  relief="flat", padx=16, pady=4).pack(side=tk.LEFT, padx=6)
        tk.Button(bf, text="Cancel", command=cancel,
                  bg="#333", fg="white", font=("Courier", 10),
                  relief="flat", padx=16, pady=4).pack(side=tk.LEFT, padx=6)

        dialog.wait_window()
        return result["value"]

    # =========================================================================
    # Status
    # =========================================================================
    def status(self, msg: str):
        self.status_var.set(msg)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = LabelerApp()
    app.mainloop()
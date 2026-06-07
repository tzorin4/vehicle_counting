"""
Vehicle Counter — Tkinter desktop app
Counts cars, motorbikes, trucks, buses crossing a configurable line using YOLO.
Run:  python vehicle_counter_tk.py
Deps: pip install opencv-python numpy pillow ultralytics   (or onnxruntime for .onnx)
"""

import os, sys, glob, threading, time
from pathlib import Path
from collections import deque

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    sys.exit("tkinter not found. Install python3-tk (Linux) or use the standard Python installer (Windows/macOS).")

try:
    from PIL import Image, ImageTk
except ImportError:
    sys.exit("Pillow not found. Run: pip install pillow")

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = {"car": 0, "motorbike": 1, "truck": 2, "bus": 3}
CLASS_COLORS_BGR = {
    "car":       (153, 211,  52),
    "motorbike": ( 36, 191, 251),
    "truck":     ( 68,  68, 239),
    "bus":       (246,  92, 139),
}
CLASS_COLORS_HEX = {
    "car":       "#34d399",
    "motorbike": "#fbbf24",
    "truck":     "#f87171",
    "bus":       "#a78bfa",
}

# Dark-theme palette
BG      = "#0f1117"
SURFACE = "#1a1d27"
SURFACE2= "#20243a"
BORDER  = "#2a2f45"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"
ACCENT  = "#38bdf8"
GREEN   = "#34d399"
RED     = "#f87171"

# ─────────────────────────────────────────────────────────────────────────────
#  Model loader
# ─────────────────────────────────────────────────────────────────────────────
_model_cache = {}

def load_model(path: str):
    if path in _model_cache:
        return _model_cache[path]
    ext = Path(path).suffix.lower()
    if ext == ".onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _model_cache[path] = ("onnx", sess)
    elif ext in (".pt", ".pth"):
        from ultralytics import YOLO
        m = YOLO(path)
        _model_cache[path] = ("ultralytics", m)
    else:
        raise ValueError(f"Unsupported model: {ext}  (use .pt or .onnx)")
    return _model_cache[path]


def infer(model_info, frame, conf, detect_classes):
    kind, model = model_info
    if kind == "ultralytics":
        ids = [CLASSES[c] for c in detect_classes if c in CLASSES]
        res = model(frame, conf=conf, classes=ids, verbose=False)[0]
        out = []
        id2n = {v: k for k, v in CLASSES.items()}
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cn = id2n.get(cls_id)
            if cn is None: continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            out.append((x1,y1,x2,y2,cn,float(box.conf[0])))
        return out
    else:                          # ONNX
        input_size = 640
        h0,w0 = frame.shape[:2]
        scale = input_size / max(h0, w0)
        nw,nh = int(w0*scale), int(h0*scale)
        resized = cv2.resize(frame, (nw, nh))
        pw,ph = (input_size-nw)//2, (input_size-nh)//2
        padded = np.full((input_size,input_size,3), 114, np.uint8)
        padded[ph:ph+nh, pw:pw+nw] = resized
        blob = padded.astype(np.float32)/255.
        blob = blob.transpose(2,0,1)[np.newaxis]
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: blob})[0]
        preds = outputs[0].T
        id2n = {v: k for k, v in CLASSES.items()}
        out = []
        for pred in preds:
            scores = pred[4:]
            cls_id = int(np.argmax(scores))
            c = float(scores[cls_id])
            if c < conf: continue
            cn = id2n.get(cls_id)
            if cn is None or cn not in detect_classes: continue
            cx,cy,w,h = pred[:4]
            cx=(cx-pw)/scale; cy=(cy-ph)/scale; w/=scale; h/=scale
            out.append((int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2),cn,c))
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Centroid Tracker
# ─────────────────────────────────────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_gone=15):
        self.next_id = 0
        self.objects = {}      # id->(cx,cy)
        self.gone    = {}      # id->int
        self.cls     = {}      # id->str
        self.max_gone = max_gone

    def _reg(self, cx, cy, cn):
        i = self.next_id
        self.objects[i]=(cx,cy); self.gone[i]=0; self.cls[i]=cn
        self.next_id+=1
        return i

    def _del(self, i):
        del self.objects[i], self.gone[i], self.cls[i]

    def update(self, dets):   # dets: [(cx,cy,cn), ...]
        if not dets:
            for i in list(self.gone):
                self.gone[i]+=1
                if self.gone[i]>self.max_gone: self._del(i)
            return dict(self.objects)

        if not self.objects:
            for cx,cy,cn in dets: self._reg(cx,cy,cn)
            return {i:(self.objects[i][0],self.objects[i][1],self.cls[i]) for i in self.objects}

        ids  = list(self.objects)
        oc   = np.array(list(self.objects.values()))
        dc   = np.array([(d[0],d[1]) for d in dets])
        D    = np.linalg.norm(oc[:,None]-dc[None,:], axis=2)
        rows = D.min(1).argsort()
        cols = D.argmin(1)[rows]
        ur,uc = set(),set()
        for r,c in zip(rows,cols):
            if r in ur or c in uc: continue
            i = ids[r]
            self.objects[i]=(int(dc[c][0]),int(dc[c][1]))
            self.cls[i]=dets[c][2]; self.gone[i]=0
            ur.add(r); uc.add(c)
        for r in set(range(len(ids)))-ur:
            i=ids[r]; self.gone[i]+=1
            if self.gone[i]>self.max_gone: self._del(i)
        for c in set(range(len(dets)))-uc:
            self._reg(dets[c][0],dets[c][1],dets[c][2])
        return {i:(self.objects[i][0],self.objects[i][1],self.cls[i]) for i in self.objects}


def side(px,py,x1,y1,x2,y2):
    return (x2-x1)*(py-y1)-(y2-y1)*(px-x1)


# ─────────────────────────────────────────────────────────────────────────────
#  Draw helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_frame(frame, lx1,ly1,lx2,ly2, detections, active, newly_counted, counts, frame_idx, total):
    vis = frame.copy()
    h,w = vis.shape[:2]

    # Counting line
    cv2.line(vis,(lx1,ly1),(lx2,ly2),(0,220,255),3)
    cv2.line(vis,(lx1,ly1),(lx2,ly2),(255,255,255),1)

    # Bounding boxes
    for x1,y1b,x2,y2b,cn,cf in detections:
        bgr = CLASS_COLORS_BGR.get(cn,(200,200,200))
        cv2.rectangle(vis,(x1,y1b),(x2,y2b),bgr,2)
        lbl = f"{cn} {cf:.0%}"
        (lw,lh),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
        cv2.rectangle(vis,(x1,y1b-lh-8),(x1+lw+4,y1b),bgr,-1)
        cv2.putText(vis,lbl,(x1+2,y1b-4),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)

    # Centroids + IDs
    for oid,(cx,cy,cn) in active.items():
        bgr = CLASS_COLORS_BGR.get(cn,(200,200,200))
        cv2.circle(vis,(cx,cy),5,bgr,-1)
        cv2.putText(vis,f"#{oid}",(cx+6,cy-6),cv2.FONT_HERSHEY_SIMPLEX,0.38,bgr,1,cv2.LINE_AA)

    # Flash events
    for oid,cn in newly_counted:
        bgr = CLASS_COLORS_BGR.get(cn,(200,200,200))
        info = active.get(oid)
        if info:
            cx,cy,_ = info
            cv2.circle(vis,(cx,cy),22,bgr,3)
            cv2.putText(vis,f"+1 {cn}",(cx+26,cy),cv2.FONT_HERSHEY_SIMPLEX,0.65,bgr,2,cv2.LINE_AA)

    # Count overlay
    ov = vis.copy()
    cv2.rectangle(ov,(6,6),(215,34+len(CLASSES)*27),(0,0,0),-1)
    cv2.addWeighted(ov,0.55,vis,0.45,0,vis)
    cv2.putText(vis,"COUNTS",(14,26),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),1,cv2.LINE_AA)
    for i,(cn,cnt) in enumerate(counts.items()):
        bgr = CLASS_COLORS_BGR.get(cn,(200,200,200))
        cv2.putText(vis,f"{cn}: {cnt}",(14,52+i*27),cv2.FONT_HERSHEY_SIMPLEX,0.54,bgr,1,cv2.LINE_AA)

    # Frame counter
    cv2.putText(vis,f"{frame_idx+1}/{total}",(w-110,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1,cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
#  Main GUI
# ─────────────────────────────────────────────────────────────────────────────
class VehicleCounterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vehicle Counter — YOLO Line Crossing")
        self.configure(bg=BG)
        self.minsize(1100, 700)
        self._apply_style()

        # State
        self.frames_data   = []     # list of np arrays (annotated frames)
        self.current_frame = 0
        self.total_frames  = 0
        self.playing       = False
        self._play_after   = None
        self._proc_thread  = None
        self._cancel_flag  = False
        self.counts        = {c:0 for c in CLASSES}
        self._photo        = None   # keep ImageTk ref

        # Line endpoints as % (0-100)
        self.line_x1 = tk.IntVar(value=0)
        self.line_y1 = tk.IntVar(value=50)
        self.line_x2 = tk.IntVar(value=100)
        self.line_y2 = tk.IntVar(value=50)

        # Settings vars
        self.model_path   = tk.StringVar()
        self.source_path  = tk.StringVar()
        self.source_type  = tk.StringVar(value="video")   # "video" | "folder"
        self.conf_var     = tk.DoubleVar(value=0.40)
        self.fps_var      = tk.IntVar(value=10)
        self.max_gone_var = tk.IntVar(value=15)
        self.class_vars   = {c: tk.BooleanVar(value=True) for c in CLASSES}

        self._build_ui()
        self._draw_line_canvas()
        self.bind("<Configure>", lambda e: None)   # absorb resize events

    # ── Style ────────────────────────────────────────────────────────────────
    def _apply_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, troughcolor=SURFACE2,
                     fieldbackground=SURFACE2, borderwidth=0, relief="flat")
        s.configure("TFrame",      background=BG)
        s.configure("Surface.TFrame", background=SURFACE)
        s.configure("TLabel",      background=BG,      foreground=TEXT, font=("Segoe UI",10))
        s.configure("Muted.TLabel",background=BG,      foreground=MUTED,  font=("Segoe UI",9))
        s.configure("Head.TLabel", background=BG,      foreground=ACCENT, font=("Consolas",11,"bold"))
        s.configure("Count.TLabel",background=SURFACE, foreground=TEXT,   font=("Consolas",28,"bold"))
        s.configure("CName.TLabel",background=SURFACE, foreground=MUTED,  font=("Segoe UI",9))
        s.configure("TEntry",      fieldbackground=SURFACE2, foreground=TEXT,
                     insertcolor=TEXT, borderwidth=1, relief="flat")
        s.configure("TCheckbutton",background=BG, foreground=TEXT, font=("Segoe UI",10))
        s.map("TCheckbutton", background=[("active",BG)])
        s.configure("TScale",  background=BG, troughcolor=SURFACE2)
        s.configure("TProgressbar", troughcolor=SURFACE2, background=ACCENT, thickness=6)
        s.configure("TButton", background=SURFACE2, foreground=TEXT,
                     font=("Segoe UI",10), borderwidth=0, padding=(10,6))
        s.map("TButton",
              background=[("active",BORDER),("disabled",SURFACE)],
              foreground=[("disabled",MUTED)])
        s.configure("Accent.TButton", background=ACCENT, foreground="#000000",
                     font=("Segoe UI",10,"bold"), padding=(10,7))
        s.map("Accent.TButton", background=[("active","#29a8d8"),("disabled",BORDER)])
        s.configure("Danger.TButton", background="#ef4444", foreground="#ffffff",
                     font=("Segoe UI",10,"bold"), padding=(10,7))
        s.map("Danger.TButton", background=[("active","#c53030")])
        s.configure("Horizontal.TScale", background=BG)

    # ── Build UI ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top header ──
        hdr = tk.Frame(self, bg=SURFACE, height=48)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🚗  VEHICLE COUNTER", bg=SURFACE, fg=ACCENT,
                 font=("Consolas",13,"bold")).pack(side="left", padx=18, pady=12)
        tk.Label(hdr, text="YOLO · Line Crossing · Centroid Tracker", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI",9)).pack(side="left", pady=12)
        self._status_lbl = tk.Label(hdr, text="● IDLE", bg=SURFACE, fg=MUTED,
                                     font=("Consolas",9,"bold"))
        self._status_lbl.pack(side="right", padx=18)

        # ── Main paned window ──
        paned = tk.PanedWindow(self, orient="horizontal", bg=BG,
                               sashwidth=4, sashrelief="flat", sashpad=0)
        paned.pack(fill="both", expand=True)

        # Left panel (settings)
        left = tk.Frame(paned, bg=BG, width=300)
        left.pack_propagate(False)
        paned.add(left, minsize=260)

        # Right panel (viewer + counts)
        right = tk.Frame(paned, bg=BG)
        paned.add(right, minsize=500)

        self._build_left(left)
        self._build_right(right)

    # ── Left panel ───────────────────────────────────────────────────────────
    def _build_left(self, parent):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0,0), window=inner, anchor="nw")

        def on_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", on_resize)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _mousewheel(e):
            canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _mousewheel)

        P = 14  # padding

        # ── Source ──
        self._section(inner, "SOURCE")
        src_frame = tk.Frame(inner, bg=BG)
        src_frame.pack(fill="x", padx=P, pady=(0,4))
        ttk.Radiobutton(src_frame, text="Video file", variable=self.source_type,
                        value="video",  command=self._pick_source).pack(side="left")
        ttk.Radiobutton(src_frame, text="Image folder", variable=self.source_type,
                        value="folder", command=self._pick_source).pack(side="left", padx=8)

        src_row = tk.Frame(inner, bg=BG)
        src_row.pack(fill="x", padx=P, pady=(0,2))
        self._src_entry = ttk.Entry(src_row, textvariable=self.source_path, width=26)
        self._src_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(src_row, text="Browse", command=self._pick_source).pack(side="left", padx=(4,0))

        # ── Model ──
        self._section(inner, "MODEL  (.pt or .onnx)")
        mdl_row = tk.Frame(inner, bg=BG)
        mdl_row.pack(fill="x", padx=P, pady=(0,4))
        ttk.Entry(mdl_row, textvariable=self.model_path, width=26).pack(side="left", fill="x", expand=True)
        ttk.Button(mdl_row, text="Browse",
                   command=lambda: self.model_path.set(
                       filedialog.askopenfilename(title="Select model",
                           filetypes=[("Model","*.pt *.pth *.onnx"),("All","*.*")])
                   )).pack(side="left", padx=(4,0))

        # ── Counting line ──
        self._section(inner, "COUNTING LINE")
        ttk.Label(inner, text="Drag endpoints on the preview below (% of frame)",
                  style="Muted.TLabel").pack(padx=P, anchor="w")

        # Canvas for line preview
        self._line_canvas = tk.Canvas(inner, bg="#0a0c14", height=140,
                                      highlightthickness=1, highlightbackground=BORDER,
                                      cursor="crosshair")
        self._line_canvas.pack(fill="x", padx=P, pady=6)
        self._line_canvas.bind("<ButtonPress-1>",  self._lc_press)
        self._line_canvas.bind("<B1-Motion>",       self._lc_drag)
        self._line_canvas.bind("<ButtonRelease-1>", self._lc_release)
        self._line_canvas.bind("<Configure>",       lambda e: self._draw_line_canvas())
        self._drag_pt = None  # "p1" or "p2"

        # Numeric inputs for line
        grid = tk.Frame(inner, bg=BG)
        grid.pack(fill="x", padx=P, pady=(0,6))
        for col,(lbl,var) in enumerate([("X1 %",self.line_x1),("Y1 %",self.line_y1),
                                         ("X2 %",self.line_x2),("Y2 %",self.line_y2)]):
            f = tk.Frame(grid, bg=BG)
            f.grid(row=0, column=col, padx=3, sticky="ew")
            grid.columnconfigure(col, weight=1)
            tk.Label(f, text=lbl, bg=BG, fg=MUTED, font=("Segoe UI",8)).pack(anchor="w")
            e = ttk.Entry(f, textvariable=var, width=5)
            e.pack(fill="x")
            var.trace_add("write", lambda *a: self._draw_line_canvas())

        # ── Classes ──
        self._section(inner, "DETECT CLASSES")
        cls_frame = tk.Frame(inner, bg=BG)
        cls_frame.pack(fill="x", padx=P, pady=(0,6))
        for i,(cn,var) in enumerate(self.class_vars.items()):
            col = CLASS_COLORS_HEX.get(cn,"#aaa")
            cb = tk.Checkbutton(cls_frame, text=cn, variable=var,
                                bg=BG, fg=col, selectcolor=SURFACE2,
                                activebackground=BG, activeforeground=col,
                                font=("Consolas",10), bd=0, highlightthickness=0)
            cb.grid(row=i//2, column=i%2, sticky="w", padx=4, pady=2)
        cls_frame.columnconfigure(0,weight=1); cls_frame.columnconfigure(1,weight=1)

        # ── Parameters ──
        self._section(inner, "PARAMETERS")
        self._slider_row(inner, "Confidence",   self.conf_var,     0.05, 0.95, 0.05, ".2f", P)
        self._slider_row(inner, "Max Disappeared", self.max_gone_var, 3,  60,  1,   "d",  P)
        self._slider_row(inner, "Playback FPS",  self.fps_var,      1,   60,  1,   "d",  P)

        # ── Buttons ──
        self._section(inner, "ACTIONS")
        btn_frame = tk.Frame(inner, bg=BG)
        btn_frame.pack(fill="x", padx=P, pady=(0,14))

        self._start_btn = ttk.Button(btn_frame, text="▶  Start Processing",
                                     style="Accent.TButton", command=self._start)
        self._start_btn.pack(fill="x", pady=(0,6))

        self._cancel_btn = ttk.Button(btn_frame, text="⏹  Cancel",
                                      style="Danger.TButton", command=self._cancel,
                                      state="disabled")
        self._cancel_btn.pack(fill="x", pady=(0,6))

        self._export_btn = ttk.Button(btn_frame, text="⬇  Export Annotated MP4",
                                      command=self._export, state="disabled")
        self._export_btn.pack(fill="x")

        # Progress bar
        self._prog_var = tk.DoubleVar(value=0)
        self._prog_bar = ttk.Progressbar(inner, variable=self._prog_var,
                                         maximum=100, style="TProgressbar")
        self._prog_bar.pack(fill="x", padx=P, pady=(0,4))
        self._prog_lbl = ttk.Label(inner, text="", style="Muted.TLabel")
        self._prog_lbl.pack(padx=P, anchor="w")

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", padx=14, pady=(14,4))
        tk.Label(f, text=title, bg=BG, fg=MUTED,
                 font=("Consolas",8)).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=(6,0), pady=6)

    def _slider_row(self, parent, label, var, from_, to, res, fmt, padx):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", padx=padx, pady=2)
        lbl = tk.Label(f, bg=BG, fg=MUTED, font=("Segoe UI",9), anchor="w", width=18)
        lbl.pack(side="left")
        val_lbl = tk.Label(f, bg=BG, fg=TEXT, font=("Consolas",9), width=6, anchor="e")
        val_lbl.pack(side="right")

        def update(*_):
            v = var.get()
            val_lbl.config(text=format(v, fmt))
            lbl.config(text=label)
        var.trace_add("write", update)
        update()

        sl = ttk.Scale(f, from_=from_, to=to, variable=var, orient="horizontal")
        sl.pack(side="left", fill="x", expand=True, padx=(4,4))
        # Snap to resolution on release for int vars
        if fmt == "d":
            sl.bind("<ButtonRelease-1>", lambda e: var.set(round(var.get())))

    # ── Right panel ──────────────────────────────────────────────────────────
    def _build_right(self, parent):
        # Count cards row
        cards = tk.Frame(parent, bg=BG)
        cards.pack(fill="x", padx=14, pady=(12,0))
        self._count_labels = {}
        for cn in CLASSES:
            col = CLASS_COLORS_HEX.get(cn,"#aaa")
            card = tk.Frame(cards, bg=SURFACE, bd=0, highlightthickness=2,
                            highlightbackground=BORDER)
            card.pack(side="left", expand=True, fill="x", padx=4)
            # Colored top bar via label
            tk.Frame(card, bg=col, height=3).pack(fill="x")
            tk.Label(card, text=cn.upper(), bg=SURFACE, fg=MUTED,
                     font=("Segoe UI",8)).pack(padx=10, pady=(6,0), anchor="w")
            lbl = tk.Label(card, text="0", bg=SURFACE, fg=col,
                           font=("Consolas",30,"bold"))
            lbl.pack(padx=10, pady=(0,8), anchor="w")
            self._count_labels[cn] = lbl

        # Viewer area
        viewer = tk.Frame(parent, bg=SURFACE, bd=0,
                          highlightthickness=1, highlightbackground=BORDER)
        viewer.pack(fill="both", expand=True, padx=14, pady=10)

        # Toolbar
        toolbar = tk.Frame(viewer, bg=SURFACE2, height=42)
        toolbar.pack(fill="x", side="bottom")
        toolbar.pack_propagate(False)

        self._play_btn = ttk.Button(toolbar, text="▶", width=3,
                                    command=self._toggle_play, state="disabled")
        self._play_btn.pack(side="left", padx=(8,4), pady=6)

        self._frame_slider = ttk.Scale(toolbar, from_=0, to=0, orient="horizontal",
                                        command=self._on_slider)
        self._frame_slider.pack(side="left", fill="x", expand=True, padx=4, pady=6)
        self._frame_slider.state(["disabled"])

        self._frame_lbl = tk.Label(toolbar, text="0 / 0", bg=SURFACE2, fg=MUTED,
                                    font=("Consolas",9), width=10)
        self._frame_lbl.pack(side="left", padx=4)

        tk.Label(toolbar, text="FPS:", bg=SURFACE2, fg=MUTED,
                 font=("Segoe UI",9)).pack(side="left")
        self._fps_display = tk.Label(toolbar, textvariable=self.fps_var,
                                      bg=SURFACE2, fg=TEXT, font=("Consolas",9), width=3)
        self._fps_display.pack(side="left")
        ttk.Scale(toolbar, from_=1, to=60, variable=self.fps_var,
                  orient="horizontal", length=80).pack(side="left", padx=(0,12), pady=6)

        # Canvas for video
        self._vid_canvas = tk.Canvas(viewer, bg="#000000", highlightthickness=0)
        self._vid_canvas.pack(fill="both", expand=True)
        self._vid_canvas.bind("<Configure>", self._on_canvas_resize)

        # Placeholder text
        self._placeholder_id = self._vid_canvas.create_text(
            400, 200,
            text="Configure settings and press  ▶ Start Processing",
            fill=MUTED, font=("Segoe UI",12), anchor="center"
        )

    # ── Line canvas ──────────────────────────────────────────────────────────
    def _draw_line_canvas(self):
        c = self._line_canvas
        c.delete("all")
        W = c.winfo_width()  or 260
        H = c.winfo_height() or 140

        # Grid
        c.configure(bg="#0a0c14")
        for x in range(0, W, W//8):
            c.create_line(x,0,x,H, fill="#ffffff0a" if x else "#0a0c14", width=1)
        for y in range(0, H, H//5):
            c.create_line(0,y,W,y, fill="#ffffff0a", width=1)

        x1 = int(self.line_x1.get()/100*W)
        y1 = int(self.line_y1.get()/100*H)
        x2 = int(self.line_x2.get()/100*W)
        y2 = int(self.line_y2.get()/100*H)

        # Glow (simulated with thicker line)
        c.create_line(x1,y1,x2,y2, fill="#facc1566", width=6, capstyle="round")
        c.create_line(x1,y1,x2,y2, fill="#facc15",   width=2, capstyle="round")
        c.create_line(x1,y1,x2,y2, fill="#ffffff88",  width=1, capstyle="round")

        # Endpoints
        for (ex,ey,col) in [(x1,y1,"#38bdf8"),(x2,y2,"#f472b6")]:
            c.create_oval(ex-7,ey-7,ex+7,ey+7, fill=col, outline="#ffffff", width=1.5)

    def _lc_press(self, e):
        W = self._line_canvas.winfo_width()
        H = self._line_canvas.winfo_height()
        p1x = int(self.line_x1.get()/100*W)
        p1y = int(self.line_y1.get()/100*H)
        p2x = int(self.line_x2.get()/100*W)
        p2y = int(self.line_y2.get()/100*H)
        d1 = (e.x-p1x)**2+(e.y-p1y)**2
        d2 = (e.x-p2x)**2+(e.y-p2y)**2
        self._drag_pt = "p1" if d1<d2 else "p2"

    def _lc_drag(self, e):
        if not self._drag_pt: return
        W = self._line_canvas.winfo_width()
        H = self._line_canvas.winfo_height()
        xp = max(0,min(100,round(e.x/W*100)))
        yp = max(0,min(100,round(e.y/H*100)))
        if self._drag_pt=="p1":
            self.line_x1.set(xp); self.line_y1.set(yp)
        else:
            self.line_x2.set(xp); self.line_y2.set(yp)

    def _lc_release(self, e): self._drag_pt = None

    # ── Source picker ─────────────────────────────────────────────────────────
    def _pick_source(self):
        if self.source_type.get() == "video":
            p = filedialog.askopenfilename(
                title="Select video",
                filetypes=[("Video","*.mp4 *.avi *.mov *.mkv *.webm"),("All","*.*")])
        else:
            p = filedialog.askdirectory(title="Select image folder")
        if p:
            self.source_path.set(p)

    # ── Processing ───────────────────────────────────────────────────────────
    def _start(self):
        mp = self.model_path.get().strip()
        sp = self.source_path.get().strip()
        if not mp:
            messagebox.showerror("Missing", "Please specify a model path."); return
        if not os.path.exists(mp):
            messagebox.showerror("Not found", f"Model not found:\n{mp}"); return
        if not sp:
            messagebox.showerror("Missing", "Please select a video or image folder."); return

        st = self.source_type.get()
        if st == "video" and not os.path.isfile(sp):
            messagebox.showerror("Not found", f"Video file not found:\n{sp}"); return
        if st == "folder" and not os.path.isdir(sp):
            messagebox.showerror("Not found", f"Folder not found:\n{sp}"); return

        chosen = [cn for cn,v in self.class_vars.items() if v.get()]
        if not chosen:
            messagebox.showerror("No classes", "Select at least one class."); return

        # Reset
        self.frames_data.clear()
        self.current_frame = 0
        self.total_frames  = 0
        self.counts = {c:0 for c in CLASSES}
        self._cancel_flag  = False
        self._update_counts({c:0 for c in CLASSES})
        self._prog_var.set(0)
        self._play_btn.state(["disabled"])
        self._frame_slider.state(["disabled"])
        self._export_btn.state(["disabled"])
        self._start_btn.state(["disabled"])
        self._cancel_btn.state(["!disabled"])
        self._set_status("PROCESSING", ACCENT)
        self._vid_canvas.delete("all")
        self._placeholder_id = self._vid_canvas.create_text(
            self._vid_canvas.winfo_width()//2 or 400,
            self._vid_canvas.winfo_height()//2 or 200,
            text="Processing…", fill=MUTED, font=("Segoe UI",12))

        params = dict(
            model_path      = mp,
            source_path     = sp,
            source_type     = st,
            conf_thresh     = self.conf_var.get(),
            line_x1         = self.line_x1.get(),
            line_y1         = self.line_y1.get(),
            line_x2         = self.line_x2.get(),
            line_y2         = self.line_y2.get(),
            classes         = chosen,
            max_disappeared = int(self.max_gone_var.get()),
        )
        self._proc_thread = threading.Thread(target=self._process_thread,
                                              args=(params,), daemon=True)
        self._proc_thread.start()
        self._poll_progress()

    def _cancel(self):
        self._cancel_flag = True
        self._set_status("CANCELLING…", RED)

    def _process_thread(self, p):
        """Runs in background thread — fills self.frames_data."""
        try:
            model_info = load_model(p["model_path"])
        except Exception as e:
            self._post(lambda: messagebox.showerror("Model error", str(e)))
            self._post(self._reset_buttons)
            return

        st   = p["source_type"]
        conf = p["conf_thresh"]
        lx1p,ly1p = p["line_x1"], p["line_y1"]
        lx2p,ly2p = p["line_x2"], p["line_y2"]
        classes   = p["classes"]
        max_gone  = p["max_disappeared"]

        tracker     = CentroidTracker(max_gone)
        prev_pos    = {}
        counted_ids = set()
        counts      = {c:0 for c in CLASSES}

        if st == "video":
            cap   = cv2.VideoCapture(p["source_path"])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9999
        else:
            exts = sum([glob.glob(os.path.join(p["source_path"],f"*{e}"))
                        for e in [".jpg",".jpeg",".png",".bmp",".webp",
                                  ".JPG",".JPEG",".PNG",".BMP",".WEBP"]], [])
            frame_files = sorted(set(exts))
            total = len(frame_files)
            cap   = None

        self.total_frames = total
        frame_idx = 0

        def get_lc(w, h):
            return (int(lx1p/100*w),int(ly1p/100*h),
                    int(lx2p/100*w),int(ly2p/100*h))

        while True:
            if self._cancel_flag:
                break
            if st == "video":
                ret, frame = cap.read()
                if not ret: break
            else:
                if frame_idx >= len(frame_files): break
                frame = cv2.imread(frame_files[frame_idx])
                if frame is None: frame_idx+=1; continue

            h,w = frame.shape[:2]
            lx1,ly1,lx2,ly2 = get_lc(w,h)

            dets = infer(model_info, frame, conf, classes)
            cents = [(int((x1+x2)//2), int((y1+y2)//2), cn)
                     for x1,y1,x2,y2,cn,_ in dets]
            active = tracker.update(cents)

            newly = []
            for oid,(cx,cy,cn) in active.items():
                if oid not in prev_pos:
                    prev_pos[oid]=(cx,cy); continue
                px,py = prev_pos[oid]
                prev_pos[oid]=(cx,cy)
                if oid in counted_ids: continue
                if side(px,py,lx1,ly1,lx2,ly2)*side(cx,cy,lx1,ly1,lx2,ly2)<0:
                    counts[cn]+=1; counted_ids.add(oid); newly.append((oid,cn))

            vis = draw_frame(frame,lx1,ly1,lx2,ly2,dets,active,newly,counts,frame_idx,total)
            self.frames_data.append(vis)

            # Update counts in UI every 5 frames
            if frame_idx % 5 == 0:
                snap = dict(counts)
                self._post(lambda s=snap: self._update_counts(s))

            self._prog_var.set(int((frame_idx+1)/max(total,1)*100))
            frame_idx+=1

        if cap: cap.release()

        final_counts = dict(counts)
        self._post(lambda: self._update_counts(final_counts))
        self._post(self._on_processing_done)

    def _poll_progress(self):
        """Check progress on the main thread."""
        if self._proc_thread and self._proc_thread.is_alive():
            pct = self._prog_var.get()
            done = len(self.frames_data)
            self._prog_lbl.config(text=f"Frame {done} / {self.total_frames}  ({pct:.0f}%)")
            self.after(250, self._poll_progress)
        else:
            self._prog_lbl.config(text="")

    def _on_processing_done(self):
        self.total_frames = len(self.frames_data)
        if self.total_frames == 0:
            self._set_status("NO FRAMES", RED)
            self._reset_buttons()
            return

        self._set_status("DONE", GREEN)
        self._reset_buttons()
        self._export_btn.state(["!disabled"])
        self._play_btn.state(["!disabled"])
        self._frame_slider.state(["!disabled"])
        self._frame_slider.config(to=max(0,self.total_frames-1))
        self._show_frame(0)
        self._prog_var.set(100)
        self._prog_lbl.config(text=f"Done — {self.total_frames} frames")

    def _reset_buttons(self):
        self._start_btn.state(["!disabled"])
        self._cancel_btn.state(["disabled"])

    def _post(self, fn):
        """Thread-safe call to main thread."""
        try: self.after(0, fn)
        except: pass

    # ── Viewer ───────────────────────────────────────────────────────────────
    def _show_frame(self, idx):
        if not self.frames_data: return
        idx = max(0, min(len(self.frames_data)-1, int(idx)))
        self.current_frame = idx
        self._frame_slider.set(idx)
        self._frame_lbl.config(text=f"{idx+1} / {self.total_frames}")

        frame_bgr = self.frames_data[idx]
        cw = self._vid_canvas.winfo_width()  or 640
        ch = self._vid_canvas.winfo_height() or 480
        fh,fw = frame_bgr.shape[:2]
        scale  = min(cw/fw, ch/fh)
        nw,nh  = int(fw*scale), int(fh*scale)
        disp   = cv2.resize(frame_bgr,(nw,nh), interpolation=cv2.INTER_AREA)
        rgb    = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img    = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self._vid_canvas.delete("all")
        self._vid_canvas.create_image(cw//2, ch//2, image=self._photo, anchor="center")

    def _on_slider(self, val):
        if self.playing: self._stop_play()
        self._show_frame(int(float(val)))

    def _on_canvas_resize(self, e):
        self._show_frame(self.current_frame)

    def _toggle_play(self):
        if self.playing: self._stop_play()
        else:            self._start_play()

    def _start_play(self):
        self.playing = True
        self._play_btn.config(text="⏸")
        self._schedule_play()

    def _stop_play(self):
        self.playing = False
        self._play_btn.config(text="▶")
        if self._play_after:
            self.after_cancel(self._play_after)
            self._play_after = None

    def _schedule_play(self):
        if not self.playing: return
        fps = max(1, int(self.fps_var.get()))
        delay = max(1, int(1000/fps))
        if self.current_frame >= self.total_frames-1:
            self._stop_play()
            return
        self._show_frame(self.current_frame+1)
        self._play_after = self.after(delay, self._schedule_play)

    # ── Export ───────────────────────────────────────────────────────────────
    def _export(self):
        if not self.frames_data:
            messagebox.showwarning("Empty","No frames to export."); return
        out = filedialog.asksaveasfilename(
            title="Save annotated video",
            defaultextension=".mp4",
            filetypes=[("MP4","*.mp4"),("All","*.*")])
        if not out: return

        fps = max(1, int(self.fps_var.get()))
        h,w = self.frames_data[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out, fourcc, fps, (w,h))
        for f in self.frames_data:
            writer.write(f)
        writer.release()
        messagebox.showinfo("Exported", f"Saved to:\n{out}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _update_counts(self, counts):
        for cn,val in counts.items():
            if cn in self._count_labels:
                self._count_labels[cn].config(text=str(val))

    def _set_status(self, text, color=MUTED):
        self._status_lbl.config(text=f"● {text}", fg=color)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = VehicleCounterApp()
    app.mainloop()
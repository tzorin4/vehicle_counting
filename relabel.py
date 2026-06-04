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
UA_DETRAC_MAP = {'bus': 3, 'car': 1, 'van': 2, 'others': 4}

# Auto-skip mappings based on instructions (Map Car and Van -> target Car)
# It leaves 2 (Bus) and 3 (Others) for manual GUI review
AUTO_MAP_RULES = {
    1: 0, # Previous 1 (Car) -> Target 0 (Car)
    2: 0, # van -> car
    3: 3, #bus -> bus

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




if __name__ == "__main__":
    root = tk.Tk()
    app = RelabelApp(root)
    root.mainloop()
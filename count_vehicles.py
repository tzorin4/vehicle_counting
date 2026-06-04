import os
import glob
import shutil
import random
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET

TARGET_CLASSES = {0: "Car", 1: "Motorcycle", 2: "Truck", 3: "Bus"}
PREVIOUS_CLASSES = {'car': 1, 'van': 2, 'bus': 3, 'others': 4}
PREVIOUS_CLASSES_REVERSED = {v: k for k, v in PREVIOUS_CLASSES.items()}

AUTO_MAP_RULES = {
    1: 0,
    2: 0,
    3: 3
}

class RelabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Relabel & Split Tool (YOLO & UA-Detrac)")
        self.root.geometry("1000x800")

        self.images_list = []
        self.current_img_idx = -1
        self.current_bboxes = []
        print("helo")




if __name__ == "__main__":
    root = tk.Tk()
    app = RelabelApp(root)
    root.mainloop()
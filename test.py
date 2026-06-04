import os
import re
import glob
from typing import List, Tuple

def natural_sort_key(path: str) -> List: # works
    basename = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', basename)]

def load_img_and_lbl(image_dir:str, label_dir:str) -> List[Tuple[str, str]]:
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    image_paths.sort(key=natural_sort_key)

    img_lbl_pairs = []
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(label_dir, base_name + ".txt")
        if os.path.exists(lbl_path):
            img_lbl_pairs.append((img_path, lbl_path))
    return img_lbl_pairs

def find_and_parse_labels_2(lbl_path:str) -> List[Tuple[int, int, int, int, int]]:
    bboxes = []
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            bboxes.append((int(class_id), x_center, y_center, width, height))
    return bboxes

def find_and_parse_labels(lbl_path:str):
    boxes = []
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            boxes.append({
                'id': int(class_id),
                'bbox': [x_center, y_center, width, height],
                'needs_review': True
            })
    return boxes

def load_img_test():
    img_dir = r"archive/content/UA-DETRAC\DETRAC_Upload/images/train"
    lbl_dir = r"archive/content/UA-DETRAC\DETRAC_Upload/labels/train"
    list_of_pairs = load_img_and_lbl(img_dir, lbl_dir)
    for img_path, lbl_path in list_of_pairs[:10]:
        print(f"Image: {img_path} | Label: {lbl_path}")


def find_and_parse_labels_test():
    path = r"archive/content/UA-DETRAC/DETRAC_Upload/labels/train/MVI_20065_img00788.txt"
    print(find_and_parse_labels(path))

if __name__ == "__main__":
    find_and_parse_labels_test()


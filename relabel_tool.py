import cv2
import os
import glob

# --- CONFIGURATION ---
# Change these to point to your dataset folders
IMAGES_DIR = "images"
LABELS_DIR = "labels"

# Your target YOLO classes mapped to keyboard keys
# 0: Car, 1: Motorcycle, 2: Truck, 3: Bus
# We will use keys: 'c' for car, 'm' for motorcycle, 't' for truck, 'b' for bus
KEY_MAP = {
    ord('c'): 0, # Car
    ord('m'): 1, # Motorcycle
    ord('t'): 2, # Truck
    ord('b'): 3, # Bus
}

CLASS_NAMES = {
    0: "Car",
    1: "Motorcycle",
    2: "Truck",
    3: "Bus"
}

def get_yolo_box(img_shape, yolo_bbox):
    """Convert YOLO format (x_center, y_center, w, h) to pixel coordinates (x1, y1, x2, y2)"""
    h, w, _ = img_shape
    x_c, y_c, bw, bh = map(float, yolo_bbox)
    x1 = int((x_c - bw / 2) * w)
    y1 = int((y_c - bh / 2) * h)
    x2 = int((x_c + bw / 2) * w)
    y2 = int((y_c + bh / 2) * h)
    return (x1, y1), (x2, y2)

def main():
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(LABELS_DIR):
        print(f"Please create '{IMAGES_DIR}' and '{LABELS_DIR}' folders and put your data there.")
        return

    image_paths = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    
    print("\n--- YOLO RELABELING TOOL ---")
    print("Controls for highlighted object:")
    print(" 'c' - Car")
    print(" 'm' - Motorcycle")
    print(" 't' - Truck")
    print(" 'b' - Bus")
    print(" 'delete' or 'x' - Remove bounding box")
    print(" 's' or 'space' - Skip/Keep current label")
    print(" 'q' - Quit tool")
    print("----------------------------\n")

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS_DIR, base_name + ".txt")

        if not os.path.exists(label_path):
            continue # Skip images without labels

        with open(label_path, "r") as f:
            lines = f.readlines()

        if not lines:
            continue

        img = cv2.imread(img_path)
        new_lines = []
        modified = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            cls_id = int(parts[0])
            bbox = parts[1:]

            display_img = img.copy()
            pt1, pt2 = get_yolo_box(display_img.shape, bbox)
            
            # Draw the current box in bright Red to highlight it
            cv2.rectangle(display_img, pt1, pt2, (0, 0, 255), 3)
            current_label = CLASS_NAMES.get(cls_id, f"Unknown ({cls_id})")
            cv2.putText(display_img, f"? Current: {current_label}", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Fast Relabel Tool", display_img)
            
            key = cv2.waitKey(0)

            if key == ord('q'):
                print("Quitting...")
                cv2.destroyAllWindows()
                return
            elif key in KEY_MAP:
                new_cls = KEY_MAP[key]
                new_lines.append(f"{new_cls} " + " ".join(bbox) + "\n")
                modified = True
                print(f"Changed to {CLASS_NAMES[new_cls]}")
            elif key == ord('x') or key == 255 or key == 8: # 'x', Delete, Backspace
                print("Box deleted.")
                modified = True
                continue
            else:
                # Keep original (space or 's')
                new_lines.append(line)
        
        # Save overwritten file if modified
        if modified:
            with open(label_path, "w") as f:
                f.writelines(new_lines)

    cv2.destroyAllWindows()
    print("Finished checking all files!")

if __name__ == "__main__":
    main()
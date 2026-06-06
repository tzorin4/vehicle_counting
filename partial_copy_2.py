from test import load_img_and_lbl
import os
import random
import shutil

if __name__ == "__main__":
    img_dir = r"archive/content/UA-DETRAC\DETRAC_Upload/images/train"
    lbl_dir = r"archive/content/UA-DETRAC\DETRAC_Upload/labels/train"
    number_of_pairs_to_copy = 1000
    list_of_pairs = load_img_and_lbl(img_dir, lbl_dir)
    train_split = 0.8
    output_dir = r"dataset_output_2"
    
    for img_path, lbl_path in list_of_pairs[:number_of_pairs_to_copy]:
        split = 'train' if random.random() <= train_split else 'val'
        base_name = os.path.basename(img_path)
        dest_img_path = os.path.join(output_dir, split, 'images', base_name)
        dest_lbl_path = os.path.join(output_dir, split, 'labels', os.path.splitext(base_name)[0] + '.txt')
        os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(dest_lbl_path), exist_ok=True)
        shutil.copy2(img_path, dest_img_path)
        shutil.copy2(lbl_path, dest_lbl_path)
    print(f"Copied {number_of_pairs_to_copy} pairs to {output_dir} with train/val split of {train_split*100:.0f}/{(1-train_split)*100:.0f}.")
from test import load_img_and_lbl
import os
import random
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy a subset of image-label pairs with train/val split")
    parser.add_argument("--img-dir","-i", default=r"archive/content/UA-DETRAC\DETRAC_Upload/images/train", help="Directory containing images")
    parser.add_argument("--lbl-dir", "-l", default=r"archive/content/UA-DETRAC\DETRAC_Upload/labels/train", help="Directory containing label files")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for copied dataset")
    parser.add_argument("--num-pairs", "-n", type=int, default=1000, help="Number of image-label pairs to copy")
    parser.add_argument("--train-split", "-t", type=float, default=0.8, help="Proportion of data to use for training (default: 0.8)")
    args = parser.parse_args()
    img_dir = args.img_dir
    lbl_dir = args.lbl_dir
    number_of_pairs_to_copy = min(args.num_pairs, len(load_img_and_lbl(img_dir, lbl_dir)))
    list_of_pairs = load_img_and_lbl(img_dir, lbl_dir)
    train_split = args.train_split
    output_dir = args.output_dir
    
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
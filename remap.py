import argparse
import glob
import os


# =====================================================
# REMAP RULES
#
# key = original class id
# value = new class id
#
# Classes not listed here remain unchanged.
# =====================================================

CLASS_MAP = {
    1: 0,  # car -> car
    2: 50,
    3: 100

}


def remap_label_file(input_file, output_file):
    lines_out = []

    with open(input_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))

            # remap if rule exists, otherwise keep original
            class_id = CLASS_MAP.get(class_id, class_id)

            parts[0] = str(class_id)
            lines_out.append(" ".join(parts))

    with open(output_file, "w") as f:
        f.write("\n".join(lines_out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", "-l", required=True,
                        help="Directory containing YOLO label txt files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(args.labels, "*.txt")))

    for label_file in label_files:
        out_file = os.path.join(
            args.output,
            os.path.basename(label_file)
        )

        remap_label_file(label_file, out_file)

    print(f"Processed {len(label_files)} label files")


if __name__ == "__main__":
    main()
import os
import shutil
from PIL import Image

def clean_mismatched_images(img_dir="img", label_dir="labels", out_dir="bad_images"):
    # Ensure the output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Filter for .jpeg files in the img directory
    images = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpeg')]

    mismatch_count = 0

    for img_name in images:
        # Construct paths
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, f"{base_name}.png")

        # Check if the label counterpart exists
        if os.path.exists(label_path):
            try:
                with Image.open(img_path) as img, Image.open(label_path) as lbl:
                    if img.size != lbl.size:
                        print(f"Mismatch: {base_name} (Img: {img.size} != Lbl: {lbl.size})")

                        # Move both files
                        shutil.move(img_path, os.path.join(out_dir, img_name))
                        shutil.move(label_path, os.path.join(out_dir, f"{base_name}.png"))
                        mismatch_count += 1
            except Exception as e:
                print(f"Error processing {base_name}: {e}")
        else:
            print(f"Warning: No label found for {img_name}")

    print(f"\nTask complete. Moved {mismatch_count} pairs to '{out_dir}'.")

if __name__ == "__main__":
    clean_mismatched_images()

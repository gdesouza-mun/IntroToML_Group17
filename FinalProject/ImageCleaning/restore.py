import os
import shutil
from PIL import Image

def fix_and_restore(bad_dir="bad_images", img_dest="img", label_dest="labels"):
    # Ensure destination directories exist
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(label_dest, exist_ok=True)

    # List all .jpeg files in the bad_images folder
    bad_files = [f for f in os.listdir(bad_dir) if f.lower().endswith('.jpeg')]

    if not bad_files:
        print("No .jpeg files found in bad_images folder.")
        return

    fixed_count = 0

    for img_name in bad_files:
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.png"

        img_path = os.path.join(bad_dir, img_name)
        label_path = os.path.join(bad_dir, label_name)

        # Ensure the label actually exists in bad_images
        if not os.path.exists(label_path):
            print(f"Skipping {img_name}: No corresponding .png found in {bad_dir}.")
            continue

        try:
            with Image.open(img_path) as img, Image.open(label_path) as lbl:
                img_w, img_h = img.size
                lbl_w, lbl_h = lbl.size

                # Calculate cropping box for the label
                # left, top, right, bottom
                left = (lbl_w - img_w) / 2
                top = (lbl_h - img_h) / 2
                right = (lbl_w + img_w) / 2
                bottom = (lbl_h + img_h) / 2

                # Perform the crop
                cropped_lbl = lbl.crop((left, top, right, bottom))

                # Save the corrected label to the final destination
                cropped_lbl.save(os.path.join(label_dest, label_name))

                # Move the original image to the final destination
                shutil.move(img_path, os.path.join(img_dest, img_name))

                # Clean up: delete the old oversized label from bad_images
                os.remove(label_path)

                fixed_count += 1
                print(f"Fixed and restored: {base_name} ({lbl_w}x{lbl_h} -> {img_w}x{img_h})")

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    print(f"\nFinished! Successfully restored {fixed_count} pairs.")

if __name__ == "__main__":
    fix_and_restore()

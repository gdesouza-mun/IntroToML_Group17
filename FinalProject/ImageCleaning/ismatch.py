import os
from PIL import Image

def run_sanity_check(img_dir="img", label_dir="labels"):
    # Get all .jpeg images
    images = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpeg')]

    total = len(images)
    matches = 0
    mismatches = 0
    missing = 0

    print(f"--- Starting Sanity Check on {total} pairs ---")

    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.png"

        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, label_name)

        # 1. Check if label exists
        if not os.path.exists(label_path):
            print(f"❌ MISSING LABEL: {label_name}")
            missing += 1
            continue

        # 2. Check Dimensions
        try:
            with Image.open(img_path) as img, Image.open(label_path) as lbl:
                if img.size == lbl.size:
                    matches += 1
                else:
                    print(f"⚠️ SIZE MISMATCH: {base_name} | Img: {img.size} vs Lbl: {lbl.size}")
                    mismatches += 1
        except Exception as e:
            print(f"🔥 CORRUPT FILE: {img_name} - {e}")

    # Final Report
    print("\n--- Final Report ---")
    print(f"✅ Identical Sizes: {matches}")
    print(f"⚠️ Mismatched Sizes: {mismatches}")
    print(f"❌ Missing Labels:   {missing}")

    if mismatches == 0 and missing == 0:
        print("\n🎉 All clear! Your dataset is synchronized.")
    else:
        print("\n❗ Issues found. Please review the log above.")

if __name__ == "__main__":
    run_sanity_check()

import os
import shutil

# ================= USER SETTINGS =================
BASE_DIR = "../images_new"
IMAGE_EXTS = ".png"
# =================================================


def get_images(folder):
    if not os.path.exists(folder):
        return set()
    return {f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)}


for date in sorted(os.listdir(BASE_DIR)):
    date_path = os.path.join(BASE_DIR, date)
    if not os.path.isdir(date_path):
        continue

    sat_dir = os.path.join(date_path, "sat")
    rad_dir = os.path.join(date_path, "rad")

    if not os.path.isdir(sat_dir) or not os.path.isdir(rad_dir):
        print(f"{date}: missing sat or rad folder — skipped")
        continue

    sat_images = get_images(sat_dir)
    rad_images = get_images(rad_dir)

    root_images = get_images(date_path)

    expected_images = sat_images | rad_images
    missing_in_root = expected_images - root_images

    # Copy only what is missing in the root folder
    copied = 0
    for img in missing_in_root:
        src = (
            os.path.join(sat_dir, img)
            if img in sat_images
            else os.path.join(rad_dir, img)
        )
        dst = os.path.join(date_path, img)
        shutil.copy2(src, dst)
        copied += 1

    total_root = len(get_images(date_path))

    if copied == 0:
        print(f"{date}: OK ({total_root} images in root)")
    else:
        print(f"{date}: FIXED (copied {copied}, root={total_root})")

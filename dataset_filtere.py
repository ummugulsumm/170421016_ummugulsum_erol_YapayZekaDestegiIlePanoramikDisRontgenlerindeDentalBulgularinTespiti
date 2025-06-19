import os
import shutil
from collections import defaultdict
from PIL import Image


BASE_DIR = os.getenv('BASE_DIR')
CLAHE_SUBDIR = os.getenv('CLAHE_SUBDIR')
YAML_FILENAME = os.getenv('YAML_FILENAME')
FILTERED_SUBDIR = os.getenv('FILTERED_SUBDIR')

CLAHE_DIR = os.path.join(BASE_DIR, CLAHE_SUBDIR)
YAML_PATH = os.path.join(CLAHE_DIR, YAML_FILENAME)

FILTERED_DIR = os.path.join(BASE_DIR, FILTERED_SUBDIR)

# Hedef boyutlar
TARGET_WIDTH = 1615
TARGET_HEIGHT = 840

splits = ["train", "valid", "test"]

for split in splits:
    split_img_dir = os.path.join(FILTERED_DIR, split, "images")
    split_lbl_dir = os.path.join(FILTERED_DIR, split, "labels")
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)

    images_dir = os.path.join(CLAHE_DIR, split, "images")
    labels_dir = os.path.join(CLAHE_DIR, split, "labels")

    # Görselleri hasta bazlı grupla
    patient_images = defaultdict(list)
    for file in os.listdir(images_dir):
        if file.endswith((".jpg", ".png")):
            patient_id = file.split("-")[0]
            patient_images[patient_id].append(os.path.join(images_dir, file))

    print(f"[{split.upper()}] Toplam hasta sayısı: {len(patient_images)}")

    selected_images = []
    for files in patient_images.values():
        selected = None
        fallback = None
        for img_path in files:
            with Image.open(img_path) as im:
                if im.size == (TARGET_WIDTH, TARGET_HEIGHT):
                    selected = img_path
                    break
                if fallback is None:
                    fallback = img_path
        selected_images.append(selected if selected else fallback)

    print(f"[{split.upper()}] Seçilen görseller: {len(selected_images)}")

    saved_count = 0
    for img_path in selected_images:
        file_name = os.path.basename(img_path)
        label_file = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue

        shutil.copy2(img_path, os.path.join(split_img_dir, file_name))
        shutil.copy2(label_path, os.path.join(split_lbl_dir, label_file))
        saved_count += 1

    print(f"[{split.upper()}] Kopyalanan görsel ve etiket sayısı: {saved_count}")
    print("-" * 50)

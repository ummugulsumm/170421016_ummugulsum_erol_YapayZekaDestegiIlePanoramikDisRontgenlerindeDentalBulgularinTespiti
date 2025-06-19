import os
import cv2
from tqdm import tqdm
import yaml
import shutil
from glob import glob
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv('BASE_DIR')
YOLO_SUBDIR = os.getenv('YOLO_SUBDIR')
CLAHE_SUBDIR = os.getenv('CLAHE_SUBDIR')
YAML_FILENAME = os.getenv('YAML_FILENAME')

YOLO_DIR = os.path.join(BASE_DIR, YOLO_SUBDIR)
CLAHE_DIR = os.path.join(BASE_DIR, CLAHE_SUBDIR)
YAML_PATH = os.path.join(CLAHE_DIR, YAML_FILENAME)

class_map = {
    2: 0,
    3: 1,
    9: 2,
    11: 3,
}


NEW_CLASS_NAMES = {
    0: 'Filling',
    1: 'Implant',
    2: 'Root Canal Treatment',
    3: 'Impacted tooth',

}


def remap_class_indices():
    """Etiket dosyalarındaki sınıf indekslerini yeniden eşler.
    Geçersiz hale gelen (hiç etiketi kalmayan) görüntüleri de siler.
    """
    label_dirs = [
        os.path.join(YOLO_DIR, 'train', 'labels'),
        os.path.join(YOLO_DIR, 'valid', 'labels'),
        os.path.join(YOLO_DIR, 'test', 'labels')
    ]

    total_processed = 0
    total_deleted = 0

    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            print(f"Uyarı: Label klasörü bulunamadı: {label_dir}")
            continue

        split = label_dir.split(os.sep)[-2]  # 'train', 'valid', ya da 'test'
        image_dir = os.path.join(YOLO_DIR, split, 'images')

        files_to_delete = []
        processed_files = 0

        for label_file in glob(os.path.join(label_dir, '*.txt')):
            new_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    if class_id in class_map:
                        parts[0] = str(class_map[class_id])
                        new_lines.append(' '.join(parts))

            if not new_lines:
                files_to_delete.append(label_file)
                base_filename = os.path.splitext(os.path.basename(label_file))[0]
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_path = os.path.join(image_dir, base_filename + ext)
                    if os.path.exists(image_path):
                        files_to_delete.append(image_path)
                        break
            else:
                with open(label_file, 'w') as f:
                    for l in new_lines:
                        f.write(l + '\n')
                processed_files += 1

        for f_path in files_to_delete:
            if os.path.exists(f_path):
                os.remove(f_path)
                print(f"Silindi: {f_path}")
                total_deleted += 1

        total_processed += processed_files
        print(f"{split}: {processed_files} dosya işlendi, {len(files_to_delete)} dosya silindi")

    print(f"Toplam: {total_processed} dosya işlendi, {total_deleted} dosya silindi")
    print("Sınıf indeksleri yeniden eşlendi ve geçersiz görüntüler kaldırıldı.")


def apply_clahe(img_path, output_path):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) uygular"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Görüntü yüklenemedi: {img_path}")
            return False

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)

        hsv_clahe = cv2.merge([h, s, v_clahe])
        img_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_clahe)
        return True
    except Exception as e:
        print(f"CLAHE uygulama hatası {img_path}: {str(e)}")
        return False


def create_clahe_dataset():

    total_processed = 0
    total_copied = 0

    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(CLAHE_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(CLAHE_DIR, split, "labels"), exist_ok=True)

        img_src_dir = os.path.join(YOLO_DIR, split, "images")
        label_src_dir = os.path.join(YOLO_DIR, split, "labels")

        new_img_dst_dir = os.path.join(CLAHE_DIR, split, "images")
        new_label_dst_dir = os.path.join(CLAHE_DIR, split, "labels")

        if not os.path.exists(label_src_dir):
            print(f"Uyarı: Label klasörü bulunamadı: {label_src_dir}")
            continue

        label_files = [f for f in os.listdir(label_src_dir) if f.endswith('.txt')]
        processed_images_count = 0
        copied_labels_count = 0

        for label_file in tqdm(label_files, desc=f"CLAHE uygulama ve etiket kopyalama {split}"):
            base_filename = os.path.splitext(label_file)[0]

            image_found = False
            for ext in ['.jpg', '.jpeg', '.png']:
                img_src_path = os.path.join(img_src_dir, base_filename + ext)
                if os.path.exists(img_src_path):
                    img_dst_path = os.path.join(new_img_dst_dir, base_filename + ext)
                    if apply_clahe(img_src_path, img_dst_path):
                        processed_images_count += 1
                    else:
                        shutil.copy2(img_src_path, img_dst_path)
                        processed_images_count += 1
                    image_found = True
                    break

            if image_found:
                label_src_path = os.path.join(label_src_dir, label_file)
                label_dst_path = os.path.join(new_label_dst_dir, label_file)
                shutil.copy2(label_src_path, label_dst_path)
                copied_labels_count += 1

        print(f"CLAHE uygulanan {split} görüntüsü: {processed_images_count}")
        print(f"Kopyalanan {split} etiketi: {copied_labels_count}")

        total_processed += processed_images_count
        total_copied += copied_labels_count

    create_new_yaml()

    print(f"CLAHE optimize edilmiş veri seti oluşturuldu: {CLAHE_DIR}")
    print(f"Toplam işlenen görüntü: {total_processed}")
    print(f"Toplam kopyalanan etiket: {total_copied}")

    return YAML_PATH


def create_new_yaml():
    """Yeni data.yaml dosyasını oluştur"""
    print("Yeni data.yaml dosyası oluşturuluyor...")

    yaml_content = {
        'train': os.path.join(CLAHE_DIR, "train", "images"),
        'val': os.path.join(CLAHE_DIR, "valid", "images"),
        'test': os.path.join(CLAHE_DIR, "test", "images"),
        'nc': len(NEW_CLASS_NAMES),
        'names': NEW_CLASS_NAMES
    }

    try:
        os.makedirs(os.path.dirname(YAML_PATH), exist_ok=True)
        with open(YAML_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"Yeni data.yaml dosyası oluşturuldu: {YAML_PATH}")

        print("\n data.yaml içeriği:")
        with open(YAML_PATH, 'r', encoding='utf-8') as f:
            print(f.read())

    except Exception as e:
        print(f"data.yaml oluşturma hatası: {str(e)}")

        try:
            with open(YAML_PATH, 'w') as f:
                f.write(f"train: {os.path.join(CLAHE_DIR, 'train', 'images')}\n")
                f.write(f"val: {os.path.join(CLAHE_DIR, 'valid', 'images')}\n")
                f.write(f"test: {os.path.join(CLAHE_DIR, 'test', 'images')}\n")
                f.write(f"nc: {len(NEW_CLASS_NAMES)}\n")
                f.write("names:\n")
                for idx, name in NEW_CLASS_NAMES.items():
                    f.write(f"  {idx}: {name}\n")
            print("Basit YAML formatında oluşturuldu")
        except Exception as e2:
            print(f"Basit YAML oluşturma da başarısız: {str(e2)}")


def check_environment():
    print("Çevre değişkenleri kontrol ediliyor...")

    required_vars = ['BASE_DIR', 'YOLO_SUBDIR', 'CLAHE_SUBDIR', 'YAML_FILENAME']
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            print(f"{var}: {value}")

    if missing_vars:
        print(f"Eksik çevre değişkenleri: {missing_vars}")
        return False

    print(f"\nDizin kontrolleri:")
    print(f"YOLO_DIR: {YOLO_DIR}")
    print(f"CLAHE_DIR: {CLAHE_DIR}")
    print(f"YAML_PATH: {YAML_PATH}")

    if not os.path.exists(YOLO_DIR):
        print(f"YOLO dizini bulunamadı: {YOLO_DIR}")
        return False
    else:
        print(f"YOLO dizini mevcut")

    return True


def main():

    print("Dataset İşleme Başlatılıyor...")
    print("=" * 50)

    if not check_environment():
        print("Çevre değişkenleri eksik veya dizinler bulunamadı!")
        return

    try:
        print("\n1️Sınıf indeksleri yeniden eşleniyor...")
        remap_class_indices()

        print("\n2️CLAHE veri seti oluşturuluyor...")
        clahe_yaml_path = create_clahe_dataset()

        print(f"\nİşlem tamamlandı!")
        print(f"Yeni veri seti: {CLAHE_DIR}")
        print(f"YAML dosyası: {clahe_yaml_path}")

    except Exception as e:
        print(f"\nHata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
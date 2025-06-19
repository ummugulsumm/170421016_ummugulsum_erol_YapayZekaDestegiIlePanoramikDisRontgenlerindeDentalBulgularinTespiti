import os
import yaml
import torch
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv('BASE_DIR')
CLAHE_SUBDIR = os.getenv('CLAHE_SUBDIR')
YAML_FILENAME = os.getenv('YAML_FILENAME')

CLAHE_DIR = os.path.join(BASE_DIR, CLAHE_SUBDIR)
YAML_PATH = os.path.join(CLAHE_DIR, YAML_FILENAME)

def check_gpu():
    if torch.cuda.is_available():
        device = 0
        print(f"GPU bulundu: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("GPU bulunamadı, CPU kullanılacak.")
    return device


def train_model(yaml_path, model_type="yolo11m-seg.pt", epochs=200):
    device = check_gpu()

    print(f"\nModel eğitimi başlatılıyor: {model_type}")
    print(f"Veri seti: {yaml_path}")
    print(f"Cihaz: {'GPU' if device == 0 else 'CPU'}")

    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        print(f"Sınıf sayısı: {yaml_content.get('nc', 'Bilinmiyor')}")
        print(f"Sınıflar: {yaml_content.get('names', 'Bilinmiyor')}")
    else:
        print(f"Uyarı: YAML dosyası bulunamadı: {yaml_path}")
        return None

    model = YOLO(model_type)

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=1024,
        batch=4,
        device=0,
        patience=75,
        optimizer="Adam",  # Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto,
        lr0=0.001,
        val=True,
        project=os.path.join(BASE_DIR, "runs_yeni"),
        name=f"train_{model_type.split('.')[0]}_4class_{epochs}_epochs_imgsize_1024_Adam_0.001_lr",
        resume=False,
    )

    best_pt_path = os.path.join(BASE_DIR, "runs_yeni", f"train_{model_type.split('.')[0]}_4class_{epochs}_epochs_imgsize_1024_Adam_0.001_lr", "weights", "best.pt")
    if os.path.exists(best_pt_path):
        print(f"\nEğitim tamamlandı!")
        print(f"En iyi model: {best_pt_path}")
    else:
        print("\nEğitim tamamlandı ancak best.pt dosyası bulunamadı!")

    return best_pt_path if os.path.exists(best_pt_path) else None


def evaluate_model(model_path, yaml_path):
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı: {model_path}")
        return None

    device = check_gpu()
    print(f"\nModel değerlendiriliyor: {os.path.basename(model_path)}")

    model = YOLO(model_path)
    results = model.val(data=yaml_path, device=device)

    print("\nDeğerlendirme sonuçları:")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"mAP50: {results.box.map50:.4f}")

    return results

def main():
    print("\n===== DENTAL NESNE TESPİTİ MODEL EĞİTİMİ =====\n")

    best_model_path = train_model(YAML_PATH)

    if best_model_path:
        evaluate_model(best_model_path, YAML_PATH)

    print("\n===== İŞLEM TAMAMLANDI =====")


if __name__ == "__main__":
    main()
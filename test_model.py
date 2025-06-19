import os
from ultralytics import YOLO
from datetime import datetime
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv('BASE_DIR')
CLAHE_SUBDIR = os.getenv('CLAHE_SUBDIR')
YAML_FILENAME = os.getenv('YAML_FILENAME')
RUNS_SUBDIR = os.getenv('RUNS_SUBDIR')
MODEL_SUBDIR = os.getenv('MODEL_SUBDIR')

CLAHE_DIR = os.path.join(BASE_DIR, CLAHE_SUBDIR)
YAML_PATH = os.path.join(CLAHE_DIR, YAML_FILENAME)
BEST_MODEL_PATH = os.path.join(BASE_DIR, RUNS_SUBDIR,MODEL_SUBDIR ,"weights", "best.pt")

def main():
    best_model_path = BEST_MODEL_PATH
    if best_model_path:
        print("\nDetaylı model değerlendirmesi başlatılıyor...")
        evaluation_results = evaluate_model_detailed(best_model_path, YAML_PATH)
        if evaluation_results:
            save_evaluation_results(evaluation_results, best_model_path)
            print("Model değerlendirme tamamlandı ve dosyalara kaydedildi.")
        else:
            print("Model değerlendirme başarısız oldu.")


def evaluate_model_detailed(model_path, yaml_path):
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı: {model_path}")
        return None

    print(f"\nDetaylı model değerlendirmesi: {os.path.basename(model_path)}")
    model = YOLO(model_path)

    conf_thresholds = [0.1, 0.25, 0.5, 0.7]
    results_summary = []

    for conf_thresh in conf_thresholds:
        print(f"\n--- Confidence Threshold: {conf_thresh} ---")
        results = model.val(
            data=yaml_path,
            device=0,
            imgsz=640,
            batch=1,
            conf=conf_thresh,
            iou=0.5,
            plots=False,
        )

        result_data = {
            'conf': conf_thresh,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if hasattr(results, 'box') and results.box is not None:
            print(f"Box Detection (conf={conf_thresh}):")
            print(f"  mAP50-95: {results.box.map:.4f}")
            print(f"  mAP50:    {results.box.map50:.4f}")
            result_data['box_map'] = float(results.box.map)
            result_data['box_map50'] = float(results.box.map50)

            if hasattr(results.box, 'mp') and results.box.mp is not None:
                result_data['box_precision'] = float(results.box.mp)
            if hasattr(results.box, 'mr') and results.box.mr is not None:
                result_data['box_recall'] = float(results.box.mr)

        if hasattr(results, 'seg') and results.seg is not None:
            print(f"Segmentation (conf={conf_thresh}):")
            print(f"  mAP50-95: {results.seg.map:.4f}")
            print(f"  mAP50:    {results.seg.map50:.4f}")
            result_data['seg_map'] = float(results.seg.map)
            result_data['seg_map50'] = float(results.seg.map50)

            if hasattr(results.seg, 'mp') and results.seg.mp is not None:
                result_data['seg_precision'] = float(results.seg.mp)
            if hasattr(results.seg, 'mr') and results.seg.mr is not None:
                result_data['seg_recall'] = float(results.seg.mr)

        results_summary.append(result_data)

    return results_summary


def save_evaluation_results(results_summary, model_path):

    results_dir = os.path.join(BASE_DIR, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    json_file = os.path.join(results_dir, f"{model_name}_evaluation_{timestamp}.json")
    evaluation_data = {
        'model_path': model_path,
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'results': results_summary
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    print(f"JSON sonuçları kaydedildi: {json_file}")

    try:
        df = pd.DataFrame(results_summary)
        csv_file = os.path.join(results_dir, f"{model_name}_evaluation_{timestamp}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"CSV sonuçları kaydedildi: {csv_file}")
    except Exception as e:
        print(f"CSV kaydetme hatası: {e}")

    txt_file = os.path.join(results_dir, f"{model_name}_evaluation_report_{timestamp}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("DENTAL MODEL DEĞERLENDIRME RAPORU\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Değerlendirme Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {YAML_PATH}\n\n")

        f.write("SONUÇLAR:\n")
        f.write("-" * 40 + "\n\n")

        for result in results_summary:
            f.write(f"Confidence Threshold: {result['conf']}\n")
            f.write("-" * 25 + "\n")

            if 'box_map' in result:
                f.write("Box Detection:\n")
                f.write(f"  mAP50-95: {result['box_map']:.4f}\n")
                f.write(f"  mAP50:    {result['box_map50']:.4f}\n")
                if 'box_precision' in result:
                    f.write(f"  Precision: {result['box_precision']:.4f}\n")
                if 'box_recall' in result:
                    f.write(f"  Recall:    {result['box_recall']:.4f}\n")
                f.write("\n")

            if 'seg_map' in result:
                f.write("Segmentation:\n")
                f.write(f"  mAP50-95: {result['seg_map']:.4f}\n")
                f.write(f"  mAP50:    {result['seg_map50']:.4f}\n")
                if 'seg_precision' in result:
                    f.write(f"  Precision: {result['seg_precision']:.4f}\n")
                if 'seg_recall' in result:
                    f.write(f"  Recall:    {result['seg_recall']:.4f}\n")
                f.write("\n")

            f.write("\n")

        if results_summary:
            f.write("EN İYİ SONUÇLAR:\n")
            f.write("-" * 20 + "\n")

            if 'box_map50' in results_summary[0]:
                best_box = max(results_summary, key=lambda x: x.get('box_map50', 0))
                f.write(f"En İyi Box mAP50: {best_box['box_map50']:.4f} (conf={best_box['conf']})\n")

            if 'seg_map50' in results_summary[0]:
                best_seg = max(results_summary, key=lambda x: x.get('seg_map50', 0))
                f.write(f"En İyi Seg mAP50: {best_seg['seg_map50']:.4f} (conf={best_seg['conf']})\n")

    print(f"Detaylı rapor kaydedildi: {txt_file}")


if __name__ == "__main__":
    main()
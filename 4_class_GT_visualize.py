import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from dotenv import load_dotenv

load_dotenv()

class YOLOSegmentationTester:
    def __init__(self, model_path, test_path, class_names, label_to_model_mapping=None, conf_threshold=0.1,
                 iou_threshold=0.1):
        self.model = YOLO(model_path)
        self.test_path = test_path
        self.class_names = class_names
        self.label_to_model_mapping = label_to_model_mapping or {}
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Test klasör yolları
        self.images_path = os.path.join(test_path, 'images')
        self.labels_path = os.path.join(test_path, 'labels')

        self.all_true_labels = []
        self.all_pred_labels = []
        self.all_true_masks = []
        self.all_pred_masks = []

        self.detailed_results = defaultdict(lambda: {
            'tp': 0, 'fp': 0, 'fn': 0,
            'pred_masks': [], 'true_masks': []
        })

        # Görselleştirme için
        self.visualization_data = []

        # Renk paleti (her sınıf için farklı renk)
        self.colors = {
            0: [255, 0, 0],  # Kırmızı - Caries
            1: [0, 255, 0],  # Yeşil - Filling
            2: [0, 0, 255],  # Mavi - Implant
            3: [255, 255, 0],  # Sarı - Root Canal Treatment
            4: [255, 0, 255],  # Magenta - Impacted tooth
        }

    def polygon_to_mask(self, polygon_coords, img_height, img_width):
        if len(polygon_coords) < 6:  # En az 3 nokta gerekli
            return np.zeros((img_height, img_width), dtype=np.uint8)

        # Koordinatları reshape et
        points = np.array(polygon_coords).reshape(-1, 2)
        # Normalize edilmiş koordinatları pixel koordinatlarına çevir
        points[:, 0] *= img_width
        points[:, 1] *= img_height
        points = points.astype(np.int32)

        # Mask oluştur
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        return mask

    def calculate_mask_iou(self, mask1, mask2):
        if mask1.shape != mask2.shape:
            return 0.0

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def load_ground_truth(self, label_file, img_height, img_width):
        gt_data = []

        if not os.path.exists(label_file):
            return gt_data

        with open(label_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 6:  # En az class_id + 3 nokta
                    continue

                label_class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                # Label sınıf ID'sini model sınıf ID'sine çevir
                if self.label_to_model_mapping and label_class_id in self.label_to_model_mapping:
                    model_class_id = self.label_to_model_mapping[label_class_id]
                else:
                    model_class_id = label_class_id

                # Sadece modelimizde tanımlı sınıfları al
                if model_class_id not in self.class_names:
                    continue

                # Polygon'u mask'e çevir
                mask = self.polygon_to_mask(coords, img_height, img_width)

                gt_data.append({
                    'class_id': model_class_id,
                    'mask': mask,
                    'coords': coords,
                    'original_label_class': label_class_id
                })

        return gt_data

    def process_predictions(self, results, img_height, img_width):
        """Model tahminlerini işle"""
        pred_data = []

        if results[0].masks is None:
            return pred_data

        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes

        if boxes is None:
            return pred_data

        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for i, (mask, conf, class_id) in enumerate(zip(masks, confidences, class_ids)):
            if conf >= self.conf_threshold:
                # Mask'i orijinal boyuta getir
                mask_resized = cv2.resize(mask, (img_width, img_height))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                pred_data.append({
                    'class_id': class_id,
                    'confidence': conf,
                    'mask': mask_binary
                })

        return pred_data

    def match_predictions_with_ground_truth(self, gt_data, pred_data):
        """Tahminleri ground truth ile eşleştir"""
        matches = []
        used_gt = set()
        used_pred = set()

        # Her tahmin için en iyi ground truth'u bul
        for pred_idx, pred in enumerate(pred_data):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_data):
                if gt_idx in used_gt:
                    continue

                if pred['class_id'] == gt['class_id']:
                    iou = self.calculate_mask_iou(pred['mask'], gt['mask'])
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_gt_idx != -1:
                matches.append({
                    'pred_idx': pred_idx,
                    'gt_idx': best_gt_idx,
                    'iou': best_iou,
                    'class_id': pred['class_id']
                })
                used_gt.add(best_gt_idx)
                used_pred.add(pred_idx)

        return matches, used_gt, used_pred

    def create_colored_mask(self, masks_data, img_height, img_width):
        """Renkli mask oluştur"""
        colored_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        for mask_info in masks_data:
            mask = mask_info['mask']
            class_id = mask_info['class_id']
            color = self.colors.get(class_id, [255, 255, 255])

            # Mask olan yerleri renklendir
            for i in range(3):
                colored_mask[:, :, i] = np.where(mask > 0, color[i], colored_mask[:, :, i])

        return colored_mask

    def visualize_predictions(self, image_path, gt_data, pred_data, matches, save_path=None):
        """Tahminleri ve gerçek maskeleri görselleştir"""
        # Orijinal görüntüyü yükle
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]

        # Figure oluştur
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Prediction vs Ground Truth: {os.path.basename(image_path)}', fontsize=16)

        # 1. Orijinal görüntü
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # 2. Ground Truth
        gt_colored = self.create_colored_mask(gt_data, img_height, img_width)
        overlay_gt = cv2.addWeighted(image_rgb, 0.7, gt_colored, 0.3, 0)
        axes[0, 1].imshow(overlay_gt)
        axes[0, 1].set_title(f'Ground Truth ({len(gt_data)} objects)')
        axes[0, 1].axis('off')

        # 3. Predictions
        pred_colored = self.create_colored_mask(pred_data, img_height, img_width)
        overlay_pred = cv2.addWeighted(image_rgb, 0.7, pred_colored, 0.3, 0)
        axes[1, 0].imshow(overlay_pred)
        axes[1, 0].set_title(f'Predictions ({len(pred_data)} objects)')
        axes[1, 0].axis('off')

        # 4. Matches (TP'ler)
        match_colored = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        for match in matches:
            pred_idx = match['pred_idx']
            gt_idx = match['gt_idx']
            class_id = match['class_id']
            color = self.colors.get(class_id, [255, 255, 255])

            # Hem prediction hem de GT maskını göster
            pred_mask = pred_data[pred_idx]['mask']
            gt_mask = gt_data[gt_idx]['mask']
            combined_mask = np.logical_or(pred_mask, gt_mask)

            for i in range(3):
                match_colored[:, :, i] = np.where(combined_mask, color[i], match_colored[:, :, i])

        overlay_match = cv2.addWeighted(image_rgb, 0.7, match_colored, 0.3, 0)
        axes[1, 1].imshow(overlay_match)
        axes[1, 1].set_title(f'Matches (TP: {len(matches)})')
        axes[1, 1].axis('off')

        # Legend ekle
        legend_elements = []
        for class_id, class_name in self.class_names.items():
            color = [c / 255.0 for c in self.colors.get(class_id, [255, 255, 255])]
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=class_name))

        fig.legend(handles=legend_elements, loc='lower center', ncol=len(self.class_names),
                   bbox_to_anchor=(0.5, 0.02))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def evaluate_single_image(self, image_path):

        # Görüntüyü yükle
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]

        # Label dosyası yolu
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_file = os.path.join(self.labels_path, f"{image_name}.txt")

        # Ground truth yükle
        gt_data = self.load_ground_truth(label_file, img_height, img_width)

        # Model tahminleri
        results = self.model(image_path, verbose=False)
        pred_data = self.process_predictions(results, img_height, img_width)

        # Eşleştirme yap
        matches, used_gt, used_pred = self.match_predictions_with_ground_truth(gt_data, pred_data)

        self.visualization_data.append({
            'image_path': image_path,
            'image_name': image_name,
            'gt_data': gt_data,
            'pred_data': pred_data,
            'matches': matches,
            'used_gt': used_gt,
            'used_pred': used_pred
        })

        # Her sınıf için metrikleri hesapla
        for class_id in self.class_names.keys():
            # True Positives
            tp = len([m for m in matches if m['class_id'] == class_id])

            # False Positives
            fp = len([p for i, p in enumerate(pred_data)
                      if p['class_id'] == class_id and i not in used_pred])

            # False Negatives
            fn = len([g for i, g in enumerate(gt_data)
                      if g['class_id'] == class_id and i not in used_gt])

            self.detailed_results[class_id]['tp'] += tp
            self.detailed_results[class_id]['fp'] += fp
            self.detailed_results[class_id]['fn'] += fn

        return {
            'image_name': image_name,
            'gt_count': len(gt_data),
            'pred_count': len(pred_data),
            'matches': len(matches)
        }

    def analyze_label_distribution(self):
        """Label dosyalarındaki sınıf dağılımını analiz et"""
        print("Label dosyalarındaki sınıf dağılımı analiz ediliyor...")

        label_class_counts = defaultdict(int)
        image_files = [f for f in os.listdir(self.images_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for image_file in image_files[:50]:  # İlk 50 dosyayı kontrol et
            image_name = os.path.splitext(image_file)[0]
            label_file = os.path.join(self.labels_path, f"{image_name}.txt")

            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 6:
                                class_id = int(parts[0])
                                label_class_counts[class_id] += 1

        print("\nLabel dosyalarında bulunan sınıflar:")
        for class_id, count in sorted(label_class_counts.items()):
            print(f"Sınıf ID {class_id}: {count} adet")

        print(f"\nModelinizde tanımlı sınıflar:")
        for class_id, class_name in self.class_names.items():
            print(f"Sınıf ID {class_id}: {class_name}")

        return label_class_counts

    def calculate_metrics(self):
        """Tüm metrikleri hesapla"""
        metrics = {}

        for class_id, class_name in self.class_names.items():
            tp = self.detailed_results[class_id]['tp']
            fp = self.detailed_results[class_id]['fp']
            fn = self.detailed_results[class_id]['fn']

            # Precision, Recall, F1-Score hesapla
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[class_name] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }

        # Genel metrikler
        total_tp = sum(self.detailed_results[c]['tp'] for c in self.class_names.keys())
        total_fp = sum(self.detailed_results[c]['fp'] for c in self.class_names.keys())
        total_fn = sum(self.detailed_results[c]['fn'] for c in self.class_names.keys())

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (
                                                                                                                    overall_precision + overall_recall) > 0 else 0.0

        metrics['Overall'] = {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1
        }

        return metrics

    def create_all_visualizations(self, output_dir="visualization_results_4_class_0.5"):
        """Tüm görüntüler için görselleştirmeler oluştur"""
        print(f"\nTüm görüntüler için görselleştirmeler oluşturuluyor...")

        # Çıktı klasörü oluştur
        os.makedirs(output_dir, exist_ok=True)

        for i, vis_data in enumerate(self.visualization_data):
            if i % 10 == 0:
                print(f"Görselleştirme: {i + 1}/{len(self.visualization_data)}")

            # Görselleştirme oluştur
            save_path = os.path.join(output_dir, f"{vis_data['image_name']}_comparison.png")

            fig = self.visualize_predictions(
                vis_data['image_path'],
                vis_data['gt_data'],
                vis_data['pred_data'],
                vis_data['matches'],
                save_path
            )

            plt.close(fig)  # Bellek tasarrufu için

        print(f"Tüm görselleştirmeler {output_dir} klasörüne kaydedildi.")

    def create_summary_plots(self, metrics, output_dir="visualization_results"):
        """Özet grafikler oluştur"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Metrik karşılaştırma grafiği
        class_names = list(metrics.keys())[:-1]  # Overall hariç
        precisions = [metrics[name]['precision'] for name in class_names]
        recalls = [metrics[name]['recall'] for name in class_names]
        f1_scores = [metrics[name]['f1_score'] for name in class_names]

        x = np.arange(len(class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Classes')
        ax.set_ylabel('Scores')
        ax.set_title('Performance Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Değerleri bar'ların üstüne yaz
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Confusion Matrix benzeri TP/FP/FN grafiği
        tps = [metrics[name]['tp'] for name in class_names]
        fps = [metrics[name]['fp'] for name in class_names]
        fns = [metrics[name]['fn'] for name in class_names]

        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(x - width, tps, width, label='True Positives', alpha=0.8, color='green')
        bars2 = ax.bar(x, fps, width, label='False Positives', alpha=0.8, color='red')
        bars3 = ax.bar(x + width, fns, width, label='False Negatives', alpha=0.8, color='orange')

        ax.set_xlabel('Classes')
        ax.set_ylabel('Count')
        ax.set_title('Detection Results by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Değerleri bar'ların üstüne yaz
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_counts.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Özet grafikler {output_dir} klasörüne kaydedildi.")

    def run_evaluation(self):
        """Tüm test setini değerlendir"""
        print("Değerlendirme başlıyor...")

        # Önce sınıf dağılımını analiz et
        label_distribution = self.analyze_label_distribution()

        # Tüm test görüntülerini al
        image_files = [f for f in os.listdir(self.images_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Toplam {len(image_files)} görüntü bulundu.")

        # Her görüntüyü değerlendir
        results = []
        for i, image_file in enumerate(image_files):
            if i % 10 == 0:
                print(f"İşlenen: {i}/{len(image_files)}")

            image_path = os.path.join(self.images_path, image_file)
            result = self.evaluate_single_image(image_path)
            results.append(result)

        # Metrikleri hesapla
        metrics = self.calculate_metrics()

        return metrics, results, label_distribution

    def print_results(self, metrics):
        """Sonuçları yazdır"""
        print("\n" + "=" * 80)
        print("YOLO SEGMENTASYON MODEL DEĞERLENDİRME SONUÇLARI")
        print("=" * 80)

        print(f"\nKullanılan parametreler:")
        print(f"- Confidence Threshold: {self.conf_threshold}")
        print(f"- IoU Threshold: {self.iou_threshold}")

        print(f"\n{'Sınıf':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
        print("-" * 80)

        for class_name, metric in metrics.items():
            print(f"{class_name:<20} {metric['precision']:<10.3f} {metric['recall']:<10.3f} "
                  f"{metric['f1_score']:<10.3f} {metric['tp']:<5} {metric['fp']:<5} {metric['fn']:<5}")

    def save_results(self, metrics, output_file="evaluation_results.json"):
        """Sonuçları dosyaya kaydet"""
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSonuçlar {output_file} dosyasına kaydedildi.")


# Ana kod
if __name__ == "__main__":
    # Parametreler



    BASE_DIR = os.getenv('BASE_DIR')
    CLAHE_SUBDIR = os.getenv('CLAHE_SUBDIR')
    CLAHE_DIR = os.path.join(BASE_DIR, CLAHE_SUBDIR)
    RUNS_SUBDIR = os.getenv('RUNS_SUBDIR')
    MODEL_SUBDIR = os.getenv('MODEL_SUBDIR')

    BEST_MODEL_PATH = os.path.join(BASE_DIR, RUNS_SUBDIR, MODEL_SUBDIR, "weights", "best.pt")

    model_path = BEST_MODEL_PATH
    test_path = os.path.join(BASE_DIR,CLAHE_DIR,"test")

    # Sınıf isimleri
    class_names = {
        0: 'Filling',
        1: 'Implant',
        2: 'Root Canal Treatment',
        3: 'Impacted tooth'
    }

    # Tester oluştur
    tester = YOLOSegmentationTester(
        model_path=model_path,
        test_path=test_path,
        class_names=class_names,
        conf_threshold=0.1,
        iou_threshold=0.1
    )

    # Değerlendirmeyi çalıştır
    metrics, results, label_distribution = tester.run_evaluation()

    # Sonuçları yazdır
    tester.print_results(metrics)

    # Sonuçları kaydet
    tester.save_results(metrics)

    # Tüm görüntüler için görselleştirmeler oluştur
    tester.create_all_visualizations()

    # Özet grafikler oluştur
    tester.create_summary_plots(metrics)

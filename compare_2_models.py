import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

# Stil ayarları
sns.set(style="whitegrid")
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


BASE_DIR = os.getenv('BASE_DIR')
RUNS_SUBDIR_COMPARE2_MODEL = os.getenv('RUNS_SUBDIR_COMPARE2_MODEL')

# 'runs' klasörünü tara
runs_dir = os.path.join(BASE_DIR, RUNS_SUBDIR_COMPARE2_MODEL)
results = []

for folder in os.listdir(runs_dir):
    subfolder_path = os.path.join(runs_dir, folder)
    results_csv = os.path.join(subfolder_path, "results.csv")
    if os.path.isfile(results_csv):
        try:
            df = pd.read_csv(results_csv)
            df['model'] = folder
            results.append(df)
        except Exception as e:
            print(f"{results_csv} okunamadı: {e}")

# Sonuçları birleştir
if results:
    all_results = pd.concat(results, ignore_index=True)

    # EN İYİ EPOCH'U AL (en yüksek mAP50(B) değerine sahip)
    best_results = all_results.loc[all_results.groupby("model")["metrics/mAP50(B)"].idxmax()]

    # Karşılaştırmak istediğimiz metrik grupları
    box_metrics = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)"
    ]
    mask_metrics = [
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)"
    ]

    # Pandas görüntüleme ayarları - tam precision için
    pd.set_option('display.precision', 4)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # Sonuçları yazdır
    print("BOX (B) En İyi Epoch Sonuçları:")
    print(best_results[["model", "epoch"] + box_metrics], "\n")
    print("MASK (M) En İyi Epoch Sonuçları:")
    print(best_results[["model", "epoch"] + mask_metrics])


    # Görselleştirme
    def plot_metric_group(metric_list, title):
        melted = best_results[["model"] + metric_list].melt(id_vars="model", var_name="Metric", value_name="Value")
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=melted, x="Metric", y="Value", hue="model")

        # Bar üzerinde değerleri tam detaylı göster
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', fontsize=9)

        plt.title(title)
        plt.xticks(rotation=20)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()


    # Grafiklerle karşılaştır
    plot_metric_group(box_metrics, "BOX Tahmin Performans Karşılaştırması")
    plot_metric_group(mask_metrics, "MASK Tahmin Performans Karşılaştırması")

else:
    print("Hiçbir results.csv dosyası bulunamadı.")
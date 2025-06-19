import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv('BASE_DIR')
RUNS_SUBDIR = os.getenv('RUNS_SUBDIR')

runs_dir = os.path.join(BASE_DIR, RUNS_SUBDIR)
results = []

for folder in os.listdir(runs_dir):
    subfolder_path = os.path.join(runs_dir, folder)
    results_csv = os.path.join(subfolder_path, 'results.csv')
    if os.path.isfile(results_csv):
        try:
            df = pd.read_csv(results_csv)
            df['model'] = folder
            results.append(df)
        except Exception as e:
            print(f"{results_csv} okunamadı: {e}")

if results:
    all_results = pd.concat(results, ignore_index=True)

    # EN İYİ EPOCH'U AL (en yüksek mAP50 değerine sahip)
    best_results = all_results.loc[all_results.groupby("model")["metrics/mAP50(B)"].idxmax()]

    metrics_to_compare = [
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)"
    ]

    pd.set_option('display.precision', 4)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print(best_results[["model", "epoch"] + metrics_to_compare])

    # Her bir metrik için grafik çiz
    for metric in metrics_to_compare:
        plt.figure(figsize=(12, 6))

        plot_data = best_results.sort_values('model').reset_index(drop=True)
        bars = plt.bar(range(len(plot_data)), plot_data[metric])

        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        plt.xticks(range(len(plot_data)), plot_data['model'], rotation=45, ha='right')
        plt.title(f'Model Karşılaştırması {metric}')
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.ylim(0, max(plot_data[metric]) * 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

else:
    print("Hiçbir results.csv dosyası bulunamadı.")
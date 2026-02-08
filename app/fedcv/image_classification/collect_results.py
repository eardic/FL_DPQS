import argparse
import os
import pandas as pd
import yaml
import glob
import numpy as np
import os.path as osp
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="D:/FedMLLogs")
    args = parser.parse_args()

    yaml.SafeLoader.add_constructor(
        'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
        lambda loader, node: np.array(node.value))

    not_comp = 0
    table = []
    all_keys = set()

    # İlk olarak tüm conf_yaml dosyalarındaki anahtarları topla
    for p in tqdm(os.listdir(args.path)):
        if not osp.isdir(osp.join(args.path, p)): continue
        conf_path = osp.join(args.path, p, "config.yaml")
        if not osp.exists(conf_path): continue

        with open(conf_path, "r") as f:
            if not "!!python" in f.readline():
                f.seek(0)
            conf_yaml = yaml.safe_load(f)
            all_keys.update(conf_yaml.keys())

    # Anahtarları alfabetik sıraya koy
    all_keys = sorted(all_keys)

    # Tüm key'leri işleme sok
    for p in tqdm(os.listdir(args.path)):
        if not osp.isdir(osp.join(args.path, p)): continue
        conf_path = osp.join(args.path, p, "config.yaml")
        if not osp.exists(conf_path):
            print("Config yaml not found: ", conf_path)
            not_comp += 1
            continue
        with open(conf_path, "r") as f:
            if not "!!python" in f.readline():
                f.seek(0)
            conf_yaml = yaml.safe_load(f)

        yaml_path = osp.join(args.path, p, "weights/best_metrics.yaml")
        if not osp.exists(yaml_path):
            print("Yaml not found: ", yaml_path)
            not_comp += 1
            continue
        save_dval_scores = "data_val_save_scores" in conf_yaml and conf_yaml["data_val_save_scores"]
        if save_dval_scores and len(glob.glob(osp.join(args.path, p, "data_val/*"))) < conf_yaml["comm_round"]:
            print("Training not completed ! : ", p)
            not_comp += 1
            continue
        with open(yaml_path, "r") as f:
            best_yaml = yaml.safe_load(f)
            round = best_yaml["round"]
            acc = 100 * best_yaml["test_acc"]
            prec = 100 * best_yaml["metrics"]["test_precision"]
            recall = 100 * best_yaml["metrics"]["test_recall"]
            f1 = 100 * best_yaml["metrics"]["test_f1"]
            psnr = best_yaml["metrics"]["test_psnr"] / best_yaml["metrics"]["test_total"]
            ssim = 100 * best_yaml["metrics"]["test_ssim"] / best_yaml["metrics"]["test_total"]
            bacc = 100 * best_yaml["metrics"]["test_bacc"] if "test_bacc" in best_yaml["metrics"] else 0
            # Tüm key'leri dolaşarak row listesi oluştur
            row = [p, acc, prec, recall, f1, bacc, psnr, ssim, round]
            for key in all_keys:
                row.append(conf_yaml.get(key, ""))  # Anahtar yoksa boş değer ekle
            table.append(row)

    # Kolon isimlerini belirle
    columns = ["name", "acc", "prec", "recall", "f1", "bacc", "psnr", "ssim", "round"] + all_keys

    df = pd.DataFrame(table, columns=columns)
    df.to_excel(osp.join(args.path, "summary.xlsx"))

    print(f"Total not completed / completed: {not_comp}/{len(table)}")

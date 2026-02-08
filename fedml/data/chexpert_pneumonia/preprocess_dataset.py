from pathlib import Path

import numpy as np
import glob
import os
import shutil

from tqdm import tqdm

import os
import shutil
import pandas as pd


def organize_images(csv_path, dest_dir):
    """
    CSV dosyasına göre görüntüleri sınıflar klasörlerine kopyalar.

    Args:
        csv_path (str): CSV dosyasının yolu.
        src_img_dir (str): Orijinal görüntülerin bulunduğu klasör yolu.
        dest_dir (str): Düzenlenmiş görüntülerin hedef klasörü.
    """
    # CSV dosyasını yükle
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows()):
        img_name = row['Image']
        class_label = str(row['Pneumonia'])  # Sınıf etiketini al

        dest_class_dir = os.path.join(dest_dir, class_label)  # Hedef sınıf klasörü yolu

        # Sınıf klasörü yoksa oluştur
        os.makedirs(dest_class_dir, exist_ok=True)

        img_path = Path(img_name)
        new_name = f"{img_path.parent.parent.name}_{img_path.parent.name}_{img_path.name}"

        # Görüntüyü sınıf klasörüne kopyala
        shutil.copy(img_name, os.path.join(dest_class_dir, new_name))

    print(f"Görüntüler {dest_dir} dizinine organize edildi.")


if __name__ == '__main__':
    # Kullanım
    train_csv_path = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\chexpert\train_pneumonia.csv"  # Eğitim CSV dosyasının yolu
    test_csv_path = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\chexpert\test_pneumonia.csv"  # Test CSV dosyasının yolu
    dest_dir = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\chexpert_pneumonia"  # Orijinal görüntülerin bulunduğu dizin

    # Eğitim ve test görüntülerini organize et
    organize_images(train_csv_path, os.path.join(dest_dir, "train"))
    organize_images(test_csv_path, os.path.join(dest_dir, "test"))

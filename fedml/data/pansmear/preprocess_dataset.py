from pathlib import Path

import numpy as np
import glob
import os
import shutil

from tqdm import tqdm

if __name__ == '__main__':

    class_folders = {
        "C:\\Datasets\\sipakmed\\im_Dyskeratotic\\CROPPED": 0,
        "C:\\Datasets\\sipakmed\\im_Koilocytotic\\CROPPED": 1,
        "C:\\Datasets\\sipakmed\\im_Metaplastic\\CROPPED": 2,
        "C:\\Datasets\\sipakmed\\im_Parabasal\\CROPPED": 3,
        "C:\\Datasets\\sipakmed\\im_Superficial-Intermediate\\CROPPED": 4,
    }
    output = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\pansmear"
    os.makedirs(output, exist_ok=True)
    # Her klasördeki dosyaları tarar ve dosya yolunu ve sınıf ID'sini yazar
    for folder, class_id in class_folders.items():
        for filename in tqdm(os.listdir(folder)):
            if filename.endswith(".bmp"):
                file_path = os.path.join(folder, filename)
                dst = os.path.join(output, str(class_id))
                os.makedirs(dst, exist_ok=True)
                shutil.copy(file_path, os.path.join(dst, filename))
    # input_dir = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\pansmear"
    # files = list(glob.glob(os.path.join(input_dir, "**", "*.bmp")))
    # np.random.shuffle(files)
    # outdir = os.path.join(input_dir, "test")
    # os.makedirs(outdir, exist_ok=True)
    # for f in files[:1000]:
    #     p = Path(f).parent.name
    #     dir = os.path.join(outdir, p)
    #     os.makedirs(dir, exist_ok=True)
    #     shutil.move(f, dir)

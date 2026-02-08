import os
import random
import shutil
from collections import defaultdict

# Veri kümesi kök yolu
dataset_path = r"C:\Datasets\BreaKHis_v1\BreaKHis_v1\histology_slides\breast"
output_train_path = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\breakhis_v1\train"
output_test_path = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\breakhis_v1\test"

# Hedef klasörler oluştur
os.makedirs(output_train_path, exist_ok=True)
os.makedirs(output_test_path, exist_ok=True)

# Hasta bazında verileri organize et
patient_data = defaultdict(list)

# Ana sınıflar
classes = ["benign", "malignant"]

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    for root, _, files in os.walk(class_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                # Hasta kimliğini belirle
                patient_id = root.split(os.sep)[-2]  # Hasta ID dizin isimlerinden elde edilir
                file_path = os.path.join(root, file)
                patient_data[patient_id].append((file_path, class_name))

# Hasta listesi
patients = list(patient_data.keys())
random.shuffle(patients)  # Hastaları karıştır

# 7:3 oranında böl
split_idx = int(0.7 * len(patients))
train_patients = patients[:split_idx]
test_patients = patients[split_idx:]
print(f"Train hasta sayısı: {len(train_patients)}")
print(f"Test hasta sayısı: {len(test_patients)}")

# Eğitim ve test setlerine dosyaları kopyala
def copy_files(patients, output_path):
    for patient_id in patients:
        for file_path, class_name in patient_data[patient_id]:
            target_class_folder = os.path.join(output_path, class_name)
            os.makedirs(target_class_folder, exist_ok=True)
            shutil.copy(file_path, os.path.join(target_class_folder, os.path.basename(file_path)))


print("Eğitim setine kopyalanıyor...")
copy_files(train_patients, output_train_path)

print("Test setine kopyalanıyor...")
copy_files(test_patients, output_test_path)

print("Veri seti başarıyla 7:3 oranında bölündü!")

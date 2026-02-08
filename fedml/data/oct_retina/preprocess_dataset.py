import os
import random
import time

import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from pathlib import Path

import numpy as np
import glob
import os
import shutil

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

label_index = {
    "CNV": 0,
    "DME": 1,
    "DRUSEN": 2,
    "NORMAL": 3
}


def pad_to_square(img):
    # Görüntünün boyutlarını al
    w, h = img.size
    max_dim = max(w, h)

    # Sol, üst, sağ, alt padding miktarlarını hesapla
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2
    pad_right = max_dim - w - pad_left
    pad_bottom = max_dim - h - pad_top

    return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def apply_transforms_and_save(input_dir, output_dir, transform=None):
    """
    Verilen klasördeki görüntülere transform uygular ve sonuçları belirtilen yapıdaki klasörlere kaydeder.

    Args:
        input_dir (str): Girdi veri klasörü. (örneğin, "C:\\Datasets\\oct2018\\train")
        output_dir (str): Çıktı klasörü (örneğin, "C:\\Processed\\train")
        transform (callable): PyTorch transformları.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in tqdm(os.listdir(input_dir), desc=f"Processing {input_dir}"):
        label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, str(label_index[label.upper()]))

        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)

            try:
                img = Image.open(img_path).convert("RGB")  # Görüntüyü yükle ve RGB'ye çevir
                if transform is not None:
                    img = transform(img)  # Transformları uygula

                save_path = os.path.join(output_label_dir, img_file)
                img.save(save_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


if __name__ == '__main__':
    # Transformlar
    image_size = 224
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))
    # ])
    # test_transform = transforms.Compose([
    #     transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    # ])
    train_transform = None
    test_transform = None

    # Giriş ve çıkış klasörleri
    base_input_dir = r"C:\Datasets\OCT"
    base_output_dir = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\oct_retina"

    # Uygulama
    apply_transforms_and_save(os.path.join(base_input_dir, "train"),
                              os.path.join(base_output_dir, "train"),
                              train_transform)
    apply_transforms_and_save(os.path.join(base_input_dir, "test"),
                              os.path.join(base_output_dir, "test"),
                              test_transform)

    # import matplotlib.pyplot as plt
    # from PIL import Image
    # import torchvision.transforms as transforms
    # from torchvision.transforms import InterpolationMode
    #
    # # Özel Gaussian Noise Sınıfı
    # class AddGaussianNoise:
    #     def __init__(self, mean=0.0, std_range=(0.01, 0.05)):
    #         self.mean = mean
    #         self.std_range = std_range
    #
    #     def __call__(self, img):
    #         std = random.uniform(*self.std_range)  # Rastgele std seç
    #         noise = torch.randn(img.size()) * std + self.mean
    #         noisy_img = img + noise
    #         return torch.clamp(noisy_img, 0.0, 1.0)  # Piksel değerlerini 0-1 aralığında sınırla
    #
    # torch.random.manual_seed(cv2.getTickCount())
    #
    # # Augmentasyon pipeline'ı
    # image_size = (224, 224)  # Örneğin model input boyutu
    # train_transform = transforms.Compose([
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    #     transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(1, 5), interpolation=InterpolationMode.BICUBIC),
    #     transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(1.5, 3.5), interpolation=InterpolationMode.BICUBIC),
    #     transforms.ToTensor(),
    #     AddGaussianNoise(mean=0.0, std_range=(0.01, 0.03)),  # Gaussian Noise ekle
    #     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizasyon
    # ])
    #
    # # Örnek bir görüntüyü yükle ve augmentasyon uygula
    # image_path = r'C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\oct_retina\test\0\CNV-1569-1.jpeg'  # Örnek OCT Retina görüntüsü
    # image = Image.open(image_path).convert('L')  # Gri ölçekli olarak yükle
    #
    # # Augmentasyon uygulanmış görüntüyü al
    # augmented_image = train_transform(image)
    #
    #
    # # Görselleştirme (normalize edilmiş görüntüyle)
    # def denormalize(tensor, mean, std):
    #     return tensor * std[0] + mean[0]  # Normalizasyonu geri al
    #
    #
    # augmented_image = denormalize(augmented_image, mean=[0.5], std=[0.5])
    #
    # # Tensor'u görüntülemek için numpy array'e çevir
    # augmented_image_np = augmented_image.squeeze().numpy()
    #
    # # Orijinal ve augment edilmiş görüntüyü görselleştir
    # plt.figure(figsize=(10, 5))
    #
    # # Orijinal görüntü
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')
    #
    # # Augment edilmiş görüntü
    # plt.subplot(1, 2, 2)
    # plt.imshow(augmented_image_np, cmap='gray')
    # plt.title('Augmented Image')
    # plt.axis('off')
    #
    # plt.show()

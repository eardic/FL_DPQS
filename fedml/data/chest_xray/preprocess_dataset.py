import os

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

label_index = {
    "NORMAL": 0,
    "PNEUMONIA": 1
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


def apply_transforms_and_save(input_dir, output_dir, transform):
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
        if label[0] == ".": continue
        label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, str(label_index[label.upper()]))

        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        for img_file in os.listdir(label_dir):
            if img_file[0] == ".": continue
            img_path = os.path.join(label_dir, img_file)

            try:
                img = Image.open(img_path).convert("RGB")  # Görüntüyü yükle ve RGB'ye çevir
                transformed_img = transform(img)  # Transformları uygula

                # Kaydedilecek dosya yolu
                save_path = os.path.join(output_label_dir, img_file)
                transformed_img.save(
                    save_path)  # Tensor'u tekrar PIL formatına çevir ve kaydet
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


if __name__ == '__main__':
    np.random.seed(10)

    image_size = 224
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        ]
    )

    # Giriş ve çıkış klasörleri
    base_input_dir = r"C:\Datasets\chest_xray"
    base_output_dir = r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\chest_xray_v2"

    # Uygulama
    apply_transforms_and_save(os.path.join(base_input_dir, "train"), os.path.join(base_output_dir, "train"),
                              train_transform)
    apply_transforms_and_save(os.path.join(base_input_dir, "test"), os.path.join(base_output_dir, "test"),
                              test_transform)

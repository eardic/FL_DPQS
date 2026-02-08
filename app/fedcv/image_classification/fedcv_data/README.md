# Dataset Preparation Guide

This document explains how to prepare datasets for the experiments in this repository.

All datasets **must be placed under** the following directory:

```
app/fedcv/image_classification/fedcv_data
```

Training is performed primarily on **CIFAR-10** and **MNIST**, so these datasets should be prepared first.  
If **open-set noise** experiments are required, the corresponding noise datasets must also be downloaded.

Dataset preparation scripts and loaders can be found under the `fedml/data` directory.

---

## 1. CIFAR-10

**Download from:**  
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Extract the contents into the `fedcv_data` folder.

**Expected folder structure:**

```
fedcv_data/
└── cifar-10-batches-py/
    ├── batches.meta
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── readme.html
```

---

## 2. MNIST

**Download from:**  
https://fedcv.s3.us-west-1.amazonaws.com/MNIST.zip

Extract the archive into the `fedcv_data` directory.

**Expected folder structure:**

```
fedcv_data/MNIST/
├── train/all_data_0_niid_0_keep_10_train_9.json
└── test/all_data_0_niid_0_keep_10_test_9.json
```

### Normalization

After extraction, run the following script to normalize images to the `[0, 1]` range:

```bash
python fedml/data/MNIST/renormalize.py
```

---

## 3. Noise Datasets (Open-Set Noise)

The following datasets are used as **open-set noise** sources:
- ImageNet32
- FEMNIST
- SVHN

PyTorch `DataLoader` implementations are provided at:

- `fedml/data/ImageNet/datasets.py`  
  - Class: `ImageNet32Noise`
- `fedml/data/FederatedEMNIST/data_loader.py`  
  - Class: `FEMNISTLetterNoise`
- `fedml/data/cifar10/data_loader.py`  
  - Class: `ImageNoiseFolder` (used for SVHN)

### ImageNet32

**Download from:**  
https://www.image-net.org/data/downsample/Imagenet32_train.zip

Extract **only** `train_data_batch_1` into:

```
fedcv_data/imagenet32/train_data_batch_1/
```

---

### SVHN

**Download from:**  
https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers?select=train

Extract training images into:

```
fedcv_data/SVHN/train/
```

---

### FEMNIST

**Download from:**  
https://fedml.s3-us-west-1.amazonaws.com/fed_emnist.tar.bz2

Use the file `fed_emnist_train.h5` and place it directly under:

```
fedcv_data/
```

---

## 4. Medical Imaging Datasets

The following medical datasets are used in **differential privacy** and **adaptive quantization** experiments.

Each dataset has an associated preprocessing script.  
After downloading the dataset, **update the dataset paths inside the script and run it**.

Final directory structures should match the examples below.

---

### Chest X-ray (Pneumonia)

- **Preprocessing Script:**  
  `fedml/data/chest_xray/preprocess_dataset.py`
- **Dataset:**  
  https://data.mendeley.com/datasets/rscbjbr9sj/3

**Expected folder structure:**

```
fedcv_data/chest_xray/
├── train/
│   ├── normal/*.jpeg
│   └── pneumonia/*.jpeg
└── test/
    ├── normal/*.jpeg
    └── pneumonia/*.jpeg
```

---

### BreaKHis (Breast Cancer Histopathology)

- **Preprocessing Script:**  
  `fedml/data/breakhis_v1/preprocess.py`
- **Dataset:**  
  http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz

**Expected folder structure:**

```
fedcv_data/breakhis_v1/
├── train/
│   ├── Benign/*.png
│   └── Malignant/*.png
└── test/
    ├── Benign/*.png
    └── Malignant/*.png
```

---

### PAP Smear (SipakMeD)

- **Preprocessing Script:**  
  `fedml/data/pansmear/preprocess_dataset.py`
- **Dataset:**  
  https://www.cs.uoi.gr/~marina/sipakmed.html

**Expected folder structure:**

```
fedcv_data/pansmear/
├── train/
│   ├── class1/*.bmp
│   ├── class2/*.bmp
│   ├── class3/*.bmp
│   ├── class4/*.bmp
│   └── class5/*.bmp
└── test/
    ├── class1/*.bmp
    ├── class2/*.bmp
    ├── class3/*.bmp
    ├── class4/*.bmp
    └── class5/*.bmp
```

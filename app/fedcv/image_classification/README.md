# Batch Training Scripts (Windows)

To facilitate batch training in a Windows environment, example `.bat` (batch) scripts are provided.  
Each script corresponds to a specific set of experiments conducted in the thesis.

---

## Available Scripts

### 1. Sample Selection Experiments

- **Script Path:**  
  `app/fedcv/image_classification/sample_selection.bat`

- **Description:**  
  Used for the sample selection experiments discussed in the thesis.

- **Datasets:**  
  - CIFAR-10  
  - MNIST

- **Open-Set Noise Datasets:**  
  - ImageNet32  
  - SVHN  
  - FEMNIST

---

### 2. Differential Privacy and Quantization (CIFAR-10 & MNIST)

- **Script Path:**  
  `app/fedcv/image_classification/mnist_cifar10_dp_and_quant.bat`

- **Description:**  
  Used for differential privacy and adaptive quantization experiments on CIFAR-10 and MNIST datasets.

---

### 3. Differential Privacy and Quantization (Medical Datasets)

- **Script Path:**  
  `app/fedcv/image_classification/medical_dp_and_quant.bat`

- **Description:**  
  Used for differential privacy and quantization experiments on medical imaging datasets.

---

## Execution Notes

> **Important:**  
> All batch scripts **must be executed from within** the following directory:

```
app/fedcv/image_classification
```

Running the scripts from a different working directory may result in incorrect paths or failed executions.

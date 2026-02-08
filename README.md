# FL_DPQS

**Federated Learning with Sample Selection, Differential Privacy, and Adaptive Quantization**

This repository contains the codebase accompanying the following papers:

## Related Publications

1. **Enhanced Privacy and Communication Efficiency in Non-IID Federated Learning With Adaptive Quantization and Differential Privacy**  
   IEEE Access  
   https://ieeexplore.ieee.org/document/10937694

2. **Sample Selection Using Multi-Task Autoencoders in Federated Learning with Non-IID Data**  
   Engineering Science and Technology, an International Journal (Elsevier)  
   https://www.sciencedirect.com/science/article/pii/S2215098624003069

---

## Framework

This codebase are built on top of the open-source **FedML** framework for large-scale federated learning experiments:

- FedML GitHub Repository:  
  https://github.com/FedML-AI/FedML

---

## Implemented Algorithms and Code Structure

### 1. Sample Selection Algorithms
- **Location:** `fedml/core/data_valuation/`
- **Description:**  
  Algorithms for detecting anomalous, noisy, or low-quality samples in federated clients using data valuation and outlier detection techniques.

### 2. Laplacian-based Differential Privacy (DP)
- **Location:** `fedml/core/dp/mechanisms/`
- **Description:**  
  Implements Laplace-based differential privacy mechanisms to protect local model updates before communication.

### 3. Quantization Algorithms
- **Location:** `fedml/utils/quantize.py`
- **Description:**  
  Fixed and adaptive bit-length quantization methods designed to reduce communication overhead in federated learning.

### 4. Server and Client Algorithms
- **Location:** `fedml/simulation/sp/fedavg/`
- **Description:**  
  Server and client logic based on the Federated Averaging (FedAvg) algorithm, extended with privacy and efficiency enhancements.

### 5. Environment Variable Configuration
- **Location:** `fedml/__init__.py`
- **Description:**  
  Centralized definition and management of training hyperparameters via environment variables.

### 6. Model Training and Evaluation
- **Location:** `fedml/ml/trainer/my_model_trainer_classification.py`
- **Description:**  
  Training and evaluation logic for classification models used in federated learning experiments.

### 7. Multi-Task AutoEncoder and Classification Models
- **Location:** `fedml/model/cv/`
- **Description:**  
  Implementation of Multi-Task AutoEncoders and classification models.  
  Refer to `model_hub.py` for model definitions and configurations.

---

## Usage Instructions

### Environment
- **Operating System:** Windows 11  
- **Python Version:** Python 3.6

### Installation

1. **(Optional but Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate

### Project Directory

Location: `app\fedcv\image_classification`  
   - Visit this directory to access training scripts and learn about the required datasets.  
   - Refer to the following instruction files:  
     - `image_classification\README.txt`  
     - `image_classification\fedcv_data\README.txt`
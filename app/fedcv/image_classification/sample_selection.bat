@REM Important ! Please add main project path to PYTHONPATH like below
@REM set PYTHONPATH=%PYTHONPATH%;D:\Projects\FL_DPQS;
set CLIENT_MAX_WORKERS=1
set MODEL_NAME=aec
set AEC_USE_VAE=0
set BATCH_SIZE=64
set DP_ENABLE=0
set Q_ENABLE=0

set CLIENT_NUM_IN_TOTAL=1000
set CLIENT_NUM_PER_ROUND=100
set CLIENT_OPTIMIZER=sgd
set CLIENT_LR=0.1
set CLIENT_WD=0.001

set DATA_VAL_TYPE=global
set DATA_VAL_UPDATE_PERIOD=5
set DATA_VAL_SELECT_ALGO=all
set DATA_VAL_ROUND_START=400

@REM "loss" or "feature" based sample selection, default: loss
@REM IMPORTANT: don't run fedbal_loss or apply closed-set noise when "feature" is used !!
@REM set ANOM_ALGO_SOURCE=loss

@REM when using feature-based sample selection, you can enable the proposed SVDD algorithm like below:
@REM set DATA_VAL_ROUND_START=600
@REM set AEC_IAE_ROUND_START=500
@REM set AEC_IAE_NU=0.1
@REM set AEC_IAE_LAMBDA=0.00001

@REM MNIST Experiments
set DATASET=mnist
set DATASET_PARTITION=hetero

set CLIENT_LABEL_NOISE=0
python main_fedml_image_classification.py --cf config2/fedml_config_aec.yaml
set CLIENT_LABEL_NOISE=0.4
python main_fedml_image_classification.py --cf config2/fedml_config_aec.yaml

set CLIENT_NOISE_TYPE=openset
set CLIENT_NOISE_SOURCE=femnist
python main_fedml_image_classification.py --cf config2/fedml_config_aec_fedbal_loss.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_iforest.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_ocsvm.yaml
set CLIENT_NOISE_SOURCE=imagenet32
python main_fedml_image_classification.py --cf config2/fedml_config_aec_fedbal_loss.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_iforest.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_ocsvm.yaml

set CLIENT_LABEL_NOISE=0.4
set CLIENT_NOISE_TYPE=closedset
python main_fedml_image_classification.py --cf config2/fedml_config_aec_fedbal_loss.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_iforest.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_ocsvm.yaml

set CLIENT_NOISE_TYPE=
set CLIENT_NOISE_SOURCE=

@REM @REM CIFAR10 Experiments

set DATASET=cifar10
set DATASET_PARTITION=hetero

set CLIENT_LABEL_NOISE=0
python main_fedml_image_classification.py --cf config2/fedml_config_aec.yaml
set CLIENT_LABEL_NOISE=0.4
python main_fedml_image_classification.py --cf config2/fedml_config_aec.yaml

set CLIENT_NOISE_TYPE=openset
set CLIENT_NOISE_SOURCE=svhn
python main_fedml_image_classification.py --cf config2/fedml_config_aec_fedbal_loss.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_iforest.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_ocsvm.yaml

set CLIENT_NOISE_SOURCE=imagenet32
python main_fedml_image_classification.py --cf config2/fedml_config_aec_fedbal_loss.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_iforest.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_ocsvm.yaml

set CLIENT_NOISE_TYPE=closedset
python main_fedml_image_classification.py --cf config2/fedml_config_aec_fedbal_loss.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_iforest.yaml
python main_fedml_image_classification.py --cf config2/fedml_config_aec_ocsvm.yaml

set CLIENT_NOISE_TYPE=
set CLIENT_NOISE_SOURCE=
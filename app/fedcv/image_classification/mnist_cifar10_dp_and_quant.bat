@REM Important ! Please add main project path to PYTHONPATH like below
@REM set PYTHONPATH=%PYTHONPATH%;D:\Projects\FL_DPQS;
set CLIENT_MAX_WORKERS=1
set BATCH_SIZE=64
set CLIENT_LR=0.1
set CLIENT_WD=0.001
set CLIENT_OPTIMIZER=sgd
set COMM_ROUND=1000
set LOCAL_EPOCHS=5
set TEST_FREQ=10
set MODEL_NAME=cnn
@REM for cifar10 
@REM set MODEL_NAME=vgg7_bn

set CLIENT_NUM_IN_TOTAL=1000
set CLIENT_NUM_PER_ROUND=100

@REM for cifar10 
@REM set DATASET=cifar10
set DATASET=mnist
set DATASET_PARTITION=hetero

@REM without gradient norm clipping
python main_fedml_image_classification.py --cf config2/fedml_config_dpq.yaml

set GRAD_CLIP=100
set GRAD_CLIP_NORM=1
set DP_ENABLE=1
set DP_M_TYPE=laplace
set DP_EPSILON=10000

@REM with gradient clipping but without quantization
python main_fedml_image_classification.py --cf config2/fedml_config_dpq.yaml

set Q_ENABLE=1
set Q_OP=per_batch
set Q_ALGO=stochastic
set Q_SCHEDULE=cosine
set Q_SCHEDULE_MIN_B=8

@REM with gradient clipping and cosine-based dynamic quantization
python main_fedml_image_classification.py --cf config2/fedml_config_dpq.yaml

set Q_DYNAMIC_ENABLE=1
set Q_DYNAMIC_HS_W=0.5
set Q_DYNAMIC_SS_W=0.5

@REM with gradient clipping and cosine-based dynamic quantization with client importance (based on shannon entropy and client data size)
python main_fedml_image_classification.py --cf config2/fedml_config_dpq.yaml

set Q_SCHEDULE=
set Q_DYNAMIC_ENABLE=
set Q_SCHEDULE_MIN_B=

@REM static quantization 8-bit
set Q_PRECISION=8
python main_fedml_image_classification.py --cf config2/fedml_config_dpq.yaml

@REM static quantization 12-bit
set Q_PRECISION=12
python main_fedml_image_classification.py --cf config2/fedml_config_dpq.yaml

@REM static quantization 16-bit
set Q_PRECISION=16
python main_fedml_image_classification.py --cf config2/fedml_config_dpq.yaml

set Q_PRECISION=
set Q_ENABLE=
set DP_ENABLE=
set GRAD_CLIP=
set GRAD_CLIP_NORM=
set DP_M_TYPE=
set DP_EPSILON=

pause
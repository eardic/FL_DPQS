@REM Important ! Please add main project path to PYTHONPATH like below
@REM set PYTHONPATH=%PYTHONPATH%;D:\Projects\FL_DPQS;
set CLIENT_MAX_WORKERS=1
set BATCH_SIZE=64
set CLIENT_LR=0.0003
set CLIENT_WD=0.0005
set CLIENT_OPTIMIZER=adam
set COMM_ROUND=100
set LOCAL_EPOCHS=2
set TEST_FREQ=3
set MODEL_NAME=efficientnet-b0

set CLIENT_NUM_IN_TOTAL=10
set CLIENT_NUM_PER_ROUND=10

@REM Medical imaging experiments
@REM set DATASET=chest_xray
@REM set DATASET=pansmear
set BEST_METRIC=bacc
set DATASET=breakhis_v1
set DATASET_PARTITION=hetero

@REM without gradient norm clipping
python main_fedml_image_classification.py --cf config2/fedml_config_dpq.yaml

set GRAD_CLIP=1000
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
set Q_SCHEDULE_MIN_B=12

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
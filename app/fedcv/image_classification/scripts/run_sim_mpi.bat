@echo off

set WORKER_NUM=%1
echo "Workers: %WORKER_NUM%"

set /a "PROCESS_NUM=%WORKER_NUM%+1"
echo "Processes: %PROCESS_NUM%"

set PYTHONPATH=%PYTHONPATH%;C:\Users\eardic\Documents\PyCharmProjects\FedML_PHD;

mpiexec.exe -n %PROCESS_NUM% python main_fedml_image_classification.py --cf config/simulation/fedml_config.yaml
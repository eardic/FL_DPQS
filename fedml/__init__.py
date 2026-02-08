import logging
import multiprocess as multiprocessing
import os
import random

import numpy as np
import torch

import fedml
from .cli.env.collect_env import collect_env
from .constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_SP,
    FEDML_SIMULATION_TYPE_MPI,
    FEDML_SIMULATION_TYPE_NCCL,
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
    FEDML_TRAINING_PLATFORM_CROSS_DEVICE,
)
from .core.common.ml_engine_backend import MLEngineBackend

_global_training_type = None
_global_comm_backend = None

__version__ = "0.7.302"


def init(args=None):
    """Initialize FedML Engine."""
    collect_env()

    if args is None:
        args = load_arguments(fedml._global_training_type, fedml._global_comm_backend)

    fedml._global_training_type = args.training_type
    fedml._global_comm_backend = args.backend

    """
    # Windows/Linux/MacOS compatability issues on multi-processing
    # https://github.com/pytorch/pytorch/issues/3492
    """
    if multiprocessing.get_start_method() != "spawn":
        # force all platforms (Windows/Linux/MacOS) to use the same way (spawn) for multiprocessing
        multiprocessing.set_start_method("spawn", force=True)

    """
    # https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    if torch.is_anomaly_enabled():
        print("Pytorch anomaly detection is enabled !")

    mlops.pre_setup(args)

    if args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION and hasattr(args, "backend") and args.backend == "MPI":
        args = init_simulation_mpi(args)

    elif args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION and hasattr(args, "backend") and args.backend == "sp":
        args = init_simulation_sp(args)
    elif (
            args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
            and hasattr(args, "backend")
            and args.backend == FEDML_SIMULATION_TYPE_NCCL
    ):
        from .simulation.nccl.base_framework.common import FedML_NCCL_Similulation_init

        args = FedML_NCCL_Similulation_init(args)

    elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO:
        if not hasattr(args, "scenario"):
            args.scenario = "horizontal"
        if args.scenario == "horizontal":
            init_cross_silo_horizontal(args)
        elif args.scenario == "hierarchical":
            args = init_cross_silo_hierarchical(args)

    elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_DEVICE:
        args = init_cross_device(args)
    else:
        raise Exception("no such setting: training_type = {}, backend = {}".format(args.training_type, args.backend))

    from fedml.data.config import DataValArgs
    DataValArgs.client_test_size = getattr(args, "client_max_test_size", None)
    DataValArgs.client_label_noise = getattr(args, "client_label_noise", None)
    DataValArgs.client_noise_type = getattr(args, "client_noise_type", None)
    DataValArgs.client_noise_source = getattr(args, "client_noise_source", None)

    if "WANDB_PROJECT" in os.environ and hasattr(args, "wandb_project"):
        args.wandb_project = os.environ["WANDB_PROJECT"]
        print("Wandb project is changed to ", args.wandb_project)

    if "CLIENT_NUM_PER_ROUND" in os.environ and hasattr(args, "client_num_per_round"):
        args.client_num_per_round = int(os.environ["CLIENT_NUM_PER_ROUND"])
    if "CLIENT_NUM_IN_TOTAL" in os.environ and hasattr(args, "client_num_in_total"):
        args.client_num_in_total = int(os.environ["CLIENT_NUM_IN_TOTAL"])
    if "DATASET" in os.environ and hasattr(args, "dataset"):
        args.dataset = os.environ["DATASET"]
    if "COMM_ROUND" in os.environ and hasattr(args, "comm_round"):
        args.comm_round = int(os.environ["COMM_ROUND"])
    if "LOCAL_EPOCHS" in os.environ and hasattr(args, "epochs"):
        args.epochs = int(os.environ["LOCAL_EPOCHS"])
    if "BEST_METRIC" in os.environ and hasattr(args, "best_metric"):
        args.best_metric = os.environ["BEST_METRIC"]

    args.wandb_name = f"{args.dataset}-{args.client_num_in_total}-{args.client_num_per_round}-e{args.epochs}-{args.wandb_name}"

    if "BATCH_SIZE" in os.environ and hasattr(args, "batch_size"):
        args.batch_size = int(os.environ["BATCH_SIZE"])
        args.wandb_name += f"-b{args.batch_size}"

    if "DATASET_PARTITION" in os.environ and hasattr(args, "partition_method"):
        args.partition_method = os.environ["DATASET_PARTITION"]
    if "PARTITION_ALPHA" in os.environ and hasattr(args, "partition_alpha"):
        args.partition_alpha = float(os.environ["PARTITION_ALPHA"])
    if "MODEL_NAME" in os.environ and hasattr(args, "model"):
        args.model = os.environ["MODEL_NAME"]
    if "AEC_USE_VAE" in os.environ and hasattr(args, "aec_use_vae"):
        args.aec_use_vae = int(os.environ["AEC_USE_VAE"])
        if args.aec_use_vae:
            args.wandb_name += f"-vae"
    if "CLIENT_MAX_WORKERS" in os.environ and hasattr(args, "client_max_workers"):
        args.client_max_workers = int(os.environ["CLIENT_MAX_WORKERS"])

    if "DATA_VAL_TYPE" in os.environ and hasattr(args, "data_val_type"):
        args.data_val_type = os.environ["DATA_VAL_TYPE"]
        args.wandb_name += f"-{args.data_val_type}"
    if "DATA_VAL_SELECT_ALGO" in os.environ and hasattr(args, "data_val_select_algo"):
        args.data_val_select_algo = os.environ["DATA_VAL_SELECT_ALGO"]
        args.wandb_name += f"-{args.data_val_select_algo}"
    if "DATA_VAL_UPDATE_PERIOD" in os.environ and hasattr(args, "data_val_update_period"):
        args.data_val_update_period = int(os.environ["DATA_VAL_UPDATE_PERIOD"])
        args.wandb_name += f"-up{args.data_val_update_period}"
    if "DATA_VAL_SELECT_RATIO" in os.environ and hasattr(args, "data_val_select_ratio"):
        args.data_val_select_ratio = float(os.environ["DATA_VAL_SELECT_RATIO"])
        if hasattr(args, "data_val_select_algo") and args.data_val_select_algo != "all":
            args.wandb_name += f"-{args.data_val_select_ratio}"
    if "DATA_VAL_ROUND_START" in os.environ and hasattr(args, "data_val_round_start"):
        def_val = args.data_val_round_start
        args.data_val_round_start = int(os.environ["DATA_VAL_ROUND_START"])
        args.wandb_name = args.wandb_name.replace(f"-e{def_val}", f"-e{args.data_val_round_start}")
    if "DATA_VAL_PER_CLASS" in os.environ and hasattr(args, "data_val_per_class"):
        args.data_val_per_class = int(os.environ["DATA_VAL_PER_CLASS"])
        if args.data_val_per_class:
            args.wandb_name += f"-pcls"

    if "FEDBAL_PROB" in os.environ and hasattr(args, "fedbal_prob"):
        args.fedbal_prob = float(os.environ["FEDBAL_PROB"])
        args.wandb_name += f"-fbp{args.fedbal_prob}"

    if "ANOM_OCSVM_NU" in os.environ and hasattr(args, "anomaly_ocsvm_nu"):
        args.anomaly_ocsvm_nu = float(os.environ["ANOM_OCSVM_NU"])
        args.wandb_name += f"-nu{args.anomaly_ocsvm_nu}"

    if "ANOM_IFOREST_NU" in os.environ and hasattr(args, "anomaly_iforest_c"):
        args.anomaly_iforest_c = float(os.environ["ANOM_IFOREST_NU"])
        args.wandb_name += f"-c{args.anomaly_iforest_c}"

    if "ANOM_ALGO_SOURCE" in os.environ and hasattr(args, "anomaly_algo_source"):
        args.anomaly_algo_source = os.environ["ANOM_ALGO_SOURCE"]
        args.wandb_name = args.wandb_name.replace(args.anomaly_algo,
                                                  f"{args.anomaly_algo}_{args.anomaly_algo_source[0]}")

    iae = False
    if "AEC_IAE_ROUND_START" in os.environ and hasattr(args, "aec_iae_round_start"):
        args.aec_iae_round_start = int(os.environ["AEC_IAE_ROUND_START"])
        iae = True
    if "AEC_IAE_NU" in os.environ and hasattr(args, "aec_iae_nu"):
        args.aec_iae_nu = float(os.environ["AEC_IAE_NU"])
        iae = True
    if "AEC_IAE_LAMBDA" in os.environ and hasattr(args, "aec_iae_lamda"):
        args.aec_iae_lamda = float(os.environ["AEC_IAE_LAMBDA"])
        iae = True
    if iae:
        args.wandb_name += f"-iaev4({args.aec_iae_round_start},{args.aec_iae_nu},{args.aec_iae_lamda})"

    if "CLIENT_OPTIMIZER" in os.environ:
        args.client_optimizer = os.environ["CLIENT_OPTIMIZER"]
        args.wandb_name += f"-{args.client_optimizer}"
    if "CLIENT_LR" in os.environ:
        args.learning_rate = float(os.environ["CLIENT_LR"])
    if "CLIENT_WD" in os.environ:
        args.weight_decay = float(os.environ["CLIENT_WD"])

    if "CLIENT_NOISE_TYPE" in os.environ:
        DataValArgs.client_noise_type = os.environ["CLIENT_NOISE_TYPE"]
        args.client_noise_type = DataValArgs.client_noise_type
        args.wandb_name += f"-{DataValArgs.client_noise_type}"
    if "CLIENT_LABEL_NOISE" in os.environ:
        DataValArgs.client_label_noise = float(os.environ["CLIENT_LABEL_NOISE"])
        args.client_label_noise = DataValArgs.client_label_noise
        args.wandb_name += f"-nsy{DataValArgs.client_label_noise:.1f}"
    if "CLIENT_NOISE_SOURCE" in os.environ:
        DataValArgs.client_noise_source = os.environ["CLIENT_NOISE_SOURCE"]
        args.client_noise_source = DataValArgs.client_noise_source
        args.wandb_name += f"-{DataValArgs.client_noise_source}"

    dp_enable = 0
    if "DP_ENABLE" in os.environ and hasattr(args, "enable_dp"):
        dp_enable = args.enable_dp = int(os.environ["DP_ENABLE"])
    if dp_enable:
        if "DP_M_TYPE" in os.environ and hasattr(args, "mechanism_type"):
            args.mechanism_type = os.environ["DP_M_TYPE"]
        if "DP_TYPE" in os.environ and hasattr(args, "dp_type"):
            args.dp_type = os.environ["dp_type"]
        if "DP_EPSILON" in os.environ and hasattr(args, "epsilon"):
            args.epsilon = int(os.environ["DP_EPSILON"])
            args.wandb_name += f"-dpe{args.epsilon}"
        if "DP_SENS" in os.environ and hasattr(args, "sensitivity"):
            args.sensitivity = float(os.environ["DP_SENS"])
        if "DP_DELTA" in os.environ and hasattr(args, "delta"):
            args.delta = float(os.environ["DP_DELTA"])

    q_enable = 0
    if "Q_ENABLE" in os.environ and hasattr(args, "enable_quantization"):
        q_enable = args.enable_quantization = int(os.environ["Q_ENABLE"])
    if q_enable:
        if "Q_PRECISION" in os.environ and hasattr(args, "quantize_precision"):
            args.quantize_precision = int(os.environ["Q_PRECISION"])
            args.wandb_name += f"-i{args.quantize_precision}"
        if "Q_ALGO" in os.environ and hasattr(args, "quantize_algo"):
            args.quantize_algo = os.environ["Q_ALGO"]
            args.wandb_name += f"-{args.quantize_algo}"
        if "Q_OP" in os.environ and hasattr(args, "quantize_op"):
            args.quantize_op = os.environ["Q_OP"]
            args.wandb_name += f"-{args.quantize_op}"
        if "Q_SCHEDULE" in os.environ and hasattr(args, "quantize_schedule"):
            args.quantize_schedule = os.environ["Q_SCHEDULE"]
            args.wandb_name += f"-q{args.quantize_schedule[:3]}"
        if "Q_SCHEDULE_MIN_B" in os.environ and hasattr(args, "quantize_schedule_min_b"):
            args.quantize_schedule_min_b = int(os.environ["Q_SCHEDULE_MIN_B"])
            args.wandb_name += f"-qb{args.quantize_schedule_min_b}"
        if "Q_DYNAMIC_ENABLE" in os.environ and hasattr(args, "quantize_dyn"):
            args.quantize_dyn = int(os.environ["Q_DYNAMIC_ENABLE"])
            if args.quantize_dyn:
                args.wandb_name += f"-qdyn"
                if "Q_DYNAMIC_HS_W" in os.environ and hasattr(args, "quantize_dyn_hs_w"):
                    args.quantize_dyn_hs_w = float(os.environ["Q_DYNAMIC_HS_W"])
                    args.wandb_name += f"-hs{args.quantize_dyn_hs_w}"
                if "Q_DYNAMIC_SS_W" in os.environ and hasattr(args, "quantize_dyn_ss_w"):
                    args.quantize_dyn_ss_w = float(os.environ["Q_DYNAMIC_SS_W"])
                    args.wandb_name += f"-ss{args.quantize_dyn_ss_w}"

    if "GRAD_CLIP" in os.environ and hasattr(args, "gradient_clip_t"):
        args.gradient_clip_t = float(os.environ["GRAD_CLIP"])
        args.wandb_name += f"-gc{args.gradient_clip_t}"
        if "GRAD_CLIP_NORM" in os.environ and hasattr(args, "gradient_clip_norm"):
            args.gradient_clip_norm = int(os.environ["GRAD_CLIP_NORM"])
            args.wandb_name += f"-gn{args.gradient_clip_norm}"

    if "TEST_FREQ" in os.environ and hasattr(args, "frequency_of_the_test"):
        args.frequency_of_the_test = int(os.environ["TEST_FREQ"])
        print("Test frequency is overrided to:", args.frequency_of_the_test)

    manage_profiling_args(args)

    update_client_id_list(args)

    mlops.init(args)

    logging.info("==== args = {}".format(vars(args)))
    return args


def init_simulation_mpi(args):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    world_size = comm.Get_size()
    args.comm = comm
    args.process_id = process_id
    args.worker_num = world_size
    if process_id == 0:
        args.role = "server"
    return args


def init_simulation_sp(args):
    return args


def init_simulation_nccl(args):
    return


def manage_profiling_args(args):
    if not hasattr(args, "sys_perf_profiling"):
        args.sys_perf_profiling = True
    if not hasattr(args, "sys_perf_profiling"):
        args.sys_perf_profiling = True

    if args.sys_perf_profiling:
        from .core.mlops.mlops_profiler_event import MLOpsProfilerEvent

        MLOpsProfilerEvent.enable_sys_perf_profiling()

    if args.enable_wandb:
        wandb_only_server = getattr(args, "wandb_only_server", None)
        if (wandb_only_server and args.rank == 0 and args.process_id == 0) or not wandb_only_server:
            wandb_entity = getattr(args, "wandb_entity", None)
            if wandb_entity is not None:
                wandb_args = {
                    "entity": args.wandb_entity,
                    "project": args.wandb_project,
                    "config": args,
                }
            else:
                wandb_args = {
                    "project": args.wandb_project,
                    "config": args,
                }

            if hasattr(args, "wandb_name"):
                wandb_args["name"] = args.wandb_name

            if hasattr(args, "wandb_group_id"):
                # wandb_args["group"] = args.wandb_group_id
                wandb_args["group"] = "Test1"
                wandb_args["name"] = f"Client {args.rank}"
                wandb_args["job_type"] = str(args.rank)

            import wandb

            wandb.init(**wandb_args)

            from .core.mlops.mlops_profiler_event import MLOpsProfilerEvent

            MLOpsProfilerEvent.enable_wandb_tracking()


def manage_cuda_rpc_args(args):
    if (not hasattr(args, "enable_cuda_rpc")) or (not args.using_gpu):
        args.enable_cuda_rpc = False

    if args.enable_cuda_rpc and args.backend != "TRPC":
        args.enable_cuda_rpc = False
        print("Argument enable_cuda_rpc is ignored. Cuda RPC only works with TRPC backend.")

    # When Cuda RPC is not used, tensors should be moved to cpu before transfer with TRPC
    if (not args.enable_cuda_rpc) and args.backend == "TRPC":
        args.cpu_transfer = True
    else:
        args.cpu_transfer = False

    # Valudate arguments related to cuda rpc
    if args.enable_cuda_rpc:
        if not hasattr(args, "cuda_rpc_gpu_mapping"):
            raise Exception("Invalid config. cuda_rpc_gpu_mapping is required when enable_cuda_rpc=True")
        assert type(args.cuda_rpc_gpu_mapping) is dict, "Invalid cuda_rpc_gpu_mapping type. Expected dict"
        assert (
                len(args.cuda_rpc_gpu_mapping) == args.worker_num + 1
        ), f"Invalid cuda_rpc_gpu_mapping. Expected list of size {args.worker_num + 1}"

    print(f"cpu_transfer: {args.cpu_transfer}")
    print(f"enable_cuda_rpc: {args.enable_cuda_rpc}")


def manage_mpi_args(args):
    if hasattr(args, "backend") and args.backend == "MPI":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        process_id = comm.Get_rank()
        world_size = comm.Get_size()
        args.comm = comm
        args.rank = process_id
        if process_id == 0:
            args.role = "server"
        # args.worker_num = worker_num
        assert args.worker_num + 1 == world_size, f"Invalid number of mpi processes. Expected {args.worker_num + 1}"
    else:
        args.comm = None


def init_cross_silo_horizontal(args):
    args.n_proc_in_silo = 1
    args.proc_rank_in_silo = 0
    manage_mpi_args(args)
    manage_cuda_rpc_args(args)
    args.process_id = args.rank
    return args


def init_cross_silo_hierarchical(args):
    manage_mpi_args(args)
    manage_cuda_rpc_args(args)

    # Set intra-silo arguments
    if args.rank == 0:
        args.n_node_in_silo = 1
        args.n_proc_in_silo = 1
        args.rank_in_node = 0
        args.proc_rank_in_silo = 0
    else:
        # Modify arguments to match info set in env by torchrun
        # Silo Topology

        args.n_proc_in_silo = int(os.environ.get("WORLD_SIZE", 1))

        # Rank in node
        args.rank_in_node = int(os.environ.get("LOCAL_RANK", 0))
        args.process_id = args.rank_in_node

        # Rank in silo (process group)
        args.proc_rank_in_silo = int(os.environ.get("RANK", 0))

        # Process group master endpoint
        args.pg_master_address = os.environ.get("MASTER_ADDR", "127.0.0.1")
        args.pg_master_port = os.environ.get("MASTER_PORT", 29300)

        # Launcher Rendezvous
        if not hasattr(args, "launcher_rdzv_port"):
            args.launcher_rdzv_port = 29400

        if not hasattr(args, "n_node_in_silo"):
            args.n_node_in_silo = 1
        if not (hasattr(args, "n_proc_per_node") and args.n_proc_per_node):
            if args.n_node_in_silo == 1 and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                # Checking if launcher is has spawned enoug processes.
                if gpu_count == args.n_proc_in_silo:
                    print(f"Auto assigning GPU to processes.")
                    args.gpu_id = args.proc_rank_in_silo
                else:
                    args.n_proc_per_node = 1
            else:
                args.n_proc_per_node = 1

    return args


def update_client_id_list(args):
    """
        generate args.client_id_list for CLI mode where args.client_id_list is set to None
        In MLOps mode, args.client_id_list will be set to real-time client id list selected by UI (not starting from 1)
    """
    if not hasattr(args, "using_mlops") or (hasattr(args, "using_mlops") and not args.using_mlops):
        print("args.client_id_list = {}".format(print(args.client_id_list)))
        if args.client_id_list is None or args.client_id_list == "None" or args.client_id_list == "[]":
            if (
                    args.training_type == FEDML_TRAINING_PLATFORM_CROSS_DEVICE
                    or args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO
            ):
                if args.rank == 0:
                    client_id_list = []
                    for client_idx in range(args.client_num_per_round):
                        client_id_list.append(client_idx + 1)
                    args.client_id_list = str(client_id_list)
                    print("------------------server client_id_list = {}-------------------".format(args.client_id_list))
                else:
                    # for the client, we only specify its client id in the list, not including others.
                    client_id_list = []
                    client_id_list.append(args.rank)
                    args.client_id_list = str(client_id_list)
                    print("------------------client client_id_list = {}-------------------".format(args.client_id_list))
            else:
                print(
                    "training_type != FEDML_TRAINING_PLATFORM_CROSS_DEVICE and training_type != FEDML_TRAINING_PLATFORM_CROSS_SILO"
                )
        else:
            print("args.client_id_list is not None")
    else:
        print("using_mlops = true")


def init_cross_device(args):
    args.rank = 0  # only server runs on Python package
    return args


def run_distributed():
    pass


from fedml import device
from fedml import data
from fedml import model
from fedml import mlops

from .arguments import load_arguments

from .launch_simulation import run_simulation

from .launch_cross_silo_horizontal import run_cross_silo_server
from .launch_cross_silo_horizontal import run_cross_silo_client

from .launch_cross_silo_hi import run_hierarchical_cross_silo_server
from .launch_cross_silo_hi import run_hierarchical_cross_silo_client

from .launch_cross_device import run_mnn_server

from .core.common.ml_engine_backend import MLEngineBackend

from .runner import FedMLRunner

__all__ = [
    "MLEngineBackend",
    "device",
    "data",
    "model",
    "mlops",
    "FedMLRunner",
    "run_simulation",
    "run_cross_silo_server",
    "run_cross_silo_client",
    "run_hierarchical_cross_silo_server",
    "run_hierarchical_cross_silo_client",
    "run_mnn_server",
]

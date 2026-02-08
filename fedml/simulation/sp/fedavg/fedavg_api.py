import copy
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import tqdm
import wandb
import yaml

import fedml.utils.quantize as quantize
from fedml import mlops
from fedml.core.data_valuation.anomaly_valuator import AnomalyBasedValuator
from fedml.core.data_valuation.fedbal_data_valuator import FedBalDataValuator
from fedml.core.data_valuation.fedbal_manager import FedBalThreshController
from fedml.ml.trainer.iae_loss import IAE
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client
import fedml.utils.statistics as stats_f


class FedAvgAPI(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.class_num = class_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, args)
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )
        self.fedbal_manager = None
        self.data_val_algo = args.data_val_algo if hasattr(args, "data_val_algo") else None
        if self.data_val_algo is not None and "fedbal" in self.data_val_algo:
            self.fedbal_manager = FedBalThreshController(self.args)

        self.data_val_save_scores = hasattr(args, "data_val_save_scores") and args.data_val_save_scores
        cur_datetime = time.strftime("%Y%m%d-%H%M%S")
        valid_fname = f"{args.wandb_name}".replace("/", "-") \
            .replace("\\", "-") \
            .replace(":", ";") \
            .replace("\"", "'") \
            .replace("*", "~") \
            .replace("?", "!")
        log_dir = os.path.join("./FedMLLogs", f"{valid_fname}-{cur_datetime}")
        os.makedirs(log_dir, exist_ok=True)
        if self.data_val_save_scores:
            self.data_val_save_path = os.path.join(log_dir, "data_val")
            os.makedirs(self.data_val_save_path, exist_ok=True)

        self.quantizer = None
        if hasattr(self.args, "enable_quantization") and self.args.enable_quantization:
            logging.info(f"Quantization has been enabled !")
            self.quantizer = quantize.Quantizer(args)

        self.best_metrics = {"test_acc": 0, "test_bacc": 0, "round": 0, "metrics": {}}
        self.model_save_path = os.path.join(log_dir, "weights")
        os.makedirs(self.model_save_path, exist_ok=True)
        with open(os.path.join(log_dir, "config.yaml"), "w") as f:
            f.write(yaml.dump(self.args))
        print("Log dir created at: ", log_dir)

        self.data_val_global = hasattr(self.args, "data_val_type") and (self.args.data_val_type == "global")
        if self.data_val_global:
            self.data_val_update_period = self.args.data_val_update_period if hasattr(self.args,
                                                                                      "data_val_update_period") else 1
            from fedml.core.data_valuation.data_val_creator import create_data_valuator
            self.data_valuator = create_data_valuator(self.model_trainer, self.args)
        self.data_val_round_start = 1e10
        if hasattr(self.args, "data_val_round_start"):
            self.data_val_round_start = self.args.data_val_round_start
            if self.data_val_round_start < 0:
                self.data_val_round_start = 1e10
            print(f"Data val. round start is set to : {self.data_val_round_start}")

        self.data_val_per_class = self.args.data_val_per_class if hasattr(self.args, "data_val_per_class") else False

        self.aec_iae_round_start = 1e10
        if hasattr(self.args, "aec_iae_round_start"):
            self.aec_iae_round_start = self.args.aec_iae_round_start
            if self.aec_iae_round_start < 0:
                self.aec_iae_round_start = 1e10
            self.aec_iae_nu = self.args.aec_iae_nu
            self.aec_iae_lamda = self.args.aec_iae_lamda
            self.iae_loss = IAE(self.aec_iae_nu, self.aec_iae_lamda, class_num, device)
            print(f"IAE round start is set to : {self.aec_iae_round_start}")

        if hasattr(self.args, "best_metric") and len(self.args.best_metric) > 0:
            print("Using " + self.args.best_metric + " to track the best model !")

    def _setup_clients(
            self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                copy.deepcopy(self.model_trainer)  # self.model_trainer
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    @staticmethod
    def train_client_worker(client: Client, w_global, global_qdata, return_iae=False):
        try:
            logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)
            threading.currentThread().setName(f"Client :{client.client_idx}")
            w, local_qdata = client.train(w_global, global_qdata)
            return client.get_sample_number(), \
                copy.deepcopy(w), \
                copy.deepcopy(local_qdata) if local_qdata is not None else None, \
                copy.deepcopy(client.get_data_valuator()), \
                copy.deepcopy(client.get_trainer().get_iae_algo().get_dists()) if return_iae else None
        except Exception as ex:
            logging.exception(ex)

    def save_data_val_scores(self, round_ix):
        try:
            # wlog = {}
            for client in self.client_list:
                save_path = os.path.join(self.data_val_save_path, f"round_{round_ix}")
                os.makedirs(save_path, exist_ok=True)
                vals = client.get_data_valuator().get_last_scores()
                np.save(os.path.join(save_path, f"{client.client_idx}"), vals)
        except Exception as ex:
            print("Failed to save data values: ", ex)

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        q_enable = self.quantizer is not None
        w_global = self.model_trainer.get_model_params()
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        max_workers = self.args.client_max_workers if hasattr(self.args, "client_max_workers") else None
        worker_pool = ProcessPoolExecutor(max_workers=max_workers)
        total_com_up_bytes = 0  # to measure size of only model transfer
        total_com_down_bytes = 0  # to measure size of only model transfer
        try:
            for round_idx in range(self.args.comm_round):

                logging.info("################Communication round : {}".format(round_idx))
                """
                for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
                Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
                """
                client_indexes = self._client_sampling(
                    round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )

                max_n_samples = 0
                for idx, client in enumerate(self.client_list):
                    # update dataset
                    client_idx = client_indexes[idx]
                    client.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                    )
                    max_n_samples = max(client.get_sample_number(), max_n_samples)
                    client.set_round_index(round_idx)
                    if self.fedbal_manager is not None:
                        dvaluer: FedBalDataValuator = client.get_data_valuator()
                        dvaluer.update_threshold(self.fedbal_manager.get_thresh())

                    if self.data_val_global and round_idx > self.data_val_round_start:
                        client.set_data_valuator(copy.deepcopy(self.data_valuator))

                    if round_idx > self.aec_iae_round_start:
                        client.get_trainer().set_iae_algo(copy.deepcopy(self.iae_loss))

                logging.info("Round {}: Max # samples: {}".format(round_idx, max_n_samples))
                for idx, client in enumerate(self.client_list):
                    client.set_max_samples(max_n_samples)

                global_qdata = None
                if q_enable:
                    if hasattr(self.args, "quantize_schedule") and self.args.quantize_schedule == "cosine":
                        b = stats_f.consine_schedule(round_idx, self.args.comm_round, 32,
                                                     self.args.quantize_schedule_min_b)
                        b = int(round(b))
                        self.quantizer.set_bit_length(b)
                        logging.info("Server: Bit-length: {}".format(b))
                        if self.args.enable_wandb:
                            wandb.log({"Train/QuantServerBits": b, "round": round_idx})
                    st = time.time()
                    w_global, global_qdata = self.quantizer.quantize_state_dict(w_global, err=True)
                    et = time.time()
                    logging.info("Server: Quantization has been completed in {} secs !".format((et - st)))
                    if self.args.enable_wandb:
                        wandb.log({"Train/QuantServerQErr": global_qdata["q_err"], "round": round_idx})

                update_dval = self.data_val_global and \
                              self.data_val_algo == "anomaly" and \
                              round_idx >= self.data_val_round_start and \
                              (round_idx == self.data_val_round_start or
                               round_idx % self.data_val_update_period == 0)
                iae_dists = {}
                w_locals = []
                local_dvals = []
                n_selected, n_total = 0, 0
                mean_q_err = 0.0
                mean_q_bits = 0.0
                if max_workers > 1:
                    logging.info(f"Waiting for {len(self.client_list)} client processes...")
                    return_iae = round_idx > self.aec_iae_round_start
                    client_futures = (
                        (client, worker_pool.submit(FedAvgAPI.train_client_worker,
                                                    client=copy.deepcopy(client),
                                                    w_global=copy.deepcopy(w_global),
                                                    global_qdata=copy.deepcopy(global_qdata),
                                                    return_iae=copy.deepcopy(return_iae)))
                        for client in self.client_list
                    )
                    tqdm_iters = tqdm.tqdm(total=len(self.client_list))
                    for client, c_ftr in client_futures:
                        tqdm_iters.set_description(f"Client {client.client_idx}")
                        n_total += client.get_total_sample_number()
                        if q_enable:
                            total_com_up_bytes += stats_f.get_state_dict_size(w_global,
                                                                              self.quantizer.get_bit_length()) / 8
                        else:
                            total_com_up_bytes += stats_f.get_state_dict_size(w_global)
                        n_train_samp, local_w, qdata, dvaluer, iae_d = c_ftr.result()
                        if q_enable:
                            total_com_down_bytes += stats_f.get_state_dict_size(local_w, qdata["q_bits"]) / 8
                            mean_q_err += (qdata["q_err"] if "q_err" in qdata else 0.0)
                            mean_q_bits += qdata["q_bits"]
                            local_w = self.quantizer.dequantize_state_dict(local_w, qdata)
                        else:
                            total_com_down_bytes += stats_f.get_state_dict_size(local_w)
                        w_locals.append((n_train_samp, local_w))
                        client.set_data_valuator(dvaluer)
                        tqdm_iters.update()
                        if update_dval:
                            dvaluer: AnomalyBasedValuator = client.get_data_valuator()
                            local_dvals.append(dvaluer.get_sample_dvals())
                        if round_idx > self.aec_iae_round_start:
                            # Collect distances from clients
                            for lbl, dist in iae_d.items():
                                if lbl in iae_dists:
                                    iae_dists[lbl] = np.concatenate([iae_dists[lbl], dist])
                                else:
                                    iae_dists[lbl] = dist
                        n_selected += n_train_samp
                        logging.info(
                            f"Client {client.client_idx}: Trained {n_train_samp}/{client.get_total_sample_number()}")

                    tqdm_iters.close()
                    del client_futures
                else:
                    logging.info(f"Waiting for {len(self.client_list)} clients...")
                    tqdm_iters = tqdm.tqdm(self.client_list, total=len(self.client_list))
                    for client in tqdm_iters:
                        tqdm_iters.set_description(f"Client {client.client_idx}")
                        n_total += client.get_total_sample_number()
                        if q_enable:
                            total_com_up_bytes += stats_f.get_state_dict_size(w_global,
                                                                              self.quantizer.get_bit_length()) / 8
                        else:
                            total_com_up_bytes += stats_f.get_state_dict_size(w_global)
                        local_w, qdata = client.train(w_global, global_qdata)
                        if q_enable:
                            total_com_down_bytes += stats_f.get_state_dict_size(local_w, qdata["q_bits"]) / 8
                            mean_q_err += (qdata["q_err"] if "q_err" in qdata else 0.0)
                            mean_q_bits += qdata["q_bits"]
                            local_w = self.quantizer.dequantize_state_dict(local_w, qdata)
                        else:
                            total_com_down_bytes += stats_f.get_state_dict_size(local_w)
                        w_locals.append((client.get_sample_number(), local_w))
                        n_selected += client.get_sample_number()
                        if update_dval:
                            dvaluer: AnomalyBasedValuator = client.get_data_valuator()
                            local_dvals.append(dvaluer.get_sample_dvals())
                        if round_idx > self.aec_iae_round_start:
                            # Collect distances from clients
                            d: dict = client.get_trainer().get_iae_algo().get_dists()
                            for lbl, dist in d.items():
                                if lbl in iae_dists:
                                    iae_dists[lbl] = np.concatenate([iae_dists[lbl], dist])
                                else:
                                    iae_dists[lbl] = dist

                        logging.info(
                            f"Client {client.client_idx}: Trained {client.get_sample_number()}/{client.get_total_sample_number()}")

                    tqdm_iters.close()
                mean_q_err = mean_q_err / len(self.client_list)
                mean_q_bits = mean_q_bits / len(self.client_list)

                if round_idx > self.aec_iae_round_start:
                    self.iae_loss.update_r(iae_dists)
                    logging.info(f"Updated iae radius !")

                if self.args.enable_wandb:
                    wandb.log({"Train/SelectedData": n_selected, "round": round_idx})
                    wandb.log({"Train/TotalData": n_total, "round": round_idx})
                    wandb.log({"Train/UnselectedData": n_total - n_selected, "round": round_idx})
                    wandb.log({"Train/TotalComBytes": total_com_up_bytes + total_com_down_bytes, "round": round_idx})
                    wandb.log({"Train/TotalComUpBytes": total_com_up_bytes, "round": round_idx})
                    wandb.log({"Train/TotalComDownBytes": total_com_down_bytes, "round": round_idx})
                    if q_enable:
                        wandb.log({"Train/QuantMeanErr": mean_q_err, "round": round_idx})
                        wandb.log({"Train/QuantMeanBits": mean_q_bits, "round": round_idx})

                logging.info(f"Total Selected Data for {round_idx}: {n_selected} / {n_total}")

                # train odet model right before data val starts and train periodically afterwards
                if update_dval:
                    if self.data_val_per_class:
                        combined_vals = [[] for _ in range(10)]
                        for vals in local_dvals:
                            for i in range(self.class_num):
                                combined_vals[i].extend(vals[i])
                        self.data_valuator.train_odet_model(combined_vals)
                    else:
                        local_dvals = np.concatenate(local_dvals, axis=0)
                        self.data_valuator.train_odet_model(local_dvals)
                    if self.args.enable_wandb:
                        wandb.log({"Train/OdetTrainSize": len(local_dvals), "round": round_idx})
                    logging.info(f"Updated global data valuator in round {round_idx}")

                # update fedbal
                if self.fedbal_manager is not None:
                    logging.info(f"Updating fedbal threshold...")
                    for client in self.client_list:
                        dvaluer: FedBalDataValuator = client.get_data_valuator()
                        self.fedbal_manager.add_meta(dvaluer.get_meta())
                    newt = self.fedbal_manager.update(round_idx)
                    logging.info(f"New fedbal threshold at round {round_idx}: {newt}")

                logging.info(f"Aggregating {len(w_locals)} models...")
                # update global weights
                mlops.event("agg", event_started=True, event_value=str(round_idx))
                w_global = self._aggregate(w_locals)
                self.model_trainer.set_model_params(w_global)
                mlops.event("agg", event_started=False, event_value=str(round_idx))

                if self.data_val_save_scores:
                    self.save_data_val_scores(round_idx)

                if round_idx == self.aec_iae_round_start:
                    reps, lbls = self.model_trainer.extract_features(self.test_global, self.device, self.args)
                    self.iae_loss.update_center(reps, lbls)
                    logging.info(f"IAE: Updated class centers !")

                # test results
                # at last round
                if round_idx == self.args.comm_round - 1:
                    self._local_test_on_all_clients(round_idx)
                # per {frequency_of_the_test} round
                elif round_idx % self.args.frequency_of_the_test == 0:
                    if self.args.dataset.startswith("stackoverflow"):
                        self._local_test_on_validation_set(round_idx)
                    else:
                        self._local_test_on_all_clients(round_idx)

                mlops.log_round_info(self.args.comm_round, round_idx)

                torch.save(w_global, os.path.join(self.model_save_path, "latest.pth"))

                del w_locals
        finally:
            mlops.log_training_finished_status()
            mlops.log_aggregation_finished_status()
            worker_pool.shutdown(wait=False)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        if training_num > 0:
            (_, averaged_params) = w_locals[0]
            for k in averaged_params.keys():
                for i in range(0, len(w_locals)):
                    local_sample_number, local_model_params = w_locals[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w
            return averaged_params
        else:
            return self.model_trainer.get_model_params()

    def _local_test_on_all_clients(self, round_idx):

        logging.info(f"################ local_test_on_all_clients for round: {round_idx}")

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        if self.args.model == "aec":
            train_metrics["re_losses"] = []
            train_metrics["ce_losses"] = []
            train_metrics["kl_losses"] = []
            train_metrics["psnr"] = []
            train_metrics["ssim"] = []
            test_metrics["re_losses"] = []
            test_metrics["ce_losses"] = []
            test_metrics["kl_losses"] = []
            test_metrics["psnr"] = []
            test_metrics["ssim"] = []

        for client_idx in tqdm.tqdm(range(self.args.client_num_in_total)):
            train_data = self.train_data_local_dict[client_idx]
            train_local_metrics = self.model_trainer.test(train_data, self.device, self.args)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))
            if self.args.model == "aec":
                train_metrics["re_losses"].append(copy.deepcopy(train_local_metrics["test_re_loss"]))
                train_metrics["ce_losses"].append(copy.deepcopy(train_local_metrics["test_ce_loss"]))
                train_metrics["psnr"].append(copy.deepcopy(train_local_metrics["test_psnr"]))
                train_metrics["ssim"].append(copy.deepcopy(train_local_metrics["test_ssim"]))
                if self.args.aec_use_vae == 1:
                    train_metrics["kl_losses"].append(copy.deepcopy(train_local_metrics["test_kl_loss"]))

        test_local_metrics = self.model_trainer.test(self.test_global, self.device, self.args)
        test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
        test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
        test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))
        if self.args.model == "aec":
            test_metrics["re_losses"].append(copy.deepcopy(test_local_metrics["test_re_loss"]))
            test_metrics["ce_losses"].append(copy.deepcopy(test_local_metrics["test_ce_loss"]))
            test_metrics["psnr"].append(copy.deepcopy(test_local_metrics["test_psnr"]))
            test_metrics["ssim"].append(copy.deepcopy(test_local_metrics["test_ssim"]))
            if self.args.aec_use_vae == 1:
                test_metrics["kl_losses"].append(copy.deepcopy(test_local_metrics["test_kl_loss"]))

        # test on training dataset
        train_n_samp = sum(train_metrics["num_samples"])
        train_acc = sum(train_metrics["num_correct"]) / train_n_samp
        train_loss = sum(train_metrics["losses"]) / train_n_samp
        # test on test dataset
        test_n_samp = sum(test_metrics["num_samples"])
        test_acc = sum(test_metrics["num_correct"]) / test_n_samp
        test_loss = sum(test_metrics["losses"]) / test_n_samp

        if hasattr(self.args, "best_metric") and len(self.args.best_metric) > 0:
            metric = "test_" + self.args.best_metric
            if metric in test_local_metrics:
                if metric not in self.best_metrics:
                    self.best_metrics[metric] = 0
                best_flag = (self.best_metrics[metric] < test_local_metrics[metric])
            else:
                print("Warning: Best metric is not supported ! :", self.args.best_metric)
                best_flag = (self.best_metrics["test_acc"] < test_acc)
        else:
            best_flag = (self.best_metrics["test_acc"] < test_acc)

        if best_flag:
            self.best_metrics = {
                "test_acc": test_acc,
                "test_bacc": test_local_metrics['test_bacc'],
                "round": round_idx,
                "metrics": test_local_metrics
            }
            torch.save(self.model_trainer.get_model_params(), os.path.join(self.model_save_path, "best.pth"))
            with open(os.path.join(self.model_save_path, "best_metrics.yaml"), "w") as f:
                f.write(yaml.dump(self.best_metrics))
            logging.info(f"Saved the best weights by acc !")

        logging.info(f"Train/Acc: {train_acc}, Train/Loss: {train_loss}")
        logging.info(f"Test/Acc: {test_acc}, Test/Loss: {test_loss}, "
                     f"Test/BACC:{test_local_metrics['test_bacc']}, "
                     f"Test/F1:{test_local_metrics['test_f1']}, "
                     f"Test/Precision: {test_local_metrics['test_precision']}, "
                     f"Test/Recall: {test_local_metrics['test_recall']}")

        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            wandb.log({"Test/F1": test_local_metrics['test_f1'], "round": round_idx})
            wandb.log({"Test/Precision": test_local_metrics['test_precision'], "round": round_idx})
            wandb.log({"Test/Recall": test_local_metrics['test_recall'], "round": round_idx})
            wandb.log({"Test/BACC": test_local_metrics['test_bacc'], "round": round_idx})

        if self.args.model == "aec":
            train_ssim = sum(train_metrics["ssim"]) / train_n_samp
            train_psnr = sum(train_metrics["psnr"]) / train_n_samp
            train_re_loss = sum(train_metrics["re_losses"]) / train_n_samp
            train_ce_loss = sum(train_metrics["ce_losses"]) / train_n_samp

            logging.info(f"Train/PSNR: {train_psnr}, Train/SSIM: {train_ssim}")
            logging.info(f"Train/RELoss: {train_re_loss}, Train/CELoss: {train_ce_loss}")

            if self.args.enable_wandb:
                wandb.log({"Train/SSIM": train_ssim, "round": round_idx})
                wandb.log({"Train/PSNR": train_psnr, "round": round_idx})
                wandb.log({"Train/RecLoss": train_re_loss, "round": round_idx})
                wandb.log({"Train/CELoss": train_ce_loss, "round": round_idx})
            if self.args.aec_use_vae == 1:
                train_kl_loss = sum(train_metrics["kl_losses"]) / train_n_samp
                logging.info(f"Train/KLLoss: {train_kl_loss}")
                if self.args.enable_wandb:
                    wandb.log({"Train/KLLoss": train_kl_loss, "round": round_idx})

            test_ssim = sum(test_metrics["ssim"]) / test_n_samp
            test_psnr = sum(test_metrics["psnr"]) / test_n_samp
            test_re_loss = sum(test_metrics["re_losses"]) / test_n_samp
            test_ce_loss = sum(test_metrics["ce_losses"]) / test_n_samp
            logging.info(f"Test/PSNR: {test_psnr}, Test/SSIM: {test_ssim}")
            logging.info(f"Test/RELoss: {test_re_loss}, Test/CELoss: {test_ce_loss}")
            if self.args.enable_wandb:
                wandb.log({"Test/SSIM": test_ssim, "round": round_idx})
                wandb.log({"Test/PSNR": test_psnr, "round": round_idx})
                wandb.log({"Test/RecLoss": test_re_loss, "round": round_idx})
                wandb.log({"Test/CELoss": test_ce_loss, "round": round_idx})
            if self.args.aec_use_vae == 1:
                test_kl_loss = sum(test_metrics["kl_losses"]) / test_n_samp
                logging.info(f"Test/KLLoss: {test_kl_loss}")
                if self.args.enable_wandb:
                    wandb.log({"Test/KLLoss": test_kl_loss, "round": round_idx})

        if round_idx == self.args.comm_round - 1:
            try:
                wandb.run.summary.update(self.best_metrics)
            except Exception as ex:
                print("Failed to report best metrics to wandb: ", ex)

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Pre": test_pre, "round": round_idx})
                wandb.log({"Test/Rec": test_rec, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Pre": test_pre, "round": round_idx})
            mlops.log({"Test/Rec": test_rec, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)

import logging
import time

import torch
from torch.utils.data.dataloader import DataLoader

import fedml.utils.quantize as quantize
from fedml.core import ClientTrainer
from fedml.core.data_valuation.data_val_creator import create_data_valuator
import fedml.utils.statistics as stats


class Client:
    def __init__(
            self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.total_local_sample_number = local_sample_number
        self.local_sample_number = local_sample_number
        self.max_n_samples_cur_round = 0
        self.args = args
        self.device = device
        self.model_trainer: ClientTrainer = model_trainer
        self.data_valuator = create_data_valuator(model_trainer, args)
        if self.data_valuator is not None:
            logging.info(
                'Client {}: Using data valuator {}'.format(self.client_idx,
                                                           self.data_valuator.__class__.__name__))
            if isinstance(local_training_data, DataLoader):
                self.local_training_data = list(local_training_data)
            if isinstance(local_test_data, DataLoader):
                self.local_test_data = list(local_test_data)

        if self.args.dataset == "chexpert":
            self.max_entropy = stats.calculate_entropy([0.5] * self.args.class_num)
        else:
            self.max_entropy = stats.calculate_entropy([1 / self.args.class_num] * self.args.class_num)

        self.quantizer = None
        if hasattr(self.args, "enable_quantization") and self.args.enable_quantization:
            self.quantizer = quantize.Quantizer(args)

        self.round_idx = -1
        self.data_val_round_start = -1
        if hasattr(self.args, "data_val_round_start"):
            self.data_val_round_start = self.args.data_val_round_start

        self.data_val_algo = args.data_val_algo if hasattr(args, "data_val_algo") else None
        self.data_val_type = self.args.data_val_type if hasattr(self.args, "data_val_type") else "local"
        # self.data_val_update_period = self.args.data_val_update_period if hasattr(self.args,
        #                                                                           "data_val_update_period") else 1
        # self.last_sample_losses = []

    def get_data_valuator(self):
        return self.data_valuator

    def set_data_valuator(self, dval):
        self.data_valuator = dval
        self.data_valuator.set_trainer(self.get_trainer())

    def set_round_index(self, r):
        self.round_idx = r

    def get_trainer(self):
        return self.model_trainer

    def set_max_samples(self, val):
        self.max_n_samples_cur_round = val

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx

        if self.data_valuator is not None:
            if isinstance(local_training_data, DataLoader):
                local_training_data = list(local_training_data)
            if isinstance(local_test_data, DataLoader):
                local_test_data = list(local_test_data)

        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.total_local_sample_number = local_sample_number
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_total_sample_number(self):
        return self.total_local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global, q_data=None):
        q_enable = self.quantizer is not None
        if q_enable:
            st = time.time()
            w_global = quantize.Quantizer.dequantize_state_dict(w_global, q_data)
            et = time.time()
            logging.info("Client {}: De-Quantization Time : {} sec".format(self.client_idx, (et - st)))

        self.model_trainer.set_model_params(w_global)

        train_data = self.local_training_data
        if self.data_valuator is not None and self.round_idx >= self.data_val_round_start:
            # collect losses to train central odet model in server right before the data val starts
            if self.data_val_type == "global" and self.data_val_algo == "anomaly" and \
                    self.round_idx == self.data_val_round_start:
                self.data_valuator.compute_features(train_data, self.device)
            if self.round_idx > self.data_val_round_start:
                train_data = self.data_valuator.pick_samples(train_data, self.local_test_data, self.device)
                train_data_size = 0
                for (_, labels) in train_data:
                    train_data_size += len(labels)
                self.local_sample_number = train_data_size

                if self.local_sample_number > 0:
                    logging.info(
                        'Client {}: Training for filtered data: {}/{}'.format(self.client_idx,
                                                                              self.local_sample_number,
                                                                              self.total_local_sample_number))
                    self.model_trainer.on_before_local_training(train_data, self.device, self.args)
                    self.model_trainer.train(train_data, self.device, self.args)
                    self.model_trainer.on_after_local_training(train_data, self.device, self.args)
                else:
                    logging.info('Client {}: All data is dirty ! Skipping the training...'.format(self.client_idx))
        else:
            self.model_trainer.on_before_local_training(train_data, self.device, self.args)
            self.model_trainer.train(train_data, self.device, self.args)
            self.model_trainer.on_after_local_training(train_data, self.device, self.args)

        weights = self.model_trainer.get_model_params()

        if q_enable:
            client_score = 1.0
            if hasattr(self.args, "quantize_dyn") and self.args.quantize_dyn:
                # apd = self.mean_pairwise_cos_dist(self.model_trainer.get_embeddings())
                # logging.info("Client {}: EOD: {}".format(self.client_idx, eod))
                # hs = self.compute_homogeneity_score()
                # client_score = self.args.quantize_dyn_acd_w * apd + hs * self.args.quantize_dyn_hs_w
                # logging.info("Client {}: APD: {}, HS: {}".format(self.client_idx, apd, hs))
                hs = self.compute_homogeneity_score()
                ss = self.get_sample_number() / self.max_n_samples_cur_round
                client_score = hs * self.args.quantize_dyn_hs_w + ss * self.args.quantize_dyn_ss_w
                logging.info("Client {}: Homogeneity Score: {}".format(self.client_idx, hs))

            if hasattr(self.args, "quantize_schedule") and self.args.quantize_schedule == "cosine":
                b = stats.consine_schedule(self.round_idx, self.args.comm_round, 32, self.args.quantize_schedule_min_b,
                                           alpha=client_score)
                b = int(round(b))
                self.quantizer.set_bit_length(b)
                logging.info("Client {}: Bit-length: {}".format(self.client_idx, b))

            st = time.time()
            weights, q_data = self.quantizer.quantize_state_dict(weights, err=True)
            et = time.time()
            logging.info("Client {}: Quantization Time : {} sec".format(self.client_idx, (et - st)))

        return weights, q_data

    def compute_homogeneity_score(self):
        train_data = self.local_training_data
        if self.args.dataset == "chexpert":
            sums = []
            count = 0
            for (_, l) in train_data:
                sums.append(l.sum(dim=0).unsqueeze(0))
                count += len(l)
            probs = (torch.cat(sums, dim=0).sum(dim=0) / (count + 1e-15)).numpy()
        else:
            all_labels = torch.cat([l for (_, l) in train_data], dim=0).numpy()
            probs = stats.compute_probs(all_labels)
        return stats.calculate_entropy(probs) / self.max_entropy

    # def mean_pairwise_cos_dist(self, embeds):
    #     return stats.mean_pairwise_cos_dist(embeds)

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

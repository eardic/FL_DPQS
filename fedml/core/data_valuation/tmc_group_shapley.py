import copy
import random
import sys
import threading

import numpy as np
import tqdm
from torch.utils.data.dataloader import DataLoader

from fedml.core.data_valuation.data_valuator import DataValuator


class TMCGroupShapley(DataValuator):
    def __init__(self, trainer, args) -> None:
        super().__init__(trainer, args)

    def _tol_mean_score(self, train, test, device, test_len):
        """Computes the average performance and its error using bagging."""
        try:
            self.reset_model()
            self.trainer.train(train, device, self.args)
            scores = []
            for _ in range(self.args.tmc_max_iter):
                test_data = self._shuffle(test, test_len)
                ref_test_metrics = self.trainer.test(test_data, device, self.args)
                del test_data
                scores.append(ref_test_metrics["test_" + self.args.data_val_metric])
            return np.std(scores), np.mean(scores)
        finally:
            self.reset_model()

    def _error(self, mem, max_iter):
        if len(mem) < max_iter:
            return 1.0
        all_vals = (np.cumsum(mem, 0) / np.reshape(np.arange(1, len(mem) + 1), (-1, 1)))[-max_iter:]  # all phis
        # converge check
        div_res = np.abs(all_vals[-max_iter:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + sys.float_info.epsilon)
        errors = np.mean(div_res, -1)
        errors = np.max(errors)
        return errors

    def _gix2ix_rand(self, g_idx, train_len):
        return min((g_idx * self.args.tmc_group_size) + random.randint(0, self.args.tmc_group_size - 1), train_len - 1)

    def _single_iter(self, train, test, device, ref_score, train_len, tol, mean_score):
        """Runs one iteration of TMC-Shapley algorithm."""
        idxs = np.random.permutation(train_len // self.args.tmc_group_size)
        marginal_contribs = np.zeros(train_len // self.args.tmc_group_size)

        truncation_counter = 0
        new_score = ref_score
        batch_size = len(train[0][0])

        data_list, label_list = [], []
        for i, g_idx in enumerate(idxs):
            old_score = new_score
            idx = self._gix2ix_rand(g_idx)
            bix, eix = idx // batch_size, idx % batch_size
            x, lbls = train[bix]
            data_tensor, lbl_tensor = x[eix], lbls[eix]
            data_list.append(data_tensor.unsqueeze(0))
            label_list.append(lbl_tensor.unsqueeze(0))
            batched_tuples = self._to_batched_tuples(data_list, label_list, batch_size)
            self.reset_model()
            self.trainer.train(batched_tuples, device, self.args)
            metrics = self.trainer.test(test, device, self.args)
            new_score = metrics["test_" + self.args.data_val_metric]
            marginal_contribs[idx] = new_score - old_score
            del batched_tuples
            distance_to_full_score = np.abs(new_score - mean_score)
            if distance_to_full_score <= tol * mean_score:
                truncation_counter += 1
                if truncation_counter > self.args.tmc_max_truncate:
                    break
            else:
                truncation_counter = 0

        return marginal_contribs

    def pick_samples_by_value(self, train, scores):
        if isinstance(train, DataLoader):
            train = list(train)
        s_scores_ix = np.argsort(scores)
        bad_count = int(np.sum(scores < 0) * self.args.remove_bad_ratio)
        selected_gixs = s_scores_ix[bad_count:]
        batch_size = len(train[0][0])
        data_list, label_list = [], []
        for s_gix in selected_gixs:
            for g in range(self.args.tmc_group_size):
                ix = (s_gix * self.args.tmc_group_size) + g
                bix, eix = ix // batch_size, ix % batch_size
                x, lbls = train[bix]
                if eix < len(x):
                    data_tensor, lbl_tensor = x[eix], lbls[eix]
                    data_list.append(data_tensor.unsqueeze(0))
                    label_list.append(lbl_tensor.unsqueeze(0))
                else:
                    continue
        return self._to_batched_tuples(data_list, label_list, batch_size)

    def compute_values(self, train, test, device):
        try:
            if self.args.remove_bad_ratio <= 0:
                return train

            if isinstance(train, DataLoader):
                train = list(train)
            if isinstance(test, DataLoader):
                test = list(test)

            self.trainer.set_verbose(False)

            train_len = np.sum([len(labels) for (x, labels) in train])
            test_len = np.sum([len(labels) for (x, labels) in test])

            metrics = self.trainer.test(test, device, self.args)
            ref_score = metrics[f"test_{self.args.data_val_metric}"]

            cur_thread = threading.currentThread()
            print(f"{cur_thread.getName()}, reference score: ", ref_score)

            mem_tmc = np.zeros((0, train_len // self.args.tmc_group_size))

            tol, mean_score = self._tol_mean_score(train, test, device, test_len)
            print(f"{cur_thread.getName()}, tol : {tol}, mean score: {mean_score}")

            while True:
                tqdm_iter = tqdm.tqdm(range(self.args.tmc_max_iter), desc=f"{cur_thread.getName()}")
                for _ in tqdm_iter:
                    marginals = self._single_iter(train, test, device,
                                                  ref_score, train_len,
                                                  tol, mean_score)
                    # dims: (iter, # train samples)
                    mem_tmc = np.concatenate([mem_tmc,
                                              np.reshape(marginals, (1, -1))])
                tmc_err = self._error(mem_tmc, self.args.tmc_max_iter)
                if tmc_err < self.args.tmc_error:
                    print(f"{cur_thread.getName()}> converged with err: {tmc_err:.5f}")
                    return np.mean(mem_tmc, 0)
        finally:
            self.reset_model()
            self.trainer.set_verbose(True)

import copy
import sys
import threading

import numpy as np
from torch.utils.data.dataloader import DataLoader

from fedml.core.data_valuation.data_valuator import DataValuator


class GShapley(DataValuator):
    def __init__(self, trainer, args) -> None:
        super().__init__(trainer, args)
        self.train_args = copy.deepcopy(args)
        self.train_args.epochs = 1

    def _find_best_lr(self, train, test, device):
        """Computes the average performance and its error using bagging."""
        best_lr = self.train_args.learning_rate
        best_acc = 0
        print("Finding the best LR...", file=sys.stderr)
        for i in np.arange(1, 5.1, 0.5):
            self.train_args.learning_rate = 10 ** (-i)
            self.reset_model()
            self.trainer.train(train, device, self.train_args)
            ref_test_metrics = self.trainer.test(test, device, self.train_args)
            if ref_test_metrics["test_" + self.train_args.data_val_metric] > best_acc:
                best_acc = ref_test_metrics["test_" + self.train_args.data_val_metric]
                best_lr = self.train_args.learning_rate
        self.reset_model()
        self.train_args.learning_rate = best_lr
        return best_lr

    def _error(self, mem, max_iter):
        if len(mem) < max_iter:
            return 1.0
        all_vals = (np.cumsum(mem, 0) / np.reshape(np.arange(1, len(mem) + 1), (-1, 1)))[-max_iter:]  # all phis
        # converge check
        div_res = np.abs(all_vals[-max_iter:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + sys.float_info.epsilon)

        errors = np.mean(div_res, -1)
        errors = np.max(errors)
        return errors

    def _single_iter(self, train, test, device, ref_score, train_len):
        """Runs one iteration of G-Shapley algorithm."""
        idxs = np.random.permutation(train_len)
        marginal_contribs = np.zeros(train_len)

        new_score = ref_score
        batch_size = len(train[0][0])

        self.reset_model()
        for i, idx in enumerate(idxs):
            old_score = new_score
            bix, eix = idx // batch_size, idx % batch_size
            x, lbls = train[bix]
            data_tensor, lbl_tensor = x[eix].unsqueeze(0), lbls[eix].unsqueeze(0)
            self.trainer.train([(data_tensor, lbl_tensor)], device, self.train_args)
            metrics = self.trainer.test(test, device, self.train_args)
            new_score = metrics["test_" + self.train_args.data_val_metric]
            marginal_contribs[idx] = new_score - old_score

        return marginal_contribs

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

            metrics = self.trainer.test(test, device, self.train_args)
            ref_score = metrics[f"test_{self.train_args.data_val_metric}"]

            cur_thread = threading.currentThread()
            print(f"{cur_thread.getName()}, reference score: ", ref_score)

            mem_gshaps = np.zeros((0, train_len))

            best_lr = self._find_best_lr(train, test, device)
            print(f"{cur_thread.getName()}, best lr: ", best_lr)

            max_iter = self.args.gnorm_max_iter * train_len
            disp_freq = max(max_iter // 10, 1)
            for it in range(max_iter):
                marginals = self._single_iter(train, test, device, ref_score, train_len)
                # dims: (iter, # train samples)
                mem_gshaps = np.concatenate([mem_gshaps,
                                             np.reshape(marginals, (1, -1))])
                if it % disp_freq == 0:
                    ierr = self._error(mem_gshaps, len(mem_gshaps))
                    print(f"{cur_thread.getName()}, Iter: {it}/{max_iter}, Err: {ierr:.3f}")

            vals_tmc = np.mean(mem_gshaps, 0)
            ierr = self._error(mem_gshaps, max_iter)
            print(f"{cur_thread.getName()}: Max iter reached ! Error: {ierr}")
            return vals_tmc
        finally:
            self.reset_model()
            self.trainer.set_verbose(True)

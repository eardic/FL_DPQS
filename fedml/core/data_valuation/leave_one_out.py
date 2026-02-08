import logging
import sys
import threading
import copy
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from fedml.core.data_valuation.data_valuator import DataValuator


class LOO(DataValuator):
    def __init__(self, trainer, args) -> None:
        super().__init__(trainer, args)
        self.dsize_thresh = self.args.loo_grouping_thresh if hasattr(self.args, "loo_grouping_thresh") else 1e6
        self.step = 1
        self.args = copy.deepcopy(args)
        if hasattr(args, "loo_lr"):
            self.args.learning_rate = args.loo_lr

    def compute_values(self, train_local, test_local, device):
        try:
            t_id = threading.current_thread().getName()

            if self.args.remove_bad_ratio <= 0:
                return train_local

            if isinstance(train_local, DataLoader):
                train_local = list(train_local)
            if isinstance(test_local, DataLoader):
                test_local = list(test_local)

            self.trainer.set_verbose(False)
            self.reset_model()
            self.trainer.train(train_local, device, self.args)
            ref_test_metrics = self.trainer.test(test_local, device, self.args)
            ref_test_acc, ref_test_total = ref_test_metrics["test_" + self.args.data_val_metric], \
                                           ref_test_metrics["test_total"]
            ref_test_acc_per_class = ref_test_metrics["test_%s_per_class" % self.args.data_val_metric]

            logging.info(f"{t_id} -> Ref. test metric for {ref_test_total} test samples: {ref_test_acc}")
            logging.info(f"{t_id} -> Ref. test class-acc for {ref_test_total} test samples: {ref_test_acc_per_class}")

            ref_cls_weights = 1.0
            if self.args.loo_type == "bal":
                ref_cls_weights = np.array([1.0 - v for k, v in ref_test_acc_per_class.items()])

            total_samp_count = 0
            for batch_idx, (x, labels) in enumerate(train_local):
                total_samp_count += len(x)

            if total_samp_count < self.dsize_thresh:
                self.step = 1
            else:
                self.step = self.args.loo_group_size if hasattr(self.args, "loo_group_size") else 1
                logging.info(f"LOO grouping enabled with the size: {self.step}")

            scores = []
            batch_num = len(train_local)
            for batch_idx, (x, labels) in enumerate(train_local):
                batch_scores = []
                logging.info('{} -> Computing scores of batch {}/{} with size:{}'.format(t_id,
                                                                                         batch_idx + 1,
                                                                                         batch_num,
                                                                                         len(labels)))
                batch_size = x.size(0)
                if batch_size <= self.step:
                    logging.warning(f"Batch size ({batch_size}) is smaller than loo group size ({self.step})!")
                for data_ix in range(0, batch_size, self.step):
                    self.reset_model()
                    wbatch = torch.cat((x[:data_ix], x[data_ix + self.step:])).to(device)
                    wlabels = torch.cat((labels[:data_ix], labels[data_ix + self.step:])).to(device)
                    train_data = [(wbatch, wlabels)] if batch_size > self.step else []
                    if len(train_data) == 0:
                        logging.warning(f"Skipped all samples ({batch_size}) in current batch {batch_idx} !")
                    train_data.extend(train_local[:batch_idx])
                    train_data.extend(train_local[batch_idx + 1:])
                    if len(train_data) == 0:
                        logging.error(f"No train data !!!")
                    self.trainer.train(train_data, device, self.args)
                    test_metrics = self.trainer.test(test_local, device, self.args)
                    acc_per_class = test_metrics["test_%s_per_class" % self.args.data_val_metric]
                    scr_per_class = [v - acc_per_class[k] for k, v in ref_test_acc_per_class.items()]

                    if self.args.loo_type == "sum":
                        bscore = np.sum(scr_per_class)
                    elif self.args.loo_type == "std":
                        bscore = scr_per_class
                    elif self.args.loo_type == "bal":
                        bscore = np.sum(np.array(scr_per_class) * ref_cls_weights)
                    else:
                        bscore = ref_test_acc - test_metrics["test_" + self.args.data_val_metric]
                    cur_miss = batch_size - wlabels.size(0)  # skipped element count
                    if cur_miss == 0:
                        logging.error(f"No samples have been skipped ! BS: {batch_size}, LOO Step: {self.step}")
                    for _ in range(cur_miss):
                        batch_scores.append(bscore)
                batch_scores = np.array(batch_scores, dtype=np.float32)
                logging.info('{} -> Mean Batch Score {}/{} : {:.6f}'.format(t_id,
                                                                            batch_idx + 1,
                                                                            batch_num,
                                                                            batch_scores.mean()))
                scores.append(batch_scores)

            if self.args.loo_type == "std":
                scr = np.concatenate(scores)
                std = np.std(scr, axis=0)
                scr = np.sum(scr * std, axis=1)
                i, new_scr = 0, []
                for s in scores:
                    new_scr.append(scr[i:i + s.shape[0]])
                    i += s.shape[0]
                scores = new_scr

            return np.concatenate(scores).flatten()
        finally:
            self.reset_model()
            self.trainer.set_verbose(True)


if __name__ == '__main__':
    loo = LOO()

import copy

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from ...core.alg_frame.client_trainer import ClientTrainer


class DataValuator(object):
    def __init__(self, trainer: ClientTrainer, args) -> None:
        self.args = args
        self.last_scores = []
        self.trainer = trainer
        self.weights = copy.deepcopy(self.trainer.get_model_params())

    def set_trainer(self, tr):
        self.trainer = tr
        self.weights = copy.deepcopy(tr.get_model_params())

    def get_trainer(self):
        return self.trainer

    def get_last_scores(self):
        return self.last_scores

    def backup_model(self):
        self.weights = copy.deepcopy(self.trainer.get_model_params())  # backup weights for reset_model()

    def reset_model(self):
        self.trainer.set_model_params(self.weights)

    def _to_batched_tuples(self, data_list, label_list, batch_size):
        """ tensor list to batched [(tx0, ty0), (tx1, ty1)] array """
        # batch_size = len(train_local[0][0])
        data = []
        for i in range(0, len(data_list), batch_size):
            x, y = data_list[i:i + batch_size], label_list[i:i + batch_size]
            x = torch.cat(x, 0)
            y = torch.cat(y, 0)
            data.append((x, y))
        return data

    def _shuffle(self, data, data_len):
        if len(data) == 0 or data_len == 0:
            return data
        batched_data = copy.deepcopy(data)
        idxs = np.random.permutation(data_len)
        batch_size = len(batched_data[0][0])
        for i, idx in enumerate(idxs):
            bix, eix = i // batch_size, i % batch_size
            bidx, eidx = idx // batch_size, idx % batch_size
            x, lbls = batched_data[bix]
            x_idx, lbls_idx = batched_data[bidx]
            x[eix], x_idx[eidx] = x_idx[eidx], x[eix]
            lbls[eix], lbls_idx[eidx] = lbls_idx[eidx], lbls[eix]
        return batched_data

    def pick_samples_by_value(self, train, scores):
        if isinstance(train, DataLoader):
            train = list(train)
        s_scores_ix = np.argsort(scores)
        bad_count = int(np.sum(scores < 0) * self.args.remove_bad_ratio)
        selected_ixs = s_scores_ix[bad_count:]
        batch_size = len(train[0][0])
        data_list, label_list = [], []
        for six in selected_ixs:
            bix, eix = six // batch_size, six % batch_size
            x, lbls = train[bix]
            data_tensor, lbl_tensor = x[eix], lbls[eix]
            data_list.append(data_tensor.unsqueeze(0))
            label_list.append(lbl_tensor.unsqueeze(0))
        return self._to_batched_tuples(data_list, label_list, batch_size)

    def compute_values(self, train, test, device):
        raise NotImplementedError()

    def pick_samples(self, train, test, device):
        if isinstance(train, DataLoader):
            train = list(train)
        if isinstance(test, DataLoader):
            test = list(test)
        self.backup_model()
        scores = self.compute_values(train, test, device)
        self.last_scores = scores
        return self.pick_samples_by_value(train, scores)

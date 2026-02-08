import copy

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from .data_valuator import DataValuator
from ...core.alg_frame.client_trainer import ClientTrainer


class FedBalDataValuator(DataValuator):
    def __init__(self, trainer, args) -> None:
        super().__init__(trainer, args)
        self.trainer = trainer
        self.args = args
        self.min = 0
        self.max = 0
        self.sum = 0
        self.thresh = 0
        self.n_samples = 0

    def get_meta(self):
        return self.min, self.max, self.sum, self.n_samples

    def update_threshold(self, t):
        self.thresh = t

    def pick_samples(self, train, test, device):
        if isinstance(train, DataLoader):
            train = list(train)

        vals = self.compute_values(train, test, device)
        vals = np.concatenate(vals).flatten()
        self.last_scores = vals

        sorted_ixs = np.argsort(vals)
        ot_ixs = sorted_ixs[vals >= self.thresh]
        ut_ixs = sorted_ixs[vals < self.thresh]
        np.random.shuffle(ot_ixs)
        ot_ixs = ot_ixs[:int(len(ot_ixs) * self.args.fedbal_prob)]
        select_ixs = np.concatenate([ot_ixs, ut_ixs])

        batch_size = len(train[0][0])
        data_list, label_list = [], []
        for six in select_ixs:
            bix, eix = six // batch_size, six % batch_size
            x, lbls = train[bix]
            data_tensor, lbl_tensor = x[eix], lbls[eix]
            data_list.append(data_tensor.unsqueeze(0))
            label_list.append(lbl_tensor.unsqueeze(0))

        self.min = vals[sorted_ixs[0]]
        self.max = vals[sorted_ixs[-1]]
        self.sum = np.sum(vals[select_ixs])
        self.n_samples = len(select_ixs)

        return self._to_batched_tuples(data_list, label_list, batch_size)


class FedBalGradNormValuation(FedBalDataValuator):
    def __init__(self, trainer, args) -> None:
        super().__init__(trainer, args)

    def compute_values(self, train, test, device):
        return self.trainer.compute_grads(train, device)


class FedBalLossValuation(FedBalDataValuator):
    def __init__(self, trainer, args) -> None:
        super().__init__(trainer, args)

    def compute_values(self, train, test, device):
        return self.trainer.compute_loss(train, device)

import copy
import logging
import math

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError
from sklearn.svm import OneClassSVM

from fedml.core import FedMLDifferentialPrivacy
from fedml.core.data_valuation.data_valuator import DataValuator


class AnomalyBasedValuator(DataValuator):

    def __init__(self, trainer, args) -> None:
        super().__init__(trainer, args)
        self.args = copy.deepcopy(args)
        self.algo = self.args.anomaly_algo.lower() if hasattr(self.args, "anomaly_algo") else "ocsvm"
        self.source = self.args.anomaly_algo_source.lower() if hasattr(self.args, "anomaly_algo_source") else "loss"
        self.select_mode = self.args.data_val_select_algo.lower() if hasattr(self.args,
                                                                             "data_val_select_algo") else "all"
        self.select_ratio = self.args.data_val_select_ratio if hasattr(self.args, "data_val_select_ratio") else 1
        self.per_class = self.args.data_val_per_class if hasattr(self.args, "data_val_per_class") else False
        self.dval_global = hasattr(self.args, "data_val_type") and (self.args.data_val_type == "global")
        self.sample_ftr = []
        self.sample_labels = []
        self.class_num = args.class_num
        if self.per_class:
            self.odet_model = [self.create_algo() for i in range(self.class_num)]
        else:
            self.odet_model = self.create_algo()

        if self.select_mode == "srand" or self.select_mode == "stratified_random":
            from sklearn.model_selection import StratifiedShuffleSplit
            self.sss = StratifiedShuffleSplit(n_splits=1, train_size=self.select_ratio, random_state=14)

    def get_sample_dvals(self):
        sftr = self.sample_ftr
        lbls = self.sample_labels
        if self.select_mode == "minmax":
            sel_loss_ixs = []
            elem_c = max(int(len(sftr) * self.select_ratio), 1)
            for i in range(sftr.ndim):
                ixs = np.argsort(sftr[:, i])
                sel_loss_ixs.extend(ixs[:elem_c].tolist())
                sel_loss_ixs.extend(ixs[-elem_c:].tolist())
            filter = np.unique(sel_loss_ixs)
            sftr = sftr[filter]
            lbls = lbls[filter]
        elif self.select_mode == "rand" or self.select_mode == "random":
            ixs = np.arange(0, len(sftr))
            np.random.shuffle(ixs)
            elem_c = max(int(len(sftr) * self.select_ratio), 1)
            sftr = sftr[ixs[:elem_c]]
            lbls = lbls[ixs[:elem_c]]
        elif self.select_mode == "srand" or self.select_mode == "stratified_random":
            sdata, _ = list(self.sss.split(sftr, self.sample_labels))[0]
            sftr = sftr[sdata]
            lbls = lbls[sdata]

        fed_dp = FedMLDifferentialPrivacy.get_instance()
        if self.dval_global and fed_dp.is_ldp_enabled():
            logging.info("-----add local DP noise ----")
            sftr = fed_dp.add_noise2(sftr, len(sftr))

        if self.per_class:
            grouped_vals = []
            for l in range(self.class_num):
                grouped_vals.append(sftr[lbls == l])
            sftr = np.array(grouped_vals, copy=False, dtype=np.object)
        return sftr

    def set_odet_model(self, model):
        self.odet_model = model

    def train_odet_model(self, dval_data):
        if self.per_class:
            for i in range(self.class_num):
                d_len = len(dval_data[i])
                if d_len == 0:
                    print(f"Warning: Could not collect training data for class id {i} !")
                    continue
                if self.algo == "iforest":
                    self.odet_model[i].set_params(n_estimators=int(math.ceil(math.sqrt(d_len))))
                self.odet_model[i].fit(np.array(dval_data[i], copy=False))
        else:
            if self.algo == "iforest":
                self.odet_model.set_params(n_estimators=int(math.ceil(math.sqrt(len(dval_data)))))
            self.odet_model.fit(dval_data)

    def create_algo(self):
        if self.algo == "ocsvm":
            nu = self.args.anomaly_ocsvm_nu if hasattr(self.args, "anomaly_ocsvm_nu") else 0.5
            return OneClassSVM(nu=nu)
        elif self.algo == "iforest":
            c = self.args.anomaly_iforest_c if hasattr(self.args, "anomaly_iforest_c") else 0.1
            return IsolationForest(contamination=c)
        return None

    def compute_features(self, train_local, device):
        if self.source == "loss":
            ftr = self.trainer.compute_loss(train_local, device, reduce='none')
            ftr = np.concatenate(ftr)
            if ftr.ndim == 1:
                ftr = ftr.reshape((-1, 1))
            nan_count = np.isnan(ftr).sum()
            if nan_count > 0:
                logging.warning(f"Detected {nan_count}/{len(ftr)} loss value for data valuation !")
            ftr = np.nan_to_num(ftr, nan=1e6, posinf=1e6, neginf=-1e6, copy=False)
        elif self.source == "feature" or self.source == "ftr":
            ftr, _ = self.trainer.extract_features(train_local, device, self.args)
            ftr = ftr.cpu().detach().numpy()
        elif self.source == "all":
            losses, embs = self.trainer.compute_loss(train_local, device, reduce='none', return_emb=True)
            losses = np.concatenate(losses)
            embs = np.concatenate(embs)
            labels = np.concatenate([lbls.cpu().detach().numpy() for _, lbls in train_local])
            emb_dists = []
            if hasattr(self.trainer, "iae_algo") and self.trainer.iae_algo is not None:
                emb_c = self.trainer.iae_algo.iae_c
                for emb, l in zip(embs, labels):
                    c = emb_c[l].cpu().detach().numpy()
                    # d = np.dot(emb, c) / (np.linalg.norm(emb) * np.linalg.norm(c))
                    # d = 1 - ((d + 1) / 2)
                    d = np.sqrt(np.sum((emb - c) ** 2))
                    emb_dists.append(d)
            if losses.ndim == 1:
                losses = losses.reshape((-1, 1))
            ftr = losses
            if len(emb_dists) == 0:
                logging.warning("IAE distance scores are not computed, so using only loss info !")
            else:
                ftr = np.hstack((ftr, np.array(emb_dists)[:, np.newaxis]))
        else:
            raise RuntimeError("Invalid source type is given : ", self.source)
        self.sample_ftr = ftr
        self.sample_labels = np.concatenate([lbls.cpu().detach().numpy() for _, lbls in train_local])
        return self.sample_ftr, self.sample_labels

    def compute_values(self, train_local, test_local, device):
        ftr, lbls = self.compute_features(train_local, device)
        if not self.dval_global:  # if local
            dvals = self.get_sample_dvals()
            self.train_odet_model(dvals)
        if self.per_class:
            preds = []
            for f, l in zip(ftr, lbls):
                try:
                    model = self.odet_model[int(l)]
                    p = model.predict(f[np.newaxis, :])
                    preds.append(p[0])
                except NotFittedError:
                    preds.append(1.)
                    print(f"Warning: Model is not fit ! Skipping prediction for class {int(l)}")
            preds = np.array(preds, copy=False)
        else:
            preds = self.odet_model.predict(ftr)
        return preds

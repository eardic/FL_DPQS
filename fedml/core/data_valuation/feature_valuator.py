import copy

import numpy as np
import torch
from torchvision.models import resnet18
import torch.nn as nn
from fedml.core.data_valuation.data_valuator import DataValuator


class FeatureValuator(DataValuator):
    ftr_model = None

    def __init__(self, trainer, args) -> None:
        super().__init__(trainer, args)
        self.thresh = self.args.ftrsim_thresh if hasattr(self.args, "ftrsim_thresh") else 0
        self.model_name = self.args.ftrsim_model if hasattr(self.args, "ftrsim_model") else "self"
        self.args = copy.deepcopy(args)

        if self.model_name == "resnet18" and FeatureValuator.ftr_model is None:
            FeatureValuator.ftr_model = resnet18(pretrained=True)
            FeatureValuator.ftr_model.fc = nn.Identity()
            FeatureValuator.ftr_model.eval()

    def extract_features(self, data, device):
        self.ftr_model = self.ftr_model.to(device)
        ftrs = []
        lbls = []
        for (x, labels) in data:
            x = x.to(device)
            if x.ndim == 3:
                x = x.unsqueeze(1).repeat((1, 3, 1, 1))
            if self.model_name == "resnet18":
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bicubic")
            emb = FeatureValuator.ftr_model(x)
            ftrs.append(emb)
            lbls.append(labels)
        return torch.cat(ftrs, dim=0), torch.cat(lbls, dim=0)

    def compute_values(self, train_local, test_local, device):
        try:
            self.trainer.set_verbose(False)

            if self.model_name == "self":
                train_ftrs, train_lbls = self.trainer.extract_features(train_local, device, self.args)
                test_ftrs, test_lbls = self.trainer.extract_features(test_local, device, self.args)
            elif self.model_name == "resnet18":
                train_ftrs, train_lbls = self.extract_features(train_local, device)
                test_ftrs, test_lbls = self.extract_features(test_local, device)
            else:
                raise RuntimeError(f"Invalid model name : {self.model_name}")

            mean_ftrs = []
            for l in torch.unique(test_lbls).sort()[0]:
                mean_ftrs.append(test_ftrs[test_lbls == l].mean(0).unsqueeze(0))
            mean_ftrs = torch.cat(mean_ftrs, dim=0)

            mean_ftrs_norm = mean_ftrs / mean_ftrs.norm(dim=1)[:, None]
            train_ftrs_norm = train_ftrs / train_ftrs.norm(dim=1)[:, None]
            cos_sims = torch.mm(train_ftrs_norm, mean_ftrs_norm.transpose(0, 1))

            scores = []
            for i, sim in enumerate(cos_sims):
                scores.append(sim[train_lbls[i]].item() - self.thresh)

            return np.array(scores, copy=False)
        finally:
            self.reset_model()
            self.trainer.set_verbose(True)

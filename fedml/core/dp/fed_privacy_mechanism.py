import logging

import numpy as np

from fedml.core.dp.budget_accountant import BudgetAccountant
from fedml.core.dp.common.utils import check_params
from fedml.core.dp.mechanisms import Laplace, Gaussian


class FedMLDifferentialPrivacy:
    _dp_instance = None

    @staticmethod
    def get_instance():
        if FedMLDifferentialPrivacy._dp_instance is None:
            FedMLDifferentialPrivacy._dp_instance = FedMLDifferentialPrivacy()
        return FedMLDifferentialPrivacy._dp_instance

    def __init__(self):
        self.is_dp_enabled = False
        self.dp_type = None
        self.dp = None

    def init(self, args):
        if hasattr(args, "enable_dp") and args.enable_dp:
            logging.info(
                ".......init dp......." + args.mechanism_type + "-" + args.dp_type
            )
            self.is_dp_enabled = True
            self.mechanism_type = args.mechanism_type.lower()
            self.dp_type = args.dp_type.lower().strip()
            check_params(args.epsilon, args.delta, args.sensitivity)
            self.epsilon = args.epsilon
            self.delta = args.delta
            if self.dp_type not in ["cdp", "ldp"]:
                raise ValueError(
                    "DP type can only be cdp (for central DP) and ldp (for local DP)! "
                )
            if self.mechanism_type == "laplace":
                self.dp = Laplace(args)
            elif self.mechanism_type == "gaussian":
                self.dp = Gaussian(args)
            else:
                raise NotImplementedError("DP mechanism not implemented!")
            self.accountant = BudgetAccountant()

    def is_enabled(self):
        return self.is_dp_enabled

    def is_cdp_enabled(self):
        return self.is_enabled() and self.get_dp_type() == "cdp"

    def is_ldp_enabled(self):
        return self.is_enabled() and self.get_dp_type() == "ldp"

    def get_dp_type(self):
        return self.dp_type

    def add_noise(self, grad, data_size=None, lambda_l=0.0):
        new_grad = dict()
        for k in grad.keys():
            new_grad[k] = self._compute_new_grad(grad[k], data_size, lambda_l)
        self.accountant.spend(epsilon=self.epsilon, delta=0)
        return new_grad

    def add_noise2(self, x, data_size=None, lambda_l=0.0):
        self.accountant.spend(epsilon=self.epsilon, delta=0)
        n = self.dp.compute_noise(x.shape, data_size, lambda_l)
        return x + (n.numpy() if isinstance(x, np.ndarray) else n)

    def _compute_new_grad(self, grad, data_size=None, lambda_l=0.0):
        noise = self.dp.compute_noise(grad.shape, data_size, lambda_l)
        # print(f"noise computed with data distribution = {noise}")
        return noise + grad

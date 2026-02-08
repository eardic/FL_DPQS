import math

import numpy as np
import torch
from .base_dp_mechanism import BaseDPMechanism


class Laplace(BaseDPMechanism):
    """
    The classical Laplace mechanism in differential privacy.
    """

    def __init__(self, args):
        self.args = args
        self.epsilon = args.epsilon
        self.delta = args.delta
        self.tl = (args.comm_round / args.epochs) * (args.client_num_per_round / args.client_num_in_total)
        if args.gradient_clip_norm != 1:
            raise RuntimeError(f"Lap-DP requires l1-norm grad clipping but given l{args.gradient_clip_norm}-norm !", )

    def compute_sensitivity(self, data_size, lambda_l):
        if lambda_l > 1e-10:
            E_0 = math.log(1 + data_size) / math.log(1 + lambda_l * self.args.learning_rate)
            if self.args.epochs < E_0:
                s = 2 * self.args.gradient_clip_t / (lambda_l * data_size)
                s *= (math.pow(1 + lambda_l * self.args.learning_rate, self.args.epochs) - 1)
            else:
                s = 2 * self.args.gradient_clip_t + 2 * self.args.learning_rate * self.args.gradient_clip_t * (
                            self.args.epochs - E_0)
        else:
            s = (2 * self.args.gradient_clip_t * self.args.learning_rate * self.args.epochs) / data_size
        return s

    def compute_noise(self, weight_size, data_size=None, lambda_l=0.0):
        s = self.compute_sensitivity(data_size, lambda_l)
        scl = (s * self.tl) / self.epsilon
        return torch.tensor(np.random.laplace(loc=0.0, scale=scl, size=weight_size))

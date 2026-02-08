import numpy as np
import torch
from .base_dp_mechanism import BaseDPMechanism


class Gaussian(BaseDPMechanism):
    def __init__(self, args):
        self.args = args
        self.epsilon = args.epsilon
        self.delta = args.delta
        if self.epsilon == 0 or self.delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")
        self.tl = (args.comm_round / args.epochs) * (args.client_num_per_round / args.client_num_in_total)
        self.sensitivity = 2 * args.learning_rate * args.gradient_clip_t * args.epochs
        if args.gradient_clip_norm != 2:
            raise RuntimeError(f"Gauss-DP requires l2-norm grad clip but given l{args.gradient_clip_norm}-norm !")
        self.scale = self.sensitivity ** 2 * (self.tl * 2 * np.log(1 / float(self.delta))) / (self.epsilon ** 2)
        print(f"Gaussian Scale: {self.scale}")

    def compute_noise(self, weight_size, data_size=None, lambda_l=0):
        std = self.scale
        if data_size is not None and data_size > 0:
            q2 = min(self.args.batch_size / data_size, 1) ** 2
            std = std * q2
            if self.epsilon > self.tl * q2:
                print("Warning: (e,d)-DP may not be provided for epsilon:", self.epsilon)
        return torch.normal(mean=0, std=std, size=weight_size)

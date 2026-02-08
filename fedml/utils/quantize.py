import argparse
import sys

import numpy as np
import torch


class Quantizer(object):

    def __init__(self, args) -> None:
        super().__init__()
        self.method = args.quantize_algo
        self.op = args.quantize_op
        self.b = args.quantize_precision

    def get_bit_length(self):
        return self.b

    def set_bit_length(self, bits):
        self.b = bits

    def get_tensor_type(self):
        if self.b <= 8:
            return torch.int8
        elif self.b <= 16:
            return torch.int16
        elif self.b <= 32:
            return torch.int32
        return torch.int64

    def __affine_quantize(self, x: torch.Tensor):
        qlim = (2 ** self.b) - 1
        qmin = -2. ** (self.b - 1.)
        qmax = 2. ** (self.b - 1.) - 1
        min_val, max_val = x.min(), x.max()

        # if max and min same
        val_range = max_val - min_val
        if 0 <= val_range <= sys.float_info.epsilon:
            # if max_val is 0, set val_range to 1 for numerical stability
            if 0 <= abs(max_val) <= sys.float_info.epsilon:
                val_range = 1.0
            else:
                val_range = max_val

        scale = qlim / val_range

        zp = (-torch.round(min_val * scale) + qmin).int()

        if self.b >= 32:
            scaled_x = (scale * x.to(torch.float64) + zp)
        else:
            scaled_x = (scale * x + zp)

        q_x = scaled_x.round().clamp_(qmin, qmax)

        return q_x.to(self.get_tensor_type()), {"s": scale, "zp": zp}

    def __symmetric_quantize(self, x: torch.Tensor):
        alpha = x.abs().max() + sys.float_info.epsilon
        qmax = 2. ** (self.b - 1.) - 1
        scale = qmax / alpha
        if self.b >= 32:
            scaled_x = (scale * x.to(torch.float64))
        else:
            scaled_x = (scale * x)
        q_x = scaled_x.round_().clamp_(-qmax, qmax)
        return q_x.to(self.get_tensor_type()), {"s": scale, "zp": 0}

    def __stochastic_quantize(self, x: torch.Tensor):
        alpha = x.abs().max() + sys.float_info.epsilon
        qmax = 2. ** (self.b - 1.) - 1
        scale = qmax / alpha

        # Quantize with stochastic rounding
        if self.b >= 32:
            scaled_x = (scale * x.to(torch.float64))
        else:
            scaled_x = (scale * x)
        quantized_floor = scaled_x.floor()
        remainder = scaled_x - quantized_floor
        stochastic_round = quantized_floor + (torch.rand_like(remainder) < remainder)

        q_x = stochastic_round.clamp_(-qmax, qmax)

        return q_x.to(self.get_tensor_type()), {"s": scale, "zp": 0}

    def __quantize_tensor(self, x):
        if self.method == "affine":
            return self.__affine_quantize(x)
        elif self.method == "scale":
            return self.__symmetric_quantize(x)
        elif self.method == "stochastic":
            return self.__stochastic_quantize(x)
        else:
            raise NotImplementedError(f"{self.method} not supported !")

    def __quantize_loop(self, x):
        q_tensor = []
        q_params = []
        for i in range(x.shape[0]):
            qx, qp = self.__quantize_tensor(x[i])
            q_tensor.append(qx)
            q_params.append(qp)
        q_tensor = torch.cat(q_tensor, dim=0).to(self.get_tensor_type())
        return q_tensor, q_params

    def quantize(self, x: torch.Tensor):
        ndim = x.dim()
        if 0 <= ndim <= 1 or self.op == "per_tensor":
            qx, qp = self.__quantize_tensor(x)
            return qx, {"op": "per_tensor", "params": qp}

        if self.op == "per_row":
            x_2d = torch.reshape(x, (-1, x.shape[-1]))
            qx, qp = self.__quantize_loop(x_2d)
            qx = torch.reshape(qx, x.shape)
            return qx, {"op": self.op, "params": qp}
        elif self.op == "per_channel":
            if ndim < 3:
                qx, qp = self.__quantize_tensor(x)
                return qx, {"op": "per_tensor", "params": qp}
            elif ndim >= 3:
                x_3d = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))
                qx, qp = self.__quantize_loop(x_3d)
                qx = torch.reshape(qx, x.shape)
                return qx, {"op": self.op, "params": qp}
        elif self.op == "per_batch":
            if ndim < 4:
                qx, qp = self.__quantize_tensor(x)
                return qx, {"op": "per_tensor", "params": qp}
            elif ndim >= 4:
                x_4d = torch.reshape(x, (-1, x.shape[-3], x.shape[-2], x.shape[-1]))
                qx, qp = self.__quantize_loop(x_4d)
                qx = torch.reshape(qx, x.shape)
                return qx, {"op": self.op, "params": qp}
        else:
            raise NotImplementedError(f"'{self.op}' not supported !")

    @staticmethod
    def __dequantize_tensor(x: torch.Tensor, qp: dict):
        return (x.float() - qp["zp"]) / qp["s"]

    @staticmethod
    def __dequantize_loop(x: torch.Tensor, qp: dict):
        deq_tensor = []
        for i in range(x.shape[0]):
            deqx = Quantizer.__dequantize_tensor(x[i], qp[i])
            deq_tensor.append(deqx)
        return torch.reshape(torch.cat(deq_tensor, dim=0), x.shape)

    @staticmethod
    def dequantize(x: torch.Tensor, qp_dict: dict):
        op_name = qp_dict["op"]
        qp = qp_dict["params"]
        if op_name == "per_tensor":
            return Quantizer.__dequantize_tensor(x, qp)
        elif op_name == "per_row":
            x_2d = torch.reshape(x, (-1, x.shape[-1]))
            deq_tensor = Quantizer.__dequantize_loop(x_2d, qp)
            return deq_tensor
        elif op_name == "per_channel":
            x_3d = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))
            deq_tensor = Quantizer.__dequantize_loop(x_3d, qp)
            return deq_tensor
        elif op_name == "per_batch":
            x_4d = torch.reshape(x, (-1, x.shape[-3], x.shape[-2], x.shape[-1]))
            deq_tensor = Quantizer.__dequantize_loop(x_4d, qp)
            return deq_tensor
        else:
            raise NotImplementedError(f"'{op_name}' not supported !")

    def quantize_state_dict(self, state_dict: dict, err=False):
        q_state_dict = {}
        q_param_dict = {"q_bits": self.b}
        for name, tens in state_dict.items():
            qt, qpar = self.quantize(tens)
            q_state_dict[name] = qt
            q_param_dict[name] = qpar
        if err:
            deq_state_dict = Quantizer.dequantize_state_dict(q_state_dict, q_param_dict)
            q_param_dict["q_err"] = Quantizer.quant_error(state_dict, deq_state_dict)
        return q_state_dict, q_param_dict

    @staticmethod
    def dequantize_state_dict(state_dict: dict, q_param_dict: dict):
        deq_state_dict = {}
        for name, tens in state_dict.items():
            q_params = q_param_dict[name]
            t = Quantizer.dequantize(tens, q_params)
            deq_state_dict[name] = t
        return deq_state_dict

    @staticmethod
    def quant_error(org_sdict, deq_sdict):
        errs = []
        for name, deq_tens in deq_sdict.items():
            org_tens = org_sdict[name]
            error = torch.mean((org_tens - deq_tens) ** 2).item()
            errs.append(error)
        return np.mean(errs)


def compute_size(state_dict: dict):
    size = 0
    for k, t in state_dict.items():
        size += t.nelement() * t.element_size()
    return size

# def to_fp32(state_dict: dict):
#     for name, tens in state_dict.items():
#         state_dict[name] = tens.to(torch.float32)
#     return state_dict
#
#
# def to_fp16(state_dict: dict):
#     for name, tens in state_dict.items():
#         state_dict[name] = tens.to(torch.float16)
#     return state_dict

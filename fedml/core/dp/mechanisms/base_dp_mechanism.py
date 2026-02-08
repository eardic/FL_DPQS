from abc import ABC, abstractmethod


class BaseDPMechanism(ABC):

    @abstractmethod
    def compute_noise(self, weight_size, data_size=None, lambda_l=0.0):
        pass

from abc import ABC, abstractmethod
import logging
from ...core.dp.fed_privacy_mechanism import FedMLDifferentialPrivacy


class ClientTrainer(ABC):
    """Abstract base class for federated learning trainer.
    1. The goal of this abstract class is to be compatible to
    any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
    2. This class can be used in both server and client side
    3. This class is an operator which does not cache any states inside.
    """

    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args
        self.verbose = True
        self.lambda_l = 0
        self.data_size = 0
        self.embeds = []
        FedMLDifferentialPrivacy.get_instance().init(args)

    def set_id(self, trainer_id):
        self.id = trainer_id

    def set_verbose(self, verb: bool):
        self.verbose = verb

    def get_lambda_l(self):
        return self.lambda_l

    def get_data_size(self):
        return self.data_size

    def get_embeddings(self):
        return self.embeds

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    def on_before_local_training(self, train_data, device, args):
        pass

    @abstractmethod
    def train(self, train_data, device, args):
        pass

    def on_after_local_training(self, train_data, device, args):
        if FedMLDifferentialPrivacy.get_instance().is_ldp_enabled():
            logging.info("-----add local DP noise ----")
            model_with_dp = FedMLDifferentialPrivacy.get_instance().add_noise(self.get_model_params(),
                                                                              self.get_data_size(),
                                                                              self.get_lambda_l())
            self.set_model_params(model_with_dp)

    def test(self, test_data, device, args):
        pass

    def compute_loss(self, train_data, device, weighted=True, reduce='sum', return_emb=False):
        pass

    def compute_grads(self, train_data, device, weighted=True):
        pass

    def extract_features(self, train_data, device, args):
        pass

    def get_trainer(self):
        pass



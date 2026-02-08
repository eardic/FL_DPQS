import logging

from fedml.core.data_valuation.data_val_creator import create_data_valuator
from .utils import transform_tensor_to_list


class FedAVGTrainer(object):
    def __init__(
            self,
            client_index,
            train_data_local_dict,
            train_data_local_num_dict,
            test_data_local_dict,
            train_data_num,
            device,
            args,
            model_trainer,
    ):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None
        self.data_valuator = create_data_valuator(self.trainer, args)
        self.device = device
        self.args = args

        if self.data_valuator is not None:
            logging.info(
                'Client {}: Using data valuator {}'.format(self.client_index,
                                                           self.data_valuator.__class__.__name__))

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def train(self, round_idx=None):
        self.args.round_idx = round_idx

        train_data = self.train_local
        train_data_size = self.local_sample_number
        if self.data_valuator is not None:
            train_data = self.data_valuator.pick_samples(train_data, self.test_local, self.device)
            train_data_size = 0
            for (x, labels) in train_data:
                train_data_size += len(labels)

            if train_data_size > 0:
                logging.info(
                    'Client {}: Training for filtered data: {}/{}'.format(self.client_index,
                                                                          train_data_size,
                                                                          self.local_sample_number))
                self.trainer.train(train_data, self.device, self.args)
            else:
                logging.info('Client {}: All data is dirty ! Skipping the training...'.format(self.client_index))
        else:
            self.trainer.train(train_data, self.device, self.args)

        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, train_data_size

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = (
            test_metrics["test_correct"],
            test_metrics["test_total"],
            test_metrics["test_loss"],
        )

        return (
            train_tot_correct,
            train_loss,
            train_num_sample,
            test_tot_correct,
            test_loss,
            test_num_sample,
        )

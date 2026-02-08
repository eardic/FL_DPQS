import logging

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode

from .dataset import CheXpert
from ..config import DataValArgs


def _get_mean_and_std(dataset: Dataset):
    """Compute the mean and std of dataset."""
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for i, (img, _) in enumerate(data_loader):
        if i % 1000 == 0:
            print(i)
        mean += img.mean(dim=(0, 2, 3))
        std += img.std(dim=(0, 2, 3))
    mean /= len(data_loader)
    std /= len(data_loader)
    return mean, std


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_chexpert():
    CHEXPERT_MEAN = [0.503, 0.503, 0.503]
    CHEXPERT_STD = [0.291, 0.291, 0.291]

    image_size = 224
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(CHEXPERT_MEAN, CHEXPERT_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CHEXPERT_MEAN, CHEXPERT_STD),
        ]
    )

    return train_transform, test_transform


def get_dataloader(
        datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None, policy="zeros",
        train_dataset=None,
        test_dataset=None
):
    dl_obj = CheXpert

    transform_train, transform_test = _data_transforms_chexpert()

    train_ds = dl_obj(
        datadir,
        dataidxs=dataidxs_train,
        train=True,
        transform=transform_train,
        policy=policy,
        dataset=train_dataset
    )
    test_ds = dl_obj(
        datadir,
        dataidxs=dataidxs_test,
        train=False,
        transform=transform_test,
        policy=policy,
        dataset=test_dataset
    )

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4
    )
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4
    )

    return train_dl, test_dl


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        # Multi-label veride, her etiket sütun bazında toplanır
        label_counts = np.sum(y_train[dataidx], axis=0)
        # Her etiketi ve sayısını bir dictionary olarak saklar
        tmp = {f"label_{i}": int(label_counts[i]) for i in range(len(label_counts))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


def partition_data(dataset, datadir, partition, n_nets, alpha):
    np.random.seed(10)
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_chexpert_data(datadir)
    n_train = len(X_train)

    ix_map_path = f"chexpert_dataidx_map_{n_nets}.pickle"
    dist_info_path = f"chexpert_dist_{n_nets}.pickle"

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        N = y_train.shape[0]  # number of samples
        net_dataidx_map = {i: [] for i in range(n_nets)}

        # Calculate the Dirichlet distribution once for the entire dataset
        proportions = np.random.dirichlet([alpha] * n_nets, N)
        assigned_clients = np.argmax(proportions, axis=1)

        # Assign each data point to one client based on the Dirichlet sampling
        for idx, client in enumerate(assigned_clients):
            net_dataidx_map[client].append(idx)

        # Ensure that each client has a minimum number of samples
        for client, indices in net_dataidx_map.items():
            if len(indices) < 10:
                additional_samples = np.random.choice(
                    np.setdiff1d(np.arange(N), indices),
                    10 - len(indices),
                    replace=False
                )
                net_dataidx_map[client].extend(additional_samples)

        # Save the data index map
        os.makedirs(datadir, exist_ok=True)
        with open(os.path.join(datadir, ix_map_path), "wb") as f:
            pickle.dump(net_dataidx_map, f)

    elif partition == "hetero-fix":
        with open(os.path.join(datadir, ix_map_path), "rb") as f:
            net_dataidx_map = pickle.load(f)

        if len(net_dataidx_map) != n_nets:
            print(
                "Warning: Data index map is different. Please recreate it with hetero !")
            raise RuntimeWarning("Invalid data mapping !")

    if partition == "hetero-fix":
        with open(os.path.join(datadir, dist_info_path), "rb") as f:
            traindata_cls_counts = pickle.load(f)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        os.makedirs(datadir, exist_ok=True)
        with open(os.path.join(datadir, dist_info_path), "wb") as f:
            pickle.dump(traindata_cls_counts, f)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def load_chexpert_data(data_dir):
    transform_train, transform_test = _data_transforms_chexpert()

    train_dataset = CheXpert(
        data_dir=data_dir,
        dataidxs=None,
        train=True,
        transform=transform_train,
        policy="zeros",
    )
    test_dataset = CheXpert(
        data_dir=data_dir,
        dataidxs=None,
        train=False,
        transform=transform_test,
        policy="zeros",
    )

    X_train, y_train = train_dataset.images, train_dataset.labels
    X_test, y_test = test_dataset.images, test_dataset.labels

    return (X_train, y_train, X_test, y_test)


def load_partition_data_chexpert(
        dataset,
        data_dir,
        partition_method,
        partition_alpha,
        client_number,
        batch_size,
        n_proc_in_silo=0,
        policy="zeros",
):
    data_dir = os.path.join(data_dir, "chexpert")

    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
    ) = partition_data(
        dataset, data_dir, partition_method, client_number, partition_alpha
    )
    class_num = y_train.shape[1]
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    test_data_num = len(y_test)

    train_dataset = (X_train, y_train, class_num)
    test_dataset = (X_test, y_test, class_num)

    client_test_ixs = None
    if DataValArgs.client_test_size is not None:
        client_test_ixs = np.arange(0, min(DataValArgs.client_test_size, len(y_test)))

    train_data_global, test_data_global = get_dataloader(data_dir, batch_size, batch_size,
                                                         train_dataset=train_dataset,
                                                         test_dataset=test_dataset)

    logging.info(f"global train batch count = {len(train_data_global)}")
    logging.info(f"global test batch count = {test_data_num}")

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    for client_idx in range(client_number):
        client_train_ixs = net_dataidx_map[client_idx]
        local_data_num = len(client_train_ixs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info(
            "client_idx = %d, local_sample_number = %d" % (
                client_idx, local_data_num)
        )

        train_data_local, test_data_local = get_dataloader(
            data_dir, batch_size, batch_size, client_train_ixs, client_test_ixs,
            train_dataset=train_dataset,
            test_dataset=test_dataset)

        logging.info(
            "client_idx = %d, train samples = %d, test batch count = %d"
            % (client_idx, len(client_train_ixs), len(test_data_local))
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )

import logging
import os
import pickle

import PIL.Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import random
from torchvision.datasets import ImageFolder

from .datasets import CIFAR10_truncated
from ..ImageNet.datasets import ImageNet32Noise
from ..config import DataValArgs


class ImageNoiseFolder(ImageFolder):
    def __init__(self, root: str, n_classes, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.n_classes = n_classes

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        return sample


def read_data_distribution(
        filename="./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt",
):
    distribution = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0]:
                tmp = x.split(":")
                if "{" == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(
                        tmp[1].strip().replace(",", "")
                    )
    return distribution


def read_net_dataidx_map(
        filename="./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt",
):
    net_dataidx_map = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0] and "]" != x[0]:
                tmp = x.split(":")
                if "[" == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(",")
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


class Cutout(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, img):
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


def _data_transforms_cifar10():
    # CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    # CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    train_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.RandomChoice([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                Cutout(10),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomRotation(45),
                transforms.RandomAffine(45, scale=(0.5, 1.5))
            ])
            # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    # train_transform.transforms.append(Cutout(10))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(
        datadir, train=True, download=True, transform=train_transform
    )
    cifar10_test_ds = CIFAR10_truncated(
        datadir, train=False, download=True, transform=test_transform
    )

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha):
    np.random.seed(10)
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    ix_map_path = f"cifar10_dataidx_map_{n_nets}.pickle"
    dist_info_path = f"cifar10_dist_{n_nets}.pickle"

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                # Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

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

    # elif partition == "hetero-fix":
    #     dataidx_map_file_path = (
    #         "./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt"
    #     )
    #     net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)
    #
    # if partition == "hetero-fix":
    #     distribution_file_path = (
    #         "./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt"
    #     )
    #     traindata_cls_counts = read_data_distribution(distribution_file_path)
    # else:
    #     traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    #
    # return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def get_dataloader(datadir, train_bs, test_bs,
                   dataidxs_train=None, dataidxs_test=None,
                   train_dataset=None, test_dataset=None,
                   label_noise=None):
    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = CIFAR10_truncated(
        datadir,
        dataidxs=dataidxs_train,
        train=True,
        transform=transform_train,
        download=True,
        label_noise=label_noise,
        dataset=train_dataset
    )

    test_ds = CIFAR10_truncated(
        datadir,
        dataidxs=dataidxs_test,
        train=False,
        transform=transform_test,
        download=True,
        dataset=test_dataset
    )

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


def load_partition_data_distributed_cifar10(
        process_id,
        dataset,
        data_dir,
        partition_method,
        partition_alpha,
        client_number,
        batch_size,
):
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
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r])
                          for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(
            dataset, data_dir, batch_size, batch_size
        )
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info(
            "rank = %d, local_sample_number = %d" % (
                process_id, local_data_num)
        )
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, dataidxs
        )
        logging.info(
            "process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (process_id, len(train_data_local), len(test_data_local))
        )
        train_data_global = None
        test_data_global = None
    return (
        train_data_num,
        train_data_global,
        test_data_global,
        local_data_num,
        train_data_local,
        test_data_local,
        class_num,
    )


def sample_benchmark_ixs(target, sample_size):
    classes = set(target)
    size_per_class = int(sample_size / len(classes))
    bench_ix = []
    other_ix = []
    for c in classes:
        ixs = np.argwhere(target == c)
        bench_ix.append(ixs[:size_per_class])
        other_ix.append(ixs[size_per_class:])
    return np.concatenate(bench_ix).flatten().tolist(), \
           np.concatenate(other_ix).flatten().tolist()


def get_noise_dataset(data_dir):
    noise_transform = torchvision.transforms.Compose([
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32))
    ])
    if DataValArgs.client_noise_source == "imagenet32":
        data_path = os.path.join(data_dir, "imagenet32", "train_data_batch_1")
        dataset = ImageNet32Noise(data_path, noise_transform, img_size=32, n_classes=10)
        return dataset
    elif DataValArgs.client_noise_source == "svhn":
        data_path = os.path.join(data_dir, "SVHN", "train")
        return ImageNoiseFolder(data_path, n_classes=10, transform=noise_transform)
    else:
        raise RuntimeError(f"Unsupported noise source : {DataValArgs.client_noise_source}")


def load_partition_data_cifar10(
        dataset,
        data_dir,
        partition_method,
        partition_alpha,
        client_number,
        batch_size,
        n_proc_in_silo=0,
):
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
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r])
                          for r in range(client_number)])

    if DataValArgs.client_test_size is not None:
        logging.info(f"Client max test size = {DataValArgs.client_test_size}")

    noise_max_data_ix = 0
    if DataValArgs.client_label_noise is not None:
        logging.info(f"Applying label noise with rate: {DataValArgs.client_label_noise}")
        noise_max_data_ix = int(len(y_train) * DataValArgs.client_label_noise)
        if DataValArgs.client_noise_type == "openset":
            logging.info(f"Noise source: {DataValArgs.client_noise_source}")
            data_noise_set = get_noise_dataset(data_dir)
            if len(data_noise_set) < noise_max_data_ix:
                raise RuntimeError("Noise source has insufficient images !")
        for i in range(noise_max_data_ix):
            rand_cls = np.random.randint(0, class_num)
            if DataValArgs.client_noise_type == "openset":
                X_train[i] = np.array(data_noise_set[i], copy=False)
            else:
                y_train[i] = rand_cls if y_train[i] != rand_cls else abs(y_train[i] - 1)

    train_dataset = (X_train, y_train, class_num)
    test_dataset = (X_test, y_test, class_num)

    client_test_ixs = None
    if DataValArgs.client_test_size is not None:
        client_test_ixs, global_test_ixs = sample_benchmark_ixs(
            y_test, DataValArgs.client_test_size)
        train_data_global, test_data_global = get_dataloader(data_dir, batch_size, batch_size,
                                                             dataidxs_train=None,
                                                             dataidxs_test=global_test_ixs,
                                                             train_dataset=train_dataset,
                                                             test_dataset=test_dataset)
    else:
        train_data_global, test_data_global = get_dataloader(data_dir, batch_size, batch_size,
                                                             train_dataset=train_dataset,
                                                             test_dataset=test_dataset)
    test_data_num = len(test_data_global)

    logging.info(f"global train batch count = {len(train_data_global)}")
    logging.info(f"global test batch count = {test_data_num}")

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    user_noise_dict = dict()
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
            test_dataset=test_dataset,
            label_noise=DataValArgs.client_label_noise
        )

        logging.info(
            "client_idx = %d, train samples = %d, test batch count = %d"
            % (client_idx, len(client_train_ixs), len(test_data_local))
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        user_noise_dict[client_idx] = np.argwhere(
            np.array(client_train_ixs) < noise_max_data_ix).flatten()

    save_path = os.path.join(data_dir, f"cifar10_client_noisy_data_idxs_{client_number}.pickle")
    with open(save_path, "wb") as outf:
        pickle.dump(user_noise_dict, outf)
    print(f"Saved noisy data indexes to: {save_path}")

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

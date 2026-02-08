import copy
import glob
import logging
import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from fedml.data.config import DataValArgs


class ISIC2019_Proc(Dataset):
    def __init__(
            self,
            root,
            dataidxs=None,
            transform=None,
            target_transform=None,
            dataset=None
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.target = self.__build_dataset__(dataset)

    def __build_dataset__(self, dataset=None):
        if dataset is None:
            data = []
            target = []
            for f in glob.glob(os.path.join(self.root, "*.jpg")):
                data.append(np.array(Image.open(f).convert("RGB")))
                fname = os.path.basename(f)
                lbl = int(os.path.splitext(fname)[0].split("_")[-1])
                target.append(lbl)
            data = np.array(data, copy=False)
            target = np.array(target, copy=False)
            print("Loaded dataset:", self.root)
        else:
            data, target, n_class = dataset

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(Image.fromarray(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_transforms():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return train_transform, val_transform


def load_isic2019_data(datadir):
    train_transform, val_transform = get_transforms()

    train_dataset = ISIC2019_Proc(root=os.path.join(datadir, "train"), transform=train_transform)
    test_dataset = ISIC2019_Proc(root=os.path.join(datadir, "test"), transform=val_transform)

    X_train, y_train = train_dataset.data, train_dataset.target
    X_test, y_test = test_dataset.data, test_dataset.target

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


def partition_data(dataset, datadir, partition, n_nets, alpha):
    np.random.seed(10)
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_isic2019_data(datadir)
    n_train = len(X_train)
    # n_test = X_test.shape[0]

    ix_map_path = f"isic2019_dataidx_map_{n_nets}.pickle"
    dist_info_path = f"isic2019_dist_{n_nets}.pickle"

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 8
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


def get_dataloader(datadir, train_bs, test_bs,
                   dataidxs_train=None, dataidxs_test=None,
                   train_dataset=None, test_dataset=None):
    train_transform, val_transform = get_transforms()

    train_ds = ISIC2019_Proc(
        os.path.join(datadir, "train"),
        dataidxs=dataidxs_train,
        transform=train_transform,
        dataset=train_dataset
    )

    test_ds = ISIC2019_Proc(
        os.path.join(datadir, "test"),
        dataidxs=dataidxs_test,
        transform=val_transform,
        dataset=test_dataset
    )

    train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


def load_partition_data_isic2019(
        dataset,
        data_dir,
        partition_method,
        partition_alpha,
        client_number,
        batch_size,
        n_proc_in_silo=0,
):
    data_dir = os.path.join(data_dir, "isic2019")

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
            test_dataset=test_dataset
        )

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

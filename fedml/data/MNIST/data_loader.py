import pickle
import sys

import torch
import torchvision

from ..FederatedEMNIST.data_loader import FEMNISTLetterNoise
from ..ImageNet.datasets import ImageNet32Noise
from ..config import DataValArgs
import logging
from ...constants import FEDML_DATA_MNIST_URL
import zipfile
import json
import os

import numpy as np
import wget
from ...ml.engine import ml_engine_adapter

cwd = os.getcwd()


def download_mnist(data_cache_dir):
    pass
    # if not os.path.exists(data_cache_dir):
    #     os.makedirs(data_cache_dir)
    #
    # file_path = os.path.join(data_cache_dir, "MNIST.zip")
    # logging.info(file_path)
    #
    # # Download the file (if we haven't already)
    # if not os.path.exists(file_path):
    #     wget.download(FEDML_DATA_MNIST_URL, out=file_path)

    # with zipfile.ZipFile(file_path, "r") as zip_ref:
    #     zip_ref.extractall(data_cache_dir)


def get_noise_dataset(data_dir):
    if DataValArgs.client_noise_source == "imagenet32":
        noise_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.Grayscale(),
        ])
        data_path = os.path.join(data_dir, "imagenet32", "train_data_batch_1")
        dataset = ImageNet32Noise(data_path, noise_transform, img_size=32, n_classes=10)
        return dataset
    elif DataValArgs.client_noise_source == "femnist":
        data_path = os.path.join(data_dir, "FEMNIST")
        return FEMNISTLetterNoise(data_path)
    else:
        raise RuntimeError(f"Unsupported noise source : {DataValArgs.client_noise_source}")


def read_data(data_dir, train_data_dir, test_data_dir):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        test_data.update(cdata["user_data"])

    clients = sorted(cdata["users"])

    user_noise_dict = {}
    lbl_noise = DataValArgs.client_label_noise
    if lbl_noise is not None:
        print(f"Applying label noise: {lbl_noise}")
        label_flips = 0
        total_label_flip = int(
            sum([len(data["y"]) for data in train_data.values()]) * lbl_noise)
        if DataValArgs.client_noise_type == "openset":
            logging.info(f"Noise source: {DataValArgs.client_noise_source}")
            data_noise_set = get_noise_dataset(data_dir)
            if len(data_noise_set) < total_label_flip:
                raise RuntimeError("Noise source has insufficient images !")
        for user, vals in train_data.items():
            lbls = vals["y"]
            imgs = vals["x"]
            noise_ix_arr = []
            for i in range(int(round(len(lbls) * lbl_noise))):
                if DataValArgs.client_noise_type == "openset":
                    imgs[i] = np.array(data_noise_set[label_flips]).flatten().tolist()
                else:
                    rand_cls = np.random.randint(0, 10)
                    lbls[i] = float(rand_cls if int(lbls[i]) != rand_cls else abs(lbls[i] - 1))
                label_flips += 1
                noise_ix_arr.append(i)
                if label_flips > total_label_flip:
                    break
            user_noise_dict[user] = noise_ix_arr
        print(f"Total flipped labels: {label_flips}/{total_label_flip}.")
        # save_path = os.path.join(train_data_dir, "user_noisy_data_idxs.pickle")
        # with open(save_path, "wb") as outf:
        #     pickle.dump(user_noise_dict, outf)
        # print(f"Saved noisy data indexes to: {save_path}")

    bench_data = None
    benchmark_size = DataValArgs.client_test_size
    if benchmark_size is not None:
        per_cls = int(benchmark_size / 10.0)
        bx, by = [], []
        stats = np.array([0] * 10)
        for user, vals in test_data.items():
            x, y = vals["x"], vals["y"]
            for lbl in np.argwhere(stats < per_cls).flatten():
                ixs = np.argwhere(y == lbl).flatten()
                if len(ixs) > 2:
                    bx.append(x.pop(ixs[0]))
                    by.append(y.pop(ixs[0]))
                    stats[lbl] += 1
                    break
        bench_data = {"x": bx, "y": by}

    return clients, groups, train_data, test_data, bench_data, user_noise_dict


def batch_data(args, data, batch_size):
    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i: i + batch_size]
        batched_y = data_y[i: i + batch_size]
        batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(
            args, batched_x, batched_y)
        batch_data.append((batched_x, batched_y))
    return batch_data


def batch_tuple(data, batch_size):
    data_x = data[0]
    data_y = data[1]
    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i: i + batch_size]
        batched_y = data_y[i: i + batch_size]
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_mnist_by_device_id(batch_size, device_id, train_path="MNIST_mobile", test_path="MNIST_mobile"):
    train_path += os.path.join("/", device_id, "train")
    test_path += os.path.join("/", device_id, "test")
    return load_partition_data_mnist(batch_size, train_path, test_path)


def merge_tuples(tup1, tup2):
    t1 = torch.cat([tup1[0], tup2[0]], dim=0)
    t2 = torch.cat([tup1[1], tup2[1]], dim=0)
    return tuple([t1, t2])


def load_partition_data_mnist(
        args,
        batch_size,
        train_path=os.path.join(os.getcwd(), "MNIST", "train"),
        test_path=os.path.join(os.getcwd(), "MNIST", "test")
):
    users, groups, train_data, test_data, bench_data, usr_nsy_dict = read_data(args.data_cache_dir,
                                                                               train_path,
                                                                               test_path)
    if bench_data is not None:
        bench_data = batch_data(args, bench_data, batch_size)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]["x"])
        train_data_num += user_train_data_num

        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(args, train_data[u], batch_size)
        # index using client index
        train_data_local_dict[client_idx] = train_batch
        train_data_global += train_batch

        test_batch = batch_data(args, test_data[u], batch_size)
        test_data_global += test_batch
        user_test_data_num = len(test_data[u]["x"])
        test_data_num += user_test_data_num

        if bench_data is not None and len(bench_data) > 0:
            test_data_local_dict[client_idx] = bench_data
        else:
            test_data_local_dict[client_idx] = test_batch

        if usr_nsy_dict is not None and len(usr_nsy_dict) > 0:
            usr_nsy_dict[client_idx] = usr_nsy_dict[u]
            usr_nsy_dict.pop(u)

        client_idx += 1

    client_num = client_idx
    class_num = 10

    if args.client_num_in_total < client_num:
        for cix in range(args.client_num_in_total, client_num, 1):
            target_ix = cix % args.client_num_in_total
            offset = train_data_local_num_dict[target_ix]
            train_data_local_num_dict[target_ix] += train_data_local_num_dict.pop(cix)
            train_data_local_dict[target_ix].extend(train_data_local_dict.pop(cix))
            for tup in train_data_local_dict[target_ix][1:]:
                train_data_local_dict[target_ix][0] = merge_tuples(train_data_local_dict[target_ix][0], tup)
            train_data_local_dict[target_ix] = train_data_local_dict[target_ix][:1]
            # for tup in train_data_local_dict.pop(cix):
            #     train_data_local_dict[target_ix][0] = merge_tuples(train_data_local_dict[target_ix][0], tup)
            if bench_data is not None and len(bench_data) > 0:
                test_data_local_dict.pop(cix)
            else:
                test_data_local_dict[target_ix].extend(test_data_local_dict.pop(cix))
                for tup in test_data_local_dict[target_ix][1:]:
                    test_data_local_dict[target_ix][0] = merge_tuples(test_data_local_dict[target_ix][0], tup)
                test_data_local_dict[target_ix] = test_data_local_dict[target_ix][:1]
                # for tup in test_data_local_dict.pop(cix):
                #     test_data_local_dict[target_ix][0] = merge_tuples(test_data_local_dict[target_ix][0], tup)
            if usr_nsy_dict is not None and len(usr_nsy_dict) > 0:
                nsy_ix = np.array(usr_nsy_dict.pop(cix)) + offset
                usr_nsy_dict[target_ix].extend(nsy_ix.tolist())

        for target_ix in range(0, args.client_num_in_total):
            train_data_local_dict[target_ix] = batch_tuple(train_data_local_dict[target_ix][0], batch_size)
            test_data_local_dict[target_ix] = batch_tuple(test_data_local_dict[target_ix][0], batch_size)

        client_num = len(train_data_local_num_dict)

    logging.info("finished the loading data")
    # import numpy as np
    # import random
    # import os
    # from sklearn.datasets import fetch_openml
    #
    # mnist = fetch_openml('mnist_784')
    # mu = np.mean(mnist.data.astype(np.float32), 0)
    # sigma = np.std(mnist.data.astype(np.float32), 0)

    if usr_nsy_dict is not None and len(usr_nsy_dict) > 0:
        save_path = os.path.join(args.data_cache_dir, f"mnist_client_noisy_data_idxs_{client_num}.pickle")
        with open(save_path, "wb") as outf:
            pickle.dump(usr_nsy_dict, outf)
        print(f"Saved noisy data indexes to: {save_path}")

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )

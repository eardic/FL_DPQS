import json
import os

import numpy as np
from sklearn.datasets import fetch_openml


def renormalize_and_save(train_data_dir, test_data_dir):
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

    mnist = fetch_openml('mnist_784')
    mu = np.array(np.mean(mnist.data.astype(np.float32), 0))
    sigma = np.array(np.std(mnist.data.astype(np.float32), 0))

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
            for uid in cdata["user_data"].keys():
                data = cdata["user_data"][uid]
                for i in range(len(data["x"])):
                    d_new = (np.array(data["x"][i], copy=False) * sigma + mu) / 255
                    d_new = np.clip(d_new, 0, 255)
                    data["x"][i] = list(d_new)
        with open(file_path, "w") as f:
            json.dump(cdata, f)
        print("Train data renormalized to 0-1")

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
            for uid in cdata["user_data"].keys():
                data = cdata["user_data"][uid]
                for i in range(len(data["x"])):
                    d_new = (np.array(data["x"][i], copy=False) * sigma + mu) / 255
                    d_new = np.clip(d_new, 0, 255)
                    data["x"][i] = list(d_new)
        with open(file_path, "w") as f:
            json.dump(cdata, f)
        print("Test data renormalized to 0-1")


if __name__ == '__main__':
    renormalize_and_save("fedcv_data/MNIST/train", "fedcv_data/MNIST/test")

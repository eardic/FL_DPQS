import argparse
import pickle

import pandas as pd
import numpy as np
from PIL import Image
import os

import logging
from collections import Counter
from torchvision.transforms import transforms

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from fedml.data.isic2019.randaugment import rand_augment_transform


class SkinDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        assert self.mode in ["train", "test"]
        self.transform = transform
        csv_file = os.path.join(root, "ISIC_2019_Training_GroundTruth.csv")
        self.file = pd.read_csv(csv_file)

        self.images = self.file["image"].values
        self.labels = self.file.iloc[:, 1:].values.astype("int")
        self.targets = np.argmax(self.labels, axis=1)

        initial_len = len(self.images)

        # data split
        np.random.seed(0)
        idxs = np.random.permutation(initial_len)
        self.images = self.images[idxs]
        self.targets = self.targets[idxs]

        if self.mode == "train":
            self.images = self.images[:int(0.8 * initial_len)]
            self.targets = self.targets[:int(0.8 * initial_len)]
        else:
            self.images = self.images[int(0.8 * initial_len):]
            self.targets = self.targets[int(0.8 * initial_len):]

        self.n_classes = len(np.unique(self.targets))
        assert self.n_classes == 8

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = os.path.join(
            self.root, "ISIC_2019_Training_Input", self.images[index] + ".jpg")
        img = Image.open(image_name).convert("RGB")
        label = self.targets[index]
        if self.transform is not None:
            if not isinstance(self.transform, list):
                img = self.transform(img)
            else:
                img0 = self.transform[0](img)
                img1 = self.transform[1](img)
                img = [img0, img1]
        return img, label

    def __len__(self):
        return len(self.images)


def load_isic2019_data(args):
    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(
        args.img_size * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    train_randaug = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform(
            'rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params)
    ])
    train_augsim = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(args.img_size + 32),
        transforms.CenterCrop(args.img_size)
    ])

    train_dataset = SkinDataset(root=args.root, mode="train", transform=[train_randaug, train_augsim])
    test_dataset = SkinDataset(root=args.root, mode="test", transform=val_transform)

    logging.info(Counter(train_dataset.targets))
    logging.info(Counter(test_dataset.targets))

    return train_dataset, test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"C:\Datasets\isic2019", type=str)
    parser.add_argument("--save_dir",
                        default=r"C:\Users\eardic\Documents\PyCharmProjects\FedML-PHD\app\fedcv\image_classification\fedcv_data\isic2019_128",
                        type=str)
    parser.add_argument("--img_size", default=128, type=int)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "test"), exist_ok=True)

    train_dataset, test_dataset = load_isic2019_data(args)

    for i in tqdm(range(len(train_dataset))):
        (img0, img1), lbl = train_dataset.__getitem__(i)
        img0.save(os.path.join(args.save_dir, "train", "ra{}_{}.jpg".format(i, lbl)))
        img1.save(os.path.join(args.save_dir, "train", "as{}_{}.jpg".format(i, lbl)))

    for i in tqdm(range(len(test_dataset))):
        img, lbl = test_dataset.__getitem__(i)
        img.save(os.path.join(args.save_dir, "test", "{}_{}.jpg".format(i, lbl)))

    print("Done !")

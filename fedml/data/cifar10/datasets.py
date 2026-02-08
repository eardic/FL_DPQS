import copy

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def default_loader(path):
    return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class CIFAR10_truncated(data.Dataset):
    def __init__(
            self, root,
            dataidxs=None,
            train=True,
            transform=None,
            target_transform=None,
            download=False,
            label_noise=None,
            dataset=None
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.label_noise = label_noise
        self.data, self.target = self.__build_truncated_dataset__(dataset)

    def __build_truncated_dataset__(self, dataset=None):
        if dataset is None:
            print("download = " + str(self.download))
            cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
            n_class = len(cifar_dataobj.classes)
        else:
            data, target, n_class = copy.deepcopy(dataset)

        # if self.train and self.label_noise is not None:
        #     print("Applying label noise: ", self.label_noise)
        #     for i in range(int(len(target) * self.label_noise)):
        #         rand_cls = np.random.randint(0, n_class)
        #         target[i] = rand_cls if target[i] != rand_cls else abs(target[i] - 1)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

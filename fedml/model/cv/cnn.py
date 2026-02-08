import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_features=False):
        # x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        ftrs = self.relu(self.linear_1(x))
        x = self.linear_2(ftrs)
        # x = self.softmax(self.linear_2(x))
        if return_features:
            return x, ftrs
        return x


class CNN_DropOut(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        # x = self.softmax(self.linear_2(x))
        return x


# https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844
class CNN_Cifar10_S(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_Cifar10_M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        # dropout
        self.dropout = nn.Dropout(p=.25)

    def forward(self, x):
        x = self.pool(F.gelu(self.bn1(self.conv1(x))))
        x = self.pool(F.gelu(self.bn2(self.conv2(x))))
        x = self.pool(F.gelu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(F.gelu(self.fc2(x)))
        x = self.fc3(x)
        return x


class CNN_Cifar10_L(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc3 = nn.Linear(512, 10)
        # dropout
        self.dropout = nn.AlphaDropout(p=.25)

    def forward(self, x):
        x = self.pool(F.gelu(self.bn1(self.conv1(x))))
        x = self.pool(F.gelu(self.bn2(self.conv2(x))))
        x = self.pool(F.gelu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.fc3(x)
        return x


class VGG7(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, bn=False):
        super(VGG7, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # convolutional layers
        if bn:
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(in_features=128 * 2 * 2, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes)

    def forward(self, x: torch.Tensor, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        ftrs = torch.relu(self.fc2(x))
        x = self.fc3(ftrs)
        if return_features:
            return x, ftrs
        return x


class CifarAEC(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, bn=True, dropout_p=0.2, vae=False, z_dim=128):
        super(CifarAEC, self).__init__()
        self.vae = vae
        self.z_dim = z_dim

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64) if bn else nn.Identity(),
            nn.Dropout(dropout_p),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.Dropout(dropout_p),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.Dropout(dropout_p),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256) if bn else nn.Identity(),
            nn.Dropout(dropout_p)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)
        )

        if self.vae:
            self.mean_fc = nn.Linear(256 * 2 * 2, self.z_dim)
            self.logvar_fc = nn.Linear(256 * 2 * 2, self.z_dim)
            self.project_fc = nn.Linear(self.z_dim, 256 * 2 * 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # [batch, -1, 4, 4]
            nn.ReLU(),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),  # [batch, -1, 8, 8]
            nn.ReLU(),
            nn.BatchNorm2d(128) if bn else nn.Identity(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # [batch, -1, 16, 16]
            nn.ReLU(),
            nn.BatchNorm2d(64) if bn else nn.Identity(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def compute_z(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.rand_like(std)
            z = mean + eps * std
        else:
            z = mean
        return z

    def forward(self, x):
        # encoder
        enc_fmap = self.encoder(x)

        code = enc_fmap.flatten(1)

        if self.vae:
            mean, logvar = self.mean_fc(code), self.logvar_fc(code)
            z = self.compute_z(mean, logvar)
            code = self.project_fc(z)
            enc_fmap = torch.reshape(code, enc_fmap.shape)

        cls = self.classifier(code)

        decoded = self.decoder(enc_fmap)

        if self.vae:
            return (code, mean, logvar, z), decoded, cls

        return code, decoded, cls


class CNNOriginalFedAvgWithAE(torch.nn.Module):

    def __init__(self, only_digits=True, vae=False, z_dim=128):
        super(CNNOriginalFedAvgWithAE, self).__init__()
        self.only_digits = only_digits
        self.vae = vae
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 10 if only_digits else 62)
        )

        if self.vae:
            self.mean_fc = nn.Linear(512, self.z_dim)
            self.logvar_fc = nn.Linear(512, self.z_dim)
            self.project_fc = nn.Linear(self.z_dim, 512)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 32, kernel_size=7, stride=1),  # 16x16 => 22x22
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=7, stride=1),  # 22x22 => 28x28
            nn.Sigmoid()
        )

    def compute_z(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        code = self.encoder(x)

        if self.vae:
            mean, logvar = self.mean_fc(code), self.logvar_fc(code)
            z = self.compute_z(mean, logvar)
            code = self.project_fc(z)

        cls = self.classifier(code)
        decoded = self.decoder(torch.reshape(code, (-1, 2, 16, 16)))

        if self.vae:
            return (code, mean, logvar, z), decoded, cls

        return code, decoded, cls

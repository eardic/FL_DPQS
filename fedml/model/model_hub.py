import logging

import torch.nn as nn
import torchvision.models

from fedml.model.cv.cnn import CNN_DropOut, CNN_OriginalFedAvg, CNN_Cifar10_S, CNN_Cifar10_M, CNN_Cifar10_L, VGG7, \
    CifarAEC, CNNOriginalFedAvgWithAE
from fedml.model.cv.darts import genotypes
from fedml.model.cv.darts.model import NetworkCIFAR
from fedml.model.cv.darts.model_search import Network
from fedml.model.cv.efficientnet import EfficientNet
from fedml.model.cv.mnist_gan import Generator, Discriminator
from fedml.model.cv.mobilenet import mobilenet
from fedml.model.cv.mobilenet_v3 import MobileNetV3
from fedml.model.cv.resnet import resnet56
from fedml.model.cv.resnet56 import resnet_client, resnet_server
from fedml.model.cv.resnet_cifar import resnet18_cifar
from fedml.model.cv.resnet_gn import resnet18
from fedml.model.linear.lr import LogisticRegression
from fedml.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow, RNN_FedShakespeare


def create(args, output_dim):
    global model
    model_name = args.model
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn" and args.dataset == "mnist" and args.federated_optimizer.lower() == 'fedavg':
        logging.info("CNN_OrgFedAVG + MNIST")
        model = CNN_OriginalFedAvg()
    elif model_name == "cnn" and args.dataset == "mnist":
        logging.info("CNN + MNIST")
        model = CNN_DropOut(False)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_FedShakespeare()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        if args.federated_optimizer == "FedGKT":
            client_model = resnet_client.resnet8_56(c=output_dim)
            server_model = resnet_server.resnet56_server(c=output_dim)
            model = (client_model, server_model)
        else:
            model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "squeezenet":
        from torchvision.models import squeezenet1_1
        model = squeezenet1_1(num_classes=output_dim)
    elif model_name == "vgg11":
        from torchvision.models import vgg11
        model = vgg11(num_classes=output_dim)
    elif model_name == "resnet18" and args.dataset == "cifar10":
        model = resnet18_cifar()
    elif model_name == "cnn_sm" and args.dataset == "cifar10":
        model = CNN_Cifar10_S()
    elif model_name == "cnn_m" and args.dataset == "cifar10":
        model = CNN_Cifar10_M()
    elif model_name == "cnn_l" and args.dataset == "cifar10":
        model = CNN_Cifar10_L()
    elif model_name == "aec" and args.dataset == "cifar10":
        logging.info("CifarAEC")
        model = CifarAEC(vae=args.aec_use_vae,
                         z_dim=args.aec_vae_zdim,
                         num_classes=output_dim,
                         dropout_p=args.aec_dropout if hasattr(args, "aec_dropout") else 0.2)
    elif model_name == "aec" and args.dataset == "mnist":
        logging.info("CNNOriginalFedAvgWithAE")
        model = CNNOriginalFedAvgWithAE(vae=args.aec_use_vae, z_dim=args.aec_vae_zdim)
    elif model_name == "vgg7":
        logging.info(f"VGG7 {output_dim} Class")
        model = VGG7(in_channels=3, num_classes=output_dim, bn=False)
    elif model_name == "vgg7_bn":
        logging.info(f"VGG7_BN {output_dim} Class")
        model = VGG7(in_channels=3, num_classes=output_dim, bn=True)
    elif model_name == "mobilenet_v3":
        """model_mode \in {LARGE: 5.15M, SMpALL: 2.94M}"""
        model = MobileNetV3(model_mode="LARGE")
    elif model_name == "mobilenet_v3_sm":
        model = torchvision.models.mobilenet_v3_small(num_classes=output_dim)
    elif model_name == "efficientnet":
        model = EfficientNet()
    elif model_name == "efficientnet-b0":
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=output_dim)
    elif model_name == "darts" and args.dataset == "cifar10":
        if args.stage == "search":
            criterion = nn.CrossEntropyLoss()
            model = Network(args.init_channels, output_dim, args.layers, criterion)
        elif args.stage == "train":
            genotype = genotypes.FedNAS_V1
            model = NetworkCIFAR(args.init_channels, output_dim, args.layers, args.auxiliary, genotype)
    elif model_name == "GAN" and args.dataset == "mnist":
        gen = Generator()
        disc = Discriminator()
        model = (gen, disc)
    elif model_name == "lenet" and hasattr(args, "deeplearning_backend") and args.deeplearning_backend == "mnn":
        from .mobile.mnn_lenet import create_mnn_lenet5_model
        create_mnn_lenet5_model(args.global_model_file_path)
        model = None  # for server MNN, the model is saved as computational graph and then send it to clients.
    elif model_name == "resnet20" and hasattr(args, "deeplearning_backend") and args.deeplearning_backend == "mnn":
        from .mobile.mnn_resnet import create_mnn_resnet20_model

        create_mnn_resnet20_model(args.global_model_file_path)
        model = None  # for server MNN, the model is saved as computational graph and then send it to clients.
    elif model_name == "resnet18":
        from torchvision.models import resnet18 as r18
        model = r18(num_classes=output_dim)
    else:
        raise Exception("no such model definition, please check the argument spelling or customize your own model")
    return model

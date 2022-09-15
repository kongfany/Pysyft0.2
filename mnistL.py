from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#
# mnist_data = datasets.MNIST("./mnist_data2", download=True, train=True,
#                             transform=transforms.Compose([
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.13066062,), (0.30810776,))
#                             ]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
import syft as sy
import torchvision

hook = sy.TorchHook(torch)
qin = sy.VirtualWorker(hook=hook, id="qin")
zheng = sy.VirtualWorker(hook=hook, id="zheng")
#设定参数
args = {
    'use_cuda': True,
    'batch_size': 64,
    'test_batch_size': 1000,
    'lr': 0.01,
    'log_interval': 10,
    'epochs': 10
}
use_cuda = args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# 下面是训练数据，需要分发给远处打工人
federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST('./minist_data_test', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    .federate((qin, zheng)),
    batch_size=args['batch_size'], shuffle=True
)
# 下面是测试数据，在我们本地
test_loader = DataLoader(
    datasets.MNIST('./minist_data_test', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['batch_size'], shuffle=True
)
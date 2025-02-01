# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2018, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" MNIST model: Including train and test code """


import torch
from torch import nn
from torch.nn import functional
from torch import optim
from torchvision import datasets, transforms

from aimet_common.data_cache_utility import is_cache_env_set, is_mnist_cache_present, copy_mnist_to_cache, copy_cache_mnist_to_local_build
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

# Training settings
args = {
    "batch_size": 64,
    "test_batch_size": 1000,
    "epochs": 2,
    "lr": 0.01,
    "momentum": 0.5,
    "no_cuda": True,
    "seed": 1,
    "log_interval": 10,
}


class DataLoaderMnist:
    """ A dataloader for the MNIST dataset """

    def __init__(self, cuda, seed, shuffle, train_batch_size=64, test_batch_size=100):
        """
        Constructor

        :param cuda: If True, data will be loaded in GPU memory
        :param seed: Seed to use for randomization (to help with reproducibility)
        :param shuffle: If we want data to be shuffled
        :param train_batch_size: Batch size for train data
        :param test_batch_size: Batch size for test data
        """

        self._cuda = cuda
        self._seed = seed
        self._shuffle = shuffle
        self._train_batch_size = train_batch_size
        self._test_batch_size = test_batch_size
        # set the GPU flags appropriately
        # to allocate data on GPU or CPU
        self._use_cuda = self._cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self._use_cuda else "cpu")
        self._kwargs = {'num_workers': 1, 'pin_memory': True} if self._use_cuda else {}
        # set the seed value
        torch.manual_seed(self._seed)
        mnist_download = True

        if is_mnist_cache_present():
            copy_cache_mnist_to_local_build()
            mnist_download = False

        # train loader
        self.train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=mnist_download,
                                                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                                                     transforms.Normalize((0.5307,), (0.9081,))])),
                                                        batch_size=self._train_batch_size, shuffle=self._shuffle, **self._kwargs)

        # test loader
        self.test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,
                                                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                                                    transforms.Normalize((0.5307,), (0.9081,))])),
                                                       batch_size=self._test_batch_size, shuffle=self._shuffle, **self._kwargs)

        if not is_mnist_cache_present() and is_cache_env_set():
            copy_mnist_to_cache()


class Net(nn.Module):
    """ Mnist Model """
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        """ Constructor """

        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=(2, 2))
        self.conv2_drop = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout2d()
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, *inputs):
        """
        Overriden implementation for the forward pass
        :param inputs: ONe or more inputs for the model
        :return: Output of the forward pass
        """

        x = self.relu1(self.maxpool1(self.conv1(*inputs)))
        x = self.relu2(self.maxpool2(self.conv2_drop(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return self.log_softmax(x)


class ExtendedNet(nn.Module):
    """ Mnist Model """

    def __init__(self):
        """ Constructor """

        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=(2, 2), bias=False)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=(2, 2))
        self.fc1 = nn.Linear(3*3*64, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, *inputs):
        """
        Overriden implementation for the forward pass
        :param inputs: ONe or more inputs for the model
        :return: Output of the forward pass
        """

        x = functional.relu(functional.max_pool2d(self.conv1(*inputs), 2))
        x = functional.relu(functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = functional.relu(functional.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = functional.relu(self.fc1(x))
        x = functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)


def train(model, epochs, num_batches=0, batch_size=50, batch_callback=None, use_cuda=False):
    """
    Train the MNIST model
    :param model: Model
    :param epochs: Number of epochs to train
    :param num_batches: Number of batches to train (to used if we want to train for less than an epoch. None otherwise)
    :param batch_size: Batch size to use for train dataloader
    :param batch_callback: Callback method called every batch
    :return: None
    """

    data_loader = DataLoaderMnist(cuda=use_cuda, seed=1, shuffle=False,
                                  train_batch_size=batch_size)

    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(data_loader.train_loader):
            data, target = data.to(data_loader.device), target.to(data_loader.device)
            optimizer.zero_grad()
            output = model(data)
            loss = functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if num_batches != 0:
                if batch_idx == num_batches:
                    return

            if batch_callback is not None:
                batch_callback(model, batch_idx)

            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                               len(data_loader.train_loader.dataset),
                                                                               100. * batch_idx / len(data_loader.train_loader),
                                                                               loss.item()))

def evaluate(model, iterations, use_cuda=False):
    """
    Evaluate the MNIST model
    :param model: Model
    :param iterations: Number of iterations to evaluate for
    :param use_cuda: If True, inference is run in CUDA mode
    :return: Output accuracy
    """

    logger.debug("Allocating input and target tensors on GPU : %r", use_cuda)

    # create the instance of data loader
    data_loader = DataLoaderMnist(cuda=use_cuda, seed=1, shuffle=False, train_batch_size=64, test_batch_size=100)

    model.eval()
    total = 0
    correct = 0
    current_iterations = 0

    with torch.no_grad():
        for inputs, labels in data_loader.test_loader:
            inputs, labels = inputs.to(data_loader.device), labels.to(data_loader.device)
            output = model(inputs)
            current_iterations += 1
            _, predicted = torch.max(output.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if iterations is not None:
                if current_iterations >= iterations:
                    break

    accuracy = correct / total
    return accuracy

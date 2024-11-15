# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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


"""
Creates trainer for Image-Net dataset
"""
import logging

from tqdm import tqdm
import torch
from torch import nn, optim

from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator

logger = logging.getLogger('Trainer')


class ImageNetTrainer:
    """
    For training a model using the ImageNet dataset.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, images_dir: str, image_size: int, batch_size: int = 128,
                 num_workers: int = 32, num_train_samples_per_class: int = None):
        """
        :param images_dir: The path to the data directory
        :param image_size: The length of the image
        :param batch_size: The batch size to use for training and validation
        :param num_workers: Indiicates to the data loader how many sub-processes to use for data loading.
        :param num_train_samples_per_class: Number of samples to use per class.
        """

        self._train_loader = ImageNetDataLoader(images_dir=images_dir, image_size=image_size, batch_size=batch_size,
                                                is_training=True, num_workers=num_workers,
                                                num_samples_per_class=num_train_samples_per_class).data_loader

        self._evaluator = ImageNetEvaluator(images_dir=images_dir, image_size=image_size, batch_size=batch_size,
                                            num_workers=num_workers)

    def _train_loop(self, model: nn.Module, criterion: torch.nn.modules.loss, optimizer: torch.optim,
                    max_iterations: int, current_epoch: int, max_epochs: int,
                    debug_steps: int = 1000, use_cuda: bool = False):
        """
        Train the specified model using the ImageNet dataset for one epoch.
        :param model: The model to train.
        :param criterion: loss function to optimize
        :param optimizer: optimizer function
        :param max_iterations: total number of iterations to train
        :param current_epoch: current epoch for model training, used for reporting logs in debug steps
        :param max_epochs: total epoch for model training, used for reporting logs in debug steps
        :param debug_steps: number of training iterations to report accuracy/loss metrics
        :param use_cuda: If True then use GPU device

        :return: None
        """
        # pylint: disable-msg=too-many-locals
        device = torch.device('cpu')
        if use_cuda:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                logger.error('use_cuda is selected but no cuda device found.')
                raise RuntimeError("Found no CUDA Device while use_cuda is selected")

        # switch to training mode
        model = model.to(device)
        model.train()

        avg_loss = 0.0

        for i, (images, target) in tqdm(enumerate(self._train_loader), total=max_iterations):
            if i == max_iterations:
                break

            images = images.to(device)
            target = target.to(device)

            # compute model output
            output = model(images)
            loss = criterion(output, target)
            avg_loss += loss.item()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % debug_steps == 0:
                eval_accuracy = self._evaluator.evaluate(model, use_cuda=use_cuda)
                logger.info('Epoch #%d/%d: iteration #%d/%d: Global Avg Loss=%f, Eval Accuracy=%f',
                            current_epoch, max_epochs, curr_iter, max_iterations,
                            avg_loss / curr_iter, eval_accuracy)
                # switch to training mode after evaluation
                model.train()

        eval_accuracy = self._evaluator.evaluate(model, use_cuda=use_cuda)
        print("eval : ", eval_accuracy)
        logger.info('At the end of Epoch #%d/%d: Global Avg Loss=%f, Eval Accuracy=%f',
                    current_epoch, max_epochs, avg_loss / max_iterations, eval_accuracy)

    def train(self, model: nn.Module, max_epochs: int = 20, learning_rate: int = 0.1,
              weight_decay: float = 1e-4, decay_rate: float = 0.1, learning_rate_schedule: list = None,
              debug_steps: int = 1000, use_cuda: bool = False):
        """
        Train the specified model using the ImageNet dataset.
        :param model: The model to train.
        :param max_epochs: Maximum number of epochs to use in training.
        :param learning_rate: Learning rate
        :param weight_decay: Weight decay
        :param decay_rate: Multiplicative factor of learning rate decay
        :param learning_rate_schedule: Adjust the learning rate based on the number of epochs
        :param debug_steps: number of training iterations to report accuracy/loss metrics
        :param use_cuda: If True then use GPU device

        :return: None
        """
        max_iterations = len(self._train_loader)

        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

        if learning_rate_schedule:
            learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, learning_rate_schedule,
                                                                     gamma=decay_rate)

        for current_epoch in range(max_epochs):
            self._train_loop(model, loss, optimizer, max_iterations, current_epoch + 1, max_epochs, debug_steps,
                             use_cuda)

            if learning_rate_schedule:
                learning_rate_scheduler.step()

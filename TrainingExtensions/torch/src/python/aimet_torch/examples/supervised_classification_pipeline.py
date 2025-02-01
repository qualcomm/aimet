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

""" Supervised classification pipeline for ImageNet (to be deprecated in favor of using DL Pipelines) """

# BSD 3-Clause License
#
# Copyright (c) 2018, PyTorch team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import collections

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy
import torch
from torch import nn

from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


def convert_tensor(input_, device=None, non_blocking=False):
    """
    This is a copy from ignite/_utils, use this until the non_blocking issue
    :param input_:
    :dummy
    :param device:
    :param non_blocking:
    :return:
    """
    # TODO: Replace .cuda with .to for Pytorch 4.1
    if torch.is_tensor(input_):
        if device == 'cpu' or device == torch.device(type='cpu'):
            input_ = input_.to(device=device)
        else:
            input_ = input_.cuda(device=device, non_blocking=non_blocking)

    elif isinstance(input_, str):
        return input_

    elif isinstance(input_, collections.Mapping):  # pylint: disable=no-member
        return {k: convert_tensor(sample, device=device) for k, sample in input_.items()}

    elif isinstance(input_, collections.Sequence):  # pylint: disable=no-member
        return [convert_tensor(sample, device=device) for sample in input_]

    else:
        raise TypeError(("input must contain tensors, dicts or lists; found {}"
                         .format(type(input_))))
    return input_


def _prepare_batch(batch, device=None):
    # TODO: This is a copy from ignite/engine/__init__.py, use this until the non_blocking issue
    # https://github.com/pytorch/ignite/issues/231
    # is solved
    x, y = batch
    return convert_tensor(x, device=device), convert_tensor(y, device=device, non_blocking=True)


def create_supervised_classification_trainer(model, loss_fn, optimizer, val_loader,
                                             learning_rate_scheduler, callback=None, use_cuda=None):
    """
    Todo: Add description
    :param model:
    :param loss_fn:
    :param optimizer:
    :param val_loader:
    :param learning_rate_scheduler:
    :param callback:
    :param use_cuda:
    :return:
    """

    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError('Trying to run using cuda, while cuda is not available')

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
            print("Using {} gpus for training".format(torch.cuda.device_count()))
    else:
        device = torch.device('cpu')

    trainer = create_trainer(model=model, optimizer=optimizer, loss_fn=loss_fn,
                             metrics={'top_1_accuracy': Accuracy(),
                                      'top_5_accuracy': TopKCategoricalAccuracy(),
                                      'loss': Loss(loss_fn),
                                      },
                             device=device)

    evaluator = create_supervised_classification_evaluator(model, loss_fn, use_cuda)

    if learning_rate_scheduler:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda _: learning_rate_scheduler.step())

    if callback is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, callback, model)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results, optimizer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluation, evaluator, val_loader)

    return trainer, evaluator


def create_trainer(model, optimizer, loss_fn, metrics, device=None):
    """
    Todo: Add description
    :param model:
    :param optimizer:
    :param loss_fn:
    :param metrics:
    :param device:
    :return:
    """

    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return y_pred, y

    engine = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_supervised_classification_evaluator(model, loss_fn, use_cuda):
    """
    Create an evaluator
    :param model:
    :param loss_fn:
    :param use_cuda:
    :return:
    """

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
        # multiple GPUs, we can remove this as well
        torch.backends.cudnn.benchmark = True
        if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
            logger.info("Using %d gpus for training", torch.cuda.device_count())
    else:
        device = torch.device('cpu')

    evaluator = create_supervised_evaluator(model,
                                            metrics={'top_1_accuracy': Accuracy(),
                                                     'top_5_accuracy': TopKCategoricalAccuracy(),
                                                     'loss': Loss(loss_fn)},
                                            device=device)
    return evaluator


def create_stand_alone_supervised_classification_evaluator(model, loss_fn, use_cuda):
    """
    Standalone classification evaluator
    :param model:
    :param loss_fn:
    :param use_cuda:
    :return:
    """

    evaluator = create_supervised_classification_evaluator(model, loss_fn, use_cuda)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_evaluation_results(engine):       # pylint: disable=unused-variable
        metrics = engine.state.metrics
        logger.info('Validation Results - Top 1 accuracy: {:.2%} Top 5 accuracy: {:.2%}  Avg loss: {:.2f}'
                    .format(metrics['top_1_accuracy'], metrics['top_5_accuracy'], metrics['loss']))

    return evaluator


def log_training_results(trainer, optimizer):
    """
    Log utility for train results at the end of an epoch
    :param trainer:
    :param optimizer:
    :return:
    """

    metrics = trainer.state.metrics
    logger.info('Training Results - Epoch: {}  Top 1 accuracy: {:.2%}  '
                'Top 5 accuracy: {:.2%}  Avg loss: {:.2f}  Learning rate {:.2E}'
                .format(trainer.state.epoch, metrics['top_1_accuracy'], metrics['top_5_accuracy'],
                        metrics['loss'], optimizer.param_groups[0]['lr']))


def run_evaluation(trainer, evaluator, val_loader):
    """
    Run an evaluation
    :param trainer:
    :param evaluator:
    :param val_loader:
    :return:
    """
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    logger.info('Validation Results - Epoch: {}  Top 1 accuracy: {:.2%}  '
                'Top 5 accuracy: {:.2%}  Avg loss: {:.2f}'
                .format(trainer.state.epoch, metrics['top_1_accuracy'], metrics['top_5_accuracy'],
                        metrics['loss']))

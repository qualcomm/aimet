# =============================================================================
#
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
#
# =============================================================================


"""
This file demonstrates the use of quantization using AIMET range learning
technique.
"""

import argparse
import copy
import logging
import os
from datetime import datetime
from functools import partial
from typing import Tuple
import torch
from torchvision import models

# imports for AIMET
from aimet_common.defs import QuantScheme, EvalFunction
from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.v1.quantsim import QuantizationSimModel

# imports for data pipelines
from Examples.common import image_net_config
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_trainer import ImageNetTrainer


logger = logging.getLogger('TorchRangeLearning')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format=formatter)

###
# This script utilizes AIMET to perform Range Learning on a resnet18 pretrained model
# with the ImageNet data set. It should re-create the same performance numbers as published in the
# AIMET release for the particular scenario as described below.

# Scenario parameters:
#    - AIMET Range Learning using simulation model
#    - Quant Scheme: QuantScheme.training_range_learning_with_tf_init
#    - rounding_mode: 'nearest'
#    - default_output_bw: 8, default_param_bw: 8
#	 - Encoding computation using 5 batches of data
#    - Input shape: [1, 3, 224, 224]
#    - Learning rate: 0.001
#    - Decay Steps: 5
###


class ImageNetDataPipeline:
    """
    Provides APIs for model quantization using evaluation and finetuning.
    """

    def __init__(self, _config: argparse.Namespace):
        """
        :param _config:
        """
        self._config = _config

    def evaluate(self, model: torch.nn.Module, iterations: int = None, use_cuda: bool = False) -> float:
        """
        Evaluate the specified model using the specified number of samples from the validation set.
        :param model: The model to be evaluated.
        :param iterations: The number of batches of the dataset.
        :param use_cuda: If True then use a GPU for inference.
        :return: The accuracy for the sample with the maximum accuracy.
        """

        # your code goes here instead of the example from below

        evaluator = ImageNetEvaluator(self._config.dataset_dir, image_size=image_net_config.dataset['image_size'],
                                      batch_size=image_net_config.evaluation['batch_size'],
                                      num_workers=image_net_config.evaluation['num_workers'])

        return evaluator.evaluate(model, iterations, use_cuda)

    def finetune(self, model: torch.nn.Module):
        """
        Finetunes the model.  The implemtation provided here is just an example,
        provide your own implementation if needed.
        :param model: The model to finetune.
        """

        # Your code goes here instead of the example from below

        trainer = ImageNetTrainer(self._config.dataset_dir, image_size=image_net_config.dataset['image_size'],
                                  batch_size=image_net_config.train['batch_size'],
                                  num_workers=image_net_config.train['num_workers'])

        trainer.train(model, max_epochs=self._config.epochs, learning_rate=self._config.learning_rate,
                      learning_rate_schedule=self._config.learning_rate_schedule, use_cuda=self._config.use_cuda)

        torch.save(model, os.path.join(self._config.logdir, 'finetuned_model.pth'))


def calculate_quantsim_accuracy(model: torch.nn.Module, evaluator: EvalFunction,
                                use_cuda: bool = False)-> Tuple[torch.nn.Module, float]:
    """
    Calculates model accuracy on quantized simulator and returns quantized model with accuracy.
    :param model: the loaded model
    :param evaluator: the Eval function to use for evaluation
    :param use_cuda: the cuda device.
    :return: a tuple of quantizer and accuracy of model on this quantizer
    """

    if use_cuda:
        model.to(torch.device('cuda'))
        dummy_input = torch.rand(1, 3, 224, 224).cuda()
    else:
        dummy_input = torch.rand(1, 3, 224, 224)

    # Number of batches to use for computing encodings
    # Only 5 batches are used here to speed up the process, also the
    # number of images in these 5 batches should be sufficient for
    # compute encodings
    iterations = 5

    quantsim = QuantizationSimModel(model=model, quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                    dummy_input=dummy_input, rounding_mode='nearest',
                                    default_output_bw=8, default_param_bw=8, in_place=False)

    quantsim.compute_encodings(forward_pass_callback=partial(evaluator, use_cuda=use_cuda),
                               forward_pass_callback_args=iterations)

    accuracy = evaluator(quantsim.model, use_cuda=use_cuda)

    return quantsim, accuracy


def apply_cross_layer_equalization(model: torch.nn.Module, input_shape: tuple):
    """
    Applies CLE on the model and calculates model accuracy on quantized simulator
    Applying CLE on the model inplace consists of:
        Batch Norm Folding
        Cross Layer Scaling
        High Bias Fold
        Converts any ReLU6 into ReLU.
    :param model: the loaded model
    :param input_shape: the shape of the input to the model
    :return:
    """

    equalize_model(model, input_shape)


def quantize_and_finetune(config: argparse.Namespace):
    """
    1. Instantiates Data Pipeline for evaluation
    2. Loads the pretrained resnet18 Pytorch model
    3. Calculates Model accuracy
        3.1. Calculates floating point accuracy
        3.2. Calculates Quant Simulator accuracy
    4. Applies AIMET CLE
        4.1. Applies AIMET CLE and calculates QuantSim accuracy
        4.2. Finetune the quantized model
    :param config: This argparse.Namespace config expects following parameters:
                   tfrecord_dir: Path to a directory containing ImageNet TFRecords.
                                This folder should conatin files starting with:
                                'train*': for training records and 'validation*': for validation records
                   use_cuda: A boolean var to indicate to run the test on GPU.
                   logdir: Path to a directory for logging.
    """
    # Instantiate Data Pipeline for evaluation and training
    data_pipeline = ImageNetDataPipeline(config)

    # Load the pretrained resnet18 model
    model = models.resnet18(pretrained=True)
    if config.use_cuda:
        model.to(torch.device('cuda'))
    model = model.eval()

    # Calculate floating point accuracy
    accuracy = data_pipeline.evaluate(model, use_cuda=config.use_cuda)
    logger.info("Original Model Top-1 accuracy = %.2f", accuracy)

    # It is recommended to fold Batch-norms before running on target.
    # But CLE needs original model to operate upon, so making a copy
    # of model here, before folding BN layers.
    logger.info("Starting Model Quantization")
    BN_folded_model = copy.deepcopy(model)
    _ = fold_all_batch_norms(BN_folded_model, input_shapes=(1, 3, 224, 224))
    _, accuracy = calculate_quantsim_accuracy(model=BN_folded_model, evaluator=data_pipeline.evaluate,
                                              use_cuda=_config.use_cuda)
    logger.info("Batch Norm applied accuracy on Quant Simulator = %.2f", accuracy)

    # Applying Cross Layer Equalization (CLE) on the model
    apply_cross_layer_equalization(model=model, input_shape=(1, 3, 224, 224))

    # Calculate and log accuracy on Quant Simulator
    quantsim, stats = calculate_quantsim_accuracy(model=model, evaluator=data_pipeline.evaluate, use_cuda=_config.use_cuda)
    logger.info("Top-1 accuracy on Quant Simulator = %.2f", stats)

    with open(os.path.join(config.logdir, 'log.txt'), "w") as outfile:
        outfile.write("%s\n\n" % (stats))

    logger.info("Model Quantization Complete")

    # Finetune the quantized model
    logger.info("Starting Model Finetuning")
    data_pipeline.finetune(quantsim.model)

    # Calculate and log the accuracy of quantized-finetuned model
    accuracy = data_pipeline.evaluate(quantsim.model, use_cuda=config.use_cuda)
    logger.info("Finetuned Quantized Model Top-1 accuracy = %.2f", accuracy)
    logger.info("Model Finetuning Complete")

    # Save the quantized model
    torch.save(model, os.path.join(config.logdir, 'quantized_model.pth'))


if __name__ == '__main__':
    default_logdir = os.path.join("benchmark_output", "range_learning" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    parser = argparse.ArgumentParser(
        description='Quantization aware training of pretrained ResNet18 model using range learning')

    parser.add_argument('--dataset_dir', type=str,
                        required=True,
                        help="Path to a directory containing ImageNet dataset.\n\
                              This folder should conatin at least 2 subfolders:\n\
                              'train': for training dataset and 'val': for validation dataset")
    parser.add_argument('--use_cuda', action='store_true',
                        required=True,
                        help='Add this flag to run the test on GPU.')

    parser.add_argument('--logdir', type=str,
                        default=default_logdir,
                        help="Path to a directory for logging.\
                              Default value is 'benchmark_output/weight_svd_<Y-m-d-H-M-S>'")

    parser.add_argument('--epochs', type=int,
                        default=15,
                        help="Number of epochs for finetuning.\n\
                              Default is 15")
    parser.add_argument('--learning_rate', type=float,
                        default=1e-2,
                        help="A float type learning rate for model finetuning.\n\
                              default is 0.01")
    parser.add_argument('--learning_rate_schedule', type=list,
                        default=[5, 10],
                        help="A list of epoch indices for learning rate schedule used in finetuning.\n\
                              Check https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR for more details.\n\
                              default is [5, 10]")

    _config = parser.parse_args()

    os.makedirs(_config.logdir, exist_ok=True)

    fileHandler = logging.FileHandler(os.path.join(_config.logdir, "test.log"))
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    if _config.use_cuda and not torch.cuda.is_available():
        logger.error('use_cuda is selected but no cuda device found.')
        raise RuntimeError("Found no CUDA Device while use_cuda is selected")

    quantize_and_finetune(_config)

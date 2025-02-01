# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Sub-sample data for weight reconstruction for channel pruning feature """

from typing import Iterator, Callable, Tuple, Union, List
import abc
import math
import numpy as np

import torch
import torch.utils.data
import torch.nn

# Import AIMET specific modules
from aimet_common.utils import AimetLogger
from aimet_common.input_match_search import InputMatchSearch
from aimet_torch import utils

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ChannelPruning)


class StopForwardException(Exception):
    """ Dummy exception to early-terminate forward-pass """


class LayerSubSampler(abc.ABC):
    """
    The LayerSubSampler declares operations common to all the layers. For now we only support Conv2d and Linear layers
    """
    @abc.abstractmethod
    def verify_layers(self, orig_layer: torch.nn.Module, pruned_layer: torch.nn.Module):
        """
        :param orig_layer: original layer from uncompressed model
        :param pruned_layer: pruned layer from compressed model
        :return:
        """

    @abc.abstractmethod
    def get_number_of_batches(self, data_loader: Iterator, orig_layer: torch.nn.Module, num_reconstruction_samples: int,
                              samples_per_image: int) -> int:
        """
        :param orig_layer:
        :param data_loader: data loader
        :param num_reconstruction_samples: The number of reconstruction samples
        :param samples_per_image: Number of samples per image
        :return: number of batches
        """

    @abc.abstractmethod
    def get_sub_sampled_data(self, orig_layer: torch.nn.Module, input_data: np.ndarray,
                             output_data: np.ndarray, samples_per_image: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param orig_layer: original layer
        :param input_data: input data collected from compressed model
        :param output_data: output data collected from un compressed model
        :param samples_per_image: samples per image (default 10)
        :return: (sub sampled input data, sub sampled output data)
        """


class Conv2dSubSampler(LayerSubSampler):
    """
    Sub Sampler for Linear layer
    """
    def verify_layers(self, orig_layer: torch.nn.Module, pruned_layer: torch.nn.Module):

        assert isinstance(orig_layer, torch.nn.Conv2d)

        assert isinstance(pruned_layer, torch.nn.Conv2d)

        assert orig_layer.dilation == (1, 1), 'No Conv2d layers supported for dilation other than (1, 1)'

        assert pruned_layer.dilation == (1, 1), 'No Conv2d layers supported for dilation other than (1, 1)'

    def get_number_of_batches(self, data_loader: Iterator, orig_layer: torch.nn.Module, num_reconstruction_samples: int,
                              samples_per_image: int) -> int:

        total_num_of_images = int(num_reconstruction_samples / samples_per_image)

        # number of possible batches - round up using ceil
        num_of_batches = math.ceil(total_num_of_images / data_loader.batch_size)
        return num_of_batches

    def get_sub_sampled_data(self, orig_layer: torch.nn.Module, input_data: np.ndarray,
                             output_data: np.ndarray, samples_per_image: int) -> Tuple[np.ndarray, np.ndarray]:

        layer_attributes = (orig_layer.kernel_size, orig_layer.stride, orig_layer.padding)

        # get the sub sampled input and output data
        sub_sampled_inp_data, sub_sampled_out_data = InputMatchSearch.subsample_data(layer_attributes, input_data,
                                                                                     output_data, samples_per_image)

        return sub_sampled_inp_data, sub_sampled_out_data


class LinearSubSampler(LayerSubSampler):
    """
    Sub Sampler for Linear layer
    """
    def verify_layers(self, orig_layer: torch.nn.Module, pruned_layer: torch.nn.Module):

        assert isinstance(orig_layer, torch.nn.Linear)

        assert isinstance(pruned_layer, torch.nn.Linear)

    def get_number_of_batches(self, data_loader: Iterator, orig_layer: torch.nn.Module, num_reconstruction_samples: int,
                              samples_per_image: int) -> int:

        assert isinstance(orig_layer, torch.nn.Linear)

        total_num_of_images = int(num_reconstruction_samples / orig_layer.out_features)

        # number of possible batches - round up using ceil
        num_of_batches = math.ceil(total_num_of_images / data_loader.batch_size)
        return num_of_batches

    def get_sub_sampled_data(self, orig_layer: torch.nn.Module, input_data: np.ndarray,
                             output_data: np.ndarray, samples_per_image: int) -> Tuple[np.ndarray, np.ndarray]:

        # just return the input and output data as it is. No sub sampling needed
        return input_data, output_data


class DataSubSampler:
    """ Utilities to sub-sample data for weight reconstruction """

    @staticmethod
    def _forward_pass(model: torch.nn.Module, batch: Union[torch.Tensor, List, Tuple]):
        """
        forward pass depending model allocation on CPU / GPU till StopForwardException
        :param model: model
        :param batch: batch
        """

        # get the model's device placement information
        device = utils.get_device(model)
        # place the batch to appropriate device
        batch = utils.change_tensor_device_placement(batch, device)

        if isinstance(batch, torch.Tensor):
            batch = [batch]

        try:
            # keep the model in eval mode
            with utils.in_eval_mode(model), torch.no_grad():
                _ = model(*batch)
        except StopForwardException:
            pass

    @staticmethod
    def _register_fwd_hook_for_layer(layer: torch.nn.Module, hook: Callable) -> torch.utils.hooks.RemovableHandle:
        """
        register forward hook for given layer
        :param layer: layer
        :param hook: hook function
        :return: hook handle
        """
        hook_handle = layer.register_forward_hook(hook)
        return hook_handle

    @classmethod
    def get_sub_sampled_data(cls, orig_layer: Union[torch.nn.Conv2d, torch.nn.Linear],
                             pruned_layer: Union[torch.nn.Conv2d, torch.nn.Linear],
                             orig_model: torch.nn.Module, comp_model: torch.nn.Module, data_loader: Iterator,
                             num_reconstruction_samples: int) -> (np.ndarray, np.ndarray):
        # pylint: disable=too-many-locals
        """
        Get all the input data from pruned model and output data from original model

        :param orig_layer: original layer
        :param pruned_layer: pruned layer
        :param orig_model: original model, un-pruned, used to provide the actual outputs
        :param comp_model: comp. model, this is potentially already pruned in the upstreams layers of given layer name
        :param data_loader: data loader
        :param num_reconstruction_samples: The number of reconstruction samples
        :return: input_data, output_data
        """

        def _hook_to_collect_input_data(module, inp_data, _):  # pylint: disable=unused-argument
            """
            hook to collect input data
            """
            inp_data = utils.to_numpy(inp_data[0])
            pruned_layer_inp_data.append(inp_data)
            raise StopForwardException

        def _hook_to_collect_output_data(module, _, out_data):  # pylint: disable=unused-argument
            """
            hook to collect output data
            """
            out_data = utils.to_numpy(out_data)
            orig_layer_out_data.append(out_data)
            raise StopForwardException

        if isinstance(orig_layer, torch.nn.Conv2d) and isinstance(pruned_layer, torch.nn.Conv2d):
            sub_sampler = Conv2dSubSampler()

        elif isinstance(orig_layer, torch.nn.Linear) and isinstance(pruned_layer, torch.nn.Linear):
            sub_sampler = LinearSubSampler()

        else:
            raise ValueError('Layer type not supported!')

        # verify the layers
        sub_sampler.verify_layers(orig_layer, pruned_layer)

        # hard coded value
        samples_per_image = 10

        # get number of batches depending on the layer type
        num_of_batches = sub_sampler.get_number_of_batches(data_loader, orig_layer, num_reconstruction_samples,
                                                           samples_per_image)

        # Todo - I am not sure if checking the length of a data loader is a great idea.
        if num_of_batches > len(data_loader) or num_of_batches < 1:
            raise ValueError("There are insufficient batches of data in the provided data loader for the "
                             "purpose of weight reconstruction or number of reconstruction samples!")

        hook_handles = []

        orig_layer_out_data = []
        pruned_layer_inp_data = []

        all_sub_sampled_inp_data = []
        all_sub_sampled_out_data = []

        # register forward hooks
        hook_handles.append(cls._register_fwd_hook_for_layer(orig_layer, _hook_to_collect_output_data))

        hook_handles.append(cls._register_fwd_hook_for_layer(pruned_layer, _hook_to_collect_input_data))

        # forward pass for given number of batches for both original model and compressed model
        for batch_index, batch in enumerate(data_loader):

            assert isinstance(batch, (tuple, list)), 'data loader should provide data in list or tuple format' \
                                                     '(input_data, labels) or [input_data, labels]'

            batch, _ = batch

            DataSubSampler._forward_pass(orig_model, batch)
            DataSubSampler._forward_pass(comp_model, batch)

            input_data = np.vstack(pruned_layer_inp_data)
            output_data = np.vstack(orig_layer_out_data)

            # delete list entries used for hooks
            del pruned_layer_inp_data[:]
            del orig_layer_out_data[:]

            # get the sub sampled input and output data depending on layer type
            sub_sampled_inp_data, sub_sampled_out_data = sub_sampler.get_sub_sampled_data(orig_layer, input_data,
                                                                                          output_data,
                                                                                          samples_per_image)

            all_sub_sampled_inp_data.append(sub_sampled_inp_data)
            all_sub_sampled_out_data.append(sub_sampled_out_data)

            if batch_index == num_of_batches - 1:
                logger.debug("batch index : %s reached number of batches: %s", batch_index + 1, num_of_batches)
                break

        # remove hook handles
        for hook_handle in hook_handles:
            hook_handle.remove()

        # accumulate total sub sampled input and output data

        return np.vstack(all_sub_sampled_inp_data), np.vstack(all_sub_sampled_out_data)

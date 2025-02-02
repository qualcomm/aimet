# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Implementation of layer splitting logic for spatial and weight svd schemes """

import abc
import math
import numpy as np
import torch

from aimet_common.utils import AimetLogger
from aimet_common.svd_pruner import SpatialSvdPruner, WeightSvdPruner

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class SpatialSvdModuleSplitter:
    """
    Spatial SVD module splitter
    """
    @staticmethod
    def split_module(module: torch.nn.Module, rank: int):
        """
        :param module: Module to be split
        :param rank: rank for splitting
        :return: Two split modules
        """
        assert isinstance(module, torch.nn.Conv2d)
        assert module.dilation == (1, 1)

        weight_tensor = module.weight.detach().cpu().numpy()  # n c h w
        out_channels, in_channels, height, width = weight_tensor.shape

        h, v = SpatialSvdPruner.lingalg_spatial_svd(weight_tensor, rank, in_channels, out_channels,
                                                    height, width)

        first_module = torch.nn.Conv2d(in_channels=module.in_channels,
                                       out_channels=rank, kernel_size=(height, 1),
                                       stride=(module.stride[0], 1),
                                       padding=(module.padding[0], 0), dilation=1, bias=False)
        first_module.weight.data = torch.FloatTensor(v).to(device=module.weight.device)

        second_module = torch.nn.Conv2d(in_channels=rank,
                                        out_channels=module.out_channels, kernel_size=(1, width),
                                        stride=(1, module.stride[1]),
                                        padding=(0, module.padding[1]), dilation=1, bias=module.bias is not None)
        second_module.weight.data = torch.FloatTensor(h).to(device=module.weight.device)
        if module.bias is not None:
            second_module.bias.data = module.bias.data

        return first_module, second_module


class WeightSvdModuleSplitter(abc.ABC):
    """
    Weight SVD module splitter
    """
    @classmethod
    def split_module(cls, module: torch.nn.Module, rank: int, **kwargs) -> (torch.nn.Module, torch.nn.Module):
        """
        :param module: Module to be split
        :param rank: rank for splitting
        :param kwargs: Additional keyword arguments
        """
        if isinstance(module, torch.nn.Conv2d):
            split_modules = cls.split_conv_module(module, rank, **kwargs)
        elif isinstance(module, torch.nn.Linear):
            split_modules = cls.split_fc_module(module, rank, **kwargs)
        else:
            raise AssertionError('Weight SVD only supports Conv2d and FC modules currently.')

        return split_modules

    @classmethod
    @abc.abstractmethod
    def split_conv_module(cls, *args, **kwargs) -> (torch.nn.Module, torch.nn.Module):
        """
        :param args:
        :param kwargs:
        :return:
        """

    @classmethod
    @abc.abstractmethod
    def split_fc_module(cls, *args, **kwargs) -> (torch.nn.Module, torch.nn.Module):
        """
        :param args:
        :param kwargs:
        :return:
        """

    @staticmethod
    def create_conv_modules(module: torch.nn.Module, rank: int) -> (torch.nn.Module, torch.nn.Module):
        """
        Create conv modules.

        :param module: Module to be split
        :param rank: rank for splitting
        :return: Two split modules
        """
        conv_a = torch.nn.Conv2d(in_channels=module.in_channels,
                                 out_channels=rank,
                                 kernel_size=(1, 1),
                                 stride=(1, 1),
                                 dilation=module.dilation).to(device=module.weight.device,
                                                              dtype=module.weight.dtype)
        conv_b = torch.nn.Conv2d(in_channels=rank,
                                 out_channels=module.out_channels,
                                 kernel_size=module.kernel_size,
                                 stride=module.stride,
                                 padding=module.padding,
                                 dilation=module.dilation).to(device=module.weight.device,
                                                              dtype=module.weight.dtype)
        return conv_a, conv_b

    @staticmethod
    def create_fc_modules(module: torch.nn.Module, rank: int) -> (torch.nn.Module, torch.nn.Module):
        """
        Create fc modules.

        :param module: Module to be split
        :param rank: rank for splitting
        :return: Two split modules
        """
        fc_a = torch.nn.Linear(in_features=module.in_features,
                               out_features=rank).to(device=module.weight.device,
                                                     dtype=module.weight.dtype)
        fc_b = torch.nn.Linear(in_features=rank,
                               out_features=module.out_features).to(device=module.weight.device,
                                                                    dtype=module.weight.dtype)
        return fc_a, fc_b

    @staticmethod
    def _update_weight(module: torch.nn.Module, module_1: torch.nn.Module, module_2: torch.nn.Module,
                       weight_1: torch.Tensor, weight_2: torch.Tensor):
        """
        Update module weight parameters.

        :param module:
        :param module_1:
        :param module_2:
        :param weight_1:
        :param weight_2:
        :return:
        """
        assert isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))

        with torch.no_grad():
            module_1.weight.copy_(weight_1).to(device=module.weight.device,
                                               dtype=module.weight.dtype)
            module_2.weight.copy_(weight_2).to(device=module.weight.device,
                                               dtype=module.weight.dtype)


class MoWeightSvdModuleSplitter(WeightSvdModuleSplitter):
    """
    Weight SVD module splitter using Model Optimizations library. (C++)
    """
    # pylint:disable=arguments-differ
    @classmethod
    def split_conv_module(cls, module: torch.nn.Module, rank: int, name: str,
                          svd_lib_ref):
        """
        Split a given Conv2D module using weight svd

        :param module: Module to be split
        :param rank: Rank to use to split with
        :param name: Name of the module
        :param svd_lib_ref: Reference to pymo
        :return: Two split modules
        """
        split_weights, weight_sizes = [], []

        conv_a_weight_shape = (rank, module.in_channels, 1, 1)
        conv_a_weight = np.zeros(conv_a_weight_shape)

        split_weights.append(conv_a_weight.flatten().tolist())
        weight_sizes.append(conv_a_weight.size)

        conv_b_weight_shape = (module.out_channels, rank, *module.kernel_size)
        conv_b_weight = np.zeros(conv_b_weight_shape)

        split_weights.append(conv_b_weight.flatten().tolist())
        weight_sizes.append(conv_b_weight.size)

        split_weights = svd_lib_ref.SplitLayerWeights(str(name), split_weights, weight_sizes, [rank])
        weight_1 = torch.from_numpy(np.array(split_weights[0]).reshape(conv_a_weight_shape))
        weight_2 = torch.from_numpy(np.array(split_weights[1]).reshape(conv_b_weight_shape))

        logger.debug("Splitting conv module weight of shape %r into %r and %r",
                     module.weight.shape, conv_a_weight.shape, conv_b_weight.shape)

        # Split the Conv into two modules.
        conv_a, conv_b = cls.create_conv_modules(module, rank)

        # Update weight parameters.
        cls._update_weight(module, conv_a, conv_b, weight_1, weight_2)

        # Update bias parameters.
        cls._update_bias(module, conv_a, conv_b, name, rank, svd_lib_ref)

        return conv_a, conv_b

    # pylint:disable=arguments-differ
    @classmethod
    def split_fc_module(cls, module: torch.nn.Module, rank: int, name: str,
                        svd_lib_ref):
        """
        Split a given Linear module using weight svd
        :param module: Module to be split
        :param rank: Rank to use to split with
        :param name: Name of the module
        :param svd_lib_ref: Reference to pymo
        :return: Two split modules
        """
        split_weights, weight_sizes = [], []

        fc_a_weight_shape = (rank, module.in_features)
        fc_a_weight = np.zeros(fc_a_weight_shape)
        split_weights.append(fc_a_weight.flatten().tolist())
        weight_sizes.append(fc_a_weight.size)

        fc_b_weight_shape = (module.out_features, rank)
        fc_b_weight = np.zeros(fc_b_weight_shape)
        split_weights.append(fc_b_weight.flatten().tolist())
        weight_sizes.append(fc_b_weight.size)

        split_weights = svd_lib_ref.SplitLayerWeights(str(name), split_weights, weight_sizes, [rank])
        weight_1 = torch.from_numpy(np.array(split_weights[0]).reshape(fc_a_weight_shape))
        weight_2 = torch.from_numpy(np.array(split_weights[1]).reshape(fc_b_weight_shape))

        # Split FC module into two.
        fc_a, fc_b = cls.create_fc_modules(module, rank)

        # Update weight parameters.
        cls._update_weight(module, fc_a, fc_b, weight_1, weight_2)

        # Update bias parameters.
        cls._update_bias(module, fc_a, fc_b, name, rank, svd_lib_ref)

        return fc_a, fc_b

    @staticmethod
    def _update_bias(module: torch.nn.Module, module_1: torch.nn.Module, module_2: torch.nn.Module, name: str,
                     rank: int, svd_lib_ref):
        """
        :param module:
        :param module_1:
        :param module_2:
        :param name:
        :param rank:
        :param svd_lib_ref:
        :return:
        """
        assert isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))

        if module.bias is not None:
            split_biases, bias_sizes = [], []

            module_1_bias = np.zeros(rank)
            split_biases.append(module_1_bias.flatten().tolist())
            bias_sizes.append(module_1_bias.size)

            module_2_bias = np.zeros(module.out_channels if isinstance(module, torch.nn.Conv2d) else module.out_features)
            split_biases.append(module_2_bias.flatten().tolist())
            bias_sizes.append(module_2_bias.size)

            split_biases = svd_lib_ref.SplitLayerBiases(str(name), split_biases, bias_sizes, [rank])
            bias_1 = torch.from_numpy(np.array(split_biases[0]))
            bias_2 = torch.from_numpy(np.array(split_biases[1]))

            with torch.no_grad():
                module_1.bias.copy_(bias_1).to(device=module.bias.device,
                                               dtype=module.bias.dtype)
                module_2.bias.copy_(bias_2).to(device=module.bias.device,
                                               dtype=module.bias.dtype)
        else:
            module_1.bias = None
            module_2.bias = None


class PyWeightSvdModuleSplitter(WeightSvdModuleSplitter):
    """
    Weight SVD module splitter using numpy.
    """
    # pylint:disable=arguments-differ
    @classmethod
    def split_conv_module(cls, module: torch.nn.Module, rank: int) -> (torch.nn.Module, torch.nn.Module):
        """
        Split a given module using weight svd.
        :param module: Module to be split
        :param rank: rank for splitting
        :return:
        """
        weight = module.weight.detach().cpu()
        weight = weight.permute(1, 0, 2, 3).numpy()
        nkk_shape = weight.shape[-3:]
        nkk = math.prod(nkk_shape)
        weight = weight.reshape(weight.shape[0], nkk)

        # Split weight matrix.
        weight_1, weight_2 = WeightSvdPruner.lingalg_weight_svd(weight, rank)

        weight_1 = torch.from_numpy(weight_1).unsqueeze(-1).unsqueeze(-1)
        weight_1 = weight_1.permute(1, 0, 2, 3)

        weight_2 = torch.from_numpy(weight_2)
        weight_2 = weight_2.reshape(weight_2.shape[0], *nkk_shape)
        weight_2 = weight_2.permute(1, 0, 2, 3)

        # Split the Conv into two modules.
        conv_a, conv_b = cls.create_conv_modules(module, rank)

        # Update weight parameters.
        cls._update_weight(module, conv_a, conv_b, weight_1, weight_2)

        # Update bias parameters.
        cls._update_bias(module, conv_a, conv_b)

        return conv_a, conv_b

    # pylint:disable=arguments-differ
    @classmethod
    def split_fc_module(cls, module: torch.nn.Module, rank: int) -> (torch.nn.Module, torch.nn.Module):
        """
        Split a given module using weight svd.

        :param module: Module to be split
        :param rank: rank for splitting
        :return:
        """
        weight = module.weight.detach().cpu().numpy()
        weight = weight.transpose(1, 0)

        # Split weight matrix.
        weight_1, weight_2 = WeightSvdPruner.lingalg_weight_svd(weight, rank)

        weight_1 = torch.from_numpy(weight_1).transpose(1, 0)
        weight_2 = torch.from_numpy(weight_2).transpose(1, 0)

        # Split the FC into two modules.
        fc_a, fc_b = cls.create_fc_modules(module, rank)

        # Update weight parameters.
        cls._update_weight(module, fc_a, fc_b, weight_1, weight_2)

        # Update bias parameters.
        cls._update_bias(module, fc_a, fc_b)

        return fc_a, fc_b

    @staticmethod
    def _update_bias(module: torch.nn.Module, module_1: torch.nn.Module, module_2: torch.nn.Module):
        """
        Update the bias parameters.

        :param module: Original module.
        :param module_1: Split module_1.
        :param module_2: Split module_2.
        """
        assert isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))

        if module.bias is not None:
            bias = torch.zeros(
                module_1.out_channels if isinstance(module_1, torch.nn.Conv2d) else module_1.out_features,
                device=module.weight.device,
                dtype=module.weight.dtype)
            module_1.bias = torch.nn.Parameter(bias)

            with torch.no_grad():
                module_2.bias.copy_(module.bias).to(device=module.bias.device,
                                                    dtype=module.bias.dtype)
        else:
            module_1.bias = None
            module_2.bias = None

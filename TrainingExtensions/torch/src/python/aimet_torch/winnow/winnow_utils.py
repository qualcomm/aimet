# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Utilities for winnowing """

from typing import List, Tuple
import numpy as np
import torch
from torch import nn

from aimet_common.polyslice import PolySlice


class DownsampleLayer(nn.Module):
    """
    This layer down samples the input in the channel dimension according to a given mask.
    Taken from the Qrunchy repo.
    """

    def __init__(self, keep_indices_tensor):
        """ keep_indices_tensor: list of indices that should be kept. """
        super().__init__()
        self.keep_tensor = keep_indices_tensor

    # pylint: disable=arguments-differ
    def forward(self, x):
        """ Only the needed indices are kept. """
        self.keep_tensor = self.keep_tensor.to("cuda" if x.is_cuda else "cpu")
        y = torch.index_select(x, 1, self.keep_tensor)
        return y

    def get_mask_hard(self):
        """ Returns the mask for channels with zero planes.
            A 1 in the mask represents  a channel with zero planes. """
        return self.mask


class UpsampleLayer(nn.Module):
    """
    This layer up samples the input in the channel dimension according to a given mask. If a scale
    is given it also scales the up sampled output.
    """

    def __init__(self, mask, scale=None):
        super().__init__()
        self.register_buffer('mask', mask)
        self.register_buffer('indices', mask.nonzero().squeeze(1))
        if scale is not None:
            self.register_buffer('scale', scale[None, :, None, None])

    # pylint: disable=arguments-differ
    def forward(self, x):
        """ Forward  function. """
        out = torch.zeros(x.shape[0], self.mask.shape[0], x.shape[2], x.shape[3], device=x.device)
        out = out.to("cuda" if x.is_cuda else "cpu")
        if x.dtype == torch.float64:
            out = out.double()
        out[:, self.indices] = x
        return out * self.scale if hasattr(self, 'scale') else out


class ReShape(nn.Module):
    """ ReShape layer class that reshapes the input to the desired shape. """

    def __init__(self, *args):
        """ Desired shape is passed in as argument. """
        super().__init__()
        self.shape = args

    # pylint: disable=arguments-differ
    def forward(self, x):
        """ forward pass """
        return x.view(self.shape)


def to_numpy(tensor):
    """ Converts a PyTorch Tensor to a Numpy array"""
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, 'is_cuda'):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, 'detach'):
        return tensor.detach().numpy()
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()

    return np.array(tensor)


def zero_out_input_channels(module, input_channels_to_prune):
    """
    :param module: module
    :param input_channels_to_prune: list of input channels to be zeroed out
    :return:
    """
    for input_channel in input_channels_to_prune:
        if isinstance(module, nn.Conv2d):
            module.weight.data[:, input_channel, :, :] = 0

        elif isinstance(module, nn.Linear):
            module.weight.data[:, input_channel] = 0

        else:
            raise ValueError("Unsupported layer_type")


def search_for_zero_planes(model: torch.nn.Module) -> List[Tuple[torch.nn.Module, List[int]]]:
    """ If list of modules to winnow is empty to start with, search through all modules to check if any
    planes have been zeroed out. Update self._list_of_modules_to_winnow with any findings.
    :param model: torch model to search through modules for zeroed parameters
    """

    list_of_modules_to_winnow = []
    for _, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.modules.conv.Conv2d)):
            in_channels_to_winnow = _assess_weight_and_bias(module.weight, module.bias)
            if in_channels_to_winnow:
                list_of_modules_to_winnow.append((module, in_channels_to_winnow))
    return list_of_modules_to_winnow


def _assess_weight_and_bias(weight: torch.nn.Parameter, _bias: torch.nn.Parameter):
    """ 4-dim weights [CH-out, CH-in, H, W] and 1-dim bias [CH-out] """
    num_out = weight.shape[0]
    num_in = weight.shape[1]

    input_channels_to_ignore = []
    for ch_in in range(num_in):
        can_winnow = True
        for ch_out in range(num_out):
            if not _is_zero_tensor(weight[ch_out, ch_in]):
                can_winnow = False
                break
        if can_winnow:
            input_channels_to_ignore.append(ch_in)

    # Enable when implementing output pruning:
    # return input_channels_to_ignore, outputs_to_ignore

    return input_channels_to_ignore


def _is_zero_tensor(tensor: torch.Tensor):
    return (tensor != 0.0).any().item() == 0


def reduce_tensor(tensor: torch.Tensor, reduction: PolySlice):
    """Removes slices in one or more of the tensor's dimensions."""
    using_cuda = bool(tensor.is_cuda)

    result = torch.tensor(tensor)  # pylint: disable=not-callable

    for dim, index in reduction.get_all().items():
        assert dim <= len(tensor.shape)
        dim_size = tensor.shape[dim]
        assert max(index) < dim_size
        to_keep = [i for i in range(dim_size) if i not in index]
        if using_cuda:
            result = torch.index_select(result, dim, torch.tensor(to_keep).cuda())  # pylint: disable=not-callable
        else:
            result = torch.index_select(result, dim, torch.tensor(to_keep)) # pylint: disable=not-callable
    return result

# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""This module enables batch norm folding in QuantSim v2"""
from typing import List, Tuple, Iterable
import torch
from aimet_torch import utils
from aimet_torch._base.batch_norm_fold import BatchNormFoldBase, _BatchNormFoldingNotSupported
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn.base import BaseQuantizationMixin
from torch.nn.modules.conv import _ConvTransposeNd
from aimet_common.utils import AimetLogger

__all__ = [
    "fold_all_batch_norms",
    "fold_all_batch_norms_to_scale",
    "fold_given_batch_norms",
    "_is_valid_bn_fold",
    "_find_all_batch_norms_to_fold",
    "find_standalone_batchnorm_ops",
]

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.BatchNormFolding)


class BatchNormFold(BatchNormFoldBase):
    """Handles batch norm folding logic"""

    @classmethod
    def fold_all_batch_norms_to_scale(cls, sim: QuantizationSimModel)\
            -> List[Tuple[BaseQuantizationMixin, BaseQuantizationMixin]]:
        """
        Fold all batch_norm layers in a model into the quantization scale parameter
        of the corresponding conv layers

        :param sim: QuantizationSimModel
        :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
        """
        # pylint: disable=protected-access
        assert sim.model is not None
        assert sim.connected_graph is not None

        model = sim.model
        connected_graph = sim.connected_graph

        module_to_qmodule = {}
        for module_name, module in sim.connected_graph._name_to_module.items():
            if hasattr(sim.model, module_name):
                module_to_qmodule[module] = getattr(sim.model, module_name)

            if '.' in module_name and hasattr(sim.model, module_name.split('.')[1]):
                module_to_qmodule[module] = getattr(sim.model, module_name.split('.')[1])

        conv_bn_pairs, bn_conv_pairs, _ = cls._find_all_batch_norms_to_fold(connected_graph)

        conv_bn_pairs = [
            (module_to_qmodule[conv], module_to_qmodule[bn]) for conv, bn in conv_bn_pairs
        ]
        bn_conv_pairs = [
            (module_to_qmodule[bn], module_to_qmodule[conv]) for bn, conv in bn_conv_pairs
        ]

        BatchNormFold._fold_given_batch_norms(model, conv_bn_pairs, bn_conv_pairs)
        return conv_bn_pairs + [(conv, bn) for bn, conv in bn_conv_pairs]

    @classmethod
    def _fold_given_batch_norms(cls,
                                model,
                                conv_bn_pairs: Iterable[Tuple[torch.nn.Module, torch.nn.Module]],
                                bn_conv_pairs: Iterable[Tuple[torch.nn.Module, torch.nn.Module]]):
        """
        Fold a given set of batch_norm layers into conv layers

        :param model: Model
        :param conv_bn_pairs: List of (conv, bn) pairs to fold
        :param bn_conv_pairs: List of (bn, conv) pairs to fold
        :return: None
        """
        # pylint: disable=protected-access
        for bn, conv in bn_conv_pairs:
            if isinstance(conv, BaseQuantizationMixin):
                raise RuntimeError(f"Forward folding to scale is not possible. Got {conv}")

        bn_modules = []

        def _fold(conv, bn, fold_backward):
            is_quantized = isinstance(conv, BaseQuantizationMixin) or isinstance(bn, BaseQuantizationMixin)
            try:
                if is_quantized:
                    assert isinstance(conv, BaseQuantizationMixin) and isinstance(bn, BaseQuantizationMixin)
                    BatchNormFold._fold_to_scale(conv, bn)
                    bn_modules.append(bn)
                else:
                    cls._fold_to_weight(conv, bn, fold_backward=fold_backward)
            except _BatchNormFoldingNotSupported as e:
                bn_name = utils.get_layer_name(model, bn)
                conv_name = utils.get_layer_name(model, conv)
                _logger.warning(
                    "Failed to fold %s to %s. [Reason] %s", bn_name, conv_name, str(e)
                )
            else:
                bn_modules.append(bn if is_quantized else bn)


        with utils.in_eval_mode(model), torch.no_grad():
            for conv, bn in conv_bn_pairs:
                _fold(conv, bn, fold_backward=True)

            for bn, conv in bn_conv_pairs:
                _fold(conv, bn, fold_backward=False)

            cls._delete_bn_from_model(model, bn_modules)

    # pylint: disable=arguments-differ
    @classmethod
    def _fold_to_scale(cls, conv: BaseQuantizationMixin, bn: BaseQuantizationMixin):
        """
        Fold BatchNorm into the scale and bias of the given layer.

        :param conv: Quantized conv or linear layer.
        :param bn: Quantized bn layer.
        """
        # pylint: disable=protected-access, too-many-locals, too-many-branches, too-many-statements
        output_quantizer = conv.output_quantizers[0]

        if output_quantizer:
            raise _BatchNormFoldingNotSupported(
                "BatchNorm should belong to the same supergroup with the layer to be folded to."
            )

        if "bias" in conv.param_quantizers:
            bias_quantizer = conv.param_quantizers["bias"]
            if bias_quantizer:
                raise _BatchNormFoldingNotSupported(
                    "Can't fold BatchNorm to scale if bias quantizer is enabled."
                )

        weight_quantizer = conv.param_quantizers["weight"]

        if isinstance(conv, _ConvTransposeNd) and conv.groups != 1:
            raise _BatchNormFoldingNotSupported(
                "BatchNorm folding to scale is not supported for grouped ConvTransposeNd."
            )

        # Add quantization noise to the BN params (bn weight & bn bias) before folding.
        # NOTE: Quantization of foldable batchnorms is automatically disabled when
        #       initializing quantsim. However, it is still safer to call _quantize_params here
        #       as we can't guarantee this is always the case.
        #       For example, the user can manually enable quantization of batchnorms, etc...
        #       (FYI: _quantize_params takes effect only when the parameter quantizers are enabled)

        with bn._patch_quantized_parameters():
            cls._fold_to_weight(conv, bn, fold_backward=True)

            gamma = bn.weight
            sigma = torch.sqrt(bn.running_var + bn.eps)
            result = gamma / sigma

            # flatten min/max to handle ConvTranspose shape
            flat_min = torch.flatten(weight_quantizer.min)
            flat_max = torch.flatten(weight_quantizer.max)

            new_encoding_min = torch.zeros_like(flat_min)
            new_encoding_max = torch.zeros_like(flat_max)

            for i in range(torch.numel(flat_min)):
                if result[i] >= 0:
                    new_encoding_max[i] = flat_max[i] * result[i]
                    new_encoding_min[i] = flat_min[i] * result[i]
                else:
                    new_encoding_max[i] = flat_min[i] * result[i]
                    new_encoding_min[i] = flat_max[i] * result[i]

            weight_quantizer.min.copy_(new_encoding_min.view_as(weight_quantizer.min))
            weight_quantizer.max.copy_(new_encoding_max.view_as(weight_quantizer.max))

        # Copy batchnorm's output quantizers to conv output quantizers
        for i, (conv_output_quantizer, bn_output_quantizer) in\
                enumerate(zip(conv.output_quantizers, bn.output_quantizers)):

            if bn_output_quantizer:
                conv_output_quantizer = bn_output_quantizer

            conv.output_quantizers[i] = conv_output_quantizer

        conv.param_quantizers["bias"] = None

# Global variables for compatibility
fold_all_batch_norms = BatchNormFold.fold_all_batch_norms_to_weight
fold_all_batch_norms_to_scale = BatchNormFold.fold_all_batch_norms_to_scale
fold_given_batch_norms = BatchNormFold.fold_given_batch_norms
# pylint: disable=protected-access
_is_valid_bn_fold = BatchNormFold._is_valid_bn_fold
_find_all_batch_norms_to_fold = BatchNormFold._find_all_batch_norms_to_fold
find_standalone_batchnorm_ops = BatchNormFold.find_standalone_batchnorm_ops

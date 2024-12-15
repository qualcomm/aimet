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

""" Optimization code to fold batch-norm layers """

from typing import List, Tuple, Iterable
import torch
import torch.nn
from torch.nn.modules.conv import _ConvTransposeNd

import aimet_common.libpymo as libpymo
from aimet_common.utils import AimetLogger

from aimet_torch import utils
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper
from aimet_torch.v1.tensor_quantizer import LearnedGridTensorQuantizer
from aimet_torch._base.batch_norm_fold import BatchNormFoldBase, _BatchNormFoldingNotSupported

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
    def fold_all_batch_norms_to_scale(
            cls, sim: QuantizationSimModel,
    ) -> List[Tuple[QcQuantizeWrapper, QcQuantizeWrapper]]:
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

        quant_wrappers = {
            quant_wrapper._module_to_wrap: quant_wrapper
            for _, quant_wrapper in sim.quant_wrappers()
        }
        conv_bn_pairs, bn_conv_pairs, _ = cls._find_all_batch_norms_to_fold(connected_graph)
        # print(conv_bn_pairs)
        conv_bn_pairs = [
            (quant_wrappers[conv], quant_wrappers[bn]) for conv, bn in conv_bn_pairs
        ]
        # print(conv_bn_pairs)
        # print(type(conv_bn_pairs))
        bn_conv_pairs = [
            (quant_wrappers[bn], quant_wrappers[conv]) for bn, conv in bn_conv_pairs
        ]

        cls._fold_given_batch_norms(model, conv_bn_pairs, bn_conv_pairs)

        return conv_bn_pairs + [(conv, bn) for bn, conv in bn_conv_pairs]

    @classmethod
    def fold_given_batch_norms(cls, model, layer_pairs):
        super().fold_given_batch_norms(model, layer_pairs)

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
            if isinstance(conv, QcQuantizeWrapper):
                raise RuntimeError(f"Forward folding to scale is not possible. Got {conv}")

        bn_modules = []

        def _fold(conv, bn, fold_backward):
            is_wrapped = isinstance(conv, QcQuantizeWrapper) or isinstance(bn, QcQuantizeWrapper)
            try:
                if is_wrapped:
                    assert isinstance(conv, QcQuantizeWrapper) and isinstance(bn, QcQuantizeWrapper)
                    cls._fold_to_scale(conv, bn)
                    bn_modules.append(bn._module_to_wrap)
                else:
                    cls._fold_to_weight(conv, bn, fold_backward=fold_backward)
            except _BatchNormFoldingNotSupported as e:
                bn_name = utils.get_layer_name(model, bn)
                conv_name = utils.get_layer_name(model, conv)
                _logger.warning(
                    "Failed to fold %s to %s. [Reason] %s", bn_name, conv_name, str(e)
                )
            else:
                bn_modules.append(bn._module_to_wrap if is_wrapped else bn)


        with utils.in_eval_mode(model), torch.no_grad():
            for conv, bn in conv_bn_pairs:
                _fold(conv, bn, fold_backward=True)

            for bn, conv in bn_conv_pairs:
                _fold(conv, bn, fold_backward=False)

            cls._delete_bn_from_model(model, bn_modules)

    @classmethod
    def _fold_to_scale(cls, conv_wrapper: QcQuantizeWrapper, bn_wrapper: QcQuantizeWrapper):
        """
        Fold BatchNorm into the scale and bias of the given layer.

        :param conv_wrapper: QcQuantizeWrapper that wraps conv or linear layer.
        :param bn_wrapper: QcQuantizeWrapper that wraps bn.
        """
        # pylint: disable=protected-access, too-many-locals, too-many-branches, too-many-statements
        conv = conv_wrapper._module_to_wrap
        bn = bn_wrapper._module_to_wrap

        weight_quantizer = conv_wrapper.param_quantizers["weight"]

        if not isinstance(weight_quantizer, LearnedGridTensorQuantizer):
            raise _BatchNormFoldingNotSupported(
                "BatchNorm folding to scale supports LearnedGridTensorQuantizer only; "
                f"got {type(weight_quantizer)}."
            )

        output_quantizer = conv_wrapper.output_quantizers[0]

        if output_quantizer.enabled:
            raise _BatchNormFoldingNotSupported(
                "BatchNorm should belong to the same supergroup with the layer to be folded to."
            )

        if "bias" in conv_wrapper.param_quantizers:
            bias_quantizer = conv_wrapper.param_quantizers["bias"]
            if bias_quantizer.enabled:
                raise _BatchNormFoldingNotSupported(
                    "Can't fold BatchNorm to scale if bias quantizer is enabled."
                )

        encodings = weight_quantizer.encoding

        if encodings is None:
            raise RuntimeError

        if isinstance(encodings, libpymo.TfEncoding):
            encodings = [encodings]

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
        with bn_wrapper._quantize_params():
            cls._fold_to_weight(conv, bn, fold_backward=True)

            gamma = bn.weight
            sigma = torch.sqrt(bn.running_var + bn.eps)

            new_encodings = []
            for old_encoding, c in zip(encodings, gamma/sigma):
                new_encoding = libpymo.TfEncoding()
                new_encoding.delta = old_encoding.delta * abs(c)
                if c >= 0:
                    new_encoding.max = old_encoding.max * c
                    new_encoding.min = old_encoding.min * c
                else:
                    new_encoding.max = old_encoding.min * c
                    new_encoding.min = old_encoding.max * c
                new_encoding.offset = old_encoding.offset
                new_encoding.bw = old_encoding.bw
                new_encodings.append(new_encoding)

            weight_quantizer.encoding = new_encodings

        # Copy batchnorm's output quantizers to conv output quantizers
        for conv_output_quantizer, bn_output_quantizer in\
                zip(conv_wrapper.output_quantizers, bn_wrapper.output_quantizers):
            conv_output_quantizer.enabled = bn_output_quantizer.enabled

            if bn_output_quantizer.encoding is not None:
                encoding = libpymo.TfEncoding()
                encoding.delta  = bn_output_quantizer.encoding.delta
                encoding.max    = bn_output_quantizer.encoding.max
                encoding.min    = bn_output_quantizer.encoding.min
                encoding.offset = bn_output_quantizer.encoding.offset
                encoding.bw     = bn_output_quantizer.encoding.bw
                conv_output_quantizer.encoding = encoding

            bn_output_quantizer.enabled = False

        if "bias" not in conv_wrapper.param_quantizers:
            bias_quantizer = LearnedGridTensorQuantizer(weight_quantizer.bitwidth,
                                                        weight_quantizer.round_mode,
                                                        weight_quantizer.quant_scheme,
                                                        weight_quantizer.use_symmetric_encodings,
                                                        enabled_by_default=False,
                                                        data_type=weight_quantizer.data_type)
            bias_quantizer._ch_axis = weight_quantizer._ch_axis
            conv_wrapper.param_quantizers["bias"] = bias_quantizer

    @classmethod
    def _is_batchnorm(cls, module: torch.nn.Module) -> bool:
        if isinstance(module, QcQuantizeWrapper):
            module = module._module_to_wrap # pylint: disable=protected-access
        return super()._is_batchnorm(module)

    @classmethod
    def _is_conv_linear(cls, module: torch.nn.Module) -> bool:
        if isinstance(module, QcQuantizeWrapper):
            module = module._module_to_wrap # pylint: disable=protected-access
        return super()._is_conv_linear(module)



# Global variables for compatibility
fold_all_batch_norms = BatchNormFold.fold_all_batch_norms_to_weight
fold_all_batch_norms_to_scale = BatchNormFold.fold_all_batch_norms_to_scale
fold_given_batch_norms = BatchNormFold.fold_given_batch_norms
# pylint: disable=protected-access
_is_valid_bn_fold = BatchNormFold._is_valid_bn_fold
_find_all_batch_norms_to_fold = BatchNormFold._find_all_batch_norms_to_fold
find_standalone_batchnorm_ops = BatchNormFold.find_standalone_batchnorm_ops

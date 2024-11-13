# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Sequential MSE implementation """

from typing import List, Optional, Tuple
import contextlib
import torch
from torch import nn
from torch.utils.data import DataLoader

from aimet_common.utils import AimetLogger
from aimet_torch._base.seq_mse import SequentialMseBase, SeqMseParams, SUPPORTED_MODULES
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine import AffineQuantizerBase, QuantizeDequantize, GroupedBlockQuantizeDequantize
from aimet_torch.v2.quantization.affine.backends import torch_builtins
from aimet_torch.v2.nn.base import BaseQuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.utils import reduce, _is_reducible


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)

class SequentialMse(SequentialMseBase):
    """
    Sequentially minimizing activation MSE loss in layer-wise way to decide optimal param quantization encodings.
    """

    @classmethod
    def apply_seq_mse(cls,
                      model: torch.nn.Module,
                      sim: QuantizationSimModel,
                      data_loader: DataLoader,
                      params: SeqMseParams,
                      modules_to_exclude: Optional[List[torch.nn.Module]] = None,
                      checkpoints_config: Optional[str] = None):
        if not modules_to_exclude:
            modules_to_exclude = []
        modules_to_exclude.extend(cls._get_grouped_convs_with_blockwise_quantization(sim))
        with cls._handle_grouped_block_quantizers(sim):
            super().apply_seq_mse(model, sim, data_loader, params, modules_to_exclude, checkpoints_config)

    @staticmethod
    def _get_grouped_convs_with_blockwise_quantization(sim):
        """ Return a list of all grouped conv modules using blockwise quantization for weights """
        grouped_convs_with_blockwise_quantization = []
        for module in sim.model.modules():
            if isinstance(module, torch.nn.Conv2d) and \
                    isinstance(module, BaseQuantizationMixin) and \
                    module.groups != 1 and \
                    module.param_quantizers['weight'].block_size is not None and \
                    module.param_quantizers['weight'].block_size[1] != module.weight.shape[1]:
                grouped_convs_with_blockwise_quantization.append(module)
        return grouped_convs_with_blockwise_quantization

    @staticmethod
    @contextlib.contextmanager
    def _handle_grouped_block_quantizers(sim: QuantizationSimModel):
        """ Set all grouped block quantizers to regular blockwise quantization for the duration of the context manager
        """
        grouped_block_quantize_dequantizers = []
        for module in sim.model.modules():
            if isinstance(module, GroupedBlockQuantizeDequantize):
                grouped_block_quantize_dequantizers.append((module, module.block_grouping))
                module.block_grouping = tuple(1 for _ in enumerate(module.shape))

        yield

        for (module, block_grouping) in grouped_block_quantize_dequantizers:
            module.block_grouping = block_grouping

    @classmethod
    def compute_all_param_encodings(cls, sim: QuantizationSimModel):
        """
        Compute encodings for all parameters, needed for initializing Sequential MSE

        :param sim: Quant sim
        """
        for _, qmodule in sim.named_qmodules():
            qmodule._compute_param_encodings(overwrite=True) # pylint: disable=protected-access

    @classmethod
    @contextlib.contextmanager
    def temporarily_disable_quantizers(
            cls,
            model: torch.nn.Module,
            sim: QuantizationSimModel,
            modules_to_exclude: Optional[List[torch.nn.Module]],
    ):
        """
        For given quantsim model, disable quantizers needed to be diabled before applying sequential MSE.

        :param model: Original fp32 model
        :param sim: QuantizationSimModel object
        :param modules_to_exclude: List of supported modules to exclude when applying Sequential MSE
        :return: List of quantizers to be disabled.
        """
        # pylint: disable=protected-access
        name_to_fp32_module_dict = {}
        for name, fp32_module in model.named_modules():
            name_to_fp32_module_dict[name] = fp32_module

        original_input_quantizers = {}
        original_output_quantizers = {}
        original_param_quantizers = {}
        for name, qmodule in sim.named_qmodules():
            original_input_quantizers[name] = qmodule.input_quantizers
            original_output_quantizers[name] = qmodule.output_quantizers
            qmodule.input_quantizers = nn.ModuleList([None for _ in qmodule.input_quantizers])
            qmodule.output_quantizers = nn.ModuleList([None for _ in qmodule.output_quantizers])

            if not isinstance(qmodule, SUPPORTED_MODULES):
                original_param_quantizers[name] = qmodule.param_quantizers
                qmodule.param_quantizers = nn.ModuleDict({key: None for key in qmodule.param_quantizers.keys()})

            # disable param quantizers from exclusion list
            if modules_to_exclude:
                with contextlib.suppress(KeyError):
                    fp32_module = name_to_fp32_module_dict[name]
                    if fp32_module in modules_to_exclude:
                        original_param_quantizers[name] = qmodule.param_quantizers
                        qmodule.param_quantizers = nn.ModuleDict({key: None for key in qmodule.param_quantizers.keys()})

        yield

        for name, qmodule in sim.named_qmodules():
            qmodule.input_quantizers = original_input_quantizers[name]
            qmodule.output_quantizers = original_output_quantizers[name]

            if name in original_param_quantizers:
                qmodule.param_quantizers = original_param_quantizers[name]

    @classmethod
    def compute_param_encodings(cls,
                                quantizer: QuantizerBase,
                                x_min: torch.Tensor,
                                x_max: torch.Tensor):
        """
        Compute encodings for parameter quantizer using given x_min and x_max values.

        :param quantizer: Tensor quantizer
        :param x_min: min values
        :param x_max: max values
        """
        quantize_dequantize = QuantizeDequantize(quantizer.shape, quantizer.bitwidth, quantizer.symmetric,
                                                 block_size=quantizer.block_size).to(x_min.device)

        with quantize_dequantize.compute_encodings():
            _ = quantize_dequantize(torch.stack([x_min, x_max]))

        with torch.no_grad():
            quantizer.min.copy_(quantize_dequantize.min)
            quantizer.max.copy_(quantize_dequantize.max)

    @classmethod
    def _is_symmetric_quantizer(cls, quantizer: AffineQuantizerBase):
        # pylint: disable=protected-access
        return quantizer._symmetric

    @classmethod
    def _freeze_quantizer_encoding(cls, quantizer: QuantizerBase):
        # pylint: disable=protected-access
        quantizer.requires_grad_(False)
        quantizer.allow_overwrite(False)

    @classmethod
    def _get_quantized_weight(cls, quant_module: BaseQuantizationMixin):
        w = quant_module.weight
        return quant_module.param_quantizers['weight'](w)

    @classmethod
    def _get_original_module(cls, quant_module: BaseQuantizationMixin):
        return quant_module

    @staticmethod
    def _get_input_channel_block_size(quant_module):
        if not isinstance(quant_module, (torch.nn.Linear, torch.nn.Conv2d)):
            raise NotImplementedError('Unsupported module type: ', type(quant_module))
        if quant_module.param_quantizers['weight'].block_size is None:
            # Per tensor or per channel case. For either one, treat loss computation as per channel
            return quant_module.weight.shape[1]
        return quant_module.weight.shape[1] // quant_module.param_quantizers['weight'].shape[1]

    @staticmethod
    def _get_indices_to_reduce(block_size, reshaped_weight):
        """
        Return indices in reshaped_weight corresponding to block_sizes. Reshaped_weight is expected to contain
        alternating dimensions of num_blocks and block_sizes.
        """
        indices_to_reduce = []
        for idx, _ in enumerate(block_size):
            indices_to_reduce.insert(0, (len(reshaped_weight.shape) - 2 * idx) - 1)
        return indices_to_reduce

    @classmethod
    def get_min_and_max_for_candidate_selection(cls, quant_module: BaseQuantizationMixin) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Get min/max values for candidate selection.

        :param quant_module: Quant module to be optimized
        :return: Tuple of min and max values for candidate selection.
        """
        # pylint: disable=protected-access
        assert hasattr(quant_module.param_quantizers['weight'], 'block_size')
        if not isinstance(quant_module, (torch.nn.Conv2d, torch.nn.Linear)):
            raise ValueError('Unsupported module: ', quant_module)

        block_size = quant_module.param_quantizers['weight'].block_size
        if block_size is None:
            # Per tensor or per channel case
            assert _is_reducible(quant_module.weight.shape, quant_module.param_quantizers['weight'].min.shape)
            if cls._is_symmetric_quantizer(quant_module.param_quantizers['weight']):
                max_tensor = reduce(quant_module.weight.abs(),
                                    quant_module.param_quantizers['weight'].shape, torch.max).values
                min_tensor = -max_tensor
            else:
                max_tensor = reduce(quant_module.weight,
                                    quant_module.param_quantizers['weight'].shape, torch.max).values
                min_tensor = reduce(quant_module.weight,
                                    quant_module.param_quantizers['weight'].shape, torch.min).values
        else:
            # Reshape tensor so each dimension is split into (num_blocks, block_size)
            weight = torch_builtins.reshape_tensor_for_blocks(quant_module.weight,
                                                              quant_module.param_quantizers['weight'].shape,
                                                              block_size)
            indices_to_reduce = cls._get_indices_to_reduce(block_size, weight)

            # Obtain max_tensor and min_tensor which are equivalent in shape to the original weight, but with block
            # values modified to be the block minimum and maximum.
            # For example assume the original weight is 1 output channel and 6 input channels, with block size 2:
            # Original weight:           [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
            # Then, max tensor would be: [[2.0, 2.0, 4.0, 4.0, 6.0, 6.0]]
            if cls._is_symmetric_quantizer(quant_module.param_quantizers['weight']):
                max_tensor = torch.maximum(weight,
                                           torch.amax(weight.abs(),
                                                      indices_to_reduce,
                                                      keepdim=True)).reshape(quant_module.weight.shape)
                min_tensor = -max_tensor
            else:
                max_tensor = torch.maximum(weight,
                                           torch.amax(weight,
                                                      indices_to_reduce,
                                                      keepdim=True)).reshape(quant_module.weight.shape)
                min_tensor = torch.minimum(weight,
                                           torch.amin(weight,
                                                      indices_to_reduce,
                                                      keepdim=True)).reshape(quant_module.weight.shape)

        return min_tensor, max_tensor

    @classmethod
    def _get_candidate(cls, candidate_idx: int, num_candidates: int, min_tensor: torch.Tensor,
                       max_tensor: torch.Tensor):
        """
        Get candidate min and max tensors
        """
        cand_max = max_tensor / num_candidates * (candidate_idx + 1)
        cand_min = min_tensor / num_candidates * (candidate_idx + 1)
        return cand_min, cand_max

    @classmethod
    def _compute_loss(cls,
                      quant_module: BaseQuantizationMixin,
                      x: torch.Tensor,
                      xq: torch.Tensor,
                      w: torch.Tensor,
                      wq: torch.Tensor,
                      params) -> torch.Tensor:
        """
        Compute loss for the given (x, w) and (xq, wq) input/weight pairs. Assumes that block size will be on
        input_channel dimension.
        """
        # pylint: disable=too-many-locals
        # General strategy: split weights and input per block, and run a separate forward pass for each split.
        # In the case of per tensor and per channel, the entire input channel is treated as one block.
        block_size = cls._get_input_channel_block_size(quant_module)
        w_blocks = torch.split(w, block_size, dim=1)
        wq_blocks = torch.split(wq, block_size, dim=1)
        if isinstance(quant_module, torch.nn.Linear):
            x_blocks = torch.split(x, block_size, dim=-1)
            xq_blocks = torch.split(xq, block_size, dim=-1)
        else:
            x_blocks = torch.split(x, block_size, dim=-3)
            xq_blocks = torch.split(xq, block_size, dim=-3)

        block_losses = []
        for idx, x_block in enumerate(x_blocks):
            xqwq, xw = cls.compute_outputs(quant_module, x_block, xq_blocks[idx], w_blocks[idx], wq_blocks[idx])
            block_losses.append(cls.compute_recon_loss(xqwq, xw, params))
        # Stack losses in the input channel dimension
        block_losses = torch.stack(block_losses, dim=-1)
        return block_losses

    @classmethod
    def optimize_module(cls,
                        quant_module: BaseQuantizationMixin,
                        x: torch.Tensor,
                        xq: torch.Tensor,
                        params: SeqMseParams):
        """
        Find and freeze optimal parameter encodings candidate for given module.

        :param quant_module: Quant module to be optimized
        :param x: Inputs to module from FP32 model
        :param xq: Inputs to module from QuantSim model
        :param params: Sequenial MSE parameters
        """
        # pylint: disable=too-many-locals
        min_tensor, max_tensor = cls.get_min_and_max_for_candidate_selection(quant_module)

        total_loss = []
        for i in range(params.num_candidates):
            cand_min, cand_max = cls._get_candidate(i, params.num_candidates, min_tensor, max_tensor)
            cls.compute_param_encodings(quant_module.param_quantizers['weight'], cand_min, cand_max)
            w = quant_module.weight
            wq = cls._get_quantized_weight(quant_module)
            with torch.no_grad():
                for batch_idx in range(params.num_batches):
                    if batch_idx == 0:
                        loss = cls._compute_loss(quant_module, x[batch_idx], xq[batch_idx], w, wq, params)
                    else:
                        loss += cls._compute_loss(quant_module, x[batch_idx], xq[batch_idx], w, wq, params)
                total_loss.append(loss)

        best_indices = torch.stack(total_loss).min(0)[1]
        block_size = cls._get_input_channel_block_size(quant_module)
        # In the input_channels dimension, best_indices is of size num_blocks. We use repeat_interleave to expand
        # each blockwise index into block_size number of indices. This makes best_indices input_channels dimension
        # become size num_blocks * block_size, and allows for elementwise operation with min_tensor and max_tensor.
        if block_size != quant_module.weight.shape[1]:
            best_indices = best_indices.repeat_interleave(block_size, dim=-1)

        # Unsqueeze best_indices until it matches dim length of max_tensor
        while best_indices.dim() < max_tensor.dim():
            best_indices = best_indices[..., None]

        min_tensor, max_tensor = cls._get_candidate(best_indices, params.num_candidates, min_tensor, max_tensor)

        # Compute and freeze parameter encodings using best candidate
        cls.compute_param_encodings(quant_module.param_quantizers['weight'], min_tensor, max_tensor)
        cls._freeze_quantizer_encoding(quant_module.param_quantizers['weight'])

# Global variables for compatibility
apply_seq_mse = SequentialMse.apply_seq_mse
get_candidates = SequentialMse.get_candidates
optimize_module = SequentialMse.optimize_module

# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

import contextlib
from typing import Optional, Union, List
import torch
from torch.utils.data import DataLoader

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
import aimet_common.libpymo as libpymo

from aimet_torch._base.seq_mse import SequentialMseBase, SeqMseParams
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper, QcQuantizeOpMode
from aimet_torch.v1.tensor_quantizer import TensorQuantizer, StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer
from aimet_torch.v1.quantsim import QuantizationSimModel

__all__ = [
    'SequentialMse',
    'SeqMseParams',
    'apply_seq_mse',
    'get_candidates',
    'optimize_module',
]

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
        # pylint: disable=protected-access
        assert sim._quant_scheme in (QuantScheme.post_training_tf, QuantScheme.training_range_learning_with_tf_init), \
            "Use TF quant-scheme with sequential MSE."

        return super().apply_seq_mse(model, sim, data_loader, params, modules_to_exclude, checkpoints_config)


    @classmethod
    @contextlib.contextmanager
    def temporarily_disable_quantizers(
            cls,
            model: torch.nn.Module,
            sim: QuantizationSimModel,
            modules_to_exclude: Optional[List[torch.nn.Module]],
    ):
        """
        For given quantsim model, disable quantizers needed to be disabled before applying sequential MSE.

        :param model: Original fp32 model
        :param sim: QuantizationSimModel object
        :param modules_to_exclude: List of supported modules to exclude when applying Sequential MSE
        :return: List of quantizers to be disabled.
        """
        quantizers_to_be_disabled = cls._get_quantizers_to_be_disabled(model, sim, modules_to_exclude)

        for quantizer in quantizers_to_be_disabled:
            quantizer.enabled = False
        yield
        for quantizer in quantizers_to_be_disabled:
            quantizer.enabled = True

    @classmethod
    def compute_all_param_encodings(cls, sim: QuantizationSimModel):
        """
        Compute encodings for all parameters, needed for initializing Sequential MSE

        :param sim: Quant sim
        """
        for _, quant_wrapper in sim.quant_wrappers():
            for name, quantizer in quant_wrapper.param_quantizers.items():
                quantizer.reset_encoding_stats()
                quantizer.update_encoding_stats(getattr(quant_wrapper, name).data)
                quantizer.compute_encoding()

            # Wrapper mode must be set to ACTIVE because the wrapper's quantize_dequantize_params() will only call
            # into the param tensor quantizer's quantize_dequantize() if the mode isn't PASSTHROUGH.
            quant_wrapper.set_mode(QcQuantizeOpMode.ACTIVE)

    @classmethod
    def optimize_module(cls,
                        quant_module: QcQuantizeWrapper,
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
        per_channel_min, per_channel_max = cls.get_per_channel_min_and_max(quant_module)
        candidates = cls.get_candidates(params.num_candidates, per_channel_max, per_channel_min)

        total_loss = []
        for cand_max, cand_min in candidates:
            cls.compute_param_encodings(quant_module.param_quantizers['weight'], cand_min, cand_max)
            w = quant_module.weight
            wq = cls._get_quantized_weight(quant_module)
            loss = torch.zeros(len(cand_max), device=w.device)
            with torch.no_grad():
                for batch_idx in range(params.num_batches):
                    xqwq, xw = cls.compute_outputs(quant_module, x[batch_idx], xq[batch_idx], w, wq)
                    loss += cls.compute_recon_loss(xqwq, xw, params)
                total_loss.append(loss)

        best_indices = torch.stack(total_loss).min(0, keepdim=True)[1]
        _logger.debug("Indices of optimal candidate: %s", best_indices.squeeze(0)[:params.num_candidates].tolist())
        best_max = torch.stack([cand_max for cand_max, _ in candidates]).gather(0, best_indices)[0]
        best_min = torch.stack([cand_min for _, cand_min in candidates]).gather(0, best_indices)[0]

        # Compute and freeze parameter encodings using best candidate
        cls.compute_param_encodings(quant_module.param_quantizers['weight'], best_min, best_max)
        cls._freeze_quantizer_encoding(quant_module.param_quantizers['weight'])

    @classmethod
    def compute_param_encodings(cls,
                                quantizer: Union[StaticGridPerTensorQuantizer, StaticGridPerChannelQuantizer],
                                x_min: torch.Tensor,
                                x_max: torch.Tensor):
        """
        Compute encodings for parameter quantizer using given x_min and x_max values.

        :param quantizer: Tensor quantizer
        :param x_min: min values
        :param x_max: max values
        """
        tensor = torch.stack([x_min, x_max], dim=-1)
        quantizer.reset_encoding_stats()
        quantizer.update_encoding_stats(tensor)
        quantizer.compute_encoding()

    @classmethod
    def _is_symmetric_quantizer(cls, quantizer: TensorQuantizer):
        return quantizer.use_symmetric_encodings

    @classmethod
    def _freeze_quantizer_encoding(cls, quantizer: TensorQuantizer):
        return quantizer.freeze_encoding()

    @classmethod
    def _get_quantized_weight(cls, quant_module: QcQuantizeWrapper):
        w = quant_module.weight
        return quant_module.param_quantizers['weight'].quantize_dequantize(w, libpymo.RoundingMode.ROUND_NEAREST)

    @classmethod
    def _get_original_module(cls, quant_module: QcQuantizeWrapper):
        # pylint: disable=protected-access
        return quant_module._module_to_wrap


# Global variables for compatibility
apply_seq_mse = SequentialMse.apply_seq_mse
get_candidates = SequentialMse.get_candidates
optimize_module = SequentialMse.optimize_module

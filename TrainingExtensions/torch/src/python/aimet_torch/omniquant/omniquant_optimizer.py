# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Optimizer for Omniquant """

import os
import tempfile
import torch

from aimet_torch.utils import CachedDataset
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch._base.adaround.activation_sampler import get_block_inputs, get_block_outputs

from .decoder_processor import get_transformer_processor
from .omniquant_config import OmniquantConfig

class Omniquant:
    """
    Omniqunat for Post Training Quantization (PTQ)
    """
    @classmethod
    def apply_omniquant(cls, quant_sim: QuantizationSimModel, model: torch.nn.Module, omniquant_config: OmniquantConfig, dataloader,
                        output_path: str) -> torch.nn.Module:
        """
        Returns model with with omniquant weight, and save model with new weightsto output_path.

        :param quant_sim: QuantizationSimModel object to optimize with Omniquant.
        :param model: Original fp32 model from which quant_sim was created.
        :param omniquant_config: Configuration for Omniquant optimization.
        :param dataloader: Dataloader used to train model.
        :param output_path: path where to store artifacts.
        :return: Model with Omniquant weights.
        """
        quant_sim.model = cls._apply_omniquant(quant_sim, model, omniquant_config, dataloader, output_path)
        return quant_sim.model

    # pylint: disable=too-many-locals
    # pylint: disable=unused-variable
    # pylint: disable=unused-argument
    @classmethod
    def _apply_omniquant(cls, quant_sim: QuantizationSimModel, model: torch.nn.Module, omniquant_config: OmniquantConfig,
                         dataloader, output_path: str) -> torch.nn.Module:
        """
        Implemenatation to run omniquant optimization block by block. Return model with optimized weights.

        :param quant_sim: QuantizationSimModel object to optimize with Omniquant.
        :param model: Original fp32 model from which quant_sim was created.
        :param omniquant_config: Configuration for Omniquant optimization.
        :param dataloader: Dataloader used to train model.
        :param output_path: path where to store artifacts.
        :return: Model with Omniquant weights.
        """
        transformer_processor = get_transformer_processor(model)
        fp_transformer_block_list = transformer_processor.get_decoder_list(model)
        qt_transformer_block_list = transformer_processor.get_decoder_list(quant_sim.model)
        device = model.device

        def calibration_wrapper(model, dataloader):
            for batch_id, batch in enumerate(dataloader):
                if batch_id < omniquant_config.num_batch:
                    batch = tuple(batch) # Dataloader returns List[torch.Tensor]
                    model(*batch)
                else:
                    break

        quant_sim.compute_encodings(calibration_wrapper, dataloader)

        with tempfile.TemporaryDirectory() as tempdir:
            cached_dir = os.path.join(tempdir, 'cached_dataset')
            cached_dataset = CachedDataset(dataloader, omniquant_config.num_batch, cached_dir)

            cached_fp_dataset, cached_qt_dataset = get_block_inputs(
                model, quant_sim, ".".join([transformer_processor.transformer_block_list_path, "0"]), cached_dataset, omniquant_config.cache_on_cpu,
                lambda model, input: model.forward(*input), omniquant_config.num_batch, cached_dir, incl_kwargs=True
            )

            for block_num, (fp_block, qt_block) in enumerate(zip(fp_transformer_block_list, qt_transformer_block_list)):
                fp_let_pair_list = transformer_processor.get_let_module_pair(fp_block)
                qt_let_pair_list = transformer_processor.get_let_module_pair(qt_block)

                for epoch in range(omniquant_config.num_epoch):
                    for batch_num in range(omniquant_config.num_batch):
                        fp_input, qt_input = cached_fp_dataset[batch_num], cached_qt_dataset[batch_num]
                        # Do block-wise training.

                get_block_outputs(
                        fp_block, qt_block, False, cached_fp_dataset, cached_qt_dataset, omniquant_config.cache_on_cpu,
                        lambda fp_block, *args, **kwargs: fp_block(*args, **kwargs), device, cached_dir
                    )

        # pylint: disable=protected-access
        # QDQ on models to fold quantizations into weight params.
        quant_sim._apply_qdq_to_model_parameters(quant_sim.model)
        # pylint: disable=unnecessary-comprehension
        all_modules_in_original_model = [module for module in quant_sim.model.modules()]
        quant_sim._remove_quantization_wrappers(quant_sim.model, all_modules_in_original_model)

        return quant_sim.model

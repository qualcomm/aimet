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

import copy
import pytest
import tempfile
import torch
import numpy as np
from onnx import numpy_helper
from aimet_common import libpymo
from aimet_onnx.adaround.adaround_tensor_quantizer import AdaroundTensorQuantizer
from aimet_onnx.adaround.adaround_loss import AdaroundHyperParameters
from aimet_onnx.adaround.adaround_optimizer import AdaroundOptimizer
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.adaround.utils import ModelData
from .models import models_for_tests as test_models
from aimet_onnx.utils import CachedDataset


class TestAdaroundOptimizer:
    """
    Test functions in utils
    """
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="This unit-test is meant to be run on GPU")
    @pytest.mark.parametrize("warm_start", [1.0, 0.2])
    def test_optimize_rounding(self, warm_start):
        np.random.seed(0)
        torch.manual_seed(0)
        model = test_models.single_residual_model()
        model_data = ModelData(model.model)
        sim = QuantizationSimModel(copy.deepcopy(model))
        param_to_tq_dict = create_param_to_tensor_quantizer_dict(sim)

        quant_module = model_data.module_to_info['/conv1/Conv']

        old_weights = torch.from_numpy(numpy_helper.to_array(quant_module.params['weight'].tensor)).clone()

        data_loader = dataloader()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cached_dataset = CachedDataset(data_loader, 1, tmp_dir)
            opt_params = AdaroundHyperParameters(num_iterations=10, reg_param=0.01, beta_range=(20, 2),
                                                warm_start=warm_start)

            AdaroundOptimizer.adaround_module(quant_module, 'input_updated',
                                            model, sim.model, 'Relu', cached_dataset, opt_params,
                                            param_to_tq_dict, True, 0)

            new_weights = torch.from_numpy(numpy_helper.to_array(quant_module.params['weight'].tensor))
            weight_name = quant_module.params['weight'].name
            for tensor in sim.model.model.graph.initializer:
                if tensor.name == weight_name:
                    quantized_weight = torch.from_numpy(numpy_helper.to_array(tensor))
                    break
            assert not torch.all(quantized_weight.eq(new_weights))
            assert torch.all(old_weights.eq(new_weights))
            assert torch.all(param_to_tq_dict[quant_module.params['weight'].name].alpha)

    def test_compute_recons_metrics(self):
        np.random.seed(0)
        torch.manual_seed(0)
        model = test_models.single_residual_model()
        model_data = ModelData(model.model)
        sim = QuantizationSimModel(model)
        param_to_tq_dict = create_param_to_tensor_quantizer_dict(sim)

        quant_module = model_data.module_to_info['/conv1/Conv']

        inp_data = torch.randn(1, 3, 32, 32)
        out_data = torch.randn(1, 32, 18, 18)
        recon_error_soft, recon_error_hard = (
            AdaroundOptimizer._compute_recons_metrics(quant_module, None, inp_data,
                                                      out_data, param_to_tq_dict,
                                                      False))
        assert recon_error_hard > recon_error_soft > 1.4

    def test_compute_output_with_adarounded_weights(self):
        model = test_models.single_residual_model()
        model_data = ModelData(model.model)

        sim = QuantizationSimModel(model)
        param_to_tq_dict = create_param_to_tensor_quantizer_dict(sim)

        quant_module = model_data.module_to_info['/conv2/Conv']
        weights = torch.from_numpy(numpy_helper.to_array(quant_module.params['weight'].tensor))
        inp_data = torch.randn(1, 32, 32, 32)
        out_data = AdaroundOptimizer._compute_output_with_adarounded_weights(weights, quant_module, inp_data,
                                                                             param_to_tq_dict[quant_module.params['weight'].name])
        assert out_data.requires_grad == True
        assert out_data.shape == torch.Size([1, 16, 18, 18])

        quant_module = model_data.module_to_info['/fc/Gemm']
        weights = torch.from_numpy(numpy_helper.to_array(quant_module.params['weight'].tensor))
        inp_data = torch.randn(1, 72)
        out_data = AdaroundOptimizer._compute_output_with_adarounded_weights(weights, quant_module, inp_data,
                                                                             param_to_tq_dict[quant_module.params['weight'].name])
        assert out_data.shape == torch.Size([1, 10])

        model = test_models.transposed_conv_model_without_bn()
        model_data = ModelData(model.model)

        sim = QuantizationSimModel(model)
        param_to_tq_dict = create_param_to_tensor_quantizer_dict(sim)

        quant_module = model_data.module_to_info['/conv1/ConvTranspose']
        weights = torch.from_numpy(numpy_helper.to_array(quant_module.params['weight'].tensor))
        inp_data = torch.randn(10, 10, 4, 4)
        out_data = AdaroundOptimizer._compute_output_with_adarounded_weights(weights, quant_module, inp_data, param_to_tq_dict[quant_module.params['weight'].name])
        assert out_data.shape == torch.Size([10, 10, 6, 6])

def create_param_to_tensor_quantizer_dict(quant_sim):
    """
    Create Adaround tensor quantizers for weight tensor

    :param quant_sim: Quant sim
    """
    param_to_tq_dict = {}
    for param_name in quant_sim.param_names:
        quantizer = quant_sim.qc_quantize_op_dict[param_name]
        ch_axis = -1
        if quantizer.quant_info.usePerChannelMode:
            ch_axis = quantizer.quant_info.channelAxis
        adaround_quantizer = AdaroundTensorQuantizer(quantizer.bitwidth, 'Adaptive', quantizer.quant_scheme,
                                                     quantizer.use_symmetric_encodings, quantizer.enabled,
                                                     ch_axis)

        adaround_quantizer.use_strict_symmetric = quantizer.use_strict_symmetric
        adaround_quantizer.use_unsigned_symmetric = quantizer.use_unsigned_symmetric

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.3
        encodings.min = 0.0
        encodings.offset = 0.2
        encodings.delta = 1
        # Set the encodings and replace by Adaround tensor quantizer
        adaround_quantizer.encoding = encodings
        param_to_tq_dict[param_name] = adaround_quantizer

    return param_to_tq_dict


def dataloader():
    class DataLoader:
        """
        Example of a Dataloader which can be used for running AMPv2
        """
        def __init__(self, batch_size: int):
            """
            :param batch_size: batch size for data loader
            """
            self.batch_size = batch_size

        def __iter__(self):
            """Iterates over dataset"""
            dummy_input = np.random.rand(1, 3, 32, 32).astype(np.float32)
            yield dummy_input

        def __len__(self):
            return 4

    dummy_dataloader = DataLoader(batch_size=2)
    return dummy_dataloader

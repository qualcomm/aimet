# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os
import unittest
import json
import tempfile
from pathlib import Path
import torch
from torch import nn
import spconv.pytorch as spconv

from aimet_common.defs import QuantScheme
from aimet_torch.v1.qc_quantize_op import StaticGridQuantWrapper, StaticGridPerTensorQuantizer,\
    StaticGridPerChannelQuantizer, LearnedGridQuantWrapper, LearnedGridTensorQuantizer
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_common.libpymo import TfEncoding


class SpconvModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=10):
        super().__init__()
        self.conv = spconv.SparseConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        # torch tensor is stored in nchw while spconv requires it to be nhwc
        nhwc_x = x.permute(0, *[i for i in range(2, len(x.shape))], 1)
        spconv_input = spconv.SparseConvTensor.from_dense(nhwc_x)
        output = self.conv(spconv_input)
        return output.dense()


class TestSparseConv(unittest.TestCase):
    def test_sparse_conv_quantsim(self):
        dummy_input = torch.rand(2, 1, 5, 5)

        spconv_model = SpconvModel()
        sim = QuantizationSimModel(spconv_model, dummy_input)

        def dummy_forward(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        sim.compute_encodings(dummy_forward, None)

        # Check if Quantizers were created
        self.assertTrue(isinstance(sim.model.conv, StaticGridQuantWrapper))
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['weight'], StaticGridPerTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['bias'], StaticGridPerTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.output_quantizers[0], StaticGridPerTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.input_quantizers[0], StaticGridPerTensorQuantizer))

        # Check if encodings were created
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['weight'].encoding, TfEncoding))
        self.assertTrue(isinstance(sim.model.conv.output_quantizers[0].encoding, TfEncoding))

    def test_sparse_conv_quantsim_enhanced(self):
        dummy_input = torch.rand(2, 1, 5, 5)

        spconv_model = SpconvModel()
        sim = QuantizationSimModel(spconv_model, dummy_input, 'tf_enhanced')

        def dummy_forward(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        sim.compute_encodings(dummy_forward, None)

        # Check if Quantizers were created
        self.assertTrue(isinstance(sim.model.conv, StaticGridQuantWrapper))
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['weight'], StaticGridPerTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['bias'], StaticGridPerTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.output_quantizers[0], StaticGridPerTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.input_quantizers[0], StaticGridPerTensorQuantizer))

        # Check if encodings were created
        self.assertTrue(sim.model.conv.param_quantizers['weight'].encoding)
        self.assertTrue(sim.model.conv.output_quantizers[0].encoding)

    def test_sparse_conv_per_channel(self):
        dummy_input = torch.rand(2, 1, 5, 5)

        spconv_model = SpconvModel()
        sim = QuantizationSimModel(spconv_model, dummy_input)

        def dummy_forward(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        sim.model.conv.enable_per_channel_quantization()
        sim.compute_encodings(dummy_forward, None)

        # Check if Quantizers were created
        self.assertTrue(isinstance(sim.model.conv, StaticGridQuantWrapper))
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['weight'], StaticGridPerChannelQuantizer))
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['bias'], StaticGridPerChannelQuantizer))
        self.assertTrue(isinstance(sim.model.conv.output_quantizers[0], StaticGridPerTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.input_quantizers[0], StaticGridPerTensorQuantizer))

        # Check if encodings were created
        for encoding in sim.model.conv.param_quantizers['weight'].encoding:
            self.assertTrue(isinstance(encoding, TfEncoding))
        self.assertTrue(isinstance(sim.model.conv.output_quantizers[0].encoding, TfEncoding))

    def test_sparse_conv_qat(self):
        # NOTE: Use asymmetric quantization for parameter, which have gradients both encoding min/max
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True",
                },
                "params": {
                    "is_quantized": "True",
                    "is_symmetric": "False"
                }
            },
            "params": {},
            "op_type": {},
            "supergroups": [],
            "model_input": {},
            "model_output": {}
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file_path = Path(tmp_dir, "quantsim_config.json")
            with open(config_file_path, "w") as f:
                json.dump(quantsim_config, f)

            dummy_input = torch.rand(1, 1, 5, 5)
            spconv_model = SpconvModel()

            def dummy_forward(model, args):
                model.eval()
                with torch.no_grad():
                    model(dummy_input)

            sim = QuantizationSimModel(spconv_model, dummy_input,
                                    quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                    config_file=config_file_path)
            sim.compute_encodings(dummy_forward, None)

        # Check if correct Quantizers are created
        self.assertTrue(isinstance(sim.model.conv, LearnedGridQuantWrapper))
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['weight'], LearnedGridTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.param_quantizers['bias'], LearnedGridTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.output_quantizers[0], LearnedGridTensorQuantizer))
        self.assertTrue(isinstance(sim.model.conv.input_quantizers[0], LearnedGridTensorQuantizer))

        # Check if gradients for weights are initialized as None
        self.assertEqual(sim.model.conv._module_to_wrap.weight.grad, None)

        # Check if gradients for encodings are initialized as None
        self.assertEqual(sim.model.conv.output0_encoding_max.grad, None)
        self.assertEqual(sim.model.conv.output0_encoding_min.grad, None)
        self.assertEqual(sim.model.conv.weight_encoding_max.grad, None)
        self.assertEqual(sim.model.conv.weight_encoding_min.grad, None)

        output = sim.model(dummy_input)
        loss = output.flatten().sum()
        loss.backward()

        # Check if gradients for weights are calculated
        self.assertTrue(isinstance(sim.model.conv._module_to_wrap.weight.grad, torch.Tensor))

        # Check if gradients for encodings are calculated
        self.assertTrue(isinstance(sim.model.conv.output0_encoding_max.grad, torch.Tensor))
        self.assertTrue(isinstance(sim.model.conv.output0_encoding_min.grad, torch.Tensor))
        self.assertTrue(isinstance(sim.model.conv.weight_encoding_max.grad, torch.Tensor))
        self.assertTrue(isinstance(sim.model.conv.weight_encoding_min.grad, torch.Tensor))

        if os.path.exists(config_file_path):
            os.remove(config_file_path)

    def test_sparse_conv_sequential(self):
        class SequentialModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = spconv.SparseSequential(
                    spconv.SparseConv2d(1, 10, 1),
                    nn.ReLU()
                )

            def forward(self, x):
                x = x.permute(0, *[i for i in range(2, len(x.shape))], 1)
                x = spconv.SparseConvTensor.from_dense(x)
                return self.seq(x).dense()

        dummy_input = torch.rand(1, 1, 5, 5)
        def dummy_forward(model, args):
            model.eval()
            with torch.no_grad():
                model(dummy_input)

        sequential_model = SequentialModel()

        from aimet_torch.utils import replace_modules_of_type1_using_constructor
        from aimet_torch.custom.custom_modules import create_quantizable_sparse_sequential, QuantizableSparseSequential
        replace_modules_of_type1_using_constructor(sequential_model, spconv.SparseSequential,
                                                   create_quantizable_sparse_sequential)
        self.assertTrue(isinstance(sequential_model.seq, QuantizableSparseSequential))

        sim = QuantizationSimModel(sequential_model, dummy_input)
        sim.compute_encodings(dummy_forward, None)

        for module in sim.model.seq._modules.values():
            self.assertTrue(isinstance(module, StaticGridQuantWrapper))

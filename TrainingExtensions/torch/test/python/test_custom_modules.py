# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

import pytest
import os
import json
import tempfile
import numpy as np
import torch
import torch.nn as nn

from unittest import mock

from aimet_common.defs import QuantScheme
import aimet_torch._base.nn.modules.custom as aimet_modules
import aimet_torch.v1.quantsim as v1
import aimet_torch.v2.quantsim as v2


class Model(nn.Module):
    def __init__(self, op):
        super(Model, self).__init__()
        self.op1 = op

    def forward(self, input1, input2):
        x = self.op1(input1, input2)
        return x


class Model2(nn.Module):
    def __init__(self, op):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(10, 10, 5, padding=2)
        self.op1 = op

    def forward(self, input):
        x = self.conv1(input)
        x = self.op1(x, input)
        return x


class Model3(nn.Module):
    def __init__(self, op):
        super(Model3, self).__init__()
        self.op1 = op

    def forward(self, *x):
        x = self.op1(*x)
        return x


def dummy_forward_pass(model, args):
    model.eval()
    with torch.no_grad():
        output = model(torch.randn((5, 10, 10, 20)))
    return output


def evaluate(model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Helper function to evaluate model given dummy input
    :param model: torch model
    :param dummy_input: dummy input to model
    """
    model.eval()
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = [dummy_input]
    with torch.no_grad():
        model(*dummy_input)


class TestTrainingExtensionElementwiseOps:
    def test_add_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.Add())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = input1 + input2
        assert np.allclose(out, out1)

    @pytest.mark.parametrize("QuantizationSimModel", [v1.QuantizationSimModel, v2.QuantizationSimModel])
    def test_quantsim_export(self, QuantizationSimModel):
        torch.manual_seed(10)
        model = Model2(aimet_modules.Add())
        dummy_input = torch.randn(5, 10, 10, 20)
        sim = QuantizationSimModel(model, dummy_input)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(path=tmp_dir, filename_prefix='quant_model', dummy_input=dummy_input)

            with open(os.path.join(tmp_dir, 'quant_model.encodings')) as f:
                data = json.load(f)

            assert len(data['activation_encodings']) == 3
            assert len(data['param_encodings']) == 1

    def test_subtract_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.Subtract())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = input1 - input2
        assert np.allclose(out, out1)

    def test_multiply_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.Multiply())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = input1 * input2
        assert np.allclose(out, out1)

    def test_divide_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.Divide())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.div(input1, input2)
        assert np.allclose(out, out1)

    def test_concat_op_two_input_tensors(self):
        torch.manual_seed(10)
        model = Model3(aimet_modules.Concat())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.cat((input1, input2), 0)
        assert np.allclose(out, out1)

    def test_concat_op_four_input_tensors(self):
        torch.manual_seed(10)
        model = Model3(aimet_modules.Concat())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        input3 = torch.rand((5, 10, 10, 20))
        input4 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2, input3, input4)
        out1 = torch.cat((input1, input2, input3, input4), 0)
        assert np.allclose(out, out1)

    @pytest.mark.parametrize("QuantizationSimModel", [v1.QuantizationSimModel, v2.QuantizationSimModel])
    def test_concat_compute_encodings(self, QuantizationSimModel):
        torch.manual_seed(10)
        model = Model3(aimet_modules.Concat())
        dummy_input = torch.randn(5, 10, 10, 20)
        sim = QuantizationSimModel(model, dummy_input)
        sim.compute_encodings(dummy_forward_pass, None)
        print(sim)
        with tempfile.TemporaryDirectory() as tmp_dir:
            sim.export(path=tmp_dir, filename_prefix='concat_model', dummy_input=dummy_input)

    def test_matmul_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.MatMul())
        tensor1 = torch.randn(10, 3, 4)
        tensor2 = torch.randn(10, 4, 5)
        out = model(tensor1, tensor2)
        out1 = torch.matmul(tensor1, tensor2)
        assert np.allclose(out, out1)

    @pytest.mark.parametrize("QuantizationSimModel", [v1.QuantizationSimModel, v2.QuantizationSimModel])
    def test_concat_op_with_qat(self, QuantizationSimModel):
        """
        Test torch.cat op for both QAT and QAT-range learning
        """
        class ModelWithConCatWrapper(nn.Module):
            """ A model with concat wrapper. Use this model for unit testing purposes.
                Expected inputs: 3 inputs, of size (1, 3, 8, 8) """

            def __init__(self):
                super(ModelWithConCatWrapper, self).__init__()
                self.conv1 = nn.Conv2d(10, 10, 5, bias=False)
                self.conv2 = nn.Conv2d(10, 10, 5)
                self.conv3 = nn.Conv2d(10, 10, 5)
                self.conv4 = nn.Conv2d(10, 10, 1)
                self.cat = aimet_modules.Concat(1)
                self.relu1 = nn.ReLU(inplace=True)
                self.relu2 = nn.ReLU(inplace=True)

            def forward(self, *inputs):
                c1 = self.conv1(inputs[0])
                c1 = self.relu1(c1)
                c2 = self.conv2(inputs[1])
                c3 = self.conv3(inputs[2])
                c3 = self.conv4(c3)
                cat_inputs = [c1, c2, c3]
                x = self.cat(*cat_inputs)
                x = self.relu2(x)
                return x

        # Test with torch.cat dim=1 with both QAT and QAT-range learning.
        input_shape = (1, 10, 10, 20)
        dummy_input = [torch.randn(*input_shape), torch.randn(*input_shape), torch.randn(*input_shape)]
        model = ModelWithConCatWrapper().eval()
        quant_schemes = [QuantScheme.post_training_tf_enhanced,
                         QuantScheme.training_range_learning_with_tf_init,
                         QuantScheme.post_training_tf_enhanced,
                         QuantScheme.training_range_learning_with_tf_enhanced_init]
        for quant_scheme in quant_schemes:
            quant_sim = QuantizationSimModel(model, dummy_input, quant_scheme=quant_scheme)
            quant_sim.compute_encodings(evaluate, dummy_input)
            quant_sim.model.train()
            out = quant_sim.model(*dummy_input)
            loss = out.flatten().sum()
            loss.backward()
            assert quant_sim.model.cat.output_quantizers[0] is not None

    @pytest.mark.parametrize("QuantizationSimModel", [v1.QuantizationSimModel, v2.QuantizationSimModel])
    def test_dtypes_to_ignore_for_quantization(self, QuantizationSimModel):
        """
        test dtypes to be ignored for quantization when inputs to elementwise ops are scalar numbers.
        We just skip quantization for scalars.
        """
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(10, 20, 5)
                self.add = aimet_modules.Add()
                self.mul = aimet_modules.Multiply()

            def forward(self, input1):
                x = self.conv(input1)
                x = self.add(x, 2)
                x = self.mul(x, 1)
                return x

        model = Model().eval()
        dummy_input = torch.randn(1, 10, 10, 20)
        model(dummy_input)
        sim = QuantizationSimModel(model, dummy_input)
        sim.compute_encodings(dummy_forward_pass, None)
        sim.model(dummy_input)

        assert all(e is None for e in sim.model.add.export_input_encodings('0.6.1'))
        assert all(e is not None for e in sim.model.add.export_output_encodings('0.6.1'))

        assert all(e is None for e in sim.model.mul.export_input_encodings('0.6.1'))
        assert all(e is not None for e in sim.model.mul.export_output_encodings('0.6.1'))

    @pytest.mark.parametrize("QuantizationSimModel", [v1.QuantizationSimModel, v2.QuantizationSimModel])
    def test_dtypes_to_ignore_for_quantization_quant_scheme_range_learning(self, QuantizationSimModel):
        """
        test dtypes to be ignored for quantization when inputs to elementwise ops are scalar numbers.
        We just skip quantization for scalars.
        """
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv = nn.Conv2d(10, 20, 5)
                self.add = aimet_modules.Add()
                self.mul = aimet_modules.Multiply()

            def forward(self, input1):
                x = self.conv(input1)
                x = self.add(x, 2)
                x = self.mul(x, 1)
                return x

        model = Model().eval()
        dummy_input = torch.randn(1, 10, 10, 20)
        model(dummy_input)
        sim = QuantizationSimModel(model, dummy_input,
                                   quant_scheme=QuantScheme.training_range_learning_with_tf_enhanced_init)
        sim.compute_encodings(dummy_forward_pass, None)
        sim.model.train()
        out = sim.model(dummy_input)
        loss = out.flatten().sum()
        loss.backward()

        assert all(e is None for e in sim.model.add.export_input_encodings('0.6.1'))
        assert all(e is not None for e in sim.model.add.export_output_encodings('0.6.1'))

        assert all(e is None for e in sim.model.mul.export_input_encodings('0.6.1'))
        assert all(e is not None for e in sim.model.mul.export_output_encodings('0.6.1'))

    def test_erf_op(self):
        """
        Test gaussian error function
        """
        model = Model3(aimet_modules.Erf())
        inputs = torch.tensor([0, -1., 10.])

        custom_module_out = model(inputs)
        original_module_out = torch.erf(inputs)
        assert np.allclose(custom_module_out, original_module_out)

    def test_erf_with_other_ops(self):
        """
        Test erf combined with other ops
        """
        class ErfWithOtherOpsModel(nn.Module):
            def __init__(self):
                super(ErfWithOtherOpsModel, self).__init__()
                self.erf = aimet_modules.Erf()
                self.sqrt = aimet_modules.Sqrt()

            def forward(self, *inputs):
                return inputs[0] / 2 * (1 + self.erf(inputs[0] / self.sqrt(torch.tensor(2))))

        model = ErfWithOtherOpsModel()
        dummy_input = torch.randn(2)

        combined_ops_output = model(dummy_input)
        gelu_output = torch.nn.GELU()(dummy_input)
        assert np.allclose(combined_ops_output, gelu_output)

    def test_greater_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.Greater())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.gt(input1, input2)
        assert np.allclose(out, out1)
        assert input1.shape == out.shape

    def test_less_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.Less())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.lt(input1, input2)
        assert np.allclose(out, out1)
        assert input1.shape == out.shape

    def test_greater_equal_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.GreaterEqual())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.ge(input1, input2)
        assert np.allclose(out, out1)
        assert input1.shape == out.shape

    def test_less_equal_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.LessEqual())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.le(input1, input2)
        assert np.allclose(out, out1)
        assert input1.shape == out.shape

    def test_not_equal_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.NotEqual())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.ne(input1, input2)
        assert np.allclose(out, out1)
        assert input1.shape == out.shape

    def test_equal_op(self):
        torch.manual_seed(10)
        model = Model(aimet_modules.Equal())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1, input2)
        out1 = torch.eq(input1, input2)
        assert np.allclose(out, out1)
        assert input1.shape == out.shape

    def test_where_op(self):
        torch.manual_seed(10)
        model = Model3(aimet_modules.Where())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = torch.rand((5, 10, 10, 20))
        out = model(input1 > input2, input1, input2)
        out1 = torch.where(input1 > input2, input1, input2)
        assert np.allclose(out, out1)
        assert input1.shape == out.shape

    def test_mean_op(self):
        torch.manual_seed(10)
        model = Model3(aimet_modules.Mean())
        input1 = torch.rand((5, 10, 10, 20))
        out = model(input1)
        out1 = torch.mean(input1)
        assert np.allclose(out, out1)

    def test_sum_op(self):
        torch.manual_seed(10)
        model = Model3(aimet_modules.Sum())
        input1 = torch.rand((5, 10, 10, 20))
        out = model(input1)
        out1 = torch.sum(input1)
        assert np.allclose(out, out1)

    def test_prod_op(self):
        torch.manual_seed(10)
        model = Model3(aimet_modules.Prod())
        input1 = torch.rand((5, 10, 10, 20))
        out = model(input1)
        out1 = torch.prod(input1)
        assert np.allclose(out, out1)

    def test_log_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.Log())
        inputs = torch.rand(4, 3, 28, 28)

        custom_module_out = model(inputs)
        original_module_out = torch.log(inputs)
        assert np.allclose(custom_module_out, original_module_out)

    def test_abs_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.Abs())
        inputs = torch.rand(4, 3, 28, 28)

        custom_module_out = model(inputs)
        original_module_out = torch.abs(inputs)
        assert np.allclose(custom_module_out, original_module_out)

    def test_neg_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.Neg())
        inputs = torch.rand(4, 3, 28, 28)

        custom_module_out = model(inputs)
        original_module_out = torch.neg(inputs)
        assert np.allclose(custom_module_out, original_module_out)

    def test_slice_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.StridedSlice())
        input1 = torch.rand(4, 3, 28, 28)
        input2 = [[0,4,2]]

        custom_module_out = model(input1, input2)
        original_module_out = input1[input2[0][0]:input2[0][1]:input2[0][2]]
        assert np.allclose(custom_module_out, original_module_out)

    def test_channel_shuffle_op(self):
        torch.manual_seed(42)

        original_module = nn.ChannelShuffle(groups=4)
        custom_module = Model3(aimet_modules.ChannelShuffle(groups=4))

        dummy_input = torch.rand(4, 16, 28, 28)
        custom_module_out = custom_module(dummy_input)
        original_module_out = original_module(dummy_input)
        assert np.allclose(custom_module_out, original_module_out)

    def test_logical_or_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.LogicalOr())
        input1 = torch.rand((5, 10, 10, 20)) > 0.5
        input2 = torch.rand((5, 10, 10, 20)) > 0.5

        custom_module_out = model(input1, input2)
        original_module_out = torch.logical_or(input1, input2)
        assert np.allclose(custom_module_out, original_module_out)

    def test_logical_and_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.LogicalAnd())
        input1 = torch.rand((5, 10, 10, 20)) > 0.5
        input2 = torch.rand((5, 10, 10, 20)) > 0.5

        custom_module_out = model(input1, input2)
        original_module_out = torch.logical_and(input1, input2)
        assert np.allclose(custom_module_out, original_module_out)

    def test_logical_not_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.LogicalNot())
        inputs = torch.rand((5, 10, 10, 20)) > 0.5

        custom_module_out = model(inputs)
        original_module_out = torch.logical_not(inputs)
        assert np.allclose(custom_module_out, original_module_out)

    def test_cast_ops(self):
        torch.manual_seed(42)
        dtypes = [torch.int8, torch.uint8, torch.int16, torch.int32, torch.float16]

        for dtype in dtypes:
            model = Model3(aimet_modules.Cast(dtype))
            inputs = torch.rand((5, 10, 10, 20))

            custom_module_out = model(inputs)
            original_module_out = inputs.to(dtype)
            assert np.allclose(custom_module_out, original_module_out)

    def test_split_ops(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.Split())
        inputs = torch.rand(10, 10, 10)

        for split_size_or_sections, dim in [(3, 0), (3, 1), ([1, 9], 0), ([2, 8], 1)]:
            custom_module_out = model(inputs, split_size_or_sections, dim)
            original_module_out = torch.split(inputs, split_size_or_sections=split_size_or_sections, dim=dim)
            for i in range(len(original_module_out)):
                assert np.allclose(custom_module_out[i], original_module_out[i])

    def test_reshape_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.Reshape())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = [5, 10, 200]

        custom_module_out = model(input1, input2)
        original_module_out = torch.reshape(input1, input2)
        assert np.allclose(custom_module_out, original_module_out)

    def test_permute_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.Permute())
        input1 = torch.rand((5, 10, 10, 20))
        input2 = [0, 1, 3, 2]

        custom_module_out = model(input1, input2)
        original_module_out = torch.permute(input1, input2)
        assert np.allclose(custom_module_out, original_module_out)

    def test_custom_gather_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.CustomGather())
        inputs = torch.rand((5, 10, 20))
        indices_list = [torch.tensor([[0, 1], [1, 2]]), torch.tensor([[0, 1], [-1, -2]])]
        axis = 1

        for indices in indices_list:
            custom_module_out = model(inputs, indices, axis)
            original_module_out = np.take(inputs, indices, axis=axis)
            assert np.allclose(custom_module_out, original_module_out)

    def test_custom_scatternd_op(self):
        torch.manual_seed(42)
        model = Model3(aimet_modules.ScatterND())
        inputs = torch.rand(1, 2, 3)
        indices = torch.tensor([[[[0, 1, 2]], [[0, 1, 0]], [[0, 0, 2]]]])
        updates = torch.rand(1, 3, 1)
        
        original_module_out = inputs.clone()
        original_module_out[0, 1, 2] = updates[0, 0, 0]
        original_module_out[0, 1, 0] = updates[0, 1, 0]
        original_module_out[0, 0, 2] = updates[0, 2, 0]

        custom_module_out = model(inputs, indices, updates)
        assert np.allclose(custom_module_out, original_module_out)

    def test_gather_nd_jit_trace(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gather_nd = aimet_modules.GatherNd(batch_dim=0)
        data = torch.tensor([[0, 1], [2, 3]], device=device)
        indices = torch.tensor([[0, 0], [1, 1]], device=device)

        # Patch torch.jit.is_tracing() as True
        with mock.patch("torch.jit.is_tracing", lambda: True), torch.inference_mode():
            outputs = gather_nd(data, indices)

        assert data.device == outputs.device

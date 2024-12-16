# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
import pickle
import time

import aimet_common.libpymo as libpymo
import numpy as np
import pytest
import torch

from aimet_common.defs import MAP_ROUND_MODE_TO_PYMO, QuantizationDataType
from aimet_torch.v1.qc_quantize_op import QuantScheme
from aimet_torch.v1.qc_quantize_op import StaticGridQuantWrapper, LearnedGridQuantWrapper, SteGatingFuncForParameters, \
    QcQuantizeOpMode
from aimet_torch.v1.tensor_quantizer import LearnedGridTensorQuantizer
from aimet_torch.v1.tensor_quantizer import StaticGridPerTensorQuantizer, QuantizeDequantizeFunc
from aimet_torch import utils
import aimet_torch._base.nn.modules.custom as aimet_modules


class TestQcQuantizeOpStaticGrid:

    def test_update_stats_with_pymo(self):

        device = torch.device('cpu')
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quantize = StaticGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                          quant_scheme=QuantScheme.post_training_tf_enhanced)

        input_var = torch.autograd.Variable(torch.randn(4, 4, 2, 2), requires_grad=False).to(device)
        print(input_var)

        quantize.set_mode(QcQuantizeOpMode.ANALYSIS)

        output = quantize.forward(input_var)
        quantize.compute_encoding()
        actual_encoding = quantize.output_quantizers[0].encoding
        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(actual_encoding.min, actual_encoding.max,
                      actual_encoding.offset, actual_encoding.delta, actual_encoding.bw))

    def test_quantize_dequantize_with_pymo(self):

        device = torch.device('cpu')
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quantize = StaticGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                          quant_scheme=QuantScheme.post_training_tf_enhanced)

        input_var = torch.autograd.Variable(torch.randn(4, 4, 2, 2), requires_grad=True).to(device)

        quantize.set_mode(QcQuantizeOpMode.ANALYSIS)
        output = quantize.forward(input_var)
        quantize.compute_encoding()
        actual_encoding = quantize.output_quantizers[0].encoding

        print("Encoding returned: min={}, max={}, offset={}. delta={}, bw={}"
              .format(quantize.output_quantizers[0].encoding.min,
                      quantize.output_quantizers[0].encoding.max,
                      quantize.output_quantizers[0].encoding.offset,
                      quantize.output_quantizers[0].encoding.delta,
                      quantize.output_quantizers[0].encoding.bw))

        quantize.set_mode(QcQuantizeOpMode.ACTIVE)
        output = quantize.forward(input_var)

    def test_qc_post_training_wrapper(self):
        torch.manual_seed(0)

        encodings = libpymo.TfEncoding()
        encodings.bw, encodings.max, encodings.min, encodings.delta, encodings.offset = 8, 0.5, -1, 1, 0.2

        encodings_new = libpymo.TfEncoding()
        encodings_new.bw, encodings_new.max, encodings_new.min, encodings_new.delta, encodings_new.offset = 8, 0.4, -0.98, 1, 0.2

        output_grad = []
        def hook_fn(m, _, i):

            for grad in i:
                try:
                    output_grad.append(grad)
                except AttributeError:
                    print ("None found for Gradient")

        conv1 = torch.nn.Conv2d(1, 2, 1)
        quantize = StaticGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                          quant_scheme=QuantScheme.post_training_tf_enhanced)
        quantize.train()
        quantize._module_to_wrap.register_backward_hook(hook_fn)

        quantize.input_quantizers[0].enabled = True
        quantize.output_quantizers[0].enabled = True
        quantize.input_quantizers[0].encoding = encodings
        quantize.output_quantizers[0].encoding = encodings

        new_input = torch.autograd.Variable(torch.tensor([[[[0.6469]]], [[[-0.9]]]]), requires_grad=True)
        quantize.set_mode(QcQuantizeOpMode.ACTIVE)
        out = quantize(new_input)

        quantize.input_quantizers[0].encoding = encodings_new
        quantize.output_quantizers[0].encoding = encodings_new
        quantize.param_quantizers['weight'].encoding = encodings_new

        loss = out.flatten().sum()
        loss.backward()

        # Check if input gradient got clipped
        for i, val in enumerate(new_input):
            if encodings_new.min > val or val > encodings_new.max:
                assert new_input.grad[0][i] == 0.0

        # Check if output gradient got clipped
        output_grad = output_grad[0].flatten()
        assert output_grad[0] == 1.0
        assert output_grad[1] == 1.0
        assert output_grad[2] == 1.0
        assert output_grad[3] == 0.0

        # Check if weight gradient got clipped
        weight_tensor = quantize._module_to_wrap.weight.flatten()
        weight_tensor_grad = quantize._module_to_wrap.weight.grad.flatten()
        for i, val in enumerate(weight_tensor):
            if encodings_new.min > val or val > encodings_new.max:
                assert weight_tensor_grad[i] == 0.0

    def test_quantize_maxpool_with_indices(self):
        """ Test that maxpool2d returning int tensor can be quantized """
        maxpool = torch.nn.MaxPool2d(2, return_indices=True)
        quantize_op = StaticGridQuantWrapper(maxpool, weight_bw=8, activation_bw=8, round_mode='nearest',
                                             quant_scheme=QuantScheme.post_training_tf_enhanced, num_outputs=2)
        inp = torch.rand((1, 3, 8, 8))
        quantize_op.set_mode(QcQuantizeOpMode.ANALYSIS)
        quantize_op(inp)
        quantize_op.compute_encoding()
        quantize_op.set_mode(QcQuantizeOpMode.ACTIVE)
        out, indices = quantize_op(inp)

        # Check that one of the outputs of quantize_op is the indices with dtype int64
        assert indices.dtype == torch.int64
        assert quantize_op.output_quantizers[0] is not None

    def test_quantize_only_asymmetric_cpu(self):
        """
        Test tensor quantizer quantize only asymmetric functionality
        Note: Quantize-only mode uses symmetric range for output tensor by default
        """
        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=False, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 2.23
        encodings.min = -5.19
        encodings.offset = -178
        quantizer.encoding = encodings

        inp_tensor = torch.tensor([-7, -5, -3, 0, .1, 2.5])
        quant_out = quantizer.quantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([-128., -122.,  -53.,   50.,   53.,  127.], dtype=torch.float32)
        assert torch.equal(quant_out, expected_out)

    def test_quantize_only_symmetric_signed_cpu(self):
        """ Test tensor quantizer quantize only symmetric signed functionality on cpu """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=True, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = -5.20
        encodings.offset = -128

        # delta is 0.040745098
        quantizer.encoding = encodings

        # Test quantize only on cpu
        inp_tensor_gpu = torch.tensor([-7, -5, -3, 0, .1, 2.5])
        quant_out = quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([-128, -123, -74, 0, 2, 61], dtype=torch.float32)
        assert torch.equal(quant_out, expected_out)

    def test_quantizer_ignore_data_type_to_quantize_static_grid(self):
        relu = torch.nn.ReLU()
        quantizer = StaticGridQuantWrapper(relu, weight_bw=8, activation_bw=8, round_mode='nearest',
                                          quant_scheme=QuantScheme.post_training_tf_enhanced)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = 0.0
        encodings.offset = 0

        # delta is 0.020352941
        quantizer.encoding = encodings

        inputs = torch.tensor([[True, True, False, False]])
        quant_out = quantizer._quantize_activation(quantizer.output_quantizers, inputs)[0]
        expected_output = torch.tensor([True, True, False, False])
        assert torch.equal(quant_out, expected_output)

    def test_quantize_only_symmetric_unsigned_cpu(self):
        """
        Test tensor quantizer quantize only symmetric unsigned functionality on cpu
        Note: Quantize-only mode uses symmetric range for output tensor by default
        """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=True, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
        quantizer.use_unsigned_symmetric = True
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = 0.0
        encodings.offset = 0

        # delta is 0.020352941
        quantizer.encoding = encodings

        # Test quantize only on cpu
        inp_tensor_gpu = torch.tensor([0, 1.2, 1.5, 4.0, 4.9, 5.3])
        quant_out = quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([  0.,  59.,  74., 197., 241., 255.], dtype=torch.float32)
        assert torch.equal(quant_out, expected_out)

    def test_quantize_dequantize_fp16_cpu(self):
        """ Test tensor quantizer quantize only symmetric unsigned functionality on cpu """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=16, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=True, enabled_by_default=True,
                                                 data_type=QuantizationDataType.float)

        # Test quantize-dequantize only on cpu
        inp_tensor = torch.tensor([0.3, 2.1, 1.1, 0.9, 0.1, 1.3])
        quant_out = quantizer.quantize_dequantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        assert torch.allclose(inp_tensor, quant_out, rtol=0.1)

    @pytest.mark.cuda
    def test_quantize_dequantize_fp16_gpu(self):
        """ Test tensor quantizer quantize only symmetric unsigned functionality on cpu """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=16, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=True, enabled_by_default=True,
                                                 data_type=QuantizationDataType.float)

        # Test quantize-dequantize only on cpu
        inp_tensor = torch.tensor([0.3, 2.1, 1.1, 0.9, 0.1, 1.3])
        quant_out = quantizer.quantize_dequantize(inp_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        assert torch.allclose(inp_tensor, quant_out, rtol=0.1)

    def test_compute_encodings_fp16_disable_tensor(self):
        """ Negative test to make sure the encodings are not computed if 'enabled' field is set to False """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=16, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=True, enabled_by_default=True,
                                                 data_type=QuantizationDataType.float)

        # Test if encodings are computed for a float tensor
        quantizer.compute_encoding()
        assert quantizer.encoding == None

    @pytest.mark.cuda
    def test_quantize_only_asymmetric_gpu(self):
        """
        Test tensor quantizer quantize only asymmetric functionality on gpu
        Note: Quantize-only mode uses symmetric range for output tensor by default
        """

        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf,
                                                 use_symmetric_encodings=False, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 2.23
        encodings.min = -5.19
        encodings.offset = -178
        quantizer.encoding = encodings

        # Test quantize only on gpu
        inp_tensor_gpu = torch.tensor([-7, -5, -3, 0, .1, 2.5], device=torch.device('cuda'))
        quant_out = quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([-128., -122.,  -53.,   50.,   53.,  127.], dtype=torch.float32, device=torch.device('cuda'))
        assert torch.equal(quant_out, expected_out)

    @pytest.mark.cuda
    def test_quantize_only_symmetric_signed_gpu(self):
        """ Test tensor quantizer quantize only symmetric signed functionality on gpu """

        post_training_tensor_quantizer = \
            StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                         quant_scheme=QuantScheme.post_training_tf,
                                         use_symmetric_encodings=True, enabled_by_default=True,
                                         data_type=QuantizationDataType.int)
        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = -5.20
        encodings.offset = -128

        # delta is 0.040745098
        post_training_tensor_quantizer.encoding = encodings

        # Test quantize only on gpu
        inp_tensor_gpu = torch.tensor([-7, -5, -3, 0, .1, 2.5], device=torch.device('cuda'))
        quant_out = post_training_tensor_quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([-128, -123, -74, 0, 2, 61], dtype=torch.float32, device=torch.device('cuda'))
        assert torch.equal(quant_out, expected_out)

    @pytest.mark.cuda
    def test_quantize_only_symmetric_unsigned_gpu(self):
        """
        Test tensor quantizer quantize only symmetric unsigned functionality on gpu
        Note: Quantize-only mode uses symmetric range for output tensor by default
        """

        post_training_tensor_quantizer = \
            StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                         quant_scheme=QuantScheme.post_training_tf,
                                         use_symmetric_encodings=True, enabled_by_default=True,
                                         data_type=QuantizationDataType.int)
        post_training_tensor_quantizer.use_unsigned_symmetric = True

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5.19
        encodings.min = 0.0
        encodings.offset = 0

        # delta is 0.020352941
        post_training_tensor_quantizer.encoding = encodings

        # Test quantize only on gpu
        inp_tensor_gpu = torch.tensor([0, 1.2, 1.5, 4.0, 4.9, 5.3], device=torch.device('cuda'))
        quant_out = post_training_tensor_quantizer.quantize(inp_tensor_gpu, MAP_ROUND_MODE_TO_PYMO['nearest'])
        expected_out = torch.tensor([0.,  59.,  74., 197., 241., 255.], dtype=torch.float32, device=torch.device('cuda'))
        assert torch.equal(quant_out, expected_out)

    @pytest.mark.skip(reason="causing random failures in CI. disabling to see in regressions if this is the root cause")
    def test_qc_post_training_wrapper_mem_leak(self):
        torch.manual_seed(0)

        rand_tensor = torch.rand(1, 10, 20, 20)
        quant = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                                           use_symmetric_encodings=False, enabled_by_default=True,
                                           data_type=QuantizationDataType.int)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        baseline_mem = None

        for i in range(1000):
            quant.reset_encoding_stats()
            quant.update_encoding_stats(rand_tensor)
            quant.compute_encoding()
            if not baseline_mem:
                baseline_mem = process.memory_info().rss

        quant.reset_encoding_stats()
        delta = process.memory_info().rss - baseline_mem
        assert 100000 >= delta

    def test_compute_encoding_for_tensor_quantizer_with_no_stats(self):
        quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                 quant_scheme=QuantScheme.post_training_tf_enhanced,
                                                 use_symmetric_encodings=False, enabled_by_default=True,
                                                 data_type=QuantizationDataType.int)
        quantizer.compute_encoding()
        assert quantizer._encoding == []

    def test_custom_stegatingfuncforparameters(self):
        """
        test SteGatingFuncForParameters function
        """
        inputs = torch.randn(3, requires_grad=True)
        tensor_to_be_copied = torch.randn(3)
        print(tensor_to_be_copied)
        outputs = SteGatingFuncForParameters.apply(None, inputs)
        print(outputs)
        outputs = [out.clone() for out in outputs]
        # After clone(), following copy operation should succeed, otherwise throws RuntimeError
        outputs = [out.copy_(tensor_to_be_copied) for out in outputs]
        print(outputs)
        
    def test_prefreezed_encodings(self):
        """ test quantize dequantize ops with prefreezed encodings """

        torch.manual_seed(2023)

        in_features = 200
        out_features = 100

        x = torch.randn((4, 200))
        weight = torch.randn((out_features, in_features)).clamp(-1, 1)
        linear = torch.nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            linear.weight.copy_(weight)

        wrapper = StaticGridQuantWrapper(linear, weight_bw=8, activation_bw=8, round_mode='nearest',
                                         quant_scheme=QuantScheme.post_training_tf)

        # Enable input/output quantizers
        for quantizer_type in ["input_quantizers", "output_quantizers"]:
            quantizers = getattr(wrapper, quantizer_type)
            assert len(quantizers) == 1
            quantizers[0].enabled = True

        gated_wrapper = copy.deepcopy(wrapper)

        assert torch.allclose(wrapper(x), gated_wrapper(x), rtol=0.1)

        enc = libpymo.TfEncoding()
        enc.bw, enc.max, enc.min, enc.delta, enc.offset = 8, 0.5, -0.5, 1, 0.2

        for quantizer_type in ["input_quantizers", "output_quantizers"]:
            quantizers = getattr(gated_wrapper, quantizer_type)
            assert len(quantizers) == 1

            # Prefreeze encodings
            quantizers[0].encoding = enc
            quantizers[0]._is_encoding_frozen = True
            assert not torch.allclose(wrapper(x), gated_wrapper(x), rtol=0.1)

            # Reset encodings to None and unfreeze encodings
            quantizers[0]._is_encoding_frozen = False
            quantizers[0].encoding = None
            assert torch.allclose(wrapper(x), gated_wrapper(x), rtol=0.1)


class RoundStraightThrough(torch.autograd.Function):
    """
    Defining gradient of rounding function as pass-though since round is a non-linearity.
    """
    @staticmethod
    def forward(ctx, *x):
        return torch.round(*x)

    @staticmethod
    def backward(ctx, *output_grad):
        return output_grad


class TestQcQuantizeOpLearnedGrid:

    def test_trainable_tensor_quantizer_forward_backward(self):
        tensor_quantizer = LearnedGridTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                      quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                      use_symmetric_encodings=True,
                                                      enabled_by_default=True,
                                                      data_type=QuantizationDataType.int)

        def test_with_options(is_unsigned_symmetric, encoding_min):
            tensor_quantizer.is_unsigned_symmetric = is_unsigned_symmetric

            encoding_min = torch.nn.Parameter(torch.FloatTensor([encoding_min]))
            encoding_max = torch.nn.Parameter(torch.FloatTensor([5]))
            # will fall into signed symmetric because encoding_min < 0
            tensor = torch.FloatTensor(2, 1, 3, 5).uniform_(-10, 10)
            tensor = tensor_quantizer.quantize_dequantize(tensor, encoding_min, encoding_max)

            if encoding_min < 0:
                # sigend symmetric case
                min_val = -(torch.max(torch.abs(encoding_min), torch.abs(encoding_max))[0]).item()
                grid_length = (5 - min_val) / (2**8 - 2)
            else:
                # unsigned symmetric case
                min_val = 0
                grid_length = (5 - encoding_min.item()) / (2**8 - 1)

            assert np.amax(tensor.detach().numpy(), axis=(0, 1, 2, 3)) <= 5 + grid_length
            assert np.amin(tensor.detach().numpy(), axis=(0, 1, 2, 3)) >= min_val - grid_length

        # this case will fall into signed symmetric since encoding_min < 0
        test_with_options(is_unsigned_symmetric=True, encoding_min=-5)

        # unsigned symmetirc case test
        test_with_options(is_unsigned_symmetric=True, encoding_min=0)

        # signed symmetric test
        test_with_options(is_unsigned_symmetric=False, encoding_min=-5)

        # signed symmetric test
        test_with_options(is_unsigned_symmetric=False, encoding_min=-1)



    @staticmethod
    def perform_auto_grad_computation(custom_input, min_value, max_value, n=0., p=255.):
        """
        helper to perform auto grad computation
        :return:
        """
        ste = RoundStraightThrough.apply

        def quant_dequant_param(x, _min, _max):
            delta = (_max - _min) / p
            offset = ste(_min / delta)
            x = ste(x / delta) - offset
            x = torch.clamp(x, n, p)
            x = (x + offset) * delta
            return x

        # use same tensor for auto and custom grad
        a_input_tensor = copy.deepcopy(custom_input)
        a_enc_min = torch.nn.Parameter(torch.Tensor([min_value]), requires_grad=True)
        a_enc_max = torch.nn.Parameter(torch.Tensor([max_value]), requires_grad=True)

        # after forward pass for quant op, we get output y
        y = quant_dequant_param(a_input_tensor, a_enc_min, a_enc_max)
        loss = y.flatten().sum()
        loss.backward()

        return a_input_tensor.grad, a_enc_min.grad, a_enc_max.grad

    def test_quantizer_ignore_data_type_to_quantize_learned_grid(self):
        relu = torch.nn.ReLU()
        quantizer = LearnedGridQuantWrapper(relu, weight_bw=8, activation_bw=8, round_mode='nearest',
                                            quant_scheme=QuantScheme.training_range_learning_with_tf_init, device='cpu')

        inputs = torch.tensor([[True, True, False, False]])
        quant_out = quantizer._quantize_activation(inputs, quantizer.output_quantizers, 'output')[0]
        expected_output = torch.tensor([True, True, False, False])
        assert torch.equal(quant_out, expected_output)

    @staticmethod
    def perform_optimized_custom_grad_computation(custom_input, min_value, max_value):
        oc_input_tensor = copy.deepcopy(custom_input)
        oc_enc_min = torch.nn.Parameter(torch.Tensor([min_value]), requires_grad=True)
        oc_enc_max = torch.nn.Parameter(torch.Tensor([max_value]), requires_grad=True)

        # To apply our Function, we use Function.apply method.
        # We alias this as 'optimized_custom_op'.
        optimized_custom_op = QuantizeDequantizeFunc.apply

        # Forward pass: compute predicted y using operations; we compute
        # using our "optimized custom" autograd operation.
        tensor_quantizer = LearnedGridTensorQuantizer(bitwidth=8, round_mode="nearest",
                                                      quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                      use_symmetric_encodings=False,
                                                      enabled_by_default=True,
                                                      data_type=QuantizationDataType.int)
        y_pred = optimized_custom_op(oc_input_tensor, oc_enc_min, oc_enc_max, tensor_quantizer)

        loss = y_pred.flatten().sum()
        # Use custom grad to compute the backward pass.ctx.p = p
        loss.backward()

        return oc_input_tensor.grad, oc_enc_min.grad, oc_enc_max.grad

    def test_custom_gradient_math_for_range_learning(self):
        """
        Unit test to validate custom gradient computation with auto grad computation.
        :return: None
        """

        dtype = torch.float
        device = torch.device("cpu")

        # check if it covers all the conditions
        _min = -0.5
        _max = 0.9
        data_size = (100, 100)

        torch.manual_seed(0)
        custom_input = torch.rand(data_size, requires_grad=True, dtype=dtype, device=device)

        a_input_grad, a_min_grad, a_max_grad = TestQcQuantizeOpLearnedGrid.perform_auto_grad_computation(
            custom_input, _min, _max
        )

        # Optimized custom gradient computation
        oc_input_grad, oc_min_grad, oc_max_grad = TestQcQuantizeOpLearnedGrid.perform_optimized_custom_grad_computation(
            custom_input, _min, _max
        )

        # validate gradients computed from autograd engine and optimized custom gradients
        # NOTE: Optimized custom grad follows same computation logic of autograd engine
        # To sanity check, it should be compared whole tensor between two results
        assert torch.allclose(a_input_grad, oc_input_grad)
        assert torch.isclose(a_min_grad, oc_min_grad.data, atol=1e-3)
        assert torch.isclose(a_max_grad, oc_max_grad.data, atol=1e-3)

    def test_custom_gradient_for_range_learning_time_taken(self):
        """
        Unit test to check the time taken by custom gradient computation against auto grad computation
        :return: None
        """

        dtype = torch.float
        device = torch.device("cpu")

        torch.manual_seed(0)
        custom_input = torch.rand((2, 2), requires_grad=True, dtype=dtype, device=device)

        time_taken_by_auto = 0
        iterations = 10
        for i in range(1, iterations):
            start_time = time.perf_counter()
            _ = TestQcQuantizeOpLearnedGrid.perform_auto_grad_computation(custom_input, 0.0015, 1.0)
            exec_time = time.perf_counter() - start_time
            time_taken_by_auto = time_taken_by_auto + exec_time
        auto_average_time = time_taken_by_auto / iterations
        print('Avg time taken by auto grad', auto_average_time)

        # custom gradient computation
        time_taken_by_custom = 0
        torch.manual_seed(0)
        custom_input = torch.rand((2, 2), requires_grad=True, dtype=dtype, device=device)

        for i in range(1, iterations):
            # compute this one time and pass it to forward function
            start_time = time.perf_counter()
            _ = TestQcQuantizeOpLearnedGrid.perform_optimized_custom_grad_computation(custom_input, 0.0015, 1.0)
            exec_time = time.perf_counter() - start_time
            time_taken_by_custom = time_taken_by_custom + exec_time

        custom_avg_time = time_taken_by_custom/iterations

        print('Avg time taken by custom grad', custom_avg_time)
        print('Total % increase is ', ((custom_avg_time-auto_average_time)/auto_average_time) * 100)

    @pytest.mark.skip
    def test_compare_quantize_dequantize_cpp_python(self):
        torch.manual_seed(10)
        random_tensor = torch.rand((2, 3), requires_grad=True)

        encoding_min = torch.nn.Parameter(torch.FloatTensor([-5]))
        encoding_max = torch.nn.Parameter(torch.FloatTensor([5]))
        tensor_quantizer = LearnedGridTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                      quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                      use_symmetric_encodings=False,
                                                      enabled_by_default=True,
                                                      data_type=QuantizationDataType.int)
        out1 = tensor_quantizer.quantize_dequantize(random_tensor, encoding_min, encoding_max)

        post_training_tensor_quantizer = StaticGridPerTensorQuantizer(bitwidth=8, round_mode='nearest',
                                                                      quant_scheme=QuantScheme.post_training_tf,
                                                                      use_symmetric_encodings=False,
                                                                      enabled_by_default=True)

        encodings = libpymo.TfEncoding()
        encodings.bw = 8
        encodings.max = 5
        encodings.min = -5
        encodings.delta = 1
        encodings.offset = 0.2
        post_training_tensor_quantizer.encoding = encodings

        out2 = post_training_tensor_quantizer.quantize_dequantize(random_tensor, MAP_ROUND_MODE_TO_PYMO['nearest'])
        assert np.allclose(out1.detach().numpy(), out2.detach().numpy())

    def test_n_p_computation(self):
        """
        validate n and p values computed for symmetric and asymmetric case.
        :return:
        """
        bitwidth = 8

        # for 8 bit , 0 to 255 if not strict symmetric
        sym_n, sym_p = LearnedGridTensorQuantizer.get_n_and_p(bitwidth, use_symmetric_encoding=True,
                                                              use_strict_symmetric=False, device="cpu")

        expected_sym_n = 0
        expected_sym_p = (2 ** bitwidth) - 1

        comp_symmetric_n = sym_n.data.item()
        comp_symmetric_p = sym_p.data.item()

        assert expected_sym_n == comp_symmetric_n
        assert expected_sym_p == comp_symmetric_p

        # for 8 bit, 0 to 254 if strict symmetric
        strict_sym_n, strict_sym_p = LearnedGridTensorQuantizer.get_n_and_p(bitwidth, use_symmetric_encoding=True,
                                                                            use_strict_symmetric=True, device="cpu")

        expected_strict_sym_n = 0
        expected_strict_sym_p = ((2 ** bitwidth) - 1) - 1

        comp_strict_symmetric_n = strict_sym_n.data[0].item()
        comp_strict_symmetric_p = strict_sym_p.data[0].item()

        assert expected_strict_sym_n == comp_strict_symmetric_n
        assert expected_strict_sym_p == comp_strict_symmetric_p

        # for 8 bit , 0 to 255
        asym_n, asym_p = LearnedGridTensorQuantizer.get_n_and_p(bitwidth, use_symmetric_encoding=False,
                                                                use_strict_symmetric=False, device="cpu")

        expected_asym_n = 0
        expected_asym_p = (2 ** bitwidth) - 1
        comp_asymmetric_n = asym_n.data.item()
        comp_asymmetric_p = asym_p.data.item()

        assert expected_asym_n == comp_asymmetric_n
        assert expected_asym_p == comp_asymmetric_p

        # Should raise exception when SymmetricEncoding=False, UseStrictSymmetric=True
        with pytest.raises(ValueError):
            _, _ = LearnedGridTensorQuantizer.get_n_and_p(bitwidth, use_symmetric_encoding=False, use_strict_symmetric=True, device="cpu")


    def test_ste_gating_for_learnable_grid_wrapper(self):
        torch.manual_seed(0)

        encodings = libpymo.TfEncoding()
        encodings.bw, encodings.max, encodings.min, encodings.delta, encodings.offset = 8, 0.5, -1, 1, 0.2

        encodings_new = libpymo.TfEncoding()
        encodings_new.bw, encodings_new.max, encodings_new.min, encodings_new.delta, encodings_new.offset = 8, 0.4, -0.98, 1, 0.2

        output_grad = []
        def hook_fn(m, _, i):

            for grad in i:
                try:
                    output_grad.append(grad)
                except AttributeError:
                    print("None found for Gradient")

        conv1 = torch.nn.Conv2d(1, 2, 1)
        conv1.weight.data = torch.Tensor([[[[-0.8]]], [[[0.9]]]])
        quantize = LearnedGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                           quant_scheme=QuantScheme.training_range_learning_with_tf_init, device='cpu',
                                           data_type=QuantizationDataType.int)
        quantize.train()
        quantize._module_to_wrap.register_backward_hook(hook_fn)

        quantize.input_quantizers[0].enabled = True
        quantize.output_quantizers[0].enabled = True
        quantize.input_quantizers[0].encoding = encodings
        quantize.output_quantizers[0].encoding = encodings
        quantize.param_quantizers['weight'].encoding = encodings
        quantize.param_quantizers['bias'].enabled = False

        new_input = torch.autograd.Variable(torch.tensor([[[[-0.8469]]], [[[0.9]]]]), requires_grad=True)
        out = quantize(new_input)

        quantize.input_quantizers[0].encoding = encodings_new
        quantize.output_quantizers[0].encoding = encodings_new
        quantize.param_quantizers['weight'].encoding = encodings_new

        loss = out.flatten().sum()
        loss.backward()

        # Check if input gradient got clipped
        for i, val in enumerate(new_input):
            if encodings_new.min > val or val > encodings_new.max:
                assert new_input.grad.flatten()[1] == 0.0

        # Check if output gradient got clipped
        output_grad = output_grad[0].flatten()
        assert output_grad[0] == 1.0
        assert output_grad[1] == 0.0
        assert output_grad[2] == 0.0
        assert output_grad[3] == 1.0

        # Check if weight gradient got clipped
        weight_tensor_grad = quantize._module_to_wrap.weight.grad.flatten()
        assert weight_tensor_grad[1] == 0.0

    def test_wrapper_for_in_place_operation(self):
        """
        Test wrapper for following in-place operation
        """
        module = torch.nn.Conv2d(3, 4, 2)
        wrapper = StaticGridQuantWrapper(module, weight_bw=8, activation_bw=8, round_mode='nearest',
                                         quant_scheme=QuantScheme.post_training_tf_enhanced)
        input_shape = (1, 3, 8, 8)
        input_var = torch.autograd.Variable(torch.randn(*input_shape), requires_grad=True)
        wrapper.set_mode(QcQuantizeOpMode.ANALYSIS)
        output = wrapper.forward(input_var)
        output += output # in-place operation should succeed
        wrapper.compute_encoding()
        wrapper.set_mode(QcQuantizeOpMode.ACTIVE)
        output = wrapper.forward(input_var)
        output += output    # in-place operation should succeed

    def test_set_enabled_for_param_quantizers(self):
        """
        Test set enabled flag for parameter quantizers
        """
        module = torch.nn.Conv2d(3, 4, 2)
        wrapper = StaticGridQuantWrapper(module, weight_bw=8, activation_bw=8, round_mode='nearest',
                                         quant_scheme=QuantScheme.post_training_tf_enhanced)
        # Disable bias quantization
        wrapper.param_quantizers['bias'].enabled = False
        wrapper.enable_param_quantizers(enabled=True)

        assert wrapper.param_quantizers['weight'].enabled == True
        assert wrapper.param_quantizers['bias'].enabled == False

        wrapper.enable_param_quantizers(enabled=False, param_name_to_exclude=("weight",))
        assert wrapper.param_quantizers['weight'].enabled == True
        assert wrapper.param_quantizers['bias'].enabled == False

        # Enable bias quantization
        wrapper.param_quantizers['bias'].enabled = True
        wrapper.enable_param_quantizers(enabled=False, param_name_to_exclude=None)
        assert wrapper.param_quantizers['weight'].enabled == False
        assert wrapper.param_quantizers['bias'].enabled == False

    def test_set_and_freeze_param_encoding_for_static_grid_quant_wrapper(self):
        """ Test set and freeze parameter encoding  """
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quant_wrapper = StaticGridQuantWrapper(conv1, weight_bw=8, activation_bw=8, round_mode='nearest',
                                               quant_scheme=QuantScheme.post_training_tf_enhanced,
                                               data_type=QuantizationDataType.int)

        param_encodings = {'conv1.weight': [{'bitwidth': 4, 'is_symmetric': 'False', 'max': 0.3, 'min': -0.2,
                                             'offset': -7.0, 'scale': 0.038, 'dtype': 'int'}]}

        quant_wrapper.set_param_encoding('conv1', param_encodings)
        quant_wrapper.freeze_param_encoding('conv1', param_encodings)

        assert quant_wrapper.param_quantizers['weight'].encoding.bw == 4
        assert quant_wrapper.param_quantizers['weight'].encoding.offset == -7.0
        assert quant_wrapper.param_quantizers['weight'].encoding.delta == 0.038
        assert not quant_wrapper.param_quantizers['weight'].use_symmetric_encodings
        assert quant_wrapper.param_quantizers['weight'].bitwidth == 4

        # Reset encoding, Since encoding are frozen they should not be None after reset encoding
        quant_wrapper.reset_encodings()

        assert quant_wrapper.param_quantizers['weight'].encoding

    def test_set_and_freeze_param_encoding_for_learned_grid_quant_wrapper(self):
        """ Test set and freeze parameter encoding  """
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quant_wrapper = LearnedGridQuantWrapper(conv1, round_mode='nearest',
                                                quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                is_output_quantized=True, activation_bw=8,
                                                weight_bw=8, device='cpu')

        enc_old = libpymo.TfEncoding()
        enc_old.bw, enc_old.max, enc_old.min, enc_old.delta, enc_old.offset = 8, 0.5, -1, 1, 0.2
        quant_wrapper.param_quantizers['weight'].encoding = enc_old

        param_encodings = {'conv1.weight': [{'bitwidth': 4, 'is_symmetric': 'False', 'max': 0.3, 'min': -0.2,
                                             'offset': -7.0, 'scale': 0.038, 'dtype': 'int'}]}

        quant_wrapper.set_param_encoding('conv1', param_encodings)
        quant_wrapper.freeze_param_encoding('conv1', param_encodings)

        assert quant_wrapper.param_quantizers['weight'].encoding.bw == 4
        assert np.isclose(quant_wrapper.param_quantizers['weight'].encoding.min, -0.2)
        assert np.isclose(quant_wrapper.param_quantizers['weight'].encoding.max, 0.3)
        assert not quant_wrapper.param_quantizers['weight'].use_symmetric_encodings

        # try to set new encoding.
        with pytest.raises(RuntimeError):
            enc_new = libpymo.TfEncoding()
            enc_new.bw, enc_new.max, enc_new.min, enc_new.delta, enc_new.offset = 8, 0.4, -0.98, 1, 0.2
            quant_wrapper.param_quantizers['weight'].encoding = enc_new

        # Once again verify.
        assert quant_wrapper.param_quantizers['weight'].encoding.bw == 4
        assert np.isclose(quant_wrapper.param_quantizers['weight'].encoding.min, -0.2)
        assert np.isclose(quant_wrapper.param_quantizers['weight'].encoding.max, 0.3)
        assert not quant_wrapper.param_quantizers['weight'].use_symmetric_encodings

    def test_static_grid_wrapper_pickle_upickle(self):
        """
        test static grid quant wrapper's freeze_encoding() with pickle and unpickle.
        """
        conv = torch.nn.Conv2d(1, 32, 5)
        quant_wrapper = StaticGridQuantWrapper(conv, weight_bw=8, activation_bw=8, round_mode='nearest',
                                               quant_scheme=QuantScheme.post_training_tf_enhanced)

        enc = libpymo.TfEncoding()
        enc.bw, enc.max, enc.min, enc.delta, enc.offset = 8, 0.5, -1, 0.01, 50

        # Set encoding for all - input, output and parameters quantizer.
        quant_wrapper.input_quantizers[0].enabled = True
        quant_wrapper.input_quantizers[0].encoding = enc
        quant_wrapper.param_quantizers['weight'].enabled = True
        quant_wrapper.param_quantizers['weight'].encoding = enc
        quant_wrapper.param_quantizers['bias'].enabled = True
        quant_wrapper.param_quantizers['bias'].encoding = enc
        quant_wrapper.output_quantizers[0].enabled = True
        quant_wrapper.output_quantizers[0].encoding = enc

        enc_cur = quant_wrapper.output_quantizers[0].encoding
        assert enc_cur.min == enc.min

        # Freeze encoding only for output quantizer.
        quant_wrapper.output_quantizers[0].freeze_encoding()

        # Serialize and De-serialize.
        pickled = pickle.dumps(quant_wrapper)
        loaded_quant_wrapper = pickle.loads(pickled)

        # verify that the state _is_encoding_frozen state is maintained.
        assert loaded_quant_wrapper.output_quantizers[0]._is_encoding_frozen == True
        assert loaded_quant_wrapper.input_quantizers[0]._is_encoding_frozen == False
        assert loaded_quant_wrapper.param_quantizers['weight']._is_encoding_frozen == False
        assert loaded_quant_wrapper.param_quantizers['bias']._is_encoding_frozen == False

        assert loaded_quant_wrapper.param_quantizers['weight'].encoding.max == 0.5
        assert loaded_quant_wrapper.param_quantizers['bias'].encoding.max == 0.5
        assert loaded_quant_wrapper.output_quantizers[0].encoding.max == 0.5
        assert loaded_quant_wrapper.input_quantizers[0].encoding.max == 0.5

        enc_new = libpymo.TfEncoding()
        enc_new.bw, enc_new.max, enc_new.min, enc_new.delta, enc_new.offset = 8, 0.4, -0.98, 1, 0.2

        # try to set new encoding except output quantizer.
        loaded_quant_wrapper.param_quantizers['weight'].encoding = enc_new
        loaded_quant_wrapper.param_quantizers['bias'].encoding = enc_new
        loaded_quant_wrapper.input_quantizers[0].encoding = enc_new
        with pytest.raises(RuntimeError):
            loaded_quant_wrapper.output_quantizers[0].encoding = enc_new

    def test_learned_grid_wrapper_pickle_upickle(self):
        """
        test learned grid quant wrapper's freeze_encoding() with pickle and unpickle.
        """
        conv1 = torch.nn.Conv2d(4, 4, 1)
        quant_wrapper = LearnedGridQuantWrapper(conv1, round_mode='nearest',
                                                quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                is_output_quantized=True, activation_bw=8,
                                                weight_bw=8, device='cpu')

        enc_old = libpymo.TfEncoding()
        enc_old.bw, enc_old.max, enc_old.min, enc_old.delta, enc_old.offset = 8, 0.5, -1, 1, 0.2

        # Set encoding for all - input, output and parameters quantizer.
        quant_wrapper.input_quantizers[0].enabled = True
        quant_wrapper.input_quantizers[0].encoding = enc_old
        quant_wrapper.param_quantizers['weight'].enabled = True
        quant_wrapper.param_quantizers['weight'].encoding = enc_old
        quant_wrapper.param_quantizers['bias'].enabled = True
        quant_wrapper.param_quantizers['bias'].encoding = enc_old
        quant_wrapper.output_quantizers[0].enabled = True
        quant_wrapper.output_quantizers[0].encoding = enc_old

        enc_cur = quant_wrapper.output_quantizers[0].encoding
        assert enc_cur.min == enc_old.min

        # Freeze encoding only for output quantizer.
        quant_wrapper.output_quantizers[0].freeze_encoding()

        # Serialize and De-serialize.
        pickled = pickle.dumps(quant_wrapper)
        loaded_quant_wrapper = pickle.loads(pickled)

        # verify that the state _is_encoding_frozen state is maintained.
        assert loaded_quant_wrapper.output_quantizers[0]._is_encoding_frozen == True
        assert loaded_quant_wrapper.input_quantizers[0]._is_encoding_frozen == False
        assert loaded_quant_wrapper.param_quantizers['weight']._is_encoding_frozen == False
        assert loaded_quant_wrapper.param_quantizers['bias']._is_encoding_frozen == False

        enc_new = libpymo.TfEncoding()
        enc_new.bw, enc_new.max, enc_new.min, enc_new.delta, enc_new.offset = 8, 0.4, -0.98, 1, 0.2

        # try to set new encoding except output quantizer.
        loaded_quant_wrapper.param_quantizers['weight'].encoding = enc_new
        loaded_quant_wrapper.param_quantizers['bias'].encoding = enc_new
        loaded_quant_wrapper.input_quantizers[0].encoding = enc_new
        with pytest.raises(RuntimeError):
            loaded_quant_wrapper.output_quantizers[0].encoding = enc_new

    @pytest.mark.parametrize('act_bw', [8, 16])
    @pytest.mark.cuda
    def test_learned_grid_preserves_fp16(self, act_bw):
        """
        Test if the forward of LearnedGridQuantWrapper preserves the dtype between
        input and output
        """
        torch.manual_seed(2023)

        in_features = 200
        out_features = 100

        input_shape = (4, 200)
        x = torch.randn(input_shape).cuda().half()
        weight = torch.randn((out_features, in_features)).clamp(-1, 1)
        bias = torch.randn(out_features)
        linear = torch.nn.Linear(in_features, out_features).cuda()
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

        linear_wrapper = LearnedGridQuantWrapper(linear,
                                                 weight_bw=4,
                                                 activation_bw=act_bw,
                                                 round_mode='round_nearest',
                                                 quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                                 device='cuda:0')
        param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(linear_wrapper)
        for quantizer in param_quantizers + input_quantizers + output_quantizers:
            quantizer.enabled = False

        linear_wrapper.param_quantizers['weight'].enabled = True
        linear_wrapper.weight_encoding_min = torch.nn.Parameter(torch.tensor([-1.0 for _ in range(out_features)]))
        linear_wrapper.weight_encoding_max = torch.nn.Parameter(torch.tensor([1.0 for _ in range(out_features)]))

        linear_wrapper.input_quantizers[0].enabled = True
        linear_wrapper.input0_encoding_min = torch.nn.Parameter(torch.tensor([-3.5]))
        linear_wrapper.input0_encoding_max = torch.nn.Parameter(torch.tensor([3.5]))

        linear_wrapper.output_quantizers[0].enabled = True
        linear_wrapper.output0_encoding_min = torch.nn.Parameter(torch.tensor([-30.]))
        linear_wrapper.output0_encoding_max = torch.nn.Parameter(torch.tensor([30.]))

        linear_wrapper.cuda().train().half()
        out_16 = linear_wrapper(x.half())
        assert out_16.dtype == torch.float16

        out_16.sum().backward()
        assert linear_wrapper.weight.grad.dtype == torch.float16
        assert linear_wrapper.weight_encoding_min.grad.dtype == torch.float16
        assert linear_wrapper.weight_encoding_max.grad.dtype == torch.float16
        assert linear_wrapper.input0_encoding_min.grad.dtype == torch.float16
        assert linear_wrapper.input0_encoding_max.grad.dtype == torch.float16
        assert linear_wrapper.output0_encoding_min.grad.dtype == torch.float16
        assert linear_wrapper.output0_encoding_max.grad.dtype == torch.float16

        for val in (torch.nan, torch.inf, -torch.inf):
            assert torch.all(linear_wrapper.weight.grad               != val)
            assert torch.all(linear_wrapper.weight_encoding_min.grad  != val)
            assert torch.all(linear_wrapper.weight_encoding_max.grad  != val)
            assert torch.all(linear_wrapper.input0_encoding_min.grad  != val)
            assert torch.all(linear_wrapper.input0_encoding_max.grad  != val)
            assert torch.all(linear_wrapper.output0_encoding_min.grad != val)
            assert torch.all(linear_wrapper.output0_encoding_max.grad != val)

    @pytest.mark.parametrize("wrapper",
                             [StaticGridQuantWrapper(aimet_modules.Addmm(),
                                                     8, 8, 'nearest',
                                                     QuantScheme.post_training_tf_enhanced,
                                                     num_inputs=3, num_outputs=1),
                              LearnedGridQuantWrapper(aimet_modules.Addmm(),
                                                      8, 8, 'nearest',
                                                      QuantScheme.training_range_learning_with_tf_init,
                                                      torch.device('cpu'),
                                                      num_inputs=3, num_outputs=1)
                              ])
    def test_wrapper_with_kwargs(self, wrapper):
        """ test wrapper with keyword arguemnts """
        wrapper.input_quantizers[0].enabled = False
        wrapper.input_quantizers[0].enabled = False
        wrapper.input_quantizers[0].enabled = False
        wrapper.output_quantizers[0].enabled = False
        out1 = wrapper(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3), beta=1, alpha=1)
        out2 = wrapper(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3), beta=1)
        out3 = wrapper(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3))
        assert out1.shape == out2.shape
        assert out2.shape == out3.shape

    @pytest.mark.parametrize("wrapper",
        [StaticGridQuantWrapper(torch.nn.Linear(10, 10),
                                8, 8, 'nearest',
                                QuantScheme.post_training_tf_enhanced,
                                num_inputs=2, num_outputs=2),
         LearnedGridQuantWrapper(torch.nn.Linear(10, 10),
                                 8, 8, 'nearest',
                                 QuantScheme.training_range_learning_with_tf_init,
                                 torch.device('cpu'),
                                 num_inputs=2, num_outputs=2)
         ])
    def test_export_quantizer_encodings(self, wrapper):
        wrapper.input_quantizers[0].enabled = True
        wrapper.input_quantizers[1].enabled = False
        wrapper.output_quantizers[0].enabled = False
        wrapper.output_quantizers[1].enabled = True
        wrapper.param_quantizers["weight"].enabled = True
        wrapper.param_quantizers["bias"].enabled = False
        encoding = libpymo.TfEncoding()
        encoding.max = 127.0
        encoding.min = -128.0
        encoding.delta = 1.0
        encoding.offset = -128
        encoding.bw = 8
        encoding_as_dict = {"min": encoding.min, "max": encoding.max, "scale": encoding.delta, "offset": int(encoding.offset),
                            "bitwidth": encoding.bw, "is_symmetric": "False", "dtype": "int"}
        wrapper.enable_per_channel_quantization()
        wrapper.param_quantizers["weight"].encoding = [encoding] * 10
        wrapper.input_quantizers[0].encoding = encoding
        wrapper.output_quantizers[1].encoding = encoding
        input_encodings = wrapper.export_input_encodings(encoding_version='0.6.1')
        output_encodings = wrapper.export_output_encodings(encoding_version='0.6.1')
        param_encodings = wrapper.export_param_encodings(encoding_version='0.6.1')
        assert len(input_encodings) == 2
        assert input_encodings[0] == [encoding_as_dict]
        assert input_encodings[1] is None
        assert len(output_encodings) == 2
        assert output_encodings[0] is None
        assert output_encodings[1] == [encoding_as_dict]
        assert len(param_encodings.items()) == 2
        assert param_encodings["bias"] is None
        assert param_encodings["weight"] == [encoding_as_dict] * 10

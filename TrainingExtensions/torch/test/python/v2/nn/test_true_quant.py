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

import functools
import itertools
from packaging import version

import pytest
import torch
from torch import randn, randint, zeros, full, arange, ones, tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten
from torch.overrides import get_ignored_functions
from aimet_torch.v2.quantization.affine.backends import quantize, quantize_dequantize, dequantize
from aimet_torch.v2.quantization.affine import Quantize, QuantizeDequantize
import aimet_torch.v2 as aimet
from aimet_torch.v2.nn import (
    QuantizedConv1d,
    QuantizedConv2d,
    QuantizedConv3d,
    QuantizedConvTranspose1d,
    QuantizedConvTranspose2d,
    QuantizedConvTranspose3d,
    QuantizedGELU,
    QuantizedLinear,
    QuantizationMixin,
    QuantizedSigmoid,
    QuantizedSoftmax,
    QuantizedLayerNorm,
    QuantizedGroupNorm,
)
from aimet_torch.v2.nn.fake_quant import _legacy_impl
from aimet_torch.v2.nn.true_quant import _dispatch, _dispatch_table
from aimet_torch.v2.quantization.affine import AffineEncoding
from aimet_torch.v2.quantization.tensor import QuantizedTensor, DequantizedTensor
from aimet_torch.v2.utils import enable_recompute
from aimet_torch.v2.nn import custom


@pytest.fixture(autouse=True)
def manual_seed():
    torch.manual_seed(724)


def affine_quantize(tensor: torch.Tensor,
                    scale: torch.Tensor,
                    offset: torch.Tensor,
                    bitwidth: int) -> QuantizedTensor:
    """
    Quantizes the input tensor into a QuantizedTensor using the quantization parameters
    """
    tensor_q = quantize(tensor, scale, offset, bitwidth)
    encoding = AffineEncoding(scale, offset, bitwidth)
    qtensor = tensor_q.as_subclass(QuantizedTensor)
    qtensor.encoding = encoding
    return qtensor


def _input(*shape):
    numel = functools.reduce(lambda x, y: x * y, shape)
    return torch.arange(1,numel+1).view(*shape) / numel


@pytest.fixture
def input():
    return _input(10, 10)


@pytest.fixture
def register_int_linear():
    def int_linear(input, weight, bias=None, *, output_encodings=None):
        # Implicit dequantization is not supported yet
        if not isinstance(input, QuantizedTensor):
            raise RuntimeError
        if not isinstance(weight, QuantizedTensor):
            raise RuntimeError

        input = input.dequantize()
        weight = weight.dequantize()

        return affine_quantize(input.mm(weight.t()) + bias,
                               output_encodings.scale,
                               output_encodings.offset,
                               output_encodings.bitwidth)

    QuantizedLinear.set_default_kernel(int_linear)
    yield
    QuantizedLinear.set_default_kernel(None)


@pytest.fixture
def register_int_conv():
    def int_convnd(kernel, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, output_encodings=None):
        # Implicit dequantization is not supported yet
        if not isinstance(input, QuantizedTensor):
            raise RuntimeError
        if not isinstance(weight, QuantizedTensor):
            raise RuntimeError

        input = input.dequantize()
        weight = weight.dequantize()
        output = kernel(input, weight, bias, stride, padding, dilation, groups)
        return affine_quantize(output,
                               output_encodings.scale,
                               output_encodings.offset,
                               output_encodings.bitwidth)

    QuantizedConv1d.set_default_kernel(functools.partial(int_convnd, F.conv1d))
    QuantizedConv2d.set_default_kernel(functools.partial(int_convnd, F.conv2d))
    QuantizedConv3d.set_default_kernel(functools.partial(int_convnd, F.conv3d))
    yield
    QuantizedConv3d.set_default_kernel(None)
    QuantizedConv2d.set_default_kernel(None)
    QuantizedConv1d.set_default_kernel(None)


@pytest.fixture
def register_int_convtranspose():
    def int_convtransposend(kernel, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, *, output_encodings=None):
        # Implicit dequantization is not supported yet
        if not isinstance(input, QuantizedTensor):
            raise RuntimeError
        if not isinstance(weight, QuantizedTensor):
            raise RuntimeError

        input = input.dequantize()
        weight = weight.dequantize()
        output = kernel(input, weight, bias, stride, padding, output_padding, groups, dilation)
        return affine_quantize(output,
                               output_encodings.scale,
                               output_encodings.offset,
                               output_encodings.bitwidth)

    QuantizedConvTranspose1d.set_default_kernel(functools.partial(int_convtransposend, F.conv_transpose1d))
    QuantizedConvTranspose2d.set_default_kernel(functools.partial(int_convtransposend, F.conv_transpose2d))
    QuantizedConvTranspose3d.set_default_kernel(functools.partial(int_convtransposend, F.conv_transpose3d))
    yield
    QuantizedConvTranspose1d.set_default_kernel(None)
    QuantizedConvTranspose2d.set_default_kernel(None)
    QuantizedConvTranspose3d.set_default_kernel(None)


@pytest.fixture
def register_int_activation():
    def wrap_functional(func):
        def wrapped_func(*args, output_encodings=None, **kwargs):
            # Implicit dequantization is not supported yet
            x, *others = args
            if not isinstance(x, QuantizedTensor):
                raise RuntimeError
            output = func(x.dequantize(), *others, **kwargs)
            return affine_quantize(output, output_encodings.scale, output_encodings.offset, output_encodings.bitwidth)

        return wrapped_func

    QuantizedSoftmax.set_default_kernel(wrap_functional(F.softmax))
    QuantizedSigmoid.set_default_kernel(wrap_functional(torch.sigmoid))
    QuantizedGELU.set_default_kernel(wrap_functional(F.gelu))
    yield
    QuantizedGELU.set_default_kernel(None)
    QuantizedSigmoid.set_default_kernel(None)
    QuantizedSoftmax.set_default_kernel(None)


@pytest.fixture
def register_int_norm():
    def wrap_functional(func):
        def int_norm(input, normalized_shape, weight, bias, eps, *, output_encodings=None):
            # Implicit dequantization is not supported yet
            if not isinstance(input, QuantizedTensor):
                raise RuntimeError
            if not isinstance(weight, QuantizedTensor):
                raise RuntimeError

            input = input.dequantize()
            weight = weight.dequantize()

            output = func(input, normalized_shape, weight, bias, eps)
            return affine_quantize(output, output_encodings.scale, output_encodings.offset, output_encodings.bitwidth)

        return int_norm

    QuantizedLayerNorm.set_default_kernel(wrap_functional(F.layer_norm))
    QuantizedGroupNorm.set_default_kernel(wrap_functional(F.group_norm))
    yield
    QuantizedGroupNorm.set_default_kernel(None)
    QuantizedLayerNorm.set_default_kernel(None)


@pytest.fixture
def register_int_custom():
    def int_elementwise(kernel, x, y, *, output_encodings=None):
        # Implicit dequantization is not supported yet
        if not isinstance(x, QuantizedTensor):
            raise RuntimeError
        if not isinstance(y, QuantizedTensor):
            raise RuntimeError
        output = kernel(x.dequantize(), y.dequantize())
        return affine_quantize(output, output_encodings.scale, output_encodings.offset, output_encodings.bitwidth)

    custom.QuantizedAdd.set_default_kernel(functools.partial(int_elementwise, torch.add))
    custom.QuantizedMultiply.set_default_kernel(functools.partial(int_elementwise, torch.multiply))
    custom.QuantizedSubtract.set_default_kernel(functools.partial(int_elementwise, torch.subtract))
    custom.QuantizedDivide.set_default_kernel(functools.partial(int_elementwise, torch.div))
    custom.QuantizedMatMul.set_default_kernel(functools.partial(int_elementwise, torch.matmul))
    yield
    custom.QuantizedMultiply.set_default_kernel(None)
    custom.QuantizedSubtract.set_default_kernel(None)
    custom.QuantizedAdd.set_default_kernel(None)
    custom.QuantizedDivide.set_default_kernel(None)
    custom.QuantizedMatMul.set_default_kernel(None)



class TestTrueQuantLinear:
    @pytest.mark.usefixtures('register_int_linear')
    def test_no_quantizers(self, input):
        """
        Given: TrueQuantLinear with no input, output, or param quantizers
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        """
        When: inspect input/output/param quantizers
        Then: quantizers are None
        """
        assert quant_linear.input_quantizers[0] is None
        assert quant_linear.output_quantizers[0] is None
        assert quant_linear.param_quantizers["weight"] is None
        assert quant_linear.param_quantizers["bias"] is None
        """
        When: call forward pass within compute encodings context
        Then: output is equal to floating point output
        """
        expected_output = F.linear(input, quant_linear.weight, quant_linear.bias)
        with quant_linear.compute_encodings():
            output = quant_linear(input)
        assert torch.all(output == expected_output)
        """
        When: call forward pass outside of compute encodings context
        Then: raise RuntimeError
        """
        with pytest.raises(RuntimeError):
            quant_linear(input)

    @pytest.mark.usefixtures('register_int_linear')
    def test_fully_specified_quantizers(self, input):
        """
        Given: TrueQuantLinear with input, output, and param quantizers
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        quant_linear.input_quantizers[0] = Quantize((1, ), bitwidth=8, symmetric=False)
        quant_linear.output_quantizers[0] = Quantize((1, ), bitwidth=8, symmetric=False)
        quant_linear.param_quantizers["weight"] = Quantize((10, ), bitwidth=8, symmetric=True)
        """
        When: Call forward pass before computing encodings
        Then: raise RuntimeError
        """
        with pytest.raises(RuntimeError):
            quant_linear(input)

        """
        When: Invoke forward pass within compute_encodings context
        Then: Output should be equal to fake quant forward pass with activation quantizers disabled
        """
        with quant_linear.compute_encodings():
            output = quant_linear(input)

        input_enc = (quant_linear.input_quantizers[0].get_scale(),
                     quant_linear.input_quantizers[0].get_offset(),
                     quant_linear.input_quantizers[0].bitwidth)
        output_enc = (quant_linear.output_quantizers[0].get_scale(),
                      quant_linear.output_quantizers[0].get_offset(),
                      quant_linear.output_quantizers[0].bitwidth)
        weight_enc = (quant_linear.param_quantizers["weight"].get_scale(),
                      quant_linear.param_quantizers["weight"].get_offset(),
                      quant_linear.param_quantizers["weight"].bitwidth)
        weight_qdq = quantize_dequantize(quant_linear.weight, *weight_enc, signed=True)
        output_expected = F.linear(input, weight_qdq, bias=quant_linear.bias)
        assert torch.equal(output, output_expected)

        """
        When: Invoke forward pass outside of compute_encodings context with an unquantized tensor
        Then: 1) output should be computed using the global true quant backend
              2) output should be a quantized tensor
              3) output should be close to fake quant output after dequantization
        """
        input_qdq = quantize_dequantize(input, *input_enc)
        output_fp = F.linear(input_qdq, weight_qdq, bias=quant_linear.bias)
        output_expected = quantize_dequantize(output_fp, *output_enc)
        output_quant = quant_linear(input)
        assert isinstance(output_quant, DequantizedTensor)
        assert torch.allclose(output_quant.dequantize(), output_expected)

        """
        When: Invoke forward pass outside of compute_encodings context with a quantized tensor
        Then: Dequantized output should be close to running fake quant on the dequantized input tensor
        """
        quantized_input = affine_quantize(input, *input_enc)
        output = quant_linear(quantized_input)
        input_qdq = dequantize(quantized_input, *input_enc[:2])
        output_fp = F.linear(input_qdq, weight_qdq, bias=quant_linear.bias)
        output_expected = quantize_dequantize(output_fp, *output_enc)
        assert torch.allclose(output.dequantize(), output_expected)

    @pytest.mark.usefixtures('register_int_linear')
    def test_no_input_quantizer(self, input):
        """
        Given: TrueQuantLinear with output and param quantizers and computed encodings
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        quant_linear.output_quantizers[0] = Quantize((1, ), bitwidth=8, symmetric=False)
        quant_linear.param_quantizers["weight"] = Quantize((10, ), bitwidth=8, symmetric=True)
        with quant_linear.compute_encodings():
            quant_linear(input)
        """
        When: Invoke forward pass outside of compute_encodings with an unquantized tensor
        Then: raise RuntimeError
        """
        with pytest.raises(RuntimeError):
            quant_linear(input)

        """
        When: Invoke forward pass with a quantized tensor
        Then: return a tensor quantized with quant_linear.output_quantizer[0].encoding
        """
        quantizer = Quantize((1, ), bitwidth=8, symmetric=False)
        with quantizer.compute_encodings():
            quantizer(input)

        input_q = quantizer(input)
        output = quant_linear(input_q)
        assert isinstance(output, DequantizedTensor)
        assert output.encoding.scale == quant_linear.output_quantizers[0].get_scale()
        assert output.encoding.offset == quant_linear.output_quantizers[0].get_offset()


    @pytest.mark.usefixtures('register_int_linear')
    def test_from_module(self, input):
        # Analogous to FakeQuantMixin.from_module test case
        """
        Given: Instantiate a true-quantized module using `TrueQuantMixin.from_module` and compute_encodings
        When: Inspect {input, output, param}_quantizers, they are the correct length
        """
        fp_linear = torch.nn.Linear(10, input.shape[-1])
        quant_linear = QuantizationMixin.from_module(fp_linear)

        assert len(quant_linear.input_quantizers) == 1
        assert len(quant_linear.output_quantizers) == 1
        assert len(quant_linear.param_quantizers) == 2

        """
        When: Inspect the parameters of the TrueQuant layer
        Then: They are identical to the parameters of the original layer
        """
        assert fp_linear.weight is quant_linear.weight
        assert fp_linear.bias is quant_linear.bias

        """
        When: Update to the parameter/buffer of the base FP module (or its submodule) using in-place operators.
              For example,
                1) fp_module.{param_or_buffer_name}.add_(1)
                2) fp_module.{submodule_name}.{param_or_buffer_name}.add_(1)
        Then: The result of in-place operation affects the parameters/buffers of the quantized module.
              In other words, the parameters/buffers of the quantized module will have been incremented by 1.
        """
        with torch.no_grad():
            fp_linear.weight.add_(1)
        assert torch.equal(fp_linear.weight, quant_linear.weight)
        with quant_linear.compute_encodings():
            quant_linear(input)

        """
        When: Reassign a new submodule/parameter/buffer to the base FP module using assignment stmt.
              For example,
                1) fp_module.{submodule_name} = torch.nn.Linear(...)
                2) fp_module.{param_or_buffer_name} = torch.empty(...)
        Then: The reassignment shouldn't affect the quantized module derived from the FP module.
              The vice versa should also hold.
        """
        fp_linear.weight = torch.nn.Parameter(torch.zeros(10, 10))
        assert not torch.all(fp_linear.weight == quant_linear.weight)


class TestQuantizedLayers:
    @pytest.mark.usefixtures('register_int_norm', 'register_int_custom', 'register_int_activation')
    @pytest.mark.parametrize(
        "module_factory,               input_factory", [
        (lambda: nn.Softmax(dim=1),    lambda: _input(10, 10)),
        (lambda: nn.Sigmoid(),         lambda: _input(10, 10)),
        (lambda: nn.GELU(),            lambda: _input(10, 10)),
        (lambda: custom.Add(),         lambda: (_input(10, 10), _input(10, 10))),
        (lambda: custom.Multiply(),    lambda: (_input(10, 10), _input(10, 10))),
        (lambda: custom.Subtract(),    lambda: (_input(10, 10), _input(10, 10))),
        (lambda: custom.MatMul(),      lambda: (_input(10, 10), _input(10, 10))),
        (lambda: custom.Divide(),      lambda: (_input(10, 10), _input(10, 10)))]
    )
    def test_layers_no_params(self, module_factory, input_factory):
        layer = module_factory()
        inputs = input_factory()

        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)

        fq_layer = _legacy_impl.FakeQuantizationMixin.from_module(layer)
        tq_layer = QuantizationMixin.from_module(layer)
        for i, _ in enumerate(inputs):
            fq_layer.input_quantizers[i] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
            tq_layer.input_quantizers[i] = Quantize(shape=(), bitwidth=8, symmetric=False)

        fq_layer.output_quantizers[0] = QuantizeDequantize(shape=(1, ), bitwidth=8, symmetric=False)
        tq_layer.output_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)

        with fq_layer.compute_encodings():
            fq_layer(*inputs)

        fq_output = fq_layer(*inputs)

        with tq_layer.compute_encodings():
            tq_layer(*inputs)
        tq_output = tq_layer(*inputs)

        assert torch.allclose(fq_output, tq_output.dequantize())

    @pytest.mark.usefixtures('register_int_linear', 'register_int_norm', 'register_int_custom', 'register_int_activation',
                             'register_int_conv', 'register_int_convtranspose')
    @pytest.mark.parametrize(
        "module_factory,                      input_factory", [
        (lambda: nn.Linear(10, 10),           lambda: _input(10, 10)),
        (lambda: nn.LayerNorm(10),            lambda: _input(10, 10)),
        (lambda: nn.GroupNorm(2, 10),         lambda: _input(10, 10)),
        (lambda: nn.Conv1d(3, 3, 3),          lambda: _input(1, 3, 10)),
        (lambda: nn.Conv2d(3, 3, 3),          lambda: _input(1, 3, 10, 10)),
        (lambda: nn.Conv3d(3, 3, 3),          lambda: _input(1, 3, 10, 10, 10)),
        (lambda: nn.ConvTranspose1d(3, 3, 3), lambda: _input(1, 3, 10)),
        (lambda: nn.ConvTranspose2d(3, 3, 3), lambda: _input(1, 3, 10, 10)),
        (lambda: nn.ConvTranspose3d(3, 3, 3), lambda: _input(1, 3, 10, 10, 10))
    ])
    def test_layers_with_weight(self, module_factory, input_factory):
        layer = module_factory()
        input = input_factory()

        fq_layer = _legacy_impl.FakeQuantizationMixin.from_module(layer)
        tq_layer = QuantizationMixin.from_module(layer)
        fq_layer.input_quantizers[0] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
        fq_layer.output_quantizers[0] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=False)
        fq_layer.param_quantizers["weight"] = QuantizeDequantize(shape=(), bitwidth=8, symmetric=True)
        tq_layer.input_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        tq_layer.output_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        tq_layer.param_quantizers["weight"] = Quantize(shape=(), bitwidth=8, symmetric=True)

        with fq_layer.compute_encodings():
            fq_layer(input)

        fq_output = fq_layer(input)


        with tq_layer.compute_encodings():
            tq_layer(input)
        tq_output = tq_layer(input)

        assert torch.allclose(fq_output, tq_output.dequantize())

    @pytest.mark.cuda
    @pytest.mark.usefixtures('register_int_linear')
    def test_layers_with_recompute(self):
        qlinear = QuantizedLinear(4096, 4096)
        qlinear.input_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        qlinear.output_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        qlinear.param_quantizers["weight"] = Quantize(shape=(), bitwidth=8, symmetric=True)
        qlinear.cuda()

        # Using dummy backend is no good for testing memory saving in real life.
        # Set kernel to None so as to use FakeQuantizedLinear under the hood.
        qlinear.set_kernel(None)

        x = torch.randn((100, 4096), device="cuda:0")

        with qlinear.compute_encodings():
            qlinear(x)

        torch.cuda.empty_cache()
        with enable_recompute():
            out = qlinear(x)
        torch.cuda.synchronize()
        mem_with_recompute = torch.cuda.memory_allocated()

        out.backward(torch.ones_like(out))
        grads_with_recompute = [param.grad.clone().detach().cpu() for param in qlinear.parameters()]
        for param in qlinear.parameters():
            param.grad = None

        del out

        torch.cuda.empty_cache()
        out = qlinear(x)
        torch.cuda.synchronize()
        mem_without_recompute = torch.cuda.memory_allocated()

        out.backward(torch.ones_like(out))
        grads_without_recompute = [param.grad.clone().detach().cpu() for param in qlinear.parameters()]
        for param in qlinear.parameters():
            param.grad = None

        # Expected memory saving:
        #   - Input quantizer save:
        #      - mask of shape [100, 4096] * 1 byte
        #      - quantized uint8 tensor of shape [100, 4096] * 1 byte
        #   - Weight quantizer saves:
        #      - mask of shape [4096, 4096] * 1 byte
        #      - quantized uint8 tensor of shape [4096, 4096] * 1 byte
        #   - F.linear saves:
        #      - quantized weight of shape [4096, 4096] * 4 bytes
        #      - quantized input of shape [100, 4096] * 4 bytes
        #   - Output quantizer saves:
        #      - linear output of shape [100, 4096] * 4 bytes
        #      - mask of shape [100, 4096] * 1 byte
        #      - quantized uint8 tensor of shape [100, 4096] * 1 byte
        expected_memory_saving = 0
        expected_memory_saving += (1 + 1) * x.numel() # input quantizer
        expected_memory_saving += (1 + 1) * qlinear.weight.numel() # weight quantizer
        expected_memory_saving += 4 * (qlinear.weight.numel() + x.numel()) # F.linear
        expected_memory_saving += (4 + 1 + 1) * out.numel() # output quantizer
        actual_memory_saving = mem_without_recompute - mem_with_recompute

        # Considering noise factors, actual memory saving should be no less than
        # 90% of the expected memory saving
        assert expected_memory_saving * 0.9 <= actual_memory_saving

        for grad_0, grad_1 in zip(grads_with_recompute, grads_without_recompute):
            assert torch.equal(grad_0, grad_1)

    def test_remove_quantizers(self, input):
        qlinear = QuantizedLinear(10, 10, bias=False)
        qlinear.input_quantizers[0] = input_qtzr = Quantize(shape=(), bitwidth=8, symmetric=False)
        qlinear.output_quantizers[0] = output_qtzr = Quantize(shape=(), bitwidth=8, symmetric=False)
        qlinear.param_quantizers["weight"] = weight_qtzr = Quantize(shape=(), bitwidth=8, symmetric=True)
        with qlinear.compute_encodings():
            qlinear(input)

        """
        When: ``with _remove_{input, param, output, activation, all}_quantizers``
        Then:
            1) The corresponding quantizers are set to None under the context.
               (Output should be computed without input, param, and output quantization respectively)
            2) The corresponding quantizers are restored when exiting the context.
        """
        with qlinear._remove_input_quantizers(0):
            assert qlinear.input_quantizers[0] is None
            expected_out = output_qtzr(
                F.linear(
                    input, weight_qtzr(qlinear.weight).dequantize()
                )
            ).dequantize()
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.input_quantizers[0] is input_qtzr

        with qlinear._remove_param_quantizers('weight'):
            assert qlinear.param_quantizers['weight'] is None
            expected_out = output_qtzr(
                F.linear(
                    input_qtzr(input).dequantize(), qlinear.weight
                )
            ).dequantize()
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.param_quantizers['weight'] is weight_qtzr

        with qlinear._remove_output_quantizers(0):
            assert qlinear.output_quantizers[0] is None
            expected_out = F.linear(input_qtzr(input).dequantize(),
                                    weight_qtzr(qlinear.weight).dequantize())
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.output_quantizers[0] is output_qtzr

        with qlinear._remove_activation_quantizers():
            assert qlinear.input_quantizers[0] is None
            assert qlinear.output_quantizers[0] is None
            expected_out = F.linear(input, weight_qtzr(qlinear.weight).dequantize())
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.input_quantizers[0] is input_qtzr
        assert qlinear.output_quantizers[0] is output_qtzr

        with qlinear._remove_all_quantizers():
            assert qlinear.input_quantizers[0] is None
            assert qlinear.output_quantizers[0] is None
            assert qlinear.param_quantizers['weight'] is None
            expected_out = F.linear(input, qlinear.weight)
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.input_quantizers[0] is input_qtzr
        assert qlinear.output_quantizers[0] is output_qtzr
        assert qlinear.param_quantizers['weight'] is weight_qtzr

        """
        When: Call ``_remove_{input, param, output}_quantizers`` without ``with`` statement
        Then: The corresponding quantizers are set to None permanently
        """
        qlinear._remove_input_quantizers(0)
        assert qlinear.input_quantizers[0] is None
        qlinear._remove_param_quantizers('weight')
        assert qlinear.param_quantizers['weight'] is None
        qlinear._remove_output_quantizers(0)
        assert qlinear.output_quantizers[0] is None


def test_dispatch_sanity():
    """
    Given: custom_add(x, y) := x + y + 1
    """
    custom_add = lambda *args, **kwargs: torch.add(*args, **kwargs) + 1

    """
    When: Dispatch custom_add in place of torch.add(x, y)
    Then: Output of torch.add(x, y) should be equal to x + y + 1
    """
    zeros = torch.zeros(10)
    with _dispatch(torch.add, custom_add):
        out = torch.add(zeros, zeros)
    assert torch.all(out == 1)

    with _dispatch(torch.Tensor.add, custom_add):
        out = zeros + zeros
    assert torch.all(out == 1)

    """
    When: Dispatch custom_add in place of torch.add
    Then: Output of the other functions should not be affected
    """
    with _dispatch(torch.add, custom_add):
        zeros = torch.zeros(10)
        ones = torch.ones(10)
        twos = ones * 2
        fours = twos.square()
        threes = fours - twos / 2

    assert torch.all(zeros == 0)
    assert torch.all(ones == 1)
    assert torch.all(twos == 2)
    assert torch.all(threes == 3)
    assert torch.all(fours == 4)

    """
    When: Try to dispatch unsupported functions
    Then: Throw runtime error
    """
    for func in get_ignored_functions() - _dispatch_table.keys():
        dummy_impl = lambda *args, **kwargs: func(*args, **kwargs)
        with pytest.raises(RuntimeError):
            with _dispatch(func, dummy_impl): pass

    """
    When: Dispatch custom_addmm in place of torch.addmm in which
          custom_add will be dispatched in place of torch.add in a nested fashion
    Then: Output of torch.addmm(x, y, z) should be equal to x + (y @ z) + 1
    """
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    z = torch.randn(10, 10)

    def custom_addmm(x, y, z):
        with _dispatch(torch.add, custom_add):
            return torch.add(x, torch.matmul(y, z))

    with _dispatch(torch.addmm, custom_addmm):
        out = torch.addmm(x, y, z)

    expected = x + (y @ z) + 1
    assert torch.all(out == expected)


def _create_legacy_fake_quantized_module(module):
    qmodule = _legacy_impl.FakeQuantizationMixin.from_module(module)

    for i, _ in enumerate(qmodule.input_quantizers):
        qmodule.input_quantizers[i] = QuantizeDequantize([], 8, False)

    for i, _ in enumerate(qmodule.output_quantizers):
        qmodule.output_quantizers[i] = QuantizeDequantize([], 8, False)

    for name, _ in qmodule.param_quantizers.items():
        qmodule.param_quantizers[name] = QuantizeDequantize([], 8, True)

    return qmodule


def _create_quantized_module(module):
    qmodule = aimet.nn.QuantizationMixin.from_module(module)

    for i, _ in enumerate(qmodule.input_quantizers):
        qmodule.input_quantizers[i] = QuantizeDequantize([], 8, False)

    for i, _ in enumerate(qmodule.output_quantizers):
        qmodule.output_quantizers[i] = QuantizeDequantize([], 8, False)

    for name, _ in qmodule.param_quantizers.items():
        qmodule.param_quantizers[name] = QuantizeDequantize([], 8, True)

    return qmodule


@pytest.mark.parametrize(
    "module_factory,                                  input_factory", itertools.chain([
    (lambda: nn.AdaptiveAvgPool1d(2),                 lambda: randn(1, 100)),
    (lambda: nn.AdaptiveAvgPool2d(2),                 lambda: randn(1, 10, 10)),
    (lambda: nn.AdaptiveAvgPool3d(2),                 lambda: randn(1, 10, 10, 11)),
    # (lambda: nn.AdaptiveLogSoftmaxWithLoss(...),    lambda: ...),
    (lambda: nn.AdaptiveMaxPool1d(2),                 lambda: randn(1, 100)),
    (lambda: nn.AdaptiveMaxPool2d(2),                 lambda: randn(1, 10, 10)),
    (lambda: nn.AdaptiveMaxPool3d(2),                 lambda: randn(1, 10, 10, 11)),
    (lambda: nn.AlphaDropout(),                       lambda: randn(100)),
    (lambda: nn.AvgPool1d(2),                         lambda: randn(1, 100)),
    (lambda: nn.AvgPool2d(2),                         lambda: randn(1, 10, 10)),
    (lambda: nn.AvgPool3d(2),                         lambda: randn(1, 10, 10, 11)),
    (lambda: nn.BCELoss(),                            lambda: (F.sigmoid(randn(100)), zeros(100))),
    (lambda: nn.BCEWithLogitsLoss(),                  lambda: (randn(100), zeros(100))),
    (lambda: nn.BatchNorm1d(10),                      lambda: randn(5, 10, 3)),
    (lambda: nn.BatchNorm2d(10),                      lambda: randn(5, 10, 3, 2)),
    (lambda: nn.BatchNorm3d(10),                      lambda: randn(5, 10, 3, 2, 1)),
    (lambda: nn.Bilinear(20, 30, 40),                 lambda: (randn(10, 20), randn(10, 30))),
    (lambda: nn.CELU(),                               lambda: randn(10, 10)),
    (lambda: nn.CTCLoss(),                            lambda: (randn(10, 11, 12).log_softmax(2),
                                                               randint(low=1, high=12, size=(11, 20)),
                                                               full(size=(11,), fill_value=10, dtype=torch.long),
                                                               randint(low=5, high=20, size=(11,), dtype=torch.long))),
    (lambda: nn.ChannelShuffle(2),                    lambda: randn(1, 8, 4, 4)),
    (lambda: nn.ConstantPad1d(2, 3.5),                lambda: randn(1, 10, 10)),
    (lambda: nn.ConstantPad2d(2, 3.5),                lambda: randn(1, 10, 10)),
    (lambda: nn.ConstantPad3d(2, 3.5),                lambda: randn(1, 10, 2, 5)),
    # (lambda: nn.Container(...),                     lambda: ...),
    (lambda: nn.Conv1d(3, 3, 3),                      lambda: randn(1, 3, 32)),
    (lambda: nn.Conv2d(3, 3, 3),                      lambda: randn(1, 3, 16, 16)),
    (lambda: nn.Conv3d(3, 3, 3),                      lambda: randn(1, 3, 16, 16, 16)),
    (lambda: nn.ConvTranspose1d(3, 3, 3),             lambda: randn(1, 3, 32)),
    (lambda: nn.ConvTranspose2d(3, 3, 3),             lambda: randn(1, 3, 16, 16)),
    (lambda: nn.ConvTranspose3d(3, 3, 3),             lambda: randn(1, 3, 16, 16, 16)),
    (lambda: nn.CosineEmbeddingLoss(),                lambda: (randn(10, 10),
                                                               zeros(10, 10),
                                                               randn(10).sign().long())),
    (lambda: nn.CosineSimilarity(),                   lambda: (randn(10, 10), zeros(10, 10))),
    (lambda: nn.CrossEntropyLoss(),                   lambda: (randn(10, 10), zeros(10, 10))),
    # (lambda: nn.CrossMapLRN2d(...),                 lambda: ...),
    (lambda: nn.Dropout(),                            lambda: randn(10, 10)),
    (lambda: nn.Dropout2d(),                          lambda: randn(10, 10)),
    (lambda: nn.Dropout3d(),                          lambda: randn(10, 10)),
    (lambda: nn.ELU(),                                lambda: randn(10, 10)),
    (lambda: nn.Embedding(100, 100),                  lambda: randint(100, (10,))),
    (lambda: nn.EmbeddingBag(100, 100, mode='sum'),   lambda: (randint(100, (10,)), arange(10), randn(10))),
    (lambda: nn.FeatureAlphaDropout(),                lambda: randn(10, 10)),
    (lambda: nn.Flatten(),                            lambda: randn(10, 10)),
    (lambda: nn.Fold((4, 5), (2, 2)),                 lambda: randn(1, 12, 12)),
    (lambda: nn.FractionalMaxPool2d(3, (5, 5)),       lambda: randn(1, 10, 10)),
    (lambda: nn.FractionalMaxPool3d(3, (5, 5, 5)),    lambda: randn(1, 10, 10, 10)),
    (lambda: nn.GELU(),                               lambda: randn(100)),
    (lambda: nn.GLU(),                                lambda: randn(100)),
    (lambda: nn.GRU(10, 20, 2),                       lambda: (randn(5, 3, 10), randn(2, 3, 20))),
    (lambda: nn.GRUCell(10, 20),                      lambda: (randn(3, 10), randn(3, 20))),
    (lambda: nn.GaussianNLLLoss(),                    lambda: (randn(1, 100), zeros(1, 100), ones(1, 100))),
    (lambda: nn.GroupNorm(2, 4),                      lambda: randn(1, 4, 25)),
    (lambda: nn.Hardshrink(0),                        lambda: randn(100)),
    (lambda: nn.Hardsigmoid(),                        lambda: randn(100)),
    (lambda: nn.Hardswish(),                          lambda: randn(100)),
    (lambda: nn.Hardtanh(),                           lambda: randn(100)),
    (lambda: nn.HingeEmbeddingLoss(),                 lambda: (randn(10, 10), randn(10).sign().long())),
    (lambda: nn.HuberLoss(),                          lambda: (randn(10, 10), zeros(10, 10))),
    # (lambda: nn.Identity(...),                      lambda: ...),
    (lambda: nn.InstanceNorm1d(10),                   lambda: randn(5, 10, 3)),
    (lambda: nn.InstanceNorm2d(10),                   lambda: randn(5, 10, 3, 2)),
    (lambda: nn.InstanceNorm3d(10),                   lambda: randn(5, 10, 3, 2, 1)),
    (lambda: nn.KLDivLoss(reduction='batchmean'),     lambda: (F.log_softmax(randn(10, 10), dim=1),
                                                               F.softmax(randn(10, 10), dim=1))),
    (lambda: nn.L1Loss(),                             lambda: (randn(10, 10), zeros(10, 10))),
    (lambda: nn.LPPool1d(2, 3),                       lambda: randn(1, 10, 10)),
    (lambda: nn.LPPool2d(2, 3),                       lambda: randn(1, 10, 10, 10)),
    (lambda: nn.LSTM(10, 20, 2),                      lambda: (randn(5, 3, 10), (randn(2, 3, 20), randn(2, 3, 20)))),
    (lambda: nn.LSTMCell(10, 20),                     lambda: (randn(3, 10), (randn(3, 20), randn(3, 20)))),
    (lambda: nn.LayerNorm((2, 3, 4)),                 lambda: randn(10, 2, 3, 4)),
    # (lambda: nn.LazyBatchNorm1d(...),               lambda: ...),
    # (lambda: nn.LazyBatchNorm2d(...),               lambda: ...),
    # (lambda: nn.LazyBatchNorm3d(...),               lambda: ...),
    # (lambda: nn.LazyConv1d(...),                    lambda: ...),
    # (lambda: nn.LazyConv2d(...),                    lambda: ...),
    # (lambda: nn.LazyConv3d(...),                    lambda: ...),
    # (lambda: nn.LazyConvTranspose1d(...),           lambda: ...),
    # (lambda: nn.LazyConvTranspose2d(...),           lambda: ...),
    # (lambda: nn.LazyConvTranspose3d(...),           lambda: ...),
    # (lambda: nn.LazyInstanceNorm1d(...),            lambda: ...),
    # (lambda: nn.LazyInstanceNorm2d(...),            lambda: ...),
    # (lambda: nn.LazyInstanceNorm3d(...),            lambda: ...),
    # (lambda: nn.LazyLinear(...),                    lambda: ...),
    (lambda: nn.LeakyReLU(),                          lambda: randn(100)),
    (lambda: nn.Linear(10, 10),                       lambda: randn(10, 10)),
    (lambda: nn.LocalResponseNorm(2),                 lambda: randn(1, 4, 5, 5)),
    (lambda: nn.LogSigmoid(),                         lambda: randn(100)),
    (lambda: nn.LogSoftmax(),                         lambda: randn(100)),
    (lambda: nn.MSELoss(),                            lambda: (randn(10, 10), zeros(10, 10))),
    (lambda: nn.MarginRankingLoss(),                  lambda: (randn(100), randn(100), randn(100).sign().long())),
    (lambda: nn.MaxPool1d(3),                         lambda: randn(1, 10, 10)),
    (lambda: nn.MaxPool2d(3),                         lambda: randn(1, 10, 10, 10)),
    (lambda: nn.MaxPool3d(3),                         lambda: randn(1, 1, 10, 10, 10)),
    (lambda: nn.MaxUnpool1d(2),                       lambda: nn.MaxPool1d(2, return_indices=True)(randn(1, 10, 10))),
    (lambda: nn.MaxUnpool2d(2),                       lambda: nn.MaxPool2d(2, return_indices=True)(randn(1, 10, 10, 10))),
    (lambda: nn.MaxUnpool3d(2),                       lambda: nn.MaxPool3d(2, return_indices=True)(randn(1, 1, 10, 10, 10))),
    (lambda: nn.Mish(),                               lambda: randn(100)),
    # (lambda: nn.Module(...),                        lambda: ...),
    # (lambda: nn.ModuleDict(...),                    lambda: ...),
    # (lambda: nn.ModuleList(...),                    lambda: ...),
    (lambda: nn.MultiLabelMarginLoss(),               lambda: (randn(10, 10), randint(-1, 10, (10, 10)))),
    (lambda: nn.MultiLabelSoftMarginLoss(),           lambda: (randn(10, 10), F.one_hot(arange(10)))),
    (lambda: nn.MultiMarginLoss(),                    lambda: (randn(10, 10), randint(0, 10, (10,)))),
    # (lambda: nn.MultiheadAttention(...),            lambda: ...),
    (lambda: nn.NLLLoss(),                            lambda: (randn(10, 10), randint(10, (10,)))),
    (lambda: nn.NLLLoss2d(),                          lambda: (randn(10, 10), randint(10, (10,)))),
    (lambda: nn.PReLU(),                              lambda: randn(100)),
    (lambda: nn.PairwiseDistance(),                   lambda: (randn(100, 10), randn(100, 10))),
    # (lambda: nn.ParameterDict(...),                 lambda: ...),
    # (lambda: nn.ParameterList(...),                 lambda: ...),
    (lambda: nn.PixelShuffle(1),                      lambda: randn(1, 1, 10, 10)),
    # (lambda: nn.PixelUnshuffle(...),                lambda: ...),
    (lambda: nn.PoissonNLLLoss(),                     lambda: (randn(100), randn(100))),
    (lambda: nn.RNN(10, 20, 2),                       lambda: (randn(5, 3, 10), randn(2, 3, 20))),
    # (lambda: nn.RNNBase(...),                       lambda: ...),
    (lambda: nn.RNNCell(10, 20),                      lambda: (randn(3, 10), randn(3, 20))),
    # (lambda: nn.RNNCellBase(...),                   lambda: ...),
    (lambda: nn.RReLU(),                              lambda: randn(100)),
    (lambda: nn.ReLU(),                               lambda: randn(100)),
    (lambda: nn.ReLU6(),                              lambda: randn(100)),
    (lambda: nn.ReflectionPad1d(2),                   lambda: randn(1, 10, 10)),
    (lambda: nn.ReflectionPad2d(2),                   lambda: randn(1, 10, 10)),
    (lambda: nn.ReplicationPad1d(2),                  lambda: randn(1, 10, 10)),
    (lambda: nn.ReplicationPad2d(2),                  lambda: randn(1, 10, 10)),
    (lambda: nn.ReplicationPad3d(2),                  lambda: randn(1, 10, 2, 5)),
    (lambda: nn.SELU(),                               lambda: randn(100)),
    # (lambda: nn.Sequential(...),                         lambda: ...),
    (lambda: nn.SiLU(),                               lambda: randn(100)),
    (lambda: nn.Sigmoid(),                            lambda: randn(100)),
    (lambda: nn.SmoothL1Loss(),                       lambda: (randn(100), zeros(100))),
    (lambda: nn.SoftMarginLoss(),                     lambda: (randn(100), randn(100).sign().long())),
    (lambda: nn.Softmax(),                            lambda: randn(100)),
    (lambda: nn.Softmax2d(),                          lambda: randn(1, 4, 25)),
    (lambda: nn.Softmin(),                            lambda: randn(100)),
    (lambda: nn.Softplus(),                           lambda: randn(100)),
    (lambda: nn.Softshrink(),                         lambda: randn(100)),
    (lambda: nn.Softsign(),                           lambda: randn(100)),
    # (lambda: nn.SyncBatchNorm(...),                      lambda: ...),
    (lambda: nn.Tanh(),                               lambda: randn(100)),
    (lambda: nn.Tanhshrink(),                         lambda: randn(100)),
    (lambda: nn.Threshold(0.1, 20),                   lambda: randn(100)),
    # (lambda: nn.Transformer(...),                   lambda: ...),
    # (lambda: nn.TransformerDecoder(...),            lambda: ...),
    # (lambda: nn.TransformerDecoderLayer(...),       lambda: ...),
    # (lambda: nn.TransformerEncoder(...),            lambda: ...),
    # (lambda: nn.TransformerEncoderLayer(...),       lambda: ...),
    (lambda: nn.TripletMarginLoss(),                  lambda: (randn(100), randn(100), randn(100),)),
    (lambda: nn.TripletMarginWithDistanceLoss(),      lambda: (randn(100), randn(100), randn(100),)),
    (lambda: nn.Unflatten(1, (2, 5, 5)),              lambda: randn(2, 50)),
    (lambda: nn.Unfold((2, 3)),                       lambda: randn(2, 5, 3, 4)),
    (lambda: nn.Upsample(scale_factor=2),             lambda: randn(1, 1, 10, 10)),
    (lambda: nn.UpsamplingBilinear2d(scale_factor=2), lambda: randn(1, 1, 10, 10)),
    (lambda: nn.UpsamplingNearest2d(scale_factor=2),  lambda: randn(1, 1, 10, 10)),
    (lambda: nn.ZeroPad2d(2),                         lambda: randn(1, 10, 10)),
],
[
    (lambda: nn.ReflectionPad3d(2),                   lambda: randn(1, 5, 5, 5)),
] if version.parse(torch.__version__) >= version.parse("1.10.0") else [],
[
    (lambda: nn.Dropout1d(),                          lambda: randn(10, 10)),
] if version.parse(torch.__version__) >= version.parse("1.12.0") else [],
[
    (lambda: nn.CircularPad1d(2),                     lambda: randn(1, 10, 10)),
    (lambda: nn.CircularPad2d(2),                     lambda: randn(1, 10, 10)),
    (lambda: nn.CircularPad3d(2),                     lambda: randn(1, 10, 2, 5)),
    (lambda: nn.ZeroPad1d(2),                         lambda: randn(1, 10, 10)),
    (lambda: nn.ZeroPad3d(2),                         lambda: randn(1, 10, 2, 5))
] if version.parse(torch.__version__) >= version.parse("2.1.0") else [],
[
    (lambda: custom.Sin(),                            lambda: randn(100)),
    (lambda: custom.Cos(),                            lambda: randn(100)),
    (lambda: custom.AvgPool2d(),                      lambda: (randn(1,10,10), (tensor(2),))),
    (lambda: custom.Reshape(),                        lambda: (randn(10,10), (tensor(100), tensor(1)))),
    (lambda: custom.RSqrt(),                          lambda: randn(100).abs()),
    (lambda: custom.Add(),                            lambda: (randn(100), randn(100))),
    (lambda: custom.Multiply(),                       lambda: (randn(100), randn(100))),
    (lambda: custom.Subtract(),                       lambda: (randn(100), randn(100))),
    (lambda: custom.Divide(),                         lambda: (randn(100), randn(100))),
    (lambda: custom.Concat(),                         lambda: (randn(1, 100), randn(3, 100))),
    # (lambda: custom.FloorDivide(),                  lambda: ...),
    (lambda: custom.Norm(),                           lambda: randn(100)),
    (lambda: custom.Exponential(),                    lambda: randn(100)),
    (lambda: custom.Erf(),                            lambda: randn(100)),
    (lambda: custom.Sqrt(),                           lambda: randn(100).abs()),
    # (lambda: custom.Maximum(),                      lambda: ...),
    # (lambda: custom.Max(),                          lambda: ...),
    # (lambda: custom.AMax(),                         lambda: ...),
    # (lambda: custom.Minimum(),                      lambda: ...),
    # (lambda: custom.Min(),                          lambda: ...),
    # (lambda: custom.AMin(),                         lambda: ...),
    # (lambda: custom.Where(),                        lambda: ...),
    # (lambda: custom.Greater(),                      lambda: ...),
    # (lambda: custom.Less(),                         lambda: ...),
    # (lambda: custom.GreaterEqual(),                 lambda: ...),
    # (lambda: custom.LessEqual(),                    lambda: ...),
    # (lambda: custom.NotEqual(),                     lambda: ...),
    # (lambda: custom.Equal(),                        lambda: ...),
    (lambda: custom.Bmm(),                            lambda: (randn(1, 100, 100), randn(1, 100, 100))),
    (lambda: custom.CumSum(),                         lambda: (randn(10, 100), tensor(0))),
    # (lambda: custom.MaskedFill(),                   lambda: ...),
    # (lambda: custom.Mean(),                         lambda: ...),
    # (lambda: custom.Sum(),                          lambda: ...),
    # (lambda: custom.Prod(),                         lambda: ...),
    (lambda: custom.Log(),                            lambda: randint(1, 1000, (10,10))),
    (lambda: custom.Abs(),                            lambda: randn(100)),
    (lambda: custom.Neg(),                            lambda: randn(100)),
    # (lambda: custom.Argmin(),                       lambda: ...),
    # (lambda: custom.Argmax(),                       lambda: ...),
    # (lambda: custom.ElementwiseCeil(),              lambda: ...),
    # (lambda: custom.ElementwiseFloor(),             lambda: ...),
    # (lambda: custom.Asin(),                         lambda: ...),
    # (lambda: custom.Atan(),                         lambda: ...),
    # (lambda: custom.Round(),                        lambda: ...),
    # (lambda: custom.Gather(),                       lambda: ...),
    # (lambda: custom.LogicalOr(),                    lambda: ...),
    # (lambda: custom.LogicalAnd(),                   lambda: ...),
    # (lambda: custom.LogicalNot(),                   lambda: ...),
    # (lambda: custom.Split(),                        lambda: ...),
    # (lambda: custom.Permute(),                      lambda: ...),
    # (lambda: custom.Remainder(),                    lambda: ...),
    # (lambda: custom.IndexSelect(),                  lambda: ...),
    # (lambda: custom.Fmod(),                         lambda: ...),
    # (lambda: custom.NonZero(),                      lambda: ...),
    # (lambda: custom.TopK(),                         lambda: ...),
    # (lambda: custom.Shape(),                        lambda: ...),
    # (lambda: custom.Tile(),                         lambda: ...),
    # (lambda: custom.ElementwiseUnarySign(),         lambda: ...),
    (lambda: custom.Baddbmm(),                        lambda: (randn(1, 100, 100), randn(1, 100, 100), randn(1, 100, 100))),
    (lambda: custom.Addmm(),                          lambda: (randn(100, 100), randn(100, 100), randn(100, 100))),
    # (lambda: custom.Square(),                       lambda: ...),
    # (lambda: custom.Select(),                       lambda: ...),
    # (lambda: custom.Interpolate(),                  lambda: ...),
    # (lambda: custom.MaxPool2d(),                    lambda: ...),
    # (lambda: custom.AdaptiveAvgPool2d(),            lambda: ...),
    (lambda: custom.BatchNorm(),                      lambda: (randn(5, 10), zeros(10).requires_grad_(), ones(10).requires_grad_())),
    (lambda: custom.BatchNorm(),                      lambda: (randn(5, 10, 3, 2), zeros(10).requires_grad_(), ones(10).requires_grad_())),
    (lambda: custom.BatchNorm(),                      lambda: (randn(5, 10, 3, 2, 5), zeros(10).requires_grad_(), ones(10).requires_grad_())),
    (lambda: custom.BatchNorm(),                      lambda: (randn(5, 10), zeros(10), ones(10))),
    (lambda: custom.BatchNorm(),                      lambda: (randn(5, 10, 3, 2), zeros(10), ones(10))),
    (lambda: custom.BatchNorm(),                      lambda: (randn(5, 10, 3, 2, 5), zeros(10), ones(10))),
    (lambda: custom.GroupNorm(),                      lambda: (randn(20, 6, 10, 10), tensor(6))),
    (lambda: custom.Normalize(),                      lambda: randn(100, 100)),
    # (lambda: custom.Pad(),                          lambda: ...),
    # (lambda: custom.GridSample(),                   lambda: ...),
    (lambda: custom.RmsNorm([5, 2, 3], [2], 1e-5),    lambda: (randn(5, 2, 3))),
    # (lambda custom.DynamicConv2d(),                 lambda: ...),
    # (lambda custom.Pow(),                           lambda: ...),
    # (lambda custom.CustomSiLU(),                    lambda: ...),
    # (lambda custom.StridedSlice(),                  lambda: ...),
    # (lambda custom.ChannelShuffle(),                lambda: ...),
    # (lambda custom.Cast(),                          lambda: ...),
    # (lambda custom.CustomGather(),                  lambda: ...),
    # (lambda custom.DepthToSpaceCRDMode(),           lambda: ...),
    # (lambda custom.DepthToSpaceDCRMode(),           lambda: ...),
    # (lambda custom.CustomSparseConv3DLayer(),       lambda: ...),
    # (lambda custom.SparseTensorWrapper(),           lambda: ...),
    # (lambda custom.ScatterDense(),                  lambda: ...),
    # (lambda custom.ScatterND(),                     lambda: ...),
    # (lambda custom.RoiAlign(),                      lambda: ...),
    # (lambda custom.NonMaxSuppression(),             lambda: ...),
    # (lambda custom.GatherNd(),                      lambda: ...),
    # (lambda custom.ScatterElements(),               lambda: ...),
    # (lambda custom.OneHot(),                        lambda: ...),
    # (lambda custom.Expand(),                        lambda: ...),
    # (lambda custom.DynamicLinear(),                 lambda: ...),
]))
def test_default_kernels(module_factory, input_factory):
    module = module_factory()
    inputs = input_factory()

    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)

    """
    When: Run quantized module forward pass with default kernel
    Then: The output should be equal to that of the legacy fake-quantized modules
    """
    legacy_qmodule = _create_legacy_fake_quantized_module(module)
    qmodule = _create_quantized_module(module)

    # NOTE: Need to fix seed again before every forward pass
    #       in case the module involves randomized behavior (e.g. RReLU)

    with legacy_qmodule.compute_encodings():
        torch.manual_seed(0)
        _ = legacy_qmodule(*inputs)


    with qmodule.compute_encodings():
        torch.manual_seed(0)
        _ = qmodule(*inputs)

    torch.manual_seed(0)
    fout = legacy_qmodule(*inputs)
    torch.manual_seed(0)
    out = qmodule(*inputs)

    for out_, fout_ in zip(tree_flatten(out)[0], tree_flatten(fout)[0]):
        assert torch.equal(out_, fout_)
        assert torch.all(out_.isfinite())

    """
    When: Trace a quantized modules with torch.jit.trace
    Then: 1) Tracing shouldn't fail
          2) The traced module should produce the same output as the original module
    """
    traced = torch.jit.trace(qmodule, inputs)
    torch.manual_seed(0)
    tout = traced(*inputs)

    for out_, tout_ in zip(tree_flatten(out)[0], tree_flatten(tout)[0]):
        assert torch.equal(out_, tout_)

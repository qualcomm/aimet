# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import ast
import itertools
import copy
import functools
from packaging import version
from typing import Optional

import pytest
import torch
from torch import randn, randint, zeros, full, arange, ones, tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.utils._pytree import tree_flatten
from torch.overrides import get_ignored_functions
import transformers

from aimet_torch.v2.quantization.affine.backends import (
    quantize,
    quantize_dequantize,
    dequantize,
)
import aimet_torch
from aimet_torch.v2.quantization.affine import (
    AffineEncoding,
    Quantize,
    QuantizeDequantize,
    GroupedBlockQuantizeDequantize,
)
import aimet_torch.v2 as aimet
from aimet_torch.v2.nn import (
    QuantizationMixin,
    QuantizedConv1d,
    QuantizedConv2d,
    QuantizedConv3d,
    QuantizedConvTranspose1d as QConvTranspose1d,
    QuantizedConvTranspose2d as QConvTranspose2d,
    QuantizedConvTranspose3d as QConvTranspose3d,
    QuantizedEmbedding,
    QuantizedGELU,
    QuantizedGroupNorm,
    QuantizedInstanceNorm1d,
    QuantizedInstanceNorm2d,
    QuantizedInstanceNorm3d,
    QuantizedLayerNorm,
    QuantizedLinear,
    QuantizedSigmoid,
    QuantizedSoftmax,
    UnknownModuleError,
)
from aimet_torch.nn.fake_quant import _legacy_impl
from aimet_torch.nn.true_quant import _dispatch, _dispatchable_torch_functions
from aimet_torch.v2.quantization.tensor import QuantizedTensor, DequantizedTensor
from aimet_torch.v2.utils import enable_recompute
from aimet_torch.v2.nn import custom
from aimet_torch.v2.quantsim.config_utils import (
    set_grouped_blockwise_quantization_for_weights,
)


@pytest.fixture(autouse=True)
def clear_torch_compile_cache():
    yield
    torch.compiler.reset()


@pytest.fixture(autouse=True)
def manual_seed():
    torch.manual_seed(724)


@pytest.fixture
def use_torch_builtin_backend():
    with aimet_torch.quantization.set_backend("torch_builtins"):
        yield


def affine_quantize(
    tensor: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, bitwidth: int
) -> QuantizedTensor:
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
    return torch.arange(1, numel + 1).view(*shape) / numel


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

        return affine_quantize(
            input.mm(weight.t()) + bias,
            output_encodings.scale,
            output_encodings.offset,
            output_encodings.bitwidth,
        )

    QuantizedLinear.set_default_kernel(int_linear)
    yield
    QuantizedLinear.set_default_kernel(None)


@pytest.fixture
def register_int_conv():
    def int_convnd(
        kernel,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        *,
        output_encodings=None,
    ):
        # Implicit dequantization is not supported yet
        if not isinstance(input, QuantizedTensor):
            raise RuntimeError
        if not isinstance(weight, QuantizedTensor):
            raise RuntimeError

        input = input.dequantize()
        weight = weight.dequantize()
        output = kernel(input, weight, bias, stride, padding, dilation, groups)
        return affine_quantize(
            output,
            output_encodings.scale,
            output_encodings.offset,
            output_encodings.bitwidth,
        )

    QuantizedConv1d.set_default_kernel(functools.partial(int_convnd, F.conv1d))
    QuantizedConv2d.set_default_kernel(functools.partial(int_convnd, F.conv2d))
    QuantizedConv3d.set_default_kernel(functools.partial(int_convnd, F.conv3d))
    yield
    QuantizedConv3d.set_default_kernel(None)
    QuantizedConv2d.set_default_kernel(None)
    QuantizedConv1d.set_default_kernel(None)


@pytest.fixture
def register_int_convtranspose():
    def int_convtransposend(
        kernel,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=1,
        *,
        output_encodings=None,
    ):
        # Implicit dequantization is not supported yet
        if not isinstance(input, QuantizedTensor):
            raise RuntimeError
        if not isinstance(weight, QuantizedTensor):
            raise RuntimeError

        input = input.dequantize()
        weight = weight.dequantize()
        output = kernel(
            input, weight, bias, stride, padding, output_padding, groups, dilation
        )
        return affine_quantize(
            output,
            output_encodings.scale,
            output_encodings.offset,
            output_encodings.bitwidth,
        )

    QConvTranspose1d.set_default_kernel(
        functools.partial(int_convtransposend, F.conv_transpose1d)
    )
    QConvTranspose2d.set_default_kernel(
        functools.partial(int_convtransposend, F.conv_transpose2d)
    )
    QConvTranspose3d.set_default_kernel(
        functools.partial(int_convtransposend, F.conv_transpose3d)
    )
    yield
    QConvTranspose1d.set_default_kernel(None)
    QConvTranspose2d.set_default_kernel(None)
    QConvTranspose3d.set_default_kernel(None)


@pytest.fixture
def register_int_activation():
    def wrap_functional(func):
        def wrapped_func(*args, output_encodings=None, **kwargs):
            # Implicit dequantization is not supported yet
            x, *others = args
            if not isinstance(x, QuantizedTensor):
                raise RuntimeError
            output = func(x.dequantize(), *others, **kwargs)
            return affine_quantize(
                output,
                output_encodings.scale,
                output_encodings.offset,
                output_encodings.bitwidth,
            )

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
        def int_norm(
            input, normalized_shape, weight, bias, eps, *, output_encodings=None
        ):
            # Implicit dequantization is not supported yet
            if not isinstance(input, QuantizedTensor):
                raise RuntimeError
            if not isinstance(weight, QuantizedTensor):
                raise RuntimeError

            input = input.dequantize()
            weight = weight.dequantize()

            output = func(input, normalized_shape, weight, bias, eps)
            return affine_quantize(
                output,
                output_encodings.scale,
                output_encodings.offset,
                output_encodings.bitwidth,
            )

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
        return affine_quantize(
            output,
            output_encodings.scale,
            output_encodings.offset,
            output_encodings.bitwidth,
        )

    custom.QuantizedAdd.set_default_kernel(
        functools.partial(int_elementwise, torch.add)
    )
    custom.QuantizedMultiply.set_default_kernel(
        functools.partial(int_elementwise, torch.multiply)
    )
    custom.QuantizedSubtract.set_default_kernel(
        functools.partial(int_elementwise, torch.subtract)
    )
    custom.QuantizedDivide.set_default_kernel(
        functools.partial(int_elementwise, torch.div)
    )
    custom.QuantizedMatMul.set_default_kernel(
        functools.partial(int_elementwise, torch.matmul)
    )
    yield
    custom.QuantizedMultiply.set_default_kernel(None)
    custom.QuantizedSubtract.set_default_kernel(None)
    custom.QuantizedAdd.set_default_kernel(None)
    custom.QuantizedDivide.set_default_kernel(None)
    custom.QuantizedMatMul.set_default_kernel(None)


class TestTrueQuantLinear:
    @pytest.mark.usefixtures("register_int_linear")
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

    @pytest.mark.usefixtures("register_int_linear")
    def test_fully_specified_quantizers(self, input):
        """
        Given: TrueQuantLinear with input, output, and param quantizers
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        quant_linear.input_quantizers[0] = Quantize((1,), bitwidth=8, symmetric=False)
        quant_linear.output_quantizers[0] = Quantize((1,), bitwidth=8, symmetric=False)
        quant_linear.param_quantizers["weight"] = Quantize(
            (10,), bitwidth=8, symmetric=True
        )
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

        input_enc = (
            quant_linear.input_quantizers[0].get_scale(),
            quant_linear.input_quantizers[0].get_offset(),
            quant_linear.input_quantizers[0].bitwidth,
        )
        output_enc = (
            quant_linear.output_quantizers[0].get_scale(),
            quant_linear.output_quantizers[0].get_offset(),
            quant_linear.output_quantizers[0].bitwidth,
        )
        weight_enc = (
            quant_linear.param_quantizers["weight"].get_scale(),
            quant_linear.param_quantizers["weight"].get_offset(),
            quant_linear.param_quantizers["weight"].bitwidth,
        )
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

    @pytest.mark.usefixtures("register_int_linear")
    def test_no_input_quantizer(self, input):
        """
        Given: TrueQuantLinear with output and param quantizers and computed encodings
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        quant_linear.output_quantizers[0] = Quantize((1,), bitwidth=8, symmetric=False)
        quant_linear.param_quantizers["weight"] = Quantize(
            (10,), bitwidth=8, symmetric=True
        )
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
        quantizer = Quantize((1,), bitwidth=8, symmetric=False)
        with quantizer.compute_encodings():
            quantizer(input)

        input_q = quantizer(input)
        output = quant_linear(input_q)
        assert isinstance(output, DequantizedTensor)
        assert output.encoding.scale == quant_linear.output_quantizers[0].get_scale()
        assert output.encoding.offset == quant_linear.output_quantizers[0].get_offset()

    @pytest.mark.usefixtures("register_int_linear")
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
    @pytest.mark.usefixtures(
        "register_int_norm", "register_int_custom", "register_int_activation"
    )
    @pytest.mark.parametrize(
        "module_factory,               input_factory",
        [
            (lambda: nn.Softmax(dim=1), lambda: _input(10, 10)),
            (lambda: nn.Sigmoid(), lambda: _input(10, 10)),
            (lambda: nn.GELU(), lambda: _input(10, 10)),
            (lambda: custom.Add(), lambda: (_input(10, 10), _input(10, 10))),
            (lambda: custom.Multiply(), lambda: (_input(10, 10), _input(10, 10))),
            (lambda: custom.Subtract(), lambda: (_input(10, 10), _input(10, 10))),
            (lambda: custom.MatMul(), lambda: (_input(10, 10), _input(10, 10))),
            (lambda: custom.Divide(), lambda: (_input(10, 10), _input(10, 10))),
        ],
    )
    def test_layers_no_params(self, module_factory, input_factory):
        layer = module_factory()
        inputs = input_factory()

        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)

        fq_layer = _legacy_impl.FakeQuantizationMixin.from_module(layer)
        tq_layer = QuantizationMixin.from_module(layer)
        for i, _ in enumerate(inputs):
            fq_layer.input_quantizers[i] = QuantizeDequantize(
                shape=(), bitwidth=8, symmetric=False
            )
            tq_layer.input_quantizers[i] = Quantize(
                shape=(), bitwidth=8, symmetric=False
            )

        fq_layer.output_quantizers[0] = QuantizeDequantize(
            shape=(1,), bitwidth=8, symmetric=False
        )
        tq_layer.output_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)

        with fq_layer.compute_encodings():
            fq_layer(*inputs)

        fq_output = fq_layer(*inputs)

        with tq_layer.compute_encodings():
            tq_layer(*inputs)
        tq_output = tq_layer(*inputs)

        assert torch.allclose(fq_output, tq_output.dequantize())

    @pytest.mark.usefixtures(
        "register_int_linear",
        "register_int_norm",
        "register_int_custom",
        "register_int_activation",
        "register_int_conv",
        "register_int_convtranspose",
    )
    @pytest.mark.parametrize(
        "module_factory,                      input_factory",
        [
            (lambda: nn.Linear(10, 10), lambda: _input(10, 10)),
            (lambda: nn.LayerNorm(10), lambda: _input(10, 10)),
            (lambda: nn.GroupNorm(2, 10), lambda: _input(10, 10)),
            (lambda: nn.Conv1d(3, 3, 3), lambda: _input(1, 3, 10)),
            (lambda: nn.Conv2d(3, 3, 3), lambda: _input(1, 3, 10, 10)),
            (lambda: nn.Conv3d(3, 3, 3), lambda: _input(1, 3, 10, 10, 10)),
            (lambda: nn.ConvTranspose1d(3, 3, 3), lambda: _input(1, 3, 10)),
            (lambda: nn.ConvTranspose2d(3, 3, 3), lambda: _input(1, 3, 10, 10)),
            (lambda: nn.ConvTranspose3d(3, 3, 3), lambda: _input(1, 3, 10, 10, 10)),
        ],
    )
    def test_layers_with_weight(self, module_factory, input_factory):
        layer = module_factory()
        input = input_factory()

        fq_layer = _legacy_impl.FakeQuantizationMixin.from_module(layer)
        tq_layer = QuantizationMixin.from_module(layer)
        fq_layer.input_quantizers[0] = QuantizeDequantize(
            shape=(), bitwidth=8, symmetric=False
        )
        fq_layer.output_quantizers[0] = QuantizeDequantize(
            shape=(), bitwidth=8, symmetric=False
        )
        fq_layer.param_quantizers["weight"] = QuantizeDequantize(
            shape=(), bitwidth=8, symmetric=True
        )
        tq_layer.input_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        tq_layer.output_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        tq_layer.param_quantizers["weight"] = Quantize(
            shape=(), bitwidth=8, symmetric=True
        )

        with fq_layer.compute_encodings():
            fq_layer(input)

        fq_output = fq_layer(input)

        with tq_layer.compute_encodings():
            tq_layer(input)
        tq_output = tq_layer(input)

        assert torch.allclose(fq_output, tq_output.dequantize())

    @pytest.mark.cuda
    @pytest.mark.usefixtures("register_int_linear")
    def test_layers_with_recompute(self, use_torch_builtin_backend):
        qlinear = QuantizedLinear(4096, 4096)
        qlinear.input_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        qlinear.output_quantizers[0] = Quantize(shape=(), bitwidth=8, symmetric=False)
        qlinear.param_quantizers["weight"] = Quantize(
            shape=(), bitwidth=8, symmetric=True
        )
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
        grads_with_recompute = [
            param.grad.clone().detach().cpu() for param in qlinear.parameters()
        ]
        for param in qlinear.parameters():
            param.grad = None

        del out

        torch.cuda.empty_cache()
        out = qlinear(x)
        torch.cuda.synchronize()
        mem_without_recompute = torch.cuda.memory_allocated()

        out.backward(torch.ones_like(out))
        grads_without_recompute = [
            param.grad.clone().detach().cpu() for param in qlinear.parameters()
        ]
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
        expected_memory_saving += (1 + 1) * x.numel()  # input quantizer
        expected_memory_saving += (1 + 1) * qlinear.weight.numel()  # weight quantizer
        expected_memory_saving += 4 * (qlinear.weight.numel() + x.numel())  # F.linear
        expected_memory_saving += (4 + 1 + 1) * out.numel()  # output quantizer
        actual_memory_saving = mem_without_recompute - mem_with_recompute

        # Considering noise factors, actual memory saving should be no less than
        # 90% of the expected memory saving
        assert expected_memory_saving * 0.9 <= actual_memory_saving

        for grad_0, grad_1 in zip(grads_with_recompute, grads_without_recompute):
            assert torch.equal(grad_0, grad_1)

    def test_remove_quantizers(self, input):
        qlinear = QuantizedLinear(10, 10, bias=False)
        qlinear.input_quantizers[0] = input_qtzr = Quantize(
            shape=(), bitwidth=8, symmetric=False
        )
        qlinear.output_quantizers[0] = output_qtzr = Quantize(
            shape=(), bitwidth=8, symmetric=False
        )
        qlinear.param_quantizers["weight"] = weight_qtzr = Quantize(
            shape=(), bitwidth=8, symmetric=True
        )
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
                F.linear(input, weight_qtzr(qlinear.weight).dequantize())
            ).dequantize()
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.input_quantizers[0] is input_qtzr

        with qlinear._remove_param_quantizers("weight"):
            assert qlinear.param_quantizers["weight"] is None
            expected_out = output_qtzr(
                F.linear(input_qtzr(input).dequantize(), qlinear.weight)
            ).dequantize()
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.param_quantizers["weight"] is weight_qtzr

        with qlinear._remove_output_quantizers(0):
            assert qlinear.output_quantizers[0] is None
            expected_out = F.linear(
                input_qtzr(input).dequantize(), weight_qtzr(qlinear.weight).dequantize()
            )
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
            assert qlinear.param_quantizers["weight"] is None
            expected_out = F.linear(input, qlinear.weight)
            assert torch.equal(qlinear(input), expected_out)
        assert qlinear.input_quantizers[0] is input_qtzr
        assert qlinear.output_quantizers[0] is output_qtzr
        assert qlinear.param_quantizers["weight"] is weight_qtzr

        """
        When: Call ``_remove_{input, param, output}_quantizers`` without ``with`` statement
        Then: The corresponding quantizers are set to None permanently
        """
        qlinear._remove_input_quantizers(0)
        assert qlinear.input_quantizers[0] is None
        qlinear._remove_param_quantizers("weight")
        assert qlinear.param_quantizers["weight"] is None
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
    for func in get_ignored_functions() - _dispatchable_torch_functions:
        dummy_impl = lambda *args, **kwargs: func(*args, **kwargs)
        with pytest.raises(RuntimeError):
            with _dispatch(func, dummy_impl):
                pass

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

    def _linear_impl(x, y):
        _linear_impl.call_cnt += 1
        return F.linear(x, y)

    _linear_impl.call_cnt = 0

    """
    When: Dispatch _linear_impl for F.linear in a nested context manager
    Then: _linear_impl should be called only once
    """
    # Dispatch F.linear in the outer context manager
    with _dispatch(F.linear, _linear_impl):
        with _dispatch(torch.mul, lambda x, y: x * y):
            F.linear(x, y)
    assert _linear_impl.call_cnt == 1
    _linear_impl.call_cnt = 0

    # Dispatch F.linear in the inner context manager
    with _dispatch(torch.mul, lambda x, y: x * y):
        with _dispatch(F.linear, _linear_impl):
            F.linear(x, y)
    assert _linear_impl.call_cnt == 1
    _linear_impl.call_cnt = 0

    """
    When: Dispatch _linear_impl for F.linear N times in a nested context manager
    Then: _linear_impl should be called N times
    """
    with _dispatch(F.linear, _linear_impl):
        with _dispatch(F.linear, _linear_impl):
            F.linear(x, y)
    assert _linear_impl.call_cnt == 2
    _linear_impl.call_cnt = 0

    with _dispatch(F.linear, _linear_impl):
        with _dispatch(F.linear, _linear_impl):
            with _dispatch(F.linear, _linear_impl):
                F.linear(x, y)
    assert _linear_impl.call_cnt == 3


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


_MODULE_FACTORIES = {
    nn.AdaptiveAvgPool1d: lambda: nn.AdaptiveAvgPool1d(2),
    nn.AdaptiveAvgPool2d: lambda: nn.AdaptiveAvgPool2d(2),
    nn.AdaptiveAvgPool3d: lambda: nn.AdaptiveAvgPool3d(2),
    # nn.AdaptiveLogSoftmaxWithLoss: lambda: nn.AdaptiveLogSoftmaxWithLoss(...),
    nn.AdaptiveMaxPool1d: lambda: nn.AdaptiveMaxPool1d(2),
    nn.AdaptiveMaxPool2d: lambda: nn.AdaptiveMaxPool2d(2),
    nn.AdaptiveMaxPool3d: lambda: nn.AdaptiveMaxPool3d(2),
    nn.AlphaDropout: lambda: nn.AlphaDropout(),
    nn.AvgPool1d: lambda: nn.AvgPool1d(2),
    nn.AvgPool2d: lambda: nn.AvgPool2d(2),
    nn.AvgPool3d: lambda: nn.AvgPool3d(2),
    nn.BCELoss: lambda: nn.BCELoss(),
    nn.BCEWithLogitsLoss: lambda: nn.BCEWithLogitsLoss(),
    nn.BatchNorm1d: lambda: nn.BatchNorm1d(10),
    nn.BatchNorm2d: lambda: nn.BatchNorm2d(10),
    nn.BatchNorm3d: lambda: nn.BatchNorm3d(10),
    nn.Bilinear: lambda: nn.Bilinear(20, 30, 40),
    nn.CELU: lambda: nn.CELU(),
    nn.CTCLoss: lambda: nn.CTCLoss(),
    nn.ChannelShuffle: lambda: nn.ChannelShuffle(2),
    nn.ConstantPad1d: lambda: nn.ConstantPad1d(2, 3.5),
    nn.ConstantPad2d: lambda: nn.ConstantPad2d(2, 3.5),
    nn.ConstantPad3d: lambda: nn.ConstantPad3d(2, 3.5),
    nn.Conv1d: lambda: nn.Conv1d(3, 3, 3),
    nn.Conv2d: lambda: nn.Conv2d(3, 3, 3),
    nn.Conv3d: lambda: nn.Conv3d(3, 3, 3),
    nn.ConvTranspose1d: lambda: nn.ConvTranspose1d(3, 3, 3),
    nn.ConvTranspose2d: lambda: nn.ConvTranspose2d(3, 3, 3),
    nn.ConvTranspose3d: lambda: nn.ConvTranspose3d(3, 3, 3),
    nn.CosineEmbeddingLoss: lambda: nn.CosineEmbeddingLoss(),
    nn.CosineSimilarity: lambda: nn.CosineSimilarity(),
    nn.CrossEntropyLoss: lambda: nn.CrossEntropyLoss(),
    # nn.CrossMapLRN2d: lambda: nn.CrossMapLRN2d(...),
    nn.Dropout: lambda: nn.Dropout(),
    nn.Dropout2d: lambda: nn.Dropout2d(),
    nn.Dropout3d: lambda: nn.Dropout3d(),
    nn.ELU: lambda: nn.ELU(),
    nn.Embedding: lambda: nn.Embedding(100, 100),
    nn.EmbeddingBag: lambda: nn.EmbeddingBag(100, 100, mode="sum"),
    nn.FeatureAlphaDropout: lambda: nn.FeatureAlphaDropout(),
    nn.Flatten: lambda: nn.Flatten(),
    nn.Fold: lambda: nn.Fold((4, 5), (2, 2)),
    nn.FractionalMaxPool2d: lambda: nn.FractionalMaxPool2d(3, (5, 5)),
    nn.FractionalMaxPool3d: lambda: nn.FractionalMaxPool3d(3, (5, 5, 5)),
    nn.GELU: lambda: nn.GELU(),
    nn.GLU: lambda: nn.GLU(),
    nn.GRU: lambda: nn.GRU(10, 20, 2),
    nn.GRUCell: lambda: nn.GRUCell(10, 20),
    nn.GaussianNLLLoss: lambda: nn.GaussianNLLLoss(),
    nn.GroupNorm: lambda: nn.GroupNorm(2, 4),
    nn.Hardshrink: lambda: nn.Hardshrink(0),
    nn.Hardsigmoid: lambda: nn.Hardsigmoid(),
    nn.Hardswish: lambda: nn.Hardswish(),
    nn.Hardtanh: lambda: nn.Hardtanh(),
    nn.HingeEmbeddingLoss: lambda: nn.HingeEmbeddingLoss(),
    nn.HuberLoss: lambda: nn.HuberLoss(),
    nn.InstanceNorm1d: lambda: nn.InstanceNorm1d(10),
    nn.InstanceNorm2d: lambda: nn.InstanceNorm2d(10),
    nn.InstanceNorm3d: lambda: nn.InstanceNorm3d(10),
    nn.KLDivLoss: lambda: nn.KLDivLoss(reduction="batchmean"),
    nn.L1Loss: lambda: nn.L1Loss(),
    nn.LPPool1d: lambda: nn.LPPool1d(2, 3),
    nn.LPPool2d: lambda: nn.LPPool2d(2, 3),
    nn.LSTM: lambda: nn.LSTM(10, 20, 2),
    nn.LSTMCell: lambda: nn.LSTMCell(10, 20),
    nn.LayerNorm: lambda: nn.LayerNorm((2, 3, 4)),
    nn.LeakyReLU: lambda: nn.LeakyReLU(),
    nn.Linear: lambda: nn.Linear(10, 10),
    NonDynamicallyQuantizableLinear: lambda: NonDynamicallyQuantizableLinear(10, 10),
    nn.LocalResponseNorm: lambda: nn.LocalResponseNorm(2),
    nn.LogSigmoid: lambda: nn.LogSigmoid(),
    nn.LogSoftmax: lambda: nn.LogSoftmax(),
    nn.MSELoss: lambda: nn.MSELoss(),
    nn.MarginRankingLoss: lambda: nn.MarginRankingLoss(),
    nn.MaxPool1d: lambda: nn.MaxPool1d(3),
    nn.MaxPool2d: lambda: nn.MaxPool2d(3),
    nn.MaxPool3d: lambda: nn.MaxPool3d(3),
    nn.MaxUnpool1d: lambda: nn.MaxUnpool1d(2),
    nn.MaxUnpool2d: lambda: nn.MaxUnpool2d(2),
    nn.MaxUnpool3d: lambda: nn.MaxUnpool3d(2),
    nn.Mish: lambda: nn.Mish(),
    nn.MultiLabelMarginLoss: lambda: nn.MultiLabelMarginLoss(),
    nn.MultiLabelSoftMarginLoss: lambda: nn.MultiLabelSoftMarginLoss(),
    nn.MultiMarginLoss: lambda: nn.MultiMarginLoss(),
    # nn.MultiheadAttention: lambda: nn.MultiheadAttention(...),
    nn.NLLLoss: lambda: nn.NLLLoss(),
    nn.NLLLoss2d: lambda: nn.NLLLoss2d(),
    nn.PReLU: lambda: nn.PReLU(),
    nn.PairwiseDistance: lambda: nn.PairwiseDistance(),
    nn.PixelShuffle: lambda: nn.PixelShuffle(1),
    # nn.PixelUnshuffle: lambda: nn.PixelUnshuffle(...),
    nn.PoissonNLLLoss: lambda: nn.PoissonNLLLoss(),
    nn.RNN: lambda: nn.RNN(10, 20, 2),
    nn.RNNCell: lambda: nn.RNNCell(10, 20),
    nn.RReLU: lambda: nn.RReLU(),
    nn.ReLU: lambda: nn.ReLU(),
    nn.ReLU6: lambda: nn.ReLU6(),
    nn.ReflectionPad1d: lambda: nn.ReflectionPad1d(2),
    nn.ReflectionPad2d: lambda: nn.ReflectionPad2d(2),
    nn.ReplicationPad1d: lambda: nn.ReplicationPad1d(2),
    nn.ReplicationPad2d: lambda: nn.ReplicationPad2d(2),
    nn.ReplicationPad3d: lambda: nn.ReplicationPad3d(2),
    nn.SELU: lambda: nn.SELU(),
    nn.SiLU: lambda: nn.SiLU(),
    nn.Sigmoid: lambda: nn.Sigmoid(),
    nn.SmoothL1Loss: lambda: nn.SmoothL1Loss(),
    nn.SoftMarginLoss: lambda: nn.SoftMarginLoss(),
    nn.Softmax: lambda: nn.Softmax(),
    nn.Softmax2d: lambda: nn.Softmax2d(),
    nn.Softmin: lambda: nn.Softmin(),
    nn.Softplus: lambda: nn.Softplus(),
    nn.Softshrink: lambda: nn.Softshrink(),
    nn.Softsign: lambda: nn.Softsign(),
    nn.SyncBatchNorm: lambda: nn.SyncBatchNorm(10),
    nn.Tanh: lambda: nn.Tanh(),
    nn.Tanhshrink: lambda: nn.Tanhshrink(),
    nn.Threshold: lambda: nn.Threshold(0.1, 20),
    # nn.Transformer: lambda: nn.Transformer(...),
    # nn.TransformerDecoder: lambda: nn.TransformerDecoder(...),
    # nn.TransformerDecoderLayer: lambda: nn.TransformerDecoderLayer(...),
    # nn.TransformerEncoder: lambda: nn.TransformerEncoder(...),
    # nn.TransformerEncoderLayer: lambda: nn.TransformerEncoderLayer(...),
    nn.TripletMarginLoss: lambda: nn.TripletMarginLoss(),
    nn.TripletMarginWithDistanceLoss: lambda: nn.TripletMarginWithDistanceLoss(),
    nn.Unflatten: lambda: nn.Unflatten(1, (2, 5, 5)),
    nn.Unfold: lambda: nn.Unfold((2, 3)),
    nn.Upsample: lambda: nn.Upsample(scale_factor=2),
    nn.UpsamplingBilinear2d: lambda: nn.UpsamplingBilinear2d(scale_factor=2),
    nn.UpsamplingNearest2d: lambda: nn.UpsamplingNearest2d(scale_factor=2),
    nn.ZeroPad2d: lambda: nn.ZeroPad2d(2),
    nn.ReflectionPad3d: lambda: nn.ReflectionPad3d(2),
    nn.Dropout1d: lambda: nn.Dropout1d(),
    nn.CircularPad1d: lambda: nn.CircularPad1d(2),
    nn.CircularPad2d: lambda: nn.CircularPad2d(2),
    nn.CircularPad3d: lambda: nn.CircularPad3d(2),
    nn.ZeroPad1d: lambda: nn.ZeroPad1d(2),
    nn.ZeroPad3d: lambda: nn.ZeroPad3d(2),
    custom.Sin: lambda: custom.Sin(),
    custom.Cos: lambda: custom.Cos(),
    custom.AvgPool2d: lambda: custom.AvgPool2d(),
    custom.Reshape: lambda: custom.Reshape(),
    custom.RSqrt: lambda: custom.RSqrt(),
    custom.Add: lambda: custom.Add(),
    custom.Multiply: lambda: custom.Multiply(),
    custom.Subtract: lambda: custom.Subtract(),
    custom.Divide: lambda: custom.Divide(),
    custom.Concat: lambda: custom.Concat(),
    custom.Outer: lambda: custom.Outer(),
    custom.FloorDivide: lambda: custom.FloorDivide(),
    custom.Norm: lambda: custom.Norm(),
    custom.Exponential: lambda: custom.Exponential(),
    custom.Erf: lambda: custom.Erf(),
    custom.Sqrt: lambda: custom.Sqrt(),
    custom.Maximum: lambda: custom.Maximum(),
    custom.Max: lambda: custom.Max(),
    custom.AMax: lambda: custom.AMax(),
    custom.Minimum: lambda: custom.Minimum(),
    custom.Min: lambda: custom.Min(),
    custom.AMin: lambda: custom.AMin(),
    # custom.Where: lambda: custom.Where(),
    # custom.Greater: lambda: custom.Greater(),
    # custom.Less: lambda: custom.Less(),
    # custom.GreaterEqual: lambda: custom.GreaterEqual(),
    # custom.LessEqual: lambda: custom.LessEqual(),
    # custom.NotEqual: lambda: custom.NotEqual(),
    # custom.Equal: lambda: custom.Equal(),
    custom.Bmm: lambda: custom.Bmm(),
    custom.CumSum: lambda: custom.CumSum(),
    custom.MaskedFill: lambda: custom.MaskedFill(),
    custom.Mean: lambda: custom.Mean(),
    custom.Sum: lambda: custom.Sum(),
    custom.Prod: lambda: custom.Prod(),
    custom.Log: lambda: custom.Log(),
    custom.Abs: lambda: custom.Abs(),
    custom.Neg: lambda: custom.Neg(),
    custom.Argmin: lambda: custom.Argmin(),
    custom.Argmax: lambda: custom.Argmax(),
    custom.ElementwiseCeil: lambda: custom.ElementwiseCeil(),
    custom.ElementwiseFloor: lambda: custom.ElementwiseFloor(),
    custom.Asin: lambda: custom.Asin(),
    custom.Atan: lambda: custom.Atan(),
    custom.Round: lambda: custom.Round(),
    custom.Gather: lambda: custom.Gather(),
    custom.LogicalOr: lambda: custom.LogicalOr(),
    custom.LogicalAnd: lambda: custom.LogicalAnd(),
    custom.LogicalNot: lambda: custom.LogicalNot(),
    custom.Split: lambda: custom.Split(),
    custom.Permute: lambda: custom.Permute(),
    custom.Remainder: lambda: custom.Remainder(),
    custom.IndexSelect: lambda: custom.IndexSelect(),
    custom.Fmod: lambda: custom.Fmod(),
    # custom.NonZero: lambda: custom.NonZero(),
    # custom.TopK: lambda: custom.TopK(),
    custom.Tile: lambda: custom.Tile(),
    custom.Baddbmm: lambda: custom.Baddbmm(),
    custom.Addmm: lambda: custom.Addmm(),
    custom.Square: lambda: custom.Square(),
    custom.Select: lambda: custom.Select(),
    custom.Interpolate: lambda: custom.Interpolate(),
    custom.MaxPool2d: lambda: custom.MaxPool2d(),
    custom.AdaptiveAvgPool2d: lambda: custom.AdaptiveAvgPool2d(),
    custom.BatchNorm: lambda: custom.BatchNorm(),
    custom.BatchNorm: lambda: custom.BatchNorm(),
    custom.BatchNorm: lambda: custom.BatchNorm(),
    custom.BatchNorm: lambda: custom.BatchNorm(),
    custom.BatchNorm: lambda: custom.BatchNorm(),
    custom.BatchNorm: lambda: custom.BatchNorm(),
    custom.GroupNorm: lambda: custom.GroupNorm(),
    custom.Normalize: lambda: custom.Normalize(),
    custom.Pad: lambda: custom.Pad(),
    custom.GridSample: lambda: custom.GridSample(),
    custom.RmsNorm: lambda: custom.RmsNorm([5, 2, 3], [2], 1e-5),
    custom.DynamicConv2d: lambda: custom.DynamicConv2d(),
    custom.Pow: lambda: custom.Pow(),
    custom.CustomSiLU: lambda: custom.CustomSiLU(),
    custom.StridedSlice: lambda: custom.StridedSlice(),
    custom.ChannelShuffle: lambda: custom.ChannelShuffle(2),
    custom.CustomGather: lambda: custom.CustomGather(),
    custom.DepthToSpaceCRDMode: lambda: custom.DepthToSpaceCRDMode([2, 2]),
    custom.DepthToSpaceDCRMode: lambda: custom.DepthToSpaceDCRMode(2),
    # custom.CustomSparseConv3DLayer: lambda: custom.CustomSparseConv3DLayer(),
    # custom.SparseTensorWrapper: lambda: custom.SparseTensorWrapper(),
    # custom.ScatterDense: lambda: custom.ScatterDense(),
    custom.ScatterND: lambda: custom.ScatterND(),
    custom.RoiAlign: lambda: custom.RoiAlign(8, 1.0, -1),
    # custom.NonMaxSuppression: lambda: custom.NonMaxSuppression(),
    custom.GatherNd: lambda: custom.GatherNd(0),
    custom.ScatterElements: lambda: custom.ScatterElements(dim=0),
    custom.OneHot: lambda: custom.OneHot(20, 0.0, 0.9),
    custom.Expand: lambda: custom.Expand(),
    custom.DynamicLinear: lambda: custom.DynamicLinear(),
}

_INPUT_FACTORIES = {
    nn.AdaptiveAvgPool1d: lambda: randn(1, 100),
    nn.AdaptiveAvgPool2d: lambda: randn(1, 10, 10),
    nn.AdaptiveAvgPool3d: lambda: randn(1, 10, 10, 11),
    # nn.AdaptiveLogSoftmaxWithLoss: lambda: ...,
    nn.AdaptiveMaxPool1d: lambda: randn(1, 100),
    nn.AdaptiveMaxPool2d: lambda: randn(1, 10, 10),
    nn.AdaptiveMaxPool3d: lambda: randn(1, 10, 10, 11),
    nn.AlphaDropout: lambda: randn(100),
    nn.AvgPool1d: lambda: randn(1, 100),
    nn.AvgPool2d: lambda: randn(1, 10, 10),
    nn.AvgPool3d: lambda: randn(1, 10, 10, 11),
    nn.BCELoss: lambda: (F.sigmoid(randn(100)), zeros(100)),
    nn.BCEWithLogitsLoss: lambda: (randn(100), zeros(100)),
    nn.BatchNorm1d: lambda: randn(5, 10, 3),
    nn.BatchNorm2d: lambda: randn(5, 10, 3, 2),
    nn.BatchNorm3d: lambda: randn(5, 10, 3, 2, 1),
    nn.Bilinear: lambda: (randn(10, 20), randn(10, 30)),
    nn.CELU: lambda: randn(10, 10),
    nn.CTCLoss: lambda: (
        randn(10, 11, 12).log_softmax(2),
        randint(low=1, high=12, size=(11, 20)),
        full(size=(11,), fill_value=10, dtype=torch.long),
        randint(low=5, high=20, size=(11,), dtype=torch.long),
    ),
    nn.ChannelShuffle: lambda: randn(1, 8, 4, 4),
    nn.ConstantPad1d: lambda: randn(1, 10, 10),
    nn.ConstantPad2d: lambda: randn(1, 10, 10),
    nn.ConstantPad3d: lambda: randn(1, 10, 2, 5),
    nn.Conv1d: lambda: randn(1, 3, 32),
    nn.Conv2d: lambda: randn(1, 3, 16, 16),
    nn.Conv3d: lambda: randn(1, 3, 16, 16, 16),
    nn.ConvTranspose1d: lambda: randn(1, 3, 32),
    nn.ConvTranspose2d: lambda: randn(1, 3, 16, 16),
    nn.ConvTranspose3d: lambda: randn(1, 3, 16, 16, 16),
    nn.CosineEmbeddingLoss: lambda: (
        randn(10, 10),
        zeros(10, 10),
        randn(10).sign().long(),
    ),
    nn.CosineSimilarity: lambda: (randn(10, 10), zeros(10, 10)),
    nn.CrossEntropyLoss: lambda: (randn(10, 10), zeros(10, 10)),
    # nn.CrossMapLRN2d: lambda: ...,
    nn.Dropout: lambda: randn(10, 10),
    nn.Dropout2d: lambda: randn(10, 10),
    nn.Dropout3d: lambda: randn(10, 10),
    nn.ELU: lambda: randn(10, 10),
    nn.Embedding: lambda: randint(100, (10,)),
    nn.EmbeddingBag: lambda: (randint(100, (10,)), arange(10), randn(10)),
    nn.FeatureAlphaDropout: lambda: randn(10, 10),
    nn.Flatten: lambda: randn(10, 10),
    nn.Fold: lambda: randn(1, 12, 12),
    nn.FractionalMaxPool2d: lambda: randn(1, 10, 10),
    nn.FractionalMaxPool3d: lambda: randn(1, 10, 10, 10),
    nn.GELU: lambda: randn(100),
    nn.GLU: lambda: randn(100),
    nn.GRU: lambda: (randn(5, 3, 10), randn(2, 3, 20)),
    nn.GRUCell: lambda: (randn(3, 10), randn(3, 20)),
    nn.GaussianNLLLoss: lambda: (randn(1, 100), zeros(1, 100), ones(1, 100)),
    nn.GroupNorm: lambda: randn(1, 4, 25),
    nn.Hardshrink: lambda: randn(100),
    nn.Hardsigmoid: lambda: randn(100),
    nn.Hardswish: lambda: randn(100),
    nn.Hardtanh: lambda: randn(100),
    nn.HingeEmbeddingLoss: lambda: (randn(10, 10), randn(10).sign().long()),
    nn.HuberLoss: lambda: (randn(10, 10), zeros(10, 10)),
    nn.InstanceNorm1d: lambda: randn(5, 10, 3),
    nn.InstanceNorm2d: lambda: randn(5, 10, 3, 2),
    nn.InstanceNorm3d: lambda: randn(5, 10, 3, 2, 1),
    nn.KLDivLoss: lambda: (
        F.log_softmax(randn(10, 10), dim=1),
        F.softmax(randn(10, 10), dim=1),
    ),
    nn.L1Loss: lambda: (randn(10, 10), zeros(10, 10)),
    nn.LPPool1d: lambda: randn(1, 10, 10),
    nn.LPPool2d: lambda: randn(1, 10, 10, 10),
    nn.LSTM: lambda: (randn(5, 3, 10), (randn(2, 3, 20), randn(2, 3, 20))),
    nn.LSTMCell: lambda: (randn(3, 10), (randn(3, 20), randn(3, 20))),
    nn.LayerNorm: lambda: randn(10, 2, 3, 4),
    nn.LeakyReLU: lambda: randn(100),
    nn.Linear: lambda: randn(10, 10),
    NonDynamicallyQuantizableLinear: lambda: randn(10, 10),
    nn.LocalResponseNorm: lambda: randn(1, 4, 5, 5),
    nn.LogSigmoid: lambda: randn(100),
    nn.LogSoftmax: lambda: randn(100),
    nn.MSELoss: lambda: (randn(10, 10), zeros(10, 10)),
    nn.MarginRankingLoss: lambda: (randn(100), randn(100), randn(100).sign().long()),
    nn.MaxPool1d: lambda: randn(1, 10, 10),
    nn.MaxPool2d: lambda: randn(1, 10, 10, 10),
    nn.MaxPool3d: lambda: randn(1, 1, 10, 10, 10),
    nn.MaxUnpool1d: lambda: nn.MaxPool1d(2, return_indices=True)(randn(1, 10, 10)),
    nn.MaxUnpool2d: lambda: nn.MaxPool2d(2, return_indices=True)(randn(1, 10, 10, 10)),
    nn.MaxUnpool3d: lambda: nn.MaxPool3d(2, return_indices=True)(
        randn(1, 1, 10, 10, 10)
    ),
    nn.Mish: lambda: randn(100),
    nn.MultiLabelMarginLoss: lambda: (randn(10, 10), randint(-1, 10, (10, 10))),
    nn.MultiLabelSoftMarginLoss: lambda: (randn(10, 10), F.one_hot(arange(10))),
    nn.MultiMarginLoss: lambda: (randn(10, 10), randint(0, 10, (10,))),
    # nn.MultiheadAttention: lambda: ...,
    nn.NLLLoss: lambda: (randn(10, 10), randint(10, (10,))),
    nn.NLLLoss2d: lambda: (randn(10, 10), randint(10, (10,))),
    nn.PReLU: lambda: randn(100),
    nn.PairwiseDistance: lambda: (randn(100, 10), randn(100, 10)),
    nn.PixelShuffle: lambda: randn(1, 1, 10, 10),
    # nn.PixelUnshuffle: lambda: ...,
    nn.PoissonNLLLoss: lambda: (randn(100), randn(100)),
    nn.RNN: lambda: (randn(5, 3, 10), randn(2, 3, 20)),
    nn.RNNCell: lambda: (randn(3, 10), randn(3, 20)),
    nn.RReLU: lambda: randn(100),
    nn.ReLU: lambda: randn(100),
    nn.ReLU6: lambda: randn(100),
    nn.ReflectionPad1d: lambda: randn(1, 10, 10),
    nn.ReflectionPad2d: lambda: randn(1, 10, 10),
    nn.ReplicationPad1d: lambda: randn(1, 10, 10),
    nn.ReplicationPad2d: lambda: randn(1, 10, 10),
    nn.ReplicationPad3d: lambda: randn(1, 10, 2, 5),
    nn.SELU: lambda: randn(100),
    nn.SiLU: lambda: randn(100),
    nn.Sigmoid: lambda: randn(100),
    nn.SmoothL1Loss: lambda: (randn(100), zeros(100)),
    nn.SoftMarginLoss: lambda: (randn(100), randn(100).sign().long()),
    nn.Softmax: lambda: randn(100),
    nn.Softmax2d: lambda: randn(1, 4, 25),
    nn.Softmin: lambda: randn(100),
    nn.Softplus: lambda: randn(100),
    nn.Softshrink: lambda: randn(100),
    nn.Softsign: lambda: randn(100),
    nn.SyncBatchNorm: lambda: randn(5, 10, 3, 2),
    nn.Tanh: lambda: randn(100),
    nn.Tanhshrink: lambda: randn(100),
    nn.Threshold: lambda: randn(100),
    # nn.Transformer: lambda: ...,
    # nn.TransformerDecoder: lambda: ...,
    # nn.TransformerDecoderLayer: lambda: ...,
    # nn.TransformerEncoder: lambda: ...,
    # nn.TransformerEncoderLayer: lambda: ...,
    nn.TripletMarginLoss: lambda: (
        randn(100),
        randn(100),
        randn(100),
    ),
    nn.TripletMarginWithDistanceLoss: lambda: (
        randn(100),
        randn(100),
        randn(100),
    ),
    nn.Unflatten: lambda: randn(2, 50),
    nn.Unfold: lambda: randn(2, 5, 3, 4),
    nn.Upsample: lambda: randn(1, 1, 10, 10),
    nn.UpsamplingBilinear2d: lambda: randn(1, 1, 10, 10),
    nn.UpsamplingNearest2d: lambda: randn(1, 1, 10, 10),
    nn.ZeroPad2d: lambda: randn(1, 10, 10),
    nn.ReflectionPad3d: lambda: randn(1, 5, 5, 5),
    nn.Dropout1d: lambda: randn(10, 10),
    nn.CircularPad1d: lambda: randn(1, 10, 10),
    nn.CircularPad2d: lambda: randn(1, 10, 10),
    nn.CircularPad3d: lambda: randn(1, 10, 2, 5),
    nn.ZeroPad1d: lambda: randn(1, 10, 10),
    nn.ZeroPad3d: lambda: randn(1, 10, 2, 5),
    custom.Sin: lambda: randn(100),
    custom.Cos: lambda: randn(100),
    custom.AvgPool2d: lambda: (randn(1, 10, 10), (tensor(2),)),
    custom.Reshape: lambda: (randn(10, 10), (tensor(100), tensor(1))),
    custom.RSqrt: lambda: randn(100).abs(),
    custom.Add: lambda: (randn(100), randn(100)),
    custom.Multiply: lambda: (randn(100), randn(100)),
    custom.Subtract: lambda: (randn(100), randn(100)),
    custom.Divide: lambda: (randn(100), randn(100)),
    custom.Concat: lambda: (randn(1, 100), randn(3, 100)),
    custom.Outer: lambda: (randn(100), randn(50)),
    custom.FloorDivide: lambda: (arange(0, 6, 0.5), tensor(2)),
    custom.Norm: lambda: randn(100),
    custom.Exponential: lambda: randn(100),
    custom.Erf: lambda: randn(100),
    custom.Sqrt: lambda: randn(100).abs(),
    custom.Maximum: lambda: (randn(100), randn(100)),
    custom.Max: lambda: randn(100),
    custom.AMax: lambda: randn(100),
    custom.Minimum: lambda: (randn(100), randn(100)),
    custom.Min: lambda: randn(100),
    custom.AMin: lambda: randn(100),
    # custom.Where: lambda: ...,
    # custom.Greater: lambda: ...,
    # custom.Less: lambda: ...,
    # custom.GreaterEqual: lambda: ...,
    # custom.LessEqual: lambda: ...,
    # custom.NotEqual: lambda: ...,
    # custom.Equal: lambda: ...,
    custom.Bmm: lambda: (randn(1, 100, 100), randn(1, 100, 100)),
    custom.CumSum: lambda: (randn(10, 100), tensor(0)),
    custom.MaskedFill: lambda: (
        randn(10, 10),
        randint(0, 1, (10, 10)).bool(),
        tensor(0.5),
    ),
    custom.Mean: lambda: randn(10, 10),
    custom.Sum: lambda: randn(10, 10),
    custom.Prod: lambda: randint(1, 10, (10,)) / 5,
    custom.Log: lambda: randint(1, 1000, (10, 10)),
    custom.Abs: lambda: randn(100),
    custom.Neg: lambda: randn(100),
    custom.Argmin: lambda: randn(100),
    custom.Argmax: lambda: randn(100),
    custom.ElementwiseCeil: lambda: randn(100),
    custom.ElementwiseFloor: lambda: randn(100),
    custom.Asin: lambda: arange(-1, 1, 0.01),
    custom.Atan: lambda: randn(100),
    custom.Round: lambda: randn(100),
    custom.Gather: lambda: (randn(10, 10), tensor(1), randint(0, 9, (4, 4))),
    custom.LogicalOr: lambda: (
        torch.randint(0, 2, (10,), dtype=torch.float32),
        torch.randint(0, 2, (10,), dtype=torch.float32),
    ),
    custom.LogicalAnd: lambda: (
        torch.randint(0, 2, (10,), dtype=torch.float32),
        torch.randint(0, 2, (10,), dtype=torch.float32),
    ),
    # #     ),
    custom.LogicalNot: lambda: torch.randint(0, 2, (10,), dtype=torch.float32),
    custom.Split: lambda: (randn(10, 10), tensor(2)),
    custom.Permute: lambda: (randn(10, 10, 1), [tensor(1), tensor(0), tensor(2)]),
    custom.Remainder: lambda: (randn(100), tensor(1.0)),
    custom.IndexSelect: lambda: (randn(10, 10), tensor(1), randint(0, 9, (4,))),
    custom.Fmod: lambda: (randn(10), tensor(1.0)),
    # custom.NonZero: lambda: ...,
    # custom.TopK: lambda: ...,
    custom.Tile: lambda: (randn(10), [tensor(3)]),
    custom.Baddbmm: lambda: (
        randn(1, 100, 100),
        randn(1, 100, 100),
        randn(1, 100, 100),
    ),
    custom.Addmm: lambda: (randn(100, 100), randn(100, 100), randn(100, 100)),
    custom.Square: lambda: randn(10),
    custom.Select: lambda: (randn(10, 10), tensor(0), tensor(5)),
    custom.Interpolate: lambda: (randn(3, 10, 10), torch.tensor(20)),
    custom.MaxPool2d: lambda: (randn(1, 10, 10, 10), [tensor(3), tensor(3)]),
    custom.AdaptiveAvgPool2d: lambda: (randn(1, 10, 10), tensor([3, 3])),
    custom.BatchNorm: lambda: (
        randn(5, 10),
        zeros(10).requires_grad_(),
        ones(10).requires_grad_(),
    ),
    custom.BatchNorm: lambda: (
        randn(5, 10, 3, 2),
        zeros(10).requires_grad_(),
        ones(10).requires_grad_(),
    ),
    custom.BatchNorm: lambda: (
        randn(5, 10, 3, 2, 5),
        zeros(10).requires_grad_(),
        ones(10).requires_grad_(),
    ),
    custom.BatchNorm: lambda: (randn(5, 10), zeros(10), ones(10)),
    custom.BatchNorm: lambda: (randn(5, 10, 3, 2), zeros(10), ones(10)),
    custom.BatchNorm: lambda: (randn(5, 10, 3, 2, 5), zeros(10), ones(10)),
    custom.GroupNorm: lambda: (randn(20, 6, 10, 10), tensor(6)),
    custom.Normalize: lambda: randn(100, 100),
    custom.Pad: lambda: (randn(10, 10), [tensor(1), tensor(1)]),
    custom.GridSample: lambda: (randn(1, 3, 30, 30), randn(1, 3, 5, 2)),
    custom.RmsNorm: lambda: (randn(5, 2, 3)),
    custom.DynamicConv2d: lambda: (
        torch.randn(1, 3, 10, 10),
        torch.randn(3, 3, 3, 3),
        torch.randn(3),
    ),
    custom.Pow: lambda: (arange(10.0), tensor(2.0)),
    custom.CustomSiLU: lambda: randn(100),
    custom.StridedSlice: lambda: (randn(10), [[tensor(0), tensor(6), tensor(2)]]),
    custom.ChannelShuffle: lambda: randn(1, 8, 4, 4),
    custom.CustomGather: lambda: (randn(10, 10), randint(0, 9, (4, 4))),
    custom.DepthToSpaceCRDMode: lambda: randn(1, 8, 10, 10),
    custom.DepthToSpaceDCRMode: lambda: randn(1, 8, 10, 10),
    # custom.CustomSparseConv3DLayer: lambda: ...,
    # custom.SparseTensorWrapper: lambda: ...,
    # custom.ScatterDense: lambda: ...,
    custom.ScatterND: lambda: (randn(10), randint(0, 9, (4, 1)), randn(4)),
    custom.RoiAlign: lambda: (
        torch.randn(1, 3, 10, 10),
        torch.tensor([[1.0, 1.0, 6.0, 6.0]]),
        torch.tensor([0]),
    ),
    # custom.NonMaxSuppression: lambda: ...,
    custom.GatherNd: lambda: (randn(5, 5), randint(0, 5, (2,))),
    custom.ScatterElements: lambda: (
        randn(3, 4, 4),
        randint(0, 3, (3, 4, 4)),
        randn(3, 4, 4),
    ),
    custom.OneHot: lambda: randint(0, 20, (10,)),
    custom.Expand: lambda: (randn(10), [tensor(5), tensor(10)]),
    custom.DynamicLinear: lambda: (
        torch.randn(10, 10),
        torch.randn(10, 10),
        torch.randn(10),
    ),
}


@pytest.mark.parametrize("module_type", _MODULE_FACTORIES.keys())
def test_default_kernels(module_type):
    module_factory = _MODULE_FACTORIES[module_type]
    input_factory = _INPUT_FACTORIES[module_type]

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
        assert torch.equal(out_, fout_), type(module)
        assert torch.all(out_.isfinite()), type(module)

    """
    When: Trace a quantized modules with torch.jit.trace
    Then: 1) Tracing shouldn't fail
          2) The traced module should produce the same output as the original module
    """
    if not (
        # torch 2.13 fails to handle MaxUnpool during jit tracing.
        # See https://github.com/pytorch/pytorch/issues/189298
        version.parse(torch.__version__) >= version.parse("2.13.0")
        and isinstance(module, (nn.MaxUnpool1d, nn.MaxUnpool2d, nn.MaxUnpool3d))
    ):
        traced = torch.jit.trace(qmodule, inputs)
        torch.manual_seed(0)
        tout = traced(*inputs)

        for out_, tout_ in zip(tree_flatten(out)[0], tree_flatten(tout)[0]):
            assert torch.equal(out_, tout_), type(module)

    if version.parse(torch.__version__) >= version.parse("2.8.0"):
        """
        When: Export a quantized modules with torch.export.export
        Then: 1) Tracing shouldn't fail
              2) The exported module should produce the same output as the original module
        """
        is_dynamo_traceable, _ = qmodule._is_dynamo_traceable()
        if not is_dynamo_traceable:
            with pytest.raises(RuntimeError):
                _ = aimet_torch.v2.experimental.export.export(qmodule, args=inputs)
            return

        ep = aimet_torch.v2.experimental.export.export(qmodule, args=inputs)
        torch.manual_seed(0)
        ep_out = ep.module()(*inputs)

        for out_, ep_out_ in zip(tree_flatten(out)[0], tree_flatten(ep_out)[0]):
            assert torch.allclose(out_, ep_out_), type(qmodule)

    if version.parse(torch.__version__) >= version.parse("2.12.0"):
        """
        When: Compile a quantized module with torch.compile(fullgraph=True)
        Then: The compiled module should produce the same output as the original module
        """
        if isinstance(
            qmodule, (nn.FractionalMaxPool2d, nn.FractionalMaxPool3d, custom.BatchNorm)
        ):
            pytest.skip(
                reason="These FP modules can't be compiled without graph breaks"
            )

        if isinstance(
            qmodule,
            (
                custom.FloorDivide,
                custom.Tile,
                custom.Pad,
            ),
        ):
            pytest.skip(
                reason="TODO: Full-graph compile for these modules are not implemented yet"
            )

        compiled_qmodule = torch.compile(qmodule, fullgraph=True)
        compiled_out = compiled_qmodule(*inputs)

        if isinstance(
            qmodule,
            (
                nn.AlphaDropout,
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d,
                nn.FeatureAlphaDropout,
                nn.RReLU,
            ),
        ):
            # These modules involve randomness and doesn't guarantee same
            # output when compiled even under same random seed.
            return

        for out_, compiled_out_ in zip(
            tree_flatten(out)[0], tree_flatten(compiled_out)[0]
        ):
            assert torch.equal(out_, compiled_out_), type(module)


@pytest.mark.parametrize(
    "module_cls",
    [
        transformers.pytorch_utils.Conv1D,
        # transformers.models.llama.modeling_llama.LlamaRotaryEmbedding, # requires latest transformer
        # transformers.models.llama.modeling_llama.LlamaRMSNorm, # requires latest transformer
        torch.nn.Linear,
        custom.Multiply,
        custom.MatMul,
    ],
)
def test_code_example(module_cls):
    """
    Given: A torch.nn.Module class defined with return annotation
    When: Generate code example with _generate_code_example
    Then: The generated code should be parseable by python interpreter
    """
    src_code = UnknownModuleError(module_cls, QuantizationMixin).generate_code_example()
    try:
        ast.parse(src_code)
    except SyntaxError as e:
        err = SyntaxError(f"The following code example is ill-formed:\n\n{src_code}")
        raise err from e


@torch.no_grad()
def test_subclassing():
    """
    When: Define a trivial subclass of an existing quantized module
    Then: The subclass should work normally

    NOTE: This test was added to prevent malicious OOP/MRO issues
          which caused an infinite recursion error in the child classes of
          qunatized modules.
    """

    # Trivial subclass. Should behave same as parent
    class MyQuantizedLinear(QuantizedLinear): ...

    qlinear = QuantizedLinear(10, 10)
    my_qlinear = MyQuantizedLinear(
        10, 10
    )  # Shouldn't run into infinite recursion error
    x = torch.randn(10, 10)

    my_qlinear.weight.copy_(qlinear.weight)
    my_qlinear.bias.copy_(qlinear.bias)

    assert torch.equal(qlinear(x), my_qlinear(x))


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor(2),  # scalar index
        torch.tensor([0, 1, 3, 5, 7, 9]),  # 1D indices
        torch.tensor([[1, 3, 5], [8, 6, 4]]),  # 2D indices
    ],
)
@pytest.mark.parametrize(
    "scale_shape,  block_size",
    [
        ((), None),  # per-tensor
        ((10, 1), None),  # per-channel with axis=0
        ((1, 10), None),  # per-channel with axis=1
        ((10,), None),  # per-channel with axis=1
        ((10,), (-1,)),  # per-channel with axis=1
        ((10, 2), (-1, 5)),  # per-block with channel_axis=0, block_axis=1
        ((2, 10), (5, -1)),  # per-block with channel_axis=1, block_axis=0
    ],
)
def test_qembedding_output_encoding(scale_shape, block_size, indices):
    """
    Given: QuantizedEmbedding with weight-only quantization
    """
    qembedding = QuantizedEmbedding(10, 10)
    weight_qtzr = QuantizeDequantize(
        scale_shape, qmin=-128, qmax=127, symmetric=True, block_size=block_size
    )
    qembedding.param_quantizers["weight"] = weight_qtzr
    qembedding.compute_param_encodings()

    """
    When: Run forward
    Then: Output should inherit the weight encodings
    """
    qout = qembedding(indices)
    qweight = weight_qtzr(qembedding.weight)

    assert isinstance(qout, DequantizedTensor)
    assert torch.equal(qout, qweight[indices])
    assert torch.equal(qout.quantize(), qweight.quantize()[indices])


@pytest.mark.parametrize(
    "qmodule_factory,                     input_shape",
    [
        (lambda: QuantizedConv2d(16, 16, 3), (1, 16, 3, 3)),
        (lambda: QConvTranspose2d(16, 16, 3), (1, 16, 3, 3)),
        (lambda: QuantizedLinear(16, 16), (1, 9, 16)),
    ],
)
def test_create_int32_bias_quantizer_trivial(qmodule_factory, input_shape):
    """
    Given: Quantized module without input or weight quantizer
    """
    qmodule = qmodule_factory()
    input = torch.randn(input_shape)
    qmodule.input_quantizers[0] = None
    qmodule.param_quantizers["weight"] = None

    """
    When: Call _create_int32_bias_quantizer
    Then: Bias encoding should be calibrated only based on the values bias,
          and hence shouldn't incur any quantization noise
    """
    qmodule._create_int32_bias_quantizer((input,), None)
    bias_qtzr = qmodule.param_quantizers["bias"]
    assert torch.allclose(bias_qtzr(qmodule.bias), qmodule.bias)


@pytest.mark.parametrize(
    "qmodule_factory,                     scale_shape,      block_size,      block_grouping,  input_shape",
    [
        (lambda: QuantizedConv1d(16, 16, 3), (), None, None, (16, 3)),
        (lambda: QuantizedConv1d(16, 16, 3), (16, 1, 1), None, None, (16, 3)),
        (lambda: QuantizedConv1d(16, 16, 3), (16, 4, 1), (1, 4, 3), None, (16, 3)),
        (lambda: QuantizedConv1d(16, 16, 3), (16, 4, 1), (1, 4, 3), (1, 4, 1), (16, 3)),
        (lambda: QuantizedConv2d(16, 16, 3), (), None, None, (16, 3, 3)),
        (lambda: QuantizedConv2d(16, 16, 3), (16, 1, 1, 1), None, None, (16, 3, 3)),
        (
            lambda: QuantizedConv2d(16, 16, 3),
            (16, 4, 1, 1),
            (1, 4, 3, 3),
            None,
            (16, 3, 3),
        ),
        (
            lambda: QuantizedConv2d(16, 16, 3),
            (16, 4, 1, 1),
            (1, 4, 3, 3),
            (1, 4, 1, 1),
            (16, 3, 3),
        ),
        (lambda: QuantizedConv3d(16, 16, 3), (), None, None, (16, 3, 3, 3)),
        (
            lambda: QuantizedConv3d(16, 16, 3),
            (16, 1, 1, 1, 1),
            None,
            None,
            (16, 3, 3, 3),
        ),
        (
            lambda: QuantizedConv3d(16, 16, 3),
            (16, 4, 1, 1, 1),
            (1, 4, 3, 3, 3),
            None,
            (16, 3, 3, 3),
        ),
        (
            lambda: QuantizedConv3d(16, 16, 3),
            (16, 4, 1, 1, 1),
            (1, 4, 3, 3, 3),
            (1, 4, 1, 1, 1),
            (16, 3, 3, 3),
        ),
        (lambda: QConvTranspose1d(16, 16, 3), (), None, None, (16, 3)),
        (lambda: QConvTranspose1d(16, 16, 3), (1, 16, 1), None, None, (16, 3)),
        (lambda: QConvTranspose1d(16, 16, 3), (4, 16, 1), (4, 1, 3), None, (16, 3)),
        (
            lambda: QConvTranspose1d(16, 16, 3),
            (4, 16, 1),
            (4, 1, 3),
            (4, 1, 1),
            (16, 3),
        ),
        (lambda: QConvTranspose2d(16, 16, 3), (), None, None, (16, 3, 3)),
        (lambda: QConvTranspose2d(16, 16, 3), (1, 16, 1, 1), None, None, (16, 3, 3)),
        (
            lambda: QConvTranspose2d(16, 16, 3),
            (4, 16, 1, 1),
            (4, 1, 3, 3),
            None,
            (16, 3, 3),
        ),
        (
            lambda: QConvTranspose2d(16, 16, 3),
            (4, 16, 1, 1),
            (4, 1, 3, 3),
            (4, 1, 1, 1),
            (16, 3, 3),
        ),
        (lambda: QConvTranspose3d(16, 16, 3), (), None, None, (16, 3, 3, 3)),
        (
            lambda: QConvTranspose3d(16, 16, 3),
            (1, 16, 1, 1, 1),
            None,
            None,
            (16, 3, 3, 3),
        ),
        (
            lambda: QConvTranspose3d(16, 16, 3),
            (4, 16, 1, 1, 1),
            (4, 1, 3, 3, 3),
            None,
            (16, 3, 3, 3),
        ),
        (
            lambda: QConvTranspose3d(16, 16, 3),
            (4, 16, 1, 1, 1),
            (4, 1, 3, 3, 3),
            (4, 1, 1, 1, 1),
            (16, 3, 3, 3),
        ),
        (lambda: QuantizedLinear(16, 16), (), None, None, (9, 16)),
        (lambda: QuantizedLinear(16, 16), (16, 1), None, None, (9, 16)),
        (lambda: QuantizedLinear(16, 16), (16, 4), (1, 4), None, (9, 16)),
        (lambda: QuantizedLinear(16, 16), (16, 4), (1, 4), (1, 4), (9, 16)),
    ],
)
def test_create_int32_bias_quantizer_analytic(
    qmodule_factory, scale_shape, block_size, block_grouping, input_shape
):
    """
    Given: Quantized module with input and weight quantizer
    """
    qmodule = qmodule_factory()

    if block_grouping:
        weight_qtzr = GroupedBlockQuantizeDequantize(
            scale_shape,
            bitwidth=4,
            decompressed_bw=8,
            symmetric=True,
            block_size=block_size,
            block_grouping=block_grouping,
        )
    else:
        weight_qtzr = QuantizeDequantize(
            scale_shape, qmin=-128, qmax=127, symmetric=True, block_size=block_size
        )

    input = torch.randn(input_shape)
    qmodule.input_quantizers[0] = QuantizeDequantize(
        (), qmin=0, qmax=255, symmetric=False
    )
    qmodule.param_quantizers["weight"] = copy.deepcopy(weight_qtzr)

    with qmodule.compute_encodings():
        _ = qmodule(input)

    """
    When: Call _create_int32_bias_quantizer
    Then: Bias encoding should be derived analytically from input and weight encodings, such that
          bias_scale = input_scale * weight_scale
    """
    qmodule._create_int32_bias_quantizer((input,), None)
    input_qtzr = qmodule.input_quantizers[0]
    weight_qtzr = qmodule.param_quantizers["weight"]
    bias_qtzr = qmodule.param_quantizers["bias"]

    input_scale = input_qtzr.get_scale()
    if block_grouping:
        weight_scale = weight_qtzr.get_per_channel_scale()
    else:
        weight_scale = weight_qtzr.get_scale()

    expected_bias_scale = input_scale * weight_scale
    if block_size is None:
        expected_bias_scale = expected_bias_scale.squeeze()
    else:
        channel_axis = 1 if isinstance(qmodule, nn.modules.conv._ConvTransposeNd) else 0
        non_channel_axes = [
            axis for axis, _ in enumerate(qmodule.weight.shape) if axis != channel_axis
        ]
        expected_bias_scale = expected_bias_scale.amax(dim=non_channel_axes)

    assert bias_qtzr.get_scale().shape == expected_bias_scale.shape
    assert torch.allclose(bias_qtzr.get_scale(), expected_bias_scale)

    """
    Given:
      * Quantized module with weight quantizer but without input quantizer
      * input is a DequantizedTensor
    """
    qmodule = qmodule_factory()

    input = torch.randn(input_shape).as_subclass(DequantizedTensor)
    input.encoding = AffineEncoding(
        scale=(input.max() - input.min()) / 255,
        offset=torch.zeros(()),
        qmin=0,
        qmax=255,
        symmetry=False,
    )
    qmodule.input_quantizers[0] = None
    qmodule.param_quantizers["weight"] = copy.deepcopy(weight_qtzr)

    with qmodule.compute_encodings():
        _ = qmodule(input)

    """
    When: Call _create_int32_bias_quantizer
    Then: Bias encoding should be derived analytically from input and weight encodings, such that
          bias_scale = input_scale * weight_scale
    """
    qmodule._create_int32_bias_quantizer((input,), None)
    weight_qtzr = qmodule.param_quantizers["weight"]
    bias_qtzr = qmodule.param_quantizers["bias"]

    input_scale = input.encoding.scale
    if block_grouping:
        weight_scale = weight_qtzr.get_per_channel_scale()
    else:
        weight_scale = weight_qtzr.get_scale()

    expected_bias_scale = input_scale * weight_scale
    if block_size is None:
        expected_bias_scale = expected_bias_scale.flatten()
    else:
        channel_axis = 1 if isinstance(qmodule, nn.modules.conv._ConvTransposeNd) else 0
        non_channel_axes = [
            axis for axis, _ in enumerate(qmodule.weight.shape) if axis != channel_axis
        ]
        expected_bias_scale = expected_bias_scale.amax(dim=non_channel_axes)

    assert torch.allclose(bias_qtzr.get_scale(), expected_bias_scale)


@pytest.mark.parametrize(
    "qmodule_factory,                                 scale_shape,      block_size,          input_shape",
    [
        (lambda: QuantizedConv1d(16, 16, 3), (1, 16, 1), None, (1, 16, 3)),
        (lambda: QuantizedConv1d(16, 16, 3), (4, 16, 1), (4, -1, -1), (1, 16, 3)),
        (lambda: QuantizedConv2d(16, 16, 3), (1, 16, 1, 1), None, (1, 16, 3, 3)),
        (
            lambda: QuantizedConv2d(16, 16, 3),
            (4, 16, 1, 1),
            (4, -1, -1, -1),
            (1, 16, 3, 3),
        ),
        (lambda: QuantizedConv3d(16, 16, 3), (1, 16, 1, 1, 1), None, (1, 16, 3, 3, 3)),
        (
            lambda: QuantizedConv3d(16, 16, 3),
            (4, 16, 1, 1, 1),
            (4, -1, -1, -1, -1),
            (1, 16, 3, 3, 3),
        ),
        (lambda: QConvTranspose1d(16, 16, 3), (16, 1, 1), None, (1, 16, 3)),
        (lambda: QConvTranspose1d(16, 16, 3), (16, 4, 1), (-1, 4, -1), (1, 16, 3)),
        (lambda: QConvTranspose2d(16, 16, 3), (16, 1, 1, 1), None, (1, 16, 3, 3)),
        (
            lambda: QConvTranspose2d(16, 16, 3),
            (16, 4, 1, 1),
            (-1, 4, -1, -1),
            (1, 16, 3, 3),
        ),
        (lambda: QConvTranspose3d(16, 16, 3), (16, 1, 1, 1, 1), None, (1, 16, 3, 3, 3)),
        (
            lambda: QConvTranspose3d(16, 16, 3),
            (16, 4, 1, 1, 1),
            (-1, 4, -1, -1, -1),
            (1, 16, 3, 3, 3),
        ),
        (lambda: QuantizedLinear(16, 16), (1, 16), None, (1, 9, 16)),
        (lambda: QuantizedLinear(16, 16), (4, 16), (4, -1), (1, 9, 16)),
        (lambda: QuantizedLayerNorm(9), (), None, (1, 4, 9)),
        (lambda: QuantizedLayerNorm(9), (9,), None, (1, 4, 9)),
        (lambda: QuantizedGroupNorm(3, 9), (), None, (1, 9, 4)),
        (lambda: QuantizedGroupNorm(3, 9), (9,), None, (1, 9, 4)),
        (lambda: QuantizedInstanceNorm1d(9, affine=True), (), None, (1, 9, 4)),
        (lambda: QuantizedInstanceNorm1d(9, affine=True), (9,), None, (1, 9, 4)),
        (lambda: QuantizedInstanceNorm2d(9, affine=True), (), None, (1, 9, 4, 4)),
        (lambda: QuantizedInstanceNorm2d(9, affine=True), (9,), None, (1, 9, 4, 4)),
        (lambda: QuantizedInstanceNorm3d(9, affine=True), (), None, (1, 9, 4, 4, 4)),
        (lambda: QuantizedInstanceNorm3d(9, affine=True), (9,), None, (1, 9, 4, 4, 4)),
    ],
)
def test_create_int32_bias_quantizer_statistical(
    qmodule_factory, scale_shape, block_size, input_shape
):
    """
    Given: Quantized module whose bias encodings should NOT be derived from input and weight encodings.
           Notable among them are:

           - nn.ConvNd with channel_axis != 0
           - nn.ConvTransposeNd with channel_axis != 1
           - nn.Linear with channel_axis != 0
           - nn.GroupNorm
           - nn.LayerNorm
           - nn.InstanceNorm
    """
    qmodule = qmodule_factory()

    input = torch.randn(input_shape)
    qmodule.input_quantizers[0] = QuantizeDequantize(
        (), qmin=0, qmax=255, symmetric=False
    )
    qmodule.param_quantizers["weight"] = QuantizeDequantize(
        scale_shape, qmin=-128, qmax=127, symmetric=True, block_size=block_size
    )

    with qmodule.compute_encodings():
        _ = qmodule(input)

    """
    When: Call _create_int32_bias_quantizer
    Then: Bias encoding should be calibrated statistically based on the values of bias,
          and hence shouldn't incur any quantization noise
    """
    qmodule._create_int32_bias_quantizer((input,), None)
    bias_qtzr = qmodule.param_quantizers["bias"]
    assert torch.allclose(bias_qtzr(qmodule.bias), qmodule.bias)


@pytest.mark.cuda
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("requires_grad", [True, False])
def test_fold_param_quantizers(device, requires_grad):
    """
    Given: QuantizedLinear with affine weight quantizer
    """
    qlinear = QuantizedLinear(10, 10).to(device)
    qlinear.weight.requires_grad_(requires_grad)
    weight_qtzr = QuantizeDequantize(shape=(10, 1), qmin=-8, qmax=7, symmetric=True).to(
        device
    )
    qlinear.param_quantizers["weight"] = weight_qtzr
    original_weight = qlinear.weight.clone()
    original_bias = qlinear.bias.clone()

    """
    When: Call _fold_param_quantizers
    """
    qlinear._fold_param_quantizers()

    """
    Then:
      1. Weight quantizer should be removed
      2. Weight should be overwritten with a pre-quantized weight,
         which is a DequantizedTensor with AffineEncoding
      3. Other parameters (bias) shouldn't be affected
    """
    assert qlinear.param_quantizers["weight"] is None

    assert qlinear.weight.device == original_weight.device
    assert qlinear.weight.requires_grad == original_weight.requires_grad
    assert isinstance(qlinear.weight, DequantizedTensor)
    assert isinstance(qlinear.weight, torch.nn.Parameter)
    assert isinstance(qlinear.weight.encoding, AffineEncoding)
    assert torch.equal(qlinear.weight.encoding.scale, weight_qtzr.get_scale())
    assert torch.equal(qlinear.weight.encoding.offset, weight_qtzr.get_offset())
    assert torch.equal(qlinear.weight, weight_qtzr(original_weight))

    assert isinstance(qlinear.bias, torch.Tensor)
    assert isinstance(qlinear.bias, torch.nn.Parameter)
    assert torch.equal(qlinear.bias, original_bias)


def test_ignore():
    """
    When: Call QuantizationMixin.ignore
    Then: Unknown modules should be ignored during quantization
    """

    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x**2

    QuantizationMixin.ignore(MyModule)

    model = torch.nn.Sequential(MyModule())
    sim = aimet_torch.QuantizationSimModel(
        model, dummy_input=torch.randn(1, 3, 224, 224)
    )
    assert type(sim.model[0]) == MyModule

    """
    When: Call QuantizationMixin.ignore_unknown_modules
    Then: Unknown modules should be ignored during quantization
    """

    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x**2

    orig = QuantizationMixin._ignore_unknown_modules
    try:
        QuantizationMixin.ignore_unknown_modules(True)

        model = torch.nn.Sequential(MyModule())
        sim = aimet_torch.QuantizationSimModel(
            model, dummy_input=torch.randn(1, 3, 224, 224)
        )
        assert type(sim.model[0]) == MyModule
    finally:
        QuantizationMixin.ignore_unknown_modules(orig)

    """
    When: Ignored layer is followed by quantized layer
    Then: The following layer's input quantizer should be enabled
    """

    @QuantizationMixin.ignore
    class Preprocessing(torch.nn.Module):
        """
        Preprocessing layer that is NOT intended to run on NPU
        """

        def __init__(self):
            super().__init__()
            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
            )
            self.register_buffer(
                "stdev",
                torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
            )

        def forward(self, x):
            return (x - self.mean) / self.stdev

    model = torch.nn.Sequential(
        Preprocessing(),
        torch.nn.Linear(10, 10),
    )
    sim = aimet_torch.QuantizationSimModel(model, torch.randn(10, 10))
    assert isinstance(sim.model[1].input_quantizers[0], QuantizeDequantize)

    """
    When: Call ignore on module whose Quantized- definition was already registered
    Then: The module type should be excluded from quantization regardless
    """

    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x**2

    @QuantizationMixin.implements(MyModule)
    class QuantizedMyModule(QuantizationMixin, MyModule):
        def forward(self, x):
            return super().forward(x)

    QuantizationMixin.ignore(MyModule)

    model = torch.nn.Sequential(MyModule())
    sim = aimet_torch.QuantizationSimModel(
        model, dummy_input=torch.randn(1, 3, 224, 224)
    )
    assert type(sim.model[0]) == MyModule

    """
    When: Register Quantized- definition of a module that was already excluded by .ignore
    Then: The module type should be included for quantization regardless
    """

    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x**2

    QuantizationMixin.ignore(MyModule)

    @QuantizationMixin.implements(MyModule)
    class QuantizedMyModule(QuantizationMixin, MyModule):
        def forward(self, x):
            return super().forward(x)

    model = torch.nn.Sequential(MyModule())
    sim = aimet_torch.QuantizationSimModel(
        model, dummy_input=torch.randn(1, 3, 224, 224)
    )
    assert type(sim.model[0]) == QuantizedMyModule


def test_patch_quantized_param_grad():
    """
    When: Patch quantized parameters with _patch_quantized_parameters and _patch_dequantized_parameters
    Then: The gradients should be able to flow through the quantizers
    """

    class MyLinear(torch.nn.Linear): ...

    @QuantizationMixin.implements(MyLinear)
    class QuantizedMyLinear(QuantizationMixin, MyLinear):
        def forward(self, input):
            with (
                self._patch_quantized_parameters(),
                self._patch_dequantized_parameters(),
            ):
                return super().forward(input)

    qlinear = QuantizedMyLinear(10, 10)
    weight_qtzr = QuantizeDequantize(shape=(10, 1), qmin=-128, qmax=127, symmetric=True)

    with torch.no_grad():
        weight_qtzr.min.copy_(-1)
        weight_qtzr.max.copy_(1)

    qlinear.param_quantizers["weight"] = weight_qtzr

    x = torch.randn(10, 10)

    qlinear(x).sum().backward()

    assert qlinear.weight.grad is not None
    assert qlinear.param_quantizers["weight"].min.grad is not None
    assert qlinear.param_quantizers["weight"].max.grad is not None


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("add_bias_kv", [False, True])
@pytest.mark.parametrize("add_zero_attn", [False, True])
@pytest.mark.parametrize("kdim", [None, 4])
@pytest.mark.parametrize("vdim", [None, 4])
@pytest.mark.parametrize("batch_dim", [0, 1, None])
def test_qmha_forward(
    dtype: torch.dtype,
    dropout: float,
    bias: bool,
    add_bias_kv: bool,
    add_zero_attn: bool,
    kdim: Optional[int],
    vdim: Optional[int],
    batch_dim: int,
):
    embed_dim = 8
    num_heads = 2

    mha = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        dropout=dropout,
        bias=bias,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_dim == 0,
    ).to(dtype)

    if mha.in_proj_bias is not None:
        mha.in_proj_bias.copy_(torch.randint_like(mha.in_proj_bias, 0, 10) * 0.01)
        mha.out_proj.bias.copy_(torch.randint_like(mha.out_proj.bias, 0, 10) * 0.01)

    qmha = QuantizationMixin.from_module(mha)

    N = 2
    L = 4

    if batch_dim == 0:
        query = torch.randn(N, L, embed_dim, dtype=dtype)
        key = torch.randn(N, L, kdim or embed_dim, dtype=dtype)
        value = torch.randn(N, L, vdim or embed_dim, dtype=dtype)
        kpm = torch.tensor(
            [
                [False, False, True, False],
                [False, False, False, False],
            ]
        )
    elif batch_dim == 1:
        query = torch.randn(L, N, embed_dim, dtype=dtype)
        key = torch.randn(L, N, kdim or embed_dim, dtype=dtype)
        value = torch.randn(L, N, vdim or embed_dim, dtype=dtype)
        kpm = torch.tensor(
            [
                [False, False, True, False],
                [False, False, False, False],
            ]
        )
    else:
        query = torch.randn(L, embed_dim, dtype=dtype)
        key = torch.randn(L, kdim or embed_dim, dtype=dtype)
        value = torch.randn(L, vdim or embed_dim, dtype=dtype)
        kpm = torch.tensor([False, False, True, False])

    ones = torch.ones(L, L, dtype=dtype)
    lower = torch.tril(ones, diagonal=-1)
    upper = torch.triu(ones, diagonal=1)

    atol = torch.finfo(dtype).eps
    rtol = 1e-3 if dtype == torch.float16 else 1e-5

    for (
        is_training,
        key_padding_mask,
        need_weights,
        attn_mask,
        average_attn_weights,
        is_causal,
    ) in itertools.product(
        [True, False],
        [kpm, None],
        [False, True],
        [None, upper.bool(), lower.bool(), upper * -10, lower * -10],
        [True, False],
        [True, False],
    ):
        if is_causal and attn_mask is None:
            # Unsupported by nn.MultiheadAttention
            continue

        mha.train(is_training)
        qmha.train(is_training)

        # Set seed to use identical dropout mask
        torch.manual_seed(0)
        out, attn = mha(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        # Set seed to use identical dropout mask
        torch.manual_seed(0)
        qout, qattn = qmha(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        assert out.shape == qout.shape
        assert out.dtype == qout.dtype
        assert torch.allclose(out, qout, equal_nan=True, rtol=rtol, atol=atol)

        if attn is None:
            assert qattn is None
        else:
            assert attn.shape == qattn.shape
            assert attn.dtype == qattn.dtype
            assert torch.allclose(attn, qattn, equal_nan=True, rtol=rtol, atol=atol)


@torch.no_grad()
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("add_bias_kv", [False, True])
@pytest.mark.parametrize("add_zero_attn", [False, True])
@pytest.mark.parametrize("kdim", [None, 4])
@pytest.mark.parametrize("vdim", [None, 4])
@pytest.mark.parametrize("batch_dim", [0, 1])
def test_qmha_error(
    bias: bool,
    add_bias_kv: bool,
    add_zero_attn: bool,
    kdim: Optional[int],
    vdim: Optional[int],
    batch_dim: int,
):
    """
    When: Run QuantizedMultiheadAttention.forward throws error
    Then: nn.MultiheadAttention.forward should also throw same type of error
    """

    embed_dim = 8
    num_heads = 2

    mha = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_dim == 0,
    )

    if mha.in_proj_bias is not None:
        mha.in_proj_bias.copy_(torch.randint_like(mha.in_proj_bias, 0, 10) * 0.01)
        mha.out_proj.bias.copy_(torch.randint_like(mha.out_proj.bias, 0, 10) * 0.01)

    qmha = QuantizationMixin.from_module(mha)

    N = 1
    L = 4

    if batch_dim == 0:
        query = torch.randn(N, L, embed_dim)
        key = torch.randn(N, L, kdim or embed_dim)
        value = torch.randn(N, L, vdim or embed_dim)
    else:
        query = torch.randn(L, N, embed_dim)
        key = torch.randn(L, N, kdim or embed_dim)
        value = torch.randn(L, N, vdim or embed_dim)

    """
    When: Call with attn_mask=None, is_causal=True
    Then: Both nn.MultiheadAttention and QuantizedMultiheadAttention should throw error
    """
    with pytest.raises(
        RuntimeError, match="Need attn_mask if specifying the is_causal hint"
    ):
        _ = mha(query, key, value, attn_mask=None, is_causal=True)

    with pytest.raises(
        RuntimeError, match="Need attn_mask if specifying the is_causal hint"
    ):
        _ = qmha(query, key, value, attn_mask=None, is_causal=True)

    """
    When: Call with heterogeneous ndims
    Then: Both nn.MultiheadAttention and QuantizedMultiheadAttention should throw error
    """
    for args in [
        (query.squeeze(batch_dim), key, value),
        (query, key.squeeze(batch_dim), value),
        (query, key, value.squeeze(batch_dim)),
    ]:
        with pytest.raises(AssertionError, match=r"For (batched|unbatched)"):
            _ = mha(*args)

        with pytest.raises(AssertionError, match=r"For (batched|unbatched)"):
            _ = qmha(*args)

    """
    When: Call with incompatible attn_mask shape
    Then: Both nn.MultiheadAttention and QuantizedMultiheadAttention should throw error
    """
    attn_mask = torch.full((L, L + 1), -10.0).triu(diagonal=1)

    with pytest.raises(
        RuntimeError, match=r"The shape of the 2D attn_mask is .+, but should be .+"
    ):
        _ = mha(query, key, value, attn_mask=attn_mask)

    with pytest.raises(
        RuntimeError, match=r"The shape of the 3D attn_mask is .+, but should be .+"
    ):
        _ = mha(query, key, value, attn_mask=attn_mask.unsqueeze(0))

    with pytest.raises(
        RuntimeError, match=r"The shape of the 2D attn_mask is .+, but should be .+"
    ):
        _ = qmha(query, key, value, attn_mask=attn_mask)

    with pytest.raises(
        RuntimeError, match=r"The shape of the 3D attn_mask is .+, but should be .+"
    ):
        _ = qmha(query, key, value, attn_mask=attn_mask.unsqueeze(0))


@torch.no_grad()
@pytest.mark.parametrize(
    "lpbq, zero_point_shift",
    [
        (False, False),
        (False, True),
        (True, False),
    ],
)
@pytest.mark.parametrize(
    "module_factory",
    [
        functools.partial(nn.Linear, 4, 4),
        functools.partial(nn.Conv1d, 4, 4, 3),
        functools.partial(nn.Conv2d, 4, 4, 3),
        functools.partial(nn.Conv3d, 4, 4, 3),
        functools.partial(nn.ConvTranspose1d, 4, 4, 3),
        functools.partial(nn.ConvTranspose2d, 4, 4, 3),
        functools.partial(nn.ConvTranspose3d, 4, 4, 3),
    ],
)
def test_int32_bias_overflow(
    module_factory,
    lpbq: bool,
    zero_point_shift: bool,
):
    """
    Given: Linear/Conv with very large bias
    When: Concretize int32 bias quantizer
    Then: Int32 bias shouldn't overflow
    """
    torch.manual_seed(0)

    module = module_factory()
    model = torch.nn.Sequential(nn.Softmax(), module)
    # Tiny weight & huge bias to trigger overflow
    module.weight.copy_(1e-7 * torch.rand_like(module.weight))
    module.bias.copy_(torch.tensor([-1024, -1, 1, 1024]))
    input_shape = tuple(
        1 if axis == 0 else 4 for axis, _ in enumerate(module.weight.shape)
    )
    dummy_input = torch.randn(input_shape)
    sim = aimet_torch.QuantizationSimModel(
        model,
        dummy_input,
        default_param_bw=4,
        default_output_bw=16,
    )

    if lpbq:
        set_grouped_blockwise_quantization_for_weights(
            sim,
            [type(module)],
            bitwidth=4,
            symmetric=True,
            decompressed_bw=8,
            block_size=2,
        )

    sim.model[1].param_quantizers["weight"].zero_point_shift = 0.5 * zero_point_shift

    def track_call_count(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            wrapper.call_count += 1
            return fn(*args, **kwargs)

        wrapper.call_count = 0
        return wrapper

    for qtzr in sim.quantizers():
        qtzr._compute_encodings = track_call_count(qtzr._compute_encodings)

    sim.compute_encodings(lambda model: model(dummy_input))

    # compute_encodings should be called exactly once for each quantizer
    assert all(qtzr._compute_encodings.call_count == 1 for qtzr in sim.quantizers())

    _ = sim.model(dummy_input)
    input_scale = sim.model[0].output_quantizers[0].get_scale()
    weight_qtzr = sim.model[1].param_quantizers["weight"]
    weight_encoding = weight_qtzr.get_encoding()
    assert weight_encoding.scale.shape == weight_qtzr.shape

    sim.model[1]._create_int32_bias_quantizer((sim.model[0](dummy_input),), None)
    bias = sim.model[1].bias
    bias_encoding = sim.model[1].param_quantizers["bias"].get_encoding()
    assert torch.all(bias.abs() / bias_encoding.scale <= 2**31)

    per_channel_weight_scale = (
        weight_encoding.per_channel_scale if lpbq else weight_encoding.scale
    )
    assert torch.allclose(
        bias_encoding.scale,
        input_scale * per_channel_weight_scale.flatten(),
    )


def test_compute_encodings_passthrough():
    """
    When: Enter aimet_torch.nn.compute_encodings context manager
    Then: Only weight should be quantized during forward pass
    """
    qlinear = QuantizedLinear(10, 10, bias=False)
    qlinear.param_quantizers["weight"] = QuantizeDequantize(
        shape=(10, 1), qmin=-128, qmax=127, symmetric=True
    )
    qlinear.input_quantizers[0] = QuantizeDequantize(
        shape=(), qmin=0, qmax=255, symmetric=False
    )
    qlinear.output_quantizers[0] = QuantizeDequantize(
        shape=(), qmin=0, qmax=255, symmetric=False
    )

    x = torch.randn(10, 10)
    with aimet_torch.nn.compute_encodings(qlinear):
        out = qlinear(x)

    expected_out = F.linear(x, qlinear.param_quantizers["weight"](qlinear.weight))
    assert torch.allclose(out, expected_out)


@torch.no_grad()
@pytest.mark.parametrize("bitwidth", [2, 4])
def test_prequantized_weight(bitwidth: int):
    """
    When: QuantizedLinear has a pre-quantized int2 weight
    Then: The weight encoding should be lossless
    """
    qmin = -(2 ** (bitwidth - 1))
    qmax = 2 ** (bitwidth - 1) - 1
    qlinear = QuantizedLinear(10, 10, bias=False)
    weight_scale = torch.arange(0.01, 0.11, step=0.01, dtype=torch.float32).view(10, 1)
    qlinear.weight.copy_(
        torch.randint(-1, 2, (10, 10), dtype=torch.float32) * weight_scale
    )
    qlinear.param_quantizers["weight"] = QuantizeDequantize(
        shape=(10, 1), qmin=qmin, qmax=qmax, symmetric=True
    )
    qlinear.compute_param_encodings()

    assert torch.allclose(qlinear.param_quantizers["weight"].get_scale(), weight_scale)


@torch.no_grad()
@pytest.mark.parametrize(
    "module_factory",
    [
        functools.partial(torch.nn.Conv2d, 10, 10, 3, bias=True),
        functools.partial(torch.nn.Conv2d, 10, 10, 3, bias=False),
        functools.partial(torch.nn.ConvTranspose2d, 10, 10, 3, bias=True),
        functools.partial(torch.nn.ConvTranspose2d, 10, 10, 3, bias=False),
        functools.partial(torch.nn.Linear, 10, 10, bias=True),
        functools.partial(torch.nn.Linear, 10, 10, bias=False),
    ],
)
def test_htp_overflow_protection(module_factory):
    """
    Given:
        Conv or Linear with (1) tiny weight scale and (2) output scale >> input scale
    """
    dummy_input = torch.ones(1, 10, 10, 10)
    module = module_factory()

    weight = (
        module.weight
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
        else module.weight.transpose(0, 1)
    )
    # Expected weight scale @ channel 0: 1.0
    weight[0].copy_(-(2**7))
    weight[0, 0] += 1
    # Expected weight scale @ channel 1: 1.0
    weight[1].copy_(2**7 - 1)
    weight[1, 0] -= 1
    # Expected weight scale @ channel 2: 2**-23 (float32 epsilon)
    weight[2:].copy_(1e-6)

    if module.bias is not None:
        module.bias.zero_()

    """
    When: Create quantsim
    Then:
      1. Requantization scale must be > 2**-24
      2. Accumulator bias must be within int32 range
    """
    sim = aimet_torch.QuantizationSimModel(module, dummy_input)
    sim.compute_encodings(lambda model: model(dummy_input))

    input_scale = sim.model.input_quantizers[0].get_scale()
    weight_scale = sim.model.param_quantizers["weight"].get_scale()
    output_scale = sim.model.output_quantizers[0].get_scale()
    output_offset = sim.model.output_quantizers[0].get_offset()

    # Requantization scale should be > 2**-24
    requant_scale = input_scale * weight_scale / output_scale
    assert torch.all(requant_scale > 2**-24)

    # Accumulator bias should be witin int32 range
    accumulator_bias = torch.abs(output_offset / requant_scale)
    assert torch.all(torch.abs(accumulator_bias) < 2**31)

    # Weight scales should remain unchanged if they already satisfy both conditions
    assert torch.all(weight_scale.flatten()[0:2] == 1.0)

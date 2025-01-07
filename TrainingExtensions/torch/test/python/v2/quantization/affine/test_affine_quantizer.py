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
import os
import copy
import math
import pickle
import pytest
import numpy as np
import random
import tempfile
import torch
import warnings
from torch import nn
from torch.optim import SGD, RMSprop, Adagrad, Adam, AdamW
from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
from aimet_torch.v2.quantization.affine import AffineQuantizerBase, Quantize, \
    QuantizeDequantize, GroupedBlockQuantizeDequantize
from aimet_torch.v2.quantization import affine
import aimet_torch.v2.quantization as Q
from ...test_deepspeed import CustomMPU


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(999)
    torch.manual_seed(0)
    np.random.seed(0)


_PARAMETER_SHAPE = (100,)

def _initialize(q, symmetric):
    min = torch.empty(_PARAMETER_SHAPE)
    max = torch.empty(_PARAMETER_SHAPE)

    bw = q.bitwidth
    total_bins = 2**bw - 1
    negative_bins = math.ceil(total_bins / 2)
    positive_bins = math.floor(total_bins / 2)
    min.copy_(-1)
    max.copy_(1 * positive_bins/negative_bins) # max is one tick smaller

    if not symmetric:
        # Move the center to 1
        min.add_(1)
        max.add_(1)

    q.min = torch.nn.Parameter(min)
    q.max = torch.nn.Parameter(max)


def quantize(symmetric, initialized, bitwidth=8):
    encoding_analyzer = MinMaxEncodingAnalyzer(shape=_PARAMETER_SHAPE)
    quantize = Quantize(shape=_PARAMETER_SHAPE,
                        bitwidth=bitwidth,
                        symmetric=symmetric,
                        encoding_analyzer=encoding_analyzer)
    if initialized:
        _initialize(quantize, symmetric)
    return quantize


def quantize_dequantize(symmetric, initialized, bitwidth=8):
    encoding_analyzer = MinMaxEncodingAnalyzer(shape=_PARAMETER_SHAPE)
    quantize_dequantize = QuantizeDequantize(shape=_PARAMETER_SHAPE,
                                             bitwidth=bitwidth,
                                             symmetric=symmetric,
                                             encoding_analyzer=encoding_analyzer)
    if initialized:
        _initialize(quantize_dequantize, symmetric)
    return quantize_dequantize


@pytest.fixture()
def x():
    """
    Returns [
        [-2., -1.99, -1.98, ..., -1.01],
        [-1., -0.99, -0.98, ..., -0.01],
        [ 0.,  0.01,  0.02, ...,  0.99],
        [ 1.,  1.01,  1.02, ...,  1.99],
    ]
    """
    return torch.arange(-200, 200).view(4, 100) / 100


@pytest.fixture
def init_process_group():
    import torch.distributed as dist

    LOCAL_RANK = os.getenv('LOCAL_RANK', None)
    try:
        # Create process group of size 1
        dist.init_process_group(backend='gloo',
                                store=dist.HashStore(),
                                world_size=1,
                                rank=0)
        os.environ['LOCAL_RANK'] = '0'
        yield dist.new_group(ranks=[0])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if LOCAL_RANK is not None:
            os.environ['LOCAL_RANK'] = LOCAL_RANK


@pytest.fixture
def deepspeed_zero3_config():
    return {
        "zero_optimization": {"stage": 3},
        "train_batch_size": 1,
    }


def minmax_to_scaleoffset(min, max, symmetric, bitwidth):
    total_bins = 2 ** bitwidth - 1
    scale = (max - min) / total_bins
    if symmetric:
        offset = torch.zeros_like(scale)
    else:
        offset = torch.round(min / scale)
    return scale, offset


@pytest.mark.parametrize('quantize', [
    quantize(symmetric=True, initialized=False),
    quantize(symmetric=True, initialized=True),
    quantize(symmetric=False, initialized=False),
    quantize(symmetric=False, initialized=True),
])
def test_quantize_compute_encodings(quantize: Quantize, x: torch.Tensor):
    """
    :param quantize: Quantize module
    :param x: Input tensor

    Given: During compute_encodings
    When:
      1. forward() invoked with input x
      2. Exit compute_encodings() context
    Then:
      1. forward() returns dynamic quantization output
      2. self.get_scale(), self.get_offset() == dynamic scale/offset of x
    """
    num_quant_bins = math.pow(2, quantize.bitwidth) - 1
    dynamic_min, dynamic_max =\
            quantize.encoding_analyzer.compute_dynamic_encodings(x, num_quant_bins, quantize.symmetric)
    dynamic_scale, dynamic_offset = minmax_to_scaleoffset(dynamic_min,
                                                          dynamic_max,
                                                          quantize.symmetric,
                                                          bitwidth=8)
    expected_x_int = Q.affine.quantize(x,
                                       dynamic_scale,
                                       dynamic_offset,
                                       quantize.qmin,
                                       quantize.qmax)

    with quantize.compute_encodings():
        x_int = quantize(x)

    assert torch.allclose(x_int.quantized_repr(), expected_x_int.to(x_int.encoding.dtype))
    assert torch.allclose(x_int.encoding.scale, dynamic_scale)
    assert torch.allclose(x_int.encoding.offset, dynamic_offset)
    assert torch.allclose(quantize.min, dynamic_min)
    assert torch.allclose(quantize.max, dynamic_max)
    assert torch.allclose(quantize.get_scale(), dynamic_scale)
    assert torch.allclose(quantize.get_offset(), dynamic_offset)


@pytest.mark.parametrize('quantize_dequantize', [
    quantize_dequantize(symmetric=True, initialized=False),
    quantize_dequantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=False, initialized=False),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_qdq_compute_encodings(quantize_dequantize: QuantizeDequantize, x: torch.Tensor):
    """
    :param q: QuantizeDequantize module
    :param x: Input tensor

    Given: During compute_encodings
    When:
      1. forward() invoked with input x
      2. Exit compute_encodings() context
    Then:
      1. forward() returns dynamic quantization output
      2. self.get_scale(), self.get_offset() == dynamic scale/offset of x
    """
    num_quant_bins = math.pow(2, quantize_dequantize.bitwidth) - 1
    dynamic_min, dynamic_max =\
            quantize_dequantize.encoding_analyzer.compute_dynamic_encodings(x,
                                                                            num_quant_bins,
                                                                            quantize_dequantize.symmetric)
    dynamic_scale, dynamic_offset = minmax_to_scaleoffset(dynamic_min,
                                                          dynamic_max,
                                                          quantize_dequantize.symmetric,
                                                          bitwidth=8)
    expected_output = Q.affine.quantize_dequantize(x,
                                                   dynamic_scale,
                                                   dynamic_offset,
                                                   quantize_dequantize.qmin,
                                                   quantize_dequantize.qmax)

    with quantize_dequantize.compute_encodings():
        output = quantize_dequantize(x)

    assert torch.allclose(output, expected_output)
    assert torch.allclose(quantize_dequantize.min, dynamic_min)
    assert torch.allclose(quantize_dequantize.max, dynamic_max)
    assert torch.allclose(quantize_dequantize.get_scale(), dynamic_scale)
    assert torch.allclose(quantize_dequantize.get_offset(), dynamic_offset)


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=False),
    quantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=True, initialized=False),
    quantize_dequantize(symmetric=True, initialized=True),
])
def test_compute_encodings_with_no_input(q: AffineQuantizerBase):
    """
    :param q: Quantize or QuantizeDequantize module

    Given: During compute_encodings
    When:
      1. forward() never invoked
      2. Exit compute_encodings() context
    Then: self.get_min(), self.get_max() doesn't change
    """

    original_min = q.get_min()
    if original_min is not None:
        original_min = original_min.clone().detach()

    original_max = q.get_max()
    if original_max is not None:
        original_max = original_max.clone().detach()

    with q.compute_encodings():
        pass

    if original_min is None:
        assert q.get_min() is original_min
    else:
        assert torch.equal(q.get_min(), original_min)

    if original_max is None:
        assert q.get_max() is None
    else:
        assert torch.equal(q.get_max(), original_max)


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=True, initialized=True),
    quantize(symmetric=False, initialized=True),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_backward_during_compute_encodings(q: AffineQuantizerBase, x: torch.Tensor):
    """
    :param q: Quantize or QuantizeDequantize module
    :param x: Input tensor

    Given: During compute_encodings
    When:
      1. forward() invoked
      2. backward() invoked
    Then: self.min.grad == self.max.grad == None
          (min/max are not trainable during compute_encodings)
    """
    x = x.clone().requires_grad_(True)

    with q.compute_encodings():
        if isinstance(q, Quantize):
            output = q(x)
        else:
            output = q(x)
        output.backward(torch.zeros_like(output))

    assert q.min.grad is None
    assert q.max.grad is None


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=False),
    quantize_dequantize(symmetric=True, initialized=False),
])
def test_compute_encodings_updates_parameters_upon_exit(q: AffineQuantizerBase, x: torch.Tensor):
    """
    :param q: Quantize or QuantizeDequantize module
    :param x: Input tensor

    Given: During compute_encodings
    When:
      1. forward() invoked
      2. Exit compute_encodings() context
    Then: min/max/scale/offset are updated when exiting compute_encodings
    """
    assert q.get_min() is None
    assert q.get_max() is None
    assert q.get_scale() is None
    assert q.get_offset() is None

    with q.compute_encodings():
        assert q.get_min() is None
        assert q.get_max() is None
        assert q.get_scale() is None
        assert q.get_offset() is None

        _ = q(x)

        assert q.get_min() is None
        assert q.get_max() is None
        assert q.get_scale() is None
        assert q.get_offset() is None


    assert q.get_min() is not None
    assert q.get_max() is not None
    assert q.get_scale() is not None
    assert q.get_offset() is not None


@pytest.mark.parametrize('quantize', [
    quantize(symmetric=True, initialized=True),
    quantize(symmetric=False, initialized=True),
])
def test_quantize_forward(quantize: Quantize, x: torch.Tensor):
    """
    :param q: Quantize module
    :param x: Input tensor

    Given:
      1. Outside compute_encodings
      2. Quantization parmeters are initialized
    When: forward() invoked
    Then: forward() returns parametric quantization output.
    """
    output = quantize(x)
    expected_output = Q.affine.quantize(x,
                                        quantize.get_scale(),
                                        quantize.get_offset(),
                                        quantize.qmin,
                                        quantize.qmax)
    assert torch.allclose(output.quantized_repr(), expected_output.to(output.encoding.dtype))


@pytest.mark.parametrize('quantize_dequantize', [
    quantize_dequantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_qdq_forward(quantize_dequantize: QuantizeDequantize, x: torch.Tensor):
    """
    :param q: QuantizeDequantize module
    :param x: Input tensor

    Given:
      1. Outside compute_encodings
      2. Quantization parmeters are initialized
    When: forward() invoked
    Then: forward() returns parametric quantization output.
    """
    output = quantize_dequantize(x)
    expected_output = Q.affine.quantize_dequantize(x,
                                                   quantize_dequantize.get_scale(),
                                                   quantize_dequantize.get_offset(),
                                                   quantize_dequantize.qmin,
                                                   quantize_dequantize.qmax)
    assert torch.allclose(output, expected_output)


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=True),
    quantize(symmetric=False, initialized=True),
    quantize_dequantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_backward(q: AffineQuantizerBase, x: torch.Tensor):
    """
    :param q: Quantize or QuantizeDequantize module
    :param x: Input tensor

    Given:
      1. Outside compute_encodings
      2. Quantization parmeters are initialized
    When:
      1. forward() invoked
      2. backward() invoked
    Then: self.min.grad and self.max.grad should be computed
    """
    if isinstance(q, Quantize):
        output = q(x)
    else:
        output = q(x)
    output.backward(torch.zeros_like(output))
    assert q.min.grad is not None
    assert q.max.grad is not None


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=True),
    quantize(symmetric=False, initialized=True),
    quantize_dequantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_backward_with_no_grad(q, x: torch.Tensor):
    """
    :param q: Quantize or QuantizeDequantize module
    :param x: Input tensor

    Given:
      1. Outside compute_encodings
      2. Quantization parmeters are initialized
    When:
      1. forward() invoked with torch.no_grad()
      2. backward() invoked
    Then: self.min.grad and self.max.grad should not be computed
    """
    x = x.clone().requires_grad_(True)
    with torch.no_grad():
        if isinstance(q, Quantize):
            output = q(x)
        else:
            output = q(x)
    output = output + x
    output.backward(torch.zeros_like(output))
    assert q.min.grad is None
    assert q.max.grad is None


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=False),
    quantize_dequantize(symmetric=True, initialized=False),
])
def test_uninitialized_quantize(q: AffineQuantizerBase, x: torch.Tensor):
    """
    :param q: Quantize or QuantizeDequantize module
    :param x: Input tensor

    Given:
      1. Outside compute_encodings
      2. Quantization parameters not initialized yet
    When: forward() invoked
    Then: Throw runtime error
    """
    assert q.get_min() is None
    assert q.get_max() is None
    assert q.get_scale() is None
    assert q.get_offset() is None

    with pytest.raises(RuntimeError):
        _ = q(x)


def _test_symmetric_invariants(q):
    """
    symmetric invaraints:
      1. min = scale * offest
      2. max = scale * -(offset + 1)
      3. scale = range / total_bins
         where range = max(-min * total_bins/negative_bins,
                            max * total_bins/positive_bins)
               total_bins = 2**bw - 1
               negative_bins = 2 ** (bw - 1)
               positive_bins = negative_bins - 1
      4. offset = 0
      5. offset is fixed (offset.requires_grad = False)
    """
    min = q.get_min()
    max = q.get_max()
    scale = q.get_scale()
    offset = q.get_offset()
    bw = q.bitwidth

    total_bins = 2**bw - 1
    positive_bins = math.floor(total_bins / 2)
    negative_bins = math.ceil(total_bins / 2)

    # min == scale * offset
    assert torch.allclose(min, - scale * negative_bins,
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    # max == scale * -(offset+1)
    assert torch.allclose(max, scale * positive_bins,
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    range = torch.maximum(-min * total_bins/negative_bins,
                          max * total_bins/positive_bins)
    assert torch.allclose(scale, range / total_bins,
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    # offset == -1 * 2 ** (bw -1)
    assert torch.equal(offset, torch.zeros_like(offset))
    # offset is fixed in symmetric quantizer
    assert not offset.requires_grad


def _test_asymmetric_invariants(q):
    """
    asymmetric invaraints:
      1. min = scale * offest
      2. max = min + (2**bw - 1)
      3. scale = (max - min) / (2**bw - 1)
      4. offset = round(min / scale)
      5. offset is trainable (offset.requires_grad = True)
    """
    min = q.get_min()
    max = q.get_max()
    scale = q.get_scale()
    offset = q.get_offset()
    bw = q.bitwidth

    # min == scale * offset
    assert torch.allclose(min, scale * offset,
                          rtol=1e-3, atol=min.abs().max().item() * 1e-5)

    # max == min + scale * (2**bw - 1)
    assert torch.allclose(max, min + scale * (2**bw - 1),
                          rtol=1e-3, atol=max.abs().max().item() * 1e-5)

    # scale == (max - min) / (2**bw - 1)
    assert torch.allclose(scale, (max - min) / (2**bw - 1),
                          rtol=1e-3, atol=scale.abs().max().item() * 1e-5)

    # offsets == round(min / scale)
    assert torch.equal(torch.round(min/scale), offset)
    # offset is learned in asymmetric quantizer
    assert offset.requires_grad


@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=False),
    quantize_dequantize(symmetric=True, initialized=False),
])
def test_symmetric_invariants(q, x: torch.Tensor):
    """
    Given: Symmetric quantizer
    When: Quantization parameters initialized with compute_encodings
    Then: Should satisfy all the symmetric quantization invariants
    """
    with q.compute_encodings():
        _ = q(x)

    _test_symmetric_invariants(q)


@pytest.mark.parametrize("optim_cls", [SGD, RMSprop, Adagrad, Adam, AdamW])
@pytest.mark.parametrize('q', [
    quantize(symmetric=True, initialized=True),
    quantize_dequantize(symmetric=True, initialized=True),
])
def test_symmetric_learning(q, x, optim_cls):
    """
    Given:
      1. Symmetric quantizer
      2. Quantization parameters are initialized
    When:
      1. forward() invoked
      2. backward() invoked
      3. optimizer.step() invoked
    Then: Should satisfy all the symmetric quantization invariants
    """

    original_min = q.get_min().clone().detach()
    original_max = q.get_max().clone().detach()
    original_scale = q.get_scale().clone().detach()
    original_offset = q.get_offset().clone().detach()

    optimizer = optim_cls(q.parameters(), lr=1.0)

    for _ in range(10):
        if isinstance(q, Quantize):
            output = q(x)
        else:
            output = q(x)
        output.backward(torch.randn_like(output))
        optimizer.step()
        _test_symmetric_invariants(q)

    assert not torch.equal(q.get_min(), original_min)
    assert not torch.equal(q.get_max(), original_max)
    assert not torch.equal(q.get_scale(), original_scale)
    assert torch.equal(q.get_offset(), original_offset)


@pytest.mark.parametrize('q', [
    quantize(symmetric=False, initialized=False),
    quantize_dequantize(symmetric=False, initialized=False),
])
def test_asymmetric_invariants(q: AffineQuantizerBase, x: torch.Tensor):
    """
    Given: Asymmetric quantizer
    When: Quantization parameters initialized with compute_encodings
    Then: Should satisfy all the symmetric quantization invariants
    """
    with q.compute_encodings():
        _ = q(x)

    _test_asymmetric_invariants(q)


@pytest.mark.parametrize("optim_cls", [SGD, RMSprop, Adagrad, Adam, AdamW])
@pytest.mark.parametrize('q', [
    quantize(symmetric=False, initialized=True),
    quantize_dequantize(symmetric=False, initialized=True),
])
def test_asymmetric_learning(q, x, optim_cls):
    """
    Given:
      1. Asymmetric quantizer
      2. Quantization parameters are initialized
    When:
      1. forward() invoked
      2. backward() invoked
      3. optimizer.step() invoked
    Then: Should satisfy all the asymmetric quantization invariants
    """
    original_min = q.get_min().clone().detach()
    original_max = q.get_max().clone().detach()
    original_scale = q.get_scale().clone().detach()
    original_offset = q.get_offset().clone().detach()

    optimizer = optim_cls(q.parameters(), lr=1.0)

    for _ in range(10):
        if isinstance(q, Quantize):
            output = q(x)
        else:
            output = q(x)
        output.backward(torch.randn_like(output))
        optimizer.step()
        _test_asymmetric_invariants(q)

    assert not torch.equal(q.get_min(), original_min)
    assert not torch.equal(q.get_max(), original_max)
    assert not torch.equal(q.get_scale(), original_scale)
    assert not torch.equal(q.get_offset(), original_offset)

def test_extreme_values_warning():
        extreme_val = torch.finfo(torch.float16).max
        dummy_input = torch.arange(start = 0, end=extreme_val, dtype=torch.float16)        
        param_shape = (1,)
        encoding_shape = (1,)
        qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with qdq.compute_encodings():
                qdq(dummy_input)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "Extreme values" in str(w[-1].message)

def test_invalid_encoding_analyzer():
    """
    When: Instantiate a quantizer with an encoding analyzer of unmatching shape
    Then: Throw runtime error
    """
    dummy_input = torch.randn((30, 10, 11))
    param_shape = (10, 11)

    encoding_shape = (12,)
    with pytest.raises(RuntimeError):
        _ = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))

    encoding_shape = (10, 11)
    qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
    with qdq.compute_encodings():
        _ = qdq(dummy_input)

    encoding_shape = (11,)
    qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
    with qdq.compute_encodings():
        _ = qdq(dummy_input)

    encoding_shape = (10, 1)
    qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
    with qdq.compute_encodings():
        _ = qdq(dummy_input)

    encoding_shape = 11
    qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
    with qdq.compute_encodings():
        _ = qdq(dummy_input)

    encoding_shape = 1
    qdq = QuantizeDequantize(param_shape, 8, True, MinMaxEncodingAnalyzer(encoding_shape))
    with qdq.compute_encodings():
        _ = qdq(dummy_input)


@torch.no_grad()
@pytest.mark.cuda
def test_is_initialized(x):
    """
    When: Instantiate a quantizer object
    Then:
      1) All the parameters readily exist as nn.Parameters (not as None or nn.UninitializedParameters)
      2) quantizer.is_initialized() returns False
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    assert isinstance(qdq.min, nn.Parameter) and not isinstance(qdq.min, nn.UninitializedParameter)
    assert isinstance(qdq.max, nn.Parameter) and not isinstance(qdq.max, nn.UninitializedParameter)
    assert not qdq.is_initialized()

    qdq.to(device="cuda", dtype=torch.float16)
    assert not qdq.is_initialized()

    """
    When: Update the parameters using in-place operation
    Then: is_initialized() returns True
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.min.copy_(torch.zeros(10))
    assert not qdq.is_initialized() # False; max is not initialized yet
    qdq.max.add_(3)
    assert qdq.is_initialized()

    """
    When: Update the parameters with assignment statement
    Then: is_initialized() returns True
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.min = nn.Parameter(-torch.ones(10) * 2)
    assert not qdq.is_initialized() # False; max is not initialized yet
    qdq.max = nn.Parameter(torch.ones(10) * 2)
    assert qdq.is_initialized()

    """
    When: Update the parameters with compute_encodings()
    Then: is_initialized() returns True
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    with qdq.compute_encodings():
        _ = qdq(torch.arange(-5, 5, dtype=torch.float))
    assert qdq.is_initialized()

    """
    When: Invoke load_state_dict() with a state dict that contains all parameters
    Then: quantizer.is_initialized() returns True
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.load_state_dict({'min': -torch.ones(10), 'max': torch.ones(10)})
    assert qdq.is_initialized()

    """
    When: Invoke load_state_dict with insufficient parameters
    Then: quantizer.is_initialized() returns False
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.load_state_dict({'min': -torch.ones(10)}, strict=False)
    assert not qdq.is_initialized() # False; max is not initialized yet
    qdq.load_state_dict({'max': torch.ones(10)}, strict=False)
    assert qdq.is_initialized()

    """
    When: Invoke load_state_dict() with a state dict that contains uninitialized parameters
    Then: quantizer.is_initialized() returns False
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    uninitialized_state_dict = qdq.state_dict()
    qdq.load_state_dict(uninitialized_state_dict)
    assert not qdq.is_initialized()

    qdq.min.mul_(1.)
    partially_initialized_state_dict = qdq.state_dict()
    qdq.load_state_dict(partially_initialized_state_dict)
    assert not qdq.is_initialized()

    qdq.max.mul_(1.)
    fully_initialized_state_dict = qdq.state_dict()
    qdq.load_state_dict(fully_initialized_state_dict)
    assert qdq.is_initialized()

    """
    When: Create a deepcopy of quantizer
    Then: quantizer.is_initialized() flag should be preserved
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq = copy.deepcopy(qdq)
    assert not qdq.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.load_state_dict({'min': -torch.ones(10), 'max': torch.ones(10)})
    qdq = copy.deepcopy(qdq)
    assert qdq.is_initialized()

    """
    When: Pickle and unpickle quantizer
    Then: quantizer.is_initialized() flag should be preserved
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    res = pickle.dumps(qdq)
    qdq = pickle.loads(res)
    assert not qdq.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.load_state_dict({'min': -torch.ones(10), 'max': torch.ones(10)})
    out_before = qdq(x.view(-1, 10))
    res = pickle.dumps(qdq)
    qdq = pickle.loads(res)
    assert qdq.is_initialized()
    assert torch.equal(qdq(x.view(-1, 10)), out_before)


@pytest.mark.cuda
def test_is_initialized_with_deepspeed_zero3(init_process_group, deepspeed_zero3_config):
    import deepspeed as ds

    """
    When: Partition a quantizer with deepspeed zero 3
    Then: quantizer.is_initialized() flag should be preserved after pertitioning
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config,
                               mpu=CustomMPU(init_process_group))
    qdq_zero3 = engine.module
    assert not qdq_zero3.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.set_range(-1, 1)
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module
    assert qdq_zero3.is_initialized()

    """
    When: Gather the partitioned quantization parameters in read-only mode
    Then: quantizer.is_initialized() flag should be preserved during/after gathering
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module

    with ds.zero.GatheredParameters(qdq_zero3.parameters()):
        assert not qdq_zero3.is_initialized()
    assert not qdq_zero3.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.set_range(-1, 1)
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module

    with ds.zero.GatheredParameters(qdq_zero3.parameters()):
        assert qdq_zero3.is_initialized()
    assert qdq_zero3.is_initialized()

    """
    When: Modify the partitioned quantization parameters
    Then: quantizer.is_initialized() returns True
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module
    qdq_zero3.set_range(-1, 1)
    assert qdq_zero3.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module
    with qdq_zero3.compute_encodings():
        _ = qdq_zero3(torch.arange(-5, 5, dtype=torch.float, device='cuda:0'))
    assert qdq_zero3.is_initialized()

    """
    When: Gather the partitioned quantization parameters in writable mode but don't modify them
    Then: quantizer.is_initialized() flag should be preserved during/after gathering
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module

    with ds.zero.GatheredParameters(qdq_zero3.parameters(), modifier_rank=0):
        assert not qdq_zero3.is_initialized()
    assert not qdq_zero3.is_initialized()

    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.set_range(-1, 1)
    engine, *_ = ds.initialize(model=qdq, config=deepspeed_zero3_config)
    qdq_zero3 = engine.module

    with ds.zero.GatheredParameters(qdq_zero3.parameters(), modifier_rank=0):
        assert qdq_zero3.is_initialized()
    assert qdq_zero3.is_initialized()


@torch.no_grad()
@pytest.mark.parametrize('symmetric', [True, False])
def test_quantize_dequantize_then_quantize_and_dequantize_equality(x, symmetric):
    qdq = QuantizeDequantize((), 8, symmetric)
    q = Quantize((), 8, symmetric)

    with qdq.compute_encodings(), q.compute_encodings():
        _ = qdq(x)
        _ = q(x)

    a = qdq(x)
    b = q(x).dequantize()
    assert torch.equal(a, b)


@pytest.mark.parametrize("q", (Quantize(_PARAMETER_SHAPE, 8, False),
                               QuantizeDequantize(_PARAMETER_SHAPE, 8, True)))
def test_allow_overwrite(x, q):
    with q.compute_encodings():
        q(x)

    """
    Given: _allow_overwrite set to True
    When: Try to recompute encodings
    Then: Encoding does NOT get overwritten by compute_encodings
    """
    q_min, q_max = q.min.detach().clone(), q.max.detach().clone()
    q.allow_overwrite(False)
    with q.compute_encodings():
        q(x * 10)

    assert torch.equal(q_min, q.min)
    assert torch.equal(q_max, q.max)


def test_bq_compute_encodings_and_forward():
    shape = (2, 2, 4)
    bq = QuantizeDequantize(shape=shape,
                            bitwidth=4,
                            symmetric=True,
                            block_size=(2, 4, 3))
    assert bq.encoding_analyzer.observer.shape == (2, 1, 2, 1, 4, 1)

    bq.eval()
    param_tensor = torch.randn(4, 8, 12)
    with bq.compute_encodings():
        _ = bq(param_tensor)

    out = bq(param_tensor)
    assert bq.get_min().shape == shape
    assert out.shape == param_tensor.shape

    qdq_out = affine.quantize_dequantize(param_tensor, bq.get_scale(), bq.get_offset(), bq.qmin, bq.qmax,
                                         block_size=bq.block_size)
    assert torch.equal(out, qdq_out)

@pytest.mark.parametrize('shape, block_sizes', [[(4, 1, 1), (1, 4, 4)],
                                                [(1, 4, 1), (4, 1, 4)],
                                                [(1, 1, 4), (4, 4, 1)]])
def test_bq_vs_per_channel_sanity(shape, block_sizes):
    bq = QuantizeDequantize(shape=shape,
                            bitwidth=4,
                            symmetric=True,
                            block_size=block_sizes)

    pc = QuantizeDequantize(shape=shape,
                            bitwidth=4,
                            symmetric=True)

    bq.eval()
    pc.eval()
    param_tensor = torch.randn(4, 4, 4)
    with bq.compute_encodings():
        _ = bq(param_tensor)

    with pc.compute_encodings():
        _ = pc(param_tensor)

    assert torch.equal(bq(param_tensor), pc(param_tensor))

def test_quantized_tensor_with_block_size():
    shape = (2, 2, 4)
    tensor = torch.randn(4, 8, 12)
    bq = Quantize(shape=shape,
                  bitwidth=4,
                  symmetric=True,
                  block_size=(2, 4, 3))
    with bq.compute_encodings():
        _ = bq(tensor)
    assert bq.get_encodings().block_size == bq.block_size
    q = bq(tensor)
    assert q.encoding.block_size == bq.block_size
    assert torch.equal(q.dequantize(), affine.dequantize(q, bq.get_scale(), bq.get_offset(), bq.block_size))

def test_gbbq_sanity():
    tensor = torch.randn(8, 12)
    gbbq = GroupedBlockQuantizeDequantize(shape=(8, 4),
                                          bitwidth=4,
                                          symmetric=True,
                                          decompressed_bw=8,
                                          block_size=(-1, -1),
                                          block_grouping=(1, -1))
    pc = QuantizeDequantize(shape=(8, 1),
                            bitwidth=4,
                            symmetric=True)

    with gbbq.compute_encodings():
        _ = gbbq(tensor)

    with pc.compute_encodings():
        _ = pc(tensor)

    assert gbbq.get_scale().shape == (8, 4)

    # The largest scale for any given channel GBBQ should equal the scale for per channel
    assert torch.equal(torch.amax(gbbq.get_scale(), dim=1, keepdim=True), pc.get_scale())

    assert not torch.equal(gbbq(tensor), pc(tensor))

@pytest.mark.parametrize('bitwidth, decompressed_bw', [[4, 8], [4, 16], [4, 12], [3, 5], [5, 9], [6, 6]])
def test_gbbq_per_block_sanity(bitwidth, decompressed_bw):
    tensor = torch.randn(4, 8, 12)
    gbbq = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                          bitwidth=bitwidth,
                                          symmetric=True,
                                          block_size=(2, 2, 2),
                                          decompressed_bw=decompressed_bw,
                                          block_grouping=(2, 2, 3))
    qdq = QuantizeDequantize(shape=(2, 4, 6),
                             bitwidth=bitwidth,
                             symmetric=True,
                             block_size=(2, 2, 2))
    with gbbq.compute_encodings():
        _ = gbbq(tensor)

    with qdq.compute_encodings():
        _ = qdq(tensor)

    for i in range(gbbq.shape[0] // gbbq.block_grouping[0]):
        for j in range(gbbq.shape[1] // gbbq.block_grouping[1]):
            for k in range(gbbq.shape[2] // gbbq.block_grouping[2]):
                gbbq_block_group = gbbq.get_scale()[i * gbbq.block_grouping[0]:(i + 1) * gbbq.block_grouping[0],
                                                    j * gbbq.block_grouping[1]:(j + 1) * gbbq.block_grouping[1],
                                                    k * gbbq.block_grouping[2]:(k + 1) * gbbq.block_grouping[2]]
                qdq_block_group = qdq.get_scale()[i * gbbq.block_grouping[0]:(i + 1) * gbbq.block_grouping[0],
                                                  j * gbbq.block_grouping[1]:(j + 1) * gbbq.block_grouping[1],
                                                  k * gbbq.block_grouping[2]:(k + 1) * gbbq.block_grouping[2]]
                max_scale = torch.max(qdq_block_group)
                compression_factor = 2 ** (decompressed_bw - bitwidth)
                gamma = max_scale / compression_factor
                int_rounded_scales = torch.maximum(torch.tensor([1.0]), torch.round(qdq_block_group / gamma))
                rounded_scales = int_rounded_scales * gamma
                assert torch.equal(rounded_scales, gbbq_block_group)

def test_gbbq_quantizer_default_grouping():
    tensor = torch.randn(4, 8, 12)
    gbbq_default_grouping = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                                           bitwidth=4,
                                                           symmetric=True,
                                                           block_size=(2, 2, 2),
                                                           decompressed_bw=8)
    gbbq_no_grouping = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                                      bitwidth=4,
                                                      symmetric=True,
                                                      block_size=(2, 2, 2),
                                                      decompressed_bw=8,
                                                      block_grouping=(1, 1, 1))
    with gbbq_default_grouping.compute_encodings():
        _ = gbbq_default_grouping(tensor)

    with gbbq_no_grouping.compute_encodings():
        _ = gbbq_no_grouping(tensor)

    assert torch.equal(gbbq_default_grouping.get_scale(), gbbq_no_grouping.get_scale())
    assert torch.equal(gbbq_default_grouping(tensor), gbbq_no_grouping(tensor))

@pytest.mark.parametrize('gbbq_shape, gbbq_decompressed_bw, gbbq_block_size, gbbq_block_grouping, qdq_shape,'
                         'qdq_block_size',
                         [[(2, 4, 6), 8, (2, 2, 2), (1, 1, 1), (2, 4, 6), (2, 2, 2)],
                          [(2, 4, 6), 4, (2, 2, 2), (-1, -1, -1), (1, 1, 1), None],
                          [(2, 8, 6), 4, (2, 1, 2), (-1, 1, -1), (1, 8, 1), None]])
def test_gbbq_equivalences(gbbq_shape, gbbq_decompressed_bw, gbbq_block_size, gbbq_block_grouping, qdq_shape,
                           qdq_block_size):
    # Test 1: GBBQ should be equal to BQ in the case when block_grouping is 1 for all dims.
    # Test 2: GBBQ should be equal to per tensor in the case when block_grouping is -1 for all dims and decompressed_bw
    #         is equal to bitwidth.
    # Test 3: GBBQ should be equal to per channel in the case when block_grouping is -1 for all dims except the channel
    #         dimension and decompressed_bw is equal to bitwidth.
    tensor = torch.randn(4, 8, 12)
    gbbq = GroupedBlockQuantizeDequantize(shape=gbbq_shape,
                                          bitwidth=4,
                                          symmetric=True,
                                          block_size=gbbq_block_size,
                                          decompressed_bw=gbbq_decompressed_bw,
                                          block_grouping=gbbq_block_grouping)
    qdq = QuantizeDequantize(shape=qdq_shape,
                             bitwidth=4,
                             symmetric=True,
                             block_size=qdq_block_size)
    with gbbq.compute_encodings():
        _ = gbbq(tensor)

    with qdq.compute_encodings():
        _ = qdq(tensor)

    assert torch.equal(gbbq.get_scale(), qdq.get_scale().expand(gbbq.get_scale().shape))
    gbbq_out = gbbq(tensor)
    qdq_out = qdq(tensor)
    assert torch.equal(gbbq_out, qdq_out)

    if all(group_size == 1 for group_size in gbbq_block_grouping):
        grad = torch.randn(size=gbbq_out.shape)
        gbbq_out.backward(grad)
        qdq_out.backward(grad)

        assert torch.equal(gbbq.min.grad, qdq.min.grad)
        assert torch.equal(gbbq.max.grad, qdq.max.grad)

def test_invalid_gbbq_settings():
    with pytest.raises(RuntimeError):
        _ = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                           bitwidth=4,
                                           symmetric=False,
                                           decompressed_bw=8)
    with pytest.raises(RuntimeError):
        _ = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                           bitwidth=4,
                                           symmetric=True,
                                           decompressed_bw=8,
                                           block_grouping=(-1, -1, -1, -1))

    with pytest.raises(RuntimeError):
        _ = GroupedBlockQuantizeDequantize(shape=(2, 4, 6),
                                           bitwidth=4,
                                           symmetric=True,
                                           decompressed_bw=3)

def test_import_signed_flag():
    symmetric_quantizer = QuantizeDequantize((1, ), 8, symmetric=True)
    asymmetric_quantizer = QuantizeDequantize((1, ), 8, symmetric=False)
    assert symmetric_quantizer.signed
    with symmetric_quantizer.compute_encodings():
        symmetric_quantizer(torch.randn(10))
    """
    When: Load signed-symmetric encodings into unsigned-asymmetric quantizer
    Then: unsigned-asymmetric quantizer becomes signed-symmetric
    """
    asymmetric_quantizer.set_legacy_encodings(symmetric_quantizer.get_legacy_encodings())
    assert asymmetric_quantizer.signed
    assert asymmetric_quantizer.symmetric
    """
    When: Load unsigned-asymmetric encodings into signed-symmetric quantizer
    Then: signed-symmetric quantizer becomes unsigned-asymmetric
    """
    asymmetric_quantizer = QuantizeDequantize((), 8, symmetric=False)
    with asymmetric_quantizer.compute_encodings():
        asymmetric_quantizer(torch.randn(10))
    symmetric_quantizer.set_legacy_encodings(asymmetric_quantizer.get_legacy_encodings())
    assert not symmetric_quantizer.signed
    assert not symmetric_quantizer.symmetric


def test_onnx_export():
    """
    When: torch.onnx.export a quantizer
    Then: export shouldn't throw error
    """
    qdq = QuantizeDequantize((10,), bitwidth=8, symmetric=True, encoding_analyzer=MinMaxEncodingAnalyzer((10,)))
    qdq.load_state_dict({'min': -torch.ones(10), 'max': torch.ones(10)})
    with tempfile.TemporaryDirectory() as tempdir:
        with open(os.path.join(tempdir, 'qtzr.onnx'), 'wb') as f:
            torch.onnx.export(qdq, torch.randn(10, 10), f)


def get_qtzn_grid(bitwidth, signed):
    if signed:
        qmin = -2 ** (bitwidth - 1)
        qmax = -qmin - 1
    else:
        qmin = 0
        qmax = 2 ** bitwidth - 1

    return torch.arange(qmin, qmax+1, dtype=torch.long)


FLOAT32_EPS = torch.tensor(torch.finfo(torch.float32).eps)
FLOAT32_TINY = torch.tensor(torch.finfo(torch.float32).tiny)


@pytest.mark.parametrize('tiny_scale', [FLOAT32_EPS, FLOAT32_TINY])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('symmetric', [True, False])
@pytest.mark.parametrize('bitwidth', [4, 8, 16])
def test_sub_float32_quantize_dequantize(dtype, bitwidth, symmetric, tiny_scale):
    """
    Given: Input of range [0, 2**bw) * tiny_scale
    """
    min_scale = FLOAT32_EPS
    x_int = get_qtzn_grid(bitwidth, signed=symmetric)
    x = (x_int * tiny_scale).to(dtype)

    """
    When: Compute encodings of QuantizeDequantize
    Then: qtzr.get_scale() should be no smaller than torch.finfo(float32).eps
    """
    qtzr = QuantizeDequantize((), bitwidth, symmetric).to(dtype)

    with qtzr.compute_encodings():
        _ = qtzr(x)

    assert torch.allclose(qtzr.get_scale(dtype=torch.float32), min_scale)
    assert torch.allclose(qtzr.get_offset(dtype=torch.float32), torch.zeros([]))

    """
    When: Run forward
    Then: Output should be equal to performing quantize-dequantize in float32
    """
    out = qtzr(x)
    assert out.dtype == x.dtype
    assert torch.all(out.isfinite())
    expected = Q.affine.quantize_dequantize(x.float(),
                                            min_scale,
                                            torch.zeros([]).float(),
                                            qmin=int(x_int.min()),
                                            qmax=int(x_int.max())).to(dtype)

    assert torch.equal(out, expected)
    assert torch.allclose(out, x, atol=float(min_scale)/2)


@pytest.mark.parametrize('tiny_scale', [FLOAT32_EPS, FLOAT32_TINY])
@pytest.mark.parametrize(
    'dtype,          bitwidth,  symmetric', [
    (torch.float16,  4,         True),
    (torch.float16,  8,         True),
    (torch.float16,  12,        True),
    (torch.float16,  4,         False),
    (torch.float16,  8,         False),
    (torch.float16,  11,        False),
    (torch.bfloat16, 4,         True),
    (torch.bfloat16, 8,         True),
    (torch.bfloat16, 9,         True),
    (torch.bfloat16, 4,         False),
    (torch.bfloat16, 8,         False),
])
def test_sub_float32_quantize(dtype, bitwidth, symmetric, tiny_scale):
    """
    Given: Input of range [0, 2**bw) * tiny_scale
    """
    min_scale = FLOAT32_EPS
    x_int = get_qtzn_grid(bitwidth, signed=symmetric)
    x = (x_int * tiny_scale).to(dtype)

    """
    When: Compute encodings of QuantizeDequantize
    Then: qtzr.get_scale() should be no smaller than torch.finfo(float32).eps
    """
    qtzr = Quantize((), bitwidth, symmetric).to(dtype)

    with qtzr.compute_encodings():
        _ = qtzr(x)

    assert torch.allclose(qtzr.get_scale(dtype=torch.float32), min_scale)
    assert torch.allclose(qtzr.get_offset(dtype=torch.float32), torch.zeros([]))

    """
    When: Run forward
    Then: Output should be equal to performing quantize in float32
    """
    out = qtzr(x)
    assert out.dtype == x.dtype
    assert torch.all(out.isfinite())
    expected = Q.affine.quantize(x.float(),
                                 min_scale,
                                 torch.zeros([]).float(),
                                 qmin=int(x_int.min()),
                                 qmax=int(x_int.max())).to(dtype)

    assert torch.equal(out.as_subclass(torch.Tensor).long(), expected.long())


@pytest.mark.parametrize(
    'dtype,          bitwidth,  symmetric', [
    (torch.float16,  13,        True),
    (torch.float16,  12,        False),
    (torch.bfloat16, 10,        True),
    (torch.bfloat16, 9,         False),
])
def test_sub_float32_error(dtype, bitwidth, symmetric):
    """
    Given: Input of range [0, 2**bw) * scale
    When: Run Quantize.forward with high bitwidth and sub-float32 dtype
    Then: Throw runtime error
    """
    x_int = get_qtzn_grid(bitwidth, signed=symmetric)
    x = (x_int.double() / x_int.max().double()).to(dtype)

    qtzr = Quantize((), bitwidth, symmetric).to(dtype)

    with qtzr.compute_encodings():
        with pytest.raises(RuntimeError):
            _ = qtzr(x)


@pytest.mark.parametrize(
    'qmin,   qmax,  bitwidth, symmetric', [
    (0,      15,    4,        False),
    (-8,     7,     4,        True),
])
def test_qmin_qmax_consistency(qmin, qmax, bitwidth, symmetric):
    """
    When: Assign new bitwidths
    Then: qmin, qmax should be updated accordingly
    """
    q = Q.affine.Quantize((), qmin, qmax, symmetric)
    x = torch.arange(2**16, dtype=torch.float)
    with q.compute_encodings():
        q(x)

    expected_qmin = qmin
    expected_qmax = qmax
    expected_bitwidth = bitwidth

    while q.bitwidth < 16:
        assert q.qmin == expected_qmin
        assert q.qmax == expected_qmax
        assert q.bitwidth == expected_bitwidth

        q.bitwidth += 1
        expected_qmin *= 2
        expected_qmax = (expected_qmax + 1) * 2 - 1
        expected_bitwidth += 1

    while q.bitwidth > 4:
        assert q.qmin == expected_qmin
        assert q.qmax == expected_qmax
        assert q.bitwidth == expected_bitwidth

        q.bitwidth -= 1
        expected_qmin /= 2
        expected_qmax = (expected_qmax + 1) / 2 - 1
        expected_bitwidth -= 1


def test_attr_translation():
    """
    Given: Quantizer with standard quantization grid [0, 15]
    """
    q = Q.affine.Quantize((), qmin=0, qmax=15, symmetric=False)

    assert q.bitwidth == 4
    assert not q.signed

    """
    When: Assign fractional value to bitwidth
    Then: Throw type error
    """
    with pytest.raises(TypeError):
        q.bitwidth = 0.1

    """
    When: Assign bitwidth < 1
    Then: Throw value error
    """
    with pytest.raises(ValueError):
        q.bitwidth = 0

    """
    When: Assign qtzr.signed = True
    Then:
        1) qtzr.signed getter should return True
        2) qtzr.{qmin, qmax} should be updated accordingly
        3) Other attributes (qtzr.bitwidth) shouldn't change
    """
    q.signed = True
    assert q.signed
    assert q.bitwidth == 4
    assert q.qmin == -8
    assert q.qmax == 7

    """
    When: Assign a floating point number that holds integer values
    Then:
        1) qtzr.bitwidth getter should return 5
        2) qtzr.{qmin, qmax} should be updated accordingly
        3) Other attributes (qtzr.signed) shouldn't change
    """
    q.bitwidth = 5.0
    assert q.bitwidth == 5
    assert q.qmin == -16
    assert q.qmax == 15
    assert q.signed

    """
    Given: Quantizer with non-standard quantization grid [-1, 14]
    """
    q = Q.affine.Quantize((), qmin=-1, qmax=14, symmetric=False)

    """
    When: Call qtzr.{signed, bitwidth} getter
    Then: Throw runtime error
    """
    with pytest.raises(RuntimeError):
        q.signed

    with pytest.raises(RuntimeError):
        q.bitwidth


def test_non_integer_bitwidth():
    """
    Given: Quantizer whose [qmin, qmax] can be represented in the form of
           [0, 2**B-1] or [2**(B-1), 2**(B-1)-1] where B is a positive non-integer
    """
    # 2**15 - 1 < 0xff7f < 2**16 - 1
    q = Q.affine.QuantizeDequantize((), qmin=0, qmax=0xff7f, symmetric=False)

    """
    When: Get bitwidth
    Then: Should return non-integer bitwidth B
    """
    with pytest.raises(RuntimeError):
        q.bitwidth

    """
    When: Set bitwidth
    Then: Should update [qmin, qmax] accordingly
    """
    q.bitwidth = 16
    assert q.bitwidth == 16
    assert q.qmin == 0
    assert q.qmax == 2**16 - 1


@pytest.mark.parametrize(
    'qmin,   qmax,  bitwidth, symmetric', [
    (0,      15,    4,        False),
    (0,      255,   8,        False),
    (0,      65535, 16,       False),
    (-8,     7,     4,        True),
    (-128,   127,   8,        True),
    (-32768, 32767, 16,       True),
])
@pytest.mark.parametrize('qtzr_cls', [Q.affine.Quantize, Q.affine.QuantizeDequantize])
def test_parse_args_equivalence(qtzr_cls, qmin, qmax, bitwidth, symmetric):
    """
    When: Instantiate quantizers with (qmin, qmax) and (bitwidth, signed)
          with mathematically equivalent values
    Then: The output of the quantizers should be equal to each other
    """
    x = torch.arange(qmin, qmax+1, dtype=torch.float32)
    quantizers = [
        qtzr_cls((), qmin, qmax, symmetric),
        qtzr_cls((), qmin, qmax, symmetric=symmetric),
        qtzr_cls((), qmin, qmax=qmax, symmetric=symmetric),
        qtzr_cls((), qmin=qmin, qmax=qmax, symmetric=symmetric),
        qtzr_cls((), bitwidth, symmetric),
        qtzr_cls((), bitwidth, symmetric=symmetric),
        qtzr_cls((), bitwidth=bitwidth, symmetric=symmetric),
    ]

    for qtzr in quantizers:
        with qtzr.compute_encodings():
            _ = qtzr(x)

    for qtzr in quantizers:
        assert torch.equal(qtzr.min, torch.tensor(qmin))
        assert torch.equal(qtzr.max, torch.tensor(qmax))
        assert torch.equal(qtzr.get_scale(), torch.tensor(1.))
        assert torch.equal(qtzr.get_offset(), torch.tensor(0.))
        assert torch.equal(qtzr(x), x)

def test_parse_args_error():
    """
    When: Instantiate with ()
    Then: Throw TypeError
    """
    with pytest.raises(TypeError):
        Quantize()

    """
    When: Instantiate with (tuple,)
    Then: Throw TypeError
    """
    with pytest.raises(TypeError):
        Quantize((1, 10))

    """
    When: Instantiate with (tuple, int)
    Then: Throw TypeError
    """
    with pytest.raises(TypeError):
        Quantize((1, 10), -128)

    """
    When: Instantiate with (tuple, int, int)
    Then: Throw TypeError
    """
    with pytest.raises(TypeError):
        Quantize((1, 10), -128, 127)
    """
    When: Instantiate with (tuple, int, int, bool, 'foo'=any)
    Then: Throw TypeError
    """
    with pytest.raises(TypeError):
        Quantize((1, 10), -128, 127, True, foo=None)

    """
    When: Instantiate with qmin >= qmax
    Then: Throw ValueError
    """
    with pytest.raises(ValueError):
        Quantize((1, 10), 127, -128, True)

    with pytest.raises(ValueError):
        Quantize((1, 10), 127, 127, True)

    """
    When: Instantiate with (tuple, int, bool)
    Then: Create quantizer normally
    """
    Quantize((1, 10), 8, True)

    """
    When: Instantiate with (tuple, int, int, bool)
    Then: Create quantizer normally
    """
    Quantize((1, 10), -128, 127, True)


@torch.no_grad()
@pytest.mark.parametrize('symmetric', [True, False])
def test_signed_doesnt_affect_output(symmetric):
    """
    When: Quantize/Dequantize the same tensor with signed and unsigned quantizers
    Then:
      1) The quantized outputs should be equal with proper shifting
      2) The quantize-dequantized outputs should be equal
    """
    q_int8 = Quantize(shape=(), bitwidth=8, symmetric=symmetric)
    q_int8.signed = True
    q_uint8 = Quantize(shape=(), bitwidth=8, symmetric=symmetric)
    q_uint8.signed = False

    x = torch.arange(-10.0, 6.0)

    with q_int8.compute_encodings(), \
            q_uint8.compute_encodings():
        _ = q_int8(x)
        _ = q_uint8(x)

    out_int8 = q_int8(x)
    out_uint8 = q_uint8(x)
    assert torch.equal(out_int8, out_uint8 - 128)
    assert torch.equal(out_int8.dequantize(), out_uint8.dequantize())

    qdq_int8 = QuantizeDequantize(shape=(), bitwidth=8, symmetric=symmetric)
    qdq_int8.signed = True
    qdq_uint8 = QuantizeDequantize(shape=(), bitwidth=8, symmetric=symmetric)
    qdq_uint8.signed = False

    x = torch.arange(-10.0, 6.0)

    with qdq_int8.compute_encodings(), \
            qdq_uint8.compute_encodings():
        _ = qdq_int8(x)
        _ = qdq_uint8(x)

    out_int8 = qdq_int8(x)
    out_uint8 = qdq_uint8(x)
    assert torch.equal(out_int8, out_uint8)
    assert torch.equal(out_int8.quantize(), out_uint8.quantize() - 128)

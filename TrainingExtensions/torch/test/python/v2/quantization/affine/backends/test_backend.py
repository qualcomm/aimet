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
import random
import torch
import pytest
from collections import namedtuple
from aimet_torch.v2.quantization import affine
from aimet_torch.v2.quantization.affine.backends import torch_builtins
from aimet_torch.v2.utils import ste_round

VectorSetForTest = namedtuple(
    "VectorSetForTest",
    ["tensor", "tensor_q", "tensor_qdq", "mask", "delta", "offset", "qmin", "qmax"],
)

bfloat16_compat_per_tensor_4b_test_set = VectorSetForTest(
    tensor=torch.tensor(
        [
            [-1004.0, -1000.0, -15.375, -11.25, -3.25, 0.25, 500.0],
            [-1.375, -0.75, -0.125, 0, 1.125, 3, 10],
        ]
    ),
    tensor_q=torch.tensor([[0, 0, 0, 0, 0, 5, 15], [2, 3, 5, 5, 7, 11, 15]]),
    tensor_qdq=torch.tensor(
        [
            [-2.5, -2.5, -2.5, -2.5, -2.5, 0.0, 5.0],
            [-1.5, -1.0, 0.0, 0.0, 1.0, 3.0, 5.0],
        ]
    ),
    mask=torch.tensor([[1, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 1]], dtype=torch.bool),
    delta=torch.tensor([0.5]),
    offset=torch.tensor([-5]),
    qmin=0,
    qmax=15,
)

bfloat16_compat_per_tensor_8b_test_set = VectorSetForTest(
    tensor=torch.tensor(
        [
            [-1004.0, -1000.0, -15.375, -11.25, -3.25, 0.25, 500.0],
            [-1.375, -0.75, -0.125, 0, 1.125, 3, 10],
        ]
    ),
    tensor_q=torch.tensor(
        [[0, 0, 102, 111, 127, 133, 255], [130, 131, 133, 133, 135, 139, 153]]
    ),
    tensor_qdq=torch.tensor(
        [
            [-66.5, -66.5, -15.5, -11, -3.0, 0, 61],
            [-1.5, -1, 0, 0, 1, 3, 10.0],
        ]
    ),
    mask=torch.tensor([[1, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
    delta=torch.tensor([0.5]),
    offset=torch.tensor([-133]),
    qmin=0,
    qmax=255,
)

per_tensor_4b_test_set = VectorSetForTest(
    tensor=torch.tensor(
        [
            [-1005.8, -1000.1, -15.4, -11.24, -3.3, 0.2, 500.4],
            [-1.3, -0.76, -0.11, 0, 1.01, 3, 10.4],
        ]
    ),
    tensor_q=torch.tensor([[0, 0, 0, 0, 0, 5, 15], [2, 3, 5, 5, 7, 11, 15]]),
    tensor_qdq=torch.tensor(
        [
            [-2.5, -2.5, -2.5, -2.5, -2.5, 0.0, 5.0],
            [-1.5, -1.0, 0.0, 0.0, 1.0, 3.0, 5.0],
        ]
    ),
    mask=torch.tensor([[1, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 1]], dtype=torch.bool),
    delta=torch.tensor([0.5]),
    offset=torch.tensor([-5]),
    qmin=0,
    qmax=15,
)

per_tensor_8b_test_set = VectorSetForTest(
    tensor=torch.tensor(
        [
            [-1005.8, -1000.1, -15.4, -11.24, -3.3, 0.2, 500.4],
            [-1.3, -0.76, -0.11, 0, 1.01, 3, 10.4],
        ]
    ),
    tensor_q=torch.tensor(
        [[0, 0, 102, 111, 126, 133, 255], [130, 131, 133, 133, 135, 139, 154]]
    ),
    tensor_qdq=torch.tensor(
        [
            [-66.5, -66.5, -15.5, -11, -3.5, 0, 61],
            [-1.5, -1, 0, 0, 1, 3, 10.5],
        ]
    ),
    mask=torch.tensor([[1, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
    delta=torch.tensor([0.5]),
    offset=torch.tensor([-133]),
    qmin=0,
    qmax=255,
)

per_channel_4b_test_set = VectorSetForTest(
    tensor=torch.tensor(
        [
            [-1005.8, -1000.1, -15.4, -11.24, -3.3, 0.2, 500.4],
            [-1.3, -0.76, -0.11, 0, 1.01, 3, 10.4],
        ]
    ),
    tensor_q=torch.tensor([[0, 0, 0, 0, 6, 13, 15], [0, 0, 5, 7, 15, 15, 15]]),
    tensor_qdq=torch.tensor(
        [
            [-6.5, -6.5, -6.5, -6.5, -3.5, 0, 1.0],
            [-0.4375, -0.4375, -0.125, 0.0, 0.5, 0.5, 0.5],
        ]
    ),
    mask=torch.tensor([[1, 1, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1, 1]], dtype=torch.bool),
    delta=torch.tensor([[0.5], [0.0625]]),
    offset=torch.tensor([[-13], [-7]]),
    qmin=0,
    qmax=15,
)

per_channel_8b_test_set = VectorSetForTest(
    tensor=torch.tensor(
        [
            [-1005.8, -1000.1, -15.4, -11.24, -3.3, 0.2, 500.4],
            [-1.3, -0.76, -0.11, 0, 1.01, 3, 10.4],
        ]
    ),
    tensor_q=torch.tensor(
        [[0, 0, 102, 111, 126, 133, 255], [106, 115, 125, 127, 143, 175, 255]]
    ),
    tensor_qdq=torch.tensor(
        [
            [-66.5, -66.5, -15.5, -11, -3.5, 0, 61],
            [-1.3125, -0.75, -0.125, 0.0, 1.0, 3.0, 8.0],
        ]
    ),
    mask=torch.tensor([[1, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]], dtype=torch.bool),
    delta=torch.tensor([[0.5], [0.0625]]),
    offset=torch.tensor([[-133], [-127]]),
    qmin=0,
    qmax=255,
)


class AutogradQuantizationModule(torch.nn.Module):
    def __init__(self, scale, offset, qmin, qmax):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.scale = torch.nn.Parameter(scale.clone())
        self.offset = torch.nn.Parameter(offset.clone())

    def forward(self, x):
        return torch.clamp(
            ste_round(x / self.scale) - ste_round(self.offset), self.qmin, self.qmax
        )


class AutogradDequantizationModule(torch.nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.scale = torch.nn.Parameter(scale.clone())
        self.offset = torch.nn.Parameter(offset.clone())

    def forward(self, x):
        return (x + ste_round(self.offset)) * self.scale


class AutogradQuantDequantModule(torch.nn.Module):
    def __init__(self, scale, offset, qmin, qmax):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.scale = torch.nn.Parameter(scale.clone())
        self.offset = torch.nn.Parameter(offset.clone())

    def forward(self, x):
        x_q = torch.clamp(
            ste_round(x / self.scale) - ste_round(self.offset), self.qmin, self.qmax
        )
        x_dq = (x_q + ste_round(self.offset)) * self.scale
        return x_dq


def copy_test_set(
    test_set: namedtuple,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    new_test_set = VectorSetForTest(
        tensor=test_set.tensor.clone().detach().to(device, dtype),
        tensor_q=test_set.tensor_q.clone().detach().to(device, dtype),
        tensor_qdq=test_set.tensor_qdq.clone().detach().to(device, dtype),
        mask=test_set.mask.clone().to(device),
        delta=test_set.delta.clone().detach().to(device, dtype),
        offset=test_set.offset.clone().detach().to(device, dtype),
        qmin=test_set.qmin,
        qmax=test_set.qmax,
    )
    return new_test_set


def get_round_safe_quantizable_tensor(
    size: tuple, scale: torch.Tensor, qmin: int, qmax: int
):
    """
    Returns round-safe quantizable random tensor by forcing
    fractional part of tensor divided by scale not to be near 0.5
    """
    return scale.cpu() * (
        torch.randint(qmin, qmax + 1, size).to(torch.float32)
        + torch.rand(size) * 0.8
        - 0.4
    )


def get_random_quantized_tensor(size: tuple, qmin: int, qmax: int):
    return torch.randint(qmin, qmax + 1, size).to(torch.float32)


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(19521)
    torch.random.manual_seed(19521)


@pytest.fixture(scope="session")
def offset():
    return torch.randint(-5, 5, []).to(torch.float32)


if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0):

    @pytest.fixture(params=[True, False])
    def use_compiled_impl(request):
        flag = request.param
        with torch_builtins._use_compiled_impl(flag):
            yield
else:

    @pytest.fixture
    def use_compiled_impl():
        with torch_builtins._use_compiled_impl(False):
            yield


@pytest.mark.parametrize("backend_module", [torch_builtins])
class TestQuantizationBackends:
    @pytest.mark.parametrize("qmin, qmax", [(0, 255), (-128, 127)])
    @pytest.mark.parametrize("scale_shape", [(4), (2,), (3, 1), (2, 1, 1)])
    def test_quantize_using_not_broadcastable_scale(
        self, backend_module, offset, scale_shape, qmin, qmax, use_compiled_impl
    ):
        # Add small value to scale to make scale not equal to 0
        scale = torch.rand(scale_shape)
        scale[scale == 0.0] = 0.1
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), qmin, qmax)

        with pytest.raises(RuntimeError):
            backend_module.quantize(random_tensor, scale, offset, qmin, qmax)

        with pytest.raises(RuntimeError):
            backend_module.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_module.quantize_dequantize(random_tensor, scale, offset, qmin, qmax)

    def test_quantize_using_wide_quantization_range(self, backend_module, offset):
        scale = torch.tensor([0.2])
        random_tensor = torch.randn(2, 3, 4, 5)

        float = torch.float32
        half = torch.half

        """
        When: [qmin, qmax] = [0, 2**16-1]
        Then: quantize() with float16 input throws runtime error
        """
        qmin, qmax = 0, 2**16 - 1
        with pytest.raises(RuntimeError):
            # float16 is unable to represent output of [0, 2**16-1]
            backend_module.quantize(
                random_tensor.to(half), scale.to(half), offset.to(half), qmin, qmax
            )
        backend_module.quantize(
            random_tensor.to(float), scale.to(float), offset.to(float), qmin, qmax
        )

        # No runtime error; Internally fall back to float32 to perform qdq
        backend_module.quantize_dequantize(
            random_tensor.to(half), scale.to(half), offset.to(half), qmin, qmax
        )
        backend_module.quantize_dequantize(
            random_tensor.to(float), scale.to(float), offset.to(float), qmin, qmax
        )

        """
        When: [qmin, qmax] = [0, 2**32-1]
        Then: quantize() with both float16 and float32 input throws runtime error
        """
        qmin, qmax = 0, 2**32 - 1
        with pytest.raises(RuntimeError):
            backend_module.quantize(
                random_tensor.to(half), scale.to(half), offset.to(half), qmin, qmax
            )
        with pytest.raises(RuntimeError):
            backend_module.quantize(
                random_tensor.to(float), scale.to(float), offset.to(float), qmin, qmax
            )

        # No runtime error; Internally fall back to float32 to perform qdq
        backend_module.quantize_dequantize(
            random_tensor.to(half), scale.to(half), offset.to(half), qmin, qmax
        )
        backend_module.quantize_dequantize(
            random_tensor.to(float), scale.to(float), offset.to(float), qmin, qmax
        )

        """
        When: [qmin, qmax] = [0, 2**64-1]
        Then: quantize() and quantize_dequantize() of both float16 and float32 input throw runtime error
        """
        qmin, qmax = 0, 2**64 - 1
        with pytest.raises(RuntimeError):
            # Both float32 and float16 are unable to represent output of [0, 2**64-1]
            backend_module.quantize(
                random_tensor.to(half), scale.to(half), offset.to(half), qmin, qmax
            )
        with pytest.raises(RuntimeError):
            # Both float32 and float16 are unable to represent output of [0, 2**64-1]
            backend_module.quantize(
                random_tensor.to(float), scale.to(float), offset.to(float), qmin, qmax
            )
        with pytest.raises((RuntimeError, ValueError)):
            # Intermediate ouput of [0, 2**64-1] cannot be represented by internal dtype float32
            backend_module.quantize_dequantize(
                random_tensor.to(half), scale.to(half), offset.to(half), qmin, qmax
            )
        with pytest.raises((RuntimeError, ValueError)):
            # Intermediate ouput of [0, 2**64-1] cannot be represented by internal dtype float32
            backend_module.quantize_dequantize(
                random_tensor.to(float), scale.to(float), offset.to(float), qmin, qmax
            )

    @pytest.mark.cuda
    def test_quantize_using_parameters_on_different_device(
        self, backend_module, offset, use_compiled_impl
    ):
        qmin, qmax = 0, 255
        scale = torch.tensor([0.2], dtype=torch.float32)
        random_tensor = torch.randn(2, 3, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), qmin, qmax)

        with pytest.raises(RuntimeError):
            backend_module.quantize(random_tensor.cuda(), scale, offset, qmin, qmax)
        with pytest.raises(RuntimeError):
            backend_module.quantize(random_tensor, scale.cuda(), offset, qmin, qmax)
        with pytest.raises(RuntimeError):
            backend_module.quantize(random_tensor, scale, offset.cuda(), qmin, qmax)

        with pytest.raises(RuntimeError):
            backend_module.dequantize(random_quantized_tensor.cuda(), scale, offset)
        with pytest.raises(RuntimeError):
            backend_module.dequantize(random_quantized_tensor, scale.cuda(), offset)
        with pytest.raises(RuntimeError):
            backend_module.dequantize(random_quantized_tensor, scale, offset.cuda())

        with pytest.raises(RuntimeError):
            backend_module.quantize_dequantize(
                random_tensor.cuda(), scale, offset, qmin, qmax
            )
        with pytest.raises(RuntimeError):
            backend_module.quantize_dequantize(
                random_tensor, scale.cuda(), offset, qmin, qmax
            )
        with pytest.raises(RuntimeError):
            backend_module.quantize_dequantize(
                random_tensor, scale, offset.cuda(), qmin, qmax
            )

    @pytest.mark.parametrize(
        "memory_format", [torch.channels_last, torch.channels_last_3d]
    )
    def test_quantize_using_non_contiguous_tensor(
        self, backend_module, offset, memory_format, use_compiled_impl
    ):
        qmin, qmax = 0, 255
        scale = torch.tensor([0.2], dtype=torch.float32)
        random_tensor = get_round_safe_quantizable_tensor(
            (2, 3, 4, 5), scale, qmin, qmax
        )
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), qmin, qmax)

        # Rank 5 tensor is required to use channels_last_3d format
        if memory_format == torch.channels_last_3d:
            random_tensor = random_tensor[..., None]
            random_quantized_tensor = random_quantized_tensor[..., None]

        channel_last_random_tensor = random_tensor.to(memory_format=memory_format)
        channel_last_random_quantized_tensor = random_quantized_tensor.to(
            memory_format=memory_format
        )

        channel_last_quantized_tensor = backend_module.quantize(
            channel_last_random_tensor, scale, offset, qmin, qmax
        )
        assert channel_last_quantized_tensor.is_contiguous(memory_format=memory_format)

        channel_last_dequantized_tensor = backend_module.dequantize(
            channel_last_random_quantized_tensor, scale, offset
        )
        assert channel_last_dequantized_tensor.is_contiguous(
            memory_format=memory_format
        )

        channel_last_qdq_tensor = backend_module.quantize_dequantize(
            channel_last_random_tensor, scale, offset, qmin, qmax
        )
        assert channel_last_qdq_tensor.is_contiguous(memory_format=memory_format)

        expected_quantized_tensor = torch.clamp(
            torch.round(random_tensor / scale) - torch.round(offset), qmin, qmax
        )
        assert torch.allclose(channel_last_quantized_tensor, expected_quantized_tensor)

        expected_dequantized_tensor = (
            random_quantized_tensor + torch.round(offset)
        ) * scale
        assert torch.allclose(
            channel_last_dequantized_tensor, expected_dequantized_tensor
        )

        expected_qdq_tensor = (expected_quantized_tensor + torch.round(offset)) * scale
        assert torch.allclose(channel_last_qdq_tensor, expected_qdq_tensor)

    @pytest.mark.parametrize("scale_shape", [(5, 1, 1, 1, 1), (3, 1, 1)])
    def test_quantize_using_inversely_broadcastable_scale(
        self, backend_module, offset, scale_shape, use_compiled_impl
    ):
        qmin, qmax = 0, 255
        # Add small value to scale to make scale not equal to 0
        scale = torch.rand(scale_shape)
        scale[scale == 0.0] = 0.1
        random_tensor = torch.randn(2, 1, 4, 5)
        random_quantized_tensor = get_random_quantized_tensor((2, 1, 4, 5), qmin, qmax)

        with pytest.raises(RuntimeError):
            backend_module.quantize(random_tensor, scale, offset, qmin, qmax)

        with pytest.raises(RuntimeError):
            backend_module.dequantize(random_quantized_tensor, scale, offset)

        with pytest.raises(RuntimeError):
            backend_module.quantize_dequantize(random_tensor, scale, offset, qmin, qmax)

    @pytest.mark.parametrize("scale_requires_grad", [True, False])
    @pytest.mark.parametrize("offset_requires_grad", [True, False])
    @pytest.mark.parametrize("input_requires_grad", [True, False])
    def test_quantize_backward_pass(
        self,
        backend_module,
        offset,
        scale_requires_grad,
        offset_requires_grad,
        input_requires_grad,
        use_compiled_impl,
    ):
        qmin, qmax = 0, 255
        scale = torch.rand([])
        scale[scale == 0.0] = 0.1
        offset = offset.detach().clone()
        random_tensor = get_round_safe_quantizable_tensor(
            (2, 3, 4, 5), scale, qmin, qmax
        )
        scale.requires_grad = scale_requires_grad
        offset.requires_grad = offset_requires_grad
        random_tensor.requires_grad = input_requires_grad

        qdq_tensor = backend_module.quantize_dequantize(
            random_tensor, scale, offset, qmin, qmax
        )
        loss = torch.sum((random_tensor - qdq_tensor) ** 2)
        if loss.requires_grad:
            loss.backward()

        if scale_requires_grad:
            assert scale.grad is not None
        else:
            assert scale.grad is None

        if offset_requires_grad:
            assert offset.grad is not None
        else:
            assert offset.grad is None

        if input_requires_grad:
            assert random_tensor.grad is not None
        else:
            assert random_tensor.grad is None

    @pytest.mark.parametrize(
        "test_set",
        (
            per_tensor_4b_test_set,
            per_tensor_8b_test_set,
            per_channel_4b_test_set,
            per_channel_8b_test_set,
        ),
    )
    @pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
    @pytest.mark.cuda
    def test_quantize_with_predefined_values(
        self, backend_module, test_set, dtype, use_compiled_impl
    ):
        test_set = copy_test_set(test_set, device="cuda:0", dtype=dtype)
        test_set.tensor.requires_grad = True
        tensor_q = backend_module.quantize(
            test_set.tensor,
            test_set.delta,
            test_set.offset,
            test_set.qmin,
            test_set.qmax,
        )
        expected_tensor_q = test_set.tensor_q
        assert torch.all(tensor_q == expected_tensor_q)
        grad_in = torch.randn_like(test_set.tensor)
        tensor_q.backward(grad_in)
        assert torch.all(test_set.tensor.grad[test_set.mask] == 0)
        assert torch.allclose(
            test_set.tensor.grad[~test_set.mask],
            (grad_in / test_set.delta)[~test_set.mask],
        )

    @pytest.mark.parametrize(
        "test_set",
        (
            per_tensor_4b_test_set,
            per_tensor_8b_test_set,
            per_channel_4b_test_set,
            per_channel_8b_test_set,
        ),
    )
    @pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
    @pytest.mark.cuda
    def test_dequantize_with_predefined_values(
        self, backend_module, test_set, dtype, use_compiled_impl
    ):
        test_set = copy_test_set(test_set, dtype=dtype, device="cuda")
        tensor_q = test_set.tensor_q
        tensor_qdq = backend_module.dequantize(
            tensor_q, test_set.delta, test_set.offset
        )
        assert torch.all(tensor_qdq == test_set.tensor_qdq)

    @pytest.mark.parametrize(
        "test_set",
        (
            per_tensor_4b_test_set,
            per_tensor_8b_test_set,
            per_channel_4b_test_set,
            per_channel_8b_test_set,
        ),
    )
    @pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
    @pytest.mark.cuda
    def test_qdq_with_predefined_values(
        self, backend_module, test_set, dtype, use_compiled_impl
    ):
        test_set = copy_test_set(test_set, dtype=dtype, device="cuda")
        test_set.tensor.requires_grad = True
        tensor_qdq = backend_module.quantize_dequantize(
            test_set.tensor,
            test_set.delta,
            test_set.offset,
            test_set.qmin,
            test_set.qmax,
        )
        assert torch.allclose(tensor_qdq, test_set.tensor_qdq)
        grad_in = torch.randn_like(test_set.tensor)
        tensor_qdq.backward(grad_in)
        assert torch.all(test_set.tensor.grad[test_set.mask] == 0)
        assert torch.all(
            test_set.tensor.grad[~test_set.mask] == grad_in[~test_set.mask]
        )

    @pytest.mark.parametrize(
        "test_set",
        (
            bfloat16_compat_per_tensor_4b_test_set,
            bfloat16_compat_per_tensor_8b_test_set,
        ),
    )
    @pytest.mark.cuda
    def test_quantize_with_predefined_bfloat_values(self, backend_module, test_set):
        test_set = copy_test_set(test_set, device="cuda:0", dtype=torch.bfloat16)
        test_set.tensor.requires_grad = True
        tensor_q = backend_module.quantize(
            test_set.tensor,
            test_set.delta,
            test_set.offset,
            test_set.qmin,
            test_set.qmax,
        )
        assert torch.all(tensor_q == test_set.tensor_q)
        grad_in = torch.randn_like(test_set.tensor)
        tensor_q.backward(grad_in)
        assert torch.all(test_set.tensor.grad[test_set.mask] == 0)
        assert torch.allclose(
            test_set.tensor.grad[~test_set.mask],
            (grad_in / test_set.delta)[~test_set.mask],
        )

    @pytest.mark.parametrize(
        "test_set",
        (
            bfloat16_compat_per_tensor_4b_test_set,
            bfloat16_compat_per_tensor_8b_test_set,
        ),
    )
    @pytest.mark.cuda
    def test_dequantize_with_predefined_bfloat_values(self, backend_module, test_set):
        test_set = copy_test_set(
            per_tensor_8b_test_set, dtype=torch.bfloat16, device="cuda"
        )
        tensor_q = test_set.tensor_q
        tensor_qdq = backend_module.dequantize(
            tensor_q, test_set.delta, test_set.offset
        )
        assert torch.all(tensor_qdq == test_set.tensor_qdq)

    @pytest.mark.parametrize(
        "test_set",
        (
            bfloat16_compat_per_tensor_4b_test_set,
            bfloat16_compat_per_tensor_8b_test_set,
        ),
    )
    @pytest.mark.cuda
    def test_qdq_with_predefined_bfloat_values(self, backend_module, test_set):
        test_set = copy_test_set(test_set, dtype=torch.bfloat16, device="cuda")
        test_set.tensor.requires_grad = True
        tensor_qdq = backend_module.quantize_dequantize(
            test_set.tensor,
            test_set.delta,
            test_set.offset,
            test_set.qmin,
            test_set.qmax,
        )
        assert torch.allclose(tensor_qdq, test_set.tensor_qdq)
        grad_in = torch.randn_like(test_set.tensor)
        tensor_qdq.backward(grad_in)
        assert torch.all(test_set.tensor.grad[test_set.mask] == 0)
        assert torch.all(
            test_set.tensor.grad[~test_set.mask] == grad_in[~test_set.mask]
        )

    @pytest.mark.parametrize("qmin, qmax", [(0, 255), (-128, 127)])
    def test_compare_quantize_gradients_with_autograd_results(
        self, backend_module, offset, qmin, qmax, use_compiled_impl
    ):
        scale = torch.rand([])
        scale[scale == 0.0] = 0.1
        offset = offset.detach().clone()
        random_tensor = get_round_safe_quantizable_tensor(
            (2, 3, 4, 5), scale, qmin, qmax
        )
        random_tensor_for_autograd = random_tensor.detach().clone()

        scale.requires_grad = True
        offset.requires_grad = True
        random_tensor.requires_grad = True
        random_tensor_for_autograd.requires_grad = True

        autograd_based_module = AutogradQuantizationModule(scale, offset, qmin, qmax)
        expected_tensor_q = autograd_based_module(random_tensor_for_autograd)
        tensor_q = backend_module.quantize(random_tensor, scale, offset, qmin, qmax)
        assert torch.allclose(tensor_q, expected_tensor_q)

        grad_in = torch.randn_like(random_tensor)
        expected_tensor_q.backward(grad_in)
        tensor_q.backward(grad_in)

        expected_tensor_grad = random_tensor_for_autograd.grad
        expected_scale_grad = autograd_based_module.scale.grad
        expected_offset_grad = autograd_based_module.offset.grad

        assert torch.allclose(random_tensor.grad, expected_tensor_grad)
        assert torch.allclose(scale.grad, expected_scale_grad)
        assert torch.allclose(offset.grad, expected_offset_grad)

    @pytest.mark.parametrize("qmin, qmax", [(0, 255), (-128, 127)])
    def test_compare_dequantize_gradients_with_autograd_results(
        self, backend_module, offset, qmin, qmax, use_compiled_impl
    ):
        scale = torch.rand([])
        scale[scale == 0.0] = 0.1
        offset = offset.detach().clone()
        random_quantized_tensor = get_random_quantized_tensor((2, 3, 4, 5), qmin, qmax)
        random_quantized_tensor_for_autograd = random_quantized_tensor.detach().clone()

        scale.requires_grad = True
        offset.requires_grad = True
        random_quantized_tensor.requires_grad = True
        random_quantized_tensor_for_autograd.requires_grad = True

        autograd_based_module = AutogradDequantizationModule(scale, offset)
        expected_tensor_dq = autograd_based_module(random_quantized_tensor_for_autograd)
        tensor_dq = backend_module.dequantize(random_quantized_tensor, scale, offset)
        assert torch.allclose(tensor_dq, expected_tensor_dq)

        grad_in = torch.randn_like(random_quantized_tensor)
        expected_tensor_dq.backward(grad_in)
        tensor_dq.backward(grad_in)

        expected_tensor_grad = random_quantized_tensor_for_autograd.grad
        expected_scale_grad = autograd_based_module.scale.grad
        expected_offset_grad = autograd_based_module.offset.grad

        assert torch.allclose(random_quantized_tensor.grad, expected_tensor_grad)
        assert torch.allclose(scale.grad, expected_scale_grad)
        assert torch.allclose(offset.grad, expected_offset_grad)

    @pytest.mark.parametrize("qmin, qmax", [(0, 255), (-128, 127)])
    def test_compare_qdq_gradients_with_autograd_results(
        self, backend_module, offset, qmin, qmax, use_compiled_impl
    ):
        scale = torch.rand([])
        scale[scale == 0.0] = 0.1
        offset = offset.detach().clone()
        random_tensor = get_round_safe_quantizable_tensor(
            (2, 3, 4, 5), scale, qmin, qmax
        )
        random_tensor_for_autograd = random_tensor.detach().clone()

        scale.requires_grad = True
        offset.requires_grad = True
        random_tensor.requires_grad = True
        random_tensor_for_autograd.requires_grad = True

        autograd_based_module = AutogradQuantDequantModule(scale, offset, qmin, qmax)
        expected_tensor_qdq = autograd_based_module(random_tensor_for_autograd)
        tensor_qdq = backend_module.quantize_dequantize(
            random_tensor, scale, offset, qmin, qmax
        )
        assert torch.allclose(tensor_qdq, expected_tensor_qdq)

        grad_in = torch.randn_like(random_tensor)
        expected_tensor_qdq.backward(grad_in)
        tensor_qdq.backward(grad_in)

        expected_tensor_grad = random_tensor_for_autograd.grad
        expected_scale_grad = autograd_based_module.scale.grad
        expected_offset_grad = autograd_based_module.offset.grad

        assert torch.allclose(random_tensor.grad, expected_tensor_grad, rtol=1e-3)
        assert torch.allclose(scale.grad, expected_scale_grad, rtol=1e-3)
        assert torch.allclose(offset.grad, expected_offset_grad, rtol=1e-3)

    def test_block_size(self, backend_module):
        scale = torch.randn(4, 3, 8, 1)
        offset = torch.randint(low=-128, high=127, size=(4, 3, 8, 1)).to(
            dtype=scale.dtype
        )
        inp = torch.randn(8, 6, 8, 3)
        block_size = [-1, 2, 1, 3]

        reshaped_inp = torch_builtins.reshape_tensor_for_blocks(
            inp, scale.shape, block_size
        )
        assert reshaped_inp.shape == (4, 2, 3, 2, 8, 1, 1, 3)

        reshaped_scale = scale.view(
            torch_builtins.get_encoding_shape_with_blocks(scale.shape, block_size)
        )
        assert reshaped_scale.shape == (4, 1, 3, 1, 8, 1, 1, 1)

        q = affine.quantize(inp, scale, offset, 8, True, block_size=block_size)
        dq = affine.dequantize(q, scale, offset, block_size=block_size)
        qdq = affine.quantize_dequantize(
            inp, scale, offset, 8, True, block_size=block_size
        )

        assert q.shape == inp.shape
        assert dq.shape == inp.shape
        assert qdq.shape == inp.shape

    @pytest.mark.parametrize(
        "scale, offset, block_size, output",
        [
            [
                torch.tensor([[0.03, 0.02]]),
                torch.zeros(1, 2),
                [2, 1],
                torch.tensor([[-40, 120], [-20, -9]]),
            ],
            [
                torch.tensor([[0.03, 0.02]]),
                torch.zeros(1, 2),
                [-1, 1],
                torch.tensor([[-40, 120], [-20, -9]]),
            ],
            [
                torch.tensor([[0.03, 0.02]]),
                torch.zeros(1, 2),
                None,
                torch.tensor([[-40, 120], [-20, -9]]),
            ],
            [
                torch.tensor([[0.03], [0.02]]),
                torch.zeros(2, 1),
                [1, 2],
                torch.tensor([[-40, 80], [-30, -9]]),
            ],
            [
                torch.tensor([[0.03], [0.02]]),
                torch.zeros(2, 1),
                [1, -1],
                torch.tensor([[-40, 80], [-30, -9]]),
            ],
            [
                torch.tensor([[0.03], [0.02]]),
                torch.zeros(2, 1),
                None,
                torch.tensor([[-40, 80], [-30, -9]]),
            ],
            [
                torch.tensor([[0.03, 0.02], [0.01, 0.09]]),
                torch.zeros(2, 2),
                [1, 1],
                torch.tensor([[-40, 120], [-60, -2]]),
            ],
        ],
    )
    def test_block_quant(
        self, backend_module, scale, offset, block_size, output, use_compiled_impl
    ):
        inp = torch.tensor([[-1.2, 2.4], [-0.6, -0.18]])

        q = affine.quantize(
            inp, scale, offset.to(scale.dtype), 8, True, block_size=block_size
        )
        assert torch.equal(q, output.to(q.dtype))

        dq = affine.dequantize(q, scale, offset.to(scale.dtype), block_size=block_size)
        assert torch.allclose(dq, inp, atol=1e-6)

        qdq = affine.quantize_dequantize(
            inp, scale, offset.to(scale.dtype), 8, True, block_size=block_size
        )
        assert torch.allclose(qdq, inp, atol=1e-6)

    def test_block_quant_2(self, backend_module, use_compiled_impl):
        inp = torch.randn(3, 5, 4, 6, 12, 9)
        scale = torch.randn(2, 1, 4, 9)
        offset = torch.randint(low=-128, high=127, size=(2, 1, 4, 9)).to(
            dtype=scale.dtype
        )
        block_size = [2, 6, 3, 1]
        q = affine.quantize(inp, scale, offset, 8, block_size=block_size)
        dq = affine.dequantize(q, scale, offset, block_size=block_size)
        qdq = affine.quantize_dequantize(inp, scale, offset, 8, block_size=block_size)

        for i in range(scale.shape[0]):
            for j in range(scale.shape[1]):
                for k in range(scale.shape[2]):
                    for l in range(scale.shape[3]):
                        inp_block = inp[
                            ...,
                            i * block_size[0] : (i + 1) * block_size[0],
                            j * block_size[1] : (j + 1) * block_size[1],
                            k * block_size[2] : (k + 1) * block_size[2],
                            l * block_size[3] : (l + 1) * block_size[3],
                        ]
                        q_block = q[
                            ...,
                            i * block_size[0] : (i + 1) * block_size[0],
                            j * block_size[1] : (j + 1) * block_size[1],
                            k * block_size[2] : (k + 1) * block_size[2],
                            l * block_size[3] : (l + 1) * block_size[3],
                        ]
                        dq_block = dq[
                            ...,
                            i * block_size[0] : (i + 1) * block_size[0],
                            j * block_size[1] : (j + 1) * block_size[1],
                            k * block_size[2] : (k + 1) * block_size[2],
                            l * block_size[3] : (l + 1) * block_size[3],
                        ]
                        qdq_block = qdq[
                            ...,
                            i * block_size[0] : (i + 1) * block_size[0],
                            j * block_size[1] : (j + 1) * block_size[1],
                            k * block_size[2] : (k + 1) * block_size[2],
                            l * block_size[3] : (l + 1) * block_size[3],
                        ]

                        assert torch.equal(
                            q_block,
                            affine.quantize(
                                inp_block, scale[i, j, k, l], offset[i, j, k, l], 8
                            ),
                        )
                        assert torch.equal(
                            dq_block,
                            affine.dequantize(
                                q_block, scale[i, j, k, l], offset[i, j, k, l]
                            ),
                        )
                        assert torch.equal(
                            qdq_block,
                            affine.quantize_dequantize(
                                inp_block, scale[i, j, k, l], offset[i, j, k, l], 8
                            ),
                        )

    def test_invalid_block_size(self, backend_module):
        # Block size length must match scale
        with pytest.raises(RuntimeError):
            backend_module._validate_arguments(
                torch.randn(4, 8),
                torch.randn(2, 2),
                torch.randn(2, 2),
                block_size=[2, 4, 1],
            )
        backend_module._validate_arguments(
            torch.randn(4, 8), torch.randn(2, 2), torch.randn(2, 2), block_size=[2, 4]
        )

        # Scale dimension must divide evenly with input dimension
        with pytest.raises(RuntimeError):
            backend_module._validate_arguments(
                torch.randn(1, 4),
                torch.randn(1, 3),
                torch.randn(1, 2),
                block_size=[1, -1],
            )
        backend_module._validate_arguments(
            torch.randn(1, 4), torch.randn(1, 2), torch.randn(1, 2), block_size=[1, -1]
        )

        # Block dim size * scale dim size must equal input dim size
        with pytest.raises(RuntimeError):
            backend_module._validate_arguments(
                torch.randn(1, 4),
                torch.randn(1, 4),
                torch.randn(1, 2),
                block_size=[1, 3],
            )
        backend_module._validate_arguments(
            torch.randn(1, 4), torch.randn(1, 2), torch.randn(1, 2), block_size=[1, 2]
        )


@pytest.mark.parametrize(
    "qmin,   qmax,    offset",
    [
        (-8, 7, 0),
        (0, 15, 0),
        (0, 15, 8),
        (-128, 127, 0),
        (0, 255, 0),
        (0, 255, 128),
        (-(2**15), 2**15 - 1, 0),
        (0, 2**16 - 1, 0),
        (0, 2**16 - 1, 2**15),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", *(("cuda",) if torch.cuda.is_available() else ())]
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize("requires_grad", [True, False])
def test_cross_validate_torch_fake_quantize(
    qmin, qmax, offset, dtype, device, requires_grad
):
    """
    Given same inputs, the following three functions should always produce the same output
      * quantize_dequantize
      * QuantDequantFunc.apply
      * _torch_fake_quantize
    """
    scale = torch.tensor(
        [0.1], dtype=torch.float32, device=device, requires_grad=requires_grad
    )
    offset = torch.tensor(
        [offset], dtype=torch.float32, device=device, requires_grad=requires_grad
    )
    tensor = scale * torch.tensor(
        [qmin - 0.5, qmin, qmin + 0.5, qmax - 0.5, qmax, qmax + 0.5], device=device
    )
    tensor = tensor.to(dtype)

    expected = (
        tensor.to(torch.float32)
        .div(scale)
        .round()
        .sub(offset)
        .clamp(qmin, qmax)
        .add(offset)
        .mul(scale)
        .to(dtype)
    )

    # Allow off-by-one error for float16 and bfloat16
    atol = scale.item() if dtype in (torch.float16, torch.bfloat16) else 1e-8

    out1 = torch_builtins.quantize_dequantize(tensor, scale, offset, qmin, qmax)
    out2 = torch_builtins.QuantDequantFunc.apply(tensor, scale, offset, qmin, qmax).to(
        dtype
    )
    out3 = torch_builtins._torch_fake_quantize(
        tensor, scale.detach(), offset.detach(), qmin, qmax
    )

    assert torch.allclose(out1, expected, atol=atol)
    assert torch.allclose(out2, expected, atol=atol)
    if out3 is not None:
        assert torch.allclose(out3, expected, atol=atol)

    scale = torch.stack(
        [
            scale,
            scale,
        ]
    )
    offset = torch.stack(
        [
            offset,
            offset,
        ]
    )
    tensor = torch.stack([tensor, tensor])
    expected = torch.stack(
        [
            expected,
            expected,
        ]
    )

    out1 = torch_builtins.quantize_dequantize(tensor, scale, offset, qmin, qmax)
    out2 = torch_builtins.QuantDequantFunc.apply(tensor, scale, offset, qmin, qmax).to(
        dtype
    )
    out3 = torch_builtins._torch_fake_quantize(
        tensor, scale.detach(), offset.detach(), qmin, qmax
    )

    assert torch.allclose(out1, expected, atol=atol)
    assert torch.allclose(out2, expected, atol=atol)
    if out3 is not None:
        assert torch.allclose(out3, expected, atol=atol)

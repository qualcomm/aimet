# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from packaging.version import parse
import pytest
from unittest.mock import patch
import itertools
import torch
import numpy as np
from aimet_torch.v2.quantization.affine.backends import (
    quantize,
    dequantize,
    quantize_dequantize,
    set_backend,
)
import aimet_torch.experimental.pgs

try:
    import triton
except ImportError:
    triton = None


@pytest.fixture(params=[True, False], scope="function")
def enable_pgs(request):
    enable = request.param

    if enable:
        try:
            aimet_torch.experimental.pgs.enable_pgs(eps=0.1, multiplier=3.0)
            yield
        finally:
            aimet_torch.experimental.pgs.disable_pgs()
    else:
        aimet_torch.experimental.pgs.disable_pgs()
        yield


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and triton
        and parse(triton.__version__) >= parse("3.0.0")
    ),
    reason="Triton backend requires CUDA and triton>=3.0.0",
)
class TestTritonBackend:
    def test_backend_switching(self):
        input = torch.randn(
            (8, 8), dtype=torch.float32, device="cuda", requires_grad=True
        )
        scale = torch.tensor(
            0.1, dtype=torch.float32, device="cuda", requires_grad=True
        )
        offset = torch.tensor(0, dtype=torch.float32, device="cuda", requires_grad=True)

        """
        When: Call quantize_dequantize with set_backend("triton")
        Then: Triton QDQ kernel should be invoked
        """
        with (
            patch(
                "aimet_torch.v2.quantization.affine.backends.triton.TritonQuantizeDequantize.apply"
            ) as triton_mock,
            patch(
                "aimet_torch.v2.quantization.affine.backends.torch_builtins.QuantDequantFunc.apply"
            ) as torch_builtin_mock,
        ):
            with set_backend("torch_builtins"):
                _ = quantize_dequantize(input, scale, offset, -128, 127)
                assert torch_builtin_mock.call_count == 1
                assert triton_mock.call_count == 0

            torch_builtin_mock.reset_mock()
            triton_mock.reset_mock()

            with set_backend("triton"):
                _ = quantize_dequantize(input, scale, offset, -128, 127)
                assert torch_builtin_mock.call_count == 0
                assert triton_mock.call_count == 1

            torch_builtin_mock.reset_mock()
            triton_mock.reset_mock()

        """
        When: Call quantize_dequantize with set_backend("triton") but with CPU tensors
        Then: Fall back to torch-builtin implementation
        """
        with (
            patch(
                "aimet_torch.v2.quantization.affine.backends.torch_builtins.QuantDequantFunc.apply"
            ) as torch_builtin_mock,
        ):
            with set_backend("triton"):
                _ = quantize_dequantize(
                    input.cpu(), scale.cpu(), offset.cpu(), -128, 127
                )
                assert torch_builtin_mock.call_count == 1

        """
        When: Call quantize_dequantize with set_backend("triton") with more than two block axes
        Then: Fall back to torch-builtin implementation
        """
        input = torch.randn(
            (32, 32, 32, 32), dtype=torch.float32, device="cuda", requires_grad=True
        )
        block_size = (4, 4, 4, 4)
        scale = torch.ones(
            8, 8, 8, 8, dtype=torch.float32, device="cuda", requires_grad=True
        )
        offset = torch.zeros_like(
            scale, dtype=torch.float32, device="cuda", requires_grad=True
        )

        with set_backend("triton"):
            _ = quantize_dequantize(
                input, scale, offset, -128, 127, block_size=block_size
            )
            assert torch_builtin_mock.call_count == 1

    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_per_tensor(self, seed):
        """
        Triton quantize kernel should should produce close-enough output
        as PyTorch built-in quantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((512, 512), dtype=torch.float32, device="cuda")
        scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        offset = torch.tensor(0, dtype=torch.float32, device="cuda")

        with set_backend("torch_builtins"):
            output_torch = quantize(input, scale, offset, -128, 127)

        with set_backend("triton"):
            output_triton = quantize(input, scale, offset, -128, 127)

        assert torch.allclose(output_triton, output_torch, atol=1)

    @pytest.mark.parametrize("channel_axis", range(4))
    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_per_channel(self, seed, channel_axis: int):
        """
        Triton quantize kernel should should produce close-enough output
        as PyTorch built-in quantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((32, 32, 32, 32), dtype=torch.float32, device="cuda")
        scale_shape = [-1 if axis == channel_axis else 1 for axis in range(input.dim())]
        scale = torch.arange(
            0.01,
            0.33,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset = torch.zeros(32, dtype=torch.float32, device="cuda").view(*scale_shape)

        with set_backend("torch_builtins"):
            output_torch = quantize(input, scale, offset, -128, 127)

        with set_backend("triton"):
            output_triton = quantize(input, scale, offset, -128, 127)

        assert torch.allclose(output_triton, output_torch, atol=1)

    @pytest.mark.parametrize(
        "channel_axis, block_axis", itertools.combinations(range(4), 2)
    )
    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_per_block(self, seed, channel_axis: int, block_axis: int):
        """
        Triton quantize kernel should should produce close-enough output
        as PyTorch built-in quantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((32, 32, 32, 32), dtype=torch.float32, device="cuda")
        block_size = tuple(
            dim // 2 if axis == block_axis else 1 if axis == channel_axis else dim
            for axis, dim in enumerate(input.shape)
        )
        scale_shape = tuple(
            dim // block_size for dim, block_size in zip(input.shape, block_size)
        )
        scale = torch.arange(
            0.01,
            0.65,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset = torch.zeros(64, dtype=torch.float32, device="cuda").view(*scale_shape)

        with set_backend("torch_builtins"):
            output_torch = quantize(
                input, scale, offset, -128, 127, block_size=block_size
            )

        with set_backend("triton"):
            output_triton = quantize(
                input, scale, offset, -128, 127, block_size=block_size
            )

        assert torch.allclose(output_triton, output_torch, atol=1)

    @pytest.mark.parametrize("seed", range(5))
    def test_dequantize_per_tensor(self, seed):
        """
        Triton dequantize kernel should should produce close-enough output
        as PyTorch built-in dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((512, 512), dtype=torch.float32, device="cuda")
        scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        offset = torch.tensor(0, dtype=torch.float32, device="cuda")

        with set_backend("torch_builtins"):
            output_torch = dequantize(input, scale, offset)

        with set_backend("triton"):
            output_triton = dequantize(input, scale, offset)

        assert torch.allclose(output_triton, output_torch)

    @pytest.mark.parametrize("channel_axis", range(4))
    @pytest.mark.parametrize("seed", range(5))
    def test_dequantize_per_channel(self, seed: int, channel_axis: int):
        """
        Triton dequantize kernel should should produce close-enough output
        as PyTorch built-in dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((32, 32, 32, 32), dtype=torch.float32, device="cuda")
        scale_shape = [-1 if axis == channel_axis else 1 for axis in range(input.dim())]
        scale = torch.arange(
            0.01,
            0.33,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset = torch.zeros(32, dtype=torch.float32, device="cuda").view(*scale_shape)

        with set_backend("torch_builtins"):
            output_torch = dequantize(input, scale, offset)

        with set_backend("triton"):
            output_triton = dequantize(input, scale, offset)

        assert torch.allclose(output_triton, output_torch)

    @pytest.mark.parametrize(
        "channel_axis, block_axis", itertools.combinations(range(4), 2)
    )
    @pytest.mark.parametrize("seed", range(5))
    def test_dequantize_per_block(self, seed, channel_axis: int, block_axis: int):
        """
        Triton dequantize kernel should should produce close-enough output
        as PyTorch built-in dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn((32, 32, 32, 32), dtype=torch.float32, device="cuda")
        block_size = tuple(
            dim // 2 if axis == block_axis else 1 if axis == channel_axis else dim
            for axis, dim in enumerate(input.shape)
        )
        scale_shape = tuple(
            dim // block_size for dim, block_size in zip(input.shape, block_size)
        )
        scale = torch.arange(
            0.01,
            0.65,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset = torch.zeros(64, dtype=torch.float32, device="cuda").view(*scale_shape)

        with set_backend("torch_builtins"):
            output_torch = dequantize(input, scale, offset, block_size=block_size)

        with set_backend("triton"):
            output_triton = dequantize(input, scale, offset, block_size=block_size)

        assert torch.allclose(output_triton, output_torch, atol=1)

    @pytest.mark.parametrize("input_requires_grad", [True, False])
    @pytest.mark.parametrize("scale_requires_grad", [True, False])
    @pytest.mark.parametrize("offset_requires_grad", [True, False])
    @pytest.mark.parametrize("zero_point_shift", [0.0, 0.5])
    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_dequantize_per_tensor(
        self,
        input_requires_grad: bool,
        scale_requires_grad: bool,
        offset_requires_grad: bool,
        zero_point_shift: float,
        enable_pgs,
        seed: int,
    ):
        """
        Triton quantize_dequantize kernel should should produce close-enough output
        as PyTorch built-in quantize_dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn(
            (512, 512),
            dtype=torch.float32,
            device="cuda",
            requires_grad=input_requires_grad,
        )
        scale = torch.tensor(
            0.01, dtype=torch.float32, device="cuda", requires_grad=scale_requires_grad
        )
        offset = torch.tensor(
            -1, dtype=torch.float32, device="cuda", requires_grad=offset_requires_grad
        )

        with set_backend("triton"):
            output_triton = quantize_dequantize(
                input, scale, offset, -128, 127, None, zero_point_shift=zero_point_shift
            )
            loss = torch.nn.functional.mse_loss(output_triton, input.detach())
            if loss.requires_grad:
                loss.backward()

        input_ = input.clone().detach().requires_grad_(input_requires_grad)
        scale_ = scale.clone().detach().requires_grad_(scale_requires_grad)
        offset_ = offset.clone().detach().requires_grad_(offset_requires_grad)

        with set_backend("torch_builtins"):
            output_torch = quantize_dequantize(
                input_, scale_, offset_, -128, 127, zero_point_shift=zero_point_shift
            )
            loss = torch.nn.functional.mse_loss(output_torch, input_.detach())
            if loss.requires_grad:
                loss.backward()

        assert torch.allclose(output_triton, output_torch, atol=scale.item())

        if input_requires_grad:
            assert input.grad is not None
            output_eq = output_triton == output_torch
            grad_eq = input.grad == input_.grad
            is_on_rounding_boundary = (
                input / scale - zero_point_shift - offset
            ) % 1 == 0.5

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients may not be exactly equal
                # even where outputs are equal due to precision error near PGS boundaries
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                assert (output_eq & grad_eq).sum() / output_eq.sum() > 0.999

                # if output is equal, then gradients should be equal unless
                # it was exactly on rounding or PGS boundary
                assert torch.all(
                    ~output_eq
                    | grad_eq
                    | is_on_rounding_boundary
                    | (input.grad == input_.grad * pgs_multiplier)
                    | (input.grad * pgs_multiplier == input_.grad)
                )
            else:
                # if output is equal, then gradients should be equal unless
                # it was exactly on rounding boundary
                assert torch.all(~output_eq | grad_eq | is_on_rounding_boundary)

            # Given MSE loss,
            # `grad_x = 2 * (x_qdq - x) / x.numel()`,
            # where `x_qdq` can differ by at most `scale` between triton and torch.
            atol = scale.item() * 2 / input.numel()

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients can further
                # differ by a factor of pgs_multiplier
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                atol *= pgs_multiplier
                isclose = torch.isclose(input.grad, input_.grad, atol=atol)
                assert isclose.sum() / input.numel() > 0.999
                assert torch.all(
                    isclose
                    | torch.isclose(input.grad, input_.grad * pgs_multiplier, atol=atol)
                    | torch.isclose(input.grad * pgs_multiplier, input_.grad, atol=atol)
                )
            else:
                torch.allclose(input.grad, input_.grad, atol=atol)

        else:
            assert input.grad is None

        if scale_requires_grad:
            assert scale.grad is not None
            assert torch.allclose(scale.grad, scale_.grad)
        else:
            assert scale.grad is None

        if offset_requires_grad:
            assert offset.grad is not None
            assert torch.allclose(offset.grad, offset_.grad)
        else:
            assert offset.grad is None

    @pytest.mark.parametrize("input_requires_grad", [True, False])
    @pytest.mark.parametrize("scale_requires_grad", [True, False])
    @pytest.mark.parametrize("offset_requires_grad", [True, False])
    @pytest.mark.parametrize("zero_point_shift", [0.0, 0.5])
    @pytest.mark.parametrize("channel_axis", range(4))
    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_dequantize_per_channel(
        self,
        channel_axis: int,
        input_requires_grad: bool,
        scale_requires_grad: bool,
        offset_requires_grad: bool,
        zero_point_shift: float,
        enable_pgs,
        seed: int,
    ):
        """
        Triton quantize_dequantize kernel should should produce close-enough output
        as PyTorch built-in quantize_dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn(
            (32, 32, 32, 32),
            dtype=torch.float32,
            device="cuda",
            requires_grad=input_requires_grad,
        )
        scale_shape = [-1 if axis == channel_axis else 1 for axis in range(input.dim())]
        scale = torch.arange(
            0.01,
            0.33,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        scale.requires_grad_(scale_requires_grad)
        offset = -torch.ones(
            32,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset.requires_grad_(offset_requires_grad)

        with set_backend("triton"):
            output_triton = quantize_dequantize(
                input, scale, offset, -128, 127, None, zero_point_shift=zero_point_shift
            )
            loss = torch.nn.functional.mse_loss(output_triton, input.detach())
            if loss.requires_grad:
                loss.backward()

            input_ = input.clone().detach().requires_grad_(input_requires_grad)
            scale_ = scale.clone().detach().requires_grad_(scale_requires_grad)
            offset_ = offset.clone().detach().requires_grad_(offset_requires_grad)

        with set_backend("torch_builtins"):
            output_torch = quantize_dequantize(
                input_, scale_, offset_, -128, 127, zero_point_shift=zero_point_shift
            )
            loss = torch.nn.functional.mse_loss(output_torch, input_.detach())
            if loss.requires_grad:
                loss.backward()

        assert np.allclose(
            output_triton.cpu().detach().numpy(),
            output_torch.cpu().detach().numpy(),
            atol=scale.cpu().detach().numpy(),
        )

        if input_requires_grad:
            assert input.grad is not None
            output_eq = output_triton == output_torch
            grad_eq = input.grad == input_.grad
            is_on_rounding_boundary = (
                input / scale - zero_point_shift - offset
            ) % 1 == 0.5

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients may not be exactly equal
                # even where outputs are equal due to precision error near PGS boundaries
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                assert (output_eq & grad_eq).sum() / output_eq.sum() > 0.999

                # if output is equal, then gradients should be equal unless
                # it was exactly on rounding or PGS boundary
                assert torch.all(
                    ~output_eq
                    | grad_eq
                    | is_on_rounding_boundary
                    | (input.grad == input_.grad * pgs_multiplier)
                    | (input.grad * pgs_multiplier == input_.grad)
                )
            else:
                # if output is equal, then gradients should be equal unless
                # it was exactly on rounding boundary
                assert torch.all(~output_eq | grad_eq | is_on_rounding_boundary)

            # Given MSE loss,
            # `grad_x = 2 * (x_qdq - x) / x.numel()`,
            # where `x_qdq` can differ by at most `scale` between triton and torch.
            atol = scale.cpu().detach().numpy() * 2 / input.numel()
            input_grad = input.grad.cpu().detach().numpy()
            input_grad_ = input_.grad.cpu().detach().numpy()

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients can further
                # differ by a factor of pgs_multiplier
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                atol *= pgs_multiplier
                isclose = np.isclose(input_grad, input_grad_, atol=atol)
                assert isclose.sum() / input.numel() > 0.999
                assert np.all(
                    isclose
                    | np.isclose(input_grad, input_grad_ * pgs_multiplier, atol=atol)
                    | np.isclose(input_grad * pgs_multiplier, input_grad_, atol=atol)
                )
            else:
                np.allclose(input_grad, input_grad_, atol=atol)

        else:
            assert input.grad is None

        if scale_requires_grad:
            assert scale.grad is not None
            assert torch.allclose(scale.grad, scale_.grad, rtol=1e-3)
        else:
            assert scale.grad is None

        if offset_requires_grad:
            assert offset.grad is not None
            assert torch.allclose(offset.grad, offset_.grad, rtol=1e-3)
        else:
            assert offset.grad is None

    @pytest.mark.parametrize("input_requires_grad", [False, True])
    @pytest.mark.parametrize("scale_requires_grad", [False, True])
    @pytest.mark.parametrize("offset_requires_grad", [False, True])
    @pytest.mark.parametrize("zero_point_shift", [0.0, 0.5])
    @pytest.mark.parametrize(
        "channel_axis, block_axis", itertools.combinations(range(4), 2)
    )
    @pytest.mark.parametrize("seed", range(5))
    def test_quantize_dequantize_per_block(
        self,
        channel_axis: int,
        block_axis: int,
        input_requires_grad: bool,
        scale_requires_grad: bool,
        offset_requires_grad: bool,
        zero_point_shift: float,
        enable_pgs,
        seed: int,
    ):
        """
        Triton quantize_dequantize kernel should should produce close-enough output
        as PyTorch built-in quantize_dequantize function.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn(
            (32, 32, 32, 32),
            dtype=torch.float32,
            device="cuda",
            requires_grad=input_requires_grad,
        )
        block_size = tuple(
            dim // 2 if axis == block_axis else 1 if axis == channel_axis else dim
            for axis, dim in enumerate(input.shape)
        )
        scale_shape = tuple(
            dim // block_size for dim, block_size in zip(input.shape, block_size)
        )
        scale = torch.arange(
            0.01,
            0.65,
            step=0.01,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        scale.requires_grad_(scale_requires_grad)
        offset = -torch.ones(
            64,
            dtype=torch.float32,
            device="cuda",
        ).view(*scale_shape)
        offset.requires_grad_(offset_requires_grad)

        with set_backend("triton"):
            output_triton = quantize_dequantize(
                input,
                scale,
                offset,
                -128,
                127,
                block_size,
                zero_point_shift=zero_point_shift,
            )
            loss = torch.nn.functional.mse_loss(output_triton, input.detach())
            if loss.requires_grad:
                loss.backward()

        input_ = input.clone().detach().requires_grad_(input_requires_grad)
        scale_ = scale.clone().detach().requires_grad_(scale_requires_grad)
        offset_ = offset.clone().detach().requires_grad_(offset_requires_grad)

        with set_backend("torch_builtins"):
            output_torch = quantize_dequantize(
                input_,
                scale_,
                offset_,
                -128,
                127,
                block_size=block_size,
                zero_point_shift=zero_point_shift,
            )
            loss = torch.nn.functional.mse_loss(output_torch, input_.detach())
            if loss.requires_grad:
                loss.backward()

        atol = scale.repeat_interleave(repeats=block_size[block_axis], dim=block_axis)
        assert np.allclose(
            output_triton.cpu().detach().numpy(),
            output_torch.cpu().detach().numpy(),
            atol=atol.cpu().detach().numpy(),
        )

        if input_requires_grad:
            assert input.grad is not None
            output_eq = output_triton == output_torch
            grad_eq = input.grad == input_.grad
            is_on_rounding_boundary = (
                input
                / scale.repeat_interleave(
                    repeats=block_size[block_axis], dim=block_axis
                )
                - zero_point_shift
                - offset.repeat_interleave(
                    repeats=block_size[block_axis], dim=block_axis
                )
            ) % 1 == 0.5

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients may not be exactly equal
                # even where outputs are equal due to precision error near PGS boundaries
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                assert (output_eq & grad_eq).sum() / output_eq.sum() > 0.999

                # if output is equal, then gradients should be equal unless
                # it was exactly on rounding or PGS boundary
                assert torch.all(
                    ~output_eq
                    | grad_eq
                    | is_on_rounding_boundary
                    | (input.grad == input_.grad * pgs_multiplier)
                    | (input.grad * pgs_multiplier == input_.grad)
                )
            else:
                # if output is equal, then gradients should be equal unless
                # it was exactly on rounding boundary
                assert torch.all(~output_eq | grad_eq | is_on_rounding_boundary)

            atol = atol.cpu().detach().numpy()
            input_grad = input.grad.cpu().detach().numpy()
            input_grad_ = input_.grad.cpu().detach().numpy()

            if aimet_torch.experimental.pgs.is_pgs_enabled():
                # When PGS is enabled, the gradients can further
                # differ by a factor of pgs_multiplier
                pgs_multiplier = aimet_torch.experimental.pgs.get_pgs_multiplier()
                atol *= pgs_multiplier
                isclose = np.isclose(input_grad, input_grad_, atol=atol)
                assert isclose.sum() / input.numel() > 0.999
                assert np.all(
                    isclose
                    | np.isclose(input_grad, input_grad_ * pgs_multiplier, atol=atol)
                    | np.isclose(input_grad * pgs_multiplier, input_grad_, atol=atol)
                )
            else:
                np.allclose(input_grad, input_grad_, atol=atol)
        else:
            assert input.grad is None

        if scale_requires_grad:
            assert scale.grad is not None
            assert torch.allclose(scale.grad, scale_.grad, rtol=1e-3)
        else:
            assert scale.grad is None

        if offset_requires_grad:
            assert offset.grad is not None
            assert torch.allclose(offset.grad, offset_.grad, rtol=1e-3)
        else:
            assert offset.grad is None

    @pytest.mark.parametrize("seed", range(5))
    def test_noncontiguous_inputs(self, seed):
        """
        When: Inputs are non-contiguous tensors
        Then: Triton backend should produce close-enough output
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        input = torch.randn(
            (512, 512), dtype=torch.float32, device="cuda", requires_grad=True
        )
        input_q = (input * 100).round().clamp(-128, 127)
        scale = torch.tensor(
            0.01, dtype=torch.float32, device="cuda", requires_grad=True
        )
        offset = torch.full_like(scale, -1, requires_grad=True)

        with set_backend("torch_builtins"):
            with torch.no_grad():
                output_q_torch = quantize(input[::2, ::2].T, scale, offset, -128, 127)
                output_dq_torch = dequantize(input_q[::2, ::2].T, scale, offset)

            output_qdq_torch = quantize_dequantize(
                input[::2, ::2].T, scale, offset, -128, 127
            )

            with torch.no_grad():
                mse_grad = 2 * (
                    quantize_dequantize(input.T, scale, offset, -128, 127) - input.T
                )
                mse_grad = (mse_grad / input[::2, ::2].numel())[::2, ::2]
                assert not mse_grad.is_contiguous()

            output_qdq_torch.backward(mse_grad)

        input_ = input.detach().clone().requires_grad_(True)
        scale_ = scale.detach().clone().requires_grad_(True)
        offset_ = offset.detach().clone().requires_grad_(True)

        with set_backend("triton"):
            with torch.no_grad():
                output_q_triton = quantize(
                    input_[::2, ::2].T, scale_, offset_, -128, 127
                )
                output_dq_triton = dequantize(input_q[::2, ::2].T, scale_, offset_)

            output_qdq_triton = quantize_dequantize(
                input_[::2, ::2].T, scale_, offset_, -128, 127
            )

            with torch.no_grad():
                mse_grad = 2 * (
                    quantize_dequantize(input_.T, scale_, offset_, -128, 127) - input_.T
                )
                mse_grad = (mse_grad / input[::2, ::2].numel())[::2, ::2]
                assert not mse_grad.is_contiguous()

            output_qdq_triton.backward(mse_grad)

        assert torch.allclose(output_q_triton, output_q_torch, atol=1)
        assert torch.allclose(output_dq_triton, output_dq_torch)
        assert torch.allclose(output_qdq_triton, output_qdq_torch, atol=scale.item())
        assert torch.allclose(
            input.grad, input_.grad, atol=scale.item() * 2 / input[::2, ::2].numel()
        )
        assert torch.allclose(scale.grad, scale_.grad)
        assert torch.allclose(offset.grad, offset_.grad)

        input.grad = scale.grad = offset.grad = None
        scale = (
            torch.arange(0.01, 5.13, step=0.01, dtype=torch.float32, device="cuda")
            .view(512, 1)
            .requires_grad_(True)
        )
        offset = torch.full_like(scale, -1, requires_grad=True)

        with set_backend("torch_builtins"):
            with torch.no_grad():
                output_q_torch = quantize(
                    input[::2, ::2].T, scale[::2], offset[::2], -128, 127
                )
                output_dq_torch = dequantize(
                    input_q[::2, ::2].T, scale[::2], offset[::2]
                )

            output_qdq_torch = quantize_dequantize(
                input[::2, ::2].T, scale[::2], offset[::2], -128, 127
            )

            with torch.no_grad():
                mse_grad = 2 * (
                    quantize_dequantize(input.T, scale, offset, -128, 127) - input.T
                )
                mse_grad = (mse_grad / input[::2, ::2].numel())[::2, ::2]
                assert not mse_grad.is_contiguous()

            output_qdq_torch.backward(mse_grad)

        input_ = input.detach().clone().requires_grad_(True)
        scale_ = scale.detach().clone().requires_grad_(True)
        offset_ = offset.detach().clone().requires_grad_(True)

        with set_backend("triton"):
            with torch.no_grad():
                output_q_triton = quantize(
                    input_[::2, ::2].T, scale_[::2], offset_[::2], -128, 127
                )
                output_dq_triton = dequantize(
                    input_q[::2, ::2].T, scale_[::2], offset_[::2]
                )

            output_qdq_triton = quantize_dequantize(
                input_[::2, ::2].T, scale_[::2], offset_[::2], -128, 127
            )

            with torch.no_grad():
                mse_grad = 2 * (
                    quantize_dequantize(input_.T, scale_, offset_, -128, 127) - input_.T
                )
                mse_grad = (mse_grad / input[::2, ::2].numel())[::2, ::2]
                assert not mse_grad.is_contiguous()

            output_qdq_triton.backward(mse_grad)

        assert torch.allclose(output_q_triton, output_q_torch, atol=1)
        assert torch.allclose(output_dq_triton, output_dq_torch)
        assert np.allclose(
            output_qdq_triton.detach().cpu().numpy(),
            output_qdq_torch.detach().cpu().numpy(),
            atol=scale[::2].detach().cpu().numpy(),
        )
        assert torch.all((input.grad[1::2] == 0) & (input_.grad[1::2] == 0))
        assert np.allclose(
            input.grad[::2].detach().cpu().numpy(),
            input_.grad[::2].detach().cpu().numpy(),
            atol=scale[::2].detach().cpu().numpy() * 2 / input[::2, ::2].numel(),
        )
        assert torch.allclose(scale.grad, scale_.grad)
        assert torch.allclose(offset.grad, offset_.grad)

        input.grad = scale.grad = offset.grad = None
        block_size = 128
        block_axis = 1
        scale = (
            scale.repeat_interleave(repeats=512 // block_size, dim=block_axis)
            .contiguous()
            .detach()
            .clone()
            .requires_grad_(True)
        )
        offset = torch.full_like(scale, -1, requires_grad=True)

        with set_backend("torch_builtins"):
            with torch.no_grad():
                output_q_torch = quantize(
                    input[::2, ::2].T,
                    scale[::2, ::2],
                    offset[::2, ::2],
                    -128,
                    127,
                    block_size=(1, block_size),
                )
                output_dq_torch = dequantize(
                    input_q[::2, ::2].T,
                    scale[::2, ::2],
                    offset[::2, ::2],
                    block_size=(1, block_size),
                )

            output_qdq_torch = quantize_dequantize(
                input[::2, ::2].T,
                scale[::2, ::2],
                offset[::2, ::2],
                -128,
                127,
                block_size=(1, block_size),
            )

            with torch.no_grad():
                mse_grad = 2 * (
                    quantize_dequantize(
                        input.T, scale, offset, -128, 127, block_size=(1, block_size)
                    )
                    - input.T
                )
                mse_grad = (mse_grad / input[::2, ::2].numel())[::2, ::2]
                assert not mse_grad.is_contiguous()

            output_qdq_torch.backward(mse_grad)

        input_ = input.detach().clone().requires_grad_(True)
        scale_ = scale.detach().clone().requires_grad_(True)
        offset_ = offset.detach().clone().requires_grad_(True)

        with set_backend("triton"):
            with torch.no_grad():
                output_q_triton = quantize(
                    input_[::2, ::2].T,
                    scale_[::2, ::2],
                    offset_[::2, ::2],
                    -128,
                    127,
                    block_size=(1, block_size),
                )
                output_dq_triton = dequantize(
                    input_q[::2, ::2].T,
                    scale_[::2, ::2],
                    offset_[::2, ::2],
                    block_size=(1, block_size),
                )

            output_qdq_triton = quantize_dequantize(
                input_[::2, ::2].T,
                scale_[::2, ::2],
                offset_[::2, ::2],
                -128,
                127,
                block_size=(1, block_size),
            )

            with torch.no_grad():
                mse_grad = 2 * (
                    quantize_dequantize(
                        input_.T, scale_, offset_, -128, 127, block_size=(1, block_size)
                    )
                    - input_.T
                )
                mse_grad = (mse_grad / input[::2, ::2].numel())[::2, ::2]
                assert not mse_grad.is_contiguous()

            output_qdq_triton.backward(mse_grad)

        atol = (
            scale[::2, ::2]
            .repeat_interleave(repeats=block_size, dim=block_axis)
            .detach()
            .cpu()
            .numpy()
        )
        assert torch.allclose(output_q_triton, output_q_torch, atol=1)
        assert torch.allclose(output_dq_triton, output_dq_torch)
        assert np.allclose(
            output_qdq_triton.detach().cpu().numpy(),
            output_qdq_torch.detach().cpu().numpy(),
            atol=atol,
        )
        assert np.allclose(
            input.grad[::2, ::2].detach().cpu().numpy(),
            input_.grad[::2, ::2].detach().cpu().numpy(),
            atol=atol * 2 / input[::2, ::2].numel(),
        )
        assert torch.allclose(scale.grad, scale_.grad)
        assert torch.allclose(offset.grad, offset_.grad)

    def test_compile_error_fallback(self):
        from aimet_torch.v2.quantization.affine.backends.triton import _compile_success

        _orig = _compile_success.copy()
        input = torch.randn((10, 10), device="cuda", requires_grad=True)
        scale = torch.tensor(0.01, device="cuda", requires_grad=True)
        offset = torch.tensor(0.0, device="cuda", requires_grad=True)

        try:
            with (
                patch(
                    "aimet_torch.v2.quantization.affine.backends.triton.TritonQuantizeDequantize.apply",
                    side_effect=triton.CompilationError(None, None),
                ) as triton_mock,
                patch(
                    "aimet_torch.v2.quantization.affine.backends.torch_builtins.QuantDequantFunc.apply"
                ) as torch_builtin_mock,
                set_backend("triton"),
            ):
                """
                When: Triton kernel failed to compile in the first call
                Then: It should fall back to PyTorch built-in implementation
                """
                _ = quantize_dequantize(input, scale, offset, -128, 127)
                assert torch_builtin_mock.call_count == 1
                assert triton_mock.call_count == 1

                """
                When: Call triton kernel again after initial compilation failure
                Then: It should fall back to PyTorch built-in implementation without
                      trying to compile Triton kernel again
                """
                _ = quantize_dequantize(input, scale, offset, -128, 127)
                assert torch_builtin_mock.call_count == 2
                assert triton_mock.call_count == 1

        finally:
            for k, v in _orig.items():
                _compile_success[k] = v

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=redefined-builtin
import functools
from typing import Callable
import torch
from torch._decomp import core_aten_decompositions
from torch._ops import OpOverloadPacket, OpOverload


@functools.lru_cache
def _qualcomm_aten_ir_ops() -> tuple[OpOverload, ...]:
    """
    List of quantization-time ATen IR ops
    """
    import yaml
    from pathlib import Path

    with open(Path(__file__).parent / "ir.yaml") as f:
        ir_yaml = yaml.safe_load(f)

    all_ops = []

    for entry in ir_yaml:
        scope, op_name = entry["name"].split("::")
        assert scope == "aten"

        op_packet_name, *overload_name = op_name.split(".")
        op_packet: OpOverloadPacket = getattr(torch.ops.aten, op_packet_name)
        (overload_name,) = overload_name or ("default",)
        op_overload = getattr(op_packet, overload_name)
        all_ops.append(op_overload)

    return tuple(all_ops)


def _qnn_decompositions() -> dict[torch._ops.OperatorBase, Callable]:
    """
    Returns quantization-friendly decomposition table for QNN.
    For the most part, this is a subset of core ATen decomposition table
    to prevent premature over-lowering before quantization

    Conceptually,

                                 Export   (`torch.export`)
                                   |
                          ATen ... |
                                   |
                                   V
                                 Lower    (`from_torch_exported_program`)
                                   |
    quantization-friendly ATen ... |
                                   |
                                   V
                                 Quantize (`compute_missing_encodings`)
                                   |
                quantized ATen ... |
                                   |
                                   V
                                 Lower
                                   |
           quantized core ATen ... |
                                   |
                                   V
                                 (...)
    """
    decomp_table = core_aten_decompositions()
    skiplist = _qualcomm_aten_ir_ops()

    for op in skiplist:
        _ = decomp_table.pop(op, None)

    decomp_table.update(_additional_decomposition_registry)
    return decomp_table


_additional_decomposition_registry: dict[torch._ops.OpOverload, Callable] = {}


def _register_qnn_decomposition(op: OpOverload):
    """
    Register decomposition for an op. If ``op`` is an overload packet,
    all of its overloads are registered. If it's a single overload, only
    that overload is registered.
    """

    def decorator(decomp: Callable):
        _additional_decomposition_registry[op] = decomp
        return decomp

    return decorator


@_register_qnn_decomposition(torch.ops.aten.relu6.default)
def relu6(self: torch.Tensor):
    """
    Technically, relu6 doesn't need further decomposition. This decomposition
    is only to promote clamp as the canonical/preferred lowering of relu6
    """
    return torch.ops.aten.clamp(self, min=0.0, max=6.0)


@_register_qnn_decomposition(torch.ops.aten.as_strided_copy.default)
def as_strided_copy(self: torch.Tensor, size, stride, storage_offset=None):
    return torch.ops.aten.as_strided(self, size, stride, storage_offset)


@_register_qnn_decomposition(torch.ops.aten.expand_copy.default)
def expand_copy(self: torch.Tensor, size, *, implicit: bool = False):
    return torch.ops.aten.expand(self, size, implicit=implicit)


@_register_qnn_decomposition(torch.ops.aten.permute_copy.default)
def permute_copy(self: torch.Tensor, dims):
    return torch.ops.aten.permute(self, dims)


@_register_qnn_decomposition(torch.ops.aten.squeeze_copy.dim)
def squeeze_copy_dim(self: torch.Tensor, dim: int):
    return torch.ops.aten.squeeze.dim(self, dim)


@_register_qnn_decomposition(torch.ops.aten.squeeze_copy.dims)
def squeeze_copy_dims(self: torch.Tensor, dim):
    return torch.ops.aten.squeeze.dims(self, dim)


@_register_qnn_decomposition(torch.ops.aten.unbind_copy.int)
def unbind_copy(self: torch.Tensor, dim: int = 0):
    return torch.ops.aten.unbind.int(self, dim)


@_register_qnn_decomposition(torch.ops.aten.unsqueeze_copy.default)
def unsqueeze_copy(self: torch.Tensor, dim: int):
    return torch.ops.aten.unsqueeze(self, dim)


def _scalar_to_tensor(value, device) -> torch.Tensor:
    """
    Wrap a Python scalar into a 0-dim tensor whose dtype follows the scalar's
    Python type, so that type promotion matches that of the *.Scalar overloads
    """
    if isinstance(value, bool):
        dtype = torch.bool
    elif isinstance(value, (float, torch.SymFloat)):
        dtype = torch.get_default_dtype()
    else:
        dtype = torch.int64
    return torch.ops.aten.scalar_tensor(value, dtype=dtype, device=device)


@_register_qnn_decomposition(torch.ops.aten.add.Scalar)
def add_scalar(self: torch.Tensor, other, alpha=1):
    return torch.ops.aten.add.Tensor(
        self, _scalar_to_tensor(other, self.device), alpha=alpha
    )


@_register_qnn_decomposition(torch.ops.aten.sub.Scalar)
def sub_scalar(self: torch.Tensor, other, alpha=1):
    return torch.ops.aten.sub.Tensor(
        self, _scalar_to_tensor(other, self.device), alpha=alpha
    )


@_register_qnn_decomposition(torch.ops.aten.div.Scalar)
def div_scalar(self: torch.Tensor, other):
    return torch.ops.aten.div.Tensor(self, _scalar_to_tensor(other, self.device))


@_register_qnn_decomposition(torch.ops.aten.div.Scalar_mode)
def div_scalar_mode(self: torch.Tensor, other, *, rounding_mode):
    return torch.ops.aten.div.Tensor_mode(
        self, _scalar_to_tensor(other, self.device), rounding_mode=rounding_mode
    )


@_register_qnn_decomposition(torch.ops.aten.mul.Scalar)
def mul_scalar(self: torch.Tensor, other):
    return torch.ops.aten.mul.Tensor(self, _scalar_to_tensor(other, self.device))


@_register_qnn_decomposition(torch.ops.aten.fmod.Scalar)
def fmod_scalar(self: torch.Tensor, other):
    return torch.ops.aten.fmod.Tensor(self, _scalar_to_tensor(other, self.device))


@_register_qnn_decomposition(torch.ops.aten.remainder.Scalar)
def remainder_scalar(self: torch.Tensor, other):
    return torch.ops.aten.remainder.Tensor(self, _scalar_to_tensor(other, self.device))


@_register_qnn_decomposition(torch.ops.aten.eq.Scalar)
def eq_scalar(self: torch.Tensor, other):
    return torch.ops.aten.eq.Tensor(self, _scalar_to_tensor(other, self.device))


@_register_qnn_decomposition(torch.ops.aten.ne.Scalar)
def ne_scalar(self: torch.Tensor, other):
    return torch.ops.aten.ne.Tensor(self, _scalar_to_tensor(other, self.device))


@_register_qnn_decomposition(torch.ops.aten.ge.Scalar)
def ge_scalar(self: torch.Tensor, other):
    return torch.ops.aten.ge.Tensor(self, _scalar_to_tensor(other, self.device))


@_register_qnn_decomposition(torch.ops.aten.gt.Scalar)
def gt_scalar(self: torch.Tensor, other):
    return torch.ops.aten.gt.Tensor(self, _scalar_to_tensor(other, self.device))


@_register_qnn_decomposition(torch.ops.aten.le.Scalar)
def le_scalar(self: torch.Tensor, other):
    return torch.ops.aten.le.Tensor(self, _scalar_to_tensor(other, self.device))


@_register_qnn_decomposition(torch.ops.aten.lt.Scalar)
def lt_scalar(self: torch.Tensor, other):
    return torch.ops.aten.lt.Tensor(self, _scalar_to_tensor(other, self.device))


@_register_qnn_decomposition(torch.ops.aten.fill.Scalar)
def fill_scalar(self: torch.Tensor, value):
    return torch.ops.aten.fill.Tensor(self, _scalar_to_tensor(value, self.device))


@_register_qnn_decomposition(torch.ops.aten.pow.Scalar)
def pow_scalar(self, exponent: torch.Tensor):
    base = _scalar_to_tensor(self, exponent.device)
    return torch.ops.aten.pow.Tensor_Tensor(base, exponent)


@_register_qnn_decomposition(torch.ops.aten.convolution.default)
def convolution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    padding,
    dilation,
    transposed: bool,
    output_padding,
    groups: int,
):
    """
    Promote aten::convolution to conv{1,2,3}d (or conv_transpose{1,2,3}d if
    ``transposed``) depending on the number of spatial dimensions of ``weight``
    """
    num_spatial_dims = weight.dim() - 2
    if num_spatial_dims not in (1, 2, 3):
        raise NotImplementedError(
            f"Expected 1D, 2D, or 3D convolution; got {num_spatial_dims} spatial dims"
        )

    if transposed:
        op = (
            torch.ops.aten.conv_transpose1d.default
            if num_spatial_dims == 1
            else torch.ops.aten.conv_transpose2d
            if num_spatial_dims == 2
            else torch.ops.aten.conv_transpose3d
        )
        return op(
            input, weight, bias, stride, padding, output_padding, groups, dilation
        )
    else:
        op = (
            torch.ops.aten.conv1d
            if num_spatial_dims == 1
            else torch.ops.aten.conv2d
            if num_spatial_dims == 2
            else torch.ops.aten.conv3d
        )
        return op(input, weight, bias, stride, padding, dilation, groups)

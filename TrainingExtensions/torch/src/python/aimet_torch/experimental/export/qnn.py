# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

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


_additional_decomposition_registry: dict[torch._ops.OperatorBase, Callable] = {}


def _register_qnn_decomposition(op: torch._ops.OpOverloadPacket):
    def decorator(decomp: Callable):
        for name in op.overloads():
            _additional_decomposition_registry[getattr(op, name)] = decomp
        return decomp

    return decorator


@_register_qnn_decomposition(torch.ops.aten.relu6)
def relu6(self: torch.Tensor):
    """
    Technically, relu6 doesn't need further decomposition. This decomposition
    is only to promote clamp as the canonical/preferred lowering of relu6
    """
    return torch.ops.aten.clamp(self, min=0.0, max=6.0)

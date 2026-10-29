# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from pathlib import Path
from typing import Callable, Any
import onnx
import torch
from torch.utils._pytree import tree_iter
import pytest
import onnx_ir

import aimet_torch
from aimet_torch.experimental.export.exported_program import (
    ExportedProgram as AimetExportedProgram,
)
from aimet_torch.common.onnx._utils import (
    _is_grid_preserving_op,
    _is_metadata_op,
    _onnx_to_qnn_ir,
)
from aimet_torch.experimental.export.qnn import _qualcomm_aten_ir_ops
from aimet_torch.experimental.export._example_inputs import _example_inputs_registry
from aimet_torch.experimental.export._utils import get_example_inputs


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(0)


def test_example_input_registry_completeness():
    """
    Example input registry should cover all Qualcomm-ATen operators
    """
    missing = set(_qualcomm_aten_ir_ops()) - set(_example_inputs_registry.keys())
    assert not missing, (
        f"Missing example inputs for operators: {[op.name() for op in missing]}"
    )


@pytest.mark.parallel
@pytest.mark.parametrize("op", _example_inputs_registry.keys())
def test_example_input_registry_correctness(op):
    """
    Example inputs in the registry should be valid for the corresponding operator
    """
    args, kwargs = get_example_inputs(op)

    try:
        _ = op(*args, **kwargs)  # Should not raise an exception
    except Exception as e:
        raise AssertionError(f"Example inputs for {op.__name__} are invalid: {e}")


# Operators that don't exist in ONNX but should be treated as supergroup in QNN IR
supergroups = {
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.rms_norm.default,
    torch.ops.aten.ne.Tensor,
    torch.ops.aten.div.Tensor_mode,
}

# Expected failures
# TODO: Fix these failures and remove them from the xfail list
xfail = {
    torch.ops.aten._native_batch_norm_legit.default: "ONNX export error",
    torch.ops.aten._native_batch_norm_legit_no_training.default: "ONNX export error",
    torch.ops.aten._safe_softmax.default: "Onnxscript bug. See https://github.com/microsoft/onnxscript/issues/2993",
    torch.ops.aten.copy.default: "Minor ONNX export error",
    torch.ops.aten.detach.default: "Minor ONNX export error",
    torch.ops.aten.native_batch_norm.default: "ONNX export error",
    torch.ops.aten.native_group_norm.default: "Missing intermediate encoding",
    torch.ops.aten.pow.Tensor_Scalar: "Exponent is missing encoding",
    torch.ops.aten.upsample_bicubic2d.default: "Missing intermediate encoding",
    torch.ops.aten._to_copy.default: "Minor ONNX export error",
    torch.ops.aten.clone.default: "Minor ONNX export error",
    torch.ops.aten.empty.memory_format: "Minor ONNX export error",
    torch.ops.aten.full.default: "Minor ONNX export error",
    torch.ops.aten.full.default: "Minor ONNX export error",
    torch.ops.aten.full_like.default: "Minor ONNX export error",
    torch.ops.aten.ones.default: "Minor ONNX export error",
    torch.ops.aten.zeros.default: "Minor ONNX export error",
    torch.ops.aten.alias.default: "Minor ONNX export error",
    torch.ops.aten.fill.Tensor: "Minor ONNX export error",
    torch.ops.aten.scalar_tensor.default: "Minor ONNX export error",
    torch.ops.aten.sym_size.int: "Minor ONNX export error",
}


@pytest.mark.parallel
@pytest.mark.parametrize("op", _example_inputs_registry.keys())
def test_qnn_ir_alignment(tmp_path: Path, op: torch._ops.OpOverload):
    """
    When: Convert AIMET ExportedProgram into ONNX
    Then:
      1. ONNX grpah should only consist of a curated subset of ONNX operators
         that are known to be 1-to-1 mappable to QNN IR operators.
      2. "QNN-friendly" ATen operators should be converted to a single ONNX operator.
    """
    if op in xfail:
        reason = xfail[op]
        pytest.xfail(f"Expected failure for {op.__name__}: {reason}")

    class Model(torch.nn.Module):
        def __init__(
            self, parameters: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor]
        ):
            super().__init__()

            for name, param in parameters.items():
                self.register_parameter(name, torch.nn.Parameter(param.clone()))

            for name, buffer in buffers.items():
                self.register_buffer(name, buffer.clone())

        def forward(self, *args, **kwargs):
            kwargs = {
                key: value
                for key, value in itertools.chain(
                    kwargs.items(),
                    self.named_parameters(),
                    self.named_buffers(),
                )
            }
            return op(*args, **kwargs)

    args, kwargs = get_example_inputs(op)

    parameters = {}
    if (weight := kwargs.get("weight")) is not None:
        parameters["weight"] = weight
    if (bias := kwargs.get("bias")) is not None:
        parameters["bias"] = bias

    buffers = {}
    if (running_mean := kwargs.get("running_mean")) is not None:
        buffers["running_mean"] = running_mean
    if (running_var := kwargs.get("running_var")) is not None:
        buffers["running_var"] = running_var

    model = Model(parameters=parameters, buffers=buffers)
    ep = torch.export.export(model, args, kwargs)
    ep = AimetExportedProgram.from_torch_exported_program(ep)

    expected = model(*args, **kwargs)
    out = ep.module()(*args, **kwargs)

    for out_, expected_ in zip(tree_iter(out), tree_iter(expected)):
        assert torch.allclose(out_, expected_, rtol=1e-3)

    with ep.compute_missing_encodings(
        param_bw=8, activation_bw=16, target_backend="htp_v69"
    ):
        out = ep.module()(*args, **kwargs)

    for out_ in tree_iter(out):
        assert not out_.isnan().any()

    torch.export.save(ep, tmp_path / f"{op.__name__}.pt2")

    aimet_torch.onnx.export(
        ep.module(),
        args=args,
        kwargs=kwargs,
        f=tmp_path / f"{op.__name__}.onnx",
        opset_version=21,
    )

    onnx_model = onnx.load(tmp_path / f"{op.__name__}.onnx")
    op_types = [node.op_type for node in onnx_model.graph.node]

    non_grid_preserving_ops = []
    for op_type in op_types:
        assert op_type in _onnx_to_qnn_ir or _is_grid_preserving_op(op_type)
        if (
            not _is_grid_preserving_op(op_type)
            and not _is_metadata_op(op_type)
            and op_type not in ("QuantizeLinear", "DequantizeLinear", "Cast")
        ):
            non_grid_preserving_ops.append(op_type)

    if op in _qualcomm_aten_ir_ops() and op not in supergroups:
        assert len(non_grid_preserving_ops) <= 1

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    model = onnx_ir.from_proto(onnx_model)

    quantizable_graph_inputs = [t for t in model.graph.inputs if _is_quantizable(t)]
    quantizable_graph_outputs = [t for t in model.graph.outputs if _is_quantizable(t)]

    # All quantizable graph inputs/outputs should be quantized
    for inp in quantizable_graph_inputs:
        for consumer in inp.consumers():
            assert consumer.op_type == "QuantizeLinear"
    for out in quantizable_graph_outputs:
        producer = out.producer()
        assert producer
        assert producer.op_type == "DequantizeLinear"

    # If op is a supergroup, intermediate tensors aren't quantized
    if op in supergroups:
        return

    # All quantizable intermediate tensors should be quantized
    for node in model.graph.all_nodes():
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue

        for inp in node.inputs:
            if _is_quantizable(inp):
                assert inp is not None
                assert (producer := inp.producer()) is not None
                assert producer.op_type == "DequantizeLinear"

        for out in node.outputs:
            if _is_quantizable(out):
                for consumer in out.consumers():
                    assert consumer.op_type == "QuantizeLinear"


def _is_batchnorm_input_statistics(value: onnx_ir.Value) -> bool:
    return (
        len(value.consumers()) == 1
        and (consumer := value.consumers()[0]).op_type == "BatchNormalization"
        and value in consumer.inputs[3:5]
    )


def _is_batchnorm_output_statistics(value: onnx_ir.Value) -> bool:
    return bool(
        (producer := value.producer())
        and producer.op_type == "BatchNormalization"
        and value in producer.outputs[1:]
    )


def _is_layernorm_output_statistics(value: onnx_ir.Value) -> bool:
    return bool(
        (producer := value.producer())
        and producer.op_type == "LayerNormalization"
        and value in producer.outputs[1:]
    )


def _is_quantizable(value: onnx_ir.Value | None) -> bool:
    return (
        value is not None
        and value.dtype == onnx.TensorProto.FLOAT
        and not _is_batchnorm_input_statistics(value)
        and not _is_batchnorm_output_statistics(value)
        and not _is_layernorm_output_statistics(value)
    )

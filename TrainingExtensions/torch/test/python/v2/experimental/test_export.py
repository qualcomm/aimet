# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, Any
import itertools
import io
from functools import partial, lru_cache
import numpy as np
from pathlib import Path
from packaging import version
import operator
import torch
from torchvision.models import resnet18, mobilenet_v3_large
from torch.export import ExportedProgram
from torch._export.utils import _detect_fake_mode_from_gm
from torch.export.graph_signature import InputKind
import aimet_torch
from aimet_torch import QuantizationSimModel
from aimet_torch.nn import QuantizationMixin
from aimet_torch.v2.experimental.export import export
import aimet_torch.v2.quantization as Q
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from aimet_torch.quantization.affine import AffineQuantizerBase
from aimet_torch.experimental.export.exported_program import (
    ExportedProgram as AimetExportedProgram,
)
from aimet_torch.model_preparer import prepare_model
from aimet_torch.experimental.export.exported_program import _is_torch_ao_qdq_node
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.nn import QuantizationMixin
import pytest


def conv(**_):
    return torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3))


def conv_flatten(**_):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 3),
        torch.nn.Flatten(),
    )


def conv_relu(**_):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 3),
        torch.nn.ReLU(),
    )


def conv_prelu(**_):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 3),
        torch.nn.PReLU(),
    )


def custom_rmsnorm(**_):
    return torch.nn.Sequential(
        aimet_torch.nn.modules.custom.RmsNorm([1, 3, 224, 224], [-1], epsilon=1e-6)
    )


def llama_rms_norm(**_):
    return torch.nn.Sequential(
        LlamaRMSNorm(hidden_size=224),
    )


@pytest.mark.parallel
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.8.0"),
    reason="aimet_torch.export.export is only supported in torch >= 2.8.0",
)
@pytest.mark.parametrize(
    "model_factory",
    [
        conv,
        conv_relu,
        conv_prelu,
        conv_flatten,
        custom_rmsnorm,
        llama_rms_norm,
        resnet18,
        mobilenet_v3_large,
    ],
)
@pytest.mark.parametrize("compile", [False, True])
def test_export(model_factory, compile: bool, tmp_path: Path):
    if compile and version.parse(torch.__version__) < version.parse("2.11.0.dev"):
        pytest.skip(
            reason="Exporting torch.compile-d model is only supported in torch >= 2.11.0"
        )

    if (
        model_factory == llama_rms_norm
        and compile
        and version.parse("2.12.0")
        <= version.parse(torch.__version__)
        < version.parse("2.13.0")
    ):
        pytest.skip(
            reason="PyTorch 2.12.0 has a compile bug when pow is involved (will be fixed in 2.13). "
            "See https://github.com/pytorch/pytorch/issues/185715"
        )

    model = model_factory(pretrained=False).requires_grad_(False).eval()
    model = prepare_model(model)
    x = torch.randn(1, 3, 224, 224)
    sim = QuantizationSimModel(model, x, default_param_bw=4, config_file="htp_v81")

    with pytest.raises(RuntimeError):
        # Before computing encodings, export should raise error
        export(sim.model, args=(x,))

    sim.compute_encodings(lambda model: model(x))

    if compile:
        sim.model = torch.compile(sim.model)

    last_layer_name = [name for name, _ in model.named_modules()][
        -2 if model_factory == conv_flatten else -1
    ]
    last_layer = sim.model.get_submodule(last_layer_name)

    """
    When: Export sim with aimet_torch.export.export
    Then: The resulting ExportedProgram should produce output close enough to sim
    """
    ep: ExportedProgram = export(sim.model, args=(x,))
    path = tmp_path / f"{model_factory.__name__}_quantized.pt2"
    torch.export.save(ep, path)
    ep = torch.export.load(path)
    sim_out = sim.model(x)
    ep_out = ep.module()(x)

    # Allow off-by-3 error
    atol = last_layer.output_quantizers[0].get_scale().item() * 3
    assert torch.allclose(sim_out, ep_out, atol=atol)

    """
    Then: The number of qdq nodes should be equal to the number of quantizers in sim
    """
    quantizers = [
        qtzr
        for qmodule in sim.qmodules()
        for qtzr in itertools.chain(
            qmodule.input_quantizers,
            qmodule.output_quantizers,
            qmodule.param_quantizers.values(),
        )
        if isinstance(qtzr, AffineQuantizerBase) and qtzr.is_initialized()
    ]
    torch_dq_nodes = [
        node
        for node in ep.graph.nodes
        if node.op == "call_function"
        and node.target.name().startswith("quantized_decomposed::dequantize")
    ]
    # Exported model can contain MORE qdq nodes than quantsim
    # as data movement op's output encodings that are generated
    # on-the-fly during export
    assert len(torch_dq_nodes) >= len(quantizers)

    """
    Then: All scales and zero_points should be constant-folded
    """
    for q_dq_node in ep.graph.nodes:
        if q_dq_node.op == "call_function" and (
            q_dq_node.target.name().startswith("quantized_decomposed::quantize")
            or q_dq_node.target.name().startswith("quantized_decomposed::dequantize")
        ):
            for inp_node in q_dq_node.all_input_nodes[1:]:
                assert inp_node.op == "placeholder"
                assert inp_node.name.endswith("scale") or inp_node.name.endswith(
                    "zero_point"
                )

    """
    Then: There should be no dangling node, graph_signature or state dict entry
    """
    stack = [ep.graph.output_node()]
    visited = set()
    while stack:
        node = stack.pop(-1)
        if node in visited:
            continue
        visited.add(node.name)
        stack += node.all_input_nodes
    assert visited == set(node.name for node in ep.graph.nodes)

    for input_spec in ep.graph_signature.input_specs:
        assert input_spec.arg.name in visited

        if input_spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
        ):
            assert (
                input_spec.target in ep.state_dict.keys()
                or input_spec.target in ep.constants.keys()
            )

    all_targets = set(
        input_spec.target for input_spec in ep.graph_signature.input_specs
    )
    assert not (ep.state_dict.keys() - all_targets)
    assert not (ep.constants.keys() - all_targets)

    _detect_fake_mode_from_gm(ep.graph_module)  # Shouldn't raise error


@pytest.mark.parallel
def test_shared_weight(tmp_path: Path):
    """
    Given: A model with shared weights
    When: Export sim with aimet_torch.export.export
    Then: The resulting ExportedProgram should contain duplicate weights,
          each with its own encodings
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 20),
        torch.nn.Linear(20, 20),
    )
    # Share weights between the two linear layers
    model[1].weight = model[0].weight
    x = torch.randn(1, 20)
    sim = aimet_torch.QuantizationSimModel(model, x, config_file="htp_v81")
    sim.compute_encodings(lambda model: model(x))

    ep = aimet_torch.experimental.export.export(sim.model, (x,))
    path = tmp_path / "model.onnx"
    torch.export.save(ep, path)
    ep = torch.export.load(path)

    sim_out = sim.model(x)
    ep_out = ep.module()(x)

    # Allow off-by-1 error
    atol = sim.model[-1].output_quantizers[0].get_scale().item()
    assert torch.allclose(sim_out, ep_out, atol=atol)

    weight_params = [
        input_spec
        for input_spec in ep.graph_signature.input_specs
        if input_spec.kind == InputKind.PARAMETER and "weight" in input_spec.arg.name
    ]
    # There should be two separate weight parameters in exported model
    assert len(weight_params) == 2


@pytest.mark.skipif(
    not Q.affine.backends.triton.is_available(),
    reason="Triton backend not available",
)
def test_triton(tmp_path: Path):
    """
    When: Export to ExportedProgram with torch_builtins and triton backends
    Then: The exported ExportedPrograms should be identical
    """
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 3),
        torch.nn.ReLU(),
    )
    x = torch.randn(1, 3, 32, 32)
    sim = aimet_torch.QuantizationSimModel(model, x, config_file="htp_v81")
    sim.compute_encodings(lambda model: model(x))

    with Q.affine.set_backend("torch_builtins"):
        ep_builtin = aimet_torch.experimental.export.export(sim.model, (x,))
        path = tmp_path / "model.pt2"
        torch.export.save(ep_builtin, path)
        ep_builtin = torch.export.load(path)

    with Q.affine.set_backend("triton"):
        ep_triton = aimet_torch.experimental.export.export(sim.model, (x,))
        path = tmp_path / "model.pt2"
        torch.export.save(ep_triton, path)
        ep_triton = torch.export.load(path)

    assert ep_builtin.graph_signature == ep_triton.graph_signature
    assert len(ep_builtin.graph.nodes) == len(ep_triton.graph.nodes)

    @lru_cache(maxsize=30)
    def node_eq(node1, node2):
        return (
            node1.name == node2.name
            and node1.op == node2.op
            and node1.target == node2.target
            and all(
                node_eq(n1, n2)
                for n1, n2 in zip(node1.all_input_nodes, node2.all_input_nodes)
            )
        )

    for node, node_ in zip(ep_builtin.graph.nodes, ep_triton.graph.nodes):
        assert node_eq(node, node_)

    for seed in range(10):
        torch.manual_seed(seed)
        x = torch.randn(1, 3, 32, 32)
        assert torch.equal(ep_builtin.module()(x), ep_triton.module()(x))


def test_fp_model():
    """
    When: Export a non-quantized model with aimet_torch.export.export
    Then: The resulting ExportedProgram should be equal to that of torch.export.export
    """

    fp_model = conv_relu()
    x = torch.randn(1, 3, 32, 32)
    ep_aimet = aimet_torch.experimental.export.export(fp_model, (x,))
    ep_torch = torch.export.export(fp_model, (x,))

    f1 = io.BytesIO()
    f2 = io.BytesIO()
    torch.export.save(ep_aimet, f1)
    torch.export.save(ep_torch, f2)
    assert f1.getvalue() == f2.getvalue()


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize("target_backend", [None, "htp_v69"])
def test_compute_missing_encodings(
    tmp_path: Path, device: str, target_backend: str | None
):
    """
    Given: Model with functional ops
    When: Export with aimet_torch.export.export and compute missing encodings
    Then: All nodes should take input from dequantize node and produce output to quantize node
    """

    model = resnet18(pretrained=False).eval().to(device=device)
    example_inputs = (torch.randn(1, 3, 224, 224, device=device),)
    fold_all_batch_norms(model, None, dummy_input=example_inputs)
    ep = aimet_torch.experimental.export.export(model, example_inputs)

    # Save and load for robustness. Legacy meta data such as "tensor_meta" are
    # intentionally dropped during torch.export.save. Serializing and deserializing here
    # helps ensure AIMET doesn't rely on the meta data that are dropped during serialization.
    torch.export.save(ep, tmp_path / "model.pt2")
    ep = torch.export.load(tmp_path / "model.pt2")

    ep = AimetExportedProgram.from_torch_exported_program(ep)

    with ep.compute_missing_encodings(
        param_bw=8, activation_bw=16, target_backend=target_backend
    ):
        output = ep.module()(*example_inputs)
        assert isinstance(output, Q.DequantizedTensor)

    for node in ep.graph.nodes:
        if _is_torch_ao_qdq_node(node):
            continue

        for inp in node.all_input_nodes:
            input_type = type(inp.meta["val"])

            if issubclass(input_type, torch.Tensor):
                assert _is_torch_ao_qdq_node(inp)
            else:
                assert not _is_torch_ao_qdq_node(inp)

        if isinstance(
            node.target, torch._ops.OpOverload
        ) and node.target.overloadpacket in (torch.ops.aten.relu, torch.ops.aten.relu_):
            # Relu input/output encoding should be identical non-negative encoding
            (input_dq,) = node.all_input_nodes
            (output_q,) = node.users
            assert input_dq.args[1:] == output_q.args[1:]
            _, _, zero_point, _, _, dtype = input_dq.args
            assert zero_point == 0
            assert dtype == torch.uint16

        for user in node.users:
            output_type = type(node.meta["val"])

            if issubclass(output_type, torch.Tensor):
                assert _is_torch_ao_qdq_node(user)
            else:
                assert not _is_torch_ao_qdq_node(user)

            if _is_torch_ao_qdq_node(user) and node is user.all_input_nodes[0]:
                quantized_dtype = user.args[-1]
                expected_dtype = (
                    torch.int32
                    if node.name.endswith("bias") or node.name.endswith("bias_int32")
                    else torch.int8
                    if node.name.endswith("weight") or node.name.endswith("weight_int8")
                    else torch.uint16
                )
                assert quantized_dtype == expected_dtype, (
                    f"{node.name} has quantized dtype {quantized_dtype}, expected {expected_dtype}"
                )

    # Should be exportable to ONNX
    aimet_torch.onnx.export(
        ep.module(),
        example_inputs,
        str(tmp_path / "model.onnx"),
        dynamo=True,
        external_data=True,
        opset_version=21,
    )


@pytest.mark.parallel
@pytest.mark.parametrize(
    "op",
    [
        partial(torch.ops.aten.split, split_size=5, dim=-1),
        partial(torch.ops.aten.split, split_size=[5, 5], dim=-1),
        partial(torch.ops.aten.split, split_size=[5, 5], dim=-1),
        partial(torch.ops.aten.min, dim=-1),
        partial(torch.ops.aten.max, dim=-1),
        partial(torch.ops.aten.max_pool2d_with_indices, kernel_size=(2, 2)),
        partial(torch.ops.aten.max_pool3d_with_indices, kernel_size=(2, 2, 2)),
    ],
)
def test_multi_output_grid_preserving_ops(op: Callable[[torch.Tensor], Any]):
    """
    Given: Grid-preserving operator with multiple outputs with input quantizer only
    When: Export with aimet_torch.export.export
    Then: All output encodings should be derived from the input encoding
    """

    class MyModule(torch.nn.Module):
        def forward(self, x):
            return op(x)

    @QuantizationMixin.implements(MyModule)
    class QuantizedMyModule(QuantizationMixin, MyModule):
        def forward(self, x):
            if self.input_quantizers[0]:
                x = self.input_quantizers[0](x)

            return super().forward(x)

    model = torch.nn.Sequential(MyModule())
    x = torch.randn(1, 3, 10, 10)
    sim = aimet_torch.QuantizationSimModel(model, x)
    sim.compute_encodings(lambda m: m.forward(x))

    ep = aimet_torch.experimental.export.export(sim.model, (x,))

    input_dq = next(
        dq
        for dq in ep.graph.nodes
        if dq.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )

    output_producers = (
        list(ep.graph.output_node().all_input_nodes)
        if op.func == torch.ops.aten.split
        else list(ep.graph.output_node().all_input_nodes[:-1])
    )

    for output_dq in output_producers:
        assert (
            output_dq.target.overloadpacket
            == torch.ops.quantized_decomposed.dequantize_per_tensor
        )
        assert output_dq.args[1:] == input_dq.args[1:]

    """
    Given: Grid-preserving operator with multiple outputs with no input quantizer
    When: Export quantsim with aimet_torch.export.export
    Then: All output encodings should be derived from the input encoding
    """

    model = MyModule()
    ep = aimet_torch.experimental.export.export(model, (x,))
    ep = AimetExportedProgram.from_torch_exported_program(ep)

    with ep.compute_missing_encodings(param_bw=8, activation_bw=16):
        _ = ep.module()(x)

    input_dq = next(
        dq
        for dq in ep.graph.nodes
        if dq.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )

    output_producers = (
        list(ep.graph.output_node().all_input_nodes)
        if op.func == torch.ops.aten.split
        else list(ep.graph.output_node().all_input_nodes[:-1])
    )

    for output_dq in output_producers:
        assert (
            output_dq.target.overloadpacket
            == torch.ops.quantized_decomposed.dequantize_per_tensor
        )
        assert output_dq.args[1:] == input_dq.args[1:]


@pytest.mark.parallel
def test_compute_missing_encodings_with_constant():
    """
    Given: Model with hardcoded constant that is neither a parameter nor a buffer
    When: Export with aimet_torch.export.export and compute missing encodings
    Then: The constant should NOT be quantized; only parameters and buffers should be quantized
    """

    @QuantizationMixin.ignore
    class MulConstantNumpyArr(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.arr = torch.from_numpy(np.random.randn(10).astype(np.float32))

        def forward(self, x):
            return x * self.arr

    @QuantizationMixin.ignore
    class AddConstantTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.arr = torch.randn(10)

        def forward(self, x):
            return x + self.arr

    model = torch.nn.Sequential(
        MulConstantNumpyArr(),
        AddConstantTensor(),
    )
    example_inputs = (torch.randn(10),)
    sim = QuantizationSimModel(model, example_inputs, config_file="htp_v81")
    sim.compute_encodings(lambda model: model(*example_inputs))

    _ = torch.export.export(sim.model, example_inputs)
    ep = aimet_torch.experimental.export.export(sim.model, example_inputs)
    ep = AimetExportedProgram.from_torch_exported_program(ep)

    with ep.compute_missing_encodings(param_bw=8, activation_bw=16):
        _ = ep.module()(*example_inputs)

    mul_node = next(
        node for node in ep.graph.nodes if node.target == torch.ops.aten.mul.Tensor
    )
    add_node = next(
        node for node in ep.graph.nodes if node.target == torch.ops.aten.add.Tensor
    )
    assert (
        mul_node.all_input_nodes[0].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert (
        mul_node.all_input_nodes[1].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert len(mul_node.users) == 1
    assert (
        list(mul_node.users)[0].target
        == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )
    assert (
        add_node.all_input_nodes[0].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert (
        add_node.all_input_nodes[1].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert len(add_node.users) == 1
    assert (
        list(add_node.users)[0].target
        == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )


def test_dynamic_shape():
    fp_model = conv_relu()
    x = torch.randn(2, 3, 32, 32)

    ep = aimet_torch.experimental.export.export(
        fp_model,
        (x,),
        dynamic_shapes={"input": {0: torch.export.Dim("batch")}},
    )

    # Should work with different batch size
    for batch_size in [1, 3, 4]:
        _ = ep.module()(torch.randn(batch_size, 3, 32, 32))

    ep = AimetExportedProgram.from_torch_exported_program(ep)

    with ep.compute_missing_encodings(param_bw=8, activation_bw=16):
        # Should still work with different batch size
        for batch_size in [1, 3, 4]:
            _ = ep.module()(torch.randn(batch_size, 3, 32, 32))

    # Should still work with different batch size
    for batch_size in [1, 3, 4]:
        _ = ep.module()(torch.randn(batch_size, 3, 32, 32))


@pytest.mark.parallel
def test_preserve_graph_signature():
    """
    Given: A model with unused user input
    When: Export with aimet_torch.export.export
    Then: The exported ExportedProgram should preserve the forward signature of the custom module
    """

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = torch.nn.Linear(3, 3)
            self.linear_2 = torch.nn.Linear(3, 3)

        def forward(self, x, _, z):
            return self.linear_1(x) + self.linear_2(z)

    model = MyModule()
    x = torch.randn(3, 3)
    _ = torch.randn(3, 3)
    z = torch.randn(3, 3)
    sim = aimet_torch.QuantizationSimModel(model, (x, _, z), config_file="htp_v81")
    sim.compute_encodings(lambda m: m(x, _, z))

    ep = aimet_torch.experimental.export.export(sim.model, (x, _, z))

    user_inputs = [
        spec
        for spec in ep.graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    ]
    assert user_inputs[0].arg.name == "x"
    assert user_inputs[1].arg.name == "_"
    assert user_inputs[2].arg.name == "z"

    ep = AimetExportedProgram.from_torch_exported_program(ep)
    with ep.compute_missing_encodings(param_bw=8, activation_bw=16):
        _ = ep.module()(x, _, z)

    assert user_inputs[0].arg.name == "x"
    assert user_inputs[1].arg.name == "_"
    assert user_inputs[2].arg.name == "z"

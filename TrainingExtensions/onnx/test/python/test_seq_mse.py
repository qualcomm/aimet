# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

import functools
import io
import re
import time
import pytest
from unittest.mock import MagicMock
from functools import partial
import ml_dtypes
import onnx
import onnx_ir
import torch
import torch.nn.functional as F
import onnxruntime as ort
import copy
import json
import numpy as np
import os
import logging
import tempfile

from aimet_onnx.common import libquant_info
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx import apply_seq_mse
from aimet_onnx.blockwise_sampler import BlockwiseSampler, SamplingPlan, _build_session
from aimet_onnx.sequential_mse.frontier import (
    SUPPORTED_MODULES,
    find_layers,
    get_weight_name,
    is_supported_op,
    plan_frontier,
)
from aimet_onnx.sequential_mse.seq_mse import (
    SeqMseParams,
    SequentialMse,
    _temporarily_disable_block_grouping,
    _rechunk_seq_inputs,
    _compute_seq_chunk_size,
)
from aimet_onnx.sequential_mse.transform import (
    modify_graph_with_grouped_conv,
    modify_graph_with_grouped_linear,
    prepare_linear_inputs,
)
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import (
    QuantizationSimModel,
    _quantize_subgraphs_enabled,
    set_blockwise_quantization_for_weights,
    set_grouped_blockwise_quantization_for_weights,
)
from aimet_onnx.utils import _run_session_to_numpy, make_dummy_input
from aimet_onnx.qc_quantize_op import (
    QcQuantizeOp,
    OpMode,
    TensorQuantizerParams,
)

from .models.test_models import single_linear_layer_model
from .models.test_models import single_conv_layer_model
from .models.test_models import model_with_split
from .models.test_models import single_residual_model
from .models import models_for_tests
from .models.test_models_onnx import model_with_multiple_inputs
from .models.test_models_onnx import model_with_multiple_outputs
from .utils import (
    PrefillWrapper,
    export_onnx,
    make_tiny_decoder_config,
    sample_prefill_inputs,
    tmp_dir,
)

AimetLogger.set_level_for_all_areas(logging.DEBUG)


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(0)
    torch.manual_seed(42)


def _get_config_file(
    is_symmetric: bool,
    strict_symmetric: bool,
    unsigned_symmetric: bool,
    pcq: bool,
    dir_path: str,
) -> str:
    """Temporary fix until the config file can be read from beq_config directory"""

    def get_bool_str(in_bool: bool) -> str:
        if in_bool:
            return "True"
        else:
            return "False"

    beq_per_channel_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True",
                "is_symmetric": get_bool_str(is_symmetric),
            },
            "params": {
                "is_quantized": "True",
                "is_symmetric": get_bool_str(is_symmetric),
            },
            "strict_symmetric": get_bool_str(strict_symmetric),
            "unsigned_symmetric": get_bool_str(unsigned_symmetric),
            "per_channel_quantization": get_bool_str(pcq),
        },
        "params": {"bias": {"is_quantized": "True"}},
        "op_type": {"PRelu": {"params": {"weight": {"is_quantized": "False"}}}},
        "supergroups": [
            {"op_list": ["Gemm", "PRelu"]},
            {"op_list": ["Gemm", "Sigmoid"]},
            {"op_list": ["Conv", "PRelu"]},
            {"op_list": ["Conv", "Sigmoid"]},
        ],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {},
    }

    file_name = os.path.join(dir_path, "beq_per_channel_config.json")
    with open(file_name, "w") as f:
        json.dump(beq_per_channel_config, f)

    return file_name


def data_loader(dummy_input):
    return [dummy_input for _ in range(10)]


def _get_weight_param_name(cg_op):
    param_names = [
        param_name
        for param_name, (_, param_type) in cg_op.parameters.items()
        if param_type == "weight"
    ]
    if len(param_names) == 1:
        return param_names[0]
    return None


def _mock_layer(op_type):
    """A layer whose node has the initializer weight ``mock_weight``."""
    weight = onnx_ir.Value(
        name="mock_weight", const_value=onnx_ir.tensor(np.zeros((1, 1), np.float32))
    )
    graph = onnx_ir.Graph([], [], nodes=[], initializers=[weight])
    node = onnx_ir.Node("", op_type, [onnx_ir.Value(name="x"), weight], name="mock")
    graph.append(node)
    layer = MagicMock()
    layer.node = node
    return layer


def _onnx_model_from_op(
    op_type: str,
    weight_tensor: np.ndarray,
    input_shape: tuple,
    output_shape: tuple,
    num_nodes: int = 1,
    **kwargs,
):
    """
    Creates an ONNX model using the specified op_type.
    """
    input_tensor = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, input_shape
    )
    output_tensors = []
    initializers = []
    nodes = []

    for i in range(num_nodes):
        weight_name = f"weight_{i}"
        output_name = f"output_{i}"

        # Create op_node
        op_node = onnx.helper.make_node(
            op_type,
            inputs=["input", weight_name],
            outputs=[output_name],
            name=f"{op_type.lower()}_{i}",
            **kwargs,
        )
        nodes.append(op_node)

        # Weight tensor to ONNX initializer
        weight_initializer = onnx.numpy_helper.from_array(
            weight_tensor.copy(), name=weight_name
        )
        initializers.append(weight_initializer)

        # Define output tensor metadata
        output_tensor = onnx.helper.make_tensor_value_info(
            output_name, onnx.TensorProto.FLOAT, output_shape
        )
        output_tensors.append(output_tensor)

    graph = onnx.helper.make_graph(
        nodes,
        f"{op_type}_graph",
        inputs=[input_tensor],
        outputs=output_tensors,
        initializer=initializers,
    )

    # Create and validate model
    model = models_for_tests.make_model(graph)
    onnx.checker.check_model(model)

    return model


def _torch_blockwise_conv2d(
    x: torch.Tensor, w: torch.Tensor, block_size: int, **kwargs
):
    """
    Perform block-wise 2D convolution operation using torch operations.

    It splits the input and weight tensors into blocks along the input channel dimension and
    applies the convolution to each block independently and stacks the results along a new dimension.
    """
    c_out, c_in, k_h, k_w = w.shape
    num_blocks = c_in // block_size

    # Reshape weight from (c_out, c_in, kh, kw) to (c_out, num_blocks, block_size, kh, kw)
    # Transpose weights from (c_out, num_blocks, block_size, kh, kw) to (num_blocks, c_out, block_size, kh, kw)
    # Reshape weight from (num_blocks, c_out, block_size, kh, kw) to (num_blocks * c_out, block_size, kh, kw)
    w = w.reshape(c_out, num_blocks, block_size, k_h, k_w)
    w = w.permute(1, 0, 2, 3, 4)
    w = w.reshape(num_blocks * c_out, block_size, k_h, k_w)
    kwargs["groups"] = kwargs.get("groups", 1) * num_blocks
    y = F.conv2d(x, w, **kwargs)

    return y


def _torch_blockwise_linear(
    x: torch.Tensor, w: torch.Tensor, block_size: int, **kwargs
):
    """
    Perform block-wise linear transformation using torch operations.

    It splits the input and weight tensors into blocks along the input channel dimension and
    applies linear transformation to each block independently and stacks the results along a new dimension.
    """
    num_blocks = x.shape[0]
    c_out, _ = w.shape

    w = w.reshape(c_out, num_blocks, block_size).permute(1, 2, 0)
    xw = torch.bmm(x, w)

    return xw


def _timed_run(fn, *args, runs=5):
    times = []
    for _ in range(runs):
        start = time.time()
        fn(*args)
        times.append(time.time() - start)
    return sum(times) / len(times)


def _weightless_matmul_model():
    """model with dynamic MatMul"""
    input_tensor = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [4, 8]
    )
    state_tensor = onnx.helper.make_tensor_value_info(
        "state", onnx.TensorProto.FLOAT, [8, 8]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [4, 8]
    )
    w0 = onnx.numpy_helper.from_array(
        np.random.randn(8, 8).astype(np.float32), name="w0"
    )
    mm = onnx.helper.make_node("MatMul", ["input", "w0"], ["mm_out"], name="mm")
    state_mm = onnx.helper.make_node(
        "MatMul", ["mm_out", "state"], ["output"], name="state_mm"
    )
    graph = onnx.helper.make_graph(
        [mm, state_mm],
        "weightless_matmul",
        inputs=[input_tensor, state_tensor],
        outputs=[output_tensor],
        initializer=[w0],
    )
    model = models_for_tests.make_model(graph)
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize(
    "granularity, shape, channel_axis, block_axis",
    [
        ("per_block", (30, 15), 0, 1),
        ("per_channel", (30, 15), 0, 1),
        ("per_tensor", (30, 15), 0, 1),
        ("per_block", (300, 150), 1, 0),  # index shift by +1
        ("per_channel", (300, 150), 1, 0),
        ("per_tensor", (300, 150), 1, 0),
        ("per_block", (30, 15, 3, 3), 0, 1),
        ("per_channel", (30, 15, 3, 3), 0, 1),
        ("per_tensor", (30, 15, 3, 3), 0, 1),
        ("per_block", (300, 150, 3, 3), 1, 0),  # index shift by +1
        ("per_channel", (300, 150, 3, 3), 1, 0),
        ("per_tensor", (300, 150, 3, 3), 1, 0),
    ],
)
def test_min_max_for_candidate_selection(granularity, shape, channel_axis, block_axis):
    """
    Test _get_min_and_max_for_candidate_selection which returns absolute min/max
    values for different quantization granularity.
    """
    mock_layer = _mock_layer("Conv")

    calibration_tensor = np.random.randn(*shape).astype(np.float32)

    # Create quantizer
    tensor_quantizer_params = TensorQuantizerParams(shape, channel_axis, block_axis)
    quant_info = libquant_info.QcQuantizeInfo()
    bitwidth = 4
    quantizer = QcQuantizeOp(
        quant_info,
        bitwidth=bitwidth,
        op_mode=OpMode.updateStats,
        quant_scheme=QuantScheme.post_training_tf,
        tensor_quantizer_params=tensor_quantizer_params,
        use_symmetric_encodings=True,
    )
    # Set granularity
    if granularity == "per_channel":
        quantizer.enable_per_channel_quantization()
    elif granularity == "per_block":
        block_size = 3
        quantizer._enable_blockwise_quantization(block_size=block_size)
    else:
        pass
    quantizer.update_encoding_stats(calibration_tensor)
    quantizer.compute_encodings()

    # Mock sim object
    mock_sim = MagicMock()
    mock_sim._quant_scheme = QuantScheme.post_training_tf
    mock_sim.qc_quantize_op_dict = {"mock_weight": quantizer}

    # Mock seq_mse object w/o calling the __init__
    seq_mse = object.__new__(SequentialMse)
    seq_mse.sim = mock_sim

    _, max_val = seq_mse._get_min_and_max_for_candidate_selection(mock_layer)
    if granularity == "per_tensor":  # per-tensor and per-channel are handled similarly
        max_val = np.max(max_val)

    encodings = quantizer.get_encodings()
    enc_max = np.array([enc.max for enc in encodings])
    enc_min = np.array([enc.min for enc in encodings])
    delta = (enc_max - enc_min) / (2**bitwidth - 1)

    assert np.allclose(
        enc_max, max_val.flatten(), atol=delta
    )  # Allow 1-tick difference


@pytest.mark.cuda
@pytest.mark.parametrize("loss_fn", ["mse"])
@pytest.mark.parametrize(
    "param_bw, use_cuda, enable_pcq, best_indices",
    [
        (4, True, True, np.array([19, 19, 12, 18, 18, 18, 19, 15, 16, 16])),
        (31, True, True, np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19])),
        (4, True, False, np.array([[16, 15, 16, 18, 19, 15, 17, 16, 14, 17]])),
        (31, True, False, np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19])),
        (4, False, True, np.array([19, 19, 12, 18, 18, 18, 19, 15, 16, 16])),
        (31, False, True, np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19])),
        (4, False, False, np.array([[16, 15, 16, 18, 19, 15, 17, 16, 14, 17]])),
        (31, False, False, np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19])),
    ],
)
def test_apply_seq_mse_for_conv(
    loss_fn, param_bw, use_cuda, enable_pcq, best_indices, tmp_dir
):
    model = single_conv_layer_model()
    providers = ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

    # TODO: Workaround for test relying on np random state, fix test and remove
    make_dummy_input(model.model)
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=param_bw,
        providers=providers,
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=enable_pcq,
            dir_path=tmp_dir,
        ),
    )
    sim._compute_param_encodings(overwrite=False)

    all_param_ops = []
    for op in sim.connected_graph.ordered_ops:
        param_name = _get_weight_param_name(op)
        if param_name is not None:
            all_param_ops.append(param_name)

    assert len(all_param_ops) == 1

    quantizer = sim.qc_quantize_op_dict[all_param_ops[0]]
    assert not quantizer.is_encoding_frozen()

    inputs = [make_dummy_input(model.model) for _ in range(10)]
    num_candidates = 20
    seq_params = SeqMseParams(num_candidates=num_candidates, num_batches=2)
    seq_params.loss_fn = loss_fn
    seq_mse = SequentialMse(model, sim, seq_params, inputs)

    # Get the initial max tensor used to compute expected values.
    (layer,) = find_layers(sim._ir_model.graph, [])
    _, init_max = seq_mse._get_min_and_max_for_candidate_selection(layer)

    # Apply seq mse optimization
    seq_mse.apply_seq_mse_algo()

    # Encodings are frozen
    assert quantizer.is_encoding_frozen()

    """
    When: Given best indices
    Then: Expected max should be max_tensor / num_candidates * (indices + 1)
    """

    encodings = quantizer.get_encodings()
    actual_max = np.array([enc.max for enc in encodings]).reshape(
        quantizer._encoding_shape()
    )

    expected_max = init_max / num_candidates * (best_indices + 1)
    if not enable_pcq:
        expected_max = np.amax(expected_max)

    print(f"actual_max={actual_max}, expected_max={expected_max}")
    assert np.allclose(actual_max, expected_max)


@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ["mse", "l1", "sqnr"])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_static_apply_seq_mse(param_bw, loss_fn, enable_pcq, tmp_dir):
    model = single_conv_layer_model()
    tmp_path = tmp_dir
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=param_bw,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=enable_pcq,
            dir_path=tmp_path,
        ),
    )
    inputs = [make_dummy_input(model.model) for _ in range(10)]
    if loss_fn != "mse":
        seq_params = SeqMseParams(num_batches=1)
        seq_params.loss_fn = loss_fn
        SequentialMse.apply_seq_mse(model, sim, seq_params, inputs)
    else:
        apply_seq_mse(sim, inputs[:1])


def test_apply_seq_mse_with_tf_enhanced():
    """
    When: Apply seq mse to a sim created with the TF-Enhanced quant scheme
    Then: Weight encodings match those of a min-max sim, since seq mse needs exact min/max
          values, and the original quant scheme is restored
    """
    model = single_conv_layer_model()
    inputs = [make_dummy_input(model.model) for _ in range(2)]
    sim_kwargs = {
        "default_activation_bw": 8,
        "default_param_bw": 4,
        "providers": ["CPUExecutionProvider"],
    }
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        **sim_kwargs,
    )
    min_max_sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        **sim_kwargs,
    )

    apply_seq_mse(sim, inputs)
    apply_seq_mse(min_max_sim, inputs)

    quantizer = sim.qc_quantize_op_dict["conv.weight"]
    expected_quantizer = min_max_sim.qc_quantize_op_dict["conv.weight"]

    assert quantizer.quant_scheme == QuantScheme.post_training_tf_enhanced
    assert quantizer.is_encoding_frozen()
    assert np.allclose(
        [enc.max for enc in quantizer.get_encodings()],
        [enc.max for enc in expected_quantizer.get_encodings()],
    )


@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ["mse", "l1", "sqnr"])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_apply_seq_mse_for_split(param_bw, loss_fn, enable_pcq, tmp_dir):
    model = model_with_split()
    tmp_path = tmp_dir
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=param_bw,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=enable_pcq,
            dir_path=tmp_path,
        ),
    )
    inputs = [make_dummy_input(model.model) for _ in range(10)]
    if loss_fn != "mse":
        seq_params = SeqMseParams(num_batches=1)
        seq_params.loss_fn = loss_fn
        seq_mse = SequentialMse(model, sim, seq_params, inputs)
        seq_mse.apply_seq_mse_algo()
    else:
        apply_seq_mse(sim, inputs[:1])

    weight_quantizer_conv_1 = sim.qc_quantize_op_dict["conv1.weight"]
    weight_quantizer_conv_2 = sim.qc_quantize_op_dict["conv2.weight"]
    weight_quantizer_conv_3 = sim.qc_quantize_op_dict["conv3.weight"]

    assert weight_quantizer_conv_1.is_encoding_frozen()
    assert weight_quantizer_conv_2.is_encoding_frozen()
    assert weight_quantizer_conv_3.is_encoding_frozen()


def _levels(seq_mse):
    """Each level of the frontier plan as (op name, input names, output names) per layer."""
    return [
        [
            (layer.node.name, layer.op_input_names, layer.op_output_names)
            for layer in level
        ]
        for level in seq_mse._frontier_plan().levels
    ]


def _plain_seq_mse(model, tmp_dir):
    onnx_model = model.model if hasattr(model, "model") else model
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=True,
            dir_path=tmp_dir,
        ),
    )
    inputs = [make_dummy_input(onnx_model) for _ in range(10)]
    return SequentialMse(model, sim, SeqMseParams(num_batches=1), inputs)


def test_frontier_levels_for_split(tmp_dir):
    seq_mse = _plain_seq_mse(model_with_split(), tmp_dir)
    assert _levels(seq_mse) == [
        [("/conv1/Conv", ["input"], ["/conv1/Conv_output_0"])],
        [
            ("/conv2/Conv", ["/conv1/Conv_output_0"], ["/conv2/Conv_output_0"]),
            ("/conv3/Conv", ["/conv1/Conv_output_0"], ["/conv3/Conv_output_0"]),
        ],
    ]


def test_frontier_levels_for_residual_model(tmp_dir):
    seq_mse = _plain_seq_mse(single_residual_model(), tmp_dir)
    levels = _levels(seq_mse)
    assert [sorted(name for name, _, _ in level) for level in levels] == [
        ["/conv1/Conv"],
        ["/conv2/Conv", "/conv4/Conv"],
        ["/conv3/Conv"],
        ["/fc/Gemm"],
    ]
    layers = {name: (ins, outs) for level in levels for name, ins, outs in level}
    assert layers["/conv1/Conv"] == (["input"], ["/conv1/Conv_output_0"])
    assert layers["/conv2/Conv"] == (
        ["/maxpool/MaxPool_output_0"],
        ["/conv2/Conv_output_0"],
    )
    assert layers["/conv4/Conv"] == (
        ["/maxpool/MaxPool_output_0"],
        ["/conv4/Conv_output_0"],
    )
    assert layers["/conv3/Conv"] == (
        ["/relu2/Relu_output_0"],
        ["/conv3/Conv_output_0"],
    )


def test_frontier_levels_for_multiple_inputs(tmp_dir):
    """The second conv reads an Add of a graph input and the first conv's output."""
    seq_mse = _plain_seq_mse(model_with_multiple_inputs(), tmp_dir)
    levels = _levels(seq_mse)
    assert [[name for name, _, _ in level] for level in levels] == [
        ["Conv1"],
        ["Conv2"],
    ]
    assert levels[0][0][1] == ["X1"]
    assert levels[1][0][1] == ["ADD_0"]


def test_names_to_optimize_skips_control_flow_body_layers():
    """
    When: A Scan body holds a MatMul whose weight W_body is quantized, next to a top-level
          MatMul whose weight W_top is optimized by the frontier plan
    Then: names_to_optimize contains W_top but not W_body, since the plan never tunes
          layers inside control-flow bodies
    """
    dim = models_for_tests.DIM
    w_body = onnx.numpy_helper.from_array(
        np.random.randn(dim, dim).astype(np.float32), "W_body"
    )
    w_top = onnx.numpy_helper.from_array(
        np.random.randn(dim, dim).astype(np.float32), "W_top"
    )
    body = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["b_c", "W_body"], ["b_m"], name="body_mm"),
            onnx.helper.make_node("Add", ["b_m", "b_x"], ["b_co"], name="body_add"),
        ],
        "scan_body",
        inputs=[
            models_for_tests._ti("b_c", [dim, dim]),
            models_for_tests._ti("b_x", [dim, dim]),
        ],
        outputs=[models_for_tests._ti("b_co", [dim, dim])],
        initializer=[w_body],
    )
    model = models_for_tests._control_flow_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "MatMul", ["x", "W_top"], ["x_mm"], name="top_mm"
                ),
                onnx.helper.make_node(
                    "Scan",
                    ["init", "xs"],
                    ["final"],
                    name="the_scan",
                    body=body,
                    num_scan_inputs=1,
                ),
            ],
            "outer",
            inputs=[
                models_for_tests._ti("init", [dim, dim]),
                models_for_tests._ti("xs", [models_for_tests.TRIP, dim, dim]),
                models_for_tests._ti("x", [dim, dim]),
            ],
            outputs=[
                models_for_tests._ti("final", [dim, dim]),
                models_for_tests._ti("x_mm", [dim, dim]),
            ],
            initializer=[w_top],
        )
    )
    inputs = {
        "init": np.random.randn(dim, dim).astype(np.float32),
        "xs": np.random.randn(models_for_tests.TRIP, dim, dim).astype(np.float32),
        "x": np.random.randn(dim, dim).astype(np.float32),
    }
    with _quantize_subgraphs_enabled(True):
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            quant_scheme=QuantScheme.post_training_tf,
            providers=["CPUExecutionProvider"],
        )
    seq_mse = SequentialMse(model, sim, SeqMseParams(num_batches=1), [inputs])

    assert "W_body" in seq_mse.sim.qc_quantize_op_dict
    plan = seq_mse._frontier_plan()
    names = seq_mse._get_quantizer_names_to_optimize(plan)
    assert "W_top" in names
    assert "W_body" not in names


@pytest.mark.parametrize("param_bw", [2, 31])
@pytest.mark.parametrize("loss_fn", ["mse", "l1", "sqnr"])
@pytest.mark.parametrize("enable_pcq", [True, False])
def test_apply_seq_mse_for_residual_model(param_bw, loss_fn, enable_pcq, tmp_dir):
    model = single_residual_model()
    tmp_path = tmp_dir
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=param_bw,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=enable_pcq,
            dir_path=tmp_path,
        ),
    )
    inputs = [make_dummy_input(model.model) for _ in range(10)]
    if loss_fn != "mse":
        seq_params = SeqMseParams(num_batches=2)
        seq_params.loss_fn = loss_fn
        seq_mse = SequentialMse(model, sim, seq_params, inputs)
        seq_mse.apply_seq_mse_algo()
    else:
        apply_seq_mse(sim, inputs[:1])

    for conn_graph_op in sim.connected_graph.ordered_ops:
        if conn_graph_op.type in SUPPORTED_MODULES:
            param_names = [
                param_name
                for param_name, (_, param_type) in conn_graph_op.parameters.items()
                if param_type == "weight"
            ]
            assert len(param_names) == 1
            quantizer = sim.qc_quantize_op_dict[param_names[0]]
            assert quantizer.is_encoding_frozen()


def test_model_with_multiple_outputs_value_info(tmp_dir):
    model = model_with_multiple_outputs()
    tmp_path = tmp_dir
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=True,
            dir_path=tmp_path,
        ),
    )
    apply_seq_mse(sim, [make_dummy_input(model) for _ in range(1)])
    assert sim.qc_quantize_op_dict["Conv1_W"].is_encoding_frozen()
    assert sim.qc_quantize_op_dict["Conv2_W"].is_encoding_frozen()


def test_concat_model(tmp_dir: str):
    model = models_for_tests.concat_model()
    tmp_path = tmp_dir
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        config_file=_get_config_file(
            is_symmetric=True,
            strict_symmetric=False,
            unsigned_symmetric=False,
            pcq=True,
            dir_path=tmp_path,
        ),
    )

    apply_seq_mse(sim, [make_dummy_input(model.model)])
    for cg_op in sim.connected_graph.ordered_ops:
        if cg_op.type in SUPPORTED_MODULES:
            param_names = [
                param_name
                for param_name, (_, param_type) in cg_op.parameters.items()
                if param_type == "weight"
            ]
            assert len(param_names) == 1
            quantizer = sim.qc_quantize_op_dict[param_names[0]]
            assert quantizer.is_encoding_frozen()


def test_disable_subgraph_quantizers():
    model = models_for_tests.build_dummy_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        providers=["CPUExecutionProvider"],
        default_param_bw=4,
    )
    sim.compute_encodings(lambda sess: sess.run(None, make_dummy_input(model)))
    seq_params = SeqMseParams(num_batches=2)
    dataloader = [make_dummy_input(model) for _ in range(2)]
    seq_mse = SequentialMse(model, sim, seq_params, dataloader)

    enabled = {q for q in sim.qc_quantize_op_dict.values() if q.enabled}
    assert enabled

    with seq_mse._disable_subgraph_quantizers(sim.model.model):
        assert not any(q.enabled for q in sim.qc_quantize_op_dict.values())

    assert enabled == {q for q in sim.qc_quantize_op_dict.values() if q.enabled}

    subgraph_path = seq_mse._split_onnx_graph(["4"], ["output"])
    subgraph = onnx.load_model(subgraph_path, load_external_data=False)
    with seq_mse._disable_subgraph_quantizers(subgraph):
        assert not sim.qc_quantize_op_dict["fc_w"].enabled
        assert not sim.qc_quantize_op_dict["4"].enabled


@pytest.mark.parametrize("quantize_subgraphs", [False, True])
def test_scan_reading_outer_activation(quantize_subgraphs):
    model = models_for_tests.scan_reading_outer_activation_model()
    with _quantize_subgraphs_enabled(quantize_subgraphs):
        sim = QuantizationSimModel(
            model=model,
            quant_scheme=QuantScheme.post_training_tf,
            providers=["CPUExecutionProvider"],
            default_param_bw=4,
        )
    inputs = [make_dummy_input(model) for _ in range(2)]
    seq_mse = SequentialMse(model, sim, SeqMseParams(num_batches=2), inputs)

    """
    When: The Scan body reads a top-level activation that is not an input of the Scan node
    Then: The split graph includes its producer, and SeqMSE optimizes the weights around the Scan
    """
    subgraph_path = seq_mse._split_onnx_graph(["state_in", "xs"], ["state_out"])
    subgraph = onnx.load_model(subgraph_path, load_external_data=False)
    assert "the_trilu" in {node.name for node in subgraph.graph.node}

    apply_seq_mse(sim, inputs, num_candidates=4)
    assert sim.qc_quantize_op_dict["w_pre"].is_encoding_frozen()
    assert sim.qc_quantize_op_dict["w_post"].is_encoding_frozen()


def test_seq_mse_with_large_split_length_input():
    model = models_for_tests.matmul_split_concat_matmul_model()
    sim = QuantizationSimModel(
        model=model,
        quant_scheme=QuantScheme.post_training_tf,
        providers=["CPUExecutionProvider"],
        default_param_bw=4,
    )
    inputs = [make_dummy_input(model) for _ in range(2)]

    """
    When: Split's split_sizes input exceeds onnx_ir's default external data threshold of 256 bytes
    Then: It stays inline in the split graph, so ORT can infer Split's output shapes
    """
    seq_mse = SequentialMse(model, sim, SeqMseParams(num_batches=2), inputs)
    subgraph_path = seq_mse._split_onnx_graph(["input"], ["concat_out"])
    subgraph = onnx.load_model(subgraph_path, load_external_data=False)
    (split_sizes,) = (
        init for init in subgraph.graph.initializer if init.name == "split_sizes"
    )
    assert split_sizes.data_location == onnx.TensorProto.DEFAULT

    apply_seq_mse(sim, inputs, num_candidates=4)
    assert sim.qc_quantize_op_dict["w0"].is_encoding_frozen()
    assert sim.qc_quantize_op_dict["w1"].is_encoding_frozen()


def test_nodes_to_exclude():
    """Skip seq mse optimization if nodes to exclude provided"""
    model = single_residual_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
    )
    inputs = [make_dummy_input(model.model)]
    excluded_node = "/conv3/Conv"
    child_node = "/fc/Gemm"
    apply_seq_mse(sim, inputs, nodes_to_exclude=[excluded_node])

    for conn_graph_op in sim.connected_graph.ordered_ops:
        if conn_graph_op.type in SUPPORTED_MODULES:
            param_name = _get_weight_param_name(conn_graph_op)
            quantizer = sim.qc_quantize_op_dict[param_name]

            if conn_graph_op.name == excluded_node:
                assert not quantizer.is_encoding_frozen()
            elif conn_graph_op.name == child_node:
                assert quantizer.is_encoding_frozen()
            else:
                assert quantizer.is_encoding_frozen()


def _two_matmul_chain_with_r3_model():
    """input -> MatMul('mm') -> MatMul('spinquant_block0_k_R3') -> output."""
    input_tensor = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [4, 8]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [4, 8]
    )
    w0 = onnx.numpy_helper.from_array(
        np.random.randn(8, 8).astype(np.float32), name="w0"
    )
    # An R3 online Hadamard is a square matrix on the last axis.
    w1 = onnx.numpy_helper.from_array(
        np.random.randn(8, 8).astype(np.float32), name="spinquant_block0_k_R3_hadamard"
    )
    mm = onnx.helper.make_node("MatMul", ["input", "w0"], ["mm_out"], name="mm")
    r3 = onnx.helper.make_node(
        "MatMul",
        ["mm_out", "spinquant_block0_k_R3_hadamard"],
        ["output"],
        name="spinquant_block0_k_R3",
    )
    graph = onnx.helper.make_graph(
        [mm, r3],
        "two_matmul_chain_with_r3",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[w0, w1],
    )
    model = models_for_tests.make_model(graph)
    onnx.checker.check_model(model)
    return model


def test_seq_mse_excludes_r3_online_rotations(tmp_dir):
    """
    When: A model contains SpinQuant R3 online rotation MatMuls
          (named ``spinquant_block{idx}_{q|k}_R3``).
    Then: Sequential MSE auto-excludes them - their
          "weight" is a fixed Hadamard, and they would otherwise crash BQ
          sequence chunking under GQA (the Q-side and K-side rotations at the
          same level carry different folded-token sequence lengths, which ORT
          rejects as an io-binding shape mismatch) - while still optimizing
          ordinary MatMuls.
    """
    model = _two_matmul_chain_with_r3_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
    )
    inputs = [{"input": np.random.randn(4, 8).astype(np.float32)}]
    apply_seq_mse(sim, inputs)

    r3_node = "spinquant_block0_k_R3"
    regular_node = "mm"

    for cg_op in sim.connected_graph.ordered_ops:
        if cg_op.type not in SUPPORTED_MODULES:
            continue
        param_name = _get_weight_param_name(cg_op)
        quantizer = sim.qc_quantize_op_dict[param_name]
        if cg_op.name == r3_node:
            # R3 must be skipped: its encodings are left unfrozen.
            assert not quantizer.is_encoding_frozen()
        elif cg_op.name == regular_node:
            assert quantizer.is_encoding_frozen()


def test_seq_mse_skips_weightless_dynamic_matmul(tmp_dir):
    model = _weightless_matmul_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
    )
    inputs = [
        {
            "input": np.random.randn(4, 8).astype(np.float32),
            "state": np.random.randn(8, 8).astype(np.float32),
        }
    ]

    seq_mse = SequentialMse(model, sim, SeqMseParams(num_batches=1), inputs)

    """
    When: A model contains a weightless dynamic MatMul (activation x model-input.
    Then: Seq MSE optimizing (freezing) only the static-weight MatMul and skipping the weightless one,
     which has no weight param to optimize.
    """
    nodes = {node.name: node for node in sim._ir_model.graph}
    assert is_supported_op(nodes["mm"])
    assert not is_supported_op(nodes["state_mm"])
    assert [[name for name, _, _ in level] for level in _levels(seq_mse)] == [["mm"]]

    seq_mse.apply_seq_mse_algo()

    assert sim.qc_quantize_op_dict["w0"].is_encoding_frozen()
    assert not sim.qc_quantize_op_dict["state"].is_encoding_frozen()


def test_find_layers_weight_through_transpose():
    """
    When: A MatMul reads its weight through a Transpose
    Then: Its layer is the weight's, and the Transpose's output is not one of its activations
    """
    weight = onnx_ir.Value(
        name="w", const_value=onnx_ir.tensor(np.zeros((4, 3), np.float32))
    )
    x = onnx_ir.Value(name="x")
    transpose = onnx_ir.Node(
        "",
        "Transpose",
        [weight],
        name="t",
        attributes=[onnx_ir.AttrInt64s("perm", [1, 0])],
    )
    layer_node = onnx_ir.Node("", "MatMul", [x, transpose.outputs[0]], name="mm")
    graph = onnx_ir.Graph(
        [x], layer_node.outputs, nodes=[transpose, layer_node], initializers=[weight]
    )

    (layer,) = find_layers(graph, [])

    assert get_weight_name(layer.node) == "w"
    assert layer.op_input_names == ["x"]


def test_find_layers_skips_gemm_weight_through_transpose():
    """
    When: A Gemm reads its weight through a Transpose
    Then: It is not a layer, because QuantSim only gives a Transpose-read weight a weight
          quantizer for MatMul
    """
    weight = onnx_ir.Value(
        name="w", const_value=onnx_ir.tensor(np.zeros((4, 3), np.float32))
    )
    x = onnx_ir.Value(name="x")
    transpose = onnx_ir.Node(
        "",
        "Transpose",
        [weight],
        name="t",
        attributes=[onnx_ir.AttrInt64s("perm", [1, 0])],
    )
    layer_node = onnx_ir.Node("", "Gemm", [x, transpose.outputs[0]], name="mm")
    graph = onnx_ir.Graph(
        [x], layer_node.outputs, nodes=[transpose, layer_node], initializers=[weight]
    )

    assert not is_supported_op(layer_node)
    assert find_layers(graph, []) == []


def test_apply_seq_mse_with_gemm_weight_through_transpose():
    """
    When: A Gemm reads its weight through a Transpose
    Then: Sequential MSE runs to completion without raising, since the Gemm is not a layer
    """
    weight = onnx.numpy_helper.from_array(
        np.random.randn(8, 4).astype(np.float32), name="W"
    )
    transpose = onnx.helper.make_node(
        "Transpose", ["W"], ["W_t"], name="t", perm=[1, 0]
    )
    gemm = onnx.helper.make_node("Gemm", ["x", "W_t"], ["y"], name="gemm", transB=0)
    graph = onnx.helper.make_graph(
        [transpose, gemm],
        "gemm_weight_through_transpose",
        inputs=[
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 8])
        ],
        initializer=[weight],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )

    seq_mse = _seq_mse_for(model, {"x": np.random.randn(1, 4).astype(np.float32)})
    assert not seq_mse._frontier_plan().levels
    seq_mse.apply_seq_mse_algo()


def test_find_layers_skips_layer_with_no_dynamic_input():
    """
    When: A Gemm's non-weight input is also a constant (a valid but unusual ONNX pattern --
          e.g. the activation got constant-folded upstream)
    Then: find_layers does not produce a layer with an empty op_input_names list
    """
    a_const = onnx_ir.Value(
        name="a", const_value=onnx_ir.tensor(np.zeros((4, 4), np.float32))
    )
    weight = onnx_ir.Value(
        name="w", const_value=onnx_ir.tensor(np.zeros((4, 4), np.float32))
    )
    gemm = onnx_ir.Node("", "Gemm", [a_const, weight], name="g", num_outputs=1)
    graph = onnx_ir.Graph(
        [], gemm.outputs, nodes=[gemm], initializers=[a_const, weight]
    )

    assert find_layers(graph, []) == []


def test_plan_frontier_outer_scope_consumer_before_producer():
    """
    When: A top-level If node is listed before the Add whose output its branches capture
          as an outer-scope value by name (legal under onnx_ir's topological sort, which
          only orders nodes by their explicit node.inputs)
    Then: plan_frontier does not raise KeyError
    """
    x = onnx_ir.Value(name="x")
    cond = onnx_ir.Value(name="cond", const_value=onnx_ir.tensor(np.array(True)))

    mid = onnx_ir.Node("", "Add", [x, x], name="mid", num_outputs=1)
    v = mid.outputs[0]
    v.name = "v"

    then_id = onnx_ir.Node("", "Identity", [v], name="then_id", num_outputs=1)
    then_branch = onnx_ir.Graph(
        [], then_id.outputs, nodes=[then_id], name="then_branch"
    )
    else_id = onnx_ir.Node("", "Identity", [v], name="else_id", num_outputs=1)
    else_branch = onnx_ir.Graph(
        [], else_id.outputs, nodes=[else_id], name="else_branch"
    )
    if_node = onnx_ir.Node(
        "",
        "If",
        [cond],
        name="n",
        num_outputs=1,
        attributes=[
            onnx_ir.AttrGraph("then_branch", then_branch),
            onnx_ir.AttrGraph("else_branch", else_branch),
        ],
    )

    # `n` (the capturing If) is listed before `mid` (the producer of the value it
    # captures) -- legal, since If's only explicit input is `cond`.
    graph = onnx_ir.Graph(
        [x, cond], list(if_node.outputs) + list(mid.outputs), nodes=[if_node, mid]
    )

    plan_frontier(graph, find_layers(graph, []))  # must not raise KeyError


@pytest.mark.cuda
@pytest.mark.parametrize("io_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_size", [None, 8])
def test_seq_mse_with_bfloat16_model(io_dtype, block_size):
    """
    Given: Model computing in bfloat16, with float32 or bfloat16 graph inputs/outputs
    When: Apply sequential MSE with per-channel or LPBQ weight quantizers
    Then: Every weight encoding is optimized and frozen, and stays close to that of the
          equivalent float32 model
    """
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDA EP not available")

    rng = np.random.default_rng(0)
    fp32_inputs = [
        {"input": rng.standard_normal((4, 16)).astype(np.float32)} for _ in range(4)
    ]

    def run_seq_mse(model, inputs):
        sim = QuantizationSimModel(
            model,
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=16,
            default_param_bw=4,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if block_size is not None:
            set_grouped_blockwise_quantization_for_weights(
                sim,
                op_types=("MatMul", "Gemm"),
                bitwidth=4,
                decompressed_bw=8,
                block_size=block_size,
                strict=True,
            )
        apply_seq_mse(sim, inputs, num_candidates=8)
        return sim

    bf16_inputs = (
        fp32_inputs
        if io_dtype == torch.float32
        else [
            {"input": item["input"].astype(ml_dtypes.bfloat16)} for item in fp32_inputs
        ]
    )
    fp32_sim = run_seq_mse(
        models_for_tests.simple_mlp_model(torch.float32), fp32_inputs
    )
    bf16_sim = run_seq_mse(
        models_for_tests.simple_mlp_model(torch.bfloat16, io_dtype=io_dtype),
        bf16_inputs,
    )

    for name in ("fc1.weight", "fc2.weight"):
        bf16_quantizer = bf16_sim.qc_quantize_op_dict[name]
        assert bf16_quantizer.is_encoding_frozen(), name
        fp32_max = np.array(
            [e.max for e in fp32_sim.qc_quantize_op_dict[name].get_encodings()]
        )
        bf16_max = np.array([e.max for e in bf16_quantizer.get_encodings()])
        # bfloat16 rounding can tip a channel to a neighboring candidate (1/8 of its range).
        assert np.allclose(fp32_max, bf16_max, atol=0.15 * np.abs(fp32_max).max()), name

    """
    When: Calibrate activations after sequential MSE
    Then: The sim runs end-to-end on bfloat16 inputs
    """
    bf16_sim.compute_encodings(bf16_inputs)


def test_temporarily_disable_grouped_block_quantizers():
    model = single_residual_model()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
    )
    set_grouped_blockwise_quantization_for_weights(
        sim, "Conv", 4, 8, block_size=2, strict=False
    )

    quantizers = [
        sim.qc_quantize_op_dict[name]
        for name in sim.param_names
        if sim.qc_quantize_op_dict[name].enabled
        and sim.qc_quantize_op_dict[name]._scale_quantizer is not None
    ]
    assert len(quantizers) == 3

    for quantizer in quantizers:
        assert quantizer._scale_quantizer is not None

    """
    When: _temporarily_disable_grouped_block_quantizers is called
    Then: scale quantization is disabled during the context manager scope and restored after
    """

    original_scale_quantizers = [quantizer._scale_quantizer for quantizer in quantizers]

    with _temporarily_disable_block_grouping(sim):
        for quantizer in quantizers:
            assert quantizer._scale_quantizer is None

    for quantizer, original in zip(quantizers, original_scale_quantizers):
        assert quantizer._scale_quantizer is original


@pytest.mark.parametrize("loss_fn", ["mse"])
@pytest.mark.parametrize(
    "op_type, tensor_shape, output_shape, channel_axis, block_axis, block_size",
    [
        ("Conv", (32, 16, 5, 5), (2, 32, 6, 6), 1, 0, 0),
        ("Gemm", (15, 30), (2, 30), 1, 0, 0),
        ("MatMul", (15, 30), (2, 10, 30), 1, 0, 0),
        (
            "Conv",
            (32, 16, 5, 5),
            (2, 128, 6, 6),
            0,
            1,
            4,
        ),  # channel_axis, block_axis (0, 1)
        ("Gemm", (30, 15), (5, 2, 30), 0, 1, 3),  # channel_axis, block_axis (0, 1)
        ("Gemm", (15, 30), (5, 2, 30), 1, 0, 3),  # channel_axis, block_axis (1, 0)
        ("MatMul", (15, 30), (5, 20, 30), 1, 0, 3),  # channel_axis, block_axis (1, 0)
    ],
)
def test_compute_reconstruction_loss(
    loss_fn, op_type, tensor_shape, output_shape, channel_axis, block_axis, block_size
):
    # Setup mocks
    mock_layer = _mock_layer(op_type)

    mock_quantizer = MagicMock()
    mock_quantizer.tensor_quantizer_params.tensor_shape = tensor_shape
    mock_quantizer.quant_info.channelAxis = channel_axis
    mock_quantizer.quant_info.blockAxis = block_axis
    mock_quantizer.quant_info.blockSize = block_size

    mock_sim = MagicMock()
    mock_sim.qc_quantize_op_dict = {"mock_weight": mock_quantizer}

    # Mock seq_mse object w/o calling the __init__
    seq_mse = object.__new__(SequentialMse)
    seq_mse.sim = mock_sim
    seq_mse.params = MagicMock()
    seq_mse.params.loss_fn = loss_fn

    # Random outputs
    sim_output = torch.from_numpy(np.random.randn(*output_shape).astype(np.float32))
    float_output = torch.from_numpy(np.random.randn(*output_shape).astype(np.float32))

    loss = seq_mse._compute_recon_loss(sim_output, float_output, mock_layer)
    print(f"actual loss: {loss.shape}")

    # block-wise quantization
    if block_size > 0:
        if op_type == "Conv":
            c_out = tensor_shape[channel_axis]
            num_blocks = output_shape[1] // c_out
            expected_shape = (c_out, num_blocks)
        elif op_type == "MatMul":
            c_out, num_blocks = output_shape[-1], output_shape[0]
            expected_shape = (num_blocks, c_out)
        elif op_type == "Gemm":
            c_out, num_blocks = output_shape[-1], output_shape[0]
            if block_axis < channel_axis:
                expected_shape = (num_blocks, c_out)
            else:
                expected_shape = (c_out, num_blocks)  # Handle transposed form of Gemm
        else:
            raise NotImplementedError

    # per-tensor, per-channel quantization
    else:
        expected_shape = (output_shape[1] if op_type == "Conv" else output_shape[-1],)
    assert loss.shape == expected_shape


@pytest.mark.parametrize("bitwidth", [4, 8])
@pytest.mark.parametrize("is_symmetric", [True, False])
@pytest.mark.parametrize(
    "granularity, tensor_shape, x_min, x_max, channel_axis, block_axis",
    [
        ("per_tensor", (4, 4), np.array([-1.0]), np.array([1.0]), 0, 0),
        (
            "per_channel",
            (4, 4),
            np.array([-1.0, -2.0, -3.0, -4.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            0,
            0,
        ),
        (
            "per_channel",
            (4, 4),
            np.array([-1.0, -2.0, -3.0, -4.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            1,
            0,
        ),
        (
            "per_block",
            (4, 4),
            np.array([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0]]),
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            0,
            1,
        ),
        (
            "per_block",
            (4, 4),
            np.array([[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]]),
            np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            1,
            0,
        ),
    ],
)
def test_compute_encoding_from_candidate(
    bitwidth,
    is_symmetric,
    granularity,
    tensor_shape,
    x_min,
    x_max,
    channel_axis,
    block_axis,
):
    # Create quantizer
    tensor_quantizer_params = TensorQuantizerParams(
        tensor_shape, channel_axis, block_axis
    )
    quant_info = libquant_info.QcQuantizeInfo()
    quantizer = QcQuantizeOp(
        quant_info,
        bitwidth=bitwidth,
        op_mode=OpMode.updateStats,
        quant_scheme=QuantScheme.post_training_tf,
        tensor_quantizer_params=tensor_quantizer_params,
        use_symmetric_encodings=is_symmetric,
    )
    if granularity == "per_channel":
        quantizer.enable_per_channel_quantization()
    elif granularity == "per_block":
        block_size = 2
        quantizer._enable_blockwise_quantization(block_size)

    tensor = np.random.randn(*tensor_shape).astype(np.float32)
    quantizer.reset_encoding_stats()
    quantizer.update_encoding_stats(tensor)
    quantizer.compute_encodings()

    mock_sim = MagicMock()
    mock_sim.qc_quantize_op_dict = {"mock_weight": quantizer}

    # Mock seq_mse object w/o calling the __init__
    seq_mse = object.__new__(SequentialMse)
    seq_mse.sim = mock_sim

    mock_layer = _mock_layer("Gemm")
    seq_mse._compute_encoding_from_candidate(mock_layer, x_min, x_max)
    encodings = quantizer.get_encodings()
    enc_min = np.array([enc.min for enc in encodings])
    enc_max = np.array([enc.max for enc in encodings])

    if granularity == "per_tensor":
        x_min, x_max = np.amin(x_min), np.amax(x_max)

    print(f"x_min={x_min}, x_max={x_max}")
    print(f"enc_min={enc_min}, enc_max={enc_max}")

    assert np.allclose(
        enc_min, x_min.flatten(), atol=(enc_max - enc_min) / (2**bitwidth - 1)
    )
    assert np.allclose(
        enc_max, x_max.flatten(), atol=(enc_max - enc_min) / (2**bitwidth - 1)
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "swap_quantizer_func",
    [
        partial(
            set_grouped_blockwise_quantization_for_weights,
            op_types=("MatMul", "Conv", "Gemm"),
            decompressed_bw=8,
            strict=True,
        ),
        partial(
            set_blockwise_quantization_for_weights,
            op_types=("MatMul", "Conv", "Gemm"),
            strict=True,
            symmetric=True,
        ),
    ],
)
@pytest.mark.parametrize(
    "use_cuda, model_kind, do_constant_folding, block_size, channel_i_best_indices",
    [
        (
            True,
            "conv",
            None,
            1,
            {0: np.array([19, 19, 17, 17, 19])},
        ),  # weights shape (10, 5, 5, 5) where c_in=5, block_axis=1
        (
            True,
            "linear",
            True,
            25,
            {0: np.array([18, 18, 19, 18])},
        ),  # weights shape (100, 100) where c_in=100, block_axis=0
        (
            True,
            "linear",
            False,
            25,
            {0: np.array([18, 18, 19, 18])},
        ),  # weights shape (100, 100) where c_in=100, block_axis=1
        (
            False,
            "conv",
            None,
            1,
            {0: np.array([19, 19, 17, 17, 19])},
        ),  # weights shape (10, 5, 5, 5) where c_in=5, block_axis=1
        (
            False,
            "linear",
            True,
            25,
            {0: np.array([18, 18, 19, 18])},
        ),  # weights shape (100, 100) where c_in=100, block_axis=0
        (
            False,
            "linear",
            False,
            25,
            {0: np.array([18, 18, 19, 18])},
        ),  # weights shape (100, 100) where c_in=100, block_axis=1
    ],
)
def test_bq_lpbq_single_layer(
    swap_quantizer_func,
    use_cuda,
    model_kind,
    do_constant_folding,
    block_size,
    channel_i_best_indices,
):
    if model_kind == "conv":
        model = single_conv_layer_model()
    elif model_kind == "linear":
        model = single_linear_layer_model(do_constant_folding=bool(do_constant_folding))
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    providers = ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

    # TODO: Workaround for test relying on np random state, fix test and remove
    make_dummy_input(model.model)
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=providers,
    )

    swap_quantizer_func(sim, bitwidth=4, block_size=block_size)
    with _temporarily_disable_block_grouping(sim):
        sim._compute_param_encodings(overwrite=False)

    all_param_ops = []
    for op in sim.connected_graph.ordered_ops:
        param_name = _get_weight_param_name(op)
        if param_name is not None:
            all_param_ops.append(param_name)

    assert len(all_param_ops) == 1

    quantizer = sim.qc_quantize_op_dict[all_param_ops[0]]
    assert not quantizer.is_encoding_frozen()

    num_candidates = 20
    seq_mse_params = SeqMseParams(num_batches=None, num_candidates=num_candidates)
    seq_mse = SequentialMse(
        None, sim, seq_mse_params, [make_dummy_input(model.model) for _ in range(3)]
    )

    (layer,) = find_layers(sim._ir_model.graph, [])
    _, init_max = seq_mse._get_min_and_max_for_candidate_selection(layer)

    seq_mse.apply_seq_mse_algo()

    # Encodings are frozen
    assert quantizer.is_encoding_frozen()

    ((channel_i, best_indices),) = channel_i_best_indices.items()

    if model_kind == "conv":
        channel_0_init_max = init_max[channel_i, :, 0, 0]
    else:
        if do_constant_folding:
            channel_0_init_max = init_max[
                :,
                channel_i,
            ]  # For MatMul, Gemm (untransposed) weights in shape (c_in, c_out)
        else:
            channel_0_init_max = init_max[channel_i, :]
    """
    When: Given best indices for output channel 0
    Then: Expected max should be max_tensor / num_candidates * (indices + 1)
    """

    encodings = quantizer.get_encodings()
    actual_max = np.array([enc.max for enc in encodings]).reshape(
        quantizer._encoding_shape()
    )
    if model_kind == "conv":
        channel_0_actual_max = actual_max[channel_i, :, 0, 0]
    else:
        if do_constant_folding:
            channel_0_actual_max = actual_max[:, channel_i]
        else:
            channel_0_actual_max = actual_max[channel_i, :]

    channel_0_expected_max = channel_0_init_max / num_candidates * (best_indices + 1)

    print(
        f"channel_0_actual_max={channel_0_actual_max}, channel_0_expected_max={channel_0_expected_max}"
    )
    assert np.allclose(channel_0_actual_max, channel_0_expected_max)


@pytest.mark.parametrize(
    "swap_quantizer_func",
    [
        partial(
            set_grouped_blockwise_quantization_for_weights,
            op_types=("MatMul", "Conv", "Gemm"),
            decompressed_bw=8,
            strict=True,
        ),
        partial(
            set_blockwise_quantization_for_weights,
            op_types=("MatMul", "Conv", "Gemm"),
            strict=True,
            symmetric=True,
        ),
    ],
)
@pytest.mark.parametrize(
    "model_factory, block_size",
    [
        (single_residual_model, 1),
        (models_for_tests.concat_model, 1),
        (models_for_tests.linear_split_into_matmul_add, 1),
        (model_with_split, 1),
    ],
)
def test_bq_lpbq_functional(swap_quantizer_func, model_factory, block_size):
    model = model_factory()
    sim = QuantizationSimModel(
        model=copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_activation_bw=8,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
    )

    """
    When: strict=True and block_size=1
    Then: Block quantization is enabled for all the supported Conv, Gemm and MatMul ops
    """

    swap_quantizer_func(sim, bitwidth=4, block_size=block_size)

    inputs = [make_dummy_input(model.model) for _ in range(10)]
    apply_seq_mse(sim, inputs[:1])

    for conn_graph_op in sim.connected_graph.ordered_ops:
        param_name = _get_weight_param_name(conn_graph_op)
        if param_name is not None:
            quantizer = sim.qc_quantize_op_dict[param_name]
            assert quantizer.is_encoding_frozen()


def test_rechunk_seq_inputs_shapes():
    """test _rechunk_seq_inputs shapes"""
    num_blocks, seq_len, block_size = 4, 100, 8
    chunk_size = 30  # yields chunks of size 30, 30, 30, 10

    c_out = 16
    arr = np.random.randn(num_blocks, seq_len, block_size).astype(np.float32)
    sim_inputs = {"x": [arr]}
    all_out_shapes = [[(num_blocks, seq_len, c_out)]]
    all_out_dtypes = [[np.dtype("float32")]]

    new_inputs, new_shapes, new_dtypes = _rechunk_seq_inputs(
        sim_inputs, chunk_size, all_out_shapes, all_out_dtypes
    )

    """
    When: _rechunk_seq_inputs
    Then: Split dim=1 into chunks of at most chunk_size, produce correct output shapes, and be lossless
     (concatenating chunks restores the original array)
    """

    chunks = new_inputs["x"]
    assert len(chunks) == 4
    assert chunks[0].shape == (num_blocks, 30, block_size)
    assert chunks[1].shape == (num_blocks, 30, block_size)
    assert chunks[2].shape == (num_blocks, 30, block_size)
    assert chunks[3].shape == (num_blocks, 10, block_size)

    assert np.allclose(np.concatenate(chunks, axis=1), arr)

    """
    When: _rechunk_seq_inputs
    Then: Output shapes must mirror the chunk seq_len
    """

    assert new_shapes[0] == [(num_blocks, 30, c_out)]
    assert new_shapes[3] == [(num_blocks, 10, c_out)]
    assert len(new_dtypes) == 4
    assert all(d == [np.dtype("float32")] for d in new_dtypes)


def test_rechunk_seq_inputs_multiple_batches():
    """_rechunk_seq_inputs with multiple batches"""
    num_blocks, seq_len, block_size, c_out = 2, 60, 4, 8
    chunk_size = 20  # 3 chunks per original batch
    num_batches = 3

    arrs = [
        np.random.randn(num_blocks, seq_len, block_size).astype(np.float32)
        for _ in range(num_batches)
    ]
    sim_inputs = {"x": arrs}
    all_out_shapes = [[(num_blocks, seq_len, c_out)]] * num_batches
    all_out_dtypes = [[np.dtype("float32")]] * num_batches

    """
    When: _rechunk_seq_inputs
    Then: should apply chunking independently to every batch in sim_inputs,
     expanding dataset_len proportionally
    """

    new_inputs, new_shapes, new_dtypes = _rechunk_seq_inputs(
        sim_inputs, chunk_size, all_out_shapes, all_out_dtypes
    )

    assert len(new_inputs["x"]) == 9
    assert len(new_shapes) == 9
    assert len(new_dtypes) == 9

    for chunk in new_inputs["x"]:
        assert chunk.shape == (num_blocks, chunk_size, block_size)


def test_compute_seq_chunk_size_caps_cudnn_output_elements(monkeypatch):
    """Limit each output chunk to the maximum element count supported by cuDNN."""
    seq_len = 3000
    output_elements_per_token = 2**20

    # Memory alone would allow the full sequence to run in one chunk.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (80 * 1024**3, 0))

    chunk_size = _compute_seq_chunk_size(
        num_blocks=1,
        sim_inputs={"input": [np.empty((1, 1, 1, seq_len), dtype=np.float32)]},
        all_out_shapes=[[(1, output_elements_per_token, 1, seq_len)]],
        all_out_dtypes=[[np.dtype("float32")]],
        torch_device=torch.device("cuda"),
        chunk_dim=3,
    )

    expected = np.iinfo(np.int32).max // output_elements_per_token
    assert chunk_size == expected


@pytest.mark.cuda
def test_lpbq_seq_mse_chunked_matches_full(monkeypatch):
    """Verify encodings full sequence vs chunk_size=1"""
    import aimet_onnx.sequential_mse.seq_mse as seq_mse_module

    model = single_linear_layer_model(do_constant_folding=True)
    providers = ["CUDAExecutionProvider"]

    def _make_sim():
        sim = QuantizationSimModel(
            model=copy.deepcopy(model),
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=8,
            default_param_bw=4,
            providers=providers,
        )
        set_grouped_blockwise_quantization_for_weights(
            sim,
            op_types=("MatMul", "Gemm"),
            bitwidth=4,
            decompressed_bw=8,
            block_size=25,
            strict=True,
        )
        with _temporarily_disable_block_grouping(sim):
            sim._compute_param_encodings(overwrite=False)
        return sim

    inputs = [make_dummy_input(model.model) for _ in range(4)]
    params = SeqMseParams(num_batches=None, num_candidates=20)

    """
    When: Sequence chunking is applied
    Then: Verify that it produces identical encodings to a full-sequence run (seq_len=100).
    """

    # normal path (no sequential chunking, seq_len=100)
    sim_full = _make_sim()
    SequentialMse(None, sim_full, params, inputs).apply_seq_mse_algo()

    # force chunk_size=1
    monkeypatch.setattr(
        seq_mse_module,
        "_compute_seq_chunk_size",
        lambda *args, **kwargs: 1,
    )
    sim_chunked = _make_sim()
    SequentialMse(None, sim_chunked, params, inputs).apply_seq_mse_algo()

    # Encoding max must be identical between both runs
    for name in sim_full.qc_quantize_op_dict:
        enc_full = sim_full.qc_quantize_op_dict[name].get_encodings()
        enc_chunked = sim_chunked.qc_quantize_op_dict[name].get_encodings()
        if enc_full is None:
            continue
        full_max = np.array([e.max for e in enc_full])
        chunked_max = np.array([e.max for e in enc_chunked])
        assert np.allclose(full_max, chunked_max)


def _full_model_values(seq_mse, names):
    """``names`` per calibration sample, computed by one run of the whole model."""
    names = list(names)
    with tempfile.TemporaryDirectory() as tempdir:
        session = _build_session(
            seq_mse._ir_model, seq_mse._inputs[0], names, seq_mse.sim.providers, tempdir
        )
        values = [
            _run_session_to_numpy(session, sample, names) for sample in seq_mse._inputs
        ]
        del session
    return values


def _assert_sampled_match_one_shot(seq_mse, atol=1e-6):
    """Sampling every level blockwise gives the tensors one run of the whole model does."""
    plan = seq_mse._frontier_plan()
    sampler = BlockwiseSampler(
        seq_mse._ir_model,
        SamplingPlan(seq_mse._ir_model.graph, plan.block_inputs, plan.block_outputs),
        seq_mse._inputs,
        seq_mse.sim.providers,
    )
    graph_inputs = set(seq_mse._inputs[0])
    for idx, names in enumerate(plan.block_inputs):
        samples = sampler.sample_block(idx)
        computed = [i for i, name in enumerate(names) if name not in graph_inputs]
        if not computed:
            continue
        expected = _full_model_values(seq_mse, [names[i] for i in computed])
        for sample, reference in zip(samples, expected):
            for i, ref in zip(computed, reference):
                assert sample[i].shape == ref.shape, names[i]
                np.testing.assert_allclose(
                    sample[i].astype(np.float32),
                    ref.astype(np.float32),
                    atol=atol,
                    err_msg=names[i],
                )


class TestFrontierLevels:
    @pytest.mark.parametrize(
        "model, cached_data",
        [
            (
                models_for_tests.single_residual_model(),
                {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)},
            ),
            (
                models_for_tests.concat_model(),
                {
                    "input1": np.random.randn(1, 3, 8, 8).astype(np.float32),
                    "input2": np.random.randn(1, 3, 8, 8).astype(np.float32),
                    "input3": np.random.randn(1, 3, 8, 8).astype(np.float32),
                },
            ),
            (
                models_for_tests.multi_input_model(),
                {
                    "input1": np.random.randn(32, 1, 28, 28).astype(np.float32),
                    "input2": np.random.randn(32, 1, 28, 28).astype(np.float32),
                },
            ),
            (
                models_for_tests.mobilenetv2(),
                {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)},
            ),
            (
                models_for_tests.resnet18(),
                {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)},
            ),
        ],
    )
    def test_sampled_inputs_match_one_shot(self, model, cached_data):
        """
        When: Every level's inputs are sampled blockwise from the cached frontier
        Then: They match one run of the whole model
        """
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            quant_scheme=QuantScheme.post_training_tf,
            providers=["CPUExecutionProvider"],
        )
        for quantizer in sim.qc_quantize_op_dict.values():
            quantizer.enabled = False
        seq_mse = SequentialMse(model, sim, SeqMseParams(num_batches=1), [cached_data])
        _assert_sampled_match_one_shot(seq_mse)

    @pytest.mark.parametrize(
        "model, expected",
        [
            (
                models_for_tests.concat_model(),
                [["/conv1/Conv", "/conv2/Conv", "/conv3/Conv"], ["/fc/Gemm"]],
            ),
            (
                models_for_tests.multi_input_model(),
                [
                    ["/conv1_a/Conv", "/conv1_b/Conv"],
                    ["/conv2/Conv"],
                    ["/fc1/Gemm"],
                    ["/fc2/Gemm"],
                ],
            ),
        ],
    )
    def test_levels(self, model, expected):
        """Layers that read only earlier levels' outputs share a level."""
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            quant_scheme=QuantScheme.post_training_tf,
            providers=["CPUExecutionProvider"],
        )
        seq_mse = SequentialMse(
            model, sim, SeqMseParams(num_batches=1), [make_dummy_input(model.model)]
        )
        assert [sorted(name for name, _, _ in level) for level in _levels(seq_mse)] == (
            expected
        )

    def test_nodes_to_exclude(self):
        """
        When: A layer is excluded
        Then: It is not planned, and the layers after it still are
        """
        model = models_for_tests.single_residual_model()
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            quant_scheme=QuantScheme.post_training_tf,
            providers=["CPUExecutionProvider"],
        )
        seq_mse = SequentialMse(
            model,
            sim,
            SeqMseParams(num_batches=1),
            [make_dummy_input(model.model)],
            nodes_to_exclude=["/conv1/Conv"],
        )
        planned = {name for level in _levels(seq_mse) for name, _, _ in level}

        assert "/conv1/Conv" not in planned
        assert {"/conv2/Conv", "/fc/Gemm"} <= planned


class TestBlockWiseReconLoss:
    @pytest.mark.cuda
    @pytest.mark.parametrize(
        "use_cuda, weight_shape, input_shape, output_shape, block_size, block_axis, kwargs",
        [
            (
                True,
                (32, 16, 5, 5),
                (2, 16, 10, 10),
                (2, 32, 6, 6),
                4,
                1,
                dict(groups=1),
            ),  # Regular conv (groups=1)
            (
                False,
                (32, 16, 5, 5),
                (2, 16, 10, 10),
                (2, 32, 6, 6),
                4,
                1,
                dict(groups=1),
            ),  # Regular conv (groups=1)
            (
                True,
                (512, 256, 5, 5),
                (1, 256, 10, 10),
                (1, 512, 6, 6),
                4,
                1,
                dict(groups=1),
            ),  # Regular conv (groups=1)
            (
                False,
                (512, 256, 5, 5),
                (1, 256, 10, 10),
                (1, 512, 6, 6),
                4,
                1,
                dict(groups=1),
            ),  # Regular conv (groups=1)
        ],
    )
    def test_blockwise_conv(
        self,
        use_cuda,
        weight_shape,
        input_shape,
        output_shape,
        block_size,
        block_axis,
        kwargs,
    ):
        providers = ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        device = torch.device("cuda:0" if use_cuda else "cpu")

        weight = np.random.randn(*weight_shape).astype(np.float32)
        kwargs_for_onnx = kwargs.copy()
        if "groups" in kwargs_for_onnx:
            kwargs_for_onnx["group"] = kwargs_for_onnx.pop("groups")

        onnx_model = _onnx_model_from_op(
            "Conv", weight, input_shape, output_shape, **kwargs_for_onnx
        )

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        session = ort.InferenceSession(
            onnx_model.SerializeToString(), providers=providers
        )
        orig_graph_output = session.run(None, {"input": dummy_input})[0]

        # In-place modify Conv with block-wise grouped Conv
        modify_graph_with_grouped_conv(onnx_model, block_size, block_axis)
        onnx.checker.check_model(onnx_model)

        session = ort.InferenceSession(
            onnx_model.SerializeToString(), providers=providers
        )
        modified_graph_output = session.run(None, {"input": dummy_input})[0]
        c_out, c_in, k_h, k_w = weight_shape
        N, _, h_out, w_out = modified_graph_output.shape

        """
        When: Replace conv node with block-wise grouped conv
        Then: Aggregating across num_blocks should match the original conv output from ONNX graph
        """

        num_blocks = weight_shape[block_axis] // block_size
        modified_graph_output = modified_graph_output.reshape(
            N, num_blocks, c_out, h_out, w_out
        )
        aggregated_output = modified_graph_output.sum(1)
        assert np.allclose(orig_graph_output, aggregated_output, atol=1e-1)

        """
        When: Replace conv node with grouped conv
        Then: Graph output should be (N, c_out, num_blocks, h_out, w_out) and match torch output
        """

        def _onnx_blockwise_conv2d(session):
            onnx_outputs = session.run(None, {"input": dummy_input})[0]
            return onnx_outputs

        onnx_outputs = _onnx_blockwise_conv2d(session)

        # Using torch operations
        torch_input = torch.from_numpy(dummy_input).to(device=device)
        torch_weight = torch.from_numpy(weight).to(device=device)
        torch_outputs = (
            _torch_blockwise_conv2d(
                torch_input, torch_weight, block_size=block_size, **kwargs
            )
            .cpu()
            .numpy()
        )

        assert onnx_outputs.shape == torch_outputs.shape
        assert np.allclose(onnx_outputs, torch_outputs, atol=1e-1)

        # Performance test, this is not a fair comparison for PyTorch and ONNX
        # but a practical sanity check to ensure ONNX inference isn't significantly slower than PyTorch
        torch_avg_time = _timed_run(
            lambda: _torch_blockwise_conv2d(
                torch_input, torch_weight, block_size=block_size, **kwargs
            )
        )
        onnx_avg_time = _timed_run(lambda: _onnx_blockwise_conv2d(session))
        print(f"torch_avg_time: {torch_avg_time} onnx_avg_time: {onnx_avg_time}")

        del session

    @pytest.mark.cuda
    @pytest.mark.parametrize(
        "op_type, use_cuda, weight_shape, input_shape, output_shape, block_size, block_axis, kwargs",
        [
            (
                "Gemm",
                True,
                (15, 30),
                (2, 15),
                (5, 2, 30),
                3,
                0,
                dict(transA=0, transB=0),
            ),
            (
                "Gemm",
                False,
                (15, 30),
                (2, 15),
                (5, 2, 30),
                3,
                0,
                dict(transA=0, transB=0),
            ),
            (
                "Gemm",
                True,
                (150, 300),
                (1, 150),
                (50, 1, 300),
                3,
                0,
                dict(transA=0, transB=0),
            ),
            (
                "Gemm",
                False,
                (150, 300),
                (1, 150),
                (50, 1, 300),
                3,
                0,
                dict(transA=0, transB=0),
            ),
            (
                "Gemm",
                True,
                (30, 15),
                (2, 15),
                (5, 2, 30),
                3,
                1,
                dict(transA=0, transB=1),
            ),  # weight shape: (c_out, c_in)
            (
                "Gemm",
                False,
                (30, 15),
                (2, 15),
                (5, 2, 30),
                3,
                1,
                dict(transA=0, transB=1),
            ),  # weight shape: (c_out, c_in)
            (
                "Gemm",
                True,
                (300, 150),
                (1, 150),
                (50, 1, 300),
                3,
                1,
                dict(transA=0, transB=1),
            ),  # weight shape: (c_out, c_in)
            (
                "Gemm",
                False,
                (300, 150),
                (1, 150),
                (50, 1, 300),
                3,
                1,
                dict(transA=0, transB=1),
            ),  # weight shape: (c_out, c_in)
            (
                "MatMul",
                True,
                (15, 30),
                (20, 15),
                (5, 20, 30),
                3,
                0,
                dict(),
            ),  # Matmul 3d input
            (
                "MatMul",
                False,
                (15, 30),
                (20, 15),
                (5, 20, 30),
                3,
                0,
                dict(),
            ),  # Matmul 3d input
            (
                "MatMul",
                True,
                (150, 300),
                (10, 150),
                (50, 10, 300),
                3,
                0,
                dict(),
            ),  # Matmul 3d input
            (
                "MatMul",
                False,
                (150, 300),
                (10, 150),
                (50, 10, 300),
                3,
                0,
                dict(),
            ),  # Matmul 3d input
        ],
    )
    def test_blockwise_linear(
        self,
        op_type,
        use_cuda,
        weight_shape,
        input_shape,
        output_shape,
        block_size,
        block_axis,
        kwargs,
    ):
        """
        by default, the Gemm operator does not transpose the second input B
        so passing untransposed weight tensor (c_in, c_out)
        """
        providers = ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        device = torch.device("cuda:0" if use_cuda else "cpu")

        weight = np.random.randn(*weight_shape).astype(np.float32)
        if op_type == "Gemm":
            onnx_model = _onnx_model_from_op(
                op_type, weight, input_shape, output_shape, **kwargs
            )
        elif op_type == "MatMul":
            onnx_model = _onnx_model_from_op(op_type, weight, input_shape, output_shape)
        else:
            raise NotImplementedError

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        session = ort.InferenceSession(
            onnx_model.SerializeToString(), providers=providers
        )
        orig_graph_output = session.run(None, {"input": dummy_input})[0]

        # In-place modify Gemm/MatMul with block-wise batched MatMul
        modify_graph_with_grouped_linear(onnx_model, block_size, block_axis)
        onnx.checker.check_model(onnx_model)

        session = ort.InferenceSession(
            onnx_model.SerializeToString(), providers=providers
        )

        """
        When: Replace Gemm/MatMul node with block-wise matmul
        Then: Aggregating across num_blocks should match the original output from ONNX graph
        """
        prepared_inputs = prepare_linear_inputs({"input": [dummy_input]}, block_size)
        prepared_inputs = {k: v[0] for k, v in prepared_inputs.items()}  # Flatten lists

        onnx_outputs = session.run(None, prepared_inputs)[0]
        aggregated_output = onnx_outputs.sum(0)
        assert np.allclose(orig_graph_output, aggregated_output, atol=1e-1)

        """
        When: Replace Gemm/MatMul nodes with block-wise batched MatMul
        Then: Graph output should be (N, c_out, num_blocks) and should match torch output
        """

        trans_b = kwargs.get("transB", 0)
        weight = (
            weight if trans_b == 1 else weight.transpose(1, 0)
        )  # torch transpose it internally x @ W.T
        torch_input = torch.from_numpy(prepared_inputs["input"]).to(device=device)
        torch_weight = torch.from_numpy(weight).to(device=device)
        torch_outputs = (
            _torch_blockwise_linear(torch_input, torch_weight, block_size=block_size)
            .cpu()
            .numpy()
        )
        assert onnx_outputs.shape == torch_outputs.shape
        assert onnx_outputs.shape[0] == weight_shape[block_axis] // block_size
        assert np.allclose(onnx_outputs, torch_outputs, atol=1e-1)

        del session

    @pytest.mark.parametrize(
        "op_type, weight_shape, input_shape, output_shape, block_size, block_axis, num_sha_ops, kwargs",
        [
            ("Conv", (32, 16, 5, 5), (2, 16, 10, 10), (2, 32, 6, 6), 4, 1, 32, dict()),
            (
                "Gemm",
                (15, 30),
                (5, 2, 3),
                (5, 2, 30),
                3,
                0,
                32,
                dict(transA=0, transB=0),
            ),
            (
                "Gemm",
                (30, 15),
                (5, 2, 3),
                (5, 2, 30),
                3,
                1,
                32,
                dict(transA=0, transB=1),
            ),
            ("MatMul", (15, 30), (5, 20, 3), (5, 20, 30), 3, 0, 32, dict()),
        ],
    )
    def test_parallel_sha_ops(
        self,
        op_type,
        weight_shape,
        input_shape,
        output_shape,
        block_size,
        block_axis,
        num_sha_ops,
        kwargs,
    ):
        providers = ["CPUExecutionProvider"]
        weight = np.random.randn(*weight_shape).astype(np.float32)
        onnx_model = _onnx_model_from_op(
            op_type, weight, input_shape, output_shape, num_sha_ops, **kwargs
        )

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        if op_type == "Conv":
            modify_graph_with_grouped_conv(onnx_model, block_size, block_axis)
        elif op_type in ["Gemm", "MatMul"]:
            modify_graph_with_grouped_linear(onnx_model, block_size, block_axis)
        else:
            raise NotImplementedError

        """
        When: Replace Gemm/MatMul nodes with block-wise batched MatMul or Conv with grouped Conv
        Then: Graph output should be (N, c_out, num_blocks) or (N, c_out, num_blocks, h_out, w_out)
        """
        onnx.checker.check_model(onnx_model)

        session = ort.InferenceSession(
            onnx_model.SerializeToString(), providers=providers
        )

        onnx_outputs = session.run(None, {"input": dummy_input})

        """
        When: multiple SHA ops of same op_type at the same level (Share same inputs, has same replicated weights)
        Then: Outputs should match
        """
        assert len(onnx_outputs) == num_sha_ops

        for i, output in enumerate(onnx_outputs[1:], start=1):
            assert output.shape == onnx_outputs[0].shape

        for i, output in enumerate(onnx_outputs[1:], start=1):
            assert np.allclose(onnx_outputs[0], output)

        del session


_SEQ_LEN, _HIDDEN, _FFN = 4, 8, 16


class _GraphBuilder:
    """Builds small ONNX graphs in which every node output is named after its node."""

    def __init__(self):
        self.nodes = []
        self.initializers = []

    def op(self, op_type, inputs, name, **attrs):
        self.nodes.append(
            onnx.helper.make_node(op_type, inputs, [name], name=name, **attrs)
        )
        return name

    def const(self, name, array):
        self.initializers.append(
            onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)
        )
        return name

    def linear(self, x, name, shape):
        weight = self.const(f"{name}.weight", np.random.randn(*shape) * 0.1)
        return self.op("MatMul", [x, weight], name)

    def model(self, inputs, output):
        graph = onnx.helper.make_graph(
            self.nodes,
            "test_graph",
            inputs,
            [
                onnx.helper.make_tensor_value_info(
                    output, onnx.TensorProto.FLOAT, [1, _SEQ_LEN, _HIDDEN]
                )
            ],
            initializer=self.initializers,
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_operatorsetid("", 20)],
            ir_version=10,
        )
        onnx.checker.check_model(model)
        return model


def _decoder_model(num_blocks: int) -> onnx.ModelProto:
    """
    Pre-norm decoder stack with RoPE, masked attention, SwiGLU MLP and two residual Adds per block.
    RoPE cos/sin and the attention mask are computed in the graph from model inputs.
    """
    b = _GraphBuilder()
    position = b.op(
        "Mul", ["position", b.const("inv_freq", np.linspace(0.1, 1.0, _HIDDEN))], "pos"
    )
    cos = b.op("Cos", [position], "rope_cos")
    sin = b.op("Sin", [position], "rope_sin")
    mask = b.op("Mul", ["attention_mask", b.const("mask_scale", [-1e4])], "attn_mask")
    h = b.linear("hidden", "in_proj", (_HIDDEN, _HIDDEN))

    for i in range(num_blocks):
        p = f"blk{i}"
        norm1 = b.op("Relu", [h], f"{p}.norm1")
        q = b.linear(norm1, f"{p}.q_proj", (_HIDDEN, _HIDDEN))
        k = b.linear(norm1, f"{p}.k_proj", (_HIDDEN, _HIDDEN))
        v = b.linear(norm1, f"{p}.v_proj", (_HIDDEN, _HIDDEN))

        def rope(t, tag):
            t_cos = b.op("Mul", [t, cos], f"{p}.{tag}_rope_cos_mul")
            t_neg = b.op("Neg", [t], f"{p}.{tag}_rope_neg")
            t_sin = b.op("Mul", [t_neg, sin], f"{p}.{tag}_rope_sin_mul")
            return b.op("Add", [t_cos, t_sin], f"{p}.{tag}_rope_add")

        q_rot = rope(q, "q")
        k_rot = rope(k, "k")
        k_t = b.op("Transpose", [k_rot], f"{p}.k_transpose", perm=[0, 2, 1])
        scores = b.op("MatMul", [q_rot, k_t], f"{p}.attn_qk")
        masked = b.op("Add", [scores, mask], f"{p}.attn_mask_add")
        probs = b.op("Softmax", [masked], f"{p}.attn_softmax", axis=-1)
        attn = b.op("MatMul", [probs, v], f"{p}.attn_pv")
        o = b.linear(attn, f"{p}.o_proj", (_HIDDEN, _HIDDEN))
        h1 = b.op("Add", [h, o], f"{p}.residual_add1")

        norm2 = b.op("Relu", [h1], f"{p}.norm2")
        gate = b.linear(norm2, f"{p}.gate_proj", (_HIDDEN, _FFN))
        up = b.linear(norm2, f"{p}.up_proj", (_HIDDEN, _FFN))
        sig = b.op("Sigmoid", [gate], f"{p}.silu_sigmoid")
        silu = b.op("Mul", [gate, sig], f"{p}.silu_mul")
        act = b.op("Mul", [silu, up], f"{p}.swiglu_mul")
        down = b.linear(act, f"{p}.down_proj", (_FFN, _HIDDEN))
        h = b.op("Add", [h1, down], f"{p}.residual_add2")

    head = b.linear(h, "lm_head", (_HIDDEN, _HIDDEN))
    inputs = [
        onnx.helper.make_tensor_value_info(
            "hidden", onnx.TensorProto.FLOAT, [1, _SEQ_LEN, _HIDDEN]
        ),
        onnx.helper.make_tensor_value_info(
            "position", onnx.TensorProto.FLOAT, [1, _SEQ_LEN, 1]
        ),
        onnx.helper.make_tensor_value_info(
            "attention_mask", onnx.TensorProto.FLOAT, [1, _SEQ_LEN, _SEQ_LEN]
        ),
    ]
    return b.model(inputs, head)


def _decoder_inputs():
    return {
        "hidden": np.random.randn(1, _SEQ_LEN, _HIDDEN).astype(np.float32),
        "position": np.arange(_SEQ_LEN, dtype=np.float32).reshape(1, _SEQ_LEN, 1),
        "attention_mask": np.triu(
            np.ones((1, _SEQ_LEN, _SEQ_LEN), dtype=np.float32), k=1
        ),
    }


def _single_op_residual_model(num_blocks: int) -> onnx.ModelProto:
    """x_{i+1} = Add(Relu(x_i), MatMul(Sigmoid(Relu(x_i)))): the skip leaves a non-weight op."""
    b = _GraphBuilder()
    h = b.linear("hidden", "in_proj", (_HIDDEN, _HIDDEN))
    for i in range(num_blocks):
        r = b.op("Relu", [h], f"blk{i}.relu")
        n = b.op("Sigmoid", [r], f"blk{i}.norm")
        m = b.linear(n, f"blk{i}.proj", (_HIDDEN, _HIDDEN))
        h = b.op("Add", [r, m], f"blk{i}.residual_add")
    head = b.linear(h, "lm_head", (_HIDDEN, _HIDDEN))
    inputs = [
        onnx.helper.make_tensor_value_info(
            "hidden", onnx.TensorProto.FLOAT, [1, _SEQ_LEN, _HIDDEN]
        )
    ]
    return b.model(inputs, head)


def _diamond_residual_model(num_blocks: int) -> onnx.ModelProto:
    """A residual block whose branch forks and re-converges on an interior op.

    blk{i}.fork is reachable from blk{i}.join both directly (through act) and through the
    weight op blk{i}.proj, so _has_path_through_weight_op must expand it once per path.
    Keyed on op.name alone, the weight-bearing path is pruned and the residual Add is missed.
    """
    b = _GraphBuilder()
    h = b.linear("hidden", "in_proj", (_HIDDEN, _HIDDEN))
    for i in range(num_blocks):
        skip = b.op("Relu", [h], f"blk{i}.skip")
        fork = b.op("Sigmoid", [skip], f"blk{i}.fork")
        weighted = b.linear(fork, f"blk{i}.proj", (_HIDDEN, _HIDDEN))
        plain = b.op("Relu", [fork], f"blk{i}.act")
        joined = b.op("Mul", [weighted, plain], f"blk{i}.join")
        h = b.op("Add", [skip, joined], f"blk{i}.residual_add")
    head = b.linear(h, "lm_head", (_HIDDEN, _HIDDEN))
    inputs = [
        onnx.helper.make_tensor_value_info(
            "hidden", onnx.TensorProto.FLOAT, [1, _SEQ_LEN, _HIDDEN]
        )
    ]
    return b.model(inputs, head)


def _decoder_levels(num_blocks):
    """Expected levels of ``_decoder_model``, each with the frontier tensors it caches."""
    levels = [(["in_proj"], [])]
    residual = "in_proj"
    for i in range(num_blocks):
        attn = [f"blk{i}.q_proj", f"blk{i}.k_proj", f"blk{i}.v_proj"]
        levels += [(attn, [residual]), ([f"blk{i}.o_proj"], [residual])]
        residual = f"blk{i}.residual_add1"
        mlp = [f"blk{i}.gate_proj", f"blk{i}.up_proj"]
        levels += [(mlp, [residual]), ([f"blk{i}.down_proj"], [residual])]
        residual = f"blk{i}.residual_add2"
    return levels + [(["lm_head"], [])]


def _residual_levels(num_blocks, branch_tensors):
    """Expected levels of the single-op and diamond residual models."""
    levels = [(["in_proj"], [])]
    for i in range(num_blocks):
        levels.append(([f"blk{i}.proj"], [f"blk{i}.{name}" for name in branch_tensors]))
    return levels + [(["lm_head"], [])]


def _seq_mse_for(model, inputs, providers=("CPUExecutionProvider",)):
    sim = QuantizationSimModel(
        copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        providers=list(providers),
    )
    return SequentialMse(model, sim, SeqMseParams(num_batches=1), [inputs])


def _levels_and_frontier(seq_mse):
    """Per level, its layer names and the sampled tensors that are not layer inputs."""
    plan = seq_mse._frontier_plan()
    result = []
    for level, names in zip(plan.levels, plan.block_inputs):
        layer_inputs = {name for layer in level for name in layer.op_input_names}
        result.append(
            (
                [layer.node.name for layer in level],
                [name for name in names if name not in layer_inputs],
            )
        )
    return result


def _output_names(seq_mse, node_names):
    names = {
        out.name
        for node in seq_mse._ir_model.graph
        if node.name in node_names
        for out in node.outputs
    }
    assert names, "the ops are not in the graph; the test would be vacuous"
    return names


def _sampled_names(seq_mse):
    return {name for names in seq_mse._frontier_plan().block_inputs for name in names}


class TestFrontierDecoder:
    """The frontier of a decoder is its residual stream."""

    @pytest.mark.parametrize("num_blocks", [1, 3])
    def test_frontier_is_residual_stream(self, num_blocks):
        """
        When: A decoder with RoPE, masked attention and a SwiGLU MLP is planned
        Then: Each level caches only the residual stream it and later levels add onto
        """
        seq_mse = _seq_mse_for(_decoder_model(num_blocks), _decoder_inputs())
        assert _levels_and_frontier(seq_mse) == _decoder_levels(num_blocks)

    @pytest.mark.parametrize(
        "op_suffix",
        [
            "attn_qk",  # activation x activation MatMul: Q.K^T
            "attn_mask_add",  # Add of attention scores and a mask computed from a model input
            "q_rope_cos_mul",
            "q_rope_sin_mul",
            "q_rope_add",
            "k_rope_cos_mul",
            "k_rope_add",
        ],
    )
    def test_level_internals_are_not_sampled(self, op_suffix):
        """
        When: Attention, RoPE and SwiGLU ops join tensors inside a level
        Then: Their outputs are recomputed with the level, so (heads, seq, seq) tensors are
              never cached
        """
        seq_mse = _seq_mse_for(_decoder_model(2), _decoder_inputs())
        outputs = _output_names(seq_mse, {f"blk{i}.{op_suffix}" for i in range(2)})
        assert not outputs & _sampled_names(seq_mse)

    def test_residual_add_with_single_weight_branch(self):
        """
        When: The skip input of a residual Add comes from a non-weight op and the branch
              contains one weight op
        Then: Each level caches only the skip input
        """
        seq_mse = _seq_mse_for(_single_op_residual_model(6), _hidden_inputs())
        assert _levels_and_frontier(seq_mse) == _residual_levels(6, ["relu"])

    @pytest.mark.parametrize("num_blocks", [1, 3])
    def test_residual_add_with_forking_branch(self, num_blocks):
        """
        When: The branch of a residual Add forks and re-converges on an op inside the branch
        Then: Each level caches the skip input and the fork's other side
        """
        seq_mse = _seq_mse_for(_diamond_residual_model(num_blocks), _hidden_inputs())
        assert _levels_and_frontier(seq_mse) == _residual_levels(
            num_blocks, ["skip", "act"]
        )

    def test_sampled_inputs_match_one_shot(self):
        """
        When: The decoder's level inputs are sampled blockwise from the cached frontier
        Then: They match one run of the whole model
        """
        model = _decoder_model(2)
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            quant_scheme=QuantScheme.post_training_tf,
            providers=["CPUExecutionProvider"],
        )
        for quantizer in sim.qc_quantize_op_dict.values():
            quantizer.enabled = False
        seq_mse = SequentialMse(
            model, sim, SeqMseParams(num_batches=1), [_decoder_inputs()]
        )
        _assert_sampled_match_one_shot(seq_mse)


# Tests on real exported Hugging Face decoders. Their graphs contain many more Adds (RoPE,
# tanh-GELU, MoE combines) and, in bfloat16, Casts around the residual stream.
# Residual Adds per decoder layer, from each architecture's decoder-layer forward. Written
# out by hand so the tests do not reuse the predicate under test to compute their oracle.
_QWEN3 = ("Qwen3Config", (), 2)
# Gemma4 also adds the per-layer-input gate back onto the residual stream.
_GEMMA4 = ("Gemma4TextConfig", (), 3)
# The MoE block's combine Add joins two sibling branches, so it is not a residual Add.
_GEMMA4_MOE = (
    "Gemma4TextConfig",
    (("enable_moe_block", True), ("top_k_experts", 2)),
    3,
)
# Hybrid linear-attention decoder; only exports under dynamo without a model transform.
_QWEN3_5 = ("Qwen3_5TextConfig", (), 2)

_LAYER_PATTERNS = {
    "Gemma4TextConfig": ["sliding_attention"] * 4 + ["full_attention"],
    "Qwen3_5TextConfig": ["linear_attention"] * 3 + ["full_attention"],
}


@functools.cache
def _export_tiny_decoder(case, backend, dtype, num_layers):
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    """Export a tiny random decoder; returns the ONNX model and one batch of inputs."""
    config_attr, extra, _ = case
    cfg = make_tiny_decoder_config(
        config_attr, num_hidden_layers=num_layers, **dict(extra)
    )
    if config_attr in _LAYER_PATTERNS:
        pattern = _LAYER_PATTERNS[config_attr]
        # Keep one full-attention layer at every depth so fan-in is comparable across depths.
        cfg.layer_types = [pattern[0]] * (num_layers - 1) + [pattern[-1]]
    if config_attr == "Gemma4TextConfig":
        cfg.hidden_size_per_layer_input = 16
        cfg.vocab_size_per_layer_input = cfg.vocab_size
        cfg.global_head_dim = cfg.head_dim
        cfg.moe_intermediate_size = 64

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, experts_implementation="eager")
    model = PrefillWrapper(model.eval()).to(dtype).eval()
    inputs = sample_prefill_inputs(cfg)
    onnx_model = export_onnx(model, inputs, backend)

    names = ["input_ids", "position_ids", "attention_mask"]
    feed = {name: tensor.numpy() for name, tensor in zip(names, inputs)}
    graph_inputs = {inp.name for inp in onnx_model.graph.input}
    return onnx_model, {k: v for k, v in feed.items() if k in graph_inputs}


# torchscript keeps the module scope in op names, so a residual Add is a direct child of the
# decoder layer, not of self_attn / mlp / act_fn / experts.
_LAYER_SCOPE_ADD = re.compile(r"/layers\.(\d+)/Add(_\d+)?$")


@pytest.mark.skip_on_windows_amd64(
    "insufficient disk for HF model export on Windows AMD64 runner"
)
class TestResidualAddBoundaryHF:
    """SeqMSE dependency boundaries on exported Hugging Face decoders."""

    @pytest.mark.parametrize(
        "case, backend, dtype",
        [
            pytest.param(_QWEN3, "torchscript", torch.float32, id="qwen3-ts-fp32"),
            pytest.param(_GEMMA4, "torchscript", torch.float32, id="gemma4-ts-fp32"),
            pytest.param(
                _GEMMA4_MOE, "torchscript", torch.float32, id="gemma4_moe-ts-fp32"
            ),
            pytest.param(_GEMMA4, "dynamo", torch.float32, id="gemma4-dynamo-fp32"),
            pytest.param(_QWEN3_5, "dynamo", torch.float32, id="qwen3_5-dynamo-fp32"),
            # torchscript cannot export bfloat16 RoPE (ComplexDouble), so bf16 uses dynamo,
            # and the CPU EP lacks bfloat16 kernels, so bf16 runs on CUDA.
            pytest.param(
                _QWEN3,
                "dynamo",
                torch.bfloat16,
                id="qwen3-dynamo-bf16",
                marks=pytest.mark.cuda,
            ),
            pytest.param(
                _GEMMA4,
                "dynamo",
                torch.bfloat16,
                id="gemma4-dynamo-bf16",
                marks=pytest.mark.cuda,
            ),
        ],
    )
    def test_every_layer_is_planned_and_sampled_tensors_are_cacheable(
        self, case, backend, dtype
    ):
        """
        When: A tiny HF decoder is exported and planned
        Then: Every layer with a static weight is planned, and every tensor a level samples is
              computed from earlier levels only, and is not a quantized copy
        """
        providers = ["CPUExecutionProvider"]
        if dtype == torch.bfloat16:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                pytest.skip("CUDA EP not available")
            providers.insert(0, "CUDAExecutionProvider")
        onnx_model, inputs = _export_tiny_decoder(case, backend, dtype, 2)
        seq_mse = _seq_mse_for(copy.deepcopy(onnx_model), inputs, providers)
        plan = seq_mse._frontier_plan()
        sampling_plan = SamplingPlan(
            seq_mse._ir_model.graph, plan.block_inputs, plan.block_outputs
        )

        planned = {layer.node.name for level in plan.levels for layer in level}
        expected = {
            node.name
            for node in seq_mse.sim._ir_model.graph.all_nodes()
            if is_supported_op(node)
        }
        assert planned == expected
        for idx, names in enumerate(plan.block_inputs):
            for name in names:
                assert sampling_plan.is_final(name, idx), (idx, name)
                assert not name.endswith(("_updated", "_qdq")), (idx, name)

    @pytest.mark.parametrize(
        "case, backend",
        [
            pytest.param(_QWEN3, "torchscript", id="qwen3-ts"),
            pytest.param(_QWEN3_5, "dynamo", id="qwen3_5-dynamo"),
        ],
    )
    def test_frontier_independent_of_depth(self, case, backend):
        """
        When: The same decoder is exported with 3 and 6 layers
        Then: No level samples more tensors, so cached activations do not accumulate along
              the residual stream

        Gemma4 is left out: it computes every layer's per-layer input up front, so each one
        is cached until its layer reads it. 3 layers give Qwen3.5 a linear-attention layer
        after the first, which also reads the residual stream.
        """

        def max_sampled(num_layers):
            onnx_model, inputs = _export_tiny_decoder(
                case, backend, torch.float32, num_layers
            )
            plan = _seq_mse_for(copy.deepcopy(onnx_model), inputs)._frontier_plan()
            return max(len(names) for names in plan.block_inputs)

        assert max_sampled(6) == max_sampled(3)

    def test_non_residual_adds_are_not_sampled(self):
        """
        When: A Gemma4 MoE decoder has RoPE, tanh-GELU, expert and MoE-combine Adds
        Then: None of their outputs are sampled
        """
        onnx_model, inputs = _export_tiny_decoder(
            _GEMMA4_MOE, "torchscript", torch.float32, 2
        )
        seq_mse = _seq_mse_for(copy.deepcopy(onnx_model), inputs)
        add_names = [
            op.name
            for op in seq_mse.sim.connected_graph.ordered_ops
            if op.type == "Add"
        ]
        non_residual = {
            # MoE combine: sum of the dense-MLP and expert branches, at decoder-layer scope
            "moe_combine": [
                n for n in add_names if re.search(r"/layers\.\d+/Add_1$", n)
            ],
            "rope": [n for n in add_names if "/self_attn/" in n],
            "gelu": [n for n in add_names if "/act_fn/" in n],
            "experts": [n for n in add_names if "/experts/" in n],
        }
        sampled = _sampled_names(seq_mse)
        for kind, names in non_residual.items():
            assert names, f"no {kind} Adds in the export; the test would be vacuous"
            assert not _output_names(seq_mse, set(names)) & sampled, kind

    @pytest.mark.parametrize(
        "backend, dtype, providers",
        [
            pytest.param(
                "torchscript",
                torch.float32,
                ["CPUExecutionProvider"],
                id="ts-fp32-cpu",
            ),
            pytest.param(
                "dynamo",
                torch.bfloat16,
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                id="dynamo-bf16-cuda",
                marks=pytest.mark.cuda,
            ),
        ],
    )
    def test_sampled_inputs_match_one_shot(self, backend, dtype, providers):
        """
        When: A Gemma4 decoder's level inputs are sampled blockwise, as SeqMSE does
        Then: They match one run of the whole model
        """
        if "CUDAExecutionProvider" in providers and (
            "CUDAExecutionProvider" not in ort.get_available_providers()
        ):
            pytest.skip("CUDA EP not available")

        onnx_model, inputs = _export_tiny_decoder(_GEMMA4, backend, dtype, 2)
        sim = QuantizationSimModel(
            copy.deepcopy(onnx_model),
            quant_scheme=QuantScheme.post_training_tf,
            providers=providers,
        )
        for quantizer in sim.qc_quantize_op_dict.values():
            quantizer.enabled = False
        seq_mse = SequentialMse(
            copy.deepcopy(onnx_model), sim, SeqMseParams(num_batches=1), [inputs]
        )
        _assert_sampled_match_one_shot(
            seq_mse, atol=1e-5 if dtype == torch.float32 else 1e-2
        )

    @pytest.mark.cuda
    def test_seq_mse_runs_on_bf16_decoder(self):
        """
        When: SeqMSE runs on a bfloat16 Gemma4 decoder with quantizers enabled
        Then: Every weight quantizer is optimized and frozen, and the
              sim can calibrate activations afterwards
        """
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CUDA EP not available")

        onnx_model, inputs = _export_tiny_decoder(_GEMMA4, "dynamo", torch.bfloat16, 2)
        sim = QuantizationSimModel(
            copy.deepcopy(onnx_model),
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=16,
            default_param_bw=4,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        apply_seq_mse(sim, [inputs], num_candidates=4)

        weight_names = [
            name
            for name, quantizer in sim.qc_quantize_op_dict.items()
            if quantizer.is_initialized()
        ]
        assert weight_names
        for name in weight_names:
            assert sim.qc_quantize_op_dict[name].is_encoding_frozen(), name

        sim.compute_encodings([inputs])


def _dynamo_two_linear_model() -> onnx.ModelProto:
    """Two Linear layers exported with dynamo, whose node names (``node_linear``) differ from
    the module paths (``fc1``, ``fc2``) recorded in their ``namespace`` metadata."""

    class TwoLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(8, 8)
            self.fc2 = torch.nn.Linear(8, 4)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    program = torch.onnx.export(
        TwoLinear().eval(), (torch.randn(1, 8),), dynamo=True, opset_version=18
    )
    return program.model_proto


class TestSeqMseProgress:
    @staticmethod
    def _run(model, caplog, **kwargs):
        sim = QuantizationSimModel(
            model=copy.deepcopy(model),
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=8,
            default_param_bw=4,
            providers=["CPUExecutionProvider"],
        )
        inputs = [make_dummy_input(model) for _ in range(2)]
        with caplog.at_level(logging.DEBUG, logger="SeqMse"):
            apply_seq_mse(sim, inputs, num_candidates=4, **kwargs)
        return [r for r in caplog.records if r.name == "SeqMse"]

    def test_group_lines_label_results_and_show_graph_names(self, caplog):
        """
        Given: A dynamo-exported model, whose node names are not module paths
        When: Sequential MSE runs
        Then: Each group line names the weight by module path and graph node name, labels
              its range and loss, and leaves the phase breakdown to debug
        """
        records = self._run(_dynamo_two_linear_model(), caplog)
        groups = [r for r in records if r.getMessage().startswith("SeqMSE group ")]
        info_groups = [r.getMessage() for r in groups if r.levelno == logging.INFO]
        assert len(info_groups) == 2
        assert all(
            re.search(r"fc[12] \[node_\w+\] range \d\.\d\dx", m) for m in info_groups
        )
        assert all("search" not in m for m in info_groups)
        debug_phases = [r.getMessage() for r in groups if r.levelno == logging.DEBUG]
        assert len(debug_phases) == 2 and all("search" in m for m in debug_phases)

    def test_group_lines_keep_torchscript_names(self, caplog):
        """
        Given: A TorchScript-exported model, whose node names are module paths
        When: Sequential MSE runs
        Then: Each group line names the weight by its node name in the graph, unchanged
        """

        class TwoLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(8, 8)
                self.fc2 = torch.nn.Linear(8, 4)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        buffer = io.BytesIO()
        torch.onnx.export(
            TwoLinear().eval(),
            (torch.randn(1, 8),),
            buffer,
            dynamo=False,
            opset_version=18,
        )
        model = onnx.load_from_string(buffer.getvalue())
        weights = [n.name for n in model.graph.node if n.op_type in ("MatMul", "Gemm")]
        assert all(name.startswith("/fc") for name in weights)
        records = self._run(model, caplog)
        info_groups = [
            r.getMessage()
            for r in records
            if r.getMessage().startswith("SeqMSE group ") and r.levelno == logging.INFO
        ]
        assert len(info_groups) == 2
        for name, message in zip(weights, info_groups):
            assert f": {name} range " in message

    def test_excluded_count_is_weights_only(self, caplog):
        """
        Given: One weight excluded by name
        When: Sequential MSE runs
        Then: The header counts one excluded weight and the run optimizes the other
        """
        model = _dynamo_two_linear_model()
        # The last weight, so the first still reads the graph input it is calibrated on
        last = [n.name for n in model.graph.node if n.op_type in ("MatMul", "Gemm")][-1]
        records = self._run(model, caplog, nodes_to_exclude=[last])
        header = next(
            r.getMessage() for r in records if "SeqMSE: optimizing" in r.getMessage()
        )
        assert "optimizing 1 weights" in header and "| 1 weights excluded" in header

    def test_summary_is_logged_when_a_group_fails(self, caplog, monkeypatch):
        """
        Given: A group that raises during optimization
        When: Sequential MSE runs
        Then: The error propagates and the end-of-run summary is still logged
        """

        def _fail(self, *args):
            raise RuntimeError("boom")

        monkeypatch.setattr(SequentialMse, "_run_seq_mse", _fail)
        with pytest.raises(RuntimeError, match="boom"):
            self._run(_dynamo_two_linear_model(), caplog)
        assert any(
            "SeqMSE: done, optimized 0 weights" in r.getMessage()
            for r in caplog.records
        )


def _hidden_inputs():
    return {"hidden": np.random.randn(1, _SEQ_LEN, _HIDDEN).astype(np.float32)}


_FRONTIER_CASES = {
    "decoder1": (lambda: _decoder_model(1), _decoder_inputs),
    "decoder3": (lambda: _decoder_model(3), _decoder_inputs),
    "single_op_residual": (lambda: _single_op_residual_model(3), _hidden_inputs),
    "diamond": (lambda: _diamond_residual_model(2), _hidden_inputs),
    "split": (model_with_split, None),
    "residual": (single_residual_model, None),
}


def _frontier_sim_and_inputs(case, num_samples=2):
    model_factory, inputs_factory = _FRONTIER_CASES[case]
    np.random.seed(0)
    torch.manual_seed(0)
    model = model_factory()
    sim = QuantizationSimModel(
        copy.deepcopy(model),
        quant_scheme=QuantScheme.post_training_tf,
        default_param_bw=4,
        providers=["CPUExecutionProvider"],
    )
    onnx_model = model.model if hasattr(model, "model") else model
    inputs = [
        inputs_factory() if inputs_factory else make_dummy_input(onnx_model)
        for _ in range(num_samples)
    ]
    return model, sim, inputs


def _frontier_seq_mse(model, sim, inputs, **kwargs):
    params = SeqMseParams(num_batches=len(inputs), num_candidates=4)
    return SequentialMse(model, sim, params, inputs, **kwargs)


class TestFrontierSampling:
    """Sequential MSE sampling each level's inputs with the frontier planner."""

    @pytest.mark.parametrize("case", list(_FRONTIER_CASES))
    def test_every_sampled_tensor_is_cacheable(self, case):
        """
        When: The frontier plan is given to SamplingPlan
        Then: Every tensor a level samples is computed from earlier levels only, so the
              sampler can cache it, and no level samples a quantized copy of a tensor
        """
        model, sim, inputs = _frontier_sim_and_inputs(case)
        seq_mse = _frontier_seq_mse(model, sim, inputs)
        plan = seq_mse._frontier_plan()
        sampling_plan = SamplingPlan(
            seq_mse._ir_model.graph, plan.block_inputs, plan.block_outputs
        )

        for idx, names in enumerate(plan.block_inputs):
            for name in names:
                assert sampling_plan.is_final(name, idx), (idx, name)
                assert not name.endswith(("_updated", "_qdq")), (idx, name)

    def test_frontier_is_flat_in_depth(self):
        """
        When: The decoder gets deeper
        Then: The number of tensors a level samples does not grow, and the attention mask
              and RoPE tables (no weight upstream) are recomputed rather than cached
        """

        def max_sampled(num_blocks):
            model = _decoder_model(num_blocks)
            sim = QuantizationSimModel(
                copy.deepcopy(model),
                quant_scheme=QuantScheme.post_training_tf,
                providers=["CPUExecutionProvider"],
            )
            plan = _frontier_seq_mse(model, sim, [_decoder_inputs()])._frontier_plan()
            sampled = {name for names in plan.block_inputs for name in names}
            assert not {"pos", "rope_cos", "rope_sin", "attn_mask"} & sampled
            return max(len(names) for names in plan.block_inputs)

        assert max_sampled(4) == max_sampled(1)

    @pytest.mark.parametrize("case", list(_FRONTIER_CASES))
    @pytest.mark.parametrize("exclude", [False, True])
    def test_optimizes_every_layer(self, case, exclude):
        """
        When: Sequential MSE runs, optionally with a layer excluded
        Then: Exactly the weights of the layers not excluded are frozen
        """
        model, sim, inputs = _frontier_sim_and_inputs(case)
        layers = [
            node for node in sim._ir_model.graph.all_nodes() if is_supported_op(node)
        ]
        excluded = [layers[len(layers) // 2]] if exclude else []
        seq_mse = _frontier_seq_mse(
            model, sim, inputs, nodes_to_exclude=[node.name for node in excluded]
        )
        seq_mse.apply_seq_mse_algo()

        frozen = {
            name
            for name, quantizer in sim.qc_quantize_op_dict.items()
            if quantizer.is_encoding_frozen()
        }
        assert frozen == {
            get_weight_name(node) for node in layers if node not in excluded
        }

    def test_missing_input_raises(self):
        """
        When: A calibration sample lacks a graph input
        Then: Sequential MSE raises before optimizing anything
        """
        model, sim, _ = _frontier_sim_and_inputs("decoder1")
        inputs = [{"hidden": _decoder_inputs()["hidden"]}]
        with pytest.raises(ValueError, match="must be present in the dataloader"):
            _frontier_seq_mse(model, sim, inputs)

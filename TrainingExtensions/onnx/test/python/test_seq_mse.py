# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

import time
import pytest
from unittest.mock import MagicMock
from functools import partial
import onnx
import torch
import torch.nn.functional as F
import onnxruntime as ort
import copy
import json
import numpy as np
import os
import logging

from aimet_onnx.common import libquant_info
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx import apply_seq_mse
from aimet_onnx.sequential_mse.dependency_graph import (
    DependencyGraph,
    SUPPORTED_MODULES,
)
from aimet_onnx.meta.connectedgraph import ConnectedGraph
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
from aimet_onnx.utils import make_dummy_input
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
from .utils import tmp_dir

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
    # Mock dependency node and graph
    mock_dep_node = MagicMock()
    mock_dep_graph = MagicMock()
    mock_dep_graph.get_param_name.return_value = "mock_weight"

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
    seq_mse.dependency_graph = mock_dep_graph

    _, max_val = seq_mse._get_min_and_max_for_candidate_selection(mock_dep_node)
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
    dep_node = list(seq_mse.dependency_graph._name_to_node.values())[0]
    _, init_max = seq_mse._get_min_and_max_for_candidate_selection(dep_node)

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


def test_dependency_graph(tmp_dir):
    model = model_with_split()
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
    seq_params = SeqMseParams(num_batches=1)
    inputs = [make_dummy_input(model.model) for _ in range(10)]
    seq_mse = SequentialMse(model, sim, seq_params, inputs)

    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].in_degree == 0
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].out_degree == 2
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].op_input_names == [
        "input"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].op_output_names == [
        "/conv1/Conv_output_0"
    ]

    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].out_degree == 0
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].op_input_names == [
        "/conv1/Conv_output_0"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].op_output_names == [
        "/conv2/Conv_output_0"
    ]

    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].out_degree == 0
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].op_input_names == [
        "/conv1/Conv_output_0"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].op_output_names == [
        "/conv3/Conv_output_0"
    ]


def test_residual_model_dependency_graph(tmp_dir):
    model = single_residual_model()
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
    seq_params = SeqMseParams(num_batches=1)
    inputs = [make_dummy_input(model.model) for _ in range(10)]
    seq_mse = SequentialMse(model, sim, seq_params, inputs)

    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].in_degree == 0
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].out_degree == 2
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].op_input_names == [
        "input"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv1/Conv"].op_output_names == [
        "/conv1/Conv_output_0"
    ]

    assert seq_mse.dependency_graph._name_to_node["/conv4/Conv"].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv4/Conv"].out_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv4/Conv"].op_input_names == [
        "/maxpool/MaxPool_output_0"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv4/Conv"].op_output_names == [
        "/conv4/Conv_output_0"
    ]

    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].out_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].op_input_names == [
        "/maxpool/MaxPool_output_0"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv2/Conv"].op_output_names == [
        "/conv2/Conv_output_0"
    ]

    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].in_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].out_degree == 1
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].op_input_names == [
        "/relu2/Relu_output_0"
    ]
    assert seq_mse.dependency_graph._name_to_node["/conv3/Conv"].op_output_names == [
        "/conv3/Conv_output_0"
    ]


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


def test_model_with_multiple_inputs_dependency_graph_utils(tmp_dir):
    model = model_with_multiple_inputs()
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
    seq_params = SeqMseParams(num_batches=1)
    inputs = [make_dummy_input(model) for _ in range(10)]
    seq_mse = SequentialMse(model, sim, seq_params, inputs)

    starting_ops_names = [
        op.name_op for op in seq_mse.dependency_graph.conn_graph.starting_ops
    ]

    assert starting_ops_names == ["Conv1"]
    assert seq_mse.dependency_graph._op_names_with_model_inputs == [
        "Conv1",
        "ADD_0",
        "ADD_1",
    ]


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
    Then: The dependency graph auto-excludes them from sequential MSE - their
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

    # The weightless MatMul must be present in the dependency graph (it consumes
    # a model input) but flagged as unsupported (no static weight).
    seq_mse = SequentialMse(model, sim, SeqMseParams(num_batches=1), inputs)
    dep_graph = seq_mse.dependency_graph

    """
    When: A model contains a weightless dynamic MatMul (activation x model-input.
    Then: Seq MSE optimizing (freezing) only the static-weight MatMul and skipping the weightless one,
     which has no weight param to optimize.
    """
    assert "state_mm" in dep_graph._name_to_node
    weighted_op = dep_graph.conn_graph.get_op_from_module_name("mm")
    weightless_op = dep_graph.conn_graph.get_op_from_module_name("state_mm")
    assert dep_graph.is_supported_op(weighted_op)
    assert not dep_graph.is_supported_op(weightless_op)

    seq_mse.apply_seq_mse_algo()

    assert sim.qc_quantize_op_dict["w0"].is_encoding_frozen()
    assert not sim.qc_quantize_op_dict["state"].is_encoding_frozen()


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
    mock_dep_node = MagicMock()
    mock_dep_node.cg_op.type = op_type

    mock_dep_graph = MagicMock()
    mock_dep_graph.get_param_name.return_value = "mock_weight"

    mock_quantizer = MagicMock()
    mock_quantizer.tensor_quantizer_params.tensor_shape = tensor_shape
    mock_quantizer.quant_info.channelAxis = channel_axis
    mock_quantizer.quant_info.blockAxis = block_axis
    mock_quantizer.quant_info.blockSize = block_size

    mock_sim = MagicMock()
    mock_sim.qc_quantize_op_dict = {"mock_weight": mock_quantizer}

    # Mock seq_mse object w/o calling the __init__
    seq_mse = object.__new__(SequentialMse)
    seq_mse.dependency_graph = mock_dep_graph
    seq_mse.sim = mock_sim
    seq_mse.params = MagicMock()
    seq_mse.params.loss_fn = loss_fn

    # Random outputs
    sim_output = torch.from_numpy(np.random.randn(*output_shape).astype(np.float32))
    float_output = torch.from_numpy(np.random.randn(*output_shape).astype(np.float32))

    loss = seq_mse._compute_recon_loss(sim_output, float_output, mock_dep_node)
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
    # Setup mocks
    mock_dep_graph = MagicMock()
    mock_dep_graph.get_param_name.return_value = "mock_weight"

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
    seq_mse.dependency_graph = mock_dep_graph
    seq_mse.sim = mock_sim

    mock_dep_node = MagicMock()
    seq_mse._compute_encoding_from_candidate(mock_dep_node, x_min, x_max)
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

    dep_node = list(seq_mse.dependency_graph._name_to_node.values())[0]
    _, init_max = seq_mse._get_min_and_max_for_candidate_selection(dep_node)

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


class TestDependencyGraph:
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
    def test_dependency_graph(self, model, cached_data):
        """Compare the one-shot and iterative outputs"""
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            quant_scheme=QuantScheme.post_training_tf,
            providers=["CPUExecutionProvider"],
        )
        for quantizer in sim.qc_quantize_op_dict.values():
            quantizer.enabled = False
        seq_mse = SequentialMse(model, sim, SeqMseParams(num_batches=1), [cached_data])
        dep_graph = seq_mse.dependency_graph

        one_shot_output = sim.session.run(None, cached_data)[0]

        # Iterate over the topologically sorted nodes to gather intermediate outputs and
        # provide them as inputs to the subsequent subgraph.
        sorted_nodes = dep_graph.get_topologically_sorted_nodes()
        for i in range(1, len(sorted_nodes)):
            subgraph_inp_names, subgraph_out_names = (
                dep_graph.get_subgraph_inp_out_names(sorted_nodes[i])
            )
            path = seq_mse._split_onnx_graph(subgraph_inp_names, subgraph_out_names)
            with seq_mse._create_session(path) as session:
                outputs = session.run(
                    None, {name: cached_data[name] for name in subgraph_inp_names}
                )
            cached_data.update(zip(subgraph_out_names, outputs))

        # Final subgraph extraction and session run
        model_outputs = [output.name for output in sim.model.model.graph.output]
        path = seq_mse._split_onnx_graph(subgraph_out_names, model_outputs)
        with seq_mse._create_session(path) as session:
            iterative_output = session.run(
                None, {name: cached_data[name] for name in subgraph_out_names}
            )[0]

        assert np.all(iterative_output == one_shot_output)

    def test_nodes_to_exclude(self):
        """When nodes_to_exclude are provided, they should be excluded from dependency graph"""
        model = models_for_tests.single_residual_model().model
        dl = [{"input": np.random.randn(1, 3, 32, 32).astype(np.float32)}]
        dep_graph = DependencyGraph(
            ConnectedGraph(model), dl, nodes_to_exclude=["/conv1/Conv"]
        )
        sorted_order = dep_graph.get_topologically_sorted_nodes()

        for i, sorted_nodes in sorted_order.items():
            print(f"{i}: {sorted_nodes}")

        """
        When: Given node is excluded
        Then: The children of excluded nodes are still processed and added to dependency graph.
        """

        assert "/conv1/Conv" not in [
            node.cg_op.name for nodes in sorted_order.values() for node in nodes
        ]
        assert "/conv2/Conv" in [
            node.cg_op.name for nodes in sorted_order.values() for node in nodes
        ]
        assert "/fc/Gemm" in [
            node.cg_op.name for nodes in sorted_order.values() for node in nodes
        ]


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

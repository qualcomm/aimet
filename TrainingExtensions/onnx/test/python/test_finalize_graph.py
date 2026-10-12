# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import gc
import sys

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import onnx_ir
import pytest

from aimet_onnx.common.onnx._utils import _add_onnx_qdq_nodes, _finalize_graph_changes

# --- Measurement primitives -------------------------------------------------
#
# `/proc/self/clear_refs` (write "5") resets this process's VmHWM (peak RSS)
# so a later read of VmHWM reflects only the peak since the reset, instead of
# being contaminated by an unrelated earlier peak (e.g. model construction).
# This was the only measurement primitive found to separate the buggy
# implementation from an in-place fix; psutil polling and `ru_maxrss` deltas
# were both tried and either added their own overhead/instability or silently
# under-reported because the HWM was already inflated before the call being
# measured. Both are Linux-only (/proc), hence the skip guard below.


def _status(field: str) -> int:
    """Read a kB-valued field (e.g. VmRSS, VmHWM) from /proc/self/status, in bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024  # kB -> bytes
    raise RuntimeError(field + " not found in /proc/self/status")


def _reset_hwm() -> bool:
    """Reset VmHWM to the current RSS. Returns False if not permitted (some
    sandboxes disallow the write), so callers can skip gracefully."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


_LAYERS = 4
_DIM = 4096  # 4 * 4096 * 4096 * 4 bytes/float32 = 256 MiB total initializer bytes
_MAX_RATIO = 1.5


def _make_model() -> onnx.ModelProto:
    """A chain of `_LAYERS` MatMul nodes, each with its own `_DIM` x `_DIM`
    fp32 initializer"""
    nodes, inits, prev = [], [], "input"
    for idx in range(_LAYERS):
        name = f"weight_{idx}"
        # np.full, not np.zeros: calloc'd zero pages may never become
        # resident, which would make the measured working-set size too small.
        arr = np.full((_DIM, _DIM), 0.5, dtype=np.float32)
        inits.append(onnx.numpy_helper.from_array(arr, name=name))
        out = f"act_{idx}" if idx < _LAYERS - 1 else "output"
        nodes.append(
            onnx.helper.make_node("MatMul", [prev, name], [out], name=f"mm_{idx}")
        )
        prev = out

    return onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            "g",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, [1, _DIM]
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.FLOAT, [1, _DIM]
                )
            ],
            initializer=inits,
        ),
    )


def _make_small_model() -> onnx.ModelProto:
    """A tiny single-MatMul model with one (2, 2) fp32 initializer
    named `w`. Intended for fast, model-light tests of `_finalize_graph_changes`.
    """
    weight_name = "w"
    weight_shape = (2, 2)
    arr = np.full(weight_shape, 0.5, dtype=np.float32)
    inits = [onnx.numpy_helper.from_array(arr, name=weight_name)]

    node = onnx.helper.make_node(
        "MatMul", ["input", weight_name], ["output"], name="mm"
    )

    return onnx.helper.make_model(
        onnx.helper.make_graph(
            [node],
            "g",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, list(weight_shape)
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.FLOAT, list(weight_shape)
                )
            ],
            initializer=inits,
        ),
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Measurement relies on /proc/self/clear_refs + VmHWM (Linux-only)",
)
def test_finalize_graph_changes_peak_memory():
    """
    Given: A model with `_LAYERS` fp32 initializers of `_DIM` x `_DIM` (256 MiB
      total), and a realistic-but-minimal change set targeting the *last*
      layer's weight: remove it, add a renamed fp32 copy plus a scale and a
      zero-point tensor, and insert a QuantizeLinear -> DequantizeLinear node
      pair that reconstructs the weight under a new name. The DQ node is
      listed *before* the Q node in `nodes_to_add` even though DQ consumes
      Q's output, so an implementation that appends nodes in list order
      (instead of topologically sorting) produces an invalid graph.
    When: `_finalize_graph_changes(...)` is called directly
    Then: (1) the change set is applied correctly -- the removed initializer
      is gone, the added initializers are present, the consumer's input is
      renamed, and `onnx.checker.check_model` accepts the result (which also
      validates that the node order is topologically sorted); and
      (2) peak RSS growth during the call stays within `_MAX_RATIO` times the
      model's total initializer bytes.
    """
    if not _reset_hwm():
        pytest.skip("/proc/self/clear_refs is not writable in this environment")

    model = _make_model()
    total_initializer_bytes = sum(len(i.raw_data) for i in model.graph.initializer)
    ir_model = onnx_ir.from_proto(model)
    graph = ir_model.graph

    last = _LAYERS - 1
    weight_name = f"weight_{last}"
    weight_init = next(i for i in model.graph.initializer if i.name == weight_name)
    weight_arr = onnx.numpy_helper.to_array(weight_init).copy()

    # Renamed fp32 copy of the removed weight, plus a scale/zero-point pair,
    # so that Q/DQ can round-trip it under new tensor names.
    weight_copy = onnx_ir.Value(
        name=f"{weight_name}_q",
        const_value=onnx_ir.tensor(weight_arr, name=f"{weight_name}_q"),
    )
    scale = onnx_ir.Value(
        name=f"{weight_name}_scale",
        const_value=onnx_ir.tensor(
            np.array(1.0, dtype=np.float32), name=f"{weight_name}_scale"
        ),
    )
    zero_point = onnx_ir.Value(
        name=f"{weight_name}_zp",
        const_value=onnx_ir.tensor(
            np.array(0, dtype=np.int8), name=f"{weight_name}_zp"
        ),
    )
    tensors_to_add = [weight_copy, scale, zero_point]

    q_node = onnx_ir.node(
        "QuantizeLinear",
        inputs=[weight_copy, scale, zero_point],
        outputs=[onnx_ir.Value(name=f"{weight_name}_i8")],
        name=f"{weight_name}_q_node",
    )
    dq_node = onnx_ir.node(
        "DequantizeLinear",
        inputs=[q_node.outputs[0], scale, zero_point],
        outputs=[onnx_ir.Value(name=f"{weight_name}_dq")],
        name=f"{weight_name}_dq_node",
    )

    nodes_to_add = [dq_node, q_node]

    input_replacements = [(graph.initializers[weight_name], dq_node.outputs[0])]

    gc.collect()
    assert _reset_hwm()
    rss_before = _status("VmRSS")

    _finalize_graph_changes(graph, nodes_to_add, input_replacements, tensors_to_add, {})

    hwm_after = _status("VmHWM")

    # --- Correctness first, so a no-op implementation can't pass the memory
    # assertion by construction. ---
    initializer_names = set(graph.initializers)
    assert f"{weight_name}_q" in initializer_names
    assert f"{weight_name}_scale" in initializer_names
    assert f"{weight_name}_zp" in initializer_names

    consumer = next(n for n in graph if n.name == f"mm_{last}")
    consumer_inputs = [v.name for v in consumer.inputs]
    assert f"{weight_name}_dq" in consumer_inputs

    onnx.checker.check_model(
        onnx_ir.to_proto(ir_model)
    )  # also validates topological node order

    ratio = (hwm_after - rss_before) / total_initializer_bytes
    assert ratio < _MAX_RATIO, (
        f"_finalize_graph_changes peaked at {ratio:.3f}x the model's "
        f"initializer bytes (limit {_MAX_RATIO}x)"
    )


def test_dependency_through_non_first_input():
    """
    Given: The LPBQ/NVFP4 double-quantization shape, where a dependency
      arrives through a *non-first* input index:
        - `w_scale_dq` (DequantizeLinear): inputs [`w_scale_q`, `w_meta_scale`,
          `w_meta_zp`] -> output `w_scale`
        - `w_q` (QuantizeLinear): inputs [`w`, `w_scale`, `w_zp`] -> output
          `w_q_out`  (depends on `w_scale_dq`'s output via input[1], not
          input[0])
        - `w_dq` (DequantizeLinear): inputs [`w_q_out`, `w_scale`, `w_zp`] ->
          output `w_dq_out`
      Passed to `_finalize_graph_changes` in the order `_add_onnx_qdq_nodes`
      emits them: `w_scale_dq`, `w_q`, `w_dq`. `w` is an original model
      initializer; `w_scale_q`, `w_meta_scale`, and `w_meta_zp` are in
      `tensors_to_add`.
    When: `_finalize_graph_changes(...)` is called
    Then: `onnx.checker.check_model` passes (implying a valid topological
      order), and explicitly, `w_scale_dq` appears before `w_q` in
      `model.graph.node` (the checker alone is the correctness guard; this
      index assertion documents the intent that a same-input-index-only BFS
      would violate).
    """
    model = _make_small_model()
    ir_model = onnx_ir.from_proto(model)
    graph = ir_model.graph
    w_value = graph.initializers["w"]

    w_scale_q = onnx_ir.Value(
        name="w_scale_q",
        const_value=onnx_ir.tensor(
            np.array([[1, 2], [3, 4]], dtype=np.int8), name="w_scale_q"
        ),
    )
    w_meta_scale = onnx_ir.Value(
        name="w_meta_scale",
        const_value=onnx_ir.tensor(
            np.array([1.0, 1.0], dtype=np.float32), name="w_meta_scale"
        ),
    )
    w_meta_zp = onnx_ir.Value(
        name="w_meta_zp",
        const_value=onnx_ir.tensor(np.array([0, 0], dtype=np.int8), name="w_meta_zp"),
    )
    w_zp = onnx_ir.Value(
        name="w_zp",
        const_value=onnx_ir.tensor(np.array([0, 0], dtype=np.int8), name="w_zp"),
    )

    w_scale_dq = onnx_ir.node(
        "DequantizeLinear",
        inputs=[w_scale_q, w_meta_scale, w_meta_zp],
        outputs=[onnx_ir.Value(name="w_scale")],
        name="w_scale_dq",
    )
    w_q = onnx_ir.node(
        "QuantizeLinear",
        inputs=[w_value, w_scale_dq.outputs[0], w_zp],
        outputs=[onnx_ir.Value(name="w_q_out")],
        name="w_q",
    )
    w_dq = onnx_ir.node(
        "DequantizeLinear",
        inputs=[w_q.outputs[0], w_scale_dq.outputs[0], w_zp],
        outputs=[onnx_ir.Value(name="w_dq_out")],
        name="w_dq",
    )

    nodes_to_add = [w_scale_dq, w_q, w_dq]
    tensors_to_add = [w_scale_q, w_meta_scale, w_meta_zp, w_zp]
    input_replacements = [(w_value, w_dq.outputs[0])]

    _finalize_graph_changes(graph, nodes_to_add, input_replacements, tensors_to_add, {})

    onnx.checker.check_model(onnx_ir.to_proto(ir_model))

    node_names = [n.name for n in ir_model.graph]
    assert node_names.index("w_scale_dq") < node_names.index("w_q")


def test_nodes_sharing_first_input_are_all_kept():
    """
    Given: Two independent Q/DQ pairs (`a_q`/`a_dq` and `b_q`/`b_dq`) that all
      have the *same* input[0] (the shared weight `w`): `a_q` and `b_q` both
      read `w` first.
    When: `_finalize_graph_changes(...)` is called with all four nodes in
      `nodes_to_add`
    Then: all four node names are present in the resulting graph, the total
      node count increases by exactly 4 relative to the original graph, and
      `onnx.checker.check_model` passes. This is a regression test
      that ensures that keying new nodes by `input[0]` doesn't silently drop nodes
      that share that input.
    """
    model = _make_small_model()
    n_original = len(model.graph.node)

    # Two consumers so both dq outputs are used and the graph stays valid.
    consumer_a = onnx.helper.make_node(
        "Identity", ["a_dq_out"], ["out_a"], name="consumer_a"
    )
    consumer_b = onnx.helper.make_node(
        "Identity", ["b_dq_out"], ["out_b"], name="consumer_b"
    )
    model.graph.node.extend([consumer_a, consumer_b])
    model.graph.output.extend(
        [
            onnx.helper.make_tensor_value_info("out_a", onnx.TensorProto.FLOAT, [2, 2]),
            onnx.helper.make_tensor_value_info("out_b", onnx.TensorProto.FLOAT, [2, 2]),
        ]
    )
    n_original = len(model.graph.node)

    ir_model = onnx_ir.from_proto(model)
    graph = ir_model.graph
    w_value = graph.initializers["w"]
    original_values = onnx_ir.convenience.create_value_mapping(
        graph, include_subgraphs=False
    )

    a_scale = onnx_ir.Value(
        name="a_scale",
        const_value=onnx_ir.tensor(np.array(1.0, dtype=np.float32), name="a_scale"),
    )
    a_zp = onnx_ir.Value(
        name="a_zp",
        const_value=onnx_ir.tensor(np.array(0, dtype=np.int8), name="a_zp"),
    )
    b_scale = onnx_ir.Value(
        name="b_scale",
        const_value=onnx_ir.tensor(np.array(1.0, dtype=np.float32), name="b_scale"),
    )
    b_zp = onnx_ir.Value(
        name="b_zp",
        const_value=onnx_ir.tensor(np.array(0, dtype=np.int8), name="b_zp"),
    )

    a_q = onnx_ir.node(
        "QuantizeLinear",
        inputs=[w_value, a_scale, a_zp],
        outputs=[onnx_ir.Value(name="a_q_out")],
        name="a_q",
    )
    # Reuse the dangling Value that `consumer_a` already references by name,
    # so the Identity node's existing input gets wired to this node's output.
    a_dq = onnx_ir.node(
        "DequantizeLinear",
        inputs=[a_q.outputs[0], a_scale, a_zp],
        outputs=[original_values["a_dq_out"]],
        name="a_dq",
    )
    b_q = onnx_ir.node(
        "QuantizeLinear",
        inputs=[w_value, b_scale, b_zp],
        outputs=[onnx_ir.Value(name="b_q_out")],
        name="b_q",
    )
    b_dq = onnx_ir.node(
        "DequantizeLinear",
        inputs=[b_q.outputs[0], b_scale, b_zp],
        outputs=[original_values["b_dq_out"]],
        name="b_dq",
    )

    nodes_to_add = [a_q, a_dq, b_q, b_dq]
    tensors_to_add = [a_scale, a_zp, b_scale, b_zp]
    input_replacements = []

    _finalize_graph_changes(graph, nodes_to_add, input_replacements, tensors_to_add, {})

    node_names = {n.name for n in ir_model.graph}
    for name in ("a_q", "a_dq", "b_q", "b_dq"):
        assert name in node_names

    assert len(ir_model.graph) == n_original + 4

    original_order = ["mm", "consumer_a", "consumer_b"]
    assert [
        n.name for n in ir_model.graph if n.name in original_order
    ] == original_order

    onnx.checker.check_model(onnx_ir.to_proto(ir_model))


def test_insertion_order_is_deterministic():
    """
    Given: The same model and change set as
      `test_dependency_through_non_first_input`, built twice from scratch.
    When: `_finalize_graph_changes(...)` is applied to each copy
    Then: Both runs produce the identical node order, and every new node
      appears after the producers of its inputs.
    """

    def _run():
        model = _make_small_model()
        ir_model = onnx_ir.from_proto(model)
        graph = ir_model.graph
        w_value = graph.initializers["w"]

        w_scale_q = onnx_ir.Value(
            name="w_scale_q",
            const_value=onnx_ir.tensor(
                np.array([[1, 2], [3, 4]], dtype=np.int8), name="w_scale_q"
            ),
        )
        w_meta_scale = onnx_ir.Value(
            name="w_meta_scale",
            const_value=onnx_ir.tensor(
                np.array([1.0, 1.0], dtype=np.float32), name="w_meta_scale"
            ),
        )
        w_meta_zp = onnx_ir.Value(
            name="w_meta_zp",
            const_value=onnx_ir.tensor(
                np.array([0, 0], dtype=np.int8), name="w_meta_zp"
            ),
        )
        w_zp = onnx_ir.Value(
            name="w_zp",
            const_value=onnx_ir.tensor(np.array([0, 0], dtype=np.int8), name="w_zp"),
        )

        w_scale_dq = onnx_ir.node(
            "DequantizeLinear",
            inputs=[w_scale_q, w_meta_scale, w_meta_zp],
            outputs=[onnx_ir.Value(name="w_scale")],
            name="w_scale_dq",
        )
        w_q = onnx_ir.node(
            "QuantizeLinear",
            inputs=[w_value, w_scale_dq.outputs[0], w_zp],
            outputs=[onnx_ir.Value(name="w_q_out")],
            name="w_q",
        )
        w_dq = onnx_ir.node(
            "DequantizeLinear",
            inputs=[w_q.outputs[0], w_scale_dq.outputs[0], w_zp],
            outputs=[onnx_ir.Value(name="w_dq_out")],
            name="w_dq",
        )

        nodes_to_add = [w_scale_dq, w_q, w_dq]
        tensors_to_add = [w_scale_q, w_meta_scale, w_meta_zp, w_zp]
        input_replacements = [(w_value, w_dq.outputs[0])]

        _finalize_graph_changes(
            graph, nodes_to_add, input_replacements, tensors_to_add, {}
        )
        return graph

    graph = _run()
    order = [n.name for n in graph]
    assert order == [n.name for n in _run()]

    position = {n.name: i for i, n in enumerate(graph)}
    for node in graph:
        for value in node.inputs:
            if value is not None and value.producer() is not None:
                assert position[value.producer().name] < position[node.name]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Measurement relies on /proc/self/clear_refs + VmHWM (Linux-only)",
)
def test_add_onnx_qdq_nodes_peak_memory():
    """
    Given: The `_LAYERS` x 64 MiB model from `test_finalize_graph_changes_peak_memory`
    When: `_add_onnx_qdq_nodes(...)` inserts per-channel weight QDQ for every
      weight and the caller rebinds `model` to the returned ModelProto
    Then: The result is a valid model with one Q/DQ pair per weight, and peak
      RSS growth stays within `_MAX_RATIO` times the initializer bytes. The
      single `to_proto` materialization costs ~1x; copying it back into the
      input model (`CopyFrom`) would cost ~2x.
    """
    if not _reset_hwm():
        pytest.skip("/proc/self/clear_refs is not writable in this environment")

    model = _make_model()
    total_initializer_bytes = sum(len(i.raw_data) for i in model.graph.initializer)
    names = [f"weight_{idx}" for idx in range(_LAYERS)]
    encodings = [
        {
            "output_dtype": "int8",
            "y_scale": [0.01] * _DIM,
            "y_zero_point": [0] * _DIM,
            "axis": 1,
        }
        for _ in names
    ]

    gc.collect()
    assert _reset_hwm()
    rss_before = _status("VmRSS")

    model = _add_onnx_qdq_nodes(
        model,
        names,
        [n + "_qdq" for n in names],
        names,
        encodings,
        [np.float32] * _LAYERS,
        onnx_opset=21,
        prequantize_constants=False,
    )

    hwm_after = _status("VmHWM")

    op_types = [n.op_type for n in model.graph.node]
    assert op_types.count("QuantizeLinear") == _LAYERS
    assert op_types.count("DequantizeLinear") == _LAYERS
    onnx.checker.check_model(model)

    ratio = (hwm_after - rss_before) / total_initializer_bytes
    assert ratio < _MAX_RATIO, (
        f"_add_onnx_qdq_nodes peaked at {ratio:.3f}x the model's "
        f"initializer bytes (limit {_MAX_RATIO}x)"
    )


def _make_scan_model() -> onnx.ModelProto:
    """input -> MatMul(outer_w) -> scan_init -> Scan(body) -> scan_out, where the
    Scan body is itself `body_in -> MatMul(body_w) -> body_out`. `outer_w` lives in
    the top-level graph's initializer list; `body_w` lives in the Scan body's."""
    outer_w = onnx.numpy_helper.from_array(
        np.full((2, 2), 0.5, dtype=np.float32), name="outer_w"
    )
    outer_mm = onnx.helper.make_node(
        "MatMul", ["input", "outer_w"], ["scan_init"], name="outer_mm"
    )

    body_w = onnx.numpy_helper.from_array(
        np.full((2, 2), 0.5, dtype=np.float32), name="body_w"
    )
    body_mm = onnx.helper.make_node(
        "MatMul", ["body_in", "body_w"], ["body_out"], name="body_mm"
    )
    body_graph = onnx.helper.make_graph(
        [body_mm],
        "body",
        inputs=[
            onnx.helper.make_tensor_value_info(
                "body_in", onnx.TensorProto.FLOAT, [2, 2]
            )
        ],
        outputs=[
            onnx.helper.make_tensor_value_info(
                "body_out", onnx.TensorProto.FLOAT, [2, 2]
            )
        ],
        initializer=[body_w],
    )
    scan_node = onnx.helper.make_node(
        "Scan",
        ["scan_init"],
        ["scan_out"],
        name="scan1",
        body=body_graph,
        num_scan_inputs=0,
    )

    graph = onnx.helper.make_graph(
        [outer_mm, scan_node],
        "g",
        inputs=[
            onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])
        ],
        outputs=[
            onnx.helper.make_tensor_value_info(
                "scan_out", onnx.TensorProto.FLOAT, [2, 2]
            )
        ],
        initializer=[outer_w],
    )
    return onnx.helper.make_model(
        graph, ir_version=10, opset_imports=[onnx.helper.make_opsetid("", 21)]
    )


def test_add_onnx_qdq_nodes_with_scan_subgraph():
    """
    Given: `_make_scan_model()`'s top-level MatMul weight `outer_w` and its Scan
      body's MatMul weight `body_w`, quantized in the same `_add_onnx_qdq_nodes`
      call -- one encoding scoped to the top-level graph, one to the Scan body.
      This exercises the per-graph `scope` routing in `_add_onnx_qdq_nodes`: new
      nodes/tensors/input-replacements must be built against the graph that
      actually owns the corresponding name, not against whichever graph happens
      to be processed in the per-graph loop.
    When: `_add_onnx_qdq_nodes(...)` is called with `prequantize_constants=True`,
      so each original weight is dropped in favor of a prequantized int8
      constant feeding a single DequantizeLinear (the `input_q` branch).
    Then: (1) the call succeeds -- previously `inputs_to_rename` was a single
      dict shared across every graph, so building `input_replacements` for the
      Scan body looked up `outer_w` (or vice versa for the top-level graph)
      against the wrong graph's value mapping and raised `KeyError`; (2)
      `onnx.checker.check_model` accepts the result; (3) each graph's own
      consumer (`outer_mm` / `body_mm`) is rewired to its own DQ output; (4) the
      original `outer_w`/`body_w` initializers are gone from their respective
      graphs -- `body_w` in particular, since `RemoveUnusedNodesPass` only
      prunes the top-level graph's initializers and would otherwise leave an
      unused weight dangling inside the Scan body.
    """
    model = _make_scan_model()
    model = _add_onnx_qdq_nodes(
        model,
        ["outer_w", "body_w"],
        ["outer_w_qdq", "body_w_qdq"],
        ["outer_w_q", "body_w_q"],
        [
            {"output_dtype": "int8", "y_scale": 0.01, "y_zero_point": 0},
            {"output_dtype": "int8", "y_scale": 0.01, "y_zero_point": 0},
        ],
        [np.float32, np.float32],
        onnx_opset=21,
        prequantize_constants=True,
    )

    onnx.checker.check_model(model)

    outer_mm = next(n for n in model.graph.node if n.name == "outer_mm")
    assert "outer_w_qdq" in outer_mm.input

    scan_node = next(n for n in model.graph.node if n.op_type == "Scan")
    body = next(attr.g for attr in scan_node.attribute if attr.name == "body")
    body_mm = next(n for n in body.node if n.name == "body_mm")
    assert "body_w_qdq" in body_mm.input

    assert "outer_w" not in {init.name for init in model.graph.initializer}
    assert "body_w" not in {init.name for init in body.initializer}

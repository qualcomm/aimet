# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for quantizing tensors inside control-flow bodies (``_quantize_subgraphs_enabled``)."""

import numpy as np
import onnx
import pytest
from onnx import helper

import aimet_onnx.quantsim as quantsim
from aimet_onnx.common.onnx._utils import _iterate_graphs_recursive
from aimet_onnx.quantsim import QuantizationSimModel, _quantize_subgraphs_enabled

from .models import models_for_tests


def _inputs(model, seed=0):
    rng = np.random.default_rng(seed)
    return {
        inp.name: rng.standard_normal(
            [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        ).astype(np.float32)
        for inp in model.graph.input
    }


def _sim(model, flag=True, **kwargs):
    with _quantize_subgraphs_enabled(flag):
        return QuantizationSimModel(model, **kwargs)


def _quantized_tensors_per_graph(sim):
    return [
        sorted(node.input[0] for node in graph.node if node.op_type == "QcQuantizeOp")
        for graph in _iterate_graphs_recursive(sim.model.model.graph)
    ]


def _bodies(model):
    return list(_iterate_graphs_recursive(model.graph))[1:]


# Fixture, and the tensors quantized in each Scan body (in _iterate_graphs_recursive order)
BODY_QUANTIZERS = [
    (
        lambda: models_for_tests.scan_model(with_capture=True),
        [["b_c0", "b_co0", "b_scaled", "b_x"]],
    ),
    (models_for_tests.wrapped_scan_model, [["b_c", "b_sum", "b_x"]]),
    (
        models_for_tests.nested_scan_model,
        [["o_c", "o_co", "o_inner_out", "o_xs"], ["i_c", "i_co", "i_x"]],
    ),
    (
        models_for_tests.sibling_scans_model,
        [["a_c", "a_t", "a_x"], ["b_c", "b_t", "b_x"]],
    ),
    # b_c is not read by the body
    (models_for_tests.captured_activation_scan_model, [["b_co", "b_pre", "b_x"]]),
]


class TestSubgraphQuantizerInsertion:
    @pytest.mark.parametrize("make_model, _", BODY_QUANTIZERS)
    def test_flag_off_leaves_bodies_unquantized(self, make_model, _):
        """A model with Scan bodies"""
        sim = _sim(make_model(), flag=False)
        top, *bodies = _quantized_tensors_per_graph(sim)
        print(f"top-level graph: {top}, subgraphs: {bodies}")

        """
        When: QuantizationSimModel is created without subgraph quantization
        Then: No quantizer is inserted into any body
        """
        assert top
        assert all(not body for body in bodies)

    @pytest.mark.parametrize("make_model, expected", BODY_QUANTIZERS)
    def test_quantizers_inserted_in_bodies(self, make_model, expected):
        """A model with Scan bodies"""
        model = make_model()

        # w/o subgraph calibration flag
        sim_off = _sim(onnx.ModelProto.FromString(model.SerializeToString()), False)
        # w/ subgraph calibration flag
        sim_on = _sim(model)

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: 1) Body inputs and tensors produced inside each body are quantized in that body
              2) Tensors of the top-level graph are quantized as without the flag
              3) Quantizers of tensors produced inside a body are enabled
        """
        top, *bodies = _quantized_tensors_per_graph(sim_on)
        print(f"top-level graph: {top}, subgraphs: {bodies}")
        body_inputs = {inp.name for body in _bodies(model) for inp in body.input}

        assert bodies == expected
        assert top == _quantized_tensors_per_graph(sim_off)[0]

        assert all(
            sim_on.qc_quantize_op_dict[name].enabled
            for body in bodies
            for name in body
            if name not in body_inputs
        )

    @pytest.mark.parametrize("make_model, _", BODY_QUANTIZERS)
    def test_body_inputs_quantized_as_graph_inputs(self, make_model, _):
        """A model with Scan bodies"""
        model = make_model()
        body_inputs = {inp.name for body in _bodies(model) for inp in body.input}
        sim = _sim(model)

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: Body inputs read in the body are quantized, and like graph inputs,
              their quantizers are disabled by default
        """
        quantized = body_inputs & set(sim.qc_quantize_op_dict)
        assert quantized
        assert not any(sim.qc_quantize_op_dict[name].enabled for name in quantized)

    @pytest.mark.parametrize("make_model, _", BODY_QUANTIZERS)
    def test_body_outputs_read_quantized_tensors(self, make_model, _):
        """A model with Scan bodies"""
        sim = _sim(make_model())
        for body in _bodies(sim.model.model):
            produced = {out: node for node in body.node for out in node.output}
            renamed = [out.name for out in body.output if out.name.endswith("_updated")]
            """
            When: QuantizationSimModel is created with subgraph quantization
            Then: Quantized body outputs (carries, scan outputs) are renamed to the
                  quantizer output, so the quantized value leaves the body
            """
            assert renamed
            assert all(produced[name].op_type == "QcQuantizeOp" for name in renamed)

    @pytest.mark.parametrize("make_model, _", BODY_QUANTIZERS)
    def test_graphs_stay_topologically_sorted(self, make_model, _):
        """A topologically sorted model with Scan bodies"""
        sim = _sim(make_model())

        def check_sorted(graph, outer_names):
            defined = set(outer_names)
            defined |= {inp.name for inp in graph.input}
            defined |= {init.name for init in graph.initializer}
            for node in graph.node:
                # every input a node reads should already be defined when a node is reached
                assert all(not name or name in defined for name in node.input)
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        check_sorted(attr.g, defined)
                defined |= set(node.output)

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: 1) Every graph stays topologically sorted
              2) The nodes referenced by the connected graph are still part of the model
        """
        # 1)
        check_sorted(sim.model.model.graph, ())

        # 2)
        nodes = {
            id(node)
            for graph in _iterate_graphs_recursive(sim.model.model.graph)
            for node in graph.node
        }
        for op in sim.connected_graph.get_all_ops().values():
            assert id(op.get_module()) in nodes

    def test_sort_quantizer_nodes(self):
        """A sim of a Scan model with quantizers moved before all other nodes of each graph"""
        model = models_for_tests.scan_model(with_capture=True)
        sim = _sim(model)
        graphs = list(_iterate_graphs_recursive(sim.model.model.graph))
        other_nodes = [
            [node for node in graph.node if node.op_type != "QcQuantizeOp"]
            for graph in graphs
        ]

        # Put each graph out of topological order - all QcQuantizeOp at the front
        for graph in graphs:
            graph.node.sort(key=lambda node: node.op_type != "QcQuantizeOp")
        assert graphs[1].node[0].op_type == "QcQuantizeOp"

        """
        When: _sort_quantizer_nodes is called
        Then: 1) In every graph, quantizers of graph inputs and initializers come first and
                 every other quantizer is right after the producer of its input
              2) The other nodes are the same objects, in the same order
              3) The session runs

        """
        # Topo sort sim model graph
        sim._sort_quantizer_nodes()

        # 1)
        top, body = ([node.name for node in graph.node] for graph in graphs)
        assert top == [
            "QcQuantizeOp_bw",
            "QcQuantizeOp_init0",
            "QcQuantizeOp_xs",
            "the_scan",
            "QcQuantizeOp_final0",
            "QcQuantizeOp_ys",
        ]
        assert body == [
            "QcQuantizeOp_b_c0",
            "QcQuantizeOp_b_x",
            "b_add0",
            "QcQuantizeOp_b_co0",
            "b_matmul",
            "QcQuantizeOp_b_scaled",
            "b_y_id",
        ]

        # 2)
        for graph, nodes in zip(graphs, other_nodes):
            sorted_nodes = [
                node for node in graph.node if node.op_type != "QcQuantizeOp"
            ]
            assert len(sorted_nodes) == len(nodes)
            assert all(a is b for a, b in zip(sorted_nodes, nodes))

        # 3)
        sim._rebuild_session()
        inputs = _inputs(model)
        sim.compute_encodings([inputs])
        sim.session.run(None, inputs)

    def test_flag_captured_at_construction(self):
        """A sim created inside _quantize_subgraphs_enabled"""
        model = models_for_tests.scan_model(with_capture=True)
        inputs = _inputs(model)

        """
        When: The context exits before encodings are computed
        Then: 1) The global flag is restored
              2) The sim keeps quantizing its bodies
        """
        with _quantize_subgraphs_enabled():
            assert quantsim._quantize_subgraphs
            sim = QuantizationSimModel(model)
        assert not quantsim._quantize_subgraphs

        sim.compute_encodings([inputs])
        assert sim.qc_quantize_op_dict["b_co0"].is_initialized()
        assert _quantized_tensors_per_graph(sim)[1] == [
            "b_c0",
            "b_co0",
            "b_scaled",
            "b_x",
        ]

    def test_single_iteration_matches_flat_model(self):
        """A Scan running once, and the same model with the Scan replaced by its body"""
        scan_model, flat_model = (
            models_for_tests.single_iteration_scan_and_flat_models()
        )
        inputs = _inputs(scan_model)
        scan_sim, flat_sim = _sim(scan_model), _sim(flat_model)

        """
        When: Both sims are calibrated on the same data
        Then: Their outputs are identical
        """
        scan_sim.compute_encodings([inputs])
        flat_sim.compute_encodings([inputs])
        for scan_out, flat_out in zip(
            scan_sim.session.run(None, inputs), flat_sim.session.run(None, inputs)
        ):
            assert np.array_equal(scan_out, flat_out)


class TestSubgraphStaticTensors:
    @pytest.mark.parametrize("kind", ["initializer", "constant"])
    @pytest.mark.parametrize("consumer", ["Add", "MatMul"])
    def test_body_static_tensors_quantized(self, kind, consumer):
        """A Scan body reading a static tensor defined in the body"""
        model = models_for_tests.body_static_tensor_scan_model(kind, consumer)
        inputs = _inputs(model)
        sim = _sim(model)

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: 1) The static tensor is quantized in the body, as a param if it is a weight
                 and as an activation otherwise
              2) Its quantizer is enabled and calibrated, and the sim runs
        """
        # 1)
        assert _quantized_tensors_per_graph(sim)[1] == [
            "b_c",
            "b_co",
            "b_s",
            "b_x",
            "b_y",
        ]
        if consumer == "MatMul":
            assert sim.param_names == ["b_s"]
        else:
            assert sim.param_names == []
            assert "b_s" in sim.activation_names

        # 2)
        sim.compute_encodings([inputs])
        assert sim.qc_quantize_op_dict["b_s"].enabled
        assert sim.qc_quantize_op_dict["b_s"].is_initialized()
        sim.session.run(None, inputs)

    @pytest.mark.parametrize("kind", ["initializer", "constant"])
    @pytest.mark.parametrize("consumer", ["Add", "MatMul"])
    def test_body_static_tensors_flag_off(self, kind, consumer):
        """A Scan body reading a static tensor defined in the body"""
        model = models_for_tests.body_static_tensor_scan_model(kind, consumer)
        inputs = _inputs(model)
        sim = _sim(model, flag=False)

        """
        When: QuantizationSimModel is created without subgraph quantization
        Then: 1) The static tensor is not quantized, nor is any other body tensor
              2) The sim runs, also after folding param quantizers
        """
        # 1)
        assert sim.param_names == []
        assert "b_s" not in sim.qc_quantize_op_dict
        assert not _quantized_tensors_per_graph(sim)[1]

        # 2)
        sim.compute_encodings([inputs])
        sim.session.run(None, inputs)
        sim.fold_param_quantizers()
        sim.session.run(None, inputs)

    def test_captured_weight_quantized_as_param(self):
        """A Scan body reading a weight defined in the top-level graph"""
        sim = _sim(models_for_tests.scan_model(with_capture=True))

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: 1) The weight is quantized as a parameter in the top-level graph
              2) The body reads the quantized weight
        """
        assert sim.param_names == ["bw"]
        (bw_q,) = [
            node
            for node in sim.model.graph().node
            if node.op_type == "QcQuantizeOp" and node.input[0] == "bw"
        ]
        (body,) = _bodies(sim.model.model)
        (matmul,) = [node for node in body.node if node.op_type == "MatMul"]
        assert matmul.input[1] == bw_q.output[0]


class TestSubgraphQuantizerConstraints:
    def test_config_applied_in_body(self):
        """A Scan body with Softmax -> Transpose -> MatMul (second input)"""
        model = models_for_tests.body_scan_model(
            [
                helper.make_node("Mul", ["b_x", "b_c"], ["b_m"], name="b_mul"),
                helper.make_node(
                    "Softmax", ["b_m"], ["b_sm"], name="b_softmax", axis=-1
                ),
                helper.make_node(
                    "Transpose", ["b_sm"], ["b_smt"], name="b_tr", perm=[1, 0]
                ),
                helper.make_node("MatMul", ["b_c", "b_smt"], ["b_mm"], name="b_matmul"),
                helper.make_node("Add", ["b_mm", "b_c"], ["b_co"], name="b_add"),
            ]
        )
        sim = _sim(model, config_file="htp_v73")

        """
        When: QuantizationSimModel is created with subgraph quantization and htp_v73 config
        Then: The body quantizers get the same settings as the same ops at the top level
              1) Softmax output has the fixed [0, 1] range
              2) MatMul second input is symmetric, looking through the Transpose
        """

        softmax_qtzr = sim.qc_quantize_op_dict["b_sm"]
        assert softmax_qtzr.enabled
        assert softmax_qtzr._encoding_min_max_fixed_vals == (0.0, 1.0)
        assert softmax_qtzr.use_symmetric_encodings
        assert sim._get_enabled_quantizer_name("b_smt") == "b_sm"

    @pytest.mark.parametrize("multiple_consumers", [False, True])
    def test_relu_tie_in_body(self, multiple_consumers):
        """A Scan body with Mul -> Relu"""
        nodes = [
            helper.make_node("Mul", ["b_x", "b_c"], ["b_m"], name="b_mul"),
            helper.make_node("Relu", ["b_m"], ["b_r"], name="b_relu"),
        ]
        if multiple_consumers:
            nodes += [
                helper.make_node("Add", ["b_m", "b_c"], ["b_s"], name="b_add0"),
                helper.make_node("Add", ["b_r", "b_s"], ["b_co"], name="b_add"),
            ]
        else:
            nodes += [helper.make_node("Add", ["b_r", "b_c"], ["b_co"], name="b_add")]
        sim = _sim(models_for_tests.body_scan_model(nodes), config_file="htp_v73")

        """
        When: QuantizationSimModel is created with subgraph quantization and htp_v73 config
        Then: 1) The Relu input and output quantizers are tied if the Mul output has one consumer
              2) They are not tied if the Mul output has another consumer
        """

        tied = sim.qc_quantize_op_dict["b_m"] is sim.qc_quantize_op_dict["b_r"]
        assert tied != multiple_consumers

    @pytest.mark.parametrize(
        "names",
        [
            ["", "", ""],
            # The body Mul clashes with a top-level node, which keeps its name
            ["t_relu", "b_relu", "b_add"],
        ],
    )
    def test_body_node_names_filled(self, names):
        """A Scan body with Mul -> Relu -> Add, whose node names are missing or not unique"""

        def make_model(names):
            return models_for_tests.body_scan_model(
                [
                    helper.make_node("Mul", ["b_x", "b_x"], ["b_m"], name=names[0]),
                    helper.make_node("Relu", ["b_m"], ["b_r"], name=names[1]),
                    helper.make_node("Add", ["b_c", "b_r"], ["b_co"], name=names[2]),
                ],
                top_nodes=[
                    helper.make_node("Mul", ["x", "x"], ["t"], name="t_mul"),
                    helper.make_node("Relu", ["t"], ["r"], name="t_relu"),
                ],
                top_outputs=["r"],
            )

        named = _sim(make_model(["b_mul", "b_relu", "b_add"]), config_file="htp_v73")
        sim = _sim(make_model(names), config_file="htp_v73")

        """
        When: QuantizationSimModel is created with subgraph quantization and htp_v73 config
        Then: 1) Every node has a unique name, top-level nodes keep theirs
              2) Every body op is in the connected graph
              3) The body quantizers are configured as when the nodes are uniquely named
        """
        # 1)
        nodes = [
            node
            for graph in _iterate_graphs_recursive(sim.model.model.graph)
            for node in graph.node
            if node.op_type != "QcQuantizeOp"
        ]
        assert all(node.name for node in nodes)
        assert len({node.name for node in nodes}) == len(nodes)
        assert {"the_scan", "t_mul", "t_relu"} <= {node.name for node in nodes}

        # 2)
        assert len(sim.connected_graph.get_all_ops()) == len(
            named.connected_graph.get_all_ops()
        )

        # 3)
        (body_quantized,) = _quantized_tensors_per_graph(sim)[1:]
        assert body_quantized == _quantized_tensors_per_graph(named)[1]
        for name in body_quantized:
            assert (
                sim.qc_quantize_op_dict[name].enabled
                == named.qc_quantize_op_dict[name].enabled
            )

    def test_captured_tensor_not_tied_to_top_level_relu(self):
        """A top-level tensor t consumed by a Relu and captured by a Scan body"""
        model = models_for_tests.body_scan_model(
            [
                helper.make_node("Mul", ["b_x", "t"], ["b_m"], name="b_mul"),
                helper.make_node("Add", ["b_m", "b_c"], ["b_co"], name="b_add"),
            ],
            top_nodes=[
                helper.make_node("Mul", ["x", "x"], ["t"], name="t_mul"),
                helper.make_node("Relu", ["t"], ["r"], name="t_relu"),
            ],
            top_outputs=["r"],
        )
        sim = _sim(model, config_file="htp_v73")

        """
        When: QuantizationSimModel is created with subgraph quantization and htp_v73 config
        Then: The quantizer of t is not tied to the Relu output, since the body also reads t
        """
        assert sim.qc_quantize_op_dict["t"] is not sim.qc_quantize_op_dict["r"]
        assert sim.qc_quantize_op_dict["t"]._encoding_min_max_fixed_vals is None

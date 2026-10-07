# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for quantizing tensors inside control-flow bodies (``_quantize_subgraphs_enabled``)."""

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

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


class TestSubgraphQuantizerRemoval:
    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("make_model, _", BODY_QUANTIZERS)
    def test_remove_quantizers(self, make_model, _, flag):
        """A sim of a model with Scan bodies"""
        model = make_model()
        inputs = _inputs(model)
        sim = _sim(onnx.ModelProto.FromString(model.SerializeToString()), flag)

        """
        When: remove_quantizers is called on the sim model
        Then: 1) No quantizer is left in any graph
              2) The model is valid and computes the same outputs as the original model
        """
        fp_model = QuantizationSimModel.remove_quantizers(
            onnx.ModelProto.FromString(sim.model.model.SerializeToString())
        )
        assert not any(
            node.op_type == "QcQuantizeOp"
            for graph in _iterate_graphs_recursive(fp_model.graph)
            for node in graph.node
        )
        onnx.checker.check_model(fp_model, full_check=True)
        expected = ort.InferenceSession(model.SerializeToString()).run(None, inputs)
        actual = ort.InferenceSession(fp_model.SerializeToString()).run(None, inputs)

        # Before and after output should be bit-exact
        for exp, act in zip(expected, actual):
            print(f"exp: {exp.shape}, act: {act.shape}")
            assert np.array_equal(act, exp)

    @pytest.mark.parametrize("make_model, _", BODY_QUANTIZERS)
    def test_remove_quantization_nodes_restores_model(self, make_model, _):
        """A sim of a model with Scan bodies"""
        sim = _sim(make_model())
        before = sim.model.model.SerializeToString()

        """
        When: The _remove_quantization_nodes context exits
        Then: Every graph is restored, including the quantizers inside the bodies
        """
        with sim._remove_quantization_nodes():
            assert sim.model.model.SerializeToString() != before
        assert sim.model.model.SerializeToString() == before

    def test_fold_param_quantizers_with_captured_weight(self):
        """A sim of a Scan model whose body reads a weight of the top-level graph"""
        model = models_for_tests.scan_model(with_capture=True)
        inputs = _inputs(model)
        sim = _sim(model, flag=False)
        sim.compute_encodings([inputs])
        expected = sim.session.run(None, inputs)

        def bw_quantizers():
            return [
                (list(node.input), list(node.output))
                for node in sim.model.model.graph.node
                if node.op_type == "QcQuantizeOp" and node.input[0] == "bw"
            ]

        def body_matmul():
            (body,) = _bodies(sim.model.model)
            (matmul,) = [node for node in body.node if node.op_type == "MatMul"]
            return matmul

        assert bw_quantizers() == [(["bw"], ["bw_qdq"])]
        assert body_matmul().input[1] == "bw_qdq"

        """
        When: fold_param_quantizers is called
        Then: 1) bw's quantizer is removed and the body reads the folded weight
              2) The sim computes the same outputs as before folding
        """
        sim.fold_param_quantizers()
        assert not bw_quantizers()
        assert body_matmul().input[1] == "bw"
        for exp, act in zip(expected, sim.session.run(None, inputs)):
            assert np.array_equal(act, exp)


class TestSubgraphExport:
    @pytest.mark.parametrize("encoding_version", ["0.6.1", "1.0.0", "2.0.0"])
    @pytest.mark.parametrize("make_model, _", BODY_QUANTIZERS)
    def test_export(self, make_model, _, encoding_version, tmp_path):
        """A calibrated sim of a model with Scan bodies"""
        model = make_model()
        inputs = _inputs(model)
        sim = _sim(onnx.ModelProto.FromString(model.SerializeToString()))
        sim.compute_encodings([inputs])
        sim_model = sim.model.model.SerializeToString()

        """
        When: The sim is exported
        Then: 1) The exported model has no quantizer, is valid and computes the same outputs
                 as the original model
              2) The encodings of the body quantizers are exported, and every exported
                 encoding names a tensor of the exported model
              3) The sim model is unchanged
        """
        sim.export(str(tmp_path), "model", encoding_version=encoding_version)

        # 1)
        exported = onnx.load(str(tmp_path / "model.onnx"))
        graphs = list(_iterate_graphs_recursive(exported.graph))
        assert not any(
            node.op_type == "QcQuantizeOp" for graph in graphs for node in graph.node
        )
        onnx.checker.check_model(exported, full_check=True)
        expected = ort.InferenceSession(model.SerializeToString()).run(None, inputs)
        actual = ort.InferenceSession(exported.SerializeToString()).run(None, inputs)
        for exp, act in zip(expected, actual):
            assert np.array_equal(act, exp)

        # 2)
        with open(tmp_path / "model.encodings") as f:
            encodings = json.load(f)
        if encoding_version == "0.6.1":
            names = {*encodings["activation_encodings"], *encodings["param_encodings"]}
        elif encoding_version == "1.0.0":
            names = {
                enc["name"]
                for enc in encodings["activation_encodings"]
                + encodings["param_encodings"]
            }
        else:
            names = {enc["name"] for enc in encodings["encodings"]}

        # Body inputs are quantized with disabled quantizers, which export no encoding
        body_quantized = {
            name
            for body in _quantized_tensors_per_graph(sim)[1:]
            for name in body
            if sim.qc_quantize_op_dict[name].enabled
        }
        # Ensure every enabled body quantizer is in the exported encodings.
        assert body_quantized <= names

        tensors = {
            name
            for graph in graphs
            for name in [
                *(inp.name for inp in graph.input),
                *(init.name for init in graph.initializer),
                *(out for node in graph.node for out in node.output),
            ]
        }
        # Ensure every exported encoding names a real tensor in the exported model.
        assert names <= tensors

        # 3)
        assert sim.model.model.SerializeToString() == sim_model

    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("make_model, expected", BODY_QUANTIZERS)
    def test_to_onnx_qdq(self, make_model, expected, flag):
        """A calibrated sim of a model with Scan bodies"""
        model = make_model()
        inputs = _inputs(model)
        sim = _sim(model, flag)
        sim.compute_encodings([inputs])

        """
        When: The sim is exported to onnx QDQ
        Then: 1) No quantizer is left, and each body quantizes the tensors its enabled
                 quantizers did (none with the flag off)
              2) The model is valid and computes the same outputs as the sim
        """
        qdq = sim.to_onnx_qdq()

        # 1)
        graphs = list(_iterate_graphs_recursive(qdq.graph))
        assert not any(
            node.op_type == "QcQuantizeOp" for graph in graphs for node in graph.node
        )
        # Graph outputs swap names with their DequantizeLinear, so read the name off the scale
        quantized = [
            {
                node.input[1].removesuffix("_scale")
                for node in graph.node
                if node.op_type == "QuantizeLinear"
            }
            for graph in graphs[1:]
        ]
        if flag:
            for body, names in zip(quantized, expected):
                # With >= - check that nothing is missing and tolerates derived extra
                # like QDQ derived from intput for Identity
                assert body >= {
                    name for name in names if sim.qc_quantize_op_dict[name].enabled
                }
        else:
            assert not any(quantized)

        # 2)
        onnx.checker.check_model(qdq, full_check=True)
        actual = ort.InferenceSession(qdq.SerializeToString()).run(None, inputs)
        for exp, act in zip(sim.session.run(None, inputs), actual):
            assert np.array_equal(act, exp)

    @pytest.mark.parametrize(
        "make_model",
        [
            lambda: models_for_tests.scan_model(with_capture=True),
            models_for_tests.wrapped_scan_model,
        ],
    )
    def test_data_movement_op_in_body(self, make_model, tmp_path):
        """A calibrated sim of a model whose Scan body writes b_y = Identity(quantized tensor)"""
        model = make_model()
        sim = _sim(model)
        sim.compute_encodings([_inputs(model)])

        """
        When: The sim is exported, and exported to onnx QDQ
        Then: b_y's encoding is derived for both
        """
        # Only 2.0.0 encoding version derives extra
        sim.export(str(tmp_path), "model", encoding_version="2.0.0")
        with open(tmp_path / "model.encodings") as f:
            names = {enc["name"] for enc in json.load(f)["encodings"]}
        assert "b_y" in names

        (body,) = _bodies(sim.to_onnx_qdq())
        (dq,) = [node for node in body.node if node.output[0] == "b_y"]
        assert dq.op_type == "DequantizeLinear"

    def test_body_reads_quantized_outer_tensor(self):
        """A sim, subgraph calibration off, of a model whose Scan body reads the quantized top-level t"""
        model = models_for_tests.body_scan_model(
            [
                helper.make_node("Add", ["b_c", "t"], ["b_s"], name="b_add"),
                helper.make_node("Add", ["b_s", "b_x"], ["b_co"], name="b_add2"),
            ],
            top_nodes=[
                helper.make_node("Mul", ["x", "x"], ["t"], name="t_mul"),
                helper.make_node("Relu", ["t"], ["r"], name="t_relu"),
            ],
            top_outputs=["r"],
        )
        inputs = _inputs(model)
        # Disable subgraph calibration
        sim = _sim(model, flag=False)
        sim.compute_encodings([inputs])
        (body,) = _bodies(sim.model.model)
        assert body.node[0].input[1] == "t_updated"

        """
        When: The sim is exported to onnx QDQ
        Then: 1) The body reads t's top-level DequantizeLinear output
              2) The model is valid and computes the same outputs as the sim
        """
        qdq = sim.to_onnx_qdq()
        # 1)
        (body,) = _bodies(qdq)
        (b_add,) = [node for node in body.node if node.name == "b_add"]
        (t_dq,) = [node for node in qdq.graph.node if b_add.input[1] in node.output]
        print(f"b_add reads {b_add.input[1]}, produced by {t_dq.op_type} {t_dq.name}")
        assert t_dq.op_type == "DequantizeLinear"
        assert t_dq.input[1] == "t_scale"

        # 2)
        onnx.checker.check_model(qdq, full_check=True)
        actual = ort.InferenceSession(qdq.SerializeToString()).run(None, inputs)
        for exp, act in zip(sim.session.run(None, inputs), actual):
            assert np.array_equal(act, exp)


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

    @pytest.mark.parametrize("kind", ["initializer", "constant"])
    def test_fold_body_param(self, kind):
        """A calibrated sim of a Scan body reading a weight defined in the body"""
        model = models_for_tests.body_static_tensor_scan_model(kind, "MatMul")
        inputs = _inputs(model)
        sim = _sim(model)
        sim.compute_encodings([inputs])
        expected = sim.session.run(None, inputs)

        """
        When: fold_param_quantizers is called
        Then: 1) The quantizer of b_s is removed from the body
              2) The sim computes the same outputs as before folding
        """
        sim.fold_param_quantizers()

        # 1)
        assert "b_s" in sim._folded_param_quantizers
        assert "b_s" not in _quantized_tensors_per_graph(sim)[1]

        # 2)
        for exp, act in zip(expected, sim.session.run(None, inputs)):
            assert np.array_equal(act, exp)

    @pytest.mark.parametrize("prequantize_constants", [False, True])
    @pytest.mark.parametrize("kind", ["initializer", "constant"])
    def test_export_body_param(self, kind, prequantize_constants, tmp_path):
        """A calibrated sim of a Scan body reading a weight defined in the body"""
        model = models_for_tests.body_static_tensor_scan_model(kind, "MatMul")
        inputs = _inputs(model)
        sim = _sim(model)
        sim.compute_encodings([inputs])

        """
        When: The sim is exported to onnx QDQ, and exported
        Then: 1) The QDQ model quantizes b_s in the body, ahead of time if prequantized, and
                 computes the same outputs as the sim
              2) The encodings of b_s are exported
        """
        # 1)
        qdq = sim.to_onnx_qdq(prequantize_constants=prequantize_constants)
        (body,) = _bodies(qdq)
        assert any(
            node.op_type == "DequantizeLinear" and node.input[1] == "b_s_scale"
            for node in body.node
        )
        assert prequantize_constants != any(
            node.op_type == "QuantizeLinear" and node.input[0] == "b_s"
            for node in body.node
        )
        onnx.checker.check_model(qdq, full_check=True)
        actual = ort.InferenceSession(qdq.SerializeToString()).run(None, inputs)
        for exp, act in zip(sim.session.run(None, inputs), actual):
            assert np.array_equal(act, exp)

        # 2)
        sim.export(str(tmp_path), "model", encoding_version="2.0.0")
        with open(tmp_path / "model.encodings") as f:
            names = {enc["name"] for enc in json.load(f)["encodings"]}
        assert "b_s" in names

    @pytest.mark.parametrize("prequantize_constants", [False, True])
    @pytest.mark.parametrize("kind", ["initializer", "constant"])
    def test_encodings_to_onnx_qdq_body_tensors(
        self, kind, prequantize_constants, tmp_path
    ):
        """An exported model and encodings of a calibrated sim with quantized body tensors"""
        model = models_for_tests.body_static_tensor_scan_model(kind, "MatMul")
        inputs = _inputs(model)
        sim = _sim(model)
        sim.compute_encodings([inputs])
        sim.export(str(tmp_path), "model", encoding_version="2.0.0")
        (body_quantized,) = _quantized_tensors_per_graph(sim)[1:]
        enabled = {
            name for name in body_quantized if sim.qc_quantize_op_dict[name].enabled
        }

        """
        When: The encodings are applied to the exported model as QDQ
        Then: 1) The body quantizes its tensors with enabled quantizers, the param b_s ahead of
                 time if prequantized
              2) The model is valid and computes the same outputs as the sim's QDQ export
        """
        qdq = quantsim.encodings_to_onnx_qdq(
            onnx.load(str(tmp_path / "model.onnx")),
            str(tmp_path / "model.encodings"),
            prequantize_constants=prequantize_constants,
        )
        # 1)
        (body,) = _bodies(qdq)
        quantized = {
            node.input[1] for node in body.node if node.op_type == "QuantizeLinear"
        }
        dequantized = {
            node.input[1] for node in body.node if node.op_type == "DequantizeLinear"
        }

        # Use >= - so extra derived Q/DQs (data movement outputs) don't break.
        assert dequantized >= {f"{name}_scale" for name in enabled}
        assert quantized >= {f"{name}_scale" for name in enabled - {"b_s"}}
        assert prequantize_constants != ("b_s_scale" in quantized)

        # 2)
        onnx.checker.check_model(qdq, full_check=True)
        actual = ort.InferenceSession(qdq.SerializeToString()).run(None, inputs)
        expected = sim.to_onnx_qdq(prequantize_constants=prequantize_constants)
        expected = ort.InferenceSession(expected.SerializeToString()).run(None, inputs)
        for exp, act in zip(expected, actual):
            assert np.array_equal(act, exp)

    @pytest.mark.parametrize("flag", [False, True])
    def test_encodings_to_onnx_qdq_fp16_body(self, flag, tmp_path):
        """An exported float32 model whose Scan body computes in float16"""
        model = models_for_tests.body_scan_model(
            [
                helper.make_node(
                    "Cast", ["b_x"], ["b_h"], to=onnx.TensorProto.FLOAT16, name="b_cast"
                ),
                helper.make_node("Relu", ["b_h"], ["b_r"], name="b_relu"),
                helper.make_node(
                    "Cast", ["b_r"], ["b_f"], to=onnx.TensorProto.FLOAT, name="b_cast2"
                ),
                helper.make_node("Add", ["b_c", "b_f"], ["b_co"], name="b_add"),
            ]
        )
        inputs = _inputs(model)
        sim = _sim(model, flag=flag)
        sim.compute_encodings([inputs])
        sim.export(str(tmp_path), "model", encoding_version="2.0.0")

        """
        When: encodings_to_onnx_qdq is called
        Then: It raises like for a float16 top-level graph
        """
        with pytest.raises(RuntimeError, match="only supported for float32 models"):
            quantsim.encodings_to_onnx_qdq(
                onnx.load(str(tmp_path / "model.onnx")),
                str(tmp_path / "model.encodings"),
            )

    @pytest.mark.parametrize("kind", ["initializer", "constant"])
    def test_load_encodings_body_tensors(self, kind, tmp_path):
        """The exported encodings of a calibrated sim with quantized body tensors"""
        model = models_for_tests.body_static_tensor_scan_model(kind, "MatMul")
        inputs = _inputs(model)
        sim = _sim(model)
        sim.compute_encodings([inputs])
        sim.export(str(tmp_path), "model", encoding_version="2.0.0")

        """
        When: The encodings are loaded into a new sim of the same model
        Then: 1) Every encoding loads, including the body tensors'
              2) The new sim computes the same outputs as the calibrated sim
        """
        new_sim = _sim(model)
        # 1)
        assert not quantsim.load_encodings_to_sim(
            new_sim, str(tmp_path / "model.encodings")
        )
        (body_quantized,) = _quantized_tensors_per_graph(new_sim)[1:]
        for name in body_quantized:
            assert (
                new_sim.qc_quantize_op_dict[name].enabled
                == sim.qc_quantize_op_dict[name].enabled
            )

        # 2)
        for exp, act in zip(
            sim.session.run(None, inputs), new_sim.session.run(None, inputs)
        ):
            assert np.array_equal(act, exp)

    @pytest.mark.parametrize(
        "make_model",
        [make_model for make_model, _ in BODY_QUANTIZERS]
        + [
            lambda: models_for_tests.body_static_tensor_scan_model(
                "initializer", "MatMul"
            ),
            lambda: models_for_tests.body_static_tensor_scan_model("constant", "Add"),
        ],
    )
    @pytest.mark.parametrize("flag", [False, True])
    def test_from_onnx_qdq_body_tensors(self, make_model, flag):
        """The QDQ model of a calibrated sim, with or without quantized body tensors"""
        model = make_model()
        inputs = _inputs(model)
        sim = _sim(model, flag=flag)
        sim.compute_encodings([inputs])
        qdq = sim.to_onnx_qdq()

        """
        When: A sim is created from the QDQ model
        Then: 1) Every Q/DQ maps to a quantizer, and the same quantizers are enabled
              2) Its QDQ model computes the same outputs
        """
        # 1)
        with _quantize_subgraphs_enabled(flag):
            new_sim = QuantizationSimModel.from_onnx_qdq(qdq, strict=True)
        enabled = lambda s: {
            n for n, q in s.qc_quantize_op_dict.items() if q and q.enabled
        }
        assert enabled(new_sim) == enabled(sim)

        # 2)
        new_qdq = new_sim.to_onnx_qdq()
        expected = ort.InferenceSession(qdq.SerializeToString()).run(None, inputs)
        actual = ort.InferenceSession(new_qdq.SerializeToString()).run(None, inputs)
        for exp, act in zip(expected, actual):
            assert np.array_equal(act, exp)

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

    def test_weight_scale_adjusted_for_body_bias(self):
        """A Scan body with a Gemm whose large bias is declared in the body"""
        dim = models_for_tests.DIM
        rng = np.random.default_rng(1)
        weight = rng.standard_normal((dim, dim)).astype(np.float32)
        bias = np.full(dim, 1e7, np.float32)
        model = models_for_tests.body_scan_model(
            [
                helper.make_node("Mul", ["b_x", "b_x"], ["b_m"], name="b_mul"),
                helper.make_node("Gemm", ["b_m", "b_w", "b_b"], ["b_g"], name="b_gemm"),
                # # b_m must not depend on b_g
                helper.make_node("Add", ["b_c", "b_x"], ["b_co"], name="b_add"),
            ],
            body_initializer=[
                numpy_helper.from_array(weight, "b_w"),
                numpy_helper.from_array(bias, "b_b"),
            ],
        )
        sim = _sim(model, config_file="htp_v73")

        """
        When: Encodings are computed
        Then: The weight scale is adjusted against the body bias, so the int32 bias fits
        """
        sim.compute_encodings([_inputs(model)])
        input_scale = sim._get_enabled_quantizer("b_m")._get_scale()
        weight_scale = sim.qc_quantize_op_dict["b_w"]._get_scale()
        assert np.abs(np.round(bias / (input_scale * weight_scale))).max() < 2**31


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

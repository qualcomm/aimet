# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for quantizing tensors inside control-flow bodies (``_quantize_subgraphs_enabled``)."""

import json

import numpy as np
import onnx
import onnx_ir
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from aimet_onnx import ir_utils
import aimet_onnx.quantsim as quantsim
from aimet_onnx.common.onnx._utils import (
    _derive_data_movement_op_encodings,
    _iterate_graphs_recursive,
)
from aimet_onnx.quantsim import QuantizationSimModel, _quantize_subgraphs_enabled

from .models import models_for_tests
from .models.models_for_tests import _control_flow_model, _ti


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


def _if_branches(model):
    """Every branch of every top-level If, keyed by (If node name, branch attribute name)."""
    return {
        (node.name, attr.name): attr.g
        for node in model.graph.node
        if node.op_type == "If"
        for attr in node.attribute
        if attr.type == onnx.AttributeProto.GRAPH
    }


def _rename_if_outputs(model, branch_names):
    """Renames the output of the named If branches to o"""
    for graph in _bodies(model):
        if graph.name in branch_names:
            graph.node[-1].output[0] = "o"
            graph.output[0].name = "o"
    return model


def _with_outer_o(model, source, after_if):
    """
    Renames the then branch output of the first If to o, and defines o again in the graph
    holding the If. Before the If, the branch shadows o, which ONNX forbids; after it, o isn't
    yet defined when the If is reached, which ONNX allows.
    """
    graph = next(
        g
        for g in _iterate_graphs_recursive(model.graph)
        for n in g.node
        if n.op_type == "If"
    )
    index, the_if = next((i, n) for i, n in enumerate(graph.node) if n.op_type == "If")
    then_branch = next(a.g for a in the_if.attribute if a.name == "then_branch")
    then_branch.node[-1].output[0] = "o"
    then_branch.output[0].name = "o"
    graph.node.insert(
        index + after_if, helper.make_node("Abs", [source], ["o"], name="o_def")
    )
    if graph is model.graph:
        graph.output.append(then_branch.output[0])
    else:
        # A Scan body's outputs are fixed by the Scan, so consume o inside the body instead
        out = graph.output[0].name
        producer = next(n for n in graph.node if out in n.output)
        producer.output[0] = f"{out}_pre"
        graph.node.append(
            helper.make_node("Add", [f"{out}_pre", "o"], [out], name="o_use")
        )
    return model


def _if_inputs(model, cond):
    """Random float inputs, with every boolean input (an If condition) set to ``cond``."""
    inputs = _inputs(model)
    inputs.update(
        (inp.name, np.array([cond]))
        for inp in model.graph.input
        if inp.type.tensor_type.elem_type == onnx.TensorProto.BOOL
    )
    return inputs


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

    @pytest.mark.parametrize(
        "make_model",
        [
            models_for_tests.if_model,
            models_for_tests.chained_if_model,
            models_for_tests.weighted_if_model,
        ],
    )
    def test_flag_off_leaves_if_branches_unquantized(self, make_model):
        """A model with If branches"""
        model = make_model()
        sim = _sim(model, flag=False)

        """
        When: QuantizationSimModel is created without subgraph quantization
        Then: 1) ConnectedGraph does not represent the If branches
              2) No quantizer is inserted into any If branch, nor is any branch param quantized
              3) The sim calibrates and runs, taking only one branch
        """
        assert all(
            not op.subgraphs
            for op in sim.connected_graph.get_all_ops().values()
            if op.type == "If"
        )
        branches = _if_branches(sim.model.model).values()
        assert branches
        assert not any(
            node.op_type == "QcQuantizeOp"
            for branch in branches
            for node in branch.node
        )
        assert not set(sim.param_names) & {
            init.name
            for branch in _if_branches(model).values()
            for init in branch.initializer
        }
        inputs = _if_inputs(model, cond=True)
        sim.compute_encodings([inputs])
        sim.session.run(None, inputs)

    @pytest.mark.parametrize(
        "make_model, expected",
        [
            (models_for_tests.if_model, [["thn_o"], ["els_o"]]),
            (models_for_tests.chained_if_model, [["t0_o"], [], ["t1_o"], []]),
            (models_for_tests.weighted_if_model, [["t_o"], ["e_o", "w_else"]]),
            (
                models_for_tests.if_in_scan_model,
                [["b_branch", "b_c", "b_co", "b_x"], ["t_o"], ["e_o"]],
            ),
        ],
    )
    def test_quantizers_inserted_in_if_branches(self, make_model, expected):
        """A model with If branches, one branch per If, in then/else order"""
        model = make_model()
        sim = _sim(model)

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: 1) Tensors produced inside each branch, and params declared there, are
                 quantized in that branch
              2) Activations of the top-level graph are quantized as without the flag
        """
        top, *bodies = [
            sorted(n.input[0] for n in graph.node if n.op_type == "QcQuantizeOp")
            for graph in _iterate_graphs_recursive(sim.model.model.graph)
        ]
        print(f"top-level graph: {top}, subgraphs: {bodies}")
        assert sorted(bodies) == sorted(expected)

        sim_off = _sim(make_model(), flag=False)
        top_off = sorted(
            n.input[0]
            for n in sim_off.model.model.graph.node
            if n.op_type == "QcQuantizeOp"
        )
        # Top-level weights read only inside a branch become params with the flag
        assert [name for name in top if name not in sim.param_names] == top_off

    def test_if_branch_params_are_quantized_where_declared(self):
        """An If whose then branch reads a top-level weight, and whose else branch its own"""
        sim = _sim(models_for_tests.weighted_if_model())
        branches = _if_branches(sim.model.model)
        then_branch = branches["the_if", "then_branch"]
        else_branch = branches["the_if", "else_branch"]

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: 1) Both weights are params
              2) The top-level weight is quantized at top level, the branch weight in its branch
              3) Each branch MatMul reads its weight's quantized output
        """
        assert {"w_top", "w_else"} <= set(sim.param_names)
        top_quantized = {
            n.input[0]
            for n in sim.model.model.graph.node
            if n.op_type == "QcQuantizeOp"
        }
        assert "w_top" in top_quantized
        assert any(
            n.op_type == "QcQuantizeOp" and n.input[0] == "w_else"
            for n in else_branch.node
        )
        (t_mm,) = [n for n in then_branch.node if n.name == "t_mm"]
        (e_mm,) = [n for n in else_branch.node if n.name == "e_mm"]
        assert t_mm.input[1] == "w_top_qdq"
        assert e_mm.input[1] == "w_else_qdq"

    @pytest.mark.parametrize(
        "names",
        [
            ["", ""],
            # The branch MatMuls clash with top-level nodes, which keep their names
            ["the_if", "pre"],
        ],
    )
    def test_if_branch_node_names_filled(self, names):
        """weighted_if_model, whose branch MatMul names are missing or not unique"""

        def make_model(names):
            model = models_for_tests.weighted_if_model()
            branches = _if_branches(model)
            branches["the_if", "then_branch"].node[0].name = names[0]
            branches["the_if", "else_branch"].node[0].name = names[1]
            return model

        named = _sim(models_for_tests.weighted_if_model())
        sim = _sim(make_model(names))

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: 1) Every node has a unique name, top-level nodes keep theirs
              2) Every op is in the connected graph
              3) Every quantizer is configured as when the nodes are uniquely named
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
        assert {"pre", "the_if"} <= {node.name for node in nodes}

        # 2)
        assert len(sim.connected_graph.get_all_ops()) == len(
            named.connected_graph.get_all_ops()
        )

        # 3)
        assert {
            name: qtzr.enabled for name, qtzr in sim.qc_quantize_op_dict.items()
        } == {name: qtzr.enabled for name, qtzr in named.qc_quantize_op_dict.items()}

    @pytest.mark.parametrize(
        "make_model",
        [
            models_for_tests.if_model,
            models_for_tests.chained_if_model,
            models_for_tests.weighted_if_model,
            models_for_tests.if_in_scan_model,
            models_for_tests.predicated_experts_model,
        ],
    )
    def test_calibrating_every_if_branch(self, make_model):
        """A sim of a model with If branches, with subgraph quantization"""
        model = make_model()
        sim = _sim(model)

        """
        When: Calibrated with inputs that take both branches of every If
        Then: 1) Every enabled quantizer, in every branch, has an encoding
              2) The sim runs either way
        """
        sim.compute_encodings(
            [_if_inputs(model, cond=True), _if_inputs(model, cond=False)]
        )
        assert all(
            q.is_initialized() for q in sim.qc_quantize_op_dict.values() if q.enabled
        )
        for cond in (True, False):
            sim.session.run(None, _if_inputs(model, cond=cond))

    @pytest.mark.parametrize(
        "make_model, cond, if_name, branch",
        [
            # w_else is read only inside else_branch
            (models_for_tests.weighted_if_model, True, "the_if", "else_branch"),
            # t0_m is produced inside if_0's then_branch, before its output
            (models_for_tests.predicated_experts_model, False, "if_0", "then_branch"),
        ],
    )
    def test_an_if_branch_that_never_ran_fails_calibration(
        self, make_model, cond, if_name, branch
    ):
        """A sim of a model with If branches, with subgraph quantization"""
        model = make_model()
        sim = _sim(model)

        """
        When: Calibrated with inputs that never take one branch of an If
        Then: Calibration fails, naming the If and the branch that never ran, rather than
              leaving the quantizers inside that branch without an encoding
        """
        with pytest.raises(RuntimeError, match=f"in {branch} of If node {if_name}"):
            sim.compute_encodings([_if_inputs(model, cond=cond)])

    @pytest.mark.parametrize(
        "make_model, cond",
        [
            (models_for_tests.if_model, True),
            (models_for_tests.weighted_if_model, False),
            (models_for_tests.chained_if_model, False),
            (models_for_tests.if_in_scan_model, True),
        ],
    )
    def test_an_if_branch_output_that_never_ran_is_calibrated_via_the_if(
        self, make_model, cond
    ):
        """A sim of a model whose If branches hold quantizers only on their outputs"""
        model = make_model()
        sim = _sim(model)

        """
        When: Calibrated with inputs that never take one branch of an If
        Then: Calibration succeeds, since the output of the branch that never ran shares the
              quantizer of the If output, which saw the data of the other branch
        """
        sim.compute_encodings([_if_inputs(model, cond=cond)])
        assert all(
            q.is_initialized() for q in sim.qc_quantize_op_dict.values() if q.enabled
        )

    @pytest.mark.parametrize(
        "make_model",
        [
            # then: x -> Relu -> o, else: x -> Neg -> o
            lambda: _rename_if_outputs(models_for_tests.if_model(), ["thn", "els"]),
            # if_0 then: h -> o, if_1 then: y0, h -> o
            lambda: _rename_if_outputs(
                models_for_tests.chained_if_model(), ["then_0", "then_1"]
            ),
            # Scan body a: a_c, a_x -> t, Scan body b: b_c, b_x -> t
            lambda: models_for_tests.sibling_scans_model(collide="tensor"),
            # x -> Abs -> o, then x -> the_if (then: x -> Relu -> o)
            lambda: _with_outer_o(models_for_tests.if_model(), "x", after_if=False),
            # x -> the_if (then: x -> Relu -> o), then x -> Abs -> o
            lambda: _with_outer_o(models_for_tests.if_model(), "x", after_if=True),
            # Scan body: b_x -> Abs -> o, then b_if (then: b_x -> Relu -> o)
            lambda: _with_outer_o(
                models_for_tests.if_in_scan_model(), "b_x", after_if=False
            ),
            # Scan body: b_if (then: b_x -> Relu -> o), then b_x -> Abs -> o
            lambda: _with_outer_o(
                models_for_tests.if_in_scan_model(), "b_x", after_if=True
            ),
        ],
        ids=[
            "if_branches",
            "two_ifs",
            "sibling_scans",
            "top_level_before_if",
            "top_level_after_if",
            "body_before_if",
            "body_after_if",
        ],
    )
    def test_tensor_defined_in_two_scopes(self, make_model):
        """
        A model in which two scopes, one of them a control-flow body, define a tensor. ONNX
        rejects the nested *_before_if ones and allows the rest.
        """
        model = make_model()

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: It raises naming the tensor, as quantizers are keyed by tensor name in every scope
        """
        with pytest.raises(
            RuntimeError, match=r"\['[ot]'\] are defined in more than one"
        ):
            _sim(model)

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
    def test_body_inputs_share_boundary_quantizers(self, make_model, _):
        """A model with Scan bodies"""
        model = make_model()
        body_inputs = {inp.name for body in _bodies(model) for inp in body.input}
        sim = _sim(model)
        qtzrs = sim.qc_quantize_op_dict

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: Body inputs read in the body are quantized, each sharing the quantizer of a
              tensor outside the bodies' inputs (the scanned input or the carry), so that
              they are quantized whenever the value they take is
        """
        quantized = body_inputs & set(qtzrs)
        assert quantized
        outside = {id(qtzr) for name, qtzr in qtzrs.items() if name not in body_inputs}
        assert all(id(qtzrs[name]) in outside for name in quantized)

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
        # The Scan ties the carry to its initial state, so the flat model does the same
        flat_sim._set_quantizers(
            {"b_co": flat_sim.qc_quantize_op_dict["init"]}, rebuild_session=True
        )

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


def test_remove_aimet_quantizers_from_scan_bodies():
    sim = _sim(models_for_tests.sibling_scans_model())
    ir_model = onnx_ir.from_proto(sim.model.model)
    scan_bodies = [
        node.attributes["body"].as_graph()
        for node in ir_model.graph
        if node.op_type == "Scan"
    ]
    assert any(
        node.op_type == "QcQuantizeOp" for graph in scan_bodies for node in graph
    )

    ir_utils.remove_aimet_quantizers(ir_model)

    assert not any(
        node.op_type == "QcQuantizeOp"
        for graph in [ir_model.graph, *scan_bodies]
        for node in graph
    )


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


IF_MODELS = [
    models_for_tests.if_model,
    models_for_tests.weighted_if_model,
    models_for_tests.chained_if_model,
    models_for_tests.if_in_scan_model,
    models_for_tests.predicated_experts_model,
]


def _ort_run(model, inputs):
    # Keep ORT from folding Q/DQ pairs, so it computes exactly what the graph says
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options)
    return session.run(None, inputs)


def _calibrated_if_sim(make_model, flag):
    model = make_model()
    sim = _sim(onnx.ModelProto.FromString(model.SerializeToString()), flag)
    sim.compute_encodings([_if_inputs(model, cond=True), _if_inputs(model, cond=False)])
    return model, sim


def _qdq_per_graph(model):
    """(scale, zero point) of every QuantizeLinear, keyed by the tensor its scale is named after,
    for each graph in _iterate_graphs_recursive order"""
    graphs = list(_iterate_graphs_recursive(model.graph))
    constants = {
        init.name: numpy_helper.to_array(init).tolist()
        for graph in graphs
        for init in graph.initializer
    }
    return [
        {
            node.input[1].removesuffix("_scale"): (
                constants[node.input[1]],
                constants[node.input[2]] if len(node.input) > 2 else None,
            )
            for node in graph.node
            if node.op_type == "QuantizeLinear"
        }
        for graph in graphs
    ]


class TestSubgraphIfExport:
    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("make_model", IF_MODELS)
    def test_export(self, make_model, flag, tmp_path):
        """A sim of a model with If branches, calibrated with every branch taken"""
        model, sim = _calibrated_if_sim(make_model, flag)

        """
        When: The sim is exported
        Then: 1) The exported model has no quantizer, is valid and computes the same outputs as
                 the original model
              2) Every enabled quantizer's encoding is exported, and every exported encoding
                 names a tensor of the exported model
        """
        sim.export(str(tmp_path), "model", encoding_version="2.0.0")

        # 1)
        exported = onnx.load(str(tmp_path / "model.onnx"))
        graphs = list(_iterate_graphs_recursive(exported.graph))
        assert not any(
            node.op_type == "QcQuantizeOp" for graph in graphs for node in graph.node
        )
        onnx.checker.check_model(exported, full_check=True)
        for cond in (True, False):
            inputs = _if_inputs(model, cond)
            for exp, act in zip(_ort_run(model, inputs), _ort_run(exported, inputs)):
                assert np.array_equal(act, exp)

        # 2)
        with open(tmp_path / "model.encodings") as f:
            names = {enc["name"] for enc in json.load(f)["encodings"]}
        assert {n for n, q in sim.qc_quantize_op_dict.items() if q.enabled} <= names
        tensors = {
            name
            for graph in graphs
            for name in [
                *(inp.name for inp in graph.input),
                *(init.name for init in graph.initializer),
                *(out for node in graph.node for out in node.output),
            ]
        }
        assert names <= tensors

    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("make_model", IF_MODELS)
    def test_to_onnx_qdq(self, make_model, flag):
        """A sim of a model with If branches, calibrated with every branch taken"""
        model, sim = _calibrated_if_sim(make_model, flag)

        """
        When: The sim is exported to onnx QDQ
        Then: 1) No quantizer is left, and every enabled quantizer is a Q/DQ in the graph that
                 defines its tensor, so none inside a branch with the flag off
              2) No body or branch reads a tensor of its enclosing graph before it is quantized
              3) The model is valid, and with the flag, computes the same outputs as the sim
        """
        qdq = sim.to_onnx_qdq()

        # 1)
        graphs = list(_iterate_graphs_recursive(qdq.graph))
        assert not any(
            node.op_type == "QcQuantizeOp" for graph in graphs for node in graph.node
        )
        quantized = _qdq_per_graph(qdq)
        for graph, sim_graph in zip(quantized, _quantized_tensors_per_graph(sim)):
            assert set(graph) >= {
                name for name in sim_graph if sim.qc_quantize_op_dict[name].enabled
            }
        if flag:
            assert any(quantized[1:])

        # 2)
        for graph in graphs:
            unquantized = {
                n.input[0] for n in graph.node if n.op_type == "QuantizeLinear"
            }
            for body in _iterate_graphs_recursive(graph):
                if body is not graph:
                    assert not unquantized & {i for n in body.node for i in n.input}

        # 3)
        onnx.checker.check_model(qdq, full_check=True)
        # NOTE: Without the flag, the sim's branches read the unquantized tensors of the
        # enclosing graph, since the ConnectedGraph doesn't see the branches.
        if not flag:
            return
        for cond in (True, False):
            inputs = _if_inputs(model, cond)
            for exp, act in zip(sim.session.run(None, inputs), _ort_run(qdq, inputs)):
                assert np.array_equal(act, exp)

    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("make_model", IF_MODELS)
    def test_encodings_to_onnx_qdq(self, make_model, flag, tmp_path):
        """The exported model and encodings of a sim of a model with If branches"""
        _, sim = _calibrated_if_sim(make_model, flag)
        sim.export(str(tmp_path), "model", encoding_version="2.0.0")

        """
        When: The encodings are applied to the exported model as QDQ
        Then: Every graph quantizes the same tensors, with the same scales and zero points, as
              the sim's QDQ export
        """
        qdq = quantsim.encodings_to_onnx_qdq(
            onnx.load(str(tmp_path / "model.onnx")),
            str(tmp_path / "model.encodings"),
            strict=True,
        )
        onnx.checker.check_model(qdq, full_check=True)
        assert _qdq_per_graph(qdq) == _qdq_per_graph(sim.to_onnx_qdq())

    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("make_model", IF_MODELS)
    def test_from_onnx_qdq(self, make_model, flag):
        """The QDQ model of a sim of a model with If branches"""
        model, sim = _calibrated_if_sim(make_model, flag)
        qdq = sim.to_onnx_qdq()

        """
        When: A sim is created from the QDQ model, strictly
        Then: 1) Every Q/DQ maps to a quantizer, and the same quantizers are enabled
              2) Its QDQ model is the same as the one it was created from
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
        assert _qdq_per_graph(new_qdq) == _qdq_per_graph(qdq)
        for cond in (True, False):
            inputs = _if_inputs(model, cond)
            for exp, act in zip(_ort_run(qdq, inputs), _ort_run(new_qdq, inputs)):
                assert np.array_equal(act, exp)

    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("make_model", IF_MODELS)
    def test_load_encodings(self, make_model, flag, tmp_path):
        """The exported encodings of a sim of a model with If branches"""
        model, sim = _calibrated_if_sim(make_model, flag)
        sim.export(str(tmp_path), "model", encoding_version="2.0.0")

        """
        When: The encodings are loaded, strictly, into a new sim of the same model
        Then: The new sim computes the same outputs as the calibrated sim
        """
        new_sim = _sim(model, flag)
        quantsim.load_encodings_to_sim(new_sim, str(tmp_path / "model.encodings"))
        for cond in (True, False):
            inputs = _if_inputs(model, cond)
            for exp, act in zip(
                sim.session.run(None, inputs), new_sim.session.run(None, inputs)
            ):
                assert np.array_equal(act, exp)

    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("make_model", IF_MODELS)
    def test_fold_param_quantizers(self, make_model, flag):
        """A sim of a model with If branches, calibrated with every branch taken"""
        model, sim = _calibrated_if_sim(make_model, flag)
        params = [n for n in sim.param_names if sim.qc_quantize_op_dict[n].enabled]
        expected = {
            cond: sim.session.run(None, _if_inputs(model, cond))
            for cond in (True, False)
        }

        """
        When: fold_param_quantizers is called
        Then: 1) No param quantizer is left in any graph
              2) The sim computes the same outputs as before folding
        """
        sim.fold_param_quantizers()

        # 1)
        assert set(params) <= set(sim._folded_param_quantizers)
        assert not set(params) & {
            name for graph in _quantized_tensors_per_graph(sim) for name in graph
        }

        # 2)
        for cond in (True, False):
            for exp, act in zip(
                expected[cond], sim.session.run(None, _if_inputs(model, cond))
            ):
                assert np.array_equal(act, exp)

    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("make_model", IF_MODELS)
    def test_remove_quantizers(self, make_model, flag):
        """A sim of a model with If branches"""
        model = make_model()
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
        for cond in (True, False):
            inputs = _if_inputs(model, cond)
            for exp, act in zip(_ort_run(model, inputs), _ort_run(fp_model, inputs)):
                assert np.array_equal(act, exp)

    @pytest.mark.parametrize("flag", [False, True])
    def test_encodings_to_onnx_qdq_fp16_branch(self, flag, tmp_path):
        """An exported float32 model whose If then branch computes in float16"""
        model = models_for_tests.if_model()
        then_branch = _if_branches(model)["the_if", "then_branch"]
        then_branch.ClearField("node")
        then_branch.node.extend(
            [
                helper.make_node(
                    "Cast", ["x"], ["t_h"], to=onnx.TensorProto.FLOAT16, name="t_cast"
                ),
                helper.make_node("Relu", ["t_h"], ["t_r"], name="t_relu"),
                helper.make_node(
                    "Cast",
                    ["t_r"],
                    ["thn_o"],
                    to=onnx.TensorProto.FLOAT,
                    name="t_cast2",
                ),
            ]
        )
        sim = _sim(model, flag=flag)
        sim.compute_encodings(
            [_if_inputs(model, cond=True), _if_inputs(model, cond=False)]
        )
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

    def test_weight_scale_adjusted_for_branch_bias(self):
        """An If whose then branch holds a Gemm with a large bias declared in the branch"""
        dim = models_for_tests.DIM
        weight = np.random.default_rng(1).standard_normal((dim, dim)).astype(np.float32)
        bias = np.full(dim, 1e7, np.float32)
        model = models_for_tests.weighted_if_model()
        then_branch = _if_branches(model)["the_if", "then_branch"]
        then_branch.ClearField("node")
        then_branch.node.append(
            helper.make_node("Gemm", ["h", "t_w", "t_b"], ["t_o"], name="t_gemm")
        )
        then_branch.initializer.extend(
            [
                numpy_helper.from_array(weight, "t_w"),
                numpy_helper.from_array(bias, "t_b"),
            ]
        )
        sim = _sim(model, config_file="htp_v73")

        """
        When: Encodings are computed
        Then: The weight scale is adjusted against the branch bias, so the int32 bias fits
        """
        sim.compute_encodings(
            [_if_inputs(model, cond=True), _if_inputs(model, cond=False)]
        )
        input_scale = sim._get_enabled_quantizer("h")._get_scale()
        weight_scale = sim.qc_quantize_op_dict["t_w"]._get_scale()
        assert np.abs(np.round(bias / (input_scale * weight_scale))).max() < 2**31

    def test_branch_param_quantized_in_branch(self):
        """A calibrated sim whose If else branch declares its own weight w_else"""
        _, sim = _calibrated_if_sim(models_for_tests.weighted_if_model, flag=True)

        """
        When: The sim is exported to onnx QDQ, with and without prequantized constants
        Then: w_else is quantized inside the else branch, ahead of time if prequantized
        """
        for prequantize_constants in (False, True):
            qdq = sim.to_onnx_qdq(prequantize_constants=prequantize_constants)
            else_branch = _if_branches(qdq)["the_if", "else_branch"]
            assert any(
                node.op_type == "DequantizeLinear" and node.input[1] == "w_else_scale"
                for node in else_branch.node
            )
            assert prequantize_constants != any(
                node.op_type == "QuantizeLinear" and node.input[0] == "w_else"
                for node in else_branch.node
            )

    @pytest.mark.parametrize("flag", [False, True])
    def test_branch_data_movement_output_encoded(self, flag, tmp_path):
        """A calibrated sim whose if_1 else branch passes the quantized y0 on through Identity"""
        _, sim = _calibrated_if_sim(models_for_tests.chained_if_model, flag)
        sim.export(str(tmp_path), "model", encoding_version="2.0.0")

        """
        When: The sim is exported, and the encodings, without e1_o's, applied to the exported
              model as QDQ
        Then: 1) The Identity output e1_o gets y0's encoding, as in the top-level graph
              2) encodings_to_onnx_qdq derives it too, and quantizes e1_o inside the else
                 branch with y0's scale
        """
        # 1)
        with open(tmp_path / "model.encodings") as f:
            encodings_file = json.load(f)
        encodings = {enc["name"]: enc for enc in encodings_file["encodings"]}
        assert {**encodings["e1_o"], "name": "y0"} == encodings["y0"]

        # 2)
        encodings_file["encodings"] = [
            enc for enc in encodings_file["encodings"] if enc["name"] != "e1_o"
        ]
        qdq = quantsim.encodings_to_onnx_qdq(
            onnx.load(str(tmp_path / "model.onnx")), encodings_file
        )
        else_1 = _if_branches(qdq)["if_1", "else_branch"]
        (q,) = [node for node in else_1.node if node.op_type == "QuantizeLinear"]
        assert q.input[1] == "e1_o_scale"
        scales = {
            init.name: numpy_helper.to_array(init)
            for graph in _iterate_graphs_recursive(qdq.graph)
            for init in graph.initializer
        }
        assert scales["e1_o_scale"] == scales["y0_scale"]

    @pytest.mark.parametrize("flag", [False, True])
    def test_loop_body_left_unquantized(self, flag, tmp_path):
        """A calibrated sim of a model whose Loop body copies a quantized top-level tensor"""
        model = models_for_tests.loop_model()
        inputs = _inputs(model)
        sim = _sim(model, flag)
        sim.compute_encodings([inputs])
        sim.export(str(tmp_path), "model", encoding_version="2.0.0")

        """
        When: The sim is exported, and exported to onnx QDQ both from the sim and the encodings
        Then: 1) No encoding is derived inside the Loop body, which no API quantizes
              2) Neither QDQ model has a Q/DQ inside the Loop body
        """
        # 1)
        with open(tmp_path / "model.encodings") as f:
            names = {enc["name"] for enc in json.load(f)["encodings"]}
        assert "b_h" not in names

        # 2)
        e2q = quantsim.encodings_to_onnx_qdq(
            onnx.load(str(tmp_path / "model.onnx")), str(tmp_path / "model.encodings")
        )
        for qdq in (sim.to_onnx_qdq(), e2q):
            (loop,) = [node for node in qdq.graph.node if node.op_type == "Loop"]
            assert not any(
                node.op_type in ("QuantizeLinear", "DequantizeLinear")
                for node in loop.attribute[0].g.node
            )
            for exp, act in zip(sim.session.run(None, inputs), _ort_run(qdq, inputs)):
                assert np.array_equal(act, exp)


def test_delegation_walk_with_crosswise_branch_names():
    """Two Ifs whose then branches define each other's input:

    b = if_0(cond)   then_branch: x -> Identity -> c -> Identity -> b
    y = if_1(cond)   then_branch: b -> Identity -> c
    """

    def branch(name, *chain):
        nodes = [
            helper.make_node("Identity", [src], [dst], name=f"{name}_{src}")
            for src, dst in zip(chain, chain[1:])
        ]
        return helper.make_graph(nodes, name, [], [_ti(chain[-1], [1])])

    if_0 = helper.make_node(
        "If",
        ["cond"],
        ["b"],
        name="if_0",
        then_branch=branch("then_0", "x", "c", "b"),
        else_branch=branch("else_0", "x", "e"),
    )
    if_1 = helper.make_node(
        "If",
        ["cond"],
        ["y"],
        name="if_1",
        then_branch=branch("then_1", "b", "c"),
        else_branch=branch("else_1", "b", "d"),
    )
    model = _control_flow_model(
        helper.make_graph(
            [if_0, if_1],
            "g",
            [_ti("x", [1]), _ti("cond", [1], onnx.TensorProto.BOOL)],
            [_ti("y", [1])],
        )
    )
    sim = _sim(model, flag=False)
    sim.compute_encodings([{"x": np.ones(1, np.float32), "cond": np.array([True])}])
    encodings = {"b": sim.qc_quantize_op_dict["b"].export_encodings("1.0.0")}
    encodings["c"] = {
        **encodings["b"],
        "scale": [2 * s for s in encodings["b"]["scale"]],
    }

    """
    When: Encodings are delegated given one for c, a branch tensor differing from b's
    Then: The walk ends at the cycle of producers c -> b -> c and keeps c's encoding
    """
    quantsim._remove_delegatable_excess_encodings(sim, encodings)
    assert "c" in encodings


def test_derived_encodings_count_branch_consumers():
    """x, read by a top-level Transpose -> t and by name inside both branches of an If"""
    model = models_for_tests.if_model()
    model.graph.node.append(helper.make_node("Transpose", ["x"], ["t"], name="tr"))
    model.graph.output.append(
        helper.make_tensor_value_info(
            "t", onnx.TensorProto.FLOAT, [models_for_tests.DIM]
        )
    )
    encoding = {"name": "t", "output_dtype": "uint8", "y_scale": 0.1, "y_zero_point": 0}

    """
    When: Encodings are derived through data movement ops, given only t's
    Then: t's encoding is not reused for x, which has other consumers inside the branches
    """
    derived = _derive_data_movement_op_encodings(model, {"t": encoding})
    assert "x" not in derived


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
        # ORT otherwise fuses Q/DQ around ops into integer kernels, which round differently
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        actual = ort.InferenceSession(qdq.SerializeToString(), sess_options).run(
            None, inputs
        )
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


# Fixture, and the tensors sharing one quantizer once the Scan boundary is tied
SCAN_TIES = [
    (
        lambda: models_for_tests.scan_model(num_carries=2, with_capture=True),
        [
            ["init0", "b_c0", "b_co0", "final0"],
            ["init1", "b_c1", "b_co1", "final1"],
            ["b_scaled", "ys"],
            ["xs", "b_x"],
        ],
    ),
    (
        models_for_tests.wrapped_scan_model,
        [["state_in", "b_c", "b_sum", "state_out", "ys"], ["xs", "b_x"]],
    ),
]


class TestSubgraphScanTies:
    @pytest.mark.parametrize("make_model, groups", SCAN_TIES)
    def test_flag_off_no_scan_ties(self, make_model, groups):
        """A model with Scan bodies"""
        sim = _sim(make_model(), flag=False)
        qtzrs = sim.qc_quantize_op_dict
        top = [[name for name in group if name in qtzrs] for group in groups]
        print(f"top-level tensors of each group: {top}")

        """
        When: QuantizationSimModel is created without subgraph quantization
        Then: The initial states, final states and scan outputs keep separate quantizers
        """
        for group in top:
            assert len({id(qtzrs[name]) for name in group}) == len(group)

    @pytest.mark.parametrize("make_model, groups", SCAN_TIES)
    def test_scan_boundary_tied(self, make_model, groups):
        """A model with Scan bodies"""
        sim = _sim(make_model())
        qtzrs = sim.qc_quantize_op_dict

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: 1) Initial state, carry-in, carry-out and final state of a carry share one
                 quantizer, and a scanned input shares it with its body input
              2) A scan output shares the quantizer of the body output stacked into it
              3) Different groups keep different quantizers
              4) After calibration, each group has one enabled encoding
        """
        # 1) and 2)
        for group in groups:
            assert all(qtzrs[name] is qtzrs[group[0]] for name in group)

        # 3)
        assert len({id(qtzrs[group[0]]) for group in groups}) == len(groups)

        # 4)
        sim.compute_encodings([_inputs(sim.model.model)])
        for group in groups:
            assert qtzrs[group[0]].enabled
            assert qtzrs[group[0]].get_encodings()

    @pytest.mark.parametrize("make_model, _", SCAN_TIES)
    def test_body_inputs_exported(self, make_model, _):
        """A calibrated sim of a model with Scan bodies"""
        model = make_model()
        sim = _sim(model)
        sim.compute_encodings([_inputs(model)])

        """
        When: The sim is exported to onnx QDQ
        Then: Every body input is quantized in the body, so ops in the body read only
              quantized values
        """
        qdq = sim.to_onnx_qdq()
        (body,) = _bodies(qdq)
        quantized = {
            node.input[0] for node in body.node if node.op_type == "QuantizeLinear"
        }
        assert {inp.name for inp in body.input} <= quantized


# Fixture, and the tensors that share one quantizer with subgraph quantization
IF_TIES = [
    (models_for_tests.weighted_if_model, [["t_o", "e_o", "y"]]),
    # The else branch of if_0 returns an initializer, which has no quantizer, and that of
    # if_1 passes y0 on, so the running sum shares one quantizer across both Ifs
    (models_for_tests.chained_if_model, [["t0_o", "y0", "t1_o", "y1"]]),
    (models_for_tests.predicated_experts_model, [["t0_o", "y0", "t1_o", "y1"]]),
    # The If inside the Scan body is tied apart from the Scan carry
    (
        models_for_tests.if_in_scan_model,
        [["e_o", "b_branch"], ["init", "b_c", "b_co", "final"]],
    ),
]


def _groups(sim, names):
    """``names`` grouped by the quantizer they share"""
    qtzrs = sim.qc_quantize_op_dict
    return sorted(
        sorted(name for name in names if qtzrs[name] is qtzrs[first])
        for first in names
        if first == min(name for name in names if qtzrs[name] is qtzrs[first])
    )


class TestSubgraphIfTies:
    @pytest.mark.parametrize("make_model, groups", IF_TIES)
    def test_flag_off_no_if_ties(self, make_model, groups):
        """A model with If branches"""
        sim = _sim(make_model(), flag=False)
        qtzrs = sim.qc_quantize_op_dict
        top = [[name for name in group if name in qtzrs] for group in groups]

        """
        When: QuantizationSimModel is created without subgraph quantization
        Then: The If outputs keep separate quantizers
        """
        for group in top:
            assert len({id(qtzrs[name]) for name in group}) == len(group)

    @pytest.mark.parametrize("make_model, groups", IF_TIES)
    def test_if_outputs_tied(self, make_model, groups):
        """A model with If branches"""
        model = make_model()
        sim = _sim(model)
        qtzrs = sim.qc_quantize_op_dict

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: 1) Each If output shares one quantizer with the branch outputs it takes its value from
              2) Different groups keep different quantizers
              3) After calibrating with every branch taken, each group has one enabled encoding,
                 and the sim runs either way
        """
        # 1)
        for group in groups:
            assert all(qtzrs[name] is qtzrs[group[0]] for name in group)

        # 2)
        assert len({id(qtzrs[group[0]]) for group in groups}) == len(groups)

        # 3)
        sim.compute_encodings(
            [_if_inputs(model, cond=True), _if_inputs(model, cond=False)]
        )
        for group in groups:
            assert qtzrs[group[0]].enabled
            assert qtzrs[group[0]].get_encodings()
        for cond in (True, False):
            sim.session.run(None, _if_inputs(model, cond=cond))

    @pytest.mark.parametrize(
        "then_op, else_op",
        [
            ("Relu", "Neg"),
            ("Sigmoid", "Sigmoid"),
            ("Sigmoid", "Softmax"),
            ("Softmax", "Neg"),
            ("Softmax", "Relu"),
            ("Tanh", "Sigmoid"),
            ("Sin", "Tanh"),
        ],
    )
    def test_if_tied_as_concat(self, then_op, else_op):
        """An If whose branches end in then_op and else_op, and a Concat of the same two ops"""
        if_sim = _sim(
            models_for_tests.if_model(then_op=then_op, else_op=else_op),
            config_file="htp_v73",
        )
        concat_sim = _sim(
            models_for_tests.if_as_concat_model(then_op=then_op, else_op=else_op),
            config_file="htp_v73",
        )

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: The If output and the branch outputs share quantizers as the Concat output and its
              inputs do, with the same fixed range and symmetry. An If output takes its value
              from either branch, as a Concat output does from each of its inputs
        """
        names = ["x", "thn_o", "els_o", "y"]
        assert _groups(if_sim, names) == _groups(concat_sim, names)
        for name in names:
            if_qtzr = if_sim.qc_quantize_op_dict[name]
            concat_qtzr = concat_sim.qc_quantize_op_dict[name]
            assert (
                if_qtzr._encoding_min_max_fixed_vals
                == concat_qtzr._encoding_min_max_fixed_vals
            )
            assert (
                if_qtzr.use_symmetric_encodings == concat_qtzr.use_symmetric_encodings
            )

    def test_if_tie_keeps_constant_branch_unclamped(self):
        """An If whose then branch ends in Relu, constrained to min=0 by htp_v73, and whose else
        branch returns a negative constant, which has no quantizer"""
        model = models_for_tests.if_constant_branch_model()
        sim = _sim(model, config_file="htp_v73")
        qtzrs = sim.qc_quantize_op_dict

        """
        When: QuantizationSimModel is created with subgraph quantization, and calibrated
        Then: 1) The Relu output is not tied to the If output, which gets no fixed range
              2) The constant reaches the If output unclamped
        """
        # 1)
        assert qtzrs["thn_o"] is not qtzrs["y"]
        assert not qtzrs["y"]._encoding_min_max_fixed_vals

        # 2)
        x = np.linspace(-1, 1, models_for_tests.DIM).astype(np.float32)
        sim.compute_encodings(
            [{"x": x, "cond": np.array([True])}, {"x": x, "cond": np.array([False])}]
        )
        (else_out,) = sim.session.run(None, {"x": x, "cond": np.array([False])})
        assert np.allclose(else_out, -0.5, atol=0.01)

    def test_if_output_concat_keeps_other_input_unclamped(self):
        """An If whose branches both end in Softmax, constrained to [0, 1] by htp_v73, and whose
        output is concatenated with m = 5x"""
        model = models_for_tests.if_concat_model()
        sim = _sim(model, config_file="htp_v73")
        qtzrs = sim.qc_quantize_op_dict

        """
        When: QuantizationSimModel is created with subgraph quantization, and calibrated
        Then: 1) The Softmax range is not given to the Concat output, which m shares
              2) m reaches the Concat output unclamped
        """
        # 1)
        assert qtzrs["m"] is qtzrs["c"]
        assert not qtzrs["c"]._encoding_min_max_fixed_vals

        # 2)
        x = np.linspace(-1, 1, models_for_tests.DIM).astype(np.float32)
        sim.compute_encodings(
            [{"x": x, "cond": np.array([True])}, {"x": x, "cond": np.array([False])}]
        )
        (out,) = sim.session.run(None, {"x": x, "cond": np.array([True])})
        assert np.allclose(out[models_for_tests.DIM :], 5 * x, atol=0.05)

    def test_multi_output_if_tied_per_output(self):
        """An If with two outputs: ya from Sigmoid in either branch, yb from Neg or Abs"""
        sim = _sim(models_for_tests.multi_output_if_model(), config_file="htp_v73")
        qtzrs = sim.qc_quantize_op_dict

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: Each If output shares one quantizer with the same output of either branch, and
              takes the range they have in common
        """
        assert qtzrs["t_a"] is qtzrs["e_a"] is qtzrs["ya"]
        assert qtzrs["ya"]._encoding_min_max_fixed_vals == (0.0, 1.0)
        assert qtzrs["t_b"] is qtzrs["e_b"] is qtzrs["yb"]
        assert qtzrs["ya"] is not qtzrs["yb"]

    def test_routed_experts_match_flat_model(self):
        """Predicated experts with every expert routed, and the same experts written without If"""
        predicated = models_for_tests.predicated_experts_model()
        flat = models_for_tests.flat_experts_model()
        rng = np.random.default_rng(1)
        calib = [
            rng.standard_normal((models_for_tests.DIM,) * 2).astype(np.float32)
            for _ in range(4)
        ]
        routed = {"route0": np.array([True]), "route1": np.array([True])}
        predicated_sim, flat_sim = _sim(predicated), _sim(flat)
        # The If ties the running sum across experts, so the flat model does the same
        flat_sim._set_quantizers(
            {"y1": flat_sim.qc_quantize_op_dict["y0"]}, rebuild_session=True
        )

        """
        When: Both sims are calibrated on the same data
        Then: Their outputs are identical
        """
        predicated_sim.compute_encodings([{"x": x, **routed} for x in calib])
        flat_sim.compute_encodings([{"x": x} for x in calib])
        (predicated_out,) = predicated_sim.session.run(None, {"x": calib[0], **routed})
        (flat_out,) = flat_sim.session.run(None, {"x": calib[0]})
        assert np.array_equal(predicated_out, flat_out)

    @pytest.mark.parametrize(
        "config_file, param_type, bitwidth",
        [
            # 16-bit weights of a dynamic MatMul must be symmetric
            ("htp_v73", "int16", 16),
            # Before V73, weights of a dynamic MatMul must be 8-bit and symmetric
            ("htp_v69", "int8", 8),
        ],
    )
    def test_if_tie_keeps_exception_rules_of_if_output(
        self, config_file, param_type, bitwidth
    ):
        """An If whose output is the weight of a dynamic MatMul, with 16-bit activations"""
        kwargs = dict(
            config_file=config_file, param_type=param_type, activation_type="int16"
        )
        sim = _sim(models_for_tests.if_matmul_weight_model(), **kwargs)

        """
        When: QuantizationSimModel is created with subgraph quantization
        Then: The If output keeps the symmetry and bitwidth the exception rules give it
        """
        assert sim.qc_quantize_op_dict["y"].use_symmetric_encodings
        assert sim.qc_quantize_op_dict["y"].bitwidth == bitwidth

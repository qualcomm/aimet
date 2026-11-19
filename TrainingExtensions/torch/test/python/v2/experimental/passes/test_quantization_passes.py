# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from operator import getitem
from functools import partial, wraps
import pytest
import torch
import torch.nn.functional as F

from aimet_torch.quantization.affine import QuantizeDequantize
from aimet_torch.experimental.export.passes import (
    cleanup_duplicate_aimet_qdq_nodes,
    resolve_aimet_qdq_constraints,
)


def make_qdq(
    shape: tuple[int, ...] = (),
    qmin: int = 0,
    qmax: int = 255,
    symmetric: bool = False,
    fixed_range: tuple[float, float]
    | tuple[float, None]
    | tuple[None, float]
    | None = None,
) -> QuantizeDequantize:
    """
    Create QuantizeDequantize module

    If fixed_range is given, the quantizer is initialized with that range and
    its encoding is frozen, i.e. the quantizer is "fixed".
    Otherwise, the quantizer is left uninitialized (unconstrained).
    """
    qdq = QuantizeDequantize(shape, qmin=qmin, qmax=qmax, symmetric=symmetric)

    if fixed_range:
        fixed_min, fixed_max = fixed_range
        qdq.set_range(fixed_min, fixed_max)
        qdq.allow_overwrite(min=fixed_min is None, max=fixed_max is None)

    return qdq


def build_graph_module(
    modules: dict[str, torch.nn.Module],
    nodes: list[dict],
) -> torch.fx.GraphModule:
    """
    Build a GraphModule from a list of node specifications.

    Each node specification is a dictionary with keys:
        - name: name of the node
        - op: operation type ("placeholder", "call_module", "call_function", "output")
        - target: for call_module/call_function, the target module or function
        - inputs: list of input node names
    """
    graph = torch.fx.Graph()
    node_map: dict[str, torch.fx.Node] = {}

    def resolve(arg):
        # str -> node with that name, list -> list of resolved args, others -> literal
        if isinstance(arg, str):
            return node_map[arg]
        if isinstance(arg, list):
            return [resolve(a) for a in arg]
        return arg

    for node_spec in nodes:
        name = node_spec["name"]
        op = node_spec["op"]
        inputs = [resolve(inp) for inp in node_spec.get("inputs", [])]

        if op == "placeholder":
            node_map[name] = graph.placeholder(name)
        elif op == "call_module":
            target = node_spec["target"]
            node = graph.call_module(target, tuple(inputs))
            node.name = name
            node_map[name] = node
        elif op == "call_function":
            target = node_spec["target"]
            node = graph.call_function(target, tuple(inputs))
            node.name = name
            node_map[name] = node
        elif op == "output":
            graph.output(tuple(inputs))
        else:
            raise ValueError(f"Unknown operation type: {op}")

    root = torch.nn.Module()
    for name, module in modules.items():
        root.add_module(name, module)

    gm = torch.fx.GraphModule(root, graph)
    gm.graph.lint()
    return gm


def placeholder(name: str) -> dict:
    return {"name": name, "op": "placeholder"}


def call_module(name: str, target: str, inputs: list[str]) -> dict:
    return {"name": name, "op": "call_module", "target": target, "inputs": list(inputs)}


def call_function(name: str, target, inputs: list) -> dict:
    return {
        "name": name,
        "op": "call_function",
        "target": target,
        "inputs": list(inputs),
    }


def output(inputs: list) -> dict:
    return {"name": "output", "op": "output", "inputs": list(inputs)}


class TestCleanupDuplicateAimetQdqNodes:
    def test_back_to_back_same_module(self):
        """
        Given:
            x --> q1 ---> q2 --> output
                (qdq)   (qdq)

        Then: q2 should be removed
            x --> q1 --> output
                (qdq)
        """
        gm = build_graph_module(
            modules={"qdq": make_qdq()},
            nodes=[
                placeholder("x"),
                call_module(name="q1", target="qdq", inputs=["x"]),
                call_module(name="q2", target="qdq", inputs=["q1"]),
                output(["q2"]),
            ],
        )
        cleanup_duplicate_aimet_qdq_nodes(gm)
        assert [node.name for node in gm.graph.nodes] == ["x", "q1", "output"]
        _, q1_node, output_node = gm.graph.nodes
        assert output_node.args[0] == (q1_node,)

    def test_back_to_back_different_modules(self):
        """
        Given:
            x --> q1 ---> q2 --> output
                (qdq1)  (qdq2)

        Then: Nothing should be removed
        """
        gm = build_graph_module(
            modules={"qdq1": make_qdq(), "qdq2": make_qdq()},
            nodes=[
                placeholder("x"),
                call_module(name="q1", target="qdq1", inputs=["x"]),
                call_module(name="q2", target="qdq2", inputs=["q1"]),
                output(["q2"]),
            ],
        )
        cleanup_duplicate_aimet_qdq_nodes(gm)
        assert [node.name for node in gm.graph.nodes] == ["x", "q1", "q2", "output"]
        *_, q2_node, output_node = gm.graph.nodes
        assert output_node.args[0] == (q2_node,)

    def test_chain_of_three(self):
        """
        Given:
            x --> q1 --> q2 --> q3 --> output
                (qdq)  (qdq)  (qdq)

        Then: Only q1 should be left
            x --> q1 --> output
                (qdq)
        """
        gm = build_graph_module(
            modules={"qdq": make_qdq()},
            nodes=[
                placeholder("x"),
                call_module(name="q1", target="qdq", inputs=["x"]),
                call_module(name="q2", target="qdq", inputs=["q1"]),
                call_module(name="q3", target="qdq", inputs=["q2"]),
                output(["q3"]),
            ],
        )
        cleanup_duplicate_aimet_qdq_nodes(gm)
        assert [node.name for node in gm.graph.nodes] == ["x", "q1", "output"]
        _, q1_node, output_node = gm.graph.nodes
        assert output_node.args[0] == (q1_node,)

    def test_fan_out(self):
        """
        Given:
                       +--> q2 --> ...
            x --> q1 --+  (qdq)
                (qdq)  +--> q3 --> ...
                          (qdq)

        Then: Both q2 and q3 should be removed, and output should consume q1

                       +--> ...
            x --> q1 --+
                (qdq)  +--> ...

        """
        gm = build_graph_module(
            modules={"qdq": make_qdq()},
            nodes=[
                placeholder("x"),
                call_module(name="q1", target="qdq", inputs=["x"]),
                call_module(name="q2", target="qdq", inputs=["q1"]),
                call_module(name="q3", target="qdq", inputs=["q1"]),
                output(["q2", "q3"]),
            ],
        )
        cleanup_duplicate_aimet_qdq_nodes(gm)
        assert [node.name for node in gm.graph.nodes] == ["x", "q1", "output"]
        _, q1_node, output_node = gm.graph.nodes
        assert output_node.args[0] == (q1_node, q1_node)


class TestResolveAimetQdqConstraints:
    """
    Exhuastive tests for all cases in resolve_aimet_qdq_constraints.
    The case number follows those in the dev comment of the function.
    """

    def test_case_0(self):
        """
        Given: qdq1 unconstrained / qdq2 unconstrained (case 0)

            conv2d -> qdq1 -> qdq2 -> softmax

        Then:
            conv2d -> qdq1 -> qdq1 -> softmax
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                call_module(name="conv2d", target="conv2d", inputs=["x"]),
                call_module(name="qdq1_0", target="qdq1", inputs=["conv2d"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_function(
                    name="softmax", target=torch.softmax, inputs=["qdq2_0", -1]
                ),
                output(["softmax"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq2_0, _, _ = gm.graph.nodes
        assert qdq1_0.target == qdq2_0.target == "qdq1"

    def test_case_1a(self):
        """
        Given:
          - qdq1 unconstrained
          - qdq2 tied
          - qdq1 is the only input provider of qdq2 (case 1a)

                             +-> ...
                             |                  +-> qdq2
             conv2d -> qdq1 -+-> qdq2 -> split -+
                                                +-> qdq2

        Then:
                             +-> ...
                             |                  +-> qdq1
             conv2d -> qdq1 -+-> qdq1 -> split -+
                                                +-> qdq1
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                call_module(name="conv2d", target="conv2d", inputs=["x"]),
                call_module(name="qdq1_0", target="qdq1", inputs=["conv2d"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_function(name="split", target=torch.split, inputs=["qdq2_0", 1]),
                call_function(name="split_0", target=getitem, inputs=["split", 0]),
                call_function(name="split_1", target=getitem, inputs=["split", 1]),
                call_module(name="qdq2_1", target="qdq2", inputs=["split_0"]),
                call_module(name="qdq2_2", target="qdq2", inputs=["split_1"]),
                output(["qdq1_0", "qdq2_1", "qdq2_2"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq2_0, _, _, _, qdq2_1, qdq2_2, _ = gm.graph.nodes
        assert (
            qdq1_0.target == qdq2_0.target == qdq2_1.target == qdq2_2.target == "qdq1"
        )

    def test_case_1b(self):
        """
        Given:
          - qdq1 unconstrained
          - qdq2 tied
          - qdq2 is the only user of qdq1 (case 1b)

            conv2d -> qdq1 -> qdq2 -+
                                    +-> cat -> qdq2 -> ...
                       ... -> qdq2 -+


        Then:
            conv2d -> qdq2 -> qdq2 -+
                                    +-> cat -> qdq2 -> ...
                       ... -> qdq2 -+
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                placeholder("dots"),
                call_module(name="conv2d", target="conv2d", inputs=["x"]),
                call_module(name="qdq1_0", target="qdq1", inputs=["conv2d"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["dots"]),
                call_function(
                    name="cat", target=torch.cat, inputs=[["qdq2_0", "qdq2_1"]]
                ),
                call_module(name="qdq2_2", target="qdq2", inputs=["cat"]),
                output(["qdq2_2"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, _, qdq1_0, qdq2_0, qdq2_1, _, qdq2_2, _ = gm.graph.nodes
        assert (
            qdq1_0.target == qdq2_0.target == qdq2_1.target == qdq2_2.target == "qdq2"
        )

    def test_case_1c(self):
        """
        Given:
          - qdq1 unconstrained
          - qdq2 tied
          - qdq2 isn't the only user of qdq1 (case 1c)

                            +-> qdq3 -> softmax
                            |
            conv2d -> qdq1 -+-> qdq2 -+
                                      +-> cat -> qdq2 -> ...
                         ... -> qdq2 -+


        Then:
                            +-> qdq1 -> softmax
                            |
            conv2d -> qdq1 -+-> qdq2 -+
                                      +-> cat -> qdq2 -> ...
                         ... -> qdq2 -+
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(),
                "qdq3": make_qdq(),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                placeholder("dots"),
                call_module(name="conv2d", target="conv2d", inputs=["x"]),
                call_module(name="qdq1_0", target="qdq1", inputs=["conv2d"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["qdq1_0"]),
                call_function(
                    name="softmax", target=torch.softmax, inputs=["qdq3_0", -1]
                ),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["dots"]),
                call_function(
                    name="cat", target=torch.cat, inputs=[["qdq2_0", "qdq2_1"]]
                ),
                call_module(name="qdq2_2", target="qdq2", inputs=["cat"]),
                output(["softmax", "qdq2_2"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, _, qdq1_0, qdq3_0, _, qdq2_0, qdq2_1, _, qdq2_2, _ = gm.graph.nodes
        assert qdq1_0.target == qdq3_0.target == "qdq1"
        assert qdq2_0.target == qdq2_1.target == qdq2_2.target == "qdq2"

    def test_case_2a(self):
        """
        Given:
          - qdq1 tied
          - qdq2 unconstrained
          - qdq1 is the only input provider of qdq2 (case 2a)

                           +-> qdq1 -> ...
            qdq1 -> split -+
                           +-> qdq1 -> qdq2 -> conv2d

        Then:
                           +-> qdq1 -> ...
            qdq1 -> split -+
                           +-> qdq1 -> qdq1 -> conv2d
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_function(name="split", target=torch.split, inputs=["qdq1_0", 1]),
                call_function(name="split_0", target=getitem, inputs=["split", 0]),
                call_function(name="split_1", target=getitem, inputs=["split", 1]),
                call_module(name="qdq1_1", target="qdq1", inputs=["split_0"]),
                call_module(name="qdq1_2", target="qdq1", inputs=["split_1"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_2"]),
                call_module(name="conv2d", target="conv2d", inputs=["qdq2_0"]),
                output(["qdq1_1", "conv2d"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, qdq1_0, _, _, _, qdq1_1, qdq1_2, qdq2_0, _, _ = gm.graph.nodes
        assert (
            qdq1_0.target == qdq1_1.target == qdq1_2.target == qdq2_0.target == "qdq1"
        )

    def test_case_2b(self):
        """
        Given:
          - qdq1 tied
          - qdq2 unconstrained
          - qdq2 is the only user of qdq1 (case 2b)

            qdq1 -+
                  +-> cat -> qdq1 -> qdq2 -> conv2d
            qdq1 -+

        Then:
            qdq1 -+
                  +-> cat -> qdq1 -> qdq1 -> conv2d
            qdq1 -+
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_module(name="qdq1_1", target="qdq1", inputs=["y"]),
                call_function(
                    name="cat", target=torch.cat, inputs=[["qdq1_0", "qdq1_1"]]
                ),
                call_module(name="qdq1_2", target="qdq1", inputs=["cat"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_2"]),
                call_module(name="conv2d", target="conv2d", inputs=["qdq2_0"]),
                output(["conv2d"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq1_1, _, qdq1_2, qdq2_0, _, _ = gm.graph.nodes
        assert (
            qdq1_0.target == qdq1_1.target == qdq1_2.target == qdq2_0.target == "qdq1"
        )

    def test_case_3a(self):
        """
        Given:
          - qdq1 tied
          - qdq2 tied
          - qdq1 is the only input provider of qdq2 (case 3a)

                                                      +-> qdq2
            qdq1 -> maxpool -> qdq1 -> qdq2 -> split -+
                                                      +-> qdq2

        Then:
                                                      +-> qdq1
            qdq1 -> maxpool -> qdq1 -> qdq1 -> split -+
                                                      +-> qdq1
        """
        gm = build_graph_module(
            modules={"qdq1": make_qdq(), "qdq2": make_qdq()},
            nodes=[
                placeholder("x"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_function(
                    name="maxpool", target=F.max_pool2d, inputs=["qdq1_0", 2]
                ),
                call_module(name="qdq1_1", target="qdq1", inputs=["maxpool"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_1"]),
                call_function(name="split", target=torch.split, inputs=["qdq2_0", 1]),
                call_function(name="split_0", target=getitem, inputs=["split", 0]),
                call_function(name="split_1", target=getitem, inputs=["split", 1]),
                call_module(name="qdq2_1", target="qdq2", inputs=["split_0"]),
                call_module(name="qdq2_2", target="qdq2", inputs=["split_1"]),
                output(["qdq2_1", "qdq2_2"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, qdq1_0, _, qdq1_1, qdq2_0, _, _, _, qdq2_1, qdq2_2, _ = gm.graph.nodes
        assert (
            qdq1_0.target
            == qdq1_1.target
            == qdq2_0.target
            == qdq2_1.target
            == qdq2_2.target
            == "qdq1"
        )

    def test_case_3b(self):
        """
        Given:
          - qdq1 tied
          - qdq2 tied
          - qdq2 is the only user of qdq1 (case 3b)

            qdq1 -> maxpool -> qdq1 -> qdq2 -+
                                             +-> cat -> qdq2 ->
                                ... -> qdq2 -+

        Then:
            qdq2 -> maxpool -> qdq2 -> qdq2 -+
                                             +-> cat -> qdq2 ->
                                ... -> qdq2 -+
        """
        gm = build_graph_module(
            modules={"qdq1": make_qdq(), "qdq2": make_qdq()},
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_function(
                    name="maxpool", target=F.max_pool2d, inputs=["qdq1_0", 2]
                ),
                call_module(name="qdq1_1", target="qdq1", inputs=["maxpool"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_1"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["y"]),
                call_function(
                    name="cat", target=torch.cat, inputs=[["qdq2_0", "qdq2_1"]]
                ),
                call_module(name="qdq2_2", target="qdq2", inputs=["cat"]),
                output(["qdq2_2"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, _, qdq1_1, qdq2_0, qdq2_1, _, qdq2_2, _ = gm.graph.nodes
        assert (
            qdq1_0.target
            == qdq1_1.target
            == qdq2_0.target
            == qdq2_1.target
            == qdq2_2.target
            == "qdq2"
        )

    def test_case_3c(self):
        """
        Given:
          - qdq1 tied
          - qdq2 tied
          - qdq1 isn't the only input provider of qdq2, nor is qdq2 the only user of qdq1 (case 3c)

                           +-> qdq1 -> ...
            qdq1 -> split -+
                           +-> qdq1 -> qdq2 -+
                                             +-> cat -> qdq2 ->
                                ... -> qdq2 -+

        Then: Nothing changes

                           +-> qdq1 -> ...
            qdq1 -> split -+
                           +-> qdq1 -> qdq2 -+
                                             +-> cat -> qdq2 ->
                                ... -> qdq2 -+
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(),
            },
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_function(name="split", target=torch.split, inputs=["qdq1_0", 1]),
                call_function(name="split_0", target=getitem, inputs=["split", 0]),
                call_function(name="split_1", target=getitem, inputs=["split", 1]),
                call_module(name="qdq1_1", target="qdq1", inputs=["split_0"]),
                call_module(name="qdq1_2", target="qdq1", inputs=["split_1"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_2"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["y"]),
                call_function(
                    name="cat", target=torch.cat, inputs=[["qdq2_0", "qdq2_1"]]
                ),
                call_module(name="qdq2_2", target="qdq2", inputs=["cat"]),
                output(["qdq1_1", "qdq2_2"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        (
            _,
            _,
            qdq1_0,
            _,
            _,
            _,
            qdq1_1,
            qdq1_2,
            qdq2_0,
            qdq2_1,
            _,
            qdq2_2,
            _,
        ) = gm.graph.nodes
        assert qdq1_0.target == qdq1_1.target == qdq1_2.target == "qdq1"
        assert qdq2_0.target == qdq2_1.target == qdq2_2.target == "qdq2"

    def test_case_4a(self):
        """
        Given:
          - qdq1 unconstrained
          - qdq2 fixed
          - All users of qdq1 are fixed to the same range as qdq2 (case 4a)

                               [-1, 1]
                            +-> qdq3 -> asin -> ...
            conv2d -> qdq1 -+-> qdq2 -> asin -> ...
                     [-1, 1]   [-1, 1]

        Then:
                               [-1, 1]
                            +-> qdq2 -> asin -> ...
            conv2d -> qdq2 -+-> qdq2 -> asin -> ...
                               [-1, 1]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(fixed_range=(-1.0, 1.0)),
                "qdq3": make_qdq(fixed_range=(-1.0, 1.0)),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                call_module(name="conv2d", target="conv2d", inputs=["x"]),
                call_module(name="qdq1_0", target="qdq1", inputs=["conv2d"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_function(name="asin_0", target=torch.asin, inputs=["qdq2_0"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["qdq1_0"]),
                call_function(name="asin_1", target=torch.asin, inputs=["qdq3_0"]),
                output(["asin_0", "asin_1"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq2_0, _, qdq3_0, _, _ = gm.graph.nodes
        assert qdq1_0.target == qdq2_0.target == qdq3_0.target == "qdq2"

    def test_case_4b(self):
        """
        Given:
          - qdq1 unconstrained
          - qdq2 fixed
          - Not all users of qdq1 are fixed to the same range as qdq2 (case 4b)

                            +-> qdq3 -> softmax
                            |
            conv2d -> qdq1 -+-> qdq2 -> asin
                               [-1, 1]

        Then:
                            +-> qdq1 -> softmax
                            |
            conv2d -> qdq1 -+-> qdq2 -> asin
                               [-1, 1]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(fixed_range=(-1.0, 1.0)),
                "qdq3": make_qdq(),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                call_module(name="conv2d", target="conv2d", inputs=["x"]),
                call_module(name="qdq1_0", target="qdq1", inputs=["conv2d"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["qdq1_0"]),
                call_function(
                    name="softmax", target=torch.softmax, inputs=["qdq3_0", -1]
                ),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_function(name="asin", target=torch.asin, inputs=["qdq2_0"]),
                output(["softmax", "asin"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq3_0, _, qdq2_0, _, _ = gm.graph.nodes
        assert qdq1_0.target == qdq3_0.target == "qdq1"
        assert qdq2_0.target == "qdq2"

    def test_case_5a(self):
        """
        Given:
          - qdq1 unconstrained
          - qdq2 fixed & tied
          - All users of qdq1 are fixed to the same range as qdq2 (case 5a)

                               [0, ?]
                            +-> qdq3 -> relu -> ...
            conv2d -> qdq1 -+-> qdq2 -> relu -> qdq2 -> ...
                               [0, ?]          [0, ?]

        Then:
                               [0, ?]
                            +-> qdq2 -> relu -> ...
            conv2d -> qdq2 -+-> qdq2 -> relu -> qdq2 -> ...
                     [0, ?]    [0, ?]          [0, ?]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(fixed_range=(0.0, None)),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                call_module(name="conv2d", target="conv2d", inputs=["x"]),
                call_module(name="qdq1_0", target="qdq1", inputs=["conv2d"]),
                call_module(name="qdq2_2", target="qdq2", inputs=["qdq1_0"]),
                call_function(name="relu_0", target=torch.relu, inputs=["qdq2_2"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_function(name="relu_1", target=torch.relu, inputs=["qdq2_0"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["relu_1"]),
                output(["relu_0", "qdq2_1"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq2_2, _, qdq2_0, _, qdq2_1, _ = gm.graph.nodes
        assert (
            qdq1_0.target == qdq2_2.target == qdq2_0.target == qdq2_1.target == "qdq2"
        )

    def test_case_5b(self):
        """
        Given:
          - qdq1 unconstrained
          - qdq2 fixed & tied
          - Not all users of qdq1 are fixed to the same range as qdq2 (case 5b)

                            +-> qdq3 -> softmax
                            |
            conv2d -> qdq1 -+-> qdq2 -> relu -> qdq2 -> ...
                               [0, ?]          [0, ?]

        Then:
                            +-> qdq1 -> softmax
                            |
            conv2d -> qdq1 -+-> qdq2 -> relu -> qdq2 -> ...
                               [0, ?]          [0, ?]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(fixed_range=(0.0, None)),
                "qdq3": make_qdq(),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                call_module(name="conv2d", target="conv2d", inputs=["x"]),
                call_module(name="qdq1_0", target="qdq1", inputs=["conv2d"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["qdq1_0"]),
                call_function(
                    name="softmax", target=torch.softmax, inputs=["qdq3_0", -1]
                ),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_function(name="relu", target=torch.relu, inputs=["qdq2_0"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["relu"]),
                output(["softmax", "qdq2_1"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq3_0, _, qdq2_0, _, qdq2_1, _ = gm.graph.nodes
        assert qdq1_0.target == qdq3_0.target == "qdq1"
        assert qdq2_0.target == qdq2_1.target == "qdq2"

    def test_case_6a(self):
        """
        Given:
          - qdq1 tied
          - qdq2 fixed
          - All users of qdq1 are fixed to the same range as qdq2 (case 6a)

                                         [-1, 1]
            qdq1 -+                   +-> qdq3 -> asin
                  +-> concat -> qdq1 -+-> qdq2 -> asin
            qdq1 -+                      [-1, 1]

        Then:
           [-1, 1]                       [-1, 1]
            qdq2 -+                   +-> qdq2 -> asin
                  +-> concat -> qdq2 -+-> qdq2 -> asin
            qdq2 -+            [-1, 1]   [-1, 1]
           [-1, 1]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(fixed_range=(-1.0, 1.0)),
                "qdq3": make_qdq(fixed_range=(-1.0, 1.0)),
            },
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_module(name="qdq1_1", target="qdq1", inputs=["y"]),
                call_function(
                    name="concat", target=torch.cat, inputs=[["qdq1_0", "qdq1_1"]]
                ),
                call_module(name="qdq1_2", target="qdq1", inputs=["concat"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["qdq1_2"]),
                call_function(name="asin_0", target=torch.asin, inputs=["qdq3_0"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_2"]),
                call_function(name="asin_1", target=torch.asin, inputs=["qdq2_0"]),
                output(["asin_0", "asin_1"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq1_1, _, qdq1_2, qdq3_0, _, qdq2_0, _, _ = gm.graph.nodes
        # Either qdq2 or qdq3 can be the winner, depending on the order of graph traversal
        assert (
            qdq1_0.target
            == qdq1_1.target
            == qdq1_2.target
            == qdq3_0.target
            == qdq2_0.target
            in ("qdq2", "qdq3")
        )

    def test_case_6b(self):
        """
        Given:
          - qdq1 tied
          - qdq2 fixed
          - Not all users of qdq1 are fixed to the same range as qdq2 (case 6b)

            qdq1 -+                   +-> qdq3 -> softmax
                  +-> concat -> qdq1 -+-> qdq2 -> asin
            qdq1 -+                      [-1, 1]

        Then:
            qdq1 -+                   +-> qdq1 -> softmax
                  +-> concat -> qdq1 -+-> qdq2 -> asin
            qdq1 -+                      [-1, 1]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(fixed_range=(-1.0, 1.0)),
                "qdq3": make_qdq(),
            },
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_module(name="qdq1_1", target="qdq1", inputs=["y"]),
                call_function(
                    name="concat", target=torch.cat, inputs=[["qdq1_0", "qdq1_1"]]
                ),
                call_module(name="qdq1_2", target="qdq1", inputs=["concat"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["qdq1_2"]),
                call_function(
                    name="softmax", target=torch.softmax, inputs=["qdq3_0", -1]
                ),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_2"]),
                call_function(name="asin", target=torch.asin, inputs=["qdq2_0"]),
                output(["softmax", "asin"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq1_1, _, qdq1_2, qdq3_0, _, qdq2_0, _, _ = gm.graph.nodes
        assert (
            qdq1_0.target == qdq1_1.target == qdq1_2.target == qdq3_0.target == "qdq1"
        )
        assert qdq2_0.target == "qdq2"

    def test_case_7a(self):
        """
        Given:
          - qdq1 tied
          - qdq2 fixed & tied
          - All users of qdq1 are fixed to the same range as qdq2 (case 7a)

                                         [0, ?]
            qdq1 -+                   +-> qdq3 -> relu
                  +-> concat -> qdq1 -+-> qdq2 -> relu -> qdq2
            qdq1 -+                      [0, ?]          [0, ?]

        Then:
                                         [0, ?]
            qdq2 -+                   +-> qdq2 -> relu -> ...
                  +-> concat -> qdq2 -+-> qdq2 -> relu -> qdq2
            qdq2 -+                      [0, ?]          [0, ?]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(fixed_range=(0.0, None)),
            },
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_module(name="qdq1_1", target="qdq1", inputs=["y"]),
                call_function(
                    name="concat", target=torch.cat, inputs=[["qdq1_0", "qdq1_1"]]
                ),
                call_module(name="qdq1_2", target="qdq1", inputs=["concat"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_2"]),
                call_function(name="relu_0", target=torch.relu, inputs=["qdq2_0"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["relu_0"]),
                call_module(name="qdq2_2", target="qdq2", inputs=["qdq1_2"]),
                call_function(name="relu_1", target=torch.relu, inputs=["qdq2_2"]),
                output(["qdq2_1", "relu_1"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq1_1, _, qdq1_2, qdq2_0, _, qdq2_1, qdq2_2, _, _ = (
            gm.graph.nodes
        )
        assert (
            qdq1_0.target
            == qdq1_1.target
            == qdq1_2.target
            == qdq2_0.target
            == qdq2_1.target
            == qdq2_2.target
            == "qdq2"
        )

    def test_case_7b(self):
        """
        Given:
          - qdq1 tied
          - qdq2 fixed & tied
          - Not all users of qdq1 are fixed to the same range as qdq2 (case 7b)


            qdq1 -+                   +-> qdq3 -> softmax
                  +-> concat -> qdq1 -+-> qdq2 -> relu -> qdq2
            qdq1 -+                      [0, ?]          [0, ?]

        Then:
            qdq1 -+                   +-> qdq1 -> softmax
                  +-> concat -> qdq1 -+-> qdq2 -> relu -> qdq2
            qdq1 -+                      [0, ?]          [0, ?]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(),
                "qdq2": make_qdq(fixed_range=(0.0, None)),
                "qdq3": make_qdq(),
            },
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_module(name="qdq1_1", target="qdq1", inputs=["y"]),
                call_function(
                    name="concat", target=torch.cat, inputs=[["qdq1_0", "qdq1_1"]]
                ),
                call_module(name="qdq1_2", target="qdq1", inputs=["concat"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["qdq1_2"]),
                call_function(
                    name="softmax", target=torch.softmax, inputs=["qdq3_0", -1]
                ),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_2"]),
                call_function(name="relu", target=torch.relu, inputs=["qdq2_0"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["relu"]),
                output(["softmax", "qdq2_1"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq1_1, _, qdq1_2, qdq3_0, _, qdq2_0, _, qdq2_1, _ = (
            gm.graph.nodes
        )
        assert (
            qdq1_0.target == qdq1_1.target == qdq1_2.target == qdq3_0.target == "qdq1"
        )
        assert qdq2_0.target == qdq2_1.target == "qdq2"

    def test_case_8a(self):
        """
        Given: qdq1 fixed / qdq2 unconstrained (case 8)

            softmax -> qdq1 -> qdq2 -> conv
                      [0, 1]

        Then:
            softmax -> qdq1 -> qdq2 -> conv
                      [0, 1]  [0, 1]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(fixed_range=(0.0, 1.0)),
                "qdq2": make_qdq(),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                call_function(name="softmax", target=torch.softmax, inputs=["x", -1]),
                call_module(name="qdq1_0", target="qdq1", inputs=["softmax"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_module(name="conv2d", target="conv2d", inputs=["qdq2_0"]),
                output(["conv2d"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq2_0, _, _ = gm.graph.nodes
        assert qdq1_0.target == qdq2_0.target == "qdq1"

    def test_case_9a(self):
        """
        Given:
          - qdq1 fixed
          - qdq2 tied
          - All inputs of qdq2 shares the same encoding as qdq1 (case 9a)

                        [0, 1]
              softmax -> qdq1 -> qdq2 -+
                                       +-> cat -> qdq2 -> ...
              sigmoid -> qdq3 -> qdq2 -+
                        [0, 1]

        Then: Either qdq1 or qdq3 wins, depending on the order of graph traversal

                        [0, 1]  [0, 1]
              softmax -> qdq1 -> qdq1 -+
                                       +-> cat -> qdq1 -> ...
              sigmoid -> qdq1 -> qdq1 -+         [0, 1]
                        [0, 1]  [0, 1]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(fixed_range=(0.0, 1.0)),
                "qdq2": make_qdq(),
                "qdq3": make_qdq(fixed_range=(0.0, 1.0)),
            },
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_function(name="softmax", target=torch.softmax, inputs=["x", -1]),
                call_module(name="qdq1_0", target="qdq1", inputs=["softmax"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_function(name="sigmoid", target=torch.sigmoid, inputs=["y"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["sigmoid"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["qdq3_0"]),
                call_function(
                    name="cat", target=torch.cat, inputs=[["qdq2_0", "qdq2_1"]]
                ),
                call_module(name="qdq2_2", target="qdq2", inputs=["cat"]),
                output(["qdq2_2"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, _, qdq1_0, qdq2_0, _, qdq3_0, qdq2_1, _, qdq2_2, _ = gm.graph.nodes
        # Either qdq1 or qdq3 can be the winner, depending on the order of graph traversal
        assert (
            qdq1_0.target
            == qdq2_0.target
            == qdq3_0.target
            == qdq2_1.target
            == qdq2_2.target
            in ("qdq1", "qdq3")
        )

    def test_case_9b(self):
        """
        Given: case 9b
          - qdq1 fixed
          - qdq2 tied
          - Not all inputs of qdq2 shares the same encoding as qdq1 (case 9b)

                        [0, 1]
              softmax -> qdq1 -> qdq2 -+
                                       +-> cat -> qdq2 -> ...
               conv2d -> qdq3 -> qdq2 -+

        Then: qdq1 and qdq2 are not merged.
              qdq3 is unconstrained, so it is replaced with qdq2

                        [0, 1]
              softmax -> qdq1 -> qdq2 -+
                                       +-> cat -> qdq2 -> ...
               conv2d -> qdq2 -> qdq2 -+
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(fixed_range=(0.0, 1.0)),
                "qdq2": make_qdq(),
                "qdq3": make_qdq(fixed_range=(-1.0, 1.0)),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_function(name="softmax", target=torch.softmax, inputs=["x", -1]),
                call_module(name="qdq1_0", target="qdq1", inputs=["softmax"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_module(name="conv2d", target="conv2d", inputs=["y"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["conv2d"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["qdq3_0"]),
                call_function(
                    name="cat", target=torch.cat, inputs=[["qdq2_0", "qdq2_1"]]
                ),
                call_module(name="qdq2_2", target="qdq2", inputs=["cat"]),
                output(["qdq2_2"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, _, qdq1_0, qdq2_0, _, _, qdq2_1, _, qdq2_2, _ = gm.graph.nodes
        assert qdq1_0.target == "qdq1"
        assert qdq2_0.target == qdq2_1.target == qdq2_2.target == "qdq2"

    def test_case_10a(self):
        """
        Given: qdq1 tied & fixed / qdq2 unconstrained (case 10)

            qdq1 -> relu -> qdq1 -> qdq2 -> conv
           [0, ?]          [0, ?]

        Then:
            qdq1 -> relu -> qdq1 -> qdq2 -> conv
           [0, ?]          [0, ?]  [0, ?]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(fixed_range=(0.0, None)),
                "qdq2": make_qdq(),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_function(name="relu", target=torch.relu, inputs=["qdq1_0"]),
                call_module(name="qdq1_1", target="qdq1", inputs=["relu"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_1"]),
                call_module(name="conv2d", target="conv2d", inputs=["qdq2_0"]),
                output(["conv2d"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, qdq1_0, _, qdq1_1, qdq2_0, _, _ = gm.graph.nodes
        assert qdq1_0.target == qdq1_1.target == qdq2_0.target == "qdq1"

    def test_case_11a(self):
        """
        Given:
          - qdq1 fixed & tied
          - qdq2 tied
          - All inputs of qdq2 shares the same encoding as qdq1 (case 11a)

             [0, 1]             [0, 1]
              qdq1 -> maxpool -> qdq1 -> qdq2 -+
                                               +-> cat -> qdq2 -> ...
              qdq3 -> maxpool -> qdq3 -> qdq2 -+
             [0, 1]             [0, 1]

        Then: Either qdq1 or qdq3 wins, depending on the order of graph traversal

             [0, 1]             [0, 1]  [0, 1]
              qdq1 -> maxpool -> qdq1 -> qdq1 -+
                                               +-> cat -> qdq1 -> ...
              qdq1 -> maxpool -> qdq1 -> qdq1 -+         [0, 1]
             [0, 1]             [0, 1]  [0, 1]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(fixed_range=(0.0, 1.0)),
                "qdq2": make_qdq(),
                "qdq3": make_qdq(fixed_range=(0.0, 1.0)),
            },
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_function(
                    name="maxpool_0", target=F.max_pool2d, inputs=["qdq1_0", 2]
                ),
                call_module(name="qdq1_1", target="qdq1", inputs=["maxpool_0"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["y"]),
                call_function(
                    name="maxpool_1", target=F.max_pool2d, inputs=["qdq3_0", 2]
                ),
                call_module(name="qdq3_1", target="qdq3", inputs=["maxpool_1"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_1"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["qdq3_1"]),
                call_function(
                    name="cat", target=torch.cat, inputs=[["qdq2_0", "qdq2_1"]]
                ),
                call_module(name="qdq2_2", target="qdq2", inputs=["cat"]),
                output(["qdq2_2"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        (
            _,
            _,
            qdq1_0,
            _,
            qdq1_1,
            qdq3_0,
            _,
            qdq3_1,
            qdq2_0,
            qdq2_1,
            _,
            qdq2_2,
            _,
        ) = gm.graph.nodes
        # Either qdq1 or qdq3 can be the winner, depending on the order of graph traversal
        assert (
            qdq1_0.target
            == qdq1_1.target
            == qdq3_0.target
            == qdq3_1.target
            == qdq2_0.target
            == qdq2_1.target
            == qdq2_2.target
            in ("qdq1", "qdq3")
        )

    def test_case_11b(self):
        """
        Given:
          - qdq1 fixed & tied
          - qdq2 tied
          - Not all inputs of qdq2 shares the same encoding as qdq1 (case 11b)

             [0, ?]          [0, ?]
              qdq1 -> relu -> qdq1 -> qdq2 -+
                                            +-> cat -> qdq2 -> ...
              qdq3 -> relu -> qdq3 -> qdq2 -+
             [0, ?]          [0, ?]

             (Note that qdq1 and qdq3 don't share the same encoding. "?" is just a placeholder)

        Then: Nothing changes

             [0, ?]          [0, ?]
              qdq1 -> relu -> qdq1 -> qdq2 -+
                                            +-> cat -> qdq2 -> ...
              qdq3 -> relu -> qdq3 -> qdq2 -+
             [0, ?]          [0, ?]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(fixed_range=(0.0, None)),
                "qdq2": make_qdq(),
                "qdq3": make_qdq(fixed_range=(-1.0, 1.0)),
                "conv2d": torch.nn.Conv2d(1, 1, 1),
            },
            nodes=[
                placeholder("x"),
                placeholder("y"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_function(name="relu", target=torch.relu, inputs=["qdq1_0"]),
                call_module(name="qdq1_1", target="qdq1", inputs=["relu"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_1"]),
                call_module(name="conv2d", target="conv2d", inputs=["y"]),
                call_module(name="qdq3_0", target="qdq3", inputs=["conv2d"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["qdq3_0"]),
                call_function(
                    name="cat", target=torch.cat, inputs=[["qdq2_0", "qdq2_1"]]
                ),
                call_module(name="qdq2_2", target="qdq2", inputs=["cat"]),
                output(["qdq2_2"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, _, qdq1_1, qdq2_0, _, qdq3_0, qdq2_1, _, qdq2_2, _ = (
            gm.graph.nodes
        )
        assert qdq1_0.target == qdq1_1.target == "qdq1"
        assert qdq2_0.target == qdq2_1.target == qdq2_2.target == "qdq2"

    def test_case_12(self):
        """
        Given: qdq1 fixed / qdq2 fixed (case 12)

              softmax -> qdq1 --> qdq2 -> asin -> ...
                        [0, 1]  [-1, 1]

        Then: Nothing changes, since qdq1 and qdq2 are fixed to different ranges

              softmax -> qdq1 --> qdq2 -> asin -> ...
                        [0, 1]  [-1, 1]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(fixed_range=(0.0, 1.0)),
                "qdq2": make_qdq(fixed_range=(-1.0, 1.0)),
            },
            nodes=[
                placeholder("x"),
                call_function(name="softmax", target=torch.softmax, inputs=["x", -1]),
                call_module(name="qdq1_0", target="qdq1", inputs=["softmax"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_function(name="asin", target=torch.asin, inputs=["qdq2_0"]),
                output(["asin"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq2_0, _, _ = gm.graph.nodes
        assert qdq1_0.target == "qdq1"
        assert qdq2_0.target == "qdq2"

    def test_case_13(self):
        """
        Given: qdq1 fixed / qdq2 fixed & tied (case 13)

              softmax -> qdq1 --> qdq2 -> relu -> qdq2
                        [0, 1]   [0, ?]          [0, ?]

        Then: Nothing changes, since qdq1 and qdq2 are fixed to different ranges

              softmax -> qdq1 --> qdq2 -> relu -> qdq2
                        [0, 1]   [0, ?]          [0, ?]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(fixed_range=(0.0, 1.0)),
                "qdq2": make_qdq(fixed_range=(0.0, None)),
            },
            nodes=[
                placeholder("x"),
                call_function(name="softmax", target=torch.softmax, inputs=["x", -1]),
                call_module(name="qdq1_0", target="qdq1", inputs=["softmax"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_0"]),
                call_function(name="relu", target=torch.relu, inputs=["qdq2_0"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["relu"]),
                output(["qdq2_1"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, _, qdq1_0, qdq2_0, _, qdq2_1, _ = gm.graph.nodes
        assert qdq1_0.target == "qdq1"
        assert qdq2_0.target == qdq2_1.target == "qdq2"

    def test_case_14(self):
        """
        Given: qdq1 fixed & tied / qdq2 fixed (case 14)

              qdq1 -> relu -> qdq1 --> qdq2 -> asin
             [0, ?]          [0, ?]   [-1, 1]

        Then: Nothing changes, since qdq1 and qdq2 are fixed to different ranges

              qdq1 -> relu -> qdq1 --> qdq2 -> asin
             [0, ?]          [0, ?]   [-1, 1]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(fixed_range=(0.0, None)),
                "qdq2": make_qdq(fixed_range=(-1.0, 1.0)),
            },
            nodes=[
                placeholder("x"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_function(name="relu", target=torch.relu, inputs=["qdq1_0"]),
                call_module(name="qdq1_1", target="qdq1", inputs=["relu"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_1"]),
                call_function(name="asin", target=torch.asin, inputs=["qdq2_0"]),
                output(["asin"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, qdq1_0, _, qdq1_1, qdq2_0, _, _ = gm.graph.nodes
        assert qdq1_0.target == qdq1_1.target == "qdq1"
        assert qdq2_0.target == "qdq2"

    def test_case_15(self):
        """
        Given: qdq1 fixed & tied / qdq2 fixed & tied (case 15)

              qdq1 -> relu -> qdq1 --> qdq2 -> relu -> qdq2
             [0, ?]          [0, ?]   [0, ?]          [0, ?]

        Then: Nothing changes, since qdq1 and qdq2 are fixed to different ranges

              qdq1 -> relu -> qdq1 --> qdq2 -> relu -> qdq2
             [0, ?]          [0, ?]   [0, ?]          [0, ?]
        """
        gm = build_graph_module(
            modules={
                "qdq1": make_qdq(fixed_range=(0.0, None)),
                "qdq2": make_qdq(fixed_range=(0.0, None)),
            },
            nodes=[
                placeholder("x"),
                call_module(name="qdq1_0", target="qdq1", inputs=["x"]),
                call_function(name="relu_0", target=torch.relu, inputs=["qdq1_0"]),
                call_module(name="qdq1_1", target="qdq1", inputs=["relu_0"]),
                call_module(name="qdq2_0", target="qdq2", inputs=["qdq1_1"]),
                call_function(name="relu_1", target=torch.relu, inputs=["qdq2_0"]),
                call_module(name="qdq2_1", target="qdq2", inputs=["relu_1"]),
                output(["qdq2_1"]),
            ],
        )
        resolve_aimet_qdq_constraints(gm)
        _, qdq1_0, _, qdq1_1, qdq2_0, _, qdq2_1, _ = gm.graph.nodes
        assert qdq1_0.target == qdq1_1.target == "qdq1"
        assert qdq2_0.target == qdq2_1.target == "qdq2"

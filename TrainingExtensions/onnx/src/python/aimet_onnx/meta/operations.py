# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX Operation class and utilities"""

from typing import TYPE_CHECKING, Dict, List, Tuple, Optional
import onnx_ir
from aimet_onnx.common.connected_graph.operation import Op as _Op, OpInformation
from aimet_onnx.ir_utils import _has_transposed_b, get_constant_value, get_weight_value
from aimet_onnx.meta.product import Product

if TYPE_CHECKING:
    from aimet_onnx.meta.connectedgraph import ConnectedGraph

INPUT_INDEX = 0
WEIGHT_INDEX = 1
BIAS_INDEX = 2
RUNNING_MEAN_INDEX = 3
RUNNING_VAR_INDEX = 4


class Op(_Op):
    """
    View of an ``onnx_ir.Node`` in a :class:`ConnectedGraph`.

    Its attributes and links are read from the node on every call. Links map the values of the
    node to the products the connected graph holds for them, so they always return the same
    objects.
    """

    _parameters: Dict[str, Tuple[Product, Optional[str]]]

    def __init__(self, node: onnx_ir.Node, graph: "ConnectedGraph"):  # pylint: disable=super-init-not-called
        """
        :param node: Node this op represents
        :param graph: Connected graph holding the products of the node's values
        """
        # The base initializer assigns the links this class reads from ``node``
        self._node = node
        self._graph = graph
        self._parameters = {
            value.name: (graph.product_of(value), param_type)
            for param_type, value in self._find_parameters().items()
        }
        self._output_shape = None
        self._is_anonymous = False
        self._op_info = OpInformation()

    def _find_parameters(self) -> Dict[str, onnx_ir.Value]:
        """
        Parameters of the node by type, e.g. ``{"weight": W, "bias": B}`` for a Conv. A parameter
        is an initializer or a ``Constant`` tensor, read directly or through Identity nodes (or,
        for a MatMul weight, through a Transpose).
        """
        node = self._node
        if node.op_type == "MatMul":
            weight, _ = get_weight_value(node)
            values = {"weight": weight}
        else:
            values = {
                param_type: get_constant_value(node.inputs[index])
                for param_type, index in _PARAMETER_INDICES.get(
                    node.op_type, {}
                ).items()
                if index < len(node.inputs)
            }
        return {
            param_type: value
            for param_type, value in values.items()
            if value is not None and _holds_tensor(value)
        }

    def __repr__(self):
        return self.name

    @property
    def name(self) -> str:
        return self._node.name

    @property
    def name_op(self) -> str:
        """Name of the node"""
        return self._node.name

    @property
    def dotted_name(self) -> str:
        return self._node.name

    @property
    def type(self) -> str:
        return self._node.op_type

    @property
    def domain(self) -> str:
        """Domain of the node"""
        return self._node.domain

    @property
    def groups(self) -> Optional[int]:
        """The ``group`` attribute of a Conv or ConvTranspose, or None"""
        if self.type in ("Conv", "ConvTranspose"):
            return self._node.attributes.get_int("group")
        return None

    @property
    def transposed_params(self) -> bool:
        """Whether the weight is stored ``[out, in]``"""
        if self.type == "Gemm":
            return _has_transposed_b(self._node)
        if self.type == "MatMul":
            return get_weight_value(self._node)[1]
        return False

    @property
    def inputs(self) -> List[Product]:
        """Products of the inputs of the node, omitted optional inputs skipped"""
        return [
            self._graph.product_of(value)
            for value in self._node.inputs
            if value is not None
        ]

    @property
    def outputs(self) -> List[Product]:
        return [self._graph.product_of(value) for value in self._node.outputs]

    @property
    def input_ops(self) -> List["Op"]:
        return [inp.producer for inp in self.inputs if inp.producer]

    @property
    def output_ops(self) -> List["Op"]:
        return [consumer for output in self.outputs for consumer in output.consumers]

    def get_module(self) -> onnx_ir.Node:
        """The node this op represents"""
        return self._node

    def add_input(self, product: Product):
        raise AttributeError("Op links are read from its node; edit the model instead")

    def add_param(self, param: str, product: Product, product_type: Optional[str]):
        """Add a parameter product to parameters dictionary"""
        self._parameters[param] = (product, product_type)

    @property
    def parameters(self) -> Dict[str, Tuple[Product, Optional[str]]]:
        """returns parameters of the op"""
        return self._parameters

    @property
    def subgraph_ops(self) -> List["Op"]:
        """
        Returns the ops this op runs inside its own body, in the order they run.
        A control-flow op carries its body as a graph-valued attribute.
        """
        return [op for ops in self.subgraphs.values() for op in ops]

    @property
    def subgraphs(self) -> Dict[str, List["Op"]]:
        """
        Returns the ops of each of this op's bodies, keyed by the attribute carrying it,
        e.g. ``body`` for Scan, ``then_branch`` and ``else_branch`` for If.
        A body holding no op is still listed:

            cond -> If -> y       then_branch: h -> Add -> t_o
                                  else_branch: zeros (initializer, no op)

            If.subgraphs == {"then_branch": [Add], "else_branch": []}
        """
        return self._graph.subgraphs_of(self._node)


def _holds_tensor(value: onnx_ir.Value) -> bool:
    """Whether ``value`` is an initializer or the output of a ``Constant`` holding a tensor"""
    producer = value.producer()
    return value.is_initializer() or (
        producer is not None
        and producer.op_type == "Constant"
        and producer.attributes.get("value") is not None
    )


_WEIGHT_BIAS = {"weight": WEIGHT_INDEX, "bias": BIAS_INDEX}
_RECURRENT = {"weight": 1, "weight_r": 2, "bias": 3}
_PARAMETER_INDICES = {
    "Conv": _WEIGHT_BIAS,
    "Gemm": _WEIGHT_BIAS,
    "ConvTranspose": _WEIGHT_BIAS,
    "RNN": _RECURRENT,
    "LSTM": _RECURRENT,
    "GRU": _RECURRENT,
    "BatchNormalization": {
        "weight": WEIGHT_INDEX,
        "bias": BIAS_INDEX,
        "running_mean": RUNNING_MEAN_INDEX,
        "running_var": RUNNING_VAR_INDEX,
    },
    "InstanceNormalization": _WEIGHT_BIAS,
    "LayerNormalization": _WEIGHT_BIAS,
    "GroupNormalization": _WEIGHT_BIAS,
    "RMSNormalization": {"weight": WEIGHT_INDEX},
}

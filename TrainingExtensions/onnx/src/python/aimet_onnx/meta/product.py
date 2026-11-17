# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Product class and utilities"""

import math
from typing import TYPE_CHECKING, List, Optional
import onnx_ir
from aimet_onnx.common.connected_graph.product import Product as _Product
from aimet_onnx.ir_utils import static_tensor

if TYPE_CHECKING:
    from aimet_onnx.meta.connectedgraph import ConnectedGraph
    from aimet_onnx.meta.operations import Op


def _dims(shape: Optional[onnx_ir.Shape]) -> List[int]:
    """Dims of ``shape``, with 0 for unknown or symbolic dims, as protobuf reports them"""
    if shape is None:
        return []
    return [dim if isinstance(dim, int) else 0 for dim in shape]


class Product(_Product):
    """
    View of an ``onnx_ir.Value`` in a :class:`ConnectedGraph`.

    Its attributes and links are read from the value on every call. Links map the nodes of the
    value to the ops the connected graph holds for them, so they always return the same objects.
    """

    def __init__(self, value: onnx_ir.Value, graph: "ConnectedGraph"):
        """
        :param value: Value this product represents
        :param graph: Connected graph holding the ops of the value's producer and consumers
        """
        super().__init__(value.name, None)
        self._value = value
        self._graph = graph

    def __repr__(self):
        return self.name

    @property
    def value(self) -> onnx_ir.Value:
        """The value this product represents"""
        return self._value

    @property
    def name(self) -> str:
        return self._value.name

    @property
    def shape(self) -> Optional[List[int]]:
        """
        Shape of the tensor for parameters, constants and graph inputs, or None for the output of
        an op
        """
        if self.is_parm:
            return list(static_tensor(self._value).shape)
        if self._value.is_graph_input():
            return _dims(self._value.shape)
        if self._value.is_initializer():
            return list(self._value.const_value.shape)
        producer = self._value.producer()
        if producer is not None and producer.op_type == "Constant":
            attr = producer.attributes.get("value")
            return list(attr.as_tensor().shape) if attr is not None else []
        return None

    @property
    def is_parm(self) -> bool:
        """
        Whether an op takes this product as a parameter, directly or through Identity and
        Transpose nodes

            W -> Identity -> Conv       W is a parameter of Conv
        """
        values = [self._value]
        while values:
            value = values.pop()
            for use in value.uses():
                op = self._graph.op_of(use.node)
                if op is not None and any(
                    param is self for param, _ in op.parameters.values()
                ):
                    return True
                if use.node.op_type in ("Identity", "Transpose"):
                    values.extend(use.node.outputs)
        return False

    @property
    def is_model_input(self) -> bool:
        """Whether this product is an input of the model's own graph, not of a body"""
        return (
            self._value.is_graph_input()
            and self._value.graph is self._graph.model.graph
        )

    @property
    def is_const(self) -> bool:
        """Whether this product is an initializer or a Constant output that is not a parameter"""
        if self._value.is_graph_input() or self.is_parm:
            return False
        producer = self._value.producer()
        return self._value.is_initializer() or (
            producer is not None and producer.op_type == "Constant"
        )

    @property
    def producer(self) -> Optional["Op"]:
        return self._graph.op_of(self._value.producer())

    @property
    def consumers(self) -> List["Op"]:
        return self._graph.consumers_of(self._value)

    @property
    def tensor(self) -> Optional[onnx_ir.TensorProtocol]:
        """Current tensor of the parameter, or None if this product is not a parameter"""
        return static_tensor(self._value) if self.is_parm else None

    def add_consumer(self, op: "Op"):
        raise AttributeError(
            "Product links are read from its value; edit the model instead"
        )

    def set_consumers_to_null(self):
        raise AttributeError(
            "Product links are read from its value; edit the model instead"
        )

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def is_inter_module(self) -> bool:
        return not (self.is_const or self.is_parm or self.is_model_input)

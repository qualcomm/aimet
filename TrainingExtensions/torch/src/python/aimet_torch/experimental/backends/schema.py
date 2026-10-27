# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=protected-access
import operator
from dataclasses import dataclass
from pathlib import Path
import yaml
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._C import _fx_map_arg

from aimet_torch.common.experimental.backends.schema import (  # pylint: disable=unused-import
    qtype as _qtype,
    OpSchema as _OpSchema,
    OpSignature as _OpSignature,
    Argument as _Argument,
    FixedRange as _FixedRange,
    EqualRange as _EqualRange,
)
from aimet_torch.experimental.export._utils import (
    _is_multi_output_op,
    _is_grid_preserving_op,
    _is_torch_ao_qdq_node,
    _is_aimet_qdq_node,
    _get_aimet_qdq_module,
)
from aimet_torch.quantization.affine import QuantizeDequantize


@dataclass(frozen=True)
class FixedRange(_FixedRange):
    # Overwrite parent type hint
    arg: "Argument"

    def apply(self, qdq_nodes: dict["Argument", torch.fx.Node]):
        qdq_node = qdq_nodes.get(self.arg, None)
        qdq_module = _get_aimet_qdq_module(qdq_node)

        if not qdq_module:
            # qdq_module=None indicatese this argument is not quantized.
            # Skip applying constraints
            return

        if not isinstance(qdq_module, QuantizeDequantize):
            raise RuntimeError(f"Expected QuantizeDequantize, got {type(qdq_module)}")

        qdq_module.set_range(self.min, self.max)
        qdq_module.allow_overwrite(
            min=self.min is None,
            max=self.max is None,
        )


@dataclass(frozen=True)
class EqualRange(_EqualRange):
    # Overwrite parent type hint
    arg1: "Argument"
    arg2: "Argument"

    def apply(self, qdq_nodes: dict["Argument", torch.fx.Node]):
        qdq_node_1 = qdq_nodes.get(self.arg1, None)
        qdq_node_2 = qdq_nodes.get(self.arg2, None)

        if qdq_node_1 and qdq_node_2 and qdq_node_1.graph is not qdq_node_2.graph:
            raise RuntimeError(
                "Cannot apply EqualRange constraint between nodes "
                f"in different graphs: {qdq_node_1} and {qdq_node_2}"
            )

        qdq_module_1 = _get_aimet_qdq_module(qdq_node_1)
        qdq_module_2 = _get_aimet_qdq_module(qdq_node_2)

        if not qdq_module_1 and not qdq_module_2:
            # qdq_module=None indicatese this argument is not quantized.
            # Skip applying constraints
            return

        if (qdq_module_1 is None) != (qdq_module_2 is None):
            raise RuntimeError(
                "Cannot apply EqualRange constraint between a quantized and a non-quantized argument"
            )

        # At this point, both qdq_module_1 and qdq_module_2 are not None
        if not (
            isinstance(qdq_module_1, QuantizeDequantize)
            and isinstance(qdq_module_2, QuantizeDequantize)
        ):
            raise RuntimeError(
                f"Expected QuantizeDequantize, got {type(qdq_module_1)} and {type(qdq_module_2)}"
            )

        # torch.fx.Node's `target` attribute (str) indicates the name of nn.Module
        # associated with this node. Overwriting one node's `target` attribute with another's
        # will make them share the same underlying QuantizeDequantize module.
        qdq_node_1.target = qdq_node_2.target


@dataclass(frozen=True)
class qtype(_qtype):
    def instantiate_quantize_dequantize(
        self, input_shape: tuple[int, ...]
    ) -> QuantizeDequantize:
        if self.dtype.startswith("int"):
            bitwidth = int(self.dtype[3:])
            qmin = -(2 ** (bitwidth - 1))
            qmax = 2 ** (bitwidth - 1) - 1
            symmetric = True
        elif self.dtype.startswith("uint"):
            bitwidth = int(self.dtype[4:])
            qmin = 0
            qmax = 2**bitwidth - 1
            symmetric = False
        else:
            raise NotImplementedError

        if self.channel_axis is None:
            shape = ()
        else:
            shape = tuple(
                dim if axis == self.channel_axis else 1
                for axis, dim in enumerate(input_shape)
            )

        return QuantizeDequantize(
            shape=shape,
            qmin=qmin,
            qmax=qmax,
            symmetric=symmetric,
        )


@dataclass(frozen=True)
class Argument(_Argument):
    # Overwrite parent type hint
    qtype: "qtype | None" = None


@dataclass(frozen=True)
class OpSignature(_OpSignature):
    # Overwrite parent type hint
    inputs: tuple[Argument, ...]
    outputs: tuple[Argument, ...]


@dataclass(frozen=True)
class OpSchema(_OpSchema):
    # Overwrite parent type hint
    signature: OpSignature

    def exact_match(
        self,
        op: torch._ops.OpOverload,
        weight_bw: int,
        activation_bw: int,
        per_channel_quantization: bool = True,
    ) -> bool:
        domain, op_type = op.name().split("::")

        if (domain, op_type) != (self.signature.domain, self.signature.op_type):
            return False

        for arg in self.signature.inputs + self.signature.outputs:
            if arg.role == "weight":
                if not arg.qtype:
                    return False

                if arg.qtype.dtype not in (f"int{weight_bw}", f"uint{weight_bw}"):
                    return False

                if per_channel_quantization == (arg.qtype.channel_axis is None):
                    return False

            if arg.role == "bias":
                if not arg.qtype:
                    return False

                if per_channel_quantization == (arg.qtype.channel_axis is None):
                    return False

            if (
                arg.role is None
                and arg.qtype
                and arg.qtype.dtype
                not in (f"int{activation_bw}", f"uint{activation_bw}")
            ):
                return False

        return True

    def configure_node(self, node: torch.fx.Node):
        """
        Insert QuantizeDequantize or FloatQuantizeDequantize nodes as specified in the schema.

        For example, given aten::conv2d(input, weight, bias, ...) -> Tensor,

        Before:
            input --------->
            weight --------> conv2d --------> ...
            bias ---------->

        After:
            input --> QDQ ->
            weight -> QDQ -> conv2d -> QDQ -> ...
            bias ---> QDQ ->


        Args:
            node (torch.fx.Node): Main node to be configured.
                Corresponds to "conv2d" in the example above.

        Local Variables:
            target_node (torch.fx.Node): The node to which the QDQ will be attached.
                Corresponds to "input", "weight", "bias", and "conv2d" in the example above.
            qdq_node (torch.fx.Node): The newly created quantize-dequantize nodes,
                each of which is associated with one QuantizeDequantize module.
                Corresponds to the QDQ nodes in the example above.
            qdq_module (QuantizeDequantize): The QuantizeDequantize module associated with the qdq_node.
        """

        def replace(args, old: torch.fx.Node, new: torch.fx.Node):
            return _fx_map_arg(args, lambda arg: new if arg is old else arg)

        graph = node.graph

        if node.graph.owning_module:
            graph_module: torch.fx.GraphModule = node.graph.owning_module
        else:
            raise RuntimeError("Node is not part of a GraphModule")

        quantization_specs: dict[Argument, torch.fx.Node] = {
            arg_spec: target_node
            for arg_spec, target_node in zip(
                self.signature.inputs, node.all_input_nodes
            )
        }

        if isinstance(node.meta["val"], torch.Tensor):
            quantization_specs |= {self.signature.outputs[0]: node}
        else:
            raise NotImplementedError(f"Node {node} returns tuple of Tensors")

        for arg_spec, target_node in quantization_specs.items():
            if not arg_spec.qtype:
                # qtype=None indicates that the argument should not be quantized
                continue

            fake_input: FakeTensor = target_node.meta["val"]
            qdq_module = arg_spec.qtype.instantiate_quantize_dequantize(
                fake_input.shape
            )
            qdq_module.to(fake_input.device)
            graph_module.add_module(f"{target_node.name}_qdq", qdq_module)

            with graph.inserting_after(target_node):
                qdq_node = graph.call_module(
                    f"{target_node.name}_qdq", args=(target_node,)
                )
                qdq_node.meta["val"] = fake_input.clone().detach()

            for user in list(target_node.users):
                if user is qdq_node:
                    continue

                if _is_torch_ao_qdq_node(user):
                    continue

                user.args = replace(user.args, old=target_node, new=qdq_node)
                user.kwargs = replace(user.kwargs, old=target_node, new=qdq_node)

        self.apply_constraints(node)

    def apply_constraints(self, node: torch.fx.Node):
        qdq_nodes: dict[Argument, torch.fx.Node] = {
            arg_spec: qdq_node
            for arg_spec, qdq_node in zip(self.signature.inputs, node.all_input_nodes)
            if _is_aimet_qdq_node(qdq_node)
        }
        if isinstance(node.meta["val"], torch.Tensor) and len(node.users) == 1:
            (output_qdq_node,) = node.users
            if _is_aimet_qdq_node(output_qdq_node):
                qdq_nodes |= {self.signature.outputs[0]: output_qdq_node}
        else:
            raise NotImplementedError(
                f"Node {node} returns tuple of Tensors or has multiple users"
            )

        self._apply_fixed_range_constraints(qdq_nodes)
        self._apply_equal_range_constraints(qdq_nodes)

    def _apply_fixed_range_constraints(self, qdq_nodes: dict[Argument, torch.fx.Node]):
        for constraint in self.value_constraints:
            if isinstance(constraint, FixedRange):
                constraint.apply(qdq_nodes)

    def _apply_equal_range_constraints(
        self,
        qdq_nodes: dict[Argument, torch.fx.Node],
    ):
        for constraint in self.value_constraints:
            if isinstance(constraint, EqualRange):
                constraint.apply(qdq_nodes)


class BackendDescription:
    def __init__(
        self,
        all_ops: set[OpSchema],
        name: str | None = None,
    ):
        self.name = name
        self.all_ops: dict[tuple[str, str], set[OpSchema]] = {}

        for schema in all_ops:
            signature = schema.signature
            key = (signature.domain, signature.op_type)
            self.all_ops.setdefault(key, set()).add(schema)

    def __repr__(self) -> str:
        return f"BackendDescription(name={self.name})"

    @classmethod
    def from_backend_id(cls, backend_id: str) -> "BackendDescription":
        from aimet_torch.experimental.backends.parser import parse_aten_op

        with open(Path(__file__).parent / f"{backend_id}.yaml", "r") as f:
            yaml_config = yaml.safe_load(f)

        all_ops = set()

        for op_entry in yaml_config["ops"]:
            if op_entry["op_type"] == "*":
                continue
            all_ops |= parse_aten_op(op_entry)

        return cls(all_ops, name=yaml_config.get("name", None))

    def get_schema(
        self,
        op: torch._ops.OpOverload,
        weight_bw: int,
        activation_bw: int,
    ) -> OpSchema:
        domain, op_type = op.name().split("::")
        schemas = self.all_ops.get((domain, op_type), None)

        if not schemas:
            raise ValueError(f"Op {op.name()} not found in backend {self.name}")

        for schema in schemas:
            if schema.exact_match(op, weight_bw, activation_bw):
                return schema

        raise ValueError(
            f"Op {op.name()} found in backend {self.name}, but no matching schema found"
        )

    def configure(
        self,
        graph_module: torch.fx.GraphModule,
        weight_bw: int,
        activation_bw: int,
    ):
        """
        Configure graph module for quantization according to the backend description.
        """
        graph = graph_module.graph

        for node in graph.nodes:
            # Exclude multi-output ops which returns tuple of tensors.
            # QDQ for each output will be attached to the the succeeding getitem node
            #
            #                                      ┌─ getitem -> y0 -> QDQ
            # ... -> QDQ -> split -> (y0, y1, y2) ─┼─ getitem -> y1 -> QDQ
            #                                      └─ getitem -> y2 -> QDQ
            if _is_multi_output_op(node):
                continue

            # getitem output will inherit the previous node's input encoding
            # if the previous node is a grid-preserving op (aka data movement op)
            if node.target is operator.getitem and _is_grid_preserving_op(
                node.all_input_nodes[0]
            ):
                continue

            # Exclude grid-preserving ops (aka data movement ops) if its input is already quantized.
            # This is to avoid redundant quantize-dequantize pairs around data movement ops.
            if _is_grid_preserving_op(node) and node.target.overloadpacket not in (
                torch.ops.aten.relu,
                torch.ops.aten.relu_,
            ):
                continue

            val = node.meta.get("val", None)

            # Exclude non-tensors from quantization
            if not isinstance(val, torch.Tensor):
                continue

            # Exclude non-floating point tensors from quantization
            if not val.dtype.is_floating_point:
                continue

            if _is_aimet_qdq_node(node):
                continue

            if _is_torch_ao_qdq_node(node):
                continue

            if all(_is_torch_ao_qdq_node(user) for user in node.users):
                continue

            if not isinstance(node.target, torch._ops.OpOverload):
                continue

            schema: OpSchema = self.get_schema(
                node.target, weight_bw=weight_bw, activation_bw=activation_bw
            )
            schema.configure_node(node)

        # Recompile the graph
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module.delete_all_unused_submodules()

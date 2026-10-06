# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import List, Optional, cast

import torch
from torch import nn
from torch.nn import functional as F
from onnx import defs
from onnx import TensorProto
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.common import get_const_value
from onnx2torch.node_converters.registry import (
    OperationDescription,
    _CONVERTER_REGISTRY,
)


class OnnxMatmul(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return torch.matmul(x, y)


# disable_existing_matmul
operation_types_to_disable = {"MatMul": [1, 9, 14]}
domain = defs.ONNX_DOMAIN

for op_type, val in operation_types_to_disable.items():
    for version in val:
        try:
            version = defs.get_schema(
                op_type,
                domain=domain,
                max_inclusive_version=version,
            ).since_version
        except (RuntimeError, defs.SchemaError):
            pass

    description = OperationDescription(
        domain=domain,
        operation_type=op_type,
        version=version,
    )
    if description in _CONVERTER_REGISTRY:
        del _CONVERTER_REGISTRY[description]


@add_converter(operation_type="MatMul", version=13)
@add_converter(operation_type="MatMul", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    if node.input_values[1] in graph.initializers:
        weights = graph.initializers[node.input_values[1]].to_torch().T
        in_features, out_features = weights.shape[1], weights.shape[0]
        torch_module = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=None,
            dtype=weights.dtype,
        )

        with torch.no_grad():
            torch_module.weight.copy_(weights)

        return OperationConverterResult(
            torch_module=torch_module,
            onnx_mapping=OnnxMapping(
                inputs=(node.input_values[0],),
                outputs=node.output_values,
            ),
        )

    return OperationConverterResult(
        torch_module=OnnxMatmul(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


class OnnxFlatten(nn.Module, OnnxToTorchModule):
    """ONNX Flatten: 2D output ``(prod(dims[:axis]), prod(dims[axis:]))``.

    The existing converter crashes when ``axis`` is 0 or the input rank (it flattens
    to rank 1, then indexes a missing dim).
    """

    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        first_dim = math.prod(input_tensor.shape[: self.axis])
        return input_tensor.reshape(first_dim, -1)


# Override the existing Flatten converters (versions 9/11/13)
# Stock impl crashes when axis is 0 or the input rank.
for _flatten_version in (9, 11, 13):
    _flatten_description = OperationDescription(
        domain=defs.ONNX_DOMAIN,
        operation_type="Flatten",
        version=_flatten_version,
    )
    if _flatten_description in _CONVERTER_REGISTRY:
        del _CONVERTER_REGISTRY[_flatten_description]


@add_converter(operation_type="Flatten", version=9)
@add_converter(operation_type="Flatten", version=11)
@add_converter(operation_type="Flatten", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis: int = node.attributes.get("axis", 1)
    return OperationConverterResult(
        torch_module=OnnxFlatten(axis=axis),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


class OnnxReduceMean(nn.Module, OnnxToTorchModule):
    """ReduceMean for opset 18+ where axes is an input rather than an attribute."""

    def __init__(
        self,
        axes: Optional[List[int]] = None,
        keepdims: int = 1,
        noop_with_empty_axes: int = 0,
    ):
        super().__init__()
        self.axes = axes
        self.keepdims = bool(keepdims)
        self.noop_with_empty_axes = noop_with_empty_axes

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.axes is None or len(self.axes) == 0:
            if self.noop_with_empty_axes:
                return input_tensor
            axes = list(range(input_tensor.dim()))
        else:
            axes = self.axes
        return torch.mean(input_tensor, dim=axes, keepdim=self.keepdims)


@add_converter(operation_type="ReduceMean", version=18)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    keepdims: int = node.attributes.get("keepdims", 1)
    noop_with_empty_axes: int = node.attributes.get("noop_with_empty_axes", 0)

    axes = None
    if len(node.input_values) == 2:
        try:
            axes = cast(torch.Tensor, get_const_value(node.input_values[1], graph))
            axes = axes.tolist()
        except KeyError:
            pass

    return OperationConverterResult(
        torch_module=OnnxReduceMean(
            axes=axes,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        ),
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )


class OnnxReduceL2(nn.Module, OnnxToTorchModule):
    """ReduceL2 for opset 18+ where axes is an input rather than an attribute.

    onnx2torch only registers ReduceL2 up to opset 13 (axes-as-attribute, single
    forward arg). The Qwen3.5 dynamo export emits opset-18 ReduceL2 with axes as a
    second input, so the stock converter is called with an extra positional arg and
    fails. Mirrors the ReduceMean opset-18 handling above.
    """

    def __init__(
        self,
        axes: Optional[List[int]] = None,
        keepdims: int = 1,
        noop_with_empty_axes: int = 0,
    ):
        super().__init__()
        self.axes = axes
        self.keepdims = bool(keepdims)
        self.noop_with_empty_axes = noop_with_empty_axes

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.axes is None or len(self.axes) == 0:
            if self.noop_with_empty_axes:
                return input_tensor
            axes = list(range(input_tensor.dim()))
        else:
            axes = self.axes
        return torch.linalg.vector_norm(
            input_tensor, ord=2, dim=axes, keepdim=self.keepdims
        )


@add_converter(operation_type="ReduceL2", version=18)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    keepdims: int = node.attributes.get("keepdims", 1)
    noop_with_empty_axes: int = node.attributes.get("noop_with_empty_axes", 0)

    axes = None
    if len(node.input_values) == 2:
        try:
            axes = cast(torch.Tensor, get_const_value(node.input_values[1], graph))
            axes = axes.tolist()
        except KeyError:
            pass

    return OperationConverterResult(
        torch_module=OnnxReduceL2(
            axes=axes,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        ),
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )


class OnnxReshapeAllowZero(nn.Module, OnnxToTorchModule):
    """Reshape with allowzero=1: zeros in shape are kept as-is (not inherited from input)."""

    def forward(self, input_tensor: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        return torch.reshape(input_tensor, torch.Size(shape))


# Override the existing Reshape converter to support allowzero=1
_reshape_description = OperationDescription(
    domain=defs.ONNX_DOMAIN,
    operation_type="Reshape",
    version=14,
)
if _reshape_description in _CONVERTER_REGISTRY:
    del _CONVERTER_REGISTRY[_reshape_description]


@add_converter(operation_type="Reshape", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    from onnx2torch.node_converters.reshape import OnnxReshape

    if node.attributes.get("allowzero", 0) == 1:
        return OperationConverterResult(
            torch_module=OnnxReshapeAllowZero(),
            onnx_mapping=onnx_mapping_from_node(node=node),
        )

    return OperationConverterResult(
        torch_module=OnnxReshape(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type="Constant", version=19)
@add_converter(operation_type="Constant", version=21)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    from onnx2torch.node_converters.constant import OnnxConstant, _prepare_output_value

    attr_name, value = list(node.attributes.items())[0]
    prepared_value = _prepare_output_value(value, attr_name)

    return OperationConverterResult(
        torch_module=OnnxConstant(value=prepared_value),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


class OnnxScatterElements(nn.Module, OnnxToTorchModule):
    """ScatterElements: scatter updates into data at indices along axis."""

    # ONNX reduction -> torch.scatter_reduce reduce mode.
    _REDUCE_MAP = {"add": "sum", "mul": "prod", "max": "amax", "min": "amin"}

    def __init__(self, axis: int = 0, reduction: str = "none"):
        super().__init__()
        if reduction != "none" and reduction not in self._REDUCE_MAP:
            raise NotImplementedError(
                f"ScatterElements reduction {reduction!r} is not supported."
            )
        self.axis = axis
        self.reduction = reduction

    def forward(
        self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor
    ) -> torch.Tensor:
        if self.reduction == "none":
            return torch.scatter(data, self.axis, indices, updates)
        return data.scatter_reduce(
            self.axis,
            indices,
            updates,
            reduce=self._REDUCE_MAP[self.reduction],
            include_self=True,
        )


@add_converter(operation_type="ScatterElements", version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis: int = node.attributes.get("axis", 0)
    reduction: str = node.attributes.get("reduction", "none")
    return OperationConverterResult(
        torch_module=OnnxScatterElements(axis=axis, reduction=reduction),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


def _non_empty_inputs(node: OnnxNode) -> tuple:
    """Node input names with omitted optional inputs (empty-string names) dropped."""
    return tuple(name for name in node.input_values if name)


class OnnxNonZero(nn.Module, OnnxToTorchModule):
    """ONNX NonZero: indices shaped ``(rank, num_nonzero)``.

    ``torch.nonzero`` returns the transpose ``(num_nonzero, rank)``
    Existing converter omits the transpose the ONNX spec requires.
    """

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(input_tensor).T


# Override the existing NonZero converter, which is missing the ONNX transpose.
_nonzero_description = OperationDescription(
    domain=defs.ONNX_DOMAIN,
    operation_type="NonZero",
    version=13,
)
if _nonzero_description in _CONVERTER_REGISTRY:
    del _CONVERTER_REGISTRY[_nonzero_description]


@add_converter(operation_type="NonZero", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxNonZero(),
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )


class OnnxOneHot(nn.Module, OnnxToTorchModule):
    """
    ONNX OneHot from ``(indices, depth, values=[off, on])``.
    """

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(
        self, indices: torch.Tensor, depth: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        # ONNX OneHot `depth` input has exactly one element (scalar or [1]).
        if depth.numel() != 1:
            raise ValueError("Depth input must have exactly one value")
        depth_int = int(depth.reshape(-1)[0].item())

        # ONNX OneHot `values` input has exactly two elements ([off, on]).
        if values.numel() != 2:
            raise ValueError("Values input must have exactly two elements")
        off_value, on_value = values.reshape(-1)

        indices = indices.long()
        # ONNX permits negative indices in ``[-depth, depth-1]`` (wrap around).
        indices = torch.where(indices < 0, indices + depth_int, indices)

        one_hot = F.one_hot(indices, num_classes=depth_int).to(values.dtype)
        one_hot = off_value + one_hot * (on_value - off_value)

        # ``F.one_hot`` appends the class dim; move it to the requested axis.
        rank = one_hot.dim()
        axis = self.axis
        if axis < 0:
            axis += rank
        if axis != rank - 1:
            one_hot = one_hot.movedim(-1, axis)
        return one_hot


@add_converter(operation_type="OneHot", version=9)
@add_converter(operation_type="OneHot", version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis: int = node.attributes.get("axis", -1)
    return OperationConverterResult(
        torch_module=OnnxOneHot(axis=axis),
        onnx_mapping=OnnxMapping(
            inputs=tuple(node.input_values),
            outputs=node.output_values,
        ),
    )


class OnnxTrilu(nn.Module, OnnxToTorchModule):
    """ONNX Trilu: upper (default) or lower triangular part, optional diagonal ``k``."""

    def __init__(self, upper: bool = True):
        super().__init__()
        self.upper = upper

    def forward(
        self, input_tensor: torch.Tensor, k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        diagonal = 0 if k is None else int(k.reshape(-1)[0].item())
        if self.upper:
            return torch.triu(input_tensor, diagonal=diagonal)
        return torch.tril(input_tensor, diagonal=diagonal)


@add_converter(operation_type="Trilu", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    upper: int = node.attributes.get("upper", 1)
    return OperationConverterResult(
        torch_module=OnnxTrilu(upper=bool(upper)),
        onnx_mapping=OnnxMapping(
            inputs=_non_empty_inputs(node),
            outputs=node.output_values,
        ),
    )


# Override the existing Clip converters (versions 11/12/13): they treat an
# omitted optional min/max (an empty-string input name) as a present, dynamic
# tensor and raise "Dynamic value of min/max is not implemented". Empty-string
# names must be handled as absent.
for _clip_version in (11, 12, 13):
    _clip_description = OperationDescription(
        domain=defs.ONNX_DOMAIN,
        operation_type="Clip",
        version=_clip_version,
    )
    if _clip_description in _CONVERTER_REGISTRY:
        del _CONVERTER_REGISTRY[_clip_description]


@add_converter(operation_type="Clip", version=11)
@add_converter(operation_type="Clip", version=12)
@add_converter(operation_type="Clip", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    from onnx2torch.node_converters.clip import _create_torch_module

    inputs = node.input_values
    # Min/Max are optional, an omitted input is an empty-string name.
    min_name = inputs[1] if len(inputs) > 1 and inputs[1] else None
    max_name = inputs[2] if len(inputs) > 2 and inputs[2] else None

    try:
        min_val = (
            float(get_const_value(min_name, graph)) if min_name is not None else None
        )
        max_val = (
            float(get_const_value(max_name, graph)) if max_name is not None else None
        )
    except KeyError as exc:
        raise NotImplementedError(
            "Dynamic value of min/max is not implemented"
        ) from exc

    return OperationConverterResult(
        torch_module=_create_torch_module(min_val=min_val, max_val=max_val),
        onnx_mapping=OnnxMapping(
            inputs=(inputs[0],),
            outputs=node.output_values,
        ),
    )


# Override the existing Cast converters (versions 9/13).
# Stock impl's TENSOR_TYPE_TO_TORCH_TYPE has no entry for BFLOAT16, so a
# Cast(to=BFLOAT16) node -- e.g. after AdaScale's fp16->bf16 ONNX Cast
# retargeting in model_converter._retarget_fp16_casts_to_bf16 -- raises
# NotImplementedError. Add BFLOAT16 to the dtype map, otherwise identical
# to the stock converter.
for _cast_version in (9, 13):
    _cast_description = OperationDescription(
        domain=defs.ONNX_DOMAIN,
        operation_type="Cast",
        version=_cast_version,
    )
    if _cast_description in _CONVERTER_REGISTRY:
        del _CONVERTER_REGISTRY[_cast_description]


from onnx2torch.node_converters.cast import (
    TENSOR_TYPE_TO_TORCH_TYPE as _BASE_TENSOR_TYPE_TO_TORCH_TYPE,
)

_TENSOR_TYPE_TO_TORCH_TYPE_WITH_BF16 = {
    **_BASE_TENSOR_TYPE_TO_TORCH_TYPE,
    int(TensorProto.BFLOAT16): torch.bfloat16,
}


class OnnxCast(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    TENSOR_TYPE_TO_TORCH_TYPE = _TENSOR_TYPE_TO_TORCH_TYPE_WITH_BF16

    def __init__(self, onnx_dtype: int):
        super().__init__()
        try:
            self.torch_dtype = self.TENSOR_TYPE_TO_TORCH_TYPE[onnx_dtype]
        except KeyError as exc:
            raise NotImplementedError(
                f'Conversion to "{onnx_dtype}" is not implemented'
            ) from exc

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return input_tensor.to(self.torch_dtype)


@add_converter(operation_type="Cast", version=9)
@add_converter(operation_type="Cast", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    onnx_dtype = node.attributes.get("to", None)
    return OperationConverterResult(
        torch_module=OnnxCast(onnx_dtype),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


# Override the existing CumSum converters (versions 11/14).
# Stock onnx2torch defaults reverse=1 when the attribute is absent from the
# node, but the ONNX spec default is 0 (non-reversed). PyTorch's ONNX exporter
# omits the attribute when reverse=False (the common case), so the stock
# converter silently applies a reversed cumsum, producing wrong values.
for _cumsum_version in (11, 14):
    _cumsum_description = OperationDescription(
        domain=defs.ONNX_DOMAIN,
        operation_type="CumSum",
        version=_cumsum_version,
    )
    if _cumsum_description in _CONVERTER_REGISTRY:
        del _CONVERTER_REGISTRY[_cumsum_description]

from onnx2torch.node_converters.cumsum import OnnxCumSum


@add_converter(operation_type="CumSum", version=11)
@add_converter(operation_type="CumSum", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    exclusive = bool(node_attributes.get("exclusive", 0))
    reverse = bool(node_attributes.get("reverse", 0))  # ONNX spec default is 0, not 1
    return OperationConverterResult(
        torch_module=OnnxCumSum(exclusive=exclusive, reverse=reverse),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


# Bulk-register missing op versions for opsets 18-21.
# Many ops have unchanged schemas at newer versions but onnx2torch only registers
# up to opset 13-17. We find the latest registered converter for each op and
# re-register it at the newer since_version.
def _register_missing_opset_converters(max_opset: int = 21):
    """Register converters for op versions up to max_opset using existing converters."""
    _domain = defs.ONNX_DOMAIN

    # Build map of (domain, op_type) -> max registered version
    _registered = {}
    for desc in list(_CONVERTER_REGISTRY.keys()):
        key = (desc.domain, desc.operation_type)
        if key not in _registered or desc.version > _registered[key]:
            _registered[key] = desc.version

    for (d, op), max_ver in _registered.items():
        if d != _domain:
            continue
        for opset in range(max_ver + 1, max_opset + 1):
            try:
                schema = defs.get_schema(op, opset, _domain)
                since = schema.since_version
            except (RuntimeError, defs.SchemaError):
                continue
            if since > max_ver:
                desc_new = OperationDescription(
                    domain=_domain, operation_type=op, version=since
                )
                if desc_new not in _CONVERTER_REGISTRY:
                    # Reuse the converter from the latest registered version
                    desc_old = OperationDescription(
                        domain=_domain, operation_type=op, version=max_ver
                    )
                    _CONVERTER_REGISTRY[desc_new] = _CONVERTER_REGISTRY[desc_old]


_register_missing_opset_converters(max_opset=21)

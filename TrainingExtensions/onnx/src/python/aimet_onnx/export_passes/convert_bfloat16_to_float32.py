# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert every bfloat16 tensor in an ONNX model to float32."""

import numpy as np
import onnx
import onnx_ir


_BF16 = onnx_ir.DataType.BFLOAT16

# Attributes holding an element type, per op. Only those that can name a floating
# point type need rewriting; e.g. EyeLike/RandomNormal's ``dtype``.
_DTYPE_ATTRIBUTES = {
    "Cast": ("to",),
    "DequantizeLinear": ("output_dtype",),
    # ``precision`` (opset 23+) is the type the x / y_scale division is done in.
    "QuantizeLinear": ("output_dtype", "precision"),
    "Bernoulli": ("dtype",),
    "EyeLike": ("dtype",),
    "RandomNormal": ("dtype",),
    "RandomNormalLike": ("dtype",),
    "RandomUniform": ("dtype",),
    "RandomUniformLike": ("dtype",),
}

# Attributes holding a tensor value, per op.
_TENSOR_ATTRIBUTES = {
    "Constant": "value",
    "ConstantOfShape": "value",
}


class ConvertBFloat16ToFloat32Pass(onnx_ir.passes.InPlacePass):
    """
    Convert every bfloat16 tensor in a model to float32, in place.

    Rewrites graph inputs and outputs, initializers, intermediate value types,
    ``Constant``/``ConstantOfShape`` values and element-type attributes such as
    ``Cast(to=BFLOAT16)``, recursing into subgraphs and model-local functions.

    Every bfloat16 value is representable in float32, so constants are preserved
    exactly. The arithmetic, however, now runs in float32 instead of bfloat16, so
    activations can differ from the bfloat16 model by bfloat16 rounding, and a
    quantized tensor near a rounding boundary can land one quantization step away.

    The motivating use is a bfloat16 QDQ model from ``to_onnx_qdq()``: it is valid ONNX,
    but ONNX Runtime has no bfloat16 QuantizeLinear/DequantizeLinear kernels, so it can
    only be run after this conversion.
    """

    def call(self, model: onnx_ir.Model) -> onnx_ir.passes.PassResult:
        graphs = [model.graph, *model.graph.subgraphs()]
        for function in model.functions.values():
            graphs.extend((function, *function.subgraphs()))

        modified = False
        for graph in graphs:
            modified |= _convert_graph(graph)
        return onnx_ir.passes.PassResult(model, modified=modified)


def _convert_graph(graph) -> bool:
    modified = False

    # Functions have no initializers.
    initializers = graph.initializers.values() if hasattr(graph, "initializers") else ()
    values = [*graph.inputs, *graph.outputs, *initializers]
    for node in graph:
        values.extend(node.outputs)
        modified |= _convert_attributes(node)

    for value in values:
        # Replacing const_value does not update the value's declared type, so both
        # are converted.
        if value.const_value is not None and value.const_value.dtype == _BF16:
            value.const_value = _to_float32(value.const_value)
            modified = True
        if value.dtype == _BF16:
            value.dtype = onnx_ir.DataType.FLOAT
            modified = True

    return modified


def _convert_attributes(node: onnx_ir.Node) -> bool:
    if node.domain not in ("", "ai.onnx"):
        return False

    modified = False
    for dtype_attr in _DTYPE_ATTRIBUTES.get(node.op_type, ()):
        attr = node.attributes.get(dtype_attr)
        if attr is not None and attr.value == _BF16:
            node.attributes[dtype_attr] = onnx_ir.AttrInt64(
                dtype_attr, onnx_ir.DataType.FLOAT
            )
            modified = True

    tensor_attr = _TENSOR_ATTRIBUTES.get(node.op_type)
    if tensor_attr is not None:
        attr = node.attributes.get(tensor_attr)
        if attr is not None and attr.value.dtype == _BF16:
            node.attributes[tensor_attr] = onnx_ir.AttrTensor(
                tensor_attr, _to_float32(attr.value)
            )
            modified = True

    return modified


def _to_float32(tensor: onnx_ir.TensorProtocol) -> onnx_ir.Tensor:
    return onnx_ir.Tensor(
        tensor.numpy().astype(np.float32),
        name=tensor.name,
        doc_string=tensor.doc_string,
    )


def convert_bfloat16_to_float32_pass(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a copy of ``model`` with every bfloat16 tensor converted to float32.

    See :class:`ConvertBFloat16ToFloat32Pass` for what is converted.
    """
    ir_model = onnx_ir.from_proto(model)
    ConvertBFloat16ToFloat32Pass()(ir_model)
    return onnx_ir.to_proto(ir_model)

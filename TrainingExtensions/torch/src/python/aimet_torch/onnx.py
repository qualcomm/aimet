# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Defines onnx export API"""

from collections import defaultdict
import copy
import contextlib
import io
import os
import tempfile
import traceback
from typing import Any, List, Mapping, Tuple, Union

import onnx
import torch
from torch.onnx import _constants

from aimet_common.onnx._utils import (
    _add_onnx_qdq_nodes,
    _is_grid_preserving_op,
    _convert_version,
)

from .nn import QuantizationMixin
from .quantization import DequantizedTensor
from .quantization.base import EncodingBase
from .quantization.affine import AffineQuantizerBase, GroupedBlockQuantizeDequantize
from .quantization.float import FloatQuantizeDequantize
from .quantsim import QuantizationSimModel
from .v2.experimental import onnx as _onnx


_TORCH_DEFAULT_OPSET = _constants.ONNX_DEFAULT_OPSET
_TORCH_MIN_OPSET = _constants.ONNX_MIN_OPSET
_TORCH_MAX_OPSET = _constants.ONNX_MAX_OPSET

# Allow at least up to opset 21 to enable [u]int16 QDQ export
_AIMET_MAX_OPSET = max(_TORCH_MAX_OPSET, 21)


def export(
    model: Union[torch.nn.Module, QuantizationSimModel],
    args: Union[Tuple[Any, ...], torch.Tensor],
    f: Union[str, io.BytesIO],
    *,
    export_int32_bias: bool = True,
    **kwargs,
):
    """
    Export :class:`QuantizationSimModel` to onnx model with
    onnx `QuantizeLinear`_ and `DequantizeLinear`_ embedded in the graph.

    This function takes set of same arguments as `torch.onnx.export()`_

    Args:
        model: The model to be exported
        args: Same as `torch.onnx.export()`
        f: Same as `torch.onnx.export()`
        export_int32_bias (bool, optional):
            If true, generate and export int32 bias encoding on the fly (default: `True`)
        **kwargs: Same as `torch.onnx.export()`


    .. note::
        Unlike `torch.onnx.export()`, this function allows up to opset 21.
        to support 4/16-bit quantization only available in opset 21.
        However, exporting to opset 21 is a beta feature and not fully stable yet.
        For robustness, opset 20 or lower is recommended whenever possible.

    .. note::
        Dynamo-based export (`dynamo=True`) is not supported yet

    .. _torch.onnx.export(): https://docs.pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export
    .. _QuantizeLinear: https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
    .. _DequantizeLinear: https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html

    Examples:

        >>> aimet_torch.onnx.export(sim.model, x, f="model.onnx",
        ...                         input_names=["input"], output_names=["output"],
        ...                         opset_version=21, export_int32_bias=True)
        ...
        >>> import onnxruntime as ort
        >>> options = ort.SessionOptions()
        >>> options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        >>> sess = ort.InferenceSession("model.onnx", sess_options=options)
        >>> onnx_output, = sess.run(None, {"input": x.detach().numpy()})
        >>> torch.nn.functional.cosine_similarity(torch.from_numpy(onnx_output), sim.model(x))
        tensor([1.0000, 0.9999, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
               grad_fn=<AliasBackward0>)
    """
    if isinstance(model, QuantizationSimModel):
        model = model.model

    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            f"aimet_torch.export only supports torch.nn.Module or QuantizationSimModel; got {type(model)}"
        )

    _check_opset_version(kwargs)
    _check_unsupported_args(kwargs)
    _check_non_standard_quantizer(model)

    target_version = kwargs.pop("opset_version", _TORCH_DEFAULT_OPSET)
    kwargs["opset_version"] = min(target_version, _TORCH_MAX_OPSET)

    _assert_minimum_required_opset(model, target_version)

    with contextlib.ExitStack() as stack:
        # Unfold all param quantizers to incorporate QuantizeLinear/DequantizeLinear
        # of those parameters in tracing time
        stack.enter_context(_temporarily_unfold_param_quantizers(model))

        if export_int32_bias:
            # Temoprarily instantiate int32 bias quantizers
            stack.enter_context(
                _concretize_int32_bias_quantizers(model, args, kwargs.get("kwargs"))
            )

        # Export quantize-dequantized weight
        # pylint: disable=protected-access
        stack.enter_context(QuantizationSimModel._apply_qdq_to_model_parameters(model))

        # Remove [b]float16 quantizers
        stack.enter_context(_remove_fp16_quantizers(model))

        onnx_model, tensor_to_encoding_map = _to_onnx(model, args, **kwargs)

    if _TORCH_MAX_OPSET < target_version:
        try:
            onnx_model = _convert_version(onnx_model, target_version)
        except Exception as e:  # pylint: disable=broad-exception-caught
            f = io.StringIO()
            traceback.print_exc(file=f)
            reason = _why_do_i_need_opset21(model)

            if reason:
                detail = (
                    f"torch.onnx.export only supports opset<={_TORCH_MAX_OPSET}, "
                    f"but onnx::QuantizeLinear requires opset>={target_version} for {reason}. "
                    "As a workaround, we tried to torch.onnx.export your model "
                    f"with opset={_TORCH_MAX_OPSET} and convert the onnx model to {target_version}, "
                    "but failed with the following error:\n\n"
                )
            else:
                detail = "\n\n"

            msg = (
                f"Failed to convert onnx model to {target_version} due to {type(e).__name__}. {detail}"
                "==============================================================\n"
                f"{f.getvalue()}"
                "==============================================================\n\n"
            )
            raise RuntimeError(msg) from e

    onnx_qdq_model = _to_onnx_qdq(onnx_model, tensor_to_encoding_map)
    onnx.save(onnx_qdq_model, f)


def _why_do_i_need_opset21(model: torch.nn.Module) -> str:
    int4 = False
    int16 = False
    bq = False

    for qtzr in model.modules():
        if not isinstance(qtzr, AffineQuantizerBase):
            continue

        if qtzr.block_size is not None:
            bq = True

        if qtzr.bitwidth == 4:
            int4 = True

        if qtzr.bitwidth == 16:
            int16 = True

    reasons = []

    if int4 or int16:
        reasons.append("int4/int16 quantization")

    if bq:
        reasons.append("blockwise quantization")

    if not reasons:
        return ""  # This should never happen

    if len(reasons) == 1:
        return reasons[0]

    return f"{reasons[0]} and {reasons[1]}"


def _assert_minimum_required_opset(model: torch.nn.Module, target_opset: int):
    if target_opset < 21 and any(
        qtzr.block_size is not None
        for qtzr in model.modules()
        if isinstance(qtzr, AffineQuantizerBase)
    ):
        raise RuntimeError(
            "onnx::QuantizeLinear and DequantizeLinear with per-block are only supported in opset >= 21;"
            f" got opset={target_opset}"
        )

    if target_opset < 21 and any(
        qtzr.bitwidth in (4, 16)
        for qtzr in model.modules()
        if isinstance(qtzr, AffineQuantizerBase)
    ):
        raise RuntimeError(
            "onnx::QuantizeLinear and DequantizeLinear with INT4/INT16 are only supported in opset >= 21;"
            f" got opset={target_opset}"
        )

    if target_opset < 13 and any(
        tuple(qtzr.shape)
        for qtzr in model.modules()
        if isinstance(qtzr, AffineQuantizerBase)
    ):
        raise RuntimeError(
            "onnx::QuantizeLinear and DequantizeLinear with per-channel are only supported in opset >= 13;"
            f" got opset={target_opset}"
        )

    if target_opset < 10:
        raise RuntimeError(
            "onnx::QuantizeLinear and DequantizeLinear are only supported in opset >= 10;"
            f" got opset={target_opset}"
        )


def _check_opset_version(kwargs):
    opset_version = kwargs.get("opset_version", _TORCH_DEFAULT_OPSET)

    if not (_TORCH_MIN_OPSET <= opset_version <= _AIMET_MAX_OPSET):
        raise ValueError(f"Unsupported ONNX opset version: {opset_version}")


def _check_unsupported_args(kwargs):
    export_params = kwargs.get("export_params", True)

    if not export_params:
        raise NotImplementedError("export_params=False is not supported yet")

    keep_initializers_as_inputs = kwargs.get("keep_initializers_as_inputs", False)

    if keep_initializers_as_inputs:
        raise NotImplementedError(
            "keep_initializers_as_inputs=True is not supported yet"
        )

    dynamo = kwargs.get("dynamo", False)

    if dynamo:
        raise NotImplementedError("dynamo=True is not supported yet")

    do_constant_folding = kwargs.get("do_constant_folding", True)

    if not do_constant_folding:
        raise NotImplementedError("do_constant_folding=False is not supported yet")

    export_modules_as_functions = kwargs.get("export_modules_as_functions", False)

    if export_modules_as_functions:
        raise RuntimeError("export_modules_as_functions=True is not supported")

    operator_export_type = kwargs.get(
        "operator_export_type", torch.onnx.OperatorExportTypes.ONNX
    )

    if operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN:
        raise RuntimeError(
            "operator_export_type=OperatorExportTypes.ONNX_ATEN is not supported"
        )


def _check_non_standard_quantizer(model: torch.nn.Module):
    for name, qtzr in model.named_modules():
        if not isinstance(qtzr, AffineQuantizerBase):
            continue

        if isinstance(qtzr, GroupedBlockQuantizeDequantize):
            raise NotImplementedError(
                "torch.onnx.exoprt doesn't support GroupedBlockQuantizeDequantize (a.k.a LPBQ) yet; "
                f"got '{name}' of type GroupedBlockQuantizeDequantize"
            )

        if qtzr.bitwidth not in (4, 8, 16, 32):
            raise RuntimeError(
                "torch.onnx.exoprt only supports 4/8/16/32-bit integers; "
                f"got '{name}' with bitwidth={qtzr.bitwidth}"
            )


def _to_onnx(
    model: torch.nn.Module, args: Union[Tuple[Any, ...], torch.Tensor], **kwargs
):
    _check_float16_quantizers(model)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_onnx_path = os.path.join(tmp_dir, "quantized_model.onnx")
        _onnx.export(model, args, tmp_onnx_path, **kwargs)
        onnx_model = onnx.load(tmp_onnx_path)

        param_names = {
            f"{layer_name}.{param_name}"
            for layer_name, layer in model.named_modules()
            if isinstance(layer, QuantizationMixin)
            for param_name, quantizer in layer.param_quantizers.items()
            if quantizer
        }

    tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]]
    tensor_to_encoding_map = {
        name: (encoding, name in param_names)
        for name, encoding in _onnx.remove_quantization_nodes_from_onnx_graph(
            onnx_model
        ).items()
    }
    tensor_to_encoding_map |= _derive_data_movement_op_encoding(
        onnx_model, tensor_to_encoding_map
    )
    return onnx_model, tensor_to_encoding_map


def _derive_data_movement_op_encoding(
    model: onnx.ModelProto,
    tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]],
) -> Mapping[str, Tuple[EncodingBase, bool]]:
    data_movement_ops = [
        node for node in model.graph.node if _is_grid_preserving_op(node.op_type)
    ]

    encodings = {name: enc for name, (enc, _) in tensor_to_encoding_map.items()}
    new_encodings = {}
    consumers: Mapping[str, List[onnx.NodeProto]] = defaultdict(list)

    for node in model.graph.node:
        for inp in node.input:
            consumers[inp].append(node)

    def derive_encoding(node: onnx.NodeProto):
        input_name = node.input[0]
        output_name = node.output[0]
        inp_encoding = encodings.get(input_name)
        out_encoding = encodings.get(output_name)

        if not inp_encoding and not out_encoding:
            # No input encoding to inherit; skip
            return {}

        if inp_encoding and out_encoding:
            # Both input and output encoding already exists; skip
            return {}

        if out_encoding:
            if len(consumers[input_name]) > 1 or len(node.output) > 1:
                # If input has more than one consumer or if there are more than one output,
                # it is NOT safe to reuse output encoding for input quantization
                return {}
            else:
                # Reuse output encoding for input quantization
                return {input_name: copy.deepcopy(out_encoding)}
        else:
            # Reuse input encoding for output quantization
            return {output_name: copy.deepcopy(inp_encoding)}

    for node in data_movement_ops:
        enc = derive_encoding(node)
        new_encodings |= enc
        encodings |= enc

    # Repeat in reverse-DFS order
    for node in reversed(data_movement_ops):
        enc = derive_encoding(node)
        new_encodings |= enc
        encodings |= enc

    return {key: (enc, False) for key, enc in new_encodings.items()}


@contextlib.contextmanager
def _concretize_int32_bias_quantizers(model, args, kwargs=None):
    if not isinstance(args, (tuple, list)):
        args = (args,)

    kwargs = kwargs or {}

    handles = []
    orig_bias_quantizers = {
        qmodule: qmodule.param_quantizers["bias"]
        for qmodule in model.modules()
        if isinstance(qmodule, QuantizationMixin)
        and "bias" in qmodule.param_quantizers
        and qmodule.bias is not None
    }

    try:
        for qmodule, qtzr in orig_bias_quantizers.items():
            if qtzr is not None:
                # Bias quantizer already exists.
                # This means the user created bias quantizer by him/herself
                # In this case, we honor the custom bias quantizer defined by the user
                continue

            if "weight" in qmodule.param_quantizers and isinstance(
                qmodule.param_quantizers["weight"], AffineQuantizerBase
            ):
                # pylint: disable=protected-access
                handle = qmodule.register_forward_hook(
                    type(qmodule)._create_int32_bias_quantizer
                )
                handles.append(handle)
        try:
            model(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        yield
    finally:
        for qmodule, qtzr in orig_bias_quantizers.items():
            qmodule.param_quantizers["bias"] = qtzr


@contextlib.contextmanager
def _temporarily_unfold_param_quantizers(model: torch.nn.Module):
    # pylint: disable=protected-access
    """
    Temporarily re-instantiate param quantizers for ease of export
    """
    modules_with_folded_parameters = [
        qmodule
        for qmodule in model.modules()
        if isinstance(qmodule, QuantizationMixin)
        and any(isinstance(param, DequantizedTensor) for param in qmodule.parameters())
    ]

    try:
        for qmodule in modules_with_folded_parameters:
            qmodule._unfold_param_quantizers()
        yield
    finally:
        for qmodule in modules_with_folded_parameters:
            qmodule._fold_param_quantizers()


@contextlib.contextmanager
def _remove_fp16_quantizers(model: torch.nn.Module):
    """
    Temporarily remove [b]float16 quantizers for sim.onnx.export,
    as sim.onnx.export does NOT support exporting [b]float16 quantizers.
    """
    original_containers = {}

    try:
        for qmodule in model.modules():
            if not isinstance(qmodule, QuantizationMixin):
                continue

            for name, qtzr in qmodule.param_quantizers.items():
                if isinstance(qtzr, FloatQuantizeDequantize) and (
                    qtzr.is_float16() or qtzr.is_bfloat16()
                ):
                    original_containers[(qmodule.param_quantizers, name)] = qtzr
                    qmodule.param_quantizers[name] = None

            for i, qtzr in enumerate(qmodule.input_quantizers):
                if isinstance(qtzr, FloatQuantizeDequantize) and (
                    qtzr.is_float16() or qtzr.is_bfloat16()
                ):
                    original_containers[(qmodule.input_quantizers, i)] = qtzr
                    qmodule.input_quantizers[i] = None

            for i, qtzr in enumerate(qmodule.output_quantizers):
                if isinstance(qtzr, FloatQuantizeDequantize) and (
                    qtzr.is_float16() or qtzr.is_bfloat16()
                ):
                    original_containers[(qmodule.output_quantizers, i)] = qtzr
                    qmodule.output_quantizers[i] = None

        yield

    finally:
        for (container, key), qtzr in original_containers.items():
            container[key] = qtzr


def _to_onnx_qdq(
    onnx_model: onnx.ModelProto,
    tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]],
) -> onnx.ModelProto:
    qnn_encodings = {
        name: encoding.to_qnn_encoding_dict("2.0.0")
        for name, (encoding, _) in tensor_to_encoding_map.items()
    }
    qnn_encodings = {
        name: encoding for name, encoding in qnn_encodings.items() if encoding
    }

    qdq_tensor_names = {
        fp_tensor_name: f"{fp_tensor_name}_qdq" for fp_tensor_name in qnn_encodings
    }

    onnx_opset_version = next(
        opset.version for opset in onnx_model.opset_import if opset.domain == ""
    )

    # Add onnx QDQ nodes in batch
    _add_onnx_qdq_nodes(
        onnx_model,
        input_names=qnn_encodings.keys(),
        output_names=qdq_tensor_names.values(),
        node_name_prefixes=qnn_encodings.keys(),
        encodings=qnn_encodings.values(),
        onnx_opset=onnx_opset_version,
    )

    # Restore model output names from "{output}_qdq" to "{output}"
    _restore_model_output_names(onnx_model, qdq_tensor_names)

    return onnx_model


def _check_float16_quantizers(module: torch.nn.Module):
    for qtzr in module.modules():
        if isinstance(qtzr, FloatQuantizeDequantize):
            if not qtzr.is_float16() and not qtzr.is_bfloat16():
                msg = " ".join(
                    [
                        "sim.onnx.export doesn't support exporting floating point encodings",
                        f"except [b]float16. Got {qtzr.bitwidth}-bit float encoding",
                    ]
                )
                raise RuntimeError(msg)


def _rename_inputs(onnx_model: onnx.ModelProto, new_names: Mapping[str, str]):
    for node in onnx_model.graph.node:
        for i, old_name in enumerate(node.input):
            new_name = new_names.get(old_name, None)
            if new_name is not None:
                node.input[i] = new_name


def _rename_outputs(onnx_model: onnx.ModelProto, new_names: Mapping[str, str]):
    for node in onnx_model.graph.node:
        for i, old_name in enumerate(node.output):
            new_name = new_names.get(old_name, None)
            if new_name is not None:
                node.output[i] = new_name


def _restore_model_output_names(
    onnx_model: onnx.ModelProto, new_names: Mapping[str, str]
):
    """
    Rename model outputs. Assuming "output" is the model output,

    before:
        Softmax ----> output -------> QDQ -------> output_qdq

    after:
        Softmax ----> output__ -----> QDQ -------> output
    """
    _new_names = {
        output.name: f"{output.name}__"
        for output in onnx_model.graph.output
        if output.name in new_names
    }
    _rename_inputs(onnx_model, _new_names)

    _new_names.update(
        {
            new_names[output.name]: output.name
            for output in onnx_model.graph.output
            if output.name in new_names
        }
    )
    _rename_outputs(onnx_model, _new_names)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for replicating Torch model interface on ONNX InferenceSessions. Borrowed from AI Hub Models"""

import warnings
import onnx
import torch
import onnxruntime
from typing import Iterable, Any, Collection

# Element types are tracked as ONNX TensorProto codes rather than numpy dtypes:
# numpy has no bfloat16, but IOBinding accepts the ONNX code directly, so bf16
# tensors can be bound on CUDA without ever passing through numpy.
_TORCH_TO_ONNX_DTYPE = {
    torch.bfloat16: onnx.TensorProto.BFLOAT16,
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float64: onnx.TensorProto.DOUBLE,
    torch.int32: onnx.TensorProto.INT32,
    torch.int64: onnx.TensorProto.INT64,
    torch.int8: onnx.TensorProto.INT8,
    torch.uint8: onnx.TensorProto.UINT8,
    torch.bool: onnx.TensorProto.BOOL,
}

_ORT_TYPE_TO_TORCH_DTYPE = {
    "tensor(bfloat16)": torch.bfloat16,
    "tensor(float16)": torch.float16,
    "tensor(float)": torch.float32,
    "tensor(double)": torch.float64,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(int8)": torch.int8,
    "tensor(uint8)": torch.uint8,
    "tensor(bool)": torch.bool,
}


def _is_cuda_session(session) -> bool:
    return "CUDAExecutionProvider" in session.get_providers()


def _get_cuda_device_id(session) -> int:
    provider_options = session.get_provider_options()
    cuda_opts = provider_options.get("CUDAExecutionProvider", {})
    return int(cuda_opts.get("device_id", "0"))


def kwargs_to_dict(argnames: Iterable[str], *args, **kwargs) -> dict[str, Any]:
    input_dict: dict[str, Any] = dict()
    for idx, input_name in enumerate(argnames):
        if len(args) > idx:
            input_val = args[idx]
            if input_name in kwargs:
                raise ValueError(
                    f"Cannot pass input {input_name} twice (as a positional arg and a keyword arg)."
                )
        elif input_name in kwargs:
            input_val = kwargs[input_name]
        else:
            raise ValueError(f"Missing input {input_name}")
        input_dict[input_name] = input_val
    return input_dict


def _resolve_output_shapes(
    session: onnxruntime.InferenceSession,
    inputs: dict[str, torch.Tensor],
) -> list[tuple[int, ...]] | None:
    """Resolve concrete output shapes from session metadata and input shapes.

    Returns None if any dimension cannot be resolved, allowing the caller to
    let ORT allocate the outputs instead.
    """
    input_shapes = {name: tuple(t.shape) for name, t in inputs.items()}
    sym_table: dict[str, int] = {}
    for meta in session.get_inputs():
        for i, dim in enumerate(meta.shape):
            if isinstance(dim, str):
                sym_table[dim] = input_shapes[meta.name][i]

    if "input_ids" in input_shapes:
        seq_len = input_shapes["input_ids"][1]
        batch_size = input_shapes["input_ids"][0]
    elif "inputs_embeds" in input_shapes:
        seq_len = input_shapes["inputs_embeds"][1]
        batch_size = input_shapes["inputs_embeds"][0]
    else:
        seq_len = None
        batch_size = None

    output_shapes = []
    for meta in session.get_outputs():
        shape = []
        for dim_idx, dim in enumerate(meta.shape):
            if isinstance(dim, int):
                shape.append(dim)
            elif isinstance(dim, str) and dim in sym_table:
                shape.append(sym_table[dim])
            elif "past_" in meta.name and "_out" in meta.name and dim_idx == 2:
                in_name = meta.name.replace("_out", "_in").replace("_updated", "")
                if in_name in input_shapes and seq_len is not None:
                    shape.append(input_shapes[in_name][2] + seq_len)
                else:
                    return None
            elif "logits" in meta.name and dim_idx == 0 and batch_size is not None:
                shape.append(batch_size)
            else:
                return None
        output_shapes.append(tuple(shape))
    return output_shapes


def _override_model_inputs_with_session_dtypes(
    session: onnxruntime.InferenceSession,
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Cast each input to the dtype its ORT session declared.

    Callers (HF processors, dataset loaders) routinely produce fp32 tensors
    (e.g. ``pixel_values``) regardless of the graph's precision. When the
    graph was exported at fp16, ORT rejects the mismatch with
    ``Unexpected input data type. Actual: (tensor(float)) , expected:
    (tensor(float16))``. Cast at the boundary so callers stay dtype-agnostic;
    warn whenever a cast happens so callers comparing against a
    higher-precision reference don't attribute the induced error to the graph.
    """
    expected = {
        inp.name: _ORT_TYPE_TO_TORCH_DTYPE[inp.type] for inp in session.get_inputs()
    }
    out: dict[str, torch.Tensor] = {}
    for name, tensor in inputs.items():
        target = expected[name]
        if tensor.dtype != target:
            warnings.warn(
                f"Input {name!r}: casting {tensor.dtype} → {target} to match "
                "ORT session dtype. Cast inputs at the source to silence this.",
                stacklevel=2,
            )
            tensor = tensor.to(target)
        out[name] = tensor
    return out


def _iobinding_inference(
    session: onnxruntime.InferenceSession,
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> torch.Tensor | Collection[torch.Tensor]:
    device_id = _get_cuda_device_id(session)
    torch_device = f"cuda:{device_id}"
    input_names = [inp.name for inp in session.get_inputs()]

    inputs = kwargs_to_dict(input_names, *args, **kwargs)
    inputs = _override_model_inputs_with_session_dtypes(session, inputs)
    output_shapes = _resolve_output_shapes(session, inputs)

    binding = session.io_binding()
    output_metas = session.get_outputs()

    # Bound buffers are referenced by raw pointer, so keep the (possibly newly
    # moved or made contiguous) tensors alive until the run completes.
    bound_inputs = []
    for name, tensor in inputs.items():
        tensor = tensor.to(torch_device).contiguous()
        bound_inputs.append(tensor)
        binding.bind_input(
            name=name,
            device_type="cuda",
            device_id=device_id,
            element_type=_TORCH_TO_ONNX_DTYPE[tensor.dtype],
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    output_tensors = []
    if output_shapes is None:
        # Shapes are unknown up front: let ORT allocate the outputs on the GPU
        # and hand them to torch via DLPack (zero-copy, works for bf16).
        for meta in output_metas:
            binding.bind_output(meta.name, "cuda", device_id)
    else:
        for meta, shape in zip(output_metas, output_shapes):
            tensor = torch.empty(
                shape, dtype=_ORT_TYPE_TO_TORCH_DTYPE[meta.type], device=torch_device
            )
            output_tensors.append(tensor)
            binding.bind_output(
                name=meta.name,
                device_type="cuda",
                device_id=device_id,
                element_type=_TORCH_TO_ONNX_DTYPE[tensor.dtype],
                shape=shape,
                buffer_ptr=tensor.data_ptr(),
            )

    torch.cuda.synchronize()
    session.run_with_iobinding(binding)
    torch.cuda.synchronize()

    if output_shapes is None:
        output_tensors = [
            torch.from_dlpack(out._ortvalue.to_dlpack())
            for out in binding.get_outputs()
        ]

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors


def _numpy_inference(
    session: onnxruntime.InferenceSession,
    inputs: dict[str, torch.Tensor],
    target_device: str,
) -> torch.Tensor | Collection[torch.Tensor]:
    """Run a CPU session via numpy (``session.run``)."""
    bf16_io = [
        meta.name
        for meta in (*session.get_inputs(), *session.get_outputs())
        if meta.type == "tensor(bfloat16)"
    ]
    if bf16_io:
        # numpy has no bfloat16, and ORT's CPU EP has too few bf16 kernels to
        # run these graphs anyway.
        raise RuntimeError(
            f"bfloat16 graph inputs/outputs {bf16_io[:4]} require a session on "
            "CUDAExecutionProvider; numpy-based CPU inference cannot represent "
            "bfloat16."
        )
    np_inputs = {k: v.cpu().detach().numpy() for k, v in inputs.items()}
    output_np = session.run(None, np_inputs)
    output_tensors = [torch.from_numpy(out).to(target_device) for out in output_np]

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors


def mock_torch_onnx_inference(
    session: onnxruntime.InferenceSession,
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> torch.Tensor | Collection[torch.Tensor]:
    if _is_cuda_session(session):
        return _iobinding_inference(session, *args, **kwargs)

    input_names = [inp.name for inp in session.get_inputs()]
    inputs = kwargs_to_dict(input_names, *args, **kwargs)
    inputs = _override_model_inputs_with_session_dtypes(session, inputs)
    return _numpy_inference(session, inputs, "cpu")


def _flatten_tensor_args(args):
    """Flatten any tensor lists/dicts in positional args for ONNX session input."""
    flat = []
    for a in args:
        if isinstance(a, list) and a and isinstance(a[0], torch.Tensor):
            flat.extend(a)
        elif isinstance(a, dict) and all(
            isinstance(v, torch.Tensor) for v in a.values()
        ):
            flat.extend(a.values())
        else:
            flat.append(a)
    return tuple(flat)


def _float_dtype_from_session(session) -> torch.dtype:
    """Derive the graph's float dtype from its declared ORT input types.

    The exported session is the source of truth for precision (the same source
    :func:`_override_model_inputs_with_session_dtypes` casts inputs against), so
    there is no need to be told the dtype separately. Returns the first floating
    input type; falls back to fp32 for graphs with no float inputs.
    """
    for inp in session.get_inputs():
        torch_dtype = _ORT_TYPE_TO_TORCH_DTYPE[inp.type]
        if torch_dtype.is_floating_point:
            return torch_dtype
    return torch.float32


class TorchONNXInterface(torch.nn.Module):
    def __init__(self, quantsim, config):
        super().__init__()
        self.quantsim = quantsim
        self._config = config

    @property
    def config(self):
        return self._config

    @property
    def device(self) -> torch.device:
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @property
    def dtype(self) -> torch.dtype:
        # Derived from the wrapped session (the source of truth for the exported
        # graph's precision) so it can never drift from the graph.
        return _float_dtype_from_session(self.quantsim.session)

    def forward(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """
        QuantSim forward pass with torch.Tensor.

        Tensor lists in positional args (e.g. deepstack_visual_embeds) are
        flattened to match the ONNX graph's flat input layout.
        """
        assert self.quantsim is not None
        flat_args = _flatten_tensor_args(args)
        return mock_torch_onnx_inference(self.quantsim.session, *flat_args, **kwargs)

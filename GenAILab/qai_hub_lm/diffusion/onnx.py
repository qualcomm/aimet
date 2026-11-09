# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX export of the diffusion components, and ONNX Runtime sessions behind them.

Each component wrapper exports to its own ONNX graph, keeping the wrapper's
input/output names. :class:`ORTComponent` then stands in for the torch module, so
the same :class:`~.generator.DiffusionGenerator` runs the denoising loop over
ONNX Runtime sessions. Like the rest of this package, it does not depend on AIMET.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .base import COMPONENTS, DiffusionComponents

#: Opset of the exported graphs. On ONNX Runtime 1.23, newer opsets push Conv,
#: Resize and normalization onto the CPU (opset 22: ~30x slower UNet), and
#: opset 23's fused Attention is wrong on CUDA for SD1.5's head size (160).
#: 18 is the newest opset the dynamo exporter emits for every SD component.
ONNX_OPSET = 18

#: Dtypes the exported graphs support. bf16 Conv needs opset 22, and ONNX
#: Runtime has no bf16 CUDA Resize kernel at any opset.
SUPPORTED_DTYPES = (torch.float32, torch.float16)

MODEL_FILE = "model.onnx"

# ONNX element-type names -> torch dtypes, for the dtypes the graphs use.
_ORT_TO_TORCH = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
}


def check_dtype(dtype: torch.dtype) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Diffusion ONNX export supports {[str(d) for d in SUPPORTED_DTYPES]}, "
            f"not {dtype}: bfloat16 Conv needs ONNX opset 22, which runs Stable "
            f"Diffusion far slower on the current ONNX Runtime (opset {ONNX_OPSET} "
            f"is used)."
        )


def export_components(
    model_cls,
    components: DiffusionComponents,
    height: int,
    width: int,
    output_dir: str | os.PathLike,
) -> dict[str, Path]:
    """Export every component to ``<output_dir>/<component>/model.onnx``.

    Exports on the components' current device, with fixed shapes: batch 1 and
    the given resolution, the shapes the generator runs. Returns the graph path
    of each component.
    """
    check_dtype(components.dtype)
    paths = {}
    for name, module in components.items():
        path = Path(output_dir) / name / MODEL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Exporting {name} to ONNX...")
        with torch.no_grad():
            torch.onnx.export(
                module.eval(),
                model_cls.get_sample_inputs(name, components, height, width),
                str(path),
                input_names=list(model_cls.get_input_names(name)),
                output_names=list(model_cls.get_output_names(name)),
                opset_version=ONNX_OPSET,
                # The TorchScript exporter mixes fp16 and fp32 in the fp16 text
                # encoder's LayerNorm, which ONNX Runtime refuses to load.
                dynamo=True,
            )
        paths[name] = path
    return paths


def providers_for(device: torch.device | str) -> list:
    """ONNX Runtime execution providers for ``device``.

    Convolution algorithms are chosen heuristically: the default exhaustive
    search pre-allocates enough cuDNN workspace that the fp32 UNet runs out of
    memory on a 10 GB GPU. GPU memory comes from the process-wide arena of
    :func:`_shared_cuda_arena`, not a per-session one.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return ["CPUExecutionProvider"]
    options = {
        "device_id": device.index or 0,
        "cudnn_conv_algo_search": "HEURISTIC",
    }
    return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]


_REGISTERED_CUDA_ARENAS: set[int] = set()


def _shared_cuda_arena(device_id: int) -> None:
    """Register one CUDA memory arena for every session on ``device_id``.

    By default each session grows its own arena and never returns memory to the
    others, so with the three fp32 SD1.5 sessions resident, the VAE decoder's
    activations no longer fit on a 10 GB GPU. A shared arena lets the VAE reuse
    the memory the UNet just finished with. It grows only as needed. Registering
    is idempotent, and is done once per device per process.

    The arena outlives its sessions, so it keeps what it grew until
    :func:`_release_cuda_arena` hands it back.
    """
    if device_id in _REGISTERED_CUDA_ARENAS:
        return
    import onnxruntime

    memory_info = onnxruntime.OrtMemoryInfo(
        "Cuda",
        onnxruntime.OrtAllocatorType.ORT_ARENA_ALLOCATOR,
        device_id,
        onnxruntime.OrtMemType.DEFAULT,
    )
    arena = onnxruntime.OrtArenaCfg({"arena_extend_strategy": 1})  # kSameAsRequested
    onnxruntime.create_and_register_allocator_v2(
        "CUDAExecutionProvider", memory_info, {"device_id": str(device_id)}, arena
    )
    _REGISTERED_CUDA_ARENAS.add(device_id)


def _release_cuda_arena(session, device_id: int) -> None:
    """Return the shared arena's free memory on ``device_id`` to the GPU.

    ONNX Runtime only shrinks an arena at the end of a run that asks it to, so
    this runs ``session`` once with that option. Without it, the arena keeps up
    to ~6 GB (fp32 SD1.5) after its sessions are gone, and whatever runs next on
    the GPU -- the CLIP model of CLIPScore, the next experiment -- runs out.
    """
    import onnxruntime

    options = onnxruntime.RunOptions()
    options.add_run_config_entry(
        "memory.enable_memory_arena_shrinkage", f"gpu:{device_id}"
    )
    feeds = {
        meta.name: _zeros(meta)  # Any valid input will do; the result is unused.
        for meta in session.get_inputs()
    }
    session.run(None, feeds, options)


def _zeros(meta):
    import numpy as np

    return np.zeros(
        [d if isinstance(d, int) else 1 for d in meta.shape],
        dtype={
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
        }[meta.type],
    )


class ORTComponent(torch.nn.Module):
    """One exported component, called like the torch wrapper it replaces.

    ``forward`` takes and returns torch tensors in the wrapper's argument order.
    The session is built lazily and rebuilt when :meth:`to` moves the component
    to another device. On CUDA, inputs and outputs are bound to GPU memory, so
    tensors never round-trip through the host between denoising steps.
    """

    def __init__(self, path: str | os.PathLike, device: torch.device | str = "cpu"):
        super().__init__()
        self.path = str(path)
        self._device = torch.device(device)
        self._session = None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """The graph's float dtype, from its first floating-point input."""
        for meta in self.session.get_inputs():
            dtype = _ORT_TO_TORCH.get(meta.type)
            if dtype is not None and dtype.is_floating_point and len(meta.shape) > 1:
                return dtype
        return torch.float32

    @property
    def session(self):
        if self._session is None:
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            if self._device.type == "cuda":
                _shared_cuda_arena(self._device.index or 0)
                options.add_session_config_entry("session.use_env_allocators", "1")
            self._session = onnxruntime.InferenceSession(
                self.path, options, providers=providers_for(self._device)
            )
        return self._session

    def to(self, *args, **kwargs):
        device = _device_from_to_args(*args, **kwargs)
        if device is not None and device != self._device:
            self.release()
            self._device = device
        return self

    def release(self) -> None:
        """Drop the session, returning its GPU memory; it is rebuilt on next use."""
        session, self._session = self._session, None
        if session is not None and self._device.type == "cuda":
            _release_cuda_arena(session, self._device.index or 0)

    def eval(self):
        return self

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        session = self.session
        metas = session.get_inputs()
        if len(inputs) != len(metas):
            raise TypeError(
                f"{Path(self.path).parent.name} takes {len(metas)} inputs "
                f"({[m.name for m in metas]}), got {len(inputs)}."
            )
        # The graph fixes each input's dtype; cast to it at the boundary.
        feeds = {
            meta.name: tensor.to(_ORT_TO_TORCH[meta.type]).contiguous()
            for meta, tensor in zip(metas, inputs)
        }
        if self._device.type == "cuda":
            return self._run_on_cuda(session, feeds)
        (output,) = session.run(None, {k: v.cpu().numpy() for k, v in feeds.items()})
        return torch.from_numpy(output)

    def _run_on_cuda(self, session, feeds: dict[str, torch.Tensor]) -> torch.Tensor:
        device_id = self._device.index or 0
        binding = session.io_binding()
        # Bound buffers are raw pointers: keep the tensors alive for the run.
        bound = []
        for name, tensor in feeds.items():
            tensor = tensor.to(f"cuda:{device_id}")
            bound.append(tensor)
            binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=device_id,
                element_type=_numpy_dtype(tensor.dtype),
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )
        (meta,) = session.get_outputs()
        binding.bind_output(meta.name, "cuda", device_id)
        torch.cuda.synchronize(device_id)
        session.run_with_iobinding(binding)
        (output,) = binding.get_outputs()
        # DLPack hands ORT's GPU buffer to torch without a copy.
        return torch.from_dlpack(output._ortvalue.to_dlpack())


def load_components(
    components: DiffusionComponents,
    paths: dict[str, Path],
    device: torch.device | str = "cpu",
) -> DiffusionComponents:
    """``components`` with each torch module replaced by its ONNX session.

    The tokenizer, scheduler and shape metadata are kept from ``components``.
    """
    sessions = {name: ORTComponent(paths[name], device) for name in COMPONENTS}
    return DiffusionComponents(
        **sessions,
        tokenizer=components.tokenizer,
        scheduler=components.scheduler,
        latent_channels=components.latent_channels,
        vae_scale_factor=components.vae_scale_factor,
    )


def _numpy_dtype(dtype: torch.dtype):
    import numpy as np

    return {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }[dtype]


def _device_from_to_args(*args, **kwargs) -> torch.device | None:
    """The device in a ``Module.to(...)`` call, if any (dtypes are ignored)."""
    candidates = [*args, kwargs.get("device")]
    for value in candidates:
        if isinstance(value, (str, torch.device)):
            return torch.device(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return torch.device("cuda", value)
    return None

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization of diffusion components with an aimet_onnx QuantSim.

Each quantized component's exported graph is wrapped in a
:class:`~aimet_onnx.quantsim.QuantizationSimModel` and swapped into
:class:`~GenAILab.qai_hub_lm.diffusion.base.DiffusionComponents` in place of
its ONNX Runtime session, so the same ``DiffusionGenerator`` runs the denoising
loop over a mix of float sessions and quantized sims.

Components are calibrated one at a time, each with every other component in
float, so a component's encodings never depend on another's quantization
error. Each calibration pass drives the generator over held-out prompts as far
as that component needs: the text encoder sees the prompts (and the negative
prompt), the UNet the inputs of every denoising step, conditional and
unconditional, and the VAE decoder the float-denoised latents.
"""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Callable
from pathlib import Path

import onnx
import torch
from tqdm import tqdm

from GenAILab.bench.diffusion.config import ComponentPrecision, DiffusionConfig
from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.qai_hub_lm.diffusion import onnx as diffusion_onnx
from GenAILab.qai_hub_lm.diffusion.base import (
    COMPONENTS,
    TEXT_ENCODER,
    UNET,
    VAE_DECODER,
)
from GenAILab.qai_hub_lm.diffusion.generator import DiffusionGenerator

#: Base dtypes a component QuantSim supports. With an fp16 graph, aimet_onnx
#: raises every Conv/Gemm weight scale until its int32 bias scale is a normal
#: fp16 value (the bias QDQ is exported at the activation dtype). For SD1.5's
#: UNet that inflates half the weight scales by 600x (median) to 10000x and the
#: quantized UNet predicts noise uncorrelated with the float one.
SUPPORTED_DTYPES = (torch.float32,)


def check_dtype(dtype: torch.dtype) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Quantizing diffusion components supports base dtype "
            f"{[str(d).removeprefix('torch.') for d in SUPPORTED_DTYPES]}, not "
            f"{str(dtype).removeprefix('torch.')}: on an fp16 graph, aimet_onnx "
            f"enlarges the weight scales to keep each int32 bias scale "
            f"representable in fp16, which destroys Stable Diffusion's accuracy."
        )


class SimComponent(torch.nn.Module):
    """A component QuantSim, called like the torch wrapper it replaces.

    The sim keeps the device it was built for; it is not moved. Moving the
    component elsewhere only ever happens to free the GPU
    (``DiffusionGenerator.evicted`` and ``on_device`` on exit), so :meth:`to`
    drops the sim's session instead, returning its GPU memory, and the session
    is rebuilt with the sim's own providers on the next call.
    """

    def __init__(self, sim, device: torch.device | str):
        super().__init__()
        self.sim = sim
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def to(self, *args, **kwargs):
        device = diffusion_onnx._device_from_to_args(  # pylint: disable=protected-access
            *args, **kwargs
        )
        if device is not None and device != self._device:
            self.release()
        return self

    def release(self) -> None:
        """Drop the sim's session, returning its GPU memory; it is rebuilt on next use."""
        session, self.sim.session = self.sim.session, None
        if session is not None and self._device.type == "cuda":
            # As for the float sessions: the shared arena keeps what this
            # session grew it to until a run asks it to shrink.
            diffusion_onnx._release_cuda_arena(  # pylint: disable=protected-access
                session, self._device.index or 0
            )
        del session
        gc.collect()

    def eval(self):
        return self

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import (
            mock_torch_onnx_inference,
        )

        if self.sim.session is None:
            self.sim._rebuild_session()  # pylint: disable=protected-access
        return mock_torch_onnx_inference(self.sim.session, *inputs)


def _providers_for(device: torch.device):
    from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import get_ort_providers

    return get_ort_providers(device)


def build_quantsim(path: str | Path, precision: ComponentPrecision, device):
    """A QuantSim over the exported graph at ``path``, with ``precision``.

    Weights are quantized per channel (the HTP config's default for Conv, Gemm
    and MatMul) and every other op output at ``precision.activations``, except
    additive attention masks (see :func:`disable_mask_quantizers`).
    """
    import aimet_onnx.quantsim
    from aimet_onnx.quantsim import QuantizationSimModel

    from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import (
        _disable_float_quantizers,
    )

    model = onnx.load(str(path))
    # Lean ORT sessions, as the LLM runner uses: the fp32 SD1.5 UNet sim does
    # not otherwise fit next to the text encoder and VAE on a 10 GB GPU.
    with aimet_onnx.quantsim._low_memory_ort_sessions_enabled(True):  # pylint: disable=protected-access
        sim = QuantizationSimModel(
            model,
            quant_scheme="min_max",
            param_type=precision.weight.qtype,
            activation_type=precision.activations,
            config_file=QUANTSIM_CONFIG,
            providers=_providers_for(torch.device(device)),
        )
    _disable_float_quantizers(sim)
    disabled = disable_mask_quantizers(sim)
    if disabled:
        print(f"  Left {len(disabled)} attention-mask quantizers unquantized.")
    _use_shared_cuda_arena(sim, torch.device(device))
    return sim


def _use_shared_cuda_arena(sim, device: torch.device) -> None:
    """Build ``sim``'s sessions on the CUDA arena the float sessions share.

    By default each session grows an arena of its own and keeps it, so with the
    sims of all three SD1.5 components resident, the UNet's and VAE decoder's
    peak activations stay allocated side by side and no longer fit on a 10 GB
    GPU. On the shared arena, each component reuses what the last one freed.
    The session the sim was built with is dropped; the next call rebuilds it.
    """
    if device.type != "cuda":
        return
    diffusion_onnx._shared_cuda_arena(device.index or 0)  # pylint: disable=protected-access
    sim._ort_session_options.add_session_config_entry(  # pylint: disable=protected-access
        "session.use_env_allocators", "1"
    )
    sim.session = None
    gc.collect()


def quantize_and_calibrate(
    config: DiffusionConfig,
    generator: DiffusionGenerator,
    paths: dict[str, Path],
    prompts: list[dict],
    device: torch.device,
    meter: Callable[[], contextlib.AbstractContextManager],
) -> dict[str, object]:
    """Replace each component named in ``config.precision`` by its calibrated sim.

    Components are quantized in pipeline order, one at a time: each sim is
    built and calibrated on ``prompts`` while every other component runs in
    float, its float session included until the end. Only then are the sims
    swapped in for good. Each pass runs inside ``meter()``, whose value is
    returned per component (a profiler, for the results).
    """
    components = generator.components
    floats = {name: components.component(name) for name in config.precision}
    sims, profiles = {}, {}
    for name in COMPONENTS:
        if name not in config.precision:
            continue
        precision = config.precision[name]
        print(f"Building the {name} QuantSim ({_describe(precision)})...")
        _release(floats[name])  # The sim stands in for it while calibrating.
        sim = SimComponent(build_quantsim(paths[name], precision, device), device)
        setattr(components, name, sim)
        with generator.on_device(device), meter() as profile:
            calibrate(generator, name, prompts)
        profiles[name] = profile
        sim.release()
        # Restore the float component, so later passes see it unquantized.
        setattr(components, name, floats[name])
        sims[name] = sim
        gc.collect()

    for name, sim in sims.items():
        _release(floats[name])
        setattr(components, name, sim)
    gc.collect()
    return profiles


def calibrate(generator: DiffusionGenerator, name: str, prompts: list[dict]) -> None:
    """Compute the encodings of the sim standing in for component ``name``.

    Runs that component's calibration pass for each prompt, with the generator's
    other components as they are. Prompt ``i`` uses seed
    ``generator.config.seed + i``. The quantizers pass values through while they
    observe them, so the rest of the pipeline sees float outputs.

    The param quantizers are then folded into the weights: evaluation needs no
    gradients, and a folded sim neither re-quantizes every weight on each call
    nor holds a second copy of them on the GPU.
    """
    from aimet_onnx.quantsim import compute_encodings

    sim = generator.components.component(name).sim
    base = generator.config.seed
    seeds = range(base, base + len(prompts))
    with compute_encodings(sim):
        _CALIBRATION_PASSES[name](generator, prompts, seeds)
    sim.fold_param_quantizers()


@torch.inference_mode()
def _encode(generator: DiffusionGenerator, prompts: list[dict], seeds) -> None:
    del seeds  # Encoding is deterministic.
    for prompt in tqdm(prompts, desc="Calibrating text_encoder"):
        generator.encode_prompt(prompt["prompt"])
        if generator.config.do_classifier_free_guidance:
            generator.encode_prompt(generator.config.negative_prompt)


def _denoise(generator: DiffusionGenerator, prompts: list[dict], seeds) -> None:
    for prompt, seed in zip(tqdm(prompts, desc="Calibrating unet"), seeds):
        generator.generate_latents(prompt["prompt"], seed)


def _decode(generator: DiffusionGenerator, prompts: list[dict], seeds) -> None:
    for prompt, seed in zip(tqdm(prompts, desc="Calibrating vae_decoder"), seeds):
        generator.generate(prompt["prompt"], seed)


#: How far each component's calibration pass runs the pipeline: only as far as
#: the component itself.
_CALIBRATION_PASSES: dict[
    str, Callable[[DiffusionGenerator, list[dict], range], None]
] = {
    TEXT_ENCODER: _encode,
    UNET: _denoise,
    VAE_DECODER: _decode,
}


def _release(component) -> None:
    if hasattr(component, "release"):
        component.release()


def _describe(precision: ComponentPrecision) -> str:
    return f"W{precision.weight.qtype.bits}A{precision.activations.bits}"


#: Additive masks fill blocked positions with a large negative number (CLIP's
#: causal mask uses the float32 minimum, -3.4e38). Anything this negative is a mask.
_MASK_THRESHOLD = -1e9


def disable_mask_quantizers(sim) -> list[str]:
    """Leave additive attention masks, and the masked scores, unquantized.

    A quantizer over a mask, or over ``scores + mask``, must cover the mask's
    -3.4e38, which leaves no resolution for the scores: every attention weight
    rounds to the same value, and the text encoder's output no longer resembles
    the float one. The scores before the mask stay quantized. Returns the names
    of the quantizers disabled.
    """
    from onnx import numpy_helper

    graph = sim.model.model.graph
    # In the sim graph, consumers read each quantized tensor through its
    # QcQuantizeOp; map those outputs back to the tensor they quantize.
    source = {
        node.output[0]: node.input[0]
        for node in graph.node
        if node.op_type == "QcQuantizeOp"
    }
    constants = {init.name: init for init in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.attribute:
            constants[node.output[0]] = node.attribute[0].t

    def is_mask(name: str) -> bool:
        tensor = constants.get(source.get(name, name))
        if tensor is None or tensor.data_type not in (
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.FLOAT16,
        ):
            return False
        values = numpy_helper.to_array(tensor)
        return values.size > 1 and float(values.min()) < _MASK_THRESHOLD

    disabled = []
    for node in graph.node:
        if node.op_type != "Add" or not any(is_mask(i) for i in node.input):
            continue
        masks = [source.get(i, i) for i in node.input if is_mask(i)]
        for name in [*masks, *node.output]:
            quantizer = sim.qc_quantize_op_dict.get(name)
            if quantizer is not None and quantizer.enabled and name not in disabled:
                quantizer.enabled = False
                disabled.append(name)
    return disabled

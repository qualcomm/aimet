# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization of diffusion components with an aimet_onnx QuantSim.

Each quantized component's exported graph is wrapped in a
:class:`~aimet_onnx.quantsim.QuantizationSimModel` and swapped into
:class:`~GenAILab.qai_hub_lm.diffusion.base.DiffusionComponents` in place of
its ONNX Runtime session, so the same ``DiffusionGenerator`` runs the denoising
loop over a mix of float sessions and quantized sims. Calibration drives that
generator over held-out prompts, so the UNet's quantizers observe the inputs of
every denoising step, conditional and unconditional.
"""

from __future__ import annotations

import contextlib
import gc
from pathlib import Path

import onnx
import torch
from tqdm import tqdm

from GenAILab.bench.diffusion.config import ComponentPrecision, DiffusionConfig
from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.qai_hub_lm.diffusion import onnx as diffusion_onnx
from GenAILab.qai_hub_lm.diffusion.base import DiffusionComponents
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
            f"representable in fp16, which destroys the UNet's accuracy."
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
        """Drop the sim's session; it is rebuilt on next use."""
        if self.sim.session is not None:
            self.sim.session = None
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
    and MatMul) and every other op output at ``precision.activations``.
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
    return sim


def quantize_components(
    config: DiffusionConfig,
    components: DiffusionComponents,
    paths: dict[str, Path],
    device: torch.device,
) -> DiffusionComponents:
    """``components`` with each component named in ``config.precision`` replaced by its sim.

    The returned sims are uncalibrated; see :func:`calibrate`.
    """
    for name, precision in config.precision.items():
        current = components.component(name)
        if hasattr(current, "release"):
            current.release()  # The sim replaces the float session.
        setattr(components, name, None)
        gc.collect()
        print(f"Building the {name} QuantSim ({_describe(precision)})...")
        sim = build_quantsim(paths[name], precision, device)
        setattr(components, name, SimComponent(sim, device))
    return components


def calibrate(
    generator: DiffusionGenerator, quantized: list[str], prompts: list[dict]
) -> None:
    """Compute the encodings of each sim in ``quantized`` by generating for ``prompts``.

    Every sim calibrates in the same pass. The quantizers pass values through
    while they observe them, so the denoising trajectory is the float one.
    Prompt ``i`` uses seed ``generator.config.seed + i``; only the latents are
    generated, since the VAE decoder is never calibrated.

    The param quantizers are then folded into the weights: evaluation needs no
    gradients, and a folded sim neither re-quantizes every weight on each call
    nor holds a second copy of them on the GPU.
    """
    from aimet_onnx.quantsim import compute_encodings

    sims = [generator.components.component(name).sim for name in quantized]
    base = generator.config.seed
    with contextlib.ExitStack() as stack:
        for sim in sims:
            stack.enter_context(compute_encodings(sim))
        for i, prompt in enumerate(tqdm(prompts, desc="Calibrating")):
            generator.generate_latents(prompt["prompt"], seed=base + i)
    for sim in sims:
        sim.fold_param_quantizers()


def _describe(precision: ComponentPrecision) -> str:
    return f"W{precision.weight.qtype.bits}A{precision.activations.bits}"

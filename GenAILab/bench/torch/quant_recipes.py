# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization recipes for GenAI models using AIMET-Torch"""

from abc import ABC, abstractmethod
import itertools
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

from aimet_torch import QuantizationSimModel
from aimet_torch.adascale.adascale_optimizer import apply_adascale
from aimet_torch.v2.seq_mse import apply_seq_mse
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.v2.utils import remove_all_quantizers
from aimet_torch.utils import change_tensor_device_placement

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.schema import (
    RemoveQuantizationSpec,
    ClipSpec,
    SkipSpec,
    CalibrationSpec,
    SeqMSESpec,
    AdaScaleSpec,
    SpinQuantSpec,
)
from GenAILab.qai_hub_lm.backends.torch.quantsim_utils import apply_spinquant_pre_sim
from GenAILab.qai_hub_lm.models.generator import Generator


class PreQuantizationTechnique(ABC):
    """A technique that runs on the float model BEFORE the sim is built.

    Unlike QuantizationTechnique (which operates on the quantsim), apply() here
    receives the float-model bundle (what ``instantiate_float_model`` returns).
    A single call rotates the whole model (all components together).
    """

    @staticmethod
    @abstractmethod
    def apply(float_model, **kwargs):
        """Apply the technique to the float model, in place."""


@YAMLConfigParser.register_recipe(SpinQuantSpec)
class SpinQuant(PreQuantizationTechnique):
    """Rotate the float model (R1/R2) before the sim is built."""

    @staticmethod
    def apply(
        float_model, *, enable_r1=True, enable_r2=False, enable_r3=False, **kwargs
    ):
        # ``**kwargs`` absorbs the chain's framework-agnostic fixtures (e.g. the
        # ONNX-only ``topology``); torch SpinQuant discovers structure itself.
        apply_spinquant_pre_sim(
            float_model,
            {
                "enable_r1": enable_r1,
                "enable_r2": enable_r2,
                "enable_r3": enable_r3,
            },
        )


def _prefill_inputs(
    generator: Generator,
    dataloader: DataLoader,
    num_iterations: int = None,
    device: torch.device = None,
):
    inputs = []
    if num_iterations is not None:
        dataloader = itertools.islice(dataloader, num_iterations)

    with generator.fp_mode(), torch.no_grad():
        for sample in tqdm(
            dataloader,
            total=num_iterations if num_iterations else len(dataloader),
            desc="Pre-filling calibration data",
        ):
            sample_kwargs = {
                k: v.to(device=generator.device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }
            inputs.extend(
                change_tensor_device_placement(
                    [tuple(d.values()) for d in generator.prefill(**sample_kwargs)],
                    device=device if device else generator.device,
                )
            )

    return inputs


class QuantizationTechnique(ABC):
    """Generic GenAI quantization technique"""

    @classmethod
    def cacheable(cls):
        return False

    @staticmethod
    @abstractmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        **kwargs,
    ):
        """Apply quantization technique"""


@YAMLConfigParser.register_recipe(RemoveQuantizationSpec)
class RemoveQuantization(QuantizationTechnique):
    """Remove all quantization nodes from quantsim model"""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        **kwargs,
    ):
        # Remove all quantization wrappers to get an FP baseline.
        remove_all_quantizers(quantsim.model)


@YAMLConfigParser.register_recipe(ClipSpec)
class Clip(QuantizationTechnique):
    """Clamp activation quantizer encodings to a fixed symmetric range.

    Some models (e.g. Qwen 3.5 linear attention) produce a few activation
    tensors with very wide dynamic range (outliers up to ~3e4). Under int16 the
    resulting step size destroys the small in-range values that carry the
    signal. Clamping the activation quantizers to ``[-value, value]`` saturates
    the (information-free) outliers and gives the in-range values a finer grid.

    Must run AFTER ``Calibration`` (it overrides the calibrated min/max).
    Quantizers already within ``[-value, value]`` are unaffected (the clamp is
    a no-op for them), so only the wide-range outlier tensors are clipped.

    Config (per recipe step)::

        - name: Clip
          value: 1000           # symmetric clip magnitude
    """

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        value: float = 1000.0,
        **kwargs,
    ):
        from aimet_torch.v2.quantization.affine import QuantizeDequantize

        n_clipped = 0
        for module in quantsim.model.modules():
            if not isinstance(module, QuantizeDequantize):
                continue
            if (
                getattr(module, "min", None) is None
                or getattr(module, "max", None) is None
            ):
                continue
            new_min = torch.clamp(module.min, min=-value)
            new_max = torch.clamp(module.max, max=value)
            if torch.equal(new_min, module.min) and torch.equal(new_max, module.max):
                continue  # already in-range; nothing clipped
            module.min.copy_(new_min)
            module.max.copy_(new_max)
            n_clipped += 1
        print(f"Clip: clamped {n_clipped} activation quantizers to [-{value}, {value}]")


@YAMLConfigParser.register_recipe(SkipSpec)
class Skip(QuantizationTechnique):
    """Do nothing. Useful for testing fully precomputed encodings."""

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        **kwargs,
    ):
        pass


@YAMLConfigParser.register_recipe(CalibrationSpec)
class Calibration(QuantizationTechnique):
    """Calibrate quantization parameters.

    Granularity (PCQ/LPBQ/BQ) is configured via the ``precision:`` section.
    This recipe simply runs calibration.
    """

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
        **kwargs,
    ):
        # Calibrate quantization parameters by running data through the model.
        if num_iterations is None:
            num_iterations = len(dataloader)
        sliced_dataloader = itertools.islice(dataloader, num_iterations)

        with compute_encodings(quantsim.model), torch.no_grad():
            for batch in tqdm(
                sliced_dataloader, total=num_iterations, desc="Calibrating"
            ):
                inputs = {
                    k: v.to(device=generator.device)
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in batch.items()
                }
                generator(**inputs)


@YAMLConfigParser.register_recipe(SeqMSESpec)
class SeqMSE(QuantizationTechnique):
    """Apply SeqMSE to model"""

    @classmethod
    def cacheable(cls):
        return True

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
        **kwargs,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(
            generator, dataloader, num_iterations, torch.device("cpu")
        )

        # Step 2: Optimize weight quantization parameters to minimize layer-wise MSE.
        apply_seq_mse(quantsim, inputs, num_candidates=20)


@YAMLConfigParser.register_recipe(AdaScaleSpec)
class AdaScale(QuantizationTechnique):
    """Apply AdaScale to model"""

    @classmethod
    def cacheable(cls):
        return True

    @staticmethod
    @torch.no_grad()
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: Dataset,
        num_batches: int = 20,
        num_iterations: int = 1500,
        **kwargs,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(
            generator, dataloader, num_batches, torch.device("cpu")
        )

        # Step 2: Optimize quantization parameters using AdaScale.
        apply_adascale(
            quantsim,
            inputs,
            num_iterations=num_iterations,
        )

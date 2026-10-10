# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization recipes for GenAI models using AIMET-ONNX"""

from abc import ABC, abstractmethod
from tqdm import tqdm
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.sequential_mse.seq_mse import SeqMseParams, SequentialMse
from aimet_onnx.adascale.adascale_optimizer import (
    apply_adascale,
    adascale_model_config_dict,
)
from aimet_onnx.spinquant import apply_spinquant

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
from GenAILab.qai_hub_lm.models.generator import Generator
from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import kwargs_to_dict


class PreQuantizationTechnique(ABC):
    """A technique that runs on the float model BEFORE the sim is built.

    apply() receives the float-model bundle (what ``instantiate_float_model``
    returns: an entry with ``.backbone`` / ``.visual`` / ``.embedding``). A single
    call rotates the whole model (all components together).
    """

    @staticmethod
    @abstractmethod
    def apply(float_model, **kwargs):
        """Apply the technique to the float model, in place."""


@YAMLConfigParser.register_recipe(SpinQuantSpec)
class SpinQuant(PreQuantizationTechnique):
    """Rotate the float ONNX graph (R1/R2/R3) before the sim is built."""

    @staticmethod
    def apply(
        float_model,
        *,
        enable_r1=True,
        enable_r2=False,
        enable_r3=False,
        topology=None,
        **kwargs,
    ):
        embedding = float_model.embedding
        # Where the rotations go comes from llm_topology, which owns model analysis;
        # SpinQuant only rotates the structure it reports. The topology is analyzed
        # once by the runner on this same (still unrotated) float model.
        apply_spinquant(
            float_model.backbone,
            visual_model=float_model.visual,
            embedding=embedding.weight if embedding is not None else None,
            enable_r1=enable_r1,
            enable_r2=enable_r2,
            enable_r3=enable_r3,
            topology=topology,
        )


def _get_lm_head_node_names(quantsim: QuantizationSimModel) -> list[str]:
    lm_head_node_names = []
    vocab_size = (
        quantsim.model.model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    )
    node_input_map = {
        node.input[1]: node
        for node in quantsim.model.model.graph.node
        if node.op_type in ("Gemm", "MatMul", "Conv")
    }
    for weight in quantsim.model.model.graph.initializer:
        if vocab_size in weight.dims:
            for suffix in ("", "_updated", "_qdq"):
                candidate_name = weight.name + suffix if suffix else weight.name
                if candidate_name in node_input_map:
                    node = node_input_map[candidate_name]
                    lm_head_node_names.append(node.name)
    return lm_head_node_names


def _prefill_inputs(
    quantsim: QuantizationSimModel,
    generator: Generator,
    dataloader: DataLoader,
    num_iterations: int = None,
) -> list[dict[str, np.ndarray]]:
    input_names = [inp.name for inp in quantsim.session.get_inputs()]
    inputs = []
    if num_iterations is not None:
        dataloader = itertools.islice(dataloader, num_iterations)

    def _to_numpy(
        prepared: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> dict[str, np.ndarray]:
        if not isinstance(prepared, dict):
            prepared = kwargs_to_dict(input_names, *prepared)
        return {
            k: v.cpu().detach().numpy()
            for k, v in prepared.items()
            if isinstance(v, torch.Tensor)
        }

    with generator.fp_mode():
        for sample in tqdm(
            dataloader, total=num_iterations, desc="Pre-filling calibration data"
        ):
            sample_kwargs = {
                k: v.to(device=generator.device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }
            # Convert to CPU numpy immediately to free CUDA memory before
            # fp_mode exits and rebuilds the full quantized session.
            inputs.extend([_to_numpy(t) for t in generator.prefill(**sample_kwargs)])

    return inputs


class QuantizationTechnique(ABC):
    """Generic AIMET-ONNX GenAI quantization technique"""

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
        # Remove all quantization nodes from the ONNX model.
        quantsim.model.model = quantsim.remove_quantizers(quantsim.model.model)
        quantsim._rebuild_session()


@YAMLConfigParser.register_recipe(ClipSpec)
class Clip(QuantizationTechnique):
    """Clamp activation quantizer encodings to a fixed symmetric range.

    Some models (e.g. Qwen 3.5 linear attention) produce a handful of
    activation tensors with very wide dynamic range (outliers up to ~3e4).
    Under int16 the resulting step size destroys the small values that carry
    the signal, tanking accuracy. Clamping the activation encodings to
    ``[-value, value]`` saturates the (information-free) outliers and gives the
    in-range values a much finer grid, recovering accuracy.

    Must run AFTER ``Calibration`` (it overrides the calibrated encodings).
    Activations already within ``[-value, value]`` are unaffected (the clamp is
    a no-op for them), so only the wide-range outlier tensors are clipped.

    Config (per recipe step)::

        - name: Clip
          value: 1000           # symmetric clip magnitude
    """

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        value: float = 1000.0,
        **kwargs,
    ):
        activation_names = set(quantsim.activation_names)
        n_clipped = 0
        for name, op in quantsim.qc_quantize_op_dict.items():
            if name not in activation_names or not op.enabled:
                continue
            encs = op.get_encodings()
            if not encs:
                continue
            e = encs[0]
            lo, hi = float(e.min), float(e.max)
            nlo, nhi = max(lo, -value), min(hi, value)
            if (nlo, nhi) == (lo, hi):
                continue  # already in-range; nothing clipped
            # Mutate the existing encoding in place (preserves bw / symmetry /
            # other fields), recomputing only the affine params from the new
            # min/max, then reload it onto the op.
            e.min = nlo
            e.max = nhi
            e.delta = (nhi - nlo) / (2 ** int(e.bw) - 1) if nhi > nlo else 0.0
            e.offset = round(nlo / e.delta) if e.delta else 0
            op.load_encodings([e])
            n_clipped += 1
        quantsim._rebuild_session()
        print(f"Clip: clamped {n_clipped} activation encodings to [-{value}, {value}]")


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
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
        **kwargs,
    ):
        def _forward(_):
            sliced_dataloader = itertools.islice(dataloader, num_iterations)
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

        quantsim.compute_encodings(_forward)


@YAMLConfigParser.register_recipe(SeqMSESpec)
class SeqMSE(QuantizationTechnique):
    @classmethod
    def cacheable(cls):
        return True

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_iterations: int = 20,
        **kwargs,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(quantsim, generator, dataloader, num_iterations)

        # Step 2: Pre-compute param encodings with a real input so that
        # SequentialMse's internal _compute_param_encodings(overwrite=False)
        # finds them already initialized and skips make_dummy_input, which
        # generates random values that break structured inputs like
        # image_grid_thw in vision models.
        quantsim._compute_param_encodings(dummy_input=inputs[0], overwrite=False)

        # Step 3: Optimize weight quantization parameters to minimize layer-wise MSE.
        print("Starting Sequential MSE...")
        params = SeqMseParams(num_batches=num_iterations)
        seq_mse = SequentialMse(
            model=quantsim.model,
            sim=quantsim,
            params=params,
            data_loader=inputs,
            nodes_to_exclude=_get_lm_head_node_names(quantsim),
        )
        seq_mse.apply_seq_mse_algo()


@YAMLConfigParser.register_recipe(AdaScaleSpec)
class AdaScale(QuantizationTechnique):
    """Apply AdaScale to model"""

    @classmethod
    def cacheable(cls):
        return True

    @staticmethod
    def apply(
        quantsim: QuantizationSimModel,
        generator: Generator,
        dataloader: DataLoader,
        num_batches: int = 32,
        num_iterations: int = 64,
        topology=None,
        **kwargs,
    ):
        # Step 1: Collect calibration inputs in FP mode.
        inputs = _prefill_inputs(quantsim, generator, dataloader, num_batches)

        # Step 2: Pre-compute param encodings with a real input (same
        # reason as SeqMSE — avoids make_dummy_input for vision models).
        quantsim._compute_param_encodings(dummy_input=inputs[0], overwrite=False)

        # Step 3: Optimize quantization parameters using AdaScale. The decoder-stack
        # structure comes from llm_topology, which owns model analysis; AdaScale
        # only consumes the block boundaries it reports.
        #
        # ``topology`` was analyzed once by the runner on the float model. A
        # preceding SpinQuant step leaves those boundary names intact (it inserts
        # new tensors and rewires consumers, never renames existing ones), so the
        # same object stays valid here; apply_adascale re-validates every boundary
        # name against the live graph before optimizing.
        apply_adascale(
            quantsim,
            inputs,
            adascale_model_config_dict[generator.config.model_type],
            num_iterations,
            topology=topology,
        )

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Torch generator utils"""

import contextlib
import torch

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.utils import place_model
from aimet_torch.v2.utils import remove_all_quantizers

from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.components import spec
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator


class TorchFPModeMixin:
    """Mixin that provides fp_mode() for Torch QuantSim generators."""

    @contextlib.contextmanager
    def fp_mode(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                remove_all_quantizers(self.sim_collection.backbone.model)
            )
            for name in self.sim_collection.present_components():
                stack.enter_context(
                    remove_all_quantizers(self.sim_collection.component(name).model)
                )
            if self.sim_collection.embedding is not None:
                stack.enter_context(
                    remove_all_quantizers(self.sim_collection.embedding)
                )
            yield


class TorchDevicePlacementMixin:
    """Mixin that provides on_device() for Torch QuantSim generators."""

    @contextlib.contextmanager
    def on_device(self, device: torch.device):
        with contextlib.ExitStack() as stack:
            for sim in self.sim_collection.iter_members():
                if (
                    sim is not None
                    and isinstance(sim, QuantizationSimModel)
                    and isinstance(sim.model, torch.nn.Module)
                ):
                    stack.enter_context(place_model(sim.model, device))
                elif sim is not None and isinstance(sim, torch.nn.Module):
                    stack.enter_context(place_model(sim, device))
            yield


@contextlib.contextmanager
def place_collection(models: SimCollection, device: torch.device):
    """Temporarily place all non-None models in the collection on the specified device."""
    with contextlib.ExitStack() as stack:
        for sim in models.iter_members():
            if (
                sim is not None
                and isinstance(sim, QuantizationSimModel)
                and isinstance(sim.model, torch.nn.Module)
            ):
                stack.enter_context(place_model(sim.model, device))
            elif sim is not None and isinstance(sim, torch.nn.Module):
                stack.enter_context(place_model(sim, device))
        yield


def generator_factory(
    sim_collection: SimCollection,
    generator_cls: type[Generator],
    tokenizer,
    sequence_length,
    context_length,
    visual_output_names=None,
    **model_kwargs,
) -> Generator:
    # Compose the generator class with torch-specific mixins
    mixed_cls = type(
        generator_cls.__name__,
        (TorchFPModeMixin, TorchDevicePlacementMixin, generator_cls),
        {},
    )

    # Any modality component means the multi-modal generator: its backbone
    # consumes inputs_embeds because an encoder's embeddings are fused in.
    if sim_collection.present_components():
        assert issubclass(generator_cls, VLM_Generator)
        if sim_collection.extras:
            model_kwargs.update(sim_collection.extras)
        encoder_models = {
            spec(name).model_attr: sim_collection.component(name).model
            for name in sim_collection.present_components()
        }
        return mixed_cls(
            backbone_model=sim_collection.backbone.model,
            embedding=sim_collection.embedding,
            tokenizer=tokenizer,
            position_id_processor=sim_collection.position_id_processor,
            sequence_length=sequence_length,
            context_length=context_length,
            config=sim_collection.config,
            visual_output_names=visual_output_names,
            sim_collection=sim_collection,
            **encoder_models,
            **model_kwargs,
        )
    return mixed_cls(
        model=sim_collection.backbone.model,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        context_length=context_length,
        sim_collection=sim_collection,
        **sim_collection.extras,
        **model_kwargs,
    )

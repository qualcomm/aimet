# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for instantiating a Generator object in onnx/run_genai.py"""

import contextlib
import gc

import torch

from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.components import VISUAL, spec
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator
from GenAILab.qai_hub_lm.models.utils.layer_cache import _resolve_text_config

from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import TorchONNXInterface

from aimet_onnx.utils import disable_quantizers


class _VisualONNXAdapter(torch.nn.Module):
    """Reassembles flat ONNX visual outputs into the list structure expected
    by the generator (e.g. deepstack_visual_embeds as a list of tensors)."""

    def __init__(self, interface: TorchONNXInterface, num_list_outputs: int):
        super().__init__()
        self.interface = interface
        self.num_list_outputs = num_list_outputs

    @property
    def config(self):
        return self.interface.config

    @property
    def device(self):
        return self.interface.device

    @property
    def dtype(self):
        return self.interface.dtype

    @property
    def quantsim(self):
        return self.interface.quantsim

    def forward(self, *args, **kwargs):
        outputs = self.interface(*args, **kwargs)
        if self.num_list_outputs > 0 and isinstance(outputs, (list, tuple)):
            base = list(outputs[: -self.num_list_outputs])
            tail = list(outputs[-self.num_list_outputs :])
            return tuple(base + [tail])
        return outputs


class ONNXFPModeMixin:
    """Mixin that provides fp_mode() for ONNX QuantSim generators."""

    @contextlib.contextmanager
    def fp_mode(self):
        with contextlib.ExitStack() as stack:
            sim = self.sim_collection.backbone
            stack.enter_context(disable_quantizers(sim, sim.qc_quantize_op_dict.keys()))
            for name in self.sim_collection.present_components():
                sim = self.sim_collection.component(name)
                stack.enter_context(
                    disable_quantizers(sim, sim.qc_quantize_op_dict.keys())
                )
            yield


class ONNXDevicePlacementMixin:
    """Mixin that provides on_device() for ONNX QuantSim generators.

    Swaps execution providers and rebuilds sessions to move inference to the
    target device. Currently supports moving from CUDA to CPU.
    """

    @contextlib.contextmanager
    def on_device(self, device):
        device = torch.device(device)
        sims = [self.sim_collection.backbone]
        sims.extend(
            self.sim_collection.component(name)
            for name in self.sim_collection.present_components()
        )

        original_providers = [sim.providers for sim in sims]
        target_providers = _providers_for_device(device)

        if all(
            _current_device_type(providers) == device.type
            for providers in original_providers
        ):
            yield
            return

        # The generator's own torch tensors belong to no sim, so swapping
        # providers alone leaves them pinned on the GPU for the whole context.
        # For Gemma4 that is the two embedding tables — ~14 GB in fp32, enough
        # to push a 35B grader into host offload and slow grading ~10x.
        #
        # Read the device off a parameter rather than self.device, which reports
        # cuda whenever CUDA is available regardless of where the tensors are.
        # Defaults to the target so a generator with no parameters of its own
        # restores to a no-op rather than to None.
        params = list(self.parameters())
        torch_device = params[0].device if params else device

        try:
            for sim in sims:
                sim.providers = target_providers
                sim._rebuild_session()
            self.to(device)
            gc.collect()
            torch.cuda.empty_cache()
            yield
        finally:
            # Nested so the torch tensors go back even if a session rebuild
            # raises; otherwise a failure here leaves the generator on CPU.
            try:
                for sim, orig in zip(sims, original_providers):
                    sim.providers = orig
                    sim._rebuild_session()
            finally:
                self.to(torch_device)


def _current_device_type(providers) -> str:
    for p in providers:
        name = p if isinstance(p, str) else p[0]
        if name == "CUDAExecutionProvider":
            return "cuda"
    return "cpu"


def _providers_for_device(device: torch.device):
    if device.type == "cuda":
        if device.index is not None:
            return [
                ("CUDAExecutionProvider", {"device_id": device.index}),
                "CPUExecutionProvider",
            ]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def generator_factory(
    sim_collection: SimCollection,
    generator_cls: type[Generator],
    tokenizer,
    sequence_length,
    context_length,
    visual_output_names=None,
    **model_kwargs,
) -> Generator:
    # Compose the generator class with ONNX-specific mixins
    mixed_cls = type(
        generator_cls.__name__,
        (ONNXFPModeMixin, ONNXDevicePlacementMixin, generator_cls),
        {},
    )

    # Any modality component means the multi-modal generator: its backbone
    # consumes inputs_embeds because an encoder's embeddings are fused in.
    if sim_collection.present_components():
        assert issubclass(generator_cls, VLM_Generator)
        encoder_models = {}
        for name in sim_collection.present_components():
            interface = TorchONNXInterface(
                sim_collection.component(name), sim_collection.config
            )
            if name == VISUAL.name:
                # Wrap vision interface to reassemble list outputs (e.g. deepstack)
                vis_cfg = getattr(sim_collection.config, "vision_config", None)
                ds_indexes = getattr(vis_cfg, "deepstack_visual_indexes", None)
                if ds_indexes:
                    interface = _VisualONNXAdapter(
                        interface, num_list_outputs=len(ds_indexes)
                    )
            encoder_models[spec(name).model_attr] = interface
        if sim_collection.extras:
            model_kwargs.update(sim_collection.extras)
        return mixed_cls(
            backbone_model=TorchONNXInterface(
                sim_collection.backbone, _resolve_text_config(sim_collection.config)
            ),
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
    else:
        return mixed_cls(
            model=TorchONNXInterface(sim_collection.backbone, sim_collection.config),
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            context_length=context_length,
            sim_collection=sim_collection,
            **sim_collection.extras,
            **model_kwargs,
        )

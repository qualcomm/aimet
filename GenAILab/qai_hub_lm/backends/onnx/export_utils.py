# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for exporting models from ONNX to Torch"""

import contextlib
import inspect
import os
from collections.abc import Sequence
from pathlib import Path
from packaging.version import Version
import torch
import onnx
import glob
from transformers import AutoConfig
from huggingface_hub import HfApi

from GenAILab.qai_hub_lm.transforms.exportable_moe import (
    forced_expert_activation,
)

ONNX_OPSET_VERSION = 18

#: ``torch.export`` mode for ``_dynamo_export``. Each mode has a torch.cond bug in a
#: different release range, so pick the one that is correct for the installed torch:
#:   * torch < 2.11: non-strict fails on a cond in a loop over modules that share a
#:     submodule ("... is already tracked for mutation"; pytorch#161053).
#:   * torch >= 2.13: strict silently keeps the stale value of an attribute reassigned
#:     after a cond (e.g. the KV cache's ``keys``), so attention sees the old length.
STRICT_EXPORT = Version(torch.__version__).release < (2, 13)


def is_huggingface_ckpt(model_id: str) -> bool:
    if os.path.isdir(model_id):
        return False
    hf_api = HfApi()
    try:
        _ = hf_api.model_info(model_id)
        return True
    except Exception:
        return False


def get_model_checkpoint_path(model_id: str, classname: str | None = None) -> str:
    if is_huggingface_ckpt(model_id):
        # user has passed in a huggingface checkpoint, use default framework cache path
        if classname is not None:
            return f"onnx_checkpoints/{model_id}/{classname}"
        return f"onnx_checkpoints/{model_id}"
    else:
        # user has passed in a local path, verify that .onnx file and .config files exist and just return the path
        if not os.path.isdir(model_id):
            raise RuntimeError(
                f"Provided model_id '{model_id}' is not a valid HuggingFace model ID or a local directory."
            )

        if not any(
            filename.name.endswith(".onnx") for filename in Path(model_id).rglob("*")
        ):
            raise RuntimeError(
                f"No .onnx file found in the provided local directory '{model_id}'.'"
            )

        if not any(
            filename.name == "config.json" for filename in Path(model_id).rglob("*")
        ):
            raise RuntimeError(
                f"No config.json file found in the provided local directory '{model_id}'.'"
            )

        return model_id


def equivalent_configs(config_a, config_b) -> bool:
    config_dict_a = config_a.to_dict()
    config_dict_b = config_b.to_dict()
    del config_dict_a["_name_or_path"]
    del config_dict_b["_name_or_path"]
    return config_dict_a == config_dict_b


def get_opset(filepath):
    if not os.path.exists(filepath):
        raise RuntimeError(f"File `{filepath}` does not exist.")

    model = onnx.ModelProto()
    with open(filepath, "rb") as f:
        # Only parse specific fields (field number 8 = opset_import)
        model.MergeFromString(f.read())

    return {op.domain or "ai.onnx": op.version for op in model.opset_import}


def check_opset_equal_to(filepath, opset_version: int) -> bool:
    opset = get_opset(filepath)
    return opset.get("ai.onnx", -1) == opset_version


def _load_embedding_pth(
    path: str | os.PathLike, *, freeze: bool
) -> torch.nn.Embedding | None:
    """Load a torch-exported embedding weight (<name>.pth) into an nn.Embedding.

    Returns None if the file does not exist. Accepts either a raw 2D weight tensor
    or a state_dict with a "weight" key (both forms the torch export may write).
    """
    if not os.path.exists(path):
        return None
    weights = torch.load(path, map_location="cpu")
    if isinstance(weights, dict) and "weight" in weights:
        weights = weights["weight"]
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        raise ValueError(f"Expected a 2D embedding tensor in {path}")
    embedding = torch.nn.Embedding.from_pretrained(weights, freeze=freeze)
    return embedding.to("cuda" if torch.cuda.is_available() else "cpu")


def load_model_components_from_disk(
    checkpoint: str | os.PathLike,
    context_length: int,
    sequence_length: int | str,
) -> tuple[
    onnx.ModelProto,
    onnx.ModelProto | None,
    torch.nn.Embedding | None,
    dict[str, torch.nn.Module],
]:
    candidates = [
        os.path.join(
            checkpoint, f"model_seqlen{sequence_length}_cl{context_length}.onnx"
        ),
        os.path.join(
            checkpoint, "backbone", f"model_sl{sequence_length}_cl{context_length}.onnx"
        ),
    ]
    if sequence_length != "dynamic":
        candidates.append(
            os.path.join(
                checkpoint, "backbone", f"model_sldynamic_cl{context_length}.onnx"
            )
        )

    backbone_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])
    backbone = onnx.load(backbone_path)

    visual_path = os.path.join(checkpoint, "visual", "model.onnx")
    visual = onnx.load(visual_path) if os.path.exists(visual_path) else None

    embedding = _load_embedding_pth(
        os.path.join(checkpoint, "embedding.pth"), freeze=False
    )

    # Extra auxiliary modules the export saved under extras/ (e.g. Gemma4's per-layer-input embedding)
    extras: dict[str, torch.nn.Module] = {}
    extras_dir = os.path.join(checkpoint, "extras")
    if os.path.isdir(extras_dir):
        for pth in sorted(glob.glob(os.path.join(extras_dir, "*.pth"))):
            mod = _load_embedding_pth(pth, freeze=True)
            if mod is not None:
                extras[os.path.splitext(os.path.basename(pth))[0]] = mod

    return backbone, visual, embedding, extras


@contextlib.contextmanager
def _unique_initializer_names():
    """Stop a same-name initializer registration from evicting one still in use.

    onnxscript's rewriter commits new initializers with a bare
    ``initializers[name] = value`` (the guard above it in ``_rewrite_rule.py`` is
    dead code), which unparents the ``Value`` an earlier node still references.
    Min/Max->Clip names its bounds ``f"{input_name}_min"``, so a tensor clamped
    twice collides -- Gemma4's audio tower clamps each block three times.
    Uniquifying keeps every bound reachable and lets CSE fold the duplicate Clips.
    """
    from onnx_ir._graph_containers import GraphInitializers

    original = GraphInitializers.__setitem__

    def patched(self, key, value):
        existing = self.data.get(key)
        if existing is not None and existing is not value:
            suffix = 1
            while f"{key}_{suffix}" in self.data:
                suffix += 1
            key = f"{key}_{suffix}"
            value.name = key
        original(self, key, value)

    GraphInitializers.__setitem__ = patched
    try:
        yield
    finally:
        GraphInitializers.__setitem__ = original


def dynamic_axes_to_dynamic_shapes(
    input_names: Sequence[str],
    dynamic_axes: dict[str, dict[int, str]],
):
    """Lower a name-keyed ``dynamic_axes`` dict to a ``dynamic_shapes`` spec.

    Models declare dynamism once, as ``dynamic_axes``
    (``{input_name: {dim_index: symbol}}``). The TorchScript tracer consumes that
    directly; dynamo wants a spec positionally aligned with the inputs. This is
    the only place that converts, so the choice of tracer stays orthogonal to how
    a model declares its axes.

    Every declared axis becomes ``Dim.AUTO``, leaving ``torch.export`` to recover
    the relationships between axes from the traced guards. That is both less for
    us to state and more than we *could* state: the public ``Dim`` arithmetic
    admits only increasing integer-linear derivations, so the relation this
    codebase actually needs -- attention KV states holding
    ``context_length - sequence_length`` entries -- cannot be written by hand
    (``NotImplementedError: Attempted to negate ...``), while declaring the two
    axes independently is rejected outright as a constraint violation. ``AUTO``
    infers it correctly.

    Entries for outputs (``logits``) are ignored -- only ``input_names`` matters,
    and the result is positionally aligned with it.
    """
    return tuple(
        {dim: torch.export.Dim.AUTO for dim in dynamic_axes[name]}
        if dynamic_axes.get(name)
        else None
        for name in input_names
    )


def _match_dynamic_shapes_to_signature(model, dynamic_shapes):
    """Reshape a per-input ``dynamic_shapes`` tuple to mirror ``forward``'s params.

    ``dynamic_shapes`` is produced one entry per input tensor, but
    ``torch.export`` matches it against the *pytree of the call args*. A forward
    declared ``forward(self, *args)`` -- which is how the exportable wrappers here
    take their flattened inputs -- is a single ``VAR_POSITIONAL`` parameter, so a
    flat 67-tuple is one level too shallow and export rejects it with
    "`inputs` has 1 elements, but `dynamic_shapes` has 67 elements".

    Keying by parameter name sidesteps the nesting question: the var-positional
    parameter collects the remaining specs as a tuple, and ordinary parameters
    take one each.
    """
    if dynamic_shapes is None:
        return None

    remaining = list(dynamic_shapes)
    matched: dict[str, object] = {}
    for param in inspect.signature(model.forward).parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            matched[param.name] = tuple(remaining)
            remaining = []
            break
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if not remaining:
            break
        matched[param.name] = remaining.pop(0)

    if remaining:
        raise ValueError(
            f"{type(model).__name__}.forward accepts fewer inputs than the "
            f"{len(dynamic_shapes)} dynamic-shape entries built for it; "
            f"{len(remaining)} left unmatched."
        )
    return matched


def _materialize_view_inputs(sample_input):
    """Replace view tensors among the sample inputs with materialized copies.

    A tensor that is a view of a larger one (``t._base is not None``) makes
    ``torch.export`` emit a shape guard against the *base's* extent, e.g.::

        Guard failed: args_2.size()[1] < args_2._base.size()[1]

    ``run_decompositions`` then re-traces through AOT autograd with materialized
    tensors, whose ``_base`` is ``None``, so the generated guard evaluates
    ``None.size()`` and the export dies with "'NoneType' object has no attribute
    'size'" -- reported as a failure to decompose the FX graph, which points
    nowhere near the real cause. Only dynamic-shape exports emit those guards,
    which is why static export never tripped on it.

    ``Generator.prepare_inputs`` returns views quite legitimately -- for instance
    ``position_ids`` is a window onto a context-length ``arange`` -- so normalize
    here, where the export-specific hazard lives, rather than there.
    """
    return tuple(
        tensor.detach().clone()
        if isinstance(tensor, torch.Tensor) and tensor._base is not None
        else tensor
        for tensor in sample_input
    )


def _dynamo_export(
    model,
    sample_input,
    path,
    *,
    input_names,
    output_names,
    opset_version,
    dynamic_shapes=None,
):
    """Run a dynamo-based ONNX export.

    Produces an ExportedProgram, then hands it to ``torch.onnx.export`` which
    skips the capture step and goes straight to ONNX translation.

    Strict mode traces the forward with dynamo (TorchDynamo bytecode analysis);
    non-strict runs it as ordinary Python and only traces tensor ops. One mode is
    used per torch version, with no fallback (see ``STRICT_EXPORT``). Code either
    mode cannot trace must be rewritten for it -- see Gemma4's audio mask.

    :param dynamic_shapes: Optional ``torch.export`` dynamic-shape spec,
        positionally aligned with ``sample_input``.  This is the dynamo-path
        counterpart of ``dynamic_axes`` (which ``torch.onnx.export`` honours
        only on the TorchScript path); without it the exported graph is fully
        static regardless of any ``dynamic_axes`` passed alongside.
    """
    matched_shapes = _match_dynamic_shapes_to_signature(model, dynamic_shapes)
    sample_input = _materialize_view_inputs(sample_input)
    program = torch.export.export(
        model, sample_input, dynamic_shapes=matched_shapes, strict=STRICT_EXPORT
    )

    with _unique_initializer_names():
        torch.onnx.export(
            program,
            (),  # args ignored for ExportedProgram
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=True,
        )


def consolidate_external_data(path: str) -> onnx.ModelProto:
    """Collapse a fresh export's loose tensor files into one ``model.data`` blob.

    The tracer drops one file per large initializer beside the graph. Those are
    removed, the graph is rewritten with a single external-data file, and the
    weights are read back so the returned proto is self-contained.
    """
    directory = os.path.dirname(path)

    # Load first: torchscript writes one external reference per initializer, so
    # deleting the strays before this leaves them dangling and onnx.load raises.
    model = onnx.load(path)
    for extension in ("*.weight", "*.bias", "onnx__*", "*__value"):
        for stray in glob.glob(os.path.join(directory, extension)):
            os.remove(stray)

    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
    )
    onnx.external_data_helper.load_external_data_for_model(model, directory)
    return model


def get_onnx_model(
    checkpoint: str | os.PathLike,
    fp_backbone_model: torch.nn.Module,
    context_length: int,
    sequence_length: int | list[int],
    sample_input: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    dynamo: bool = False,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> tuple[onnx.ModelProto, bool]:
    """Export (or reuse) the backbone graph.

    Returns the graph and whether it was re-exported. Callers export the
    modality components themselves and pass that flag through as ``force``, so a
    config change re-exports every component rather than leaving stale encoders
    beside a fresh backbone.
    """
    # TODO: Always enable dynamic shape export unconditionally.
    use_dynamic = isinstance(sequence_length, list) and len(sequence_length) > 1
    if not use_dynamic:
        dynamic_axes = None

    if isinstance(sequence_length, list):
        sequence_length = max(sequence_length)

    sl_tag = "dynamic" if use_dynamic else str(sequence_length)

    # Create the checkpoint directory if it does not exist.
    os.makedirs(checkpoint, exist_ok=True)
    onnx_backbone_path = os.path.join(
        checkpoint, "backbone", f"model_sl{sl_tag}_cl{context_length}.onnx"
    )
    config_path = os.path.join(checkpoint, "config.json")

    fp_backbone_model.eval()
    fp_backbone_model.train(False)

    # re-export model if model/config is not found on disk OR if config on disk does not match model config
    if (
        not os.path.exists(onnx_backbone_path)
        or not os.path.exists(config_path)
        or not equivalent_configs(
            AutoConfig.from_pretrained(config_path), fp_backbone_model.config
        )
        or not check_opset_equal_to(onnx_backbone_path, ONNX_OPSET_VERSION)
    ):
        print("Exporting model(s) to ONNX...")
        fp_backbone_model.to(torch.device("cpu"))

        fp_backbone_model.config.save_pretrained(checkpoint)
        # The stock sparse expert path is not exportable (data-dependent
        # shapes); export under the configured realizer. No-op for non-MoE.
        with (
            forced_expert_activation(fp_backbone_model, phase="export"),
            torch.no_grad(),
        ):
            os.makedirs(os.path.join(checkpoint, "backbone"), exist_ok=True)
            print(
                "Backbone exporting..." + (" (dynamo)" if dynamo else " (torchscript)")
            )
            if dynamo:
                _dynamo_export(
                    fp_backbone_model,
                    sample_input,
                    onnx_backbone_path,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=ONNX_OPSET_VERSION,
                    # Same declaration as dynamic_axes below, lowered to the
                    # form dynamo takes.
                    dynamic_shapes=(
                        dynamic_axes_to_dynamic_shapes(input_names, dynamic_axes)
                        if dynamic_axes
                        else None
                    ),
                )
            else:
                torch.onnx.export(
                    fp_backbone_model,
                    sample_input,
                    onnx_backbone_path,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=ONNX_OPSET_VERSION,
                    dynamo=False,
                    dynamic_axes=dynamic_axes,
                )
        print("Loading ONNX model(s)...")
        return consolidate_external_data(onnx_backbone_path), True

    print("Loading cached ONNX model...")
    backbone, *_ = load_model_components_from_disk(
        checkpoint, context_length=context_length, sequence_length=sequence_length
    )
    return backbone, False

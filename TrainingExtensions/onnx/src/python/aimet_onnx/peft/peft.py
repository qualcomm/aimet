# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
LoRA for ONNX models.

Each attached layer gets one concatenated branch: ``y = base(x) + ((x @ A_cat) * s) @ B_cat``.

LoRA tensor names
-----------------
Callers that need the LoRA tensors (e.g. to quantize them) get them with
``get_lora_tensor_names``, which finds them by name. Every name is
generated from the layer's module path (``model.layers.0.self_attn.q_proj``) and its node prefix
(``/model/layers.0/self_attn/q_proj``, shared with the base layer's node)::

    x -> a_proj -> scale -> b_proj -> merge -> merged_output
    base layer output ---------------^

==================  ===============================================================
Role                Name
==================  ===============================================================
A weight            ``<module path>.lora_A.weight`` (initializer)
B weight            ``<module path>.lora_B.weight`` (initializer)
scale weight        ``<module path>.lora_scale`` (initializer)
a_proj node         ``<node prefix>/lora_A/MatMul`` (``Conv`` for conv)
scale node          ``<node prefix>/lora/Mul``
b_proj node         ``<node prefix>/lora_B/MatMul`` (``Conv`` for conv)
merge node          ``<node prefix>/lora/Add``
node outputs        ``<node name>_output_0`` for each of the four nodes
merged output       ``<node prefix>/lora/Add_output_0``, or the original graph output
                    name when the base layer feeds a graph output
base layer output   the unchanged tensor feeding the merge (the bias Add's output for a
                    linear with bias; ``<node prefix>/lora/base_output_0`` when it is
                    renamed because it fed a graph output)
==================  ===============================================================

In a model built by ``load_lora_adapters_as_inputs`` the A and B weights have the same names but
are graph inputs, and every branch's scale is the one graph input
``lora_scale``.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnx
import onnx_ir
from safetensors import safe_open

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.peft.config import (
    WEIGHTS_FILE,
    AdapterLayout,
    checkpoint_key,
    read_adapter,
)
from aimet_onnx.peft import peft_utils

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


def load_lora_adapters(
    model: onnx.ModelProto,
    adapters: Sequence[tuple[str, str | Path]],
    adapter_scales: Sequence[tuple[str, float]] | None = None,
    *,
    pad_rank: bool = False,
) -> onnx.ModelProto:
    """
    Attach LoRA adapters to a base ONNX model as one concatenated branch per layer.

    Each adapter owns a slice of the rank axis, and the scale ``s`` holds
    ``adapter_scales * alpha / r`` per rank. Every adapter passed in is active; to run with a
    different set or scaling, call this again on the base model. The order of ``adapters`` sets
    each adapter's rank slice, so use the same order every time if encodings are reused.

    Adapter layers are found by module path (``model.layers.0.self_attn.q_proj``) matched against
    the node names. If the base was exported with an extra wrapper module, so that every path
    differs from the adapter's by one shared prefix, that prefix is ignored and logged.

    Adapter layers the base doesn't have (e.g. a 28-layer adapter on a 2-layer base) are skipped
    with a warning, as torch peft does.

    A layer's branch rank is the sum of the ranks of the adapters that target it. With
    ``pad_rank``, every branch instead gets the sum of all the adapters' ranks, with zeros after
    the slices of the adapters that target it (as peft ``add_weighted_adapter(combination_type=
    "cat")``). The output is the same; layers targeted by only some adapters cost more.

    The returned model keeps the base's external-data locations (relative paths), so save it next
    to the base's weight files or copy them along.

    For dynamo exports, run ``fix_node_names_pass`` on the model first. For large models, load the
    base with ``onnx.load(path, load_external_data=False)`` so its weights are not copied.

    :param model: Base model without LoRA branches. Not modified.
    :param adapters: ``(name, PEFT checkpoint directory)`` pairs (the directory holds
        ``adapter_model.safetensors`` and ``adapter_config.json``), in rank-slice order. Names must
        be unique.
    :param adapter_scales: ``(name, multiplier)`` pairs, in the same form as ``adapters``. The
        multiplier is finite and applies to the checkpoint's ``alpha / r``. Adapters not listed
        get 1.0.
    :param pad_rank: Give every branch the same rank, the sum of all the adapters' ranks
    :return: A new model with the LoRA branches
    :raises ValueError: If an adapter or scaling value is invalid, none of the adapter layers are
        in the model, a layer doesn't fit the adapter, or ``model`` already contains LoRA branches
    """
    return _load_lora_adapters(
        model, adapters, adapter_scales=adapter_scales, pad_rank=pad_rank
    )[0]


def load_lora_adapters_as_inputs(
    model: onnx.ModelProto, adapters: Sequence[tuple[str, str | Path]]
) -> tuple[onnx.ModelProto, dict[str, dict[str, np.ndarray]]]:
    """
    Add LoRA branches whose weights are graph inputs, and read every adapter's values for them.

    The branches are the ones ``load_lora_adapters(model, adapters[:1])`` adds (no
    ``adapter_scales``, no ``pad_rank``), but their ``A`` and ``B`` are graph inputs named like its
    initializers, and every branch reads its scale from one ``[1]`` graph input, ``lora_scale``.
    One model then serves every adapter: merge an adapter's inputs into a run's inputs to run it,
    e.g. a different adapter per calibration batch. Merge them after batching: the LoRA inputs have
    no batch dimension. To run the model without an adapter, feed zeros, e.g. ``np.zeros_like`` of
    every value of one adapter's inputs.

    Every adapter must be compatible with the first: the same checkpoint keys, shapes and dtypes,
    so the same target layers and rank. Their alphas may differ. Linear ``A`` and ``B`` are
    transposed into the graph's layout, and ``lora_scale`` is the adapter's ``alpha / r``.

    :param model: Base model without LoRA branches. Not modified.
    :param adapters: ``(name, PEFT checkpoint directory)`` pairs. Names must be unique. The
        adapters' values are not in the model.
    :return: The model, and adapter name -> its inputs (input name -> value, for every ``A`` and
        ``B`` input and ``lora_scale``)
    :raises ValueError: If an adapter is invalid or incompatible with the first, or
        ``load_lora_adapters`` raises for the first
    """
    adapters = _check_adapters(adapters)
    _check_compatible(adapters)
    # exactly load_lora_adapters with the first adapter: scale 1.0, no rank padding
    lora_model, checkpoint_paths = _load_lora_adapters(
        model, adapters[:1], adapter_scales=None, pad_rank=False
    )
    mapping = _pull_to_input(lora_model, checkpoint_paths)
    adapter_inputs = {
        name: _load_lora_input(name, ckpt_dir, mapping) for name, ckpt_dir in adapters
    }
    return lora_model, adapter_inputs


def _load_lora_input(
    name: str, adapter: str | Path, mapping: Mapping[str, str]
) -> dict[str, np.ndarray]:
    """
    Read an adapter's values for the LoRA inputs of a ``_pull_to_input`` model.

    Linear ``A`` and ``B`` are transposed into the graph's layout; conv ones are used as they are.
    ``lora_scale`` is ``alpha / r``.

    :param name: The adapter's name, used in messages
    :param adapter: PEFT checkpoint directory, compatible with the model's adapter
    :param mapping: Input name -> checkpoint key, as ``_pull_to_input`` returns it
    :return: Input name -> value, for every input in ``mapping`` and ``lora_scale``
    :raises ValueError: If the adapter lacks a key in ``mapping``
    """
    info = read_adapter(name, adapter)
    inputs = {}
    with safe_open(str(Path(adapter) / WEIGHTS_FILE), framework="np") as ckpt:
        keys = set(ckpt.keys())
        missing = sorted(set(mapping.values()) - keys)
        if missing:
            raise ValueError(
                f"Adapter {info.name!r} has no {missing[0]!r}; it doesn't target the layers the "
                "model was built for"
            )
        extra = keys - set(mapping.values())
        if extra:
            _logger.warning(
                "Adapter %r: %d weight(s) have no model input and are not used, e.g. %s",
                info.name,
                len(extra),
                sorted(extra)[0],
            )
        for name, key in mapping.items():
            tensor = ckpt.get_tensor(key)
            # peft linear A [r, d] / B [m, r] -> the graph's [d, r] / [r, m]
            inputs[name] = (
                np.ascontiguousarray(tensor.T) if tensor.ndim == 2 else tensor
            )
    dtype = next(iter(inputs.values())).dtype
    inputs[peft_utils.LORA_SCALE_INPUT] = np.array([info.alpha / info.r], dtype)
    return inputs


def _load_lora_adapters(
    model: onnx.ModelProto,
    adapters: Sequence[tuple[str, str | Path]],
    adapter_scales: Sequence[tuple[str, float]] | None,
    pad_rank: bool,
) -> tuple[onnx.ModelProto, dict[str, str]]:
    """``load_lora_adapters``, also returning graph module path -> the adapters' path."""
    adapters = _check_adapters(adapters)
    names = [name for name, _ in adapters]
    adapter_scales = _resolve_adapter_scales(names, adapter_scales)
    layout = AdapterLayout.build(
        [read_adapter(name, d) for name, d in adapters], pad_rank=pad_rank
    )
    ir_model = onnx_ir.from_proto(model)
    peft_utils.check_no_branches(
        ir_model.graph
    )  # before reading any checkpoint weights
    points = peft_utils.resolve_targets(ir_model.graph, layout.slices_by_path)
    layout = layout.rekeyed({p.checkpoint_path: path for path, p in points.items()})
    weights = peft_utils.build_branch_weights(
        layout, points, dict(adapters), adapter_scales
    )
    peft_utils.insert_branches(ir_model.graph, points, weights)
    checkpoint_paths = {path: p.checkpoint_path for path, p in points.items()}
    return onnx_ir.to_proto(ir_model), checkpoint_paths


def _pull_to_input(
    model: onnx.ModelProto, checkpoint_paths: Mapping[str, str]
) -> dict[str, str]:
    """
    Turn the LoRA weights of a ``load_lora_adapters`` model into graph inputs, in place.

    Every branch's ``A`` and ``B`` initializers become graph inputs of the same name and shape, and
    every branch's scale initializer is replaced by one ``[1]`` graph input, ``lora_scale``.

    :param model: Model built from one adapter, so that every scale is the same ``alpha / r``
    :param checkpoint_paths: Graph module path -> the adapter's path for the layer
    :return: Input name -> checkpoint key, for every ``A`` and ``B`` input
    """
    mapping = {}
    for branch in peft_utils.pull_branch_weights_to_inputs(model):
        path = checkpoint_paths.get(branch.module_path, branch.module_path)
        mapping[branch.a_weight] = checkpoint_key(path, "A")
        mapping[branch.b_weight] = checkpoint_key(path, "B")
    return mapping


def _check_compatible(adapters: Sequence[tuple[str, str | Path]]) -> None:
    """Raise unless every adapter has the first one's weight keys, shapes and dtypes."""
    signatures = []
    for name, ckpt_dir in adapters:
        read_adapter(name, ckpt_dir)  # validates the config and weight keys
        with safe_open(str(Path(ckpt_dir) / WEIGHTS_FILE), framework="np") as ckpt:
            signatures.append(
                {
                    key: (
                        tuple(ckpt.get_slice(key).get_shape()),
                        ckpt.get_slice(key).get_dtype(),
                    )
                    for key in ckpt.keys()
                }
            )
    template, (first, _) = signatures[0], adapters[0]
    for (name, _), signature in zip(adapters[1:], signatures[1:]):
        differ = sorted(set(template) ^ set(signature))
        if differ:
            raise ValueError(
                f"Adapter {name!r} doesn't target the same layers as {first!r}, e.g. {differ[0]!r}"
            )
        for key, (shape, dtype) in signature.items():
            if (shape, dtype) != template[key]:
                raise ValueError(
                    f"Adapter {name!r}: {key} is {dtype} {shape} but in {first!r} it is "
                    f"{template[key][1]} {template[key][0]}; the adapters must have the same rank "
                    "and dtype"
                )


def _check_adapters(
    adapters: Sequence[tuple[str, str | Path]],
) -> list[tuple[str, str | Path]]:
    """Check ``adapters`` is a non-empty sequence of uniquely named pairs; return it as a list.

    The order is kept: it sets each adapter's rank slice.
    """
    adapters = list(adapters)
    if not adapters:
        raise ValueError("adapters must name at least one adapter")
    names = [name for name, _ in adapters]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"adapters names must be unique, got duplicates: {duplicates}")
    return adapters


def _resolve_adapter_scales(
    names: Sequence[str], adapter_scales: Sequence[tuple[str, float]] | None
) -> dict[str, float]:
    """Check ``adapter_scales`` and fill in 1.0 for every adapter that is not listed."""
    adapter_scales = dict(adapter_scales or ())
    unknown = sorted(set(adapter_scales) - set(names))
    if unknown:
        raise ValueError(f"adapter_scales names unknown adapters: {unknown}")
    for name, value in adapter_scales.items():
        if not math.isfinite(value):
            raise ValueError(f"adapter_scales for {name!r} must be finite, got {value}")
    return {name: adapter_scales.get(name, 1.0) for name in names}

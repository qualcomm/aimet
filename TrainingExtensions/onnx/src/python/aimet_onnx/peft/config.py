# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
LoRA adapters as plain data, with no ONNX knowledge.

This module reads PEFT checkpoints and decides how several adapters share one branch per layer:

* ``read_adapter`` validates a checkpoint directory and returns its ``AdapterInfo`` (rank, alpha
  and target module paths). Weights are not loaded.
* ``AdapterLayout`` packs the adapters along the rank axis. At each module path, each adapter that
  targets it owns one ``RankSlice`` of the concatenated ``A``/``B``, and the per-rank scale vector
  holds ``adapter_scale * alpha / r`` for that slice. Optionally every branch is padded with zeros
  to the sum of all the adapters' ranks.
"""

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from peft import LoraConfig, PeftType
from safetensors import safe_open

# LoraConfig fields that are read
_USED_FIELDS = frozenset({"r", "lora_alpha"})

# LoraConfig fields with no effect on inference math. ``target_modules`` is ignored because the
# module paths come from the weight keys.
_IGNORED_FIELDS = frozenset(
    {
        "target_modules",
        "peft_type",
        "task_type",
        "base_model_name_or_path",
        "revision",
        "peft_version",
        "inference_mode",
        "lora_dropout",
        "auto_mapping",
    }
)

# Allowlist of init_lora_weights values that leave the base weights unmodified
_SUPPORTED_INIT_LORA_WEIGHTS = (True, False, "gaussian")

WEIGHTS_FILE = "adapter_model.safetensors"
_KEY_PATTERN = re.compile(
    r"^base_model\.model\.(?P<path>.+)\.lora_(?P<which>[AB])\.weight$"
)


def checkpoint_key(module_path: str, which: str) -> str:
    """
    Return the safetensors key of one LoRA weight in a PEFT checkpoint.

    :param module_path: Dotted torch module path of the layer, e.g. ``model.layers.0.mlp.up_proj``
    :param which: ``"A"`` or ``"B"``
    :return: The key, e.g. ``base_model.model.model.layers.0.mlp.up_proj.lora_A.weight``
    """
    return f"base_model.model.{module_path}.lora_{which}.weight"


@dataclass(frozen=True)
class AdapterInfo:
    """
    What a PEFT checkpoint says about itself: its rank, alpha and the layers it targets.

    The adapter's output is scaled by ``alpha / r``. The module paths are read from the weight
    keys, not from ``LoraConfig.target_modules``.
    """

    name: str  # the name paired with the checkpoint in ``adapters`` passed to load_lora_adapters
    r: int  # LoraConfig.r
    alpha: float  # LoraConfig.lora_alpha
    module_paths: tuple[str, ...]  # dotted torch paths of the targeted layers, sorted


def read_adapter(name: str, ckpt_dir: str | Path) -> AdapterInfo:
    """
    Read and validate one PEFT LoRA checkpoint directory. Weights are not loaded.

    Config errors are reported before weight-key errors.

    :param name: The adapter's name, used in error messages and stored in the result
    :param ckpt_dir: Directory with ``adapter_config.json`` and ``adapter_model.safetensors``, as
        written by ``PeftModel.save_pretrained``
    :return: The adapter's rank, alpha and target module paths
    :raises ValueError: If the config uses a setting that is not supported, the weights file is
        missing, or a weight key or shape is not a valid LoRA ``A``/``B`` pair of rank ``r``
    """
    config = LoraConfig.from_pretrained(ckpt_dir)
    _check_config(name, config)  # config errors first
    paths = _read_module_paths(name, Path(ckpt_dir), config.r)  # then weight-key errors
    return AdapterInfo(name, config.r, config.lora_alpha, tuple(sorted(paths)))


def _check_config(name: str, config: LoraConfig) -> None:
    """
    Check that the config only uses settings whose inference math is reproduced here.

    Fields in ``_USED_FIELDS`` and ``_IGNORED_FIELDS`` may have any value, ``init_lora_weights``
    must be in ``_SUPPORTED_INIT_LORA_WEIGHTS``, and every other field must keep its
    ``LoraConfig`` default.

    :param name: The adapter's name, used in error messages
    :param config: The adapter's ``LoraConfig``
    :raises ValueError: If ``peft_type`` is not LORA, ``init_lora_weights`` modifies the base
        weights, another field differs from its default, or ``r`` is not positive
    """
    if config.peft_type != PeftType.LORA:
        raise ValueError(
            f"Adapter {name!r}: peft_type={config.peft_type!r} is not supported (only LORA)"
        )

    init = config.init_lora_weights
    if not any(
        type(init) is type(v) and init == v for v in _SUPPORTED_INIT_LORA_WEIGHTS
    ):
        raise ValueError(
            f"Adapter {name!r}: LoraConfig.init_lora_weights={init!r} is not supported "
            "(the adapter would be relative to a modified base model)"
        )

    defaults = LoraConfig()
    for field in dataclasses.fields(config):
        if field.name in _USED_FIELDS | _IGNORED_FIELDS | {"init_lora_weights"}:
            continue
        value = getattr(config, field.name)
        if value != getattr(defaults, field.name):
            raise ValueError(
                f"Adapter {name!r}: LoraConfig.{field.name}={value!r} is not supported"
            )

    if config.r <= 0:
        raise ValueError(f"Adapter {name!r}: r must be positive, got {config.r}")


def _read_module_paths(name: str, ckpt_dir: Path, r: int) -> list[str]:
    """
    Return the module paths in the checkpoint's weight keys, after checking each A/B pair.

    Only the tensor shapes are read. Every key must be
    ``base_model.model.<path>.lora_{A,B}.weight``, every path must have both ``A`` and ``B``, and
    ``A`` must have rank ``r`` in dim 0 and ``B`` in dim 1. Whether the other dims fit the base
    layer is checked when the weights are loaded.

    :param name: The adapter's name, used in error messages
    :param ckpt_dir: The checkpoint directory
    :param r: Rank from the adapter's config
    :return: Module paths, in the order of the weight keys
    :raises ValueError: If the weights file is missing or empty, a key is not a LoRA ``A``/``B``
        weight, a path lacks ``A`` or ``B``, or a rank dim differs from ``r``
    """
    weights_path = ckpt_dir / WEIGHTS_FILE
    if not weights_path.exists():
        raise ValueError(
            f"Adapter {name!r}: {WEIGHTS_FILE} not found in {ckpt_dir} "
            "(adapter_model.bin is not supported; re-save with safe_serialization=True)"
        )

    shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    with safe_open(str(weights_path), framework="np") as f:
        for key in f.keys():
            match = _KEY_PATTERN.match(key)
            if not match:
                raise ValueError(
                    f"Adapter {name!r}: unsupported weight key {key!r} "
                    f"(expected {checkpoint_key('<path>', '{A,B}')})"
                )
            shapes.setdefault(match["path"], {})[match["which"]] = tuple(
                f.get_slice(key).get_shape()
            )

    if not shapes:
        raise ValueError(f"Adapter {name!r}: no LoRA weights found in {weights_path}")
    for path, pair in shapes.items():
        if set(pair) != {"A", "B"}:
            missing = "lora_B" if "A" in pair else "lora_A"
            raise ValueError(f"Adapter {name!r}: {path!r} is missing {missing}.weight")
        if pair["A"][0] != r:
            raise ValueError(
                f"Adapter {name!r}: {path}.lora_A.weight has shape {pair['A']}, expected rank {r} "
                "in dim 0"
            )
        if pair["B"][1] != r:
            raise ValueError(
                f"Adapter {name!r}: {path}.lora_B.weight has shape {pair['B']}, expected rank {r} "
                "in dim 1"
            )
    return list(shapes)


@dataclass(frozen=True)
class RankSlice:
    """The ranks ``[offset, offset + r)`` one adapter owns in a layer's concatenated branch."""

    adapter: str  # the adapter's name
    offset: int  # first rank of the slice in the concatenated A / B
    r: int  # the adapter's rank
    alpha_over_r: float  # before any adapter_scales multiplier


@dataclass(frozen=True)
class AdapterLayout:
    """
    Where each adapter's ranks live in the concatenated branch of each layer.

    A layer's branch has total rank ``R``, the sum of the ranks of the adapters that target it.
    Adapters are packed along the rank axis in the order they were given, so the same order
    always gives the same layout. An adapter that doesn't target a layer has no slice there.

    With ``padded_rank`` set, every layer's branch has that rank instead; the ranks after the last
    slice are padding (zero ``A``, ``B`` and scale).
    """

    adapters: tuple[str, ...]  # in the order given to build
    # module path -> slices of the adapters that target the layer, in adapter order
    slices_by_path: dict[str, tuple[RankSlice, ...]]
    # rank of every branch when padded; None for no padding
    padded_rank: int | None = None

    @classmethod
    def build(
        cls, adapters: Sequence[AdapterInfo], pad_rank: bool = False
    ) -> "AdapterLayout":
        """
        Pack adapters into one layout over the union of their module paths.

        At each module path the adapters that target it get consecutive slices in the given order,
        e.g. ``x`` (r=4) and ``y`` (r=2) on the same layer own ranks ``[0, 4)`` and ``[4, 6)``.

        With ``pad_rank``, every branch gets the sum of all the adapters' ranks, so a layer only
        ``y`` targets owns ``[0, 2)`` and has padding ``[2, 6)``. The slices are the same either way.

        :param adapters: Adapters in the order their slices are packed
        :param pad_rank: Pad every branch to the sum of all the adapters' ranks
        :return: The layout, keyed by the adapters' module paths
        """
        slices_by_path: dict[str, list[RankSlice]] = {}
        next_offset: dict[str, int] = {}
        for adapter in adapters:
            for path in adapter.module_paths:
                slices = slices_by_path.setdefault(path, [])
                offset = next_offset.get(path, 0)
                next_offset[path] = offset + adapter.r
                slices.append(
                    RankSlice(
                        adapter.name, offset, adapter.r, adapter.alpha / adapter.r
                    )
                )
        return cls(
            adapters=tuple(a.name for a in adapters),
            slices_by_path={path: tuple(s) for path, s in slices_by_path.items()},
            padded_rank=sum(a.r for a in adapters) if pad_rank else None,
        )

    def rekeyed(self, new_paths: Mapping[str, str]) -> "AdapterLayout":
        """
        Return the same layout under new module paths, e.g. the graph's paths after matching.

        :param new_paths: Current path -> new path. Paths that are not listed are dropped.
        :return: A new layout with the same adapters, slices and padding
        :raises KeyError: If ``new_paths`` names a path that is not in the layout
        """
        return AdapterLayout(
            self.adapters,
            {new: self.slices_by_path[old] for old, new in new_paths.items()},
            self.padded_rank,
        )

    def rank(self, module_path: str) -> int:
        """
        Return the total rank ``R`` of the branch at ``module_path``.

        :param module_path: A module path in the layout
        :return: ``padded_rank`` if set, else the sum of the ranks of the adapters that target the
            layer
        """
        # KeyError for an unknown path, padded or not
        slices = self.slices_by_path[module_path]
        if self.padded_rank is not None:
            return self.padded_rank
        return sum(rank_slice.r for rank_slice in slices)

    def scale_vector(
        self, module_path: str, adapter_scales: Mapping[str, float]
    ) -> np.ndarray:
        """
        Return the per-rank scale of the branch at ``module_path``.

        Each adapter's slice holds ``adapter_scales[adapter] * alpha / r``, so the branch computes
        the sum of the scaled adapters. Padding ranks are 0.

        :param module_path: A module path in the layout
        :param adapter_scales: Adapter name -> multiplier, for every adapter in the layout
        :return: Float64 array of shape ``[R]``
        """
        scale = np.zeros(self.rank(module_path), dtype=np.float64)
        for rank_slice in self.slices_by_path[module_path]:
            ranks = slice(rank_slice.offset, rank_slice.offset + rank_slice.r)
            scale[ranks] = adapter_scales[rank_slice.adapter] * rank_slice.alpha_over_r
        return scale

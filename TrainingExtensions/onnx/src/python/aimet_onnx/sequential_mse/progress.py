# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Progress and result reporting for sequential MSE.

Everything here is derived from the dependency graph and the candidate losses sequential MSE
already computes, so it makes no assumption about the model's architecture.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.meta.operations import Op
from aimet_onnx.prepare_passes.fix_node_names_in_dynamo_exported_onnx import (
    _get_name_from_metadata,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)

# Weights listed in the end-of-run summary as gaining the most and the least
_NUM_EXTREMES = 5

# Warn when at least this fraction of a weight's channels/blocks picks the smallest candidate,
# which suggests the candidate grid is too coarse to find their best range
_SMALLEST_CANDIDATE_WARN_FRACTION = 0.25


def display_name(op: Op) -> str:
    """Name for ``op`` that can be looked up in the graph and passed to ``nodes_to_exclude``.

    TorchScript exports name nodes by module path already (``/features.3/conv/Conv``), so
    that name is used as is. Dynamo exports name them ``node_linear_12`` but record the path
    in ``namespace`` metadata, so the path is shown followed by the node's name in the graph:
    ``layers.0.mlp.down_proj [node_linear_12]``.
    """
    name = op.name
    node = op.get_module()
    if node is not None and not name.startswith("/"):
        path = _get_name_from_metadata(node)
        if path:
            return f"{_module_path(path)} [{name}]"
    return name


def _module_path(name: str) -> str:
    """``/a.b/c/MatMul`` -> ``a.b.c``; the op type is implied (every op here has a weight)."""
    parts = name.strip("/").split("/")
    if len(parts) > 1:
        parts = parts[:-1]
    return ".".join(parts)


def split_common_prefix(names: List[str]) -> Tuple[str, List[str]]:
    """Split off the dotted prefix ``names`` share: ``["a.b.q", "a.b.k"]`` -> ``("a.b", ["q", "k"])``.

    The prefix is empty when there is a single name, the names share none, or any name is
    the graph name itself, which is kept whole so it can still be looked up. A trailing
    `` [graph name]`` (see :func:`display_name`) is kept with each name, never split.
    """
    if len(names) < 2:
        return "", list(names)
    paths, suffixes = zip(*(_split_graph_name(name) for name in names))
    if not all(suffixes):
        return "", list(names)
    split = [path.split(".") for path in paths]
    common = 0
    while all(len(parts) > common + 1 for parts in split) and (
        len({parts[common] for parts in split}) == 1
    ):
        common += 1
    return ".".join(split[0][:common]), [
        ".".join(parts[common:]) + suffix for parts, suffix in zip(split, suffixes)
    ]


def _split_graph_name(name: str) -> Tuple[str, str]:
    """``a.b [node_1]`` -> ``("a.b", " [node_1]")``; ``a.b`` -> ``("a.b", "")``."""
    if name.endswith("]") and " [" in name:
        path, _, graph_name = name.rpartition(" [")
        return path, f" [{graph_name}"
    return name, ""


@dataclass
class WeightResult:
    """What the candidate search found for one weight."""

    name: str
    # Chosen range as a fraction of the min/max range, median over channels/blocks
    range_fraction: float
    # Relative change of the summed loss vs. the min/max range (the last candidate)
    loss_change: Optional[float]
    # Fraction of channels/blocks that picked the smallest candidate
    smallest_fraction: float

    def describe(self) -> str:
        """``range 0.80x, loss -31%`` (the loss change is omitted when there is none to report)."""
        text = f"range {self.range_fraction:.2f}x"
        if self.loss_change is not None:
            text += f", loss {self.loss_change * 100:+.0f}%"
        return text


def describe_group(results: List[WeightResult]) -> str:
    """Each weight's name next to its result, with a shared name prefix printed once.

    ``layers.0: q range 0.95x, loss -26%; k range 0.95x, loss -27%``
    """
    prefix, names = split_common_prefix([result.name for result in results])
    weights = "; ".join(
        f"{name} {result.describe()}" for name, result in zip(names, results)
    )
    return f"{prefix}: {weights}" if prefix else weights


def weight_result(
    name: str,
    stacked_loss: torch.Tensor,
    best_indices: torch.Tensor,
    num_candidates: int,
) -> WeightResult:
    """Summarize the search for one weight.

    :param stacked_loss: Loss per candidate, ``(num_candidates, ...)`` with one entry per
        channel/block after the first axis.
    :param best_indices: Index of the chosen candidate for each channel/block.
    """
    # Candidate i scales the min/max range by (i + 1) / num_candidates
    fractions = (best_indices.float() + 1) / num_candidates
    smallest = (best_indices == 0).float().mean().item()

    # The last candidate is the unmodified min/max range
    minmax_loss = stacked_loss[-1].sum().item()
    best_loss = stacked_loss.min(0)[0].sum().item()
    # The sqnr loss is negative (higher SQNR is better), so a relative change is meaningless
    loss_change = None
    if minmax_loss > 0 and best_loss >= 0:
        loss_change = best_loss / minmax_loss - 1
    return WeightResult(name, fractions.median().item(), loss_change, smallest)


def _format_duration(seconds: float) -> str:
    """``H:MM:SS``, or ``M:SS`` under an hour."""
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@dataclass
class _Phases:
    """Seconds spent in each phase, in the order they run."""

    seconds: Dict[str, float] = field(default_factory=dict)

    def add(self, phase: str, seconds: float):
        self.seconds[phase] = self.seconds.get(phase, 0.0) + seconds

    def describe(self, fmt) -> str:
        return ", ".join(f"{phase} {fmt(sec)}" for phase, sec in self.seconds.items())


class SeqMseProgress:
    """Logs one line per group of weights optimized together, and a summary at the end.

    :param groups: Ops to optimize, per group, in the order they are optimized.
    :param num_candidates: Candidates searched per weight.
    :param num_batches: Calibration batches.
    :param loss_fn: Name of the loss function.
    :param excluded: Names of the ops excluded from optimization.
    """

    def __init__(
        self,
        groups: List[List[Op]],
        num_candidates: int,
        num_batches: int,
        loss_fn: str,
        excluded: List[str],
    ):
        self._num_groups = len(groups)
        self._num_weights = sum(len(group) for group in groups)
        self._num_candidates = num_candidates
        self._group_count = 0
        self._weights_done = 0
        self._start = time.perf_counter()
        self._phase_start = None
        self._group = _Phases()
        self._total = _Phases()
        self._results: List[WeightResult] = []
        self._notices = set()

        _logger.info(
            "SeqMSE: optimizing %d weights in %d groups | %d candidates, %d batches, %s loss%s",
            self._num_weights,
            self._num_groups,
            num_candidates,
            num_batches,
            loss_fn,
            f" | {len(excluded)} weights excluded" if excluded else "",
        )
        _logger.info(
            "SeqMSE: each weight is reported as e.g. 'range 0.80x, loss -31%%': its chosen "
            "range is 0.80x of the min/max range, and its loss is 31%% lower than with min/max"
        )
        if excluded:
            _logger.debug("SeqMSE excluded weights: %s", ", ".join(excluded))

    def notice_once(self, key: str, message: str, *args):
        """Log ``message`` the first time ``key`` is seen."""
        if key not in self._notices:
            self._notices.add(key)
            _logger.info(message, *args)

    def start_phase(self, phase: str):
        """Start timing ``phase``, ending the phase in progress (if any)."""
        self._end_phase()
        self._phase_start = (phase, time.perf_counter())

    def _end_phase(self):
        if self._phase_start is not None:
            phase, start = self._phase_start
            seconds = time.perf_counter() - start
            self._group.add(phase, seconds)
            self._total.add(phase, seconds)
            self._phase_start = None

    def end_group(self, results: List[WeightResult]):
        """Log the line for a finished group."""
        self._end_phase()
        self._group_count += 1
        first = self._weights_done + 1
        self._weights_done += len(results)
        self._results.extend(results)

        elapsed = time.perf_counter() - self._start
        remaining = (
            elapsed / self._weights_done * (self._num_weights - self._weights_done)
        )
        weights = (
            f"weight {first}"
            if len(results) == 1
            else f"weights {first}-{self._weights_done}"
        )
        group_seconds = sum(self._group.seconds.values())
        _logger.info(
            "SeqMSE group %d of %d (%s of %d): %s | %s | ETA %s",
            self._group_count,
            self._num_groups,
            weights,
            self._num_weights,
            describe_group(results),
            _format_seconds(group_seconds),
            _format_duration(remaining),
        )
        _logger.debug(
            "SeqMSE group %d phases: %s",
            self._group_count,
            self._group.describe(_format_seconds),
        )
        for result in results:
            if result.smallest_fraction >= _SMALLEST_CANDIDATE_WARN_FRACTION:
                _logger.warning(
                    "SeqMSE: %.0f%% of %s's channels chose the smallest candidate range "
                    "(1/%d of min/max); more candidates may find a better range.",
                    result.smallest_fraction * 100,
                    result.name,
                    self._num_candidates,
                )
        self._group = _Phases()

    def finish(self):
        """Log the end-of-run summary."""
        self._end_phase()
        elapsed = time.perf_counter() - self._start
        _logger.info(
            "SeqMSE: done, optimized %d weights in %s",
            self._weights_done,
            _format_duration(elapsed),
        )
        _logger.debug("SeqMSE phases: %s", self._total.describe(_format_duration))
        if not self._results:
            return
        fractions = np.array([result.range_fraction for result in self._results])
        _logger.info(
            "SeqMSE: chosen range vs. min/max: median %.2fx, min %.2fx, max %.2fx",
            np.median(fractions),
            fractions.min(),
            fractions.max(),
        )
        with_change = [r for r in self._results if r.loss_change is not None]
        if len(with_change) > _NUM_EXTREMES:
            ranked = sorted(with_change, key=lambda result: result.loss_change)
            for label, picked in (
                ("largest", ranked[:_NUM_EXTREMES]),
                ("smallest", ranked[::-1][:_NUM_EXTREMES]),
            ):
                _logger.info(
                    "SeqMSE: %s loss reductions vs. min/max: %s",
                    label,
                    ", ".join(f"{r.name} {r.loss_change * 100:+.0f}%" for r in picked),
                )


def _format_seconds(seconds: float) -> str:
    """``12.3s`` under a minute, else ``M:SS``."""
    return f"{seconds:.1f}s" if seconds < 60 else _format_duration(seconds)

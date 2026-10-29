# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-quantizer quantization sensitivity analysis for ONNX models.

:func:`analyze_per_quantizer_sensitivity` measures how sensitive a calibrated
:class:`QuantizationSimModel` is to quantization at a finer granularity than the
op-level :func:`aimet_onnx.analyze_per_layer_sensitivity`, by enabling one
quantizer (or group of quantizers) at a time. An optional ``group_fn`` selects
and groups the quantizers to sweep -- e.g. pass a ``group_fn`` that returns
``None`` for everything but the KV-cache tensors to run a KV-cache-only sweep.

It accepts a :class:`SensitivityMetric` (a named ``eval_fn`` plus ranking
semantics) and returns a ``{name: score}`` dict ranked most-sensitive-first.
Feed that dict into :func:`aimet_onnx.lite_mp.flip_layers_to_higher_precision`
(keyed by op name) to raise the most sensitive units to a higher precision.

Quantizer names are tensor names, which for exported models are often opaque
(e.g. a torch ``Linear`` weight lands in ONNX as an initializer named
``onnx::MatMul_9772``). :func:`get_quantizer_op_names` maps each quantizer to the
name of the ONNX node that owns it -- typically well structured, e.g.
``/model/layers.0/self_attn/q_proj/MatMul`` -- and :func:`group_by_op_name`
wraps that mapping as a ``group_fn`` so the sweep reports scores keyed by node
name, ready to hand straight to
:func:`aimet_onnx.lite_mp.flip_layers_to_higher_precision`.

The metric ``eval_fn`` takes an ``onnxruntime.InferenceSession`` and returns a
float, matching the existing :func:`aimet_onnx.analyze_per_layer_sensitivity`
and :func:`aimet_onnx.utils.make_psnr_eval_fn` convention -- there is no
dependency on any particular model harness.
"""

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import onnxruntime as ort
from aimet_onnx.common.progress import progress_bar  # pylint: disable=import-error

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import _run_session_to_numpy, disable_quantizers
from aimet_onnx.common.utils import AimetLogger, compute_psnr

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class SensitivityMetric:
    """A named evaluation function with ranking semantics.

    :param name: Human-readable metric name (used in logs and plots).
    :param eval_fn: Callable taking an ``onnxruntime.InferenceSession`` and
        returning a scalar score.
    :param higher_is_worse: If ``True`` (e.g. perplexity, KL divergence), a
        larger score means greater sensitivity. If ``False`` (e.g. PSNR,
        accuracy), a smaller score means greater sensitivity. Controls the
        ordering used when ranking units most-sensitive-first.
    """

    def __init__(
        self,
        name: str,
        eval_fn: Callable[[ort.InferenceSession], float],
        higher_is_worse: bool = True,
    ):
        if not callable(eval_fn):
            raise ValueError(f"eval_fn is expected to be callable; got {type(eval_fn)}")
        self.name = name
        self.eval_fn = eval_fn
        self.higher_is_worse = higher_is_worse

    def __call__(self, session: ort.InferenceSession) -> float:
        return self.eval_fn(session)

    def sensitivity_score(self, score: float) -> float:
        """Map a raw score to a value where larger always means more sensitive."""
        return score if self.higher_is_worse else -score


def make_topk_logit_psnr_metric(
    fp_session: ort.InferenceSession,
    inputs: Iterable[Dict[str, np.ndarray]],
    k: int = 10,
    logit_output_index: int = 0,
) -> SensitivityMetric:
    """Build a top-k logit PSNR metric for LLM-style models.

    For each input sample, the reference (floating-point) logits are compared
    against the sim's logits, restricted to the top-``k`` vocabulary indices of
    the FP logits at each position. PSNR is computed over those slices,
    aggregated across all samples. Higher PSNR is better, so
    ``higher_is_worse=False`` -- a drop in PSNR indicates greater sensitivity.

    The FP logits are captured once, up front, from ``fp_session``.

    :param fp_session: ORT inference session for the floating-point model.
    :param inputs: Iterable of input feed dicts (``{input_name: np.ndarray}``). bfloat16
        inputs are ``ml_dtypes.bfloat16`` arrays.
    :param k: Number of top FP vocabulary indices to compare per position.
    :param logit_output_index: Index of the logits tensor in the model outputs.
    :return: A :class:`SensitivityMetric` wrapping the top-k logit PSNR eval fn.
    """
    inputs = list(inputs)

    def _topk_slices(logits: np.ndarray, topk_idx: np.ndarray) -> np.ndarray:
        # logits: [..., vocab]; topk_idx: [..., k] along the last axis.
        return np.take_along_axis(logits, topk_idx, axis=-1)

    # The reference logits, their top-k indices, and the gathered FP slice depend
    # only on the (fixed) FP model -- not on the sim under test. Precompute them
    # once here so the per-eval hot path (called once per quantizer in a sweep)
    # only runs the quantized session, gathers with the cached indices, and
    # computes PSNR. This hoists the expensive top-k off the vocab axis
    # (~150k wide) out of the inner loop entirely.
    #
    # argpartition (O(vocab)) rather than argsort (O(vocab log vocab)): we only
    # need *which* k indices are largest, not their order, and PSNR over the
    # gathered slice is order-independent.
    topk_indices: List[np.ndarray] = []
    fp_slices: List[np.ndarray] = []
    for feed in inputs:
        fp_arr = np.asarray(
            _run_session_to_numpy(fp_session, feed)[logit_output_index],
            dtype=np.float32,
        )
        effective_k = min(k, fp_arr.shape[-1])
        idx = np.argpartition(fp_arr, -effective_k, axis=-1)[..., -effective_k:]
        topk_indices.append(idx)
        fp_slices.append(_topk_slices(fp_arr, idx).reshape(-1))
    expected = np.concatenate(fp_slices)

    def _eval(session: ort.InferenceSession) -> float:
        qt_slices: List[np.ndarray] = []
        for feed, idx in zip(inputs, topk_indices):
            qt_arr = np.asarray(
                _run_session_to_numpy(session, feed)[logit_output_index],
                dtype=np.float32,
            )
            qt_slices.append(_topk_slices(qt_arr, idx).reshape(-1))
        actual = np.concatenate(qt_slices)
        return compute_psnr(expected, actual)

    return SensitivityMetric(
        name=f"Top{k}LogitPSNR", eval_fn=_eval, higher_is_worse=False
    )


def get_quantizer_op_names(sim: QuantizationSimModel) -> Dict[str, str]:
    """Map each quantizer name to the name of the ONNX node that owns it.

    Quantizers are keyed by tensor name, which for exported models is frequently
    unreadable (``onnx::MatMul_9772``), while the node the tensor belongs to
    usually carries the module path it came from
    (``/model/layers.0/self_attn/q_proj/MatMul``). This resolves the latter for
    every quantizer in ``sim``, mirroring the ownership rules of
    :meth:`QuantizationSimModel.get_op_quantizers`:

        * a parameter tensor belongs to the node it parameterizes;
        * an activation tensor belongs to the node that produces it;
        * a tensor with no producer (e.g. a graph input) belongs to the node
          that consumes it.

    The returned names are connected-graph op names -- the same keys accepted by
    :func:`aimet_onnx.lite_mp.flip_layers_to_higher_precision`.

    :param sim: QuantizationSimModel to inspect.
    :return: ``{quantizer_name: op_name}``. Quantizers that belong to no op
        (rare) are absent from the mapping.
    """
    ops = sim.connected_graph.get_all_ops()
    op_names: Dict[str, str] = {}

    # Ownership passes run in precedence order, and each uses setdefault, so an
    # earlier pass wins. Params first (a weight is owned by the node it feeds,
    # never by a consumer of that node), then producers, then consumers.
    for op_name, op in ops.items():
        for param_name in op.parameters:
            if param_name in sim.qc_quantize_op_dict:
                op_names.setdefault(param_name, op_name)

    for op_name, op in ops.items():
        for product in op.outputs:
            if product.name in sim.qc_quantize_op_dict:
                op_names.setdefault(product.name, op_name)

    for op_name, op in ops.items():
        for product in op.inputs:
            if product.name in sim.qc_quantize_op_dict:
                op_names.setdefault(product.name, op_name)

    return op_names


def group_by_op_name(
    sim: QuantizationSimModel,
) -> Callable[[str], Optional[str]]:
    """Build a ``group_fn`` that keys sensitivity scores by owning ONNX node name.

    Pass the result as the ``group_fn`` of
    :func:`analyze_per_quantizer_sensitivity` to get readable, node-named results
    instead of raw tensor names::

        scores = analyze_per_quantizer_sensitivity(
            sim, metric, group_fn=group_by_op_name(sim)
        )
        flip_layers_to_higher_precision(sim, scores, percent_to_flip=10)

    Quantizers that share an owning node are swept together as one group, so the
    result is per-node rather than per-quantizer wherever a node has more than
    one enabled quantizer (e.g. both a weight and an output quantizer).
    Quantizers with no owning node are skipped.

    :param sim: QuantizationSimModel to be analyzed.
    :return: Callable mapping a quantizer name to its owning op name, or
        ``None`` if it has none.
    """
    op_names = get_quantizer_op_names(sim)
    return op_names.get


def analyze_per_quantizer_sensitivity(
    sim: QuantizationSimModel,
    metric: SensitivityMetric,
    group_fn: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, float]:
    """Analyze sensitivity by enabling one quantizer (or quantizer group) at a time.

    All quantizers are first disabled. Then, for each group, that group's
    quantizers are enabled, the metric is evaluated on ``sim.session``, and the
    group is disabled again. This isolates each unit's contribution to
    quantization error at a finer granularity than the op-level
    :func:`aimet_onnx.analyze_per_layer_sensitivity`.

    :param sim: Calibrated QuantizationSimModel to analyze. Its quantizers'
        enabled state is restored on return.
    :param metric: :class:`SensitivityMetric` used to score each group.
    :param group_fn: Optional callable mapping a quantizer name to a group key.
        Quantizers sharing a key are enabled together; a key of ``None`` skips
        that quantizer. If omitted, each currently-enabled quantizer is its own
        group (keyed by its own name). To restrict the sweep to a subset (e.g.
        KV-cache tensors), return ``None`` for the quantizers to exclude::

            analyze_per_quantizer_sensitivity(
                sim, metric,
                group_fn=lambda name: name if name in kv_names else None,
            )

        Pass :func:`group_by_op_name` to key the results by owning ONNX node
        name instead of by tensor name.
    :return: Dict mapping group key to its metric score, ordered
        most-sensitive-first per ``metric``.
    """
    if group_fn is None:
        group_fn = lambda name: name  # noqa: E731 - one quantizer per group

    groups: Dict[str, List[str]] = {}
    for name, quantizer in sim.qc_quantize_op_dict.items():
        if not quantizer.enabled:
            continue
        key = group_fn(name)
        if key is None:
            continue
        groups.setdefault(key, []).append(name)

    if not groups:
        raise RuntimeError(
            "No enabled quantizers found to analyze. Ensure the sim is "
            "calibrated and has enabled quantizers."
        )

    logger.info("Analyzing per-quantizer sensitivity over %d groups", len(groups))

    scores: Dict[str, float] = {}
    # Disable every quantizer for the duration of the sweep (restored on exit),
    # then enable one group at a time to isolate its contribution.
    with disable_quantizers(sim, sim.qc_quantize_op_dict.keys()):
        for key, names in progress_bar(
            groups.items(),
            total=len(groups),
            desc=f"Per-quantizer sensitivity ({metric.name})",
        ):
            for name in names:
                sim.qc_quantize_op_dict[name].enabled = True
            try:
                scores[key] = metric(sim.session)
            finally:
                for name in names:
                    sim.qc_quantize_op_dict[name].enabled = False

    return _sorted_by_sensitivity(scores, metric)


def _sorted_by_sensitivity(
    scores: Dict[str, float], metric: SensitivityMetric
) -> Dict[str, float]:
    """Return ``scores`` reordered most-sensitive-first per ``metric``."""
    return dict(
        sorted(
            scores.items(),
            key=lambda kv: kv[1],
            reverse=metric.higher_is_worse,
        )
    )

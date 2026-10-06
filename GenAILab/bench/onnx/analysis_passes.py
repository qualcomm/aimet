# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX-only analysis passes."""

from __future__ import annotations

import contextlib
import functools
import os
import re

from GenAILab.bench.analysis import (
    AnalysisOutput,
    AnalysisPass,
    metric_detail_sections,
)
from GenAILab.bench.yaml_config_parser import YAMLConfigParser


@contextlib.contextmanager
def _truncated(sims, truncation_bits: int):
    """Insert Truncate nodes into every sim's session for the duration of the block.

    While active, each sim's ``_rebuild_session`` is patched to re-insert the
    nodes, since ``on_device`` (Grace grades on CPU) rebuilds from the clean
    graph and would otherwise silently drop the truncation for every metric
    after it. On exit, even on error, the original method is restored and the
    session rebuilt clean.
    """
    from aimet_onnx.experimental._truncation_aware import (
        create_truncation_aware_session,
    )

    last_session = {}

    def _reinstrument(sim):
        # Drop the old session first so two copies of the model never coexist.
        sim.session = None
        sim.session = create_truncation_aware_session(
            sim, truncation_bits=truncation_bits
        )
        last_session[id(sim)] = sim.session

    originals = []
    try:
        for sim in sims:
            # Recorded before instrumenting, so a failure below still gets
            # this sim rebuilt clean in the finally.
            originals.append((sim, sim._rebuild_session))
            _reinstrument(sim)
            sim._rebuild_session = functools.partial(_reinstrument, sim)
        yield
        for sim in sims:
            assert sim.session is last_session[id(sim)], (
                f"{sim} was rebuilt outside the patched _rebuild_session during "
                "TruncationSimulation -- instrumentation may have been silently lost."
            )
    finally:
        for sim, original in originals:
            sim._rebuild_session = original
            sim._rebuild_session()


@YAMLConfigParser.register_analysis
class TruncationSimulation(AnalysisPass):
    """Evaluate the pass's metrics with accumulator truncation simulated.

    For each bit width, Truncate nodes are inserted into every sim's session
    (``create_truncation_aware_session``), the metrics are evaluated, and the
    sessions are restored before the next bit width.

    ``truncation_bits`` is required even though ``create_truncation_aware_session``
    has a bit-width default: an analysis pass is a deliberate, one-off
    experiment, so the config must state which bit width is under test. It
    accepts a scalar (``truncation_bits: 8``) or a list (``[8, 12]``); each
    value is evaluated separately, in order.
    """

    def __init__(self, *, truncation_bits: int | list[int]):
        bits = (
            [truncation_bits] if isinstance(truncation_bits, int) else truncation_bits
        )
        if (
            not isinstance(bits, list)
            or not bits
            or not all(isinstance(b, int) and not isinstance(b, bool) for b in bits)
        ):
            raise ValueError(
                f"truncation_bits must be an int or a non-empty list of ints, "
                f"got {truncation_bits!r}"
            )
        self.truncation_bits = bits

    def run(self, sim_collection, *, evaluate) -> AnalysisOutput:
        sims = [
            sim_collection.backbone,
            *map(sim_collection.component, sim_collection.present_components()),
        ]
        table, sections = [], []
        for bits in self.truncation_bits:
            label = f"truncation_bits={bits}"
            with _truncated(sims, bits):
                results = evaluate(f"trunc_bits{bits}")
            table.append(
                {
                    "Condition": label,
                    **{r.metric_name: r.result for r in results},
                }
            )
            sections += metric_detail_sections(label, results)
        return AnalysisOutput(table=table, sections=sections)


# KV cache graph inputs: past_key_<layer>_in / past_value_<layer>_in.
KV_INPUT_NAME_RE = re.compile(r"^past_(key|value)_(\d+)_in$")


def kv_quantizer_names(sim) -> list[str]:
    """Return the KV cache input quantizer names present in the sim."""
    return [name for name in sim.qc_quantize_op_dict if KV_INPUT_NAME_RE.match(name)]


def quantizers_by_group(sim, group_fn, scores: dict) -> dict[str, str]:
    """Map each swept group key to the quantizer tensor names behind it.

    The weights sweep reports scores keyed by ONNX node name, which hides which
    initializer was actually quantized. Passing this to ``save_sensitivity_plot``
    /``save_sensitivity_results`` as ``details`` keeps the tensor name visible in
    the tooltip, table and JSON. Only currently-enabled quantizers are listed:
    the sweep skips disabled ones, so a disabled bias would otherwise be
    attributed to a group it never contributed to.
    """
    details = {}
    for name, quantizer in sim.qc_quantize_op_dict.items():
        if not quantizer.enabled:
            continue
        key = group_fn(name)
        if key in scores:
            details.setdefault(key, []).append(name)
    return {key: ", ".join(names) for key, names in details.items()}


def _positive_int(name: str, value) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


@YAMLConfigParser.register_analysis
class QuantizerSensitivity(AnalysisPass):
    """Rank backbone quantizers by top-k logit PSNR against the FP model.

    Runs on the sim the recipe built, so no second export or calibration.
    It is meant for a plain quantsim: keep the backbone recipe to
    ``Calibration`` (or ``model.encodings`` saved from one). After SpinQuant,
    AdaScale or SeqMSE the ranking describes the transformed model instead.
    This is not checked.

    The FP reference is the same sim with every quantizer disabled. The sweep
    (``analyze_per_quantizer_sensitivity``) then disables every quantizer and
    enables one group at a time, so each score is that group's error alone:

    * ``mode: weights`` groups the weight quantizers by owning ONNX node.
      Activation quantizers stay off for the whole sweep, so the score is pure
      weight error at the configured weight precision.
    * ``mode: kv_cache`` sweeps the ``past_key_<i>_in``/``past_value_<i>_in``
      quantizers, one each.

    Bit widths come from ``precision``, not from the mode. Inputs are
    ``num_samples`` Wikitext train samples, prefilled in FP mode. ``top_k`` is
    how many of the FP model's largest logits per position are compared.
    ``report_top_n`` only limits the report table; the JSON and plot written to
    the run directory hold every group. Backbone only: visual/audio encoders
    are not swept. The sim's quantizer enabled state is restored on return.
    """

    uses_metrics = False

    def __init__(
        self,
        *,
        mode: str = "weights",
        num_samples: int = 4,
        top_k: int = 10,
        report_top_n: int = 20,
    ):
        if mode not in ("weights", "kv_cache"):
            raise ValueError(f"mode must be 'weights' or 'kv_cache', got {mode!r}")
        self.mode = mode
        self.num_samples = _positive_int("num_samples", num_samples)
        self.top_k = _positive_int("top_k", top_k)
        self.report_top_n = _positive_int("report_top_n", report_top_n)

    def _build_feeds(self, sim, generator, tokenizer, context_length) -> list[dict]:
        """Prefill ``num_samples`` Wikitext train samples on the FP backbone."""
        from GenAILab.bench.datasets import Wikitext
        from GenAILab.bench.onnx.quant_recipes import _prefill_inputs

        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        dataset = Wikitext.load_encoded_dataset(tokenizer, context_length, "train")
        return _prefill_inputs(sim, generator, dataset, self.num_samples)

    def _group_fn(self, sim):
        from aimet_onnx.analysis import group_by_op_name

        if self.mode == "weights":
            # Keyed by owning node name, so the result reads as layer names.
            # Activation quantizers are disabled around the sweep in run(), so
            # each node's group is its weights alone.
            return group_by_op_name(sim)

        kv_names = set(kv_quantizer_names(sim))
        if not kv_names:
            raise RuntimeError(
                "kv_cache mode: no past_key_<i>_in / past_value_<i>_in "
                "quantizers found in the backbone sim"
            )
        return lambda name: name if name in kv_names else None

    def run(
        self, sim_collection, *, output_dir, generator, tokenizer, context_length
    ) -> AnalysisOutput:
        from aimet_onnx.analysis import (
            analyze_per_quantizer_sensitivity,
            make_topk_logit_psnr_metric,
            save_sensitivity_plot,
            save_sensitivity_results,
        )
        from aimet_onnx.utils import disable_quantizers

        sim = sim_collection.backbone
        group_fn = self._group_fn(sim)
        feeds = self._build_feeds(sim, generator, tokenizer, context_length)
        # The metric runs the FP logits once, here, and keeps them.
        with disable_quantizers(sim, sim.qc_quantize_op_dict.keys()):
            metric = make_topk_logit_psnr_metric(sim.session, feeds, k=self.top_k)

        # Weights mode: activations off for the sweep, so no activation error
        # mixes into a node's score. Same as the script's
        # _remove_activation_quantizers, but restored on exit.
        activations = (
            set(sim.activation_names) & sim.qc_quantize_op_dict.keys()
            if self.mode == "weights"
            else set()
        )
        with disable_quantizers(sim, activations):
            # Ranked most-sensitive-first.
            scores = analyze_per_quantizer_sensitivity(sim, metric, group_fn=group_fn)
            # Built inside the block: it lists only enabled quantizers, so
            # outside it would add each node's activations to its weight names.
            details = (
                quantizers_by_group(sim, group_fn, scores)
                if self.mode == "weights"
                else None
            )

        # The plot reads better in graph order: re-key it by quantizer order in
        # qc_quantize_op_dict, which is roughly topological.
        graph_order = {}
        for name in sim.qc_quantize_op_dict:
            key = group_fn(name)
            if key in scores and key not in graph_order:
                graph_order[key] = scores[key]

        plot_name = f"{self.mode}_sensitivity.html"
        results_name = f"{self.mode}_sensitivity.json"
        save_sensitivity_plot(
            graph_order,
            metric,
            save_path=os.path.join(output_dir, plot_name),
            details=details,
            details_label="Quantizer",
        )
        save_sensitivity_results(
            scores,
            save_path=os.path.join(output_dir, results_name),
            details=details,
        )

        top = list(scores.items())[: self.report_top_n]
        table = [
            # Pre-formatted: the report's float format (one decimal) is too
            # coarse to tell neighbouring PSNR scores apart.
            {"Rank": rank, "Name": name, metric.name: f"{score:.2f}"}
            for rank, (name, score) in enumerate(top, start=1)
        ]
        summary = (
            f"{len(scores)} groups swept ({self.mode}); the {len(top)} most "
            f"sensitive are shown above. Lower {metric.name} means more "
            f"sensitive. The full ranking is in {results_name}."
        )
        return AnalysisOutput(
            table=table,
            sections=[("Summary", summary)],
            artifacts=[plot_name, results_name],
        )

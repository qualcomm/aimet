# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM quantization sensitivity analysis driver.

An end-to-end example of the ``aimet_onnx.analysis`` API on a real LLM: export a
Hugging Face model to ONNX, calibrate a quantsim on Wikitext, rank each unit by
its contribution to quantization error, and write an interactive report. It uses

    * ``aimet_onnx.analysis.SensitivityMetric`` / ``make_topk_logit_psnr_metric``
    * ``aimet_onnx.analysis.analyze_per_quantizer_sensitivity``  (weights mode;
      kv_cache mode restricts the same sweep via a ``group_fn``)
    * ``aimet_onnx.analysis.group_by_op_name``                   (report weight
      sensitivity by ONNX node name instead of initializer name)
    * ``aimet_onnx.lite_mp.flip_layers_to_higher_precision``     (mixed precision)
    * ``aimet_onnx.analysis.save_sensitivity_plot`` / ``save_sensitivity_results``

The released API is framework-agnostic: metrics take an ``onnxruntime``
``InferenceSession`` and eval on numpy feed dicts. GenAILab is used only to
instantiate/export the model and build those feed dicts -- there is no GenAILab
dependency inside the analysis itself.

The model is selected on the command line; sequence/context length default to
2048/4096::

    python -m GenAILab.bench.onnx.llm_layer_sensitivity \
        --model-id microsoft/Phi-4-mini-instruct
    python -m GenAILab.bench.onnx.llm_layer_sensitivity \
        --model-id Qwen/Qwen3-0.6B --sequence-length 1024 --context-length 2048 \
        --mode kv_cache

The ONNX checkpoint directory and the output file names are derived from the
model id, so runs of different models do not overwrite each other.
"""

import argparse
import copy
import re

import torch
from tqdm import tqdm

from aimet_onnx import quantsim, int4, int16
from aimet_onnx.quantsim import QuantizationSimModel, compute_encodings
from aimet_onnx.utils import OrtInferenceSession
from aimet_onnx.lite_mp import flip_layers_to_higher_precision
from aimet_onnx.analysis import (
    make_topk_logit_psnr_metric,
    analyze_per_quantizer_sensitivity,
    group_by_op_name,
    save_sensitivity_plot,
    save_sensitivity_results,
)

from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.qai_hub_lm.models.base import LLM
from GenAILab.qai_hub_lm.models.generator import Generator
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.qai_hub_lm.backends.onnx.export_utils import get_onnx_model
from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import TorchONNXInterface
from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import (
    get_ort_providers,
    AttributePatch,
    _remove_activation_quantizers,
)
from GenAILab.bench.onnx.quant_recipes import _prefill_inputs
from GenAILab.bench.onnx.analysis_passes import kv_quantizer_names, quantizers_by_group
from GenAILab.bench.datasets import Wikitext

DEFAULT_SEQUENCE_LENGTH = 2048
DEFAULT_CONTEXT_LENGTH = 4096
NUM_CALIBRATION_ITERATIONS = 20
NUM_EVAL_ITERATIONS = 4
ATTENTION_MASK_MIN = -100
ONNX_CHECKPOINT_ROOT = "./onnx_checkpoints"


class _SessionHolder:
    """Trivial ``.session`` holder so ``TorchONNXInterface`` can wrap a bare session.

    ``TorchONNXInterface`` only ever reads ``self.quantsim.session``, so handing
    it this (instead of a real QuantizationSimModel) lets us reuse the interface
    -- and thus the generator -- against the floating-point session. See
    :func:`_fp_prefill_model`.
    """

    def __init__(self, session):
        self.session = session


def _fp_prefill_model(fp_session, config):
    """Wrap the FP session as a ``TorchONNXInterface`` for prefill.

    Prefilling on the floating-point graph (before quantsim exists) is much
    faster than running the uncalibrated, QDQ-laden quantsim session, and the
    cached K/V are the true FP values rather than garbage-quantized ones.
    Reusing ``TorchONNXInterface`` gives us the ``forward`` / ``device`` /
    ``dtype`` / ``config`` that ``generator.prefill`` and ``prepare_inputs``
    expect, without reimplementing them.
    """
    return TorchONNXInterface(_SessionHolder(fp_session), config)


def _model_slug(model_id: str) -> str:
    """Turn a HF model id into a filesystem-friendly stem.

    Drops the org prefix, lower-cases, and collapses anything but ``a-z0-9._``
    into underscores: ``microsoft/Phi-4-mini-instruct`` ->
    ``phi_4_mini_instruct``, ``Qwen/Qwen3-0.6B`` -> ``qwen3_0.6b`` (the dot is
    kept so parameter counts stay readable).
    """
    return re.sub(r"[^a-z0-9._]+", "_", model_id.split("/")[-1].lower())


def _print_ranking(title: str, scores: dict, metric):
    """Print a ranked sensitivity table (already ordered most-sensitive-first)."""
    print(f"\n=== {title} (most sensitive first) ===")
    print(f"{'Rank':>4}  {metric.name:>14}  Name")
    for rank, (name, score) in enumerate(scores.items(), start=1):
        print(f"{rank:>4}  {score:>14.4f}  {name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM quantization sensitivity analysis (released aimet_onnx.analysis API)."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model id to analyze, e.g. microsoft/Phi-4-mini-instruct.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Prefill sequence length (default: %(default)s).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="KV cache context length (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=("weights", "kv_cache"),
        default="weights",
        help="Which sensitivity sweep to run. 'weights' = per-weight-quantizer "
        "int4 sensitivity (default); 'kv_cache' = per-KV-cache-input int8 sensitivity.",
    )
    args = parser.parse_args()
    if args.sequence_length > args.context_length:
        parser.error(
            f"--sequence-length ({args.sequence_length}) must not exceed "
            f"--context-length ({args.context_length})"
        )
    return args


def main():
    args = _parse_args()
    sequence_length = args.sequence_length
    context_length = args.context_length
    slug = _model_slug(args.model_id)
    checkpoint_dir = f"{ONNX_CHECKPOINT_ROOT}/{slug}"

    # --- Instantiate the torch model and make it ONNX-traceable -------------
    print(f"Loading model: {args.model_id}")
    model = LLM.instantiate_model(args.model_id, small_model=False).to(
        dtype=torch.float32
    )
    tokenizer = LLM.instantiate_tokenizer(args.model_id)

    traceable_model = ONNXExportableModuleWithCache(model)
    layer_cache_descriptors = build_layer_cache_descriptors(traceable_model.config)

    dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
    dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)
    assembled_dummy_inputs = Generator.prepare_inputs(
        model=traceable_model,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        past_key_values=[],
        context_length=context_length,
        sequence_length=sequence_length,
        layer_cache_descriptors=layer_cache_descriptors,
        attention_mask_min=ATTENTION_MASK_MIN,
    )
    input_names = LLM.get_backbone_input_names(layer_cache_descriptors)
    output_names = LLM.get_backbone_output_names(layer_cache_descriptors)

    print("Exporting model to ONNX...")
    onnx_model, _ = get_onnx_model(
        checkpoint=checkpoint_dir,
        fp_backbone_model=traceable_model,
        context_length=context_length,
        sequence_length=sequence_length,
        sample_input=tuple(assembled_dummy_inputs.values()),
        input_names=input_names,
        output_names=output_names,
    )
    print("ONNX export complete!")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    providers = get_ort_providers(device)
    print(f"Using device: {device}")

    # The floating-point session is the reference for the top-k logit PSNR
    # metric, and is also what we prefill on to build the feed dicts. Create it
    # before quantsim so the torch model can be dropped from memory first.
    fp_session = OrtInferenceSession(onnx_model, providers=providers)

    model_config = copy.deepcopy(traceable_model.config)
    del traceable_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Calibration / evaluation feed dicts -------------------------------
    # Both sweeps need numpy feed dicts (prefill inputs plus the FP K/V cache).
    # Prefill on the FP graph: it is faster than the uncalibrated quantsim
    # session, and the cached K/V are true FP values. `_prefill_inputs` wants
    # something with a `.session`, hence `_SessionHolder`.
    print("Building calibration and evaluation feed dicts (FP prefill)...")
    fp_prefill_model = _fp_prefill_model(fp_session, model_config)
    fp_generator = Generator(
        fp_prefill_model,
        tokenizer,
        sequence_length,
        context_length,
        attention_mask_min=ATTENTION_MASK_MIN,
    )
    train_dataset = Wikitext.load_encoded_dataset(tokenizer, context_length, "train")
    calib_feeds = _prefill_inputs(
        _SessionHolder(fp_session),
        fp_generator,
        train_dataset,
        NUM_CALIBRATION_ITERATIONS,
    )
    eval_feeds = _prefill_inputs(
        _SessionHolder(fp_session), fp_generator, train_dataset, NUM_EVAL_ITERATIONS
    )

    # --- Quantsim ----------------------------------------------------------
    print("Creating ONNX QuantizationSimModel...")
    with (
        AttributePatch(quantsim, "op_types_to_tie_qtzrs", ["Concat"]),
        AttributePatch(quantsim, "_tie_qtzrs", True),
        AttributePatch(
            quantsim,
            "op_outputs_to_ignore",
            quantsim.op_outputs_to_ignore + ["Slice", "Constant"],
        ),
    ):
        quant_sim = QuantizationSimModel(
            model=onnx_model,
            quant_scheme="min_max",
            param_type=int4,
            activation_type=int16,
            config_file=QUANTSIM_CONFIG,
            providers=providers,
        )
    print("ONNX QuantizationSimModel created!")

    if args.mode == "weights":
        # Weight-only sweep: strip activation quantizers so the measured
        # sensitivity is attributable to int4 weights alone.
        _remove_activation_quantizers(quant_sim)
    else:
        # KV-cache sweep: only the past_key/past_value graph inputs matter.
        kv_names = set(kv_quantizer_names(quant_sim))
        if not kv_names:
            raise RuntimeError(
                "KV cache mode: no past_key_<i>_in / past_value_<i>_in quantizers found"
            )
        # The KV inputs are tied to their Concat outputs, so a quantizer object
        # is shared across the tie group. Disable everything first, then enable
        # and configure just the KV names -- setting bitwidth on the shared
        # object last is what makes the tie group land at INT8 symmetric.
        for qtzr in quant_sim.qc_quantize_op_dict.values():
            qtzr.enabled = False
        for name in kv_names:
            qtzr = quant_sim.qc_quantize_op_dict[name]
            qtzr.set_bitwidth(8)
            qtzr.use_symmetric_encodings = True
            qtzr.enabled = True
        print(
            f"KV cache mode: configured {len(kv_names)} KV quantizers at INT8 symmetric"
        )

    print("Running compute_encodings...")
    with compute_encodings(quant_sim):
        for feed in tqdm(calib_feeds, desc="Calibrating"):
            quant_sim.session.run(None, feed)
    print("Calibration complete!")

    # Top-k logit PSNR against the FP session: higher PSNR is better, so the
    # metric ranks lower PSNR as more sensitive.
    metric = make_topk_logit_psnr_metric(fp_session, eval_feeds, k=10)

    if args.mode == "weights":
        # Key the sweep by owning ONNX node name (e.g.
        # /model/layers.0/self_attn/q_proj/MatMul) rather than by the opaque
        # weight initializer name torch export produces (onnx::MatMul_9772).
        # That makes the report and plot readable, lets the plot's projection
        # highlight toggles match, and -- since node names are exactly what
        # lite_mp looks up in the connected graph -- lets `scores` feed
        # flip_layers_to_higher_precision with no remapping.
        group_fn = group_by_op_name(quant_sim)
        scores = analyze_per_quantizer_sensitivity(quant_sim, metric, group_fn=group_fn)
        # Safe to inspect enabled state after the sweep: it restores it on exit.
        details = quantizers_by_group(quant_sim, group_fn, scores)
        _print_ranking("Weight Sensitivity Report", scores, metric)

        print("\nFlipping the top-10% most sensitive weights to int16...")
        flip_layers_to_higher_precision(
            quant_sim, scores, percent_to_flip=10, override_precision=int16
        )
    else:
        # Restrict the same per-quantizer sweep to the KV inputs by returning
        # None from group_fn for everything else. KV inputs are graph inputs with
        # already-structured names (past_key_0_in), so they are not renamed.
        kv_names = set(kv_quantizer_names(quant_sim))
        group_fn = lambda name: name if name in kv_names else None  # noqa: E731
        details = None
        scores = analyze_per_quantizer_sensitivity(quant_sim, metric, group_fn=group_fn)
        _print_ranking("KV Cache Sensitivity Report", scores, metric)

    # `scores` is ranked most-sensitive-first, which is what we persist. The
    # plot reads better in graph order, so re-key it by the quantizer order in
    # qc_quantize_op_dict (roughly topological) before plotting.
    topo_scores = {}
    for name in quant_sim.qc_quantize_op_dict:
        key = group_fn(name)
        if key in scores and key not in topo_scores:
            topo_scores[key] = scores[key]

    html_path = f"./{slug}_{args.mode}_sensitivity.html"
    json_path = f"./{slug}_{args.mode}_sensitivity.json"
    save_sensitivity_plot(
        topo_scores,
        metric,
        save_path=html_path,
        details=details,
        details_label="Quantizer",
    )
    save_sensitivity_results(scores, save_path=json_path, details=details)
    print(f"Saved sensitivity plot to {html_path} and results to {json_path}")

    print("Done.")


if __name__ == "__main__":
    main()

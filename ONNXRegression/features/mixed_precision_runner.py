# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=missing-module-docstring

"""
Automatic Mixed-Precision (AMP) Feature Runner for AIMET ONNX

This module implements AIMET's automatic mixed-precision optimization using
the choose_mixed_precision() API, which automatically determines optimal precision
per layer through intelligent search.

Algorithm Overview:
1. Create a base QuantSim model with uniform quantization
2. Define candidate precisions (e.g., int8, float16)
3. AIMET searches for best precision per layer using accuracy-based evaluation
4. Optional two-phase optimization: fast Phase 1 pruning, thorough Phase 2 selection
5. Returns QuantSim with optimal mixed precision configuration

Key Features:
- Intelligent per-layer precision selection
- Two-phase optimization for faster convergence
- Pareto front analysis: Optimal bitops vs accuracy tradeoff
- Hardware-aware: Supports int4, int8, int16, float16

Configuration Examples:
    # Fast two-phase optimization
    candidates: "int8_fp16"
    enable_sqnr_pruning: true
    phase1_samples: 64    # Fewer samples for fast pruning
    phase2_samples: 256   # More samples for accurate selection

    # Thorough single-phase search
    candidates: ["int8", "float16"]
    enable_sqnr_pruning: false
    eval_samples: 256

    # Aggressive quantization
    candidates: ["int4", "int8", "float16"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import onnxruntime as ort
from qai_hub_models.utils.evaluate import evaluate_session_on_dataset

from aimet_onnx.mixed_precision import choose_mixed_precision
from aimet_onnx.common.amp.utils import AMPSearchAlgo
from aimet_onnx.common.defs import CallbackFunc, QuantizationDataType

from ONNXRegression.evaluation.metrics_utils import measure_inference_metrics
from ONNXRegression.features._common import build_quantsim, export_aimet_bundle

_ARTIFACTS_DIR = Path("./ONNXRegression/artifacts")
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_candidates_to_aimet_format(
    candidates_config: Any,
) -> List[Tuple[Tuple[int, QuantizationDataType], Tuple[int, QuantizationDataType]]]:
    """
    Parse candidate configuration into AIMET mixed_precision format.

    AIMET expects candidates as list of tuples containing parameter and activation
    configurations. Each configuration is (bitwidth, datatype).

    Args:
        candidates_config: Can be:
            - Preset string: "int8_fp16", "int4_int8_fp16", "int8_int16"
            - List of strings: ["int8", "float16"]
            - List of dicts (legacy): [{"data_type": "int8", "bitwidth": 8}, ...]

    Returns:
        List of candidate tuples in AIMET format:
        [((param_bw, param_dtype), (act_bw, act_dtype)), ...]

    Example:
        >>> _parse_candidates_to_aimet_format("int8_fp16")
        [((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
         ((16, QuantizationDataType.float), (16, QuantizationDataType.float))]
    """
    presets = {
        "int8_fp16": [
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
        ],
        "int4_int8_fp16": [
            ((4, QuantizationDataType.int), (4, QuantizationDataType.int)),
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
        ],
        "int8_int16": [
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
        ],
    }

    if isinstance(candidates_config, str) and candidates_config in presets:
        return presets[candidates_config]

    if isinstance(candidates_config, list):
        if all(isinstance(c, str) for c in candidates_config):
            dtype_map = {
                "int4": (4, QuantizationDataType.int),
                "int8": (8, QuantizationDataType.int),
                "int16": (16, QuantizationDataType.int),
                "float16": (16, QuantizationDataType.float),
            }
            return [
                (dtype_map.get(dtype, (8, QuantizationDataType.int)),) * 2
                for dtype in candidates_config
            ]
        elif all(isinstance(c, dict) for c in candidates_config):
            dtype_str_map = {
                "int": QuantizationDataType.int,
                "float": QuantizationDataType.float,
            }
            result = []
            for c in candidates_config:
                bw = c.get("bitwidth", 8)
                dt_str = (
                    c.get("data_type", "int").replace("int", "").replace("float", "")
                )
                dt = (
                    QuantizationDataType.float
                    if "float" in c.get("data_type", "int").lower()
                    else QuantizationDataType.int
                )
                result.append(((bw, dt), (bw, dt)))
            return result

    return presets["int8_fp16"]


def run_mixed_precision(
    *,
    fp32_onnx_path: str,
    model: Any,
    dataset_name: str,
    config: Dict[str, Any],
    export_dir: Optional[Path] = None,
) -> Tuple[str, float, Dict[str, str], str]:
    """
    Apply automatic mixed-precision optimization using AIMET's search algorithm.

    This technique uses AIMET's choose_mixed_precision() API to automatically
    determine the optimal precision for each layer through accuracy-based evaluation.
    Supports optional two-phase optimization where Phase 1 uses fewer samples for
    fast candidate pruning and Phase 2 uses more samples for accurate selection.

    Args:
        fp32_onnx_path: Path to FP32 ONNX model
        model: QAI Hub model object (provides preprocessing/postprocessing)
        dataset_name: Dataset name for evaluation
        config: Configuration dictionary containing:
            Required:
                - model_name: Name for output files
            AMP specific:
                - candidates: Precision candidates (default: "int8_fp16")
                  Options:
                    - Preset: "int8_fp16", "int4_int8_fp16", "int8_int16"
                    - List: ["int8", "float16"]
                - enable_sqnr_pruning: Enable two-phase optimization (default: False)
                  When True, Phase 1 uses fewer samples for faster pruning
                - allowed_accuracy_drop: Max accuracy drop from FP32 (default: 0.01)
            Base Quantization:
                - quant_scheme: Quantization scheme for base model (default: "tf_enhanced")
                  Options: "tf_enhanced", "tf", "percentile", "entropy"
                - param_type: Weight precision for base model (default: "int8")
                - activation_type: Activation precision for base model (default: "int8")
                - config_file: Optional AIMET config file path
            Evaluation:
                - calib_samples: Calibration samples (default: 256)
                - eval_samples: Final evaluation samples (default: 256)
                - phase1_samples: Phase 1 evaluation samples for fast pruning (default: 64)
                - phase2_samples: Phase 2 evaluation samples for accurate selection (default: 256)
                - metrics_samples: Performance measurement samples (default: 64)
                - metrics_runs: Number of timing runs (default: 1)
                - metrics_warmup: Warmup runs before timing (default: 0)
        export_dir: Optional export directory (default: artifacts/<model_name>)

    Returns:
        Tuple containing:
            - exported_onnx_path: Path to optimized ONNX model
            - feature_accuracy: Accuracy after AMP optimization
            - stats: Dict with "techniques", "runtime", "memory"
            - aimet_bundle_dir: Directory with ONNX + encodings for QNN

    Raises:
        Exception: If AIMET mixed_precision optimization fails

    Performance Note:
        Two-phase (enable_sqnr_pruning=true): 5-15 minutes
        Single-phase (enable_sqnr_pruning=false): 10-30 minutes
    """
    model_name = config["model_name"]

    if export_dir is None:
        export_dir = config.get("_export_dir")
        if export_dir:
            export_dir = Path(export_dir)
        else:
            export_dir = Path("./ONNXRegression/artifacts") / model_name
            export_dir.mkdir(parents=True, exist_ok=True)
    else:
        export_dir = Path(export_dir)

    candidates_config = config.get("candidates", "int8_fp16")
    candidates = _parse_candidates_to_aimet_format(candidates_config)

    enable_sqnr = config.get("enable_sqnr_pruning", False)
    allowed_accuracy_drop = float(config.get("allowed_accuracy_drop", 0.01))

    quant_scheme = str(config.get("quant_scheme", "tf_enhanced"))
    param_type = str(config.get("param_type", "int8"))
    activation_type = str(config.get("activation_type", "int8"))
    aimet_cfg_file = config.get("config_file", None)

    calib_samples = int(config.get("calib_samples", 256))
    eval_samples = int(config.get("eval_samples", 256))
    phase1_samples = int(config.get("phase1_samples", 64))
    phase2_samples = int(config.get("phase2_samples", calib_samples))
    metrics_samples = int(config.get("metrics_samples", 64))
    metrics_runs = int(config.get("metrics_runs", 1))
    metrics_warmup = int(config.get("metrics_warmup", 0))

    use_cuda = "CUDAExecutionProvider" in ort.get_available_providers()

    print(f"[AMP] Configuration:")
    print(
        f"  Base quantization: W{param_type}/A{activation_type}, scheme={quant_scheme}"
    )
    print(f"  Candidates: {len(candidates)} precision options")
    print(f"  Two-phase optimization: {'Enabled' if enable_sqnr else 'Disabled'}")
    print(f"  Allowed accuracy drop: {allowed_accuracy_drop:.2%}")
    print(f"  CUDA acceleration: {'Enabled' if use_cuda else 'Disabled'}")

    print(f"[AMP] Step 1: Creating base QuantSim model...")
    sim = build_quantsim(
        fp32_or_fpN_onnx_path=fp32_onnx_path,
        scheme=quant_scheme,
        param_type=param_type,
        activation_type=activation_type,
        config_file=aimet_cfg_file,
        use_cuda=use_cuda,
    )

    print(f"[AMP] Step 2: Performing initial calibration...")

    def calibration_callback(sess: ort.InferenceSession, _unused=None):
        """Forward pass callback for AIMET calibration."""
        evaluate_session_on_dataset(
            sess, model, dataset_name, num_samples=calib_samples
        )

    sim.compute_encodings(
        forward_pass_callback=calibration_callback, forward_pass_callback_args=None
    )

    print(f"[AMP] Step 3: Preparing evaluation callbacks...")

    if enable_sqnr:
        print(f"[AMP] Using two-phase optimization (Phase 1: fast, Phase 2: thorough)")
        phase1_eval_samples = phase1_samples
        phase2_eval_samples = phase2_samples
    else:
        print(f"[AMP] Using single-phase search (thorough evaluation)")
        phase1_eval_samples = eval_samples
        phase2_eval_samples = eval_samples

    def eval_callback_phase1(sess: ort.InferenceSession) -> float:
        """Phase 1 evaluation with fewer samples for faster candidate pruning."""
        acc, _ = evaluate_session_on_dataset(
            sess, model, dataset_name, num_samples=phase1_eval_samples
        )
        return float(acc)

    def eval_callback_phase2(sess: ort.InferenceSession) -> float:
        """Phase 2 evaluation with more samples for accurate selection."""
        acc, _ = evaluate_session_on_dataset(
            sess, model, dataset_name, num_samples=phase2_eval_samples
        )
        return float(acc)

    print(f"[AMP] Step 4: Running automatic mixed-precision search...")
    if enable_sqnr:
        print(f"[AMP] This may take 5-15 minutes with two-phase optimization...")
    else:
        print(f"[AMP] This may take 10-30 minutes with single-phase search...")

    results_dir = export_dir / "amp_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    def forward_pass_callback(sess: ort.InferenceSession, _unused=None):
        """Forward pass callback for AIMET mixed precision calibration."""
        evaluate_session_on_dataset(
            sess, model, dataset_name, num_samples=calib_samples
        )

    forward_pass_cb = CallbackFunc(forward_pass_callback, func_callback_args=None)

    pareto_list = choose_mixed_precision(
        sim=sim,
        candidates=candidates,
        eval_callback_for_phase1=eval_callback_phase1,
        eval_callback_for_phase2=eval_callback_phase2,
        allowed_accuracy_drop=allowed_accuracy_drop,
        results_dir=str(results_dir),
        clean_start=True,
        forward_pass_callback=forward_pass_cb,
        use_all_amp_candidates=False,
        phase1_optimize=enable_sqnr,
        amp_search_algo=AMPSearchAlgo.Binary,
    )

    print(f"[AMP] Search complete - optimal precision found per layer")
    print(f"[AMP] Pareto front analysis saved to: {results_dir}")

    print(f"[AMP] Step 5: Evaluating final mixed-precision model...")
    feature_acc, _ = evaluate_session_on_dataset(
        sim.session, model, dataset_name, num_samples=eval_samples
    )
    feature_acc = float(feature_acc)

    print(f"[AMP] Mixed-precision accuracy: {feature_acc:.4f}")

    print(f"[AMP] Step 6: Measuring inference performance...")
    runtime_str, memory_str = measure_inference_metrics(
        lambda: evaluate_session_on_dataset(
            sim.session, model, dataset_name, num_samples=metrics_samples
        ),
        runs=metrics_runs,
        warmup=metrics_warmup,
    )

    print(f"[AMP] Runtime: {runtime_str}, Memory: {memory_str}")

    print(f"[AMP] Step 7: Exporting mixed-precision model...")
    qdq_path, bundle_dir = export_aimet_bundle(sim, export_dir, model_name)

    print(f"[AMP] Bundle created at: {bundle_dir}")

    fp32_acc = config.get("_fp32_acc", None)
    if fp32_acc is not None:
        actual_accuracy_drop = fp32_acc - feature_acc
        accuracy_drop_pct = (
            (actual_accuracy_drop / fp32_acc) * 100 if fp32_acc > 0 else 0
        )
        print(
            f"[AMP] Accuracy drop: {actual_accuracy_drop:.4f} ({accuracy_drop_pct:.2f}%)"
        )
    else:
        actual_accuracy_drop = None
        accuracy_drop_pct = None

    candidates_str = "+".join(
        [
            f"W{c[0][0]}A{c[1][0]}{'int' if c[0][1].name == 'int' else 'fp'}"
            for c in candidates
        ]
    )

    phase_str = "2phase" if enable_sqnr else "1phase"
    allowed_drop_pct = allowed_accuracy_drop * 100

    technique_desc = (
        f"amp({candidates_str}, {phase_str}, allow_drop={allowed_drop_pct:.1f}%)"
    )
    if actual_accuracy_drop is not None:
        technique_desc += f", actual_drop={accuracy_drop_pct:.2f}%"

    stats = {
        "techniques": technique_desc,
        "runtime": runtime_str,
        "memory": memory_str,
    }

    return qdq_path, feature_acc, stats, str(bundle_dir)

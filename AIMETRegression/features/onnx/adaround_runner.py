# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
AdaRound Feature Runner for AIMET ONNX

This module implements AIMET's Adaptive Rounding (AdaRound) post-training
quantization optimization technique.

Algorithm Overview:
AdaRound optimizes the rounding of weights during quantization to minimize
reconstruction error. Traditional quantization uses "round-to-nearest" which
may not be optimal. AdaRound learns whether each weight should be rounded
up or down to minimize the layer-wise reconstruction error.

Key Benefits:
- Improved accuracy compared to naive rounding
- No retraining required (post-training optimization)
- Fast optimization (typically 15,000 iterations)
- Works well for INT8 quantization

Process:
1. Build QuantSim model with initial quantization settings
2. Calibrate encodings using representative data
3. Apply AdaRound optimization to learn optimal weight rounding
4. Re-calibrate encodings with optimized weights
5. Export the optimized quantized model

Technical Details:
AdaRound formulates weight rounding as an optimization problem where for each
weight w, we learn a continuous parameter α ∈ [0,1] that determines whether
to round down (α→0) or round up (α→1). This is optimized using unlabeled
data to minimize layer-wise reconstruction error.

Reference:
"Up or Down? Adaptive Rounding for Post-Training Quantization"
https://arxiv.org/abs/2004.10568
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import onnx
import onnxruntime as ort
from qai_hub_models.datasets import BaseDataset
from qai_hub_models.utils.evaluate.helpers import evaluate_session_on_dataset
import aimet_onnx  # For top-level AdaRound API (AIMET 2.15+)

from AIMETRegression.evaluation.calibration import run_onnx_calibration
from AIMETRegression.evaluation.metrics_utils import measure_inference_metrics
from AIMETRegression.features.onnx._common import (
    build_quantsim,
    export_onnx_qdq,
    format_quantsim_technique,
    pick_providers,
)

# Output directory for AIMET artifacts
_ARTIFACTS_DIR = Path("./AIMETRegression/artifacts")
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _capture_unlabeled_feeds(
    sess: ort.InferenceSession,
    model: Any,
    dataset_cls: type[BaseDataset],
    num_samples: int,
) -> List[Dict[str, Any]]:
    """
    Capture real input feeds for AdaRound optimization.

    AdaRound requires unlabeled data to optimize weight rounding. This function
    captures actual model inputs by intercepting calls during a mini evaluation.
    The captured feeds are used to compute layer-wise reconstruction error
    during AdaRound optimization.

    Args:
        sess: ORT InferenceSession to intercept
        model: QAI Hub model object (for preprocessing)
        dataset_cls: Dataset class to sample from (e.g., ImagenetDataset)
        num_samples: Number of samples to capture

    Returns:
        List of feed dictionaries, each containing:
        {input_name: numpy_array} formatted for AIMET's apply_adaround()

    Technical Note:
        We use monkey-patching to intercept sess.run() calls. This is cleaner
        than reimplementing the evaluation logic and ensures we capture exactly
        what the model sees during inference.
    """
    captured: List[Dict[str, Any]] = []
    original_run = sess.run

    def _intercept_run(output_names, input_feed, *args, **kwargs):
        """Intercept and capture input feeds during sess.run()."""
        if isinstance(input_feed, dict):
            # Convert to numpy and store a copy
            feed = {}
            for key, value in input_feed.items():
                # Handle both numpy arrays and torch tensors
                if hasattr(value, "numpy"):
                    feed[key] = value.numpy()
                else:
                    feed[key] = value
            captured.append(feed)

        # Call original run method
        return original_run(output_names, input_feed, *args, **kwargs)

    # Temporarily replace sess.run with our interceptor
    sess.run = _intercept_run

    try:
        # Run evaluation to trigger data capture
        run_onnx_calibration(sess, model, dataset_cls, num_samples)
    finally:
        # Always restore original method
        sess.run = original_run

    return captured


def run_adaround(
    *,
    fp32_onnx_path: str,
    model: Any,
    dataset_cls: type[BaseDataset],
    calib_dataset_cls: type[BaseDataset] | None = None,
    config: Dict[str, Any],
    export_dir: Optional[Path] = None,
) -> Tuple[str, float, Dict[str, str], str]:
    """
    Apply AdaRound optimization for improved weight quantization.

    AdaRound optimizes the rounding of weights during quantization by learning
    whether each weight should be rounded up or down to minimize reconstruction
    error. This typically improves accuracy compared to round-to-nearest.

    Args:
        fp32_onnx_path: Path to FP32 ONNX model from AI Hub
        model: QAI Hub model object (provides preprocessing/postprocessing)
        dataset_cls: Dataset class for evaluation (e.g., ImagenetDataset)
        config: Configuration dictionary containing:
            Required:
                - model_name: Name for output files
            Quantization:
                - quant_scheme: Quantization scheme (default: "tf_enhanced")
                - param_type: Weight precision (default: "int8")
                - activation_type: Activation precision (default: "int8")
                - config_file: Optional AIMET config file
            AdaRound specific:
                - adaround_samples: Samples for optimization (default: 64)
                - adaround_iters: Optimization iterations (default: 15000)
            Evaluation:
                - calib_samples: Calibration samples (default: 256)
                - eval_samples: Evaluation samples (default: 256)
                - metrics_samples: Samples for performance measurement (default: 64)
                - metrics_runs: Number of timing runs (default: 1)
                - metrics_warmup: Warmup runs (default: 0)

    Returns:
        Tuple containing:
            - exported_onnx_path: Path to optimized ONNX model
            - feature_accuracy: Accuracy after AdaRound optimization
            - stats: Dict with "techniques", "runtime", and "memory"

    Raises:
        Exception: If AdaRound optimization fails

    Performance Note:
        AdaRound optimization can be slow (5-10 minutes) depending on model
        size and iteration count. The default 15,000 iterations provides good
        results for most models, but can be reduced for faster experimentation.
    """
    # ============ Extract Configuration ============
    model_name = config["model_name"]

    # Use provided export_dir or extract from config (passed by runner.py)
    if export_dir is None:
        export_dir = config.get("_export_dir")
        if export_dir:
            export_dir = Path(export_dir)
        else:
            # Fallback to default location with model subdirectory
            export_dir = Path("./AIMETRegression/artifacts") / model_name
            export_dir.mkdir(parents=True, exist_ok=True)
    else:
        export_dir = Path(export_dir)

    # Quantization parameters
    quant_scheme = str(config.get("quant_scheme", "tf_enhanced"))
    param_type = str(config.get("param_type", "int8"))
    activation_type = str(config.get("activation_type", "int8"))
    aimet_cfg_file = config.get("config_file", None)

    # Evaluation parameters
    calib_samples = int(config.get("calib_samples", 256))
    eval_samples = int(config.get("eval_samples", 256))

    # AdaRound specific parameters
    # Note: adaround_samples should be small (32-128) for efficiency
    adaround_samples = int(config.get("adaround_samples", min(64, calib_samples)))
    adaround_iters = int(config.get("adaround_iters", 15000))

    # Performance measurement
    metrics_samples = int(config.get("metrics_samples", 64))
    metrics_runs = int(config.get("metrics_runs", 1))
    metrics_warmup = int(config.get("metrics_warmup", 0))

    providers = pick_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])

    print(f"[AdaRound] Configuration:")
    print(f"  Quantization: W{param_type}/A{activation_type}")
    print(f"  AdaRound iterations: {adaround_iters}")
    print(f"  AdaRound samples: {adaround_samples}")
    print(
        f"  CUDA acceleration: "
        f"{'Enabled' if 'CUDAExecutionProvider' in providers else 'Disabled'}"
    )

    # ============ Step 1: Build QuantSim Model ============
    print(f"[AdaRound] Building QuantSim model...")

    sim = build_quantsim(
        fp32_or_fpN_onnx_path=fp32_onnx_path,
        scheme=quant_scheme,
        param_type=param_type,
        activation_type=activation_type,
        config_file=aimet_cfg_file,
        providers=providers,
    )

    # ============ Step 2: Initial Calibration ============
    print(f"[AdaRound] Initial calibration with {calib_samples} samples...")

    def calibration_callback(sess: ort.InferenceSession, _unused=None):
        """Forward pass for encodings calibration."""
        run_onnx_calibration(
            sess, model, calib_dataset_cls or dataset_cls, calib_samples
        )

    # Compute initial encodings (before AdaRound)
    sim.compute_encodings(forward_pass_callback=calibration_callback)

    # ============ Step 3: Capture Unlabeled Data for AdaRound ============
    print(f"[AdaRound] Capturing {adaround_samples} samples for optimization...")

    # Capture real input feeds for AdaRound optimization
    # These are unlabeled - AdaRound only needs inputs, not labels
    unlabeled_feeds = _capture_unlabeled_feeds(
        sim.session,
        model,
        calib_dataset_cls or dataset_cls,
        num_samples=adaround_samples,
    )

    print(f"[AdaRound] Captured {len(unlabeled_feeds)} input feeds")

    # ============ Step 4: Apply AdaRound Optimization ============
    print(f"[AdaRound] Starting optimization ({adaround_iters} iterations)...")
    print(f"[AdaRound] This may take 5-10 minutes depending on model size...")

    # Apply AdaRound using AIMET 2.15+ top-level API
    # This optimizes weight rounding to minimize layer-wise reconstruction error
    aimet_onnx.apply_adaround(sim, unlabeled_feeds, num_iterations=adaround_iters)

    print(f"[AdaRound] Optimization complete")

    # ============ Step 5: Re-calibrate After AdaRound ============
    print(f"[AdaRound] Re-calibrating with optimized weights...")

    # AdaRound changes weight values, so we need to recompute encodings
    # This ensures activation quantization parameters are correct
    sim.compute_encodings(forward_pass_callback=calibration_callback)

    # ============ Step 6: Evaluate Optimized Model ============
    print(f"[AdaRound] Evaluating accuracy with {eval_samples} samples...")

    feature_acc, *_ = evaluate_session_on_dataset(
        sim.session, model, dataset_cls, num_samples=eval_samples
    )
    feature_acc = float(feature_acc)

    print(f"[AdaRound] Optimized accuracy: {feature_acc:.4f}")

    # ============ Step 7: Measure Performance ============
    print(f"[AdaRound] Measuring inference performance...")

    runtime_str, memory_str = measure_inference_metrics(
        lambda: evaluate_session_on_dataset(
            sim.session, model, dataset_cls, num_samples=metrics_samples
        ),
        runs=metrics_runs,
        warmup=metrics_warmup,
    )

    print(f"[AdaRound] Runtime: {runtime_str}, Memory: {memory_str}")

    # ============ Step 8: Export and Bundle ============
    print(f"[AdaRound] Exporting optimized model...")

    # Export QDQ ONNX (encodings embedded) for QNN
    qdq_path = export_onnx_qdq(sim, export_dir, model_name)

    # ============ Step 9: Prepare Results ============
    quantsim_desc = format_quantsim_technique(param_type, activation_type, quant_scheme)
    stats = {
        "techniques": f"{quantsim_desc} + adaround({adaround_iters})",
        "runtime": runtime_str,
        "memory": memory_str,
    }

    return qdq_path, feature_acc, stats

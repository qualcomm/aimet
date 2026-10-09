# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
AIMETRegression Pipeline Runner - Single Test Execution

This module orchestrates the complete AIMET quantization evaluation pipeline
for a single test configuration using the hierarchical config system.

Supports both ONNX and Torch frameworks:
- ONNX: Export FP32 ONNX → AIMET ONNX QuantSim → QDQ ONNX
- Torch: PyTorch model → AIMET Torch QuantSim → QDQ ONNX

Pipeline Steps:
1. Load configuration (merge defaults → profile → model → test)
2. Load model from QAI Hub Models
3. Framework-specific FP32 baseline and AIMET quantization:
   - ONNX: Export to ONNX → FP32 eval → AIMET ONNX quantization
   - Torch: FP32 eval (PyTorch) → AIMET Torch quantization
4. Validate quantization quality (FP32 → AIMET)
5. Evaluate QDQ ONNX model and validate export (AIMET → QDQ)
6. Optionally run on-device evaluation via QNN (AI Hub)
7. Generate comprehensive reports

Usage:
    python runner.py --model resnet50 --test quantsim_int8 --profile nightly
    python runner.py --model resnet50 --test quantsim_int8
    python runner.py --model resnet50 --test quantsim_int8 --profile nightly --dry-run
"""

import os
import sys
import onnx
import torch
import argparse
import contextlib
from pathlib import Path
from typing import Dict, Any

from qai_hub import Device
from qai_hub_models.utils.input_spec import make_torch_inputs
from tabulate import tabulate

from AIMETRegression.models.ai_hub_loader import (
    load_model_data,
    resolve_calibration_dataset_cls,
    resolve_eval_dataset_cls,
)
from AIMETRegression.evaluation.calibration import instantiate_calibration_dataset
from AIMETRegression.evaluation.eval_onnx import eval_onnx_model
from AIMETRegression.evaluation.eval_torch import eval_pytorch_model
from AIMETRegression.evaluation.eval_qnn import (
    compile_and_profile_qdq_model,
    eval_qnn_accuracy,
)
from AIMETRegression.features.torch.utils import ensure_device_patch

from AIMETRegression.report.report_writer import write_csv, write_html
from AIMETRegression.config_loader import load_config, validate_config
from AIMETRegression.baseline_comparison import (
    validate_quantization_quality,
    validate_qdq_export,
    TestResult,
)

os.environ.setdefault("TORCH_HOME", "./torch_cache")
os.environ.setdefault("QAIHM_CACHE_DIR", "./qaihm_cache")

ARTIFACTS_DIR = Path("./AIMETRegression/artifacts")
REPORTS_DIR = Path("./AIMETRegression/reports")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_max_accuracy_drop(config: Dict[str, Any]) -> float:
    """
    Resolve accuracy drop threshold from config with framework-specific override.

    Resolution order:
        1. max_accuracy_drop_{framework} (e.g., max_accuracy_drop_torch)
        2. max_accuracy_drop
        3. Default: 1.0

    Args:
        config: Test configuration dictionary

    Returns:
        Maximum allowed accuracy drop in percentage points
    """
    framework = config.get("framework", "onnx").lower()

    framework_key = f"max_accuracy_drop_{framework}"
    if framework_key in config:
        return float(config[framework_key])

    if "max_accuracy_drop" in config:
        return float(config["max_accuracy_drop"])

    return 1.0


def _resolve_device(device_name: str) -> Device:
    """Convert device name string to AI Hub Device enum."""
    try:
        return Device(device_name)
    except Exception as e:
        raise RuntimeError(
            f"Unknown/unsupported device: '{device_name}'. Error: {e}\n"
            f"Check available devices at: https://app.aihub.qualcomm.com/devices"
        )


@contextlib.contextmanager
def _disable_torch_mha_fastpath():
    """
    Context manager to temporarily disable torch MHA fastpath.

    This is needed to export some transformer models.
    """
    original_setting = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(original_setting)


def _export_torch_to_onnx_local(
    model: Any, input_spec: Dict, out_dir: Path, model_name: str
) -> Path:
    """Export PyTorch model to ONNX locally using torch.onnx.export."""
    print(f"\n[INFO] Exporting {model_name} to FP32 ONNX locally...")
    print(f"[INFO] Using torch.onnx.export (no AI Hub compilation)")

    if hasattr(model, "to_torch_model"):
        torch_model = model.to_torch_model().to("cpu").eval()
    else:
        torch_model = model.to("cpu").eval()

    sample_inputs = make_torch_inputs(input_spec)

    fp32_path = out_dir / f"{model_name}_fp32.onnx"

    print(f"[INFO] Converting model to ONNX...")

    if isinstance(sample_inputs, dict):
        input_names = list(input_spec.keys())
        input_values = list(sample_inputs.values())

        if len(input_values) == 1:
            sample_inputs_for_export = input_values[0]
        else:
            sample_inputs_for_export = tuple(input_values)
    else:
        if isinstance(sample_inputs, list):
            sample_inputs = tuple(sample_inputs)
        if isinstance(sample_inputs, tuple):
            sample_inputs_for_export = sample_inputs
            if input_spec and isinstance(input_spec, dict):
                input_names = list(input_spec.keys())
            else:
                input_names = [f"input_{i}" for i in range(len(sample_inputs))]
        else:
            sample_inputs_for_export = sample_inputs
            if input_spec and isinstance(input_spec, dict):
                input_names = list(input_spec.keys())
            else:
                input_names = ["input"]

    dynamic_axes = None
    export_success = False
    export_error = None

    with torch.no_grad(), _disable_torch_mha_fastpath():
        try:
            torch.onnx.export(
                torch_model,
                sample_inputs_for_export,
                str(fp32_path),
                input_names=input_names,
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
                export_params=True,
                dynamo=False,
            )
            export_success = True
        except Exception as e:
            export_error = e
            if isinstance(sample_inputs_for_export, torch.Tensor):
                print(
                    f"[INFO] First export attempt failed, trying with tuple wrapper..."
                )
                try:
                    torch.onnx.export(
                        torch_model,
                        (sample_inputs_for_export,),
                        str(fp32_path),
                        input_names=input_names,
                        output_names=["output"],
                        dynamic_axes=dynamic_axes,
                        opset_version=17,
                        do_constant_folding=True,
                        export_params=True,
                        dynamo=False,
                    )
                    export_success = True
                except Exception as e2:
                    export_error = e2

    if not export_success:
        raise RuntimeError(
            f"Failed to export ONNX model: {export_error}\n"
            f"Input format: {type(sample_inputs_for_export)}\n"
            f"Input names: {input_names}"
        ) from export_error

    print(f"[INFO] FP32 ONNX saved to: {fp32_path}")

    try:
        onnx_model = onnx.load(str(fp32_path))
        onnx.checker.check_model(onnx_model)
        print(f"[INFO] ONNX model validation passed")
    except Exception as e:
        print(f"[WARNING] ONNX validation warning (non-fatal): {e}")

    return fp32_path


def _build_single_batch_loader(
    model: Any, dataset_cls: type, input_spec: Dict, num_samples: int
):
    """Create a dataloader for QNN evaluation."""
    import numpy as np
    import torch
    from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
    from qai_hub_models.utils.evaluate.helpers import get_deterministic_sample

    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    dataset = instantiate_dataset(dataset_cls, DatasetSplit.VAL)
    sampler = get_deterministic_sample(
        dataset, num_samples=num_samples, samples_per_job=num_samples
    )

    inputs_list = []
    labels_list = []

    for sample in sampler:
        if isinstance(sample, (list, tuple)) and len(sample) == 2:
            inputs, label = sample
        else:
            inputs, label = sample, None

        if isinstance(inputs, dict):
            x = next(iter(inputs.values()))
        elif isinstance(inputs, (list, tuple)):
            x = inputs[0]
        else:
            x = inputs

        x = to_numpy(x)
        if x.ndim == 3:
            x = np.expand_dims(x, 0)
        inputs_list.append(x)

        if label is not None:
            lbl = to_numpy(label).flatten().astype(int)
            labels_list.extend(lbl.tolist())

    if not inputs_list:
        return [(np.empty((0,)), np.empty((0,), dtype=np.int64))]

    batch_inputs = np.concatenate(inputs_list, axis=0)
    batch_labels = (
        np.array(labels_list, dtype=np.int64)
        if labels_list
        else np.empty((0,), dtype=np.int64)
    )

    return [(batch_inputs, batch_labels)]


def _eval_pytorch_fp32(model: Any, dataset_cls: type, num_samples: int) -> float:
    """
    Evaluate FP32 PyTorch model accuracy.

    Note: QAI Hub models include preprocessing (normalization) with mean/std
    tensors that may stay on CPU. We use the device patch to handle this.
    """
    # The device patch is applied inside eval_pytorch_model, but we can
    # also evaluate on CPU to avoid any device mismatch issues
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        # Try GPU evaluation with device patch
        try:
            if hasattr(model, "to_torch_model"):
                torch_model = model.to_torch_model()
            else:
                torch_model = model

            device = torch.device("cuda")
            torch_model = torch_model.to(device).eval()

            return eval_pytorch_model(
                torch_model, model, dataset_cls, num_samples=num_samples
            )
        except RuntimeError as e:
            if "device" in str(e).lower():
                print(
                    f"[WARNING] GPU evaluation failed due to device mismatch, falling back to CPU..."
                )
                # Fall through to CPU evaluation
            else:
                raise

    # CPU evaluation (fallback or default if no CUDA)
    if hasattr(model, "to_torch_model"):
        torch_model = model.to_torch_model()
    else:
        torch_model = model

    torch_model = torch_model.cpu().eval()

    return eval_pytorch_model(torch_model, model, dataset_cls, num_samples=num_samples)


def run_single_config(
    config: Dict[str, Any], skip_reports: bool = False
) -> Dict[str, Any]:
    """Execute the complete evaluation pipeline for a single configuration.
    Args:
        config: Test configuration
        skip_reports: If True, skip generating CSV/HTML reports (when called from suite)

    Returns:
        Dictionary with test results
    """
    validate_config(config)

    model_name = config.get("model_name")
    device_name = config.get("device", "Samsung Galaxy S24 (Family)")
    feature_name = config.get("feature", "quantsim").strip().lower()
    framework = config.get("framework", "onnx").strip().lower()

    if framework == "onnx":
        from AIMETRegression.features.onnx.quantsim_runner import run_quantsim
        from AIMETRegression.features.onnx.lite_mp_runner import run_lite_mp
        from AIMETRegression.features.onnx.adaround_runner import run_adaround
        from AIMETRegression.features.onnx.mixed_precision_runner import (
            run_mixed_precision,
        )

        FEATURE_RUNNERS = {
            "quantsim": run_quantsim,
            "lite_mp": run_lite_mp,
            "adaround": run_adaround,
            "mixed_precision": run_mixed_precision,
        }

        if feature_name not in FEATURE_RUNNERS:
            raise RuntimeError(
                f"Unsupported feature for ONNX: {feature_name}. "
                f"Supported: {list(FEATURE_RUNNERS.keys())}"
            )
    elif framework == "torch":
        from AIMETRegression.features.torch.quantsim_runner import (
            run_quantsim as run_quantsim_torch,
        )
        from AIMETRegression.features.torch.adaround_runner import (
            run_adaround as run_adaround_torch,
        )
        from AIMETRegression.features.torch.mixed_precision_runner import (
            run_mixed_precision as run_mixed_precision_torch,
        )

        TORCH_FEATURE_RUNNERS = {
            "quantsim": run_quantsim_torch,
            "adaround": run_adaround_torch,
            "mixed_precision": run_mixed_precision_torch,
        }

        if feature_name not in TORCH_FEATURE_RUNNERS:
            raise RuntimeError(
                f"Unsupported feature for Torch: {feature_name}. "
                f"Supported: {list(TORCH_FEATURE_RUNNERS.keys())}"
            )
    else:
        raise RuntimeError(
            f"Unsupported framework: {framework}. Supported: onnx, torch"
        )

    print(f"\n{'=' * 60}")
    print(f"Running Single Test")
    print(f"{'=' * 60}")
    print(f"Model:     {model_name}")
    print(f"Framework: {framework}")
    print(f"Feature:   {feature_name}")
    print(f"Device:    {device_name}")
    print(f"{'=' * 60}\n")

    model_artifacts_dir = ARTIFACTS_DIR / model_name
    model_artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 1] Loading model and dataset from QAI Hub Models...")
    model, _dataset, input_spec, _ = load_model_data(model_name)
    dataset_cls = resolve_eval_dataset_cls(model)
    calib_dataset_cls = resolve_calibration_dataset_cls(model)
    print(f"Eval dataset: {dataset_cls.dataset_name()}")
    print(f"Calibration dataset: {calib_dataset_cls.dataset_name()}")

    # Clamp sample counts to dataset size so profiles with large values
    # (e.g., weekly eval_samples=3925) don't crash on smaller datasets
    # (e.g., pascal_voc=1449, ade20k=2000).
    eval_len = len(_dataset)
    if calib_dataset_cls is dataset_cls:
        calib_len = eval_len
    else:
        calib_len = len(instantiate_calibration_dataset(model, calib_dataset_cls))
    _SAMPLE_KEYS = {
        "eval_samples": eval_len,
        "calib_samples": calib_len,
        "metrics_samples": eval_len,
        "qnn_eval_samples": eval_len,
    }
    for key, limit in _SAMPLE_KEYS.items():
        val = int(config.get(key, 0))
        if val > limit:
            print(f"[Config] Clamping {key} from {val} to {limit} (dataset size)")
            config[key] = limit

    config["_export_dir"] = str(model_artifacts_dir)

    static_aten_acc = None
    eval_samples = int(config.get("eval_samples", 200))

    if framework == "onnx":
        print(f"\n[Step 2] Creating FP32 baseline via ONNX export...")
        fp32_path = _export_torch_to_onnx_local(
            model, input_spec, model_artifacts_dir, model_name
        )

        print(f"[Step 2] Evaluating FP32 accuracy with {eval_samples} samples...")
        fp32_acc = eval_onnx_model(
            fp32_path, model, dataset_cls, num_samples=eval_samples
        )
        print(f"[Step 2] FP32 Accuracy: {fp32_acc:.4f}")

        print(f"\n[Step 3] Applying {feature_name} quantization (ONNX)...")
        runner = FEATURE_RUNNERS[feature_name]

        aimet_onnx_path, feature_acc, stats = runner(
            fp32_onnx_path=str(fp32_path),
            model=model,
            dataset_cls=dataset_cls,
            calib_dataset_cls=calib_dataset_cls,
            config=config,
        )

    elif framework == "torch":
        ensure_device_patch()
        print(f"\n[Step 2] Evaluating FP32 PyTorch model accuracy...")
        fp32_acc = _eval_pytorch_fp32(model, dataset_cls, eval_samples)
        print(f"[Step 2] FP32 Accuracy: {fp32_acc:.4f}")

        print(f"\n[Step 3] Applying {feature_name} quantization (Torch)...")
        runner = TORCH_FEATURE_RUNNERS[feature_name]

        aimet_onnx_path, feature_acc, stats = runner(
            model=model,
            input_spec=input_spec,
            dataset_cls=dataset_cls,
            calib_dataset_cls=calib_dataset_cls,
            config=config,
        )
        static_aten_acc = stats.pop("static_aten_acc", None)

    aimet_onnx_path = Path(aimet_onnx_path)

    if not aimet_onnx_path.exists():
        raise FileNotFoundError(f"AIMET-exported QDQ ONNX not found: {aimet_onnx_path}")

    print(
        f"[Step 3] AIMET Accuracy: {feature_acc:.4f}"
        if feature_acc
        else "[Step 3] AIMET Accuracy: N/A"
    )

    max_drop = get_max_accuracy_drop(config)

    if fp32_acc is not None and feature_acc is not None:
        print(f"\n[Validation] FP32 → AIMET Quality Check")

        test_result = TestResult(
            model=model_name,
            feature=feature_name,
            fp32_accuracy=fp32_acc,
            aimet_accuracy=feature_acc,
            static_aten_acc=static_aten_acc,
            qdq_accuracy=0.0,
            max_accuracy_drop=max_drop,
        )

        quality = validate_quantization_quality(test_result)
        table = {
            "FP32 Accuracy": f"{fp32_acc:.4f}%",
            "AIMET Accuracy": f"{feature_acc:.4f}%",
        }
        if static_aten_acc is not None:
            table |= {
                "└─ (+static ATen calibration)": f"{static_aten_acc:.4f}%",
            }

        table |= {
            "Drop": f"{quality.drop_abs:+.4f} percentage points",
            "Threshold": f"{max_drop:.2f} percentage points",
            "Status": f"{quality.status_emoji}",
        }
        for line in tabulate(table.items(), tablefmt="plain").split("\n"):
            print(f"  {line}")

        if not quality.is_acceptable:
            print(
                f"\n  ❌ Accuracy drop exceeds threshold ({abs(quality.drop_abs):.2f} > {max_drop:.2f})"
            )
        else:
            print(f"  ✅ Quantization quality acceptable")

    print(f"\n[Step 4] Validating exported QDQ ONNX model...")
    print(f"[Step 4] Evaluating: {aimet_onnx_path}")

    if aimet_onnx_path.exists():
        file_size = aimet_onnx_path.stat().st_size
        print(f"[Step 4] File size: {file_size:,} bytes")

        try:
            onnx_model = onnx.load(str(aimet_onnx_path))
            qdq_ops = [
                n
                for n in onnx_model.graph.node
                if "QuantizeLinear" in n.op_type or "DequantizeLinear" in n.op_type
            ]

            if len(qdq_ops) == 0:
                print(
                    f"[Step 4] ❌ ERROR: No QDQ operators found in exported ONNX model"
                )
                print(f"[Step 4]        The model appears to be FP32, not quantized")
            else:
                print(
                    f"[Step 4] ✅ QDQ validation passed: Found {len(qdq_ops)} quantization nodes"
                )
                q_count = sum(1 for n in qdq_ops if n.op_type == "QuantizeLinear")
                dq_count = sum(1 for n in qdq_ops if n.op_type == "DequantizeLinear")
                print(
                    f"[Step 4]   QuantizeLinear: {q_count}, DequantizeLinear: {dq_count}"
                )

        except Exception as e:
            print(f"[Step 4] Warning: Could not inspect ONNX model: {e}")
    else:
        print(f"[Step 4] ❌ ERROR: ONNX file not found at {aimet_onnx_path}")

    qdq_acc = eval_onnx_model(
        str(aimet_onnx_path), model, dataset_cls, num_samples=eval_samples
    )
    print(f"[Step 4] QDQ Accuracy: {qdq_acc:.4f}")

    if feature_acc is not None and qdq_acc is not None:
        print(f"\n[Validation] AIMET → QDQ Export Check")

        test_result = TestResult(
            model=model_name,
            feature=feature_name,
            fp32_accuracy=fp32_acc,
            aimet_accuracy=feature_acc,
            qdq_accuracy=qdq_acc,
            static_aten_acc=static_aten_acc,
            max_accuracy_drop=max_drop,
        )

        export_val = validate_qdq_export(test_result)

        table = {
            "AIMET Accuracy": f"{feature_acc:.4f}%",
        }
        if static_aten_acc is not None:
            table |= {
                "└─ (+static ATen calibration)": f"{static_aten_acc:.4f}%",
            }
        table |= {
            "QDQ Accuracy": f"{qdq_acc:.4f}%",
            "Difference": f"{export_val.diff_abs:+.4f}%p",
            "Status": f"{export_val.status_emoji}",
        }
        for line in tabulate(table.items(), tablefmt="plain").split("\n"):
            print(f"  {line}")

        if not export_val.is_valid:
            print(f"\n  ⚠️  WARNING: Large difference between AIMET and QDQ (>0.5pp)")
            print(f"      This suggests an issue with the ONNX export")
        else:
            print(f"  ✅ Export validation passed (<0.5pp difference)")

    qnn_options = config.get("qnn_options")
    qnn_latency_ms = None
    qnn_acc = None
    qnn_job_urls = {}

    if qnn_options:
        print(f"\n[Step 5] Running QNN on-device evaluation...")
        print(f"[Step 5] QNN options: {qnn_options}")

        if not aimet_onnx_path or not os.path.isfile(str(aimet_onnx_path)):
            print(f"[Step 5] WARNING: QDQ ONNX model missing: {aimet_onnx_path}")
        else:
            try:
                ret = compile_and_profile_qdq_model(
                    qdq_model_path=str(aimet_onnx_path),
                    device_name=device_name,
                    model_name=model_name,
                    export_dir=str(ARTIFACTS_DIR),
                    options=qnn_options,
                )

                if isinstance(ret, (list, tuple)) and len(ret) >= 4:
                    qnn_latency_ms, uploaded_model, _zip, qnn_job_urls = ret
                    print(
                        f"[Step 5] QNN Latency: {qnn_latency_ms:.3f} ms"
                        if qnn_latency_ms
                        else "[Step 5] QNN Latency: N/A"
                    )
                else:
                    uploaded_model = None

            except Exception as e:
                print(f"[Step 5] ERROR: QNN compilation failed: {e}")
                uploaded_model = None

            qnn_eval_samples = int(config.get("qnn_eval_samples", 0))
            if qnn_eval_samples > 0 and uploaded_model is not None:
                print(
                    f"[Step 5] Evaluating on-device accuracy with {qnn_eval_samples} samples..."
                )

                try:
                    qnn_loader = _build_single_batch_loader(
                        model, dataset_cls, input_spec, qnn_eval_samples
                    )

                    channel_last = "--force_channel_last_input" in (qnn_options or "")
                    ret_acc = eval_qnn_accuracy(
                        target_model=uploaded_model,
                        device_name=device_name,
                        input_spec=input_spec,
                        dataset_loader=qnn_loader,
                        channel_last=channel_last,
                        debug_print_feeds=False,
                    )

                    if isinstance(ret_acc, (list, tuple)) and len(ret_acc) >= 2:
                        qnn_acc, inference_urls = ret_acc
                        if inference_urls and isinstance(inference_urls, dict):
                            qnn_job_urls.update(inference_urls)

                    if qnn_acc is not None:
                        print(f"[Step 5] QNN Accuracy: {qnn_acc:.4f}")

                except Exception as e:
                    print(f"[Step 5] ERROR: QNN accuracy evaluation failed: {e}")
    else:
        print(f"\n[Step 5] Skipping QNN evaluation (qnn_options not specified)")

    fp32_vs_aimet_formatted = None
    if fp32_acc is not None and feature_acc is not None:
        test_result = TestResult(
            model=model_name,
            feature=feature_name,
            fp32_accuracy=fp32_acc,
            aimet_accuracy=feature_acc,
            qdq_accuracy=qdq_acc if qdq_acc is not None else 0.0,
            static_aten_acc=static_aten_acc,
            max_accuracy_drop=max_drop,
        )
        quality = validate_quantization_quality(test_result)
        fp32_vs_aimet_formatted = quality.formatted_drop

    result = {
        "Model": model_name,
        "Feature": feature_name,
        "Framework": framework,
        "Techniques": (stats or {}).get("techniques", feature_name),
        "FP32_accuracy": float(fp32_acc) if fp32_acc is not None else None,
        "AIMET Accuracy": float(feature_acc) if feature_acc is not None else None,
        "Static ATen Accuracy": float(static_aten_acc)
        if static_aten_acc is not None
        else None,
        "FP32_vs_AIMET": fp32_vs_aimet_formatted,
        "Max_Accuracy_Drop": max_drop,
        "QDQ Accuracy": float(qdq_acc) if qdq_acc is not None else None,
        "QNN Accuracy": float(qnn_acc * 100) if qnn_acc is not None else None,
        "QNN Latency": f"{qnn_latency_ms:.3f} ms"
        if qnn_latency_ms is not None
        else None,
        "AIMET Runtime": (stats or {}).get("runtime", ""),
        "AIMET Memory": (stats or {}).get("memory", ""),
        "AI Hub QNN Compile Job": qnn_job_urls.get("compile", ""),
        "AI Hub QNN Profile Job": qnn_job_urls.get("profile", ""),
        "AI Hub QNN Inference Job": qnn_job_urls.get("inference", ""),
    }

    print(f"\n[Step 6] Generating reports...")

    if not skip_reports:
        csv_path = str(REPORTS_DIR / "results.csv")
        html_path = str(REPORTS_DIR / "results.html")

        write_csv([result], csv_path)
        write_html([result], html_path)

        print(f"[Step 6] CSV:  {csv_path}")
        print(f"[Step 6] HTML: {html_path}")
    else:
        print(f"[Step 6] Skipping individual reports (suite mode)")

    print(f"\n{'=' * 60}")
    print("Pipeline completed successfully!")
    print(f"{'=' * 60}")

    print("\nResults Summary:")
    for key, value in result.items():
        if value is not None and value != "":
            print(f"  {key:30s}: {value}")

    return result


def main():
    """Main entry point for single test execution."""
    parser = argparse.ArgumentParser(
        description="Run a single AIMET quantization test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., resnet50, mobilenetv2)",
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Test name from model config (e.g., quantsim_int8, lite_mp_25)",
    )

    parser.add_argument(
        "--profile",
        help="Profile name (e.g., nightly, smoke)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show merged configuration without executing the test",
    )

    args = parser.parse_args()

    try:
        print(f"Loading configuration...")
        print(f"  Model: {args.model}")
        print(f"  Test: {args.test}")
        print(f"  Profile: {args.profile or '(none - using defaults)'}")

        model_yaml = f"models/{args.model}.yaml"
        config = load_config(model_yaml, args.test, args.profile)

        if args.dry_run:
            print(f"\n{'=' * 60}")
            print("DRY RUN - Configuration Preview")
            print(f"{'=' * 60}\n")

            print("Merged configuration:")
            for key, value in sorted(config.items()):
                if not key.startswith("_"):
                    print(f"  {key:30s}: {value}")

            print(f"\n{'=' * 60}")
            print("Would execute with this configuration")
            print(f"{'=' * 60}")
            return 0

        result = run_single_config(config)
        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Configuration file not found: {e}")
        return 1

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return 1

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

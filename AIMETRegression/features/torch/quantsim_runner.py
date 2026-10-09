# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
AIMET Torch QuantSim Feature Runner

Applies AIMET's PyTorch QuantizationSimModel to simulate quantization effects
on native PyTorch models without early ONNX conversion.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.export import ExportedProgram
from qai_hub_models.datasets import BaseDataset

from AIMETRegression.evaluation.calibration import (
    get_calibration_dataloader,
    run_torch_calibration,
)
from AIMETRegression.evaluation.eval_torch import eval_pytorch_model, load_torch_dataset
from AIMETRegression.evaluation.metrics_utils import measure_inference_metrics
from AIMETRegression.features.torch._common import (
    bitwidth_from_token,
    build_quantsim_torch,
    export_torch_qdq,
    create_dummy_input,
    parse_output_names_from_qnn_options,
    run_static_aten_calibration,
)


_ARTIFACTS_DIR = Path("./AIMETRegression/artifacts")
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def run_quantsim(
    *,
    model: Any,
    input_spec: Dict,
    dataset_cls: type[BaseDataset],
    calib_dataset_cls: type[BaseDataset] | None = None,
    config: Dict[str, Any],
    export_dir: Path = None,
) -> Tuple[Path, float, Dict[str, str], str]:
    """
    Apply AIMET Torch QuantSim to simulate quantization on a PyTorch model.

    Args:
        model: QAI Hub model object
        input_spec: Input specification for dummy input creation
        dataset_cls: Dataset class for evaluation (e.g., ImagenetDataset)
        config: Configuration dictionary
        export_dir: Optional export directory

    Returns:
        Tuple of (qdq_path, accuracy, stats)
    """
    model_name = config["model_name"]

    if export_dir is None:
        export_dir = config.get("_export_dir")
        if export_dir:
            export_dir = Path(export_dir)
        else:
            export_dir = Path("./AIMETRegression/artifacts") / model_name
            export_dir.mkdir(parents=True, exist_ok=True)
    else:
        export_dir = Path(export_dir)

    # Extract quantization parameters directly using bitwidth_from_token
    quant_scheme = str(config.get("quant_scheme", "tf_enhanced"))
    default_param_bw = bitwidth_from_token(config.get("param_type", "int8"), 8)
    default_output_bw = bitwidth_from_token(config.get("activation_type", "int8"), 8)

    aimet_cfg_file = config.get("config_file", None)

    calib_samples = int(config.get("calib_samples", 256))
    eval_samples = int(config.get("eval_samples", 256))
    metrics_samples = int(config.get("metrics_samples", 64))
    metrics_runs = int(config.get("metrics_runs", 1))
    metrics_warmup = int(config.get("metrics_warmup", 0))
    apply_prepare_model = config.get("apply_prepare_model", False)
    apply_bn_fold = config.get("apply_bn_fold", True)
    apply_experimental_static_aten_calibration = config.get(
        "apply_experimental_static_aten_calibration", False
    )
    use_cuda = torch.cuda.is_available()

    print(f"[AIMET Torch QuantSim] Configuration:")
    print(f"  Scheme: {quant_scheme}")
    print(f"  Precision: W{default_param_bw}/A{default_output_bw}")
    print(f"  CUDA: {'Yes' if use_cuda else 'No'}")

    print(f"[AIMET Torch QuantSim] Extracting PyTorch model...")

    if hasattr(model, "to_torch_model"):
        torch_model = model.to_torch_model()
    else:
        torch_model = model

    torch_model.eval()

    print(f"[AIMET Torch QuantSim] Creating dummy input...")
    device = torch.device("cuda" if use_cuda else "cpu")
    dummy_input = create_dummy_input(input_spec, device)

    print(f"[AIMET Torch QuantSim] Moving model to device...")
    torch_model = torch_model.to(device)

    for buffer in torch_model.buffers():
        buffer.data = buffer.data.to(device)

    dummy_input = dummy_input.to(device)

    print(f"[AIMET Torch QuantSim] Building QuantizationSimModel...")
    sim = build_quantsim_torch(
        model=torch_model,
        dummy_input=dummy_input,
        quant_scheme=quant_scheme,
        default_param_bw=default_param_bw,
        default_output_bw=default_output_bw,
        config_file=aimet_cfg_file,
        use_cuda=use_cuda,
        apply_prepare_model=apply_prepare_model,
        apply_bn_fold=apply_bn_fold,
    )

    # Load dataset once — reused by calibration, eval, and metrics calls
    _dataset = load_torch_dataset(model, dataset_cls)

    print(
        f"[AIMET Torch QuantSim] Calibrating encodings with {calib_samples} samples..."
    )

    calib_loader = get_calibration_dataloader(
        model, calib_dataset_cls or dataset_cls, calib_samples
    )

    def calibration_callback(model_to_calibrate: torch.nn.Module):
        """Forward pass callback for encoding calibration."""
        try:
            model_to_calibrate.eval()
        except NotImplementedError:
            # torch.fx.GraphModules generated from ExportedProgram
            # doesn't support .eval() method yet
            pass

        run_torch_calibration(model_to_calibrate, calib_loader)

    sim.model.eval()
    sim.compute_encodings(calibration_callback)

    print(f"[AIMET Torch QuantSim] Calibration complete")

    feature_acc = eval_pytorch_model(
        sim.model,
        model,
        dataset_cls,
        num_samples=eval_samples,
        dataset=_dataset,
    )

    print(f"[AIMET Torch QuantSim] Quantized accuracy: {feature_acc:.4f}")

    if apply_experimental_static_aten_calibration:
        ep: ExportedProgram = run_static_aten_calibration(
            sim,
            dummy_input,
            default_param_bw,
            default_output_bw,
            calibration_callback,
        )
        static_aten_acc = eval_pytorch_model(
            ep.module(),
            model,
            dataset_cls,
            num_samples=eval_samples,
            dataset=_dataset,
        )
        static_aten_acc = float(static_aten_acc)

        print(
            "[AIMET Torch QuantSim] Fully quantized accuracy "
            f"after static ATen graph calibration: {static_aten_acc:.4f}"
        )
    else:
        static_aten_acc = None

    print(f"[AIMET Torch QuantSim] Measuring runtime and memory...")

    def eval_for_metrics():
        sim.model.eval()
        with torch.no_grad():
            return eval_pytorch_model(
                sim.model,
                model,
                dataset_cls,
                num_samples=metrics_samples,
                dataset=_dataset,
            )

    runtime_str, memory_str = measure_inference_metrics(
        eval_for_metrics,
        runs=metrics_runs,
        warmup=metrics_warmup,
    )

    print(f"[AIMET Torch QuantSim] Runtime: {runtime_str}, Memory: {memory_str}")

    print(f"[AIMET Torch QuantSim] Exporting QDQ ONNX...")
    print(f"[AIMET Torch QuantSim] Moving model and dummy_input to CPU for export...")

    sim.model.cpu()
    dummy_input_cpu = dummy_input.cpu()

    output_names = parse_output_names_from_qnn_options(config.get("qnn_options", ""))
    qdq_path = export_torch_qdq(
        sim=sim,
        dummy_input=dummy_input_cpu,
        export_dir=export_dir,
        model_name=model_name,
        input_spec=input_spec,
        output_names=output_names,
    )

    technique_str = f"quantsim(W{default_param_bw}A{default_output_bw}, {quant_scheme})"
    stats = {
        "techniques": technique_str,
        "runtime": runtime_str,
        "memory": memory_str,
        "static_aten_acc": static_aten_acc,
    }

    return qdq_path, float(feature_acc), stats

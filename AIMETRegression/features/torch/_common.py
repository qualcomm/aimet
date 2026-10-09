# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
Common Utilities for AIMET Torch Feature Runners

This module provides shared functionality used across all AIMET Torch feature runners
(QuantSim, AdaRound, etc.) to ensure consistency and reduce code duplication.

Key Components:
1. Quantization Scheme Mapping - Convert config strings to AIMET enums
2. AIMET QuantSim Construction - Build QuantSim from PyTorch model
3. AIMET Bundle Export - Dual export for validation and AI Hub deployment
4. Cleanup Utilities - Temporary file management

Design Philosophy:
- Device management handled transparently
- QNN compatibility: Export structure matches QNN requirements

Technical Notes:
- Model and inputs must be on CPU for ONNX export
- Dual export: QDQ model for validation, clean bundle for AI Hub
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import torch
from torch.export import ExportedProgram

from AIMETRegression.features.torch.utils import ensure_device_patch

from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.common.defs import QuantScheme
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
import aimet_torch
from torch.utils.data import DataLoader
from qai_hub_models.datasets import BaseDataset
from AIMETRegression.evaluation.calibration import instantiate_calibration_dataset
from qai_hub_models.utils.evaluate.helpers import get_deterministic_sample
from aimet_torch.nn import QuantizationMixin

__all__ = [
    "get_torch_model",
    "build_quantsim_torch",
    "export_torch_qdq",
    "map_quant_scheme",
    "bitwidth_from_token",
    "create_dummy_input",
    "create_calibration_dataloader",
    "run_static_aten_calibration",
]


# ==================== Quantization Scheme Mapping ====================


_QUANT_SCHEME_MAP = {
    "tf": QuantScheme.post_training_tf,
    "tf_enhanced": QuantScheme.post_training_tf_enhanced,
    "percentile": QuantScheme.post_training_percentile,
    "min_max": QuantScheme.post_training_tf,
}


def map_quant_scheme(scheme_str: str) -> QuantScheme:
    """
    Map configuration string to AIMET QuantScheme enum.

    Args:
        scheme_str: Scheme name from config (e.g., "tf_enhanced", "min_max")

    Returns:
        AIMET QuantScheme enum value

    Raises:
        ValueError: If scheme is not recognized
    """
    scheme_lower = scheme_str.lower().strip()

    if scheme_lower not in _QUANT_SCHEME_MAP:
        available = ", ".join(_QUANT_SCHEME_MAP.keys())
        raise ValueError(
            f"Unknown quantization scheme: '{scheme_str}'. Available: {available}"
        )

    return _QUANT_SCHEME_MAP[scheme_lower]


def bitwidth_from_token(token: Optional[Union[str, int]], default: int = 8) -> int:
    """
    Convert various bitwidth representations to integer.

    Args:
        token: Bitwidth specification (None, int, or string like "int8")
        default: Default bitwidth if token is None or unparseable

    Returns:
        Integer bitwidth (typically 4, 8, or 16)
    """
    if token is None:
        return default

    try:
        return int(token)
    except (TypeError, ValueError):
        pass

    token_str = str(token).lower()

    if "16" in token_str:
        return 16
    if "4" in token_str:
        return 4
    if "8" in token_str:
        return 8

    return default


# ==================== Model Extraction ====================


def get_torch_model(model: Any, device: torch.device = None) -> torch.nn.Module:
    """
    Get the PyTorch model from a QAI Hub model wrapper.

    Args:
        model: QAI Hub model instance or nn.Module
        device: Target device (default: CUDA if available, else CPU)

    Returns:
        PyTorch nn.Module ready for AIMET
    """
    ensure_device_patch()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(model, "to_torch_model"):
        torch_model = model.to_torch_model()
    elif isinstance(model, torch.nn.Module):
        torch_model = model
    else:
        raise TypeError(
            f"Cannot extract torch model from {type(model).__name__}. "
            f"Expected QAI Hub model or nn.Module."
        )

    torch_model = torch_model.to(device).eval()

    return torch_model


def _register_ignored_modules() -> None:
    """Register modules to be ignored by AIMET quantization."""
    try:
        from ultralytics.nn.modules.conv import Concat

        QuantizationMixin.ignore(Concat)
        print(f"[QuantSim Torch] Ignoring ultralytics.Concat for quantization")
    except ImportError:
        pass

    try:
        from torchvision.ops.stochastic_depth import StochasticDepth

        QuantizationMixin.ignore(StochasticDepth)
        print(f"[QuantSim Torch] Ignoring torchvision StochasticDepth for quantization")
    except ImportError:
        pass

    try:
        from torchvision.models.convnext import LayerNorm2d

        QuantizationMixin.ignore(LayerNorm2d)
        print(f"[QuantSim Torch] Ignoring torchvision LayerNorm2d for quantization")
    except ImportError:
        pass

    try:
        from torchvision.ops.misc import Permute

        QuantizationMixin.ignore(Permute)
        print(f"[QuantSim Torch] Ignoring torchvision Permute for quantization")
    except ImportError:
        pass

    try:
        from transformers.models.levit.modeling_levit import LevitSubsample

        QuantizationMixin.ignore(LevitSubsample)
        print(f"[QuantSim Torch] Ignoring LevitSubsample for quantization")
    except ImportError:
        pass

    # BEiT: non-computational modules (no learnable conv/linear weights)
    try:
        from transformers.models.beit.modeling_beit import (
            BeitRelativePositionBias,
            BeitDropPath,
        )
        from transformers.activations import GELUActivation

        QuantizationMixin.ignore(BeitRelativePositionBias)
        QuantizationMixin.ignore(BeitDropPath)
        QuantizationMixin.ignore(GELUActivation)
        print(f"[QuantSim Torch] Ignoring BEiT non-quantizable modules")
    except ImportError:
        pass

    # NASNet: same-padding pooling variants (no learnable weights)
    try:
        from timm.layers.pool2d_same import MaxPool2dSame, AvgPool2dSame

        QuantizationMixin.ignore(MaxPool2dSame)
        QuantizationMixin.ignore(AvgPool2dSame)
        print(f"[QuantSim Torch] Ignoring timm same-padding pooling modules")
    except ImportError:
        pass

    # NASNet: Conv2dSame is Conv2d with same-padding — has real conv weights that must be quantized
    try:
        from timm.layers.conv2d_same import Conv2dSame

        if Conv2dSame not in QuantizationMixin.cls_to_qcls:

            @QuantizationMixin.implements(Conv2dSame)
            class QuantizedConv2dSame(QuantizationMixin, Conv2dSame):
                def forward(self, x):  # pylint: disable=arguments-differ
                    if self.input_quantizers[0]:
                        x = self.input_quantizers[0](x)
                    with self._patch_quantized_parameters():
                        out = super().forward(x)
                    if self.output_quantizers[0]:
                        out = self.output_quantizers[0](out)
                    return out

            print(f"[QuantSim Torch] Registered quantized Conv2dSame")
    except ImportError:
        pass

    # Sequencer2D: FastAdaptiveAvgPool (no learnable weights)
    try:
        from timm.layers.adaptive_avgmax_pool import FastAdaptiveAvgPool

        QuantizationMixin.ignore(FastAdaptiveAvgPool)
        print(f"[QuantSim Torch] Ignoring timm FastAdaptiveAvgPool")
    except ImportError:
        pass

    # FFNet: UpsampleCat (upsample + concat, no learnable weights)
    # 'models.ffnet_blocks' is a vendored namespace loaded by qai_hub_models
    try:
        from models.ffnet_blocks import UpsampleCat

        QuantizationMixin.ignore(UpsampleCat)
        print(f"[QuantSim Torch] Ignoring FFNet UpsampleCat")
    except (ImportError, ModuleNotFoundError):
        pass

    # FFNet (qai-hub-models 0.54+): UpsampleCat under qai_hub_models namespace.
    # Catch broadly: importing a qai_hub_models model can raise non-import errors
    # (e.g. pydantic ValidationError on a bad release) that must not crash the suite (#7516).
    try:
        from qai_hub_models.models._shared.ffnet.external_repos.ffnet.models.ffnet_blocks import (
            UpsampleCat,
        )

        QuantizationMixin.ignore(UpsampleCat)
        print(f"[QuantSim Torch] Ignoring qai_hub_models FFNet UpsampleCat")
    except Exception as error:  # pylint: disable=broad-except
        print(
            f"[QuantSim Torch] Skipping qai_hub_models FFNet UpsampleCat registration: {error}"
        )

    # MiDaS: Interpolate (upsample-only, no learnable weights).
    # Catch broadly for the same reason as the FFNet import above (#7516).
    try:
        from qai_hub_models.models.midas.external_repos.midas.midas.blocks import (
            Interpolate,
        )

        QuantizationMixin.ignore(Interpolate)
        print(f"[QuantSim Torch] Ignoring MiDaS Interpolate")
    except Exception as error:  # pylint: disable=broad-except
        print(f"[QuantSim Torch] Skipping MiDaS Interpolate registration: {error}")

    # YOLOv7: Concat from vendored yolov7 source (no learnable weights)
    try:
        from models.common import Concat as YoloV7Concat

        QuantizationMixin.ignore(YoloV7Concat)
        print(f"[QuantSim Torch] Ignoring YOLOv7 Concat")
    except (ImportError, ModuleNotFoundError):
        pass


# ==================== AIMET QuantSim Construction ====================


def build_quantsim_torch(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    *,
    quant_scheme: str = "tf_enhanced",
    default_param_bw: int = 8,
    default_output_bw: int = 8,
    config_file: Optional[str] = None,
    apply_prepare_model: bool = False,
    apply_bn_fold: bool = True,
    use_cuda: bool = True,
) -> QuantizationSimModel:
    """
    Build an AIMET Torch QuantizationSimModel.

    Args:
        model: PyTorch model (nn.Module) in eval mode
        dummy_input: Sample input tensor on same device as model
        quant_scheme: Quantization scheme name
        default_param_bw: Parameter/weight bitwidth
        default_output_bw: Output/activation bitwidth (note: Torch uses output_bw)
        config_file: Optional AIMET config file path or built-in name (e.g., "htp_v79")
        apply_prepare_model: Whether to apply prepare_model() for AIMET compatibility

    Returns:
        Tuple of QuantizationSimModel
    """
    _register_ignored_modules()

    device = next(model.parameters()).device

    if dummy_input.device != device:
        dummy_input = dummy_input.to(device)

    if apply_prepare_model:
        print(f"[QuantSim Torch] Applying prepare_model() for AIMET compatibility...")
        model = prepare_model(model)
        model = model.to(device).eval()

    model.eval()

    # Fold BatchNorm layers into preceding Conv/Linear layers before quantization
    # This is critical for QNN compatibility - QNN cannot handle unfused BN nodes
    if apply_bn_fold:
        print(f"[QuantSim Torch] Folding BatchNorm layers...")
        fold_all_batch_norms(model, input_shapes=(tuple(dummy_input.shape),))
    else:
        print(f"[QuantSim Torch] Skipping BatchNorm folding (apply_bn_fold=False)")

    scheme_enum = map_quant_scheme(quant_scheme)

    print(
        f"[QuantSim Torch] Building with scheme={quant_scheme}, W{default_param_bw}A{default_output_bw}"
    )
    print(f"[QuantSim Torch] Device: {device}")

    sim = QuantizationSimModel(
        model=model,
        dummy_input=dummy_input,
        quant_scheme=scheme_enum,
        default_param_bw=default_param_bw,
        default_output_bw=default_output_bw,
        config_file=str(config_file) if config_file is not None else "default",
    )

    return sim


# ==================== AIMET Bundle Export ====================


def parse_output_names_from_qnn_options(qnn_options: str) -> Optional[List[str]]:
    """Extract --output_names values from a qnn_options string."""
    if not qnn_options:
        return None
    parts = qnn_options.split()
    for i, part in enumerate(parts):
        if part == "--output_names" and i + 1 < len(parts):
            return parts[i + 1].split(",")
    return None


def export_torch_qdq(
    sim: QuantizationSimModel,
    dummy_input: torch.Tensor,
    export_dir: Path,
    model_name: str,
    input_spec: Dict[str, Any],
    output_names: Optional[List[str]] = None,
) -> Path:
    """
    Export AIMET Torch QuantSim as QDQ ONNX.

    Creates a single QDQ ONNX model with QuantizeLinear/DequantizeLinear ops
    for both local validation and QNN compilation.

    Args:
        sim: QuantizationSimModel after compute_encodings()
        dummy_input: Dummy input (must be on CPU per AIMET requirements)
        export_dir: Parent directory for exports
        model_name: Model name for file naming
        input_spec: Input specification dict to get expected input name
        output_names: Output tensor names for QNN compatibility

    Returns:
        Path to exported QDQ ONNX model

    Important:
        dummy_input MUST be on CPU for export, regardless of where
        the model is located. AIMET documentation explicitly requires this.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    dummy_input_cpu = dummy_input.cpu() if dummy_input.is_cuda else dummy_input

    qdq_path = export_dir / f"{model_name}_qdq.onnx"

    # Get expected input name from input_spec for QNN compatibility
    # (e.g., "image_tensor" for --force_channel_last_input to match)
    input_names = list(input_spec.keys()) if input_spec else None
    if input_names:
        print(f"[AIMET Torch] Exporting QDQ with input_names={input_names}")
    if output_names:
        print(f"[AIMET Torch] Exporting QDQ with output_names={output_names}")

    aimet_torch.onnx.export(
        sim.model,
        (dummy_input_cpu,),
        str(qdq_path),
        dynamo=False,
        opset_version=21,  # For INT4/INT16 support
        input_names=input_names,
        output_names=output_names,
    )

    if not qdq_path.exists():
        raise RuntimeError(f"QDQ export failed: {qdq_path}")

    print(f"[AIMET Torch] QDQ model saved: {qdq_path}")

    return qdq_path


# ==================== Dummy Input Creation ====================


def create_dummy_input(
    input_spec: Dict[str, Any], device: torch.device
) -> torch.Tensor:
    """
    Create a dummy input tensor from input specification.

    Args:
        input_spec: Input specification from model.get_input_spec()
        device: Target device for the tensor

    Returns:
        Dummy input tensor on specified device
    """
    from qai_hub_models.utils.input_spec import make_torch_inputs

    sample_inputs = make_torch_inputs(input_spec)

    if isinstance(sample_inputs, dict):
        dummy_input = next(iter(sample_inputs.values()))
    elif isinstance(sample_inputs, (list, tuple)):
        dummy_input = sample_inputs[0] if sample_inputs else sample_inputs
    else:
        dummy_input = sample_inputs

    if dummy_input.dim() == 3:
        dummy_input = dummy_input.unsqueeze(0)

    return dummy_input.to(device)


def create_calibration_dataloader(
    qai_hub_model: Any,
    dataset_cls: type[BaseDataset],
    num_samples: int,
    batch_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for AIMET calibration/optimization.

    Args:
            qai_hub_model: QAI Hub model object
            dataset_cls: Dataset class to sample from (e.g., ImagenetDataset)
            num_samples: Number of samples to use
            batch_size: Batch size for the dataloader

    Returns:
            DataLoader yielding (input_tensor, label) tuples
    """
    dataset = instantiate_calibration_dataset(qai_hub_model, dataset_cls)
    num_samples = min(num_samples, len(dataset))

    sampler = get_deterministic_sample(
        dataset, num_samples=num_samples, samples_per_job=1
    )

    inputs_list = []
    labels_list = []

    for sample in sampler:
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            inputs, label = sample[0], sample[1]
        elif isinstance(sample, (list, tuple)) and len(sample) == 1:
            inputs, label = sample[0], 0
        else:
            inputs, label = sample, 0

        if isinstance(inputs, dict):
            inputs = next(iter(inputs.values()))
        elif isinstance(inputs, (list, tuple)):
            inputs = inputs[0]

        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
            inputs_list.append(inputs)

        if isinstance(label, torch.Tensor):
            labels_list.append(label.item() if label.numel() == 1 else 0)
        else:
            labels_list.append(int(label) if label is not None else 0)

    if not inputs_list:
        raise ValueError(
            f"No samples collected from dataset {dataset_cls.dataset_name()}"
        )

    all_inputs = torch.cat(inputs_list, dim=0)
    all_labels = torch.tensor(labels_list, dtype=torch.long)

    tensor_dataset = torch.utils.data.TensorDataset(all_inputs, all_labels)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)


def run_static_aten_calibration(
    sim: QuantizationSimModel,
    dummy_input: torch.Tensor,
    param_bw: int,
    activation_bw: int,
    calibration_callback: Callable[[torch.nn.Module], Any],
) -> ExportedProgram:
    from aimet_torch.experimental.export.exported_program import (
        ExportedProgram as AimetExportedProgram,
    )

    batch_size = dummy_input.shape[0]

    if batch_size <= 1:
        # Ensure batch_size > 1.
        # When batch_size <= 1, torch.export specializes the graph
        # specifically for batch_size=1 or 0, which causes issues
        # when we later run calibration with batch_size > 1.
        new_shape = (2, *dummy_input.shape[1:])
        dummy_input = torch.randn(
            new_shape,
            dtype=dummy_input.dtype,
            device=dummy_input.device,
        )

    ep = aimet_torch.experimental.export.export(
        sim.model.eval(),
        args=(dummy_input,),
        # CUDA batch normalization kernels only support batch size up to 65535
        dynamic_shapes=[
            {0: torch.export.Dim("batch_size", max=65535)},
        ],
    )
    ep = AimetExportedProgram.from_torch_exported_program(ep)

    with ep.compute_missing_encodings(param_bw=param_bw, activation_bw=activation_bw):
        calibration_callback(ep.module())

    return ep

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LoRA quantization utilities for integrating with QuantizationSimModel.

Standalone functions that interact with QuantSim's qc_quantize_op_dict to
configure, freeze, and unfreeze quantizers for multi-adapter calibration
workflows. All functions take ``lora_tensors``, the
:class:`~aimet_onnx.peft.peft_utils.LoraTensorNames` that
:func:`~aimet_onnx.peft.peft_utils.get_lora_tensor_names` returns for the model
``load_lora_adapters`` returned.
"""

import onnx
from aimet_onnx.common.defs import qtype
from aimet_onnx.common.utils import AimetLogger
from aimet_onnx.peft.peft_utils import LoraTensorNames

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


# ---------------------------------------------------------------------------
# Quantizer configuration
# ---------------------------------------------------------------------------


def set_lora_bitwidth(
    sim,
    lora_tensors: LoraTensorNames,
    param_type: str | qtype,
    activation_type: str | qtype,
) -> int:
    """Set bitwidth for LoRA parameter and activation quantizers.

    :param sim: QuantizationSimModel instance
    :param lora_tensors: LoRA tensor names from ``get_lora_tensor_names``
    :param param_type: Quantization type for LoRA weights (e.g., ``int16``).
    :param activation_type: Quantization type for LoRA activations (e.g., ``int8``).
    :return: Number of quantizers updated
    :raises RuntimeError: If any LoRA quantizer is already frozen
    """
    if isinstance(param_type, str):
        param_type = qtype.from_string(param_type)
    if isinstance(activation_type, str):
        activation_type = qtype.from_string(activation_type)
    _, param_bw = param_type.to_legacy_repr()
    _, activation_bw = activation_type.to_legacy_repr()

    count = 0
    for name, bw in _iter_lora_quantizers(sim, lora_tensors, param_bw, activation_bw):
        qtzr = sim.qc_quantize_op_dict[name]
        if qtzr.is_encoding_frozen():
            raise RuntimeError(
                f"LoRA quantizer '{name}' is frozen. "
                f"Call set_lora_bitwidth() BEFORE freeze functions."
            )
        qtzr.set_bitwidth(bw)
        count += 1

    if count == 0:
        all_names = lora_tensors.quantizer_names
        raise ValueError(
            f"No LoRA quantizers found in sim for {len(all_names)} LoRA names. "
            f"First few names: {all_names[:3]}"
        )

    # Disable scale quantizers — scales are scalar constants (alpha/rank),
    # not statistical distributions.  QuantSim wraps them as activation
    # quantizers because Mul is not in OPS_WITH_PARAMS, but quantizing a
    # single constant adds no value and breaks the freeze/unfreeze lifecycle.
    scale_count = 0
    for name in lora_tensors.scales:
        if name in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[name].enabled = False
            scale_count += 1

    _logger.info(
        "Set LoRA bitwidth: %d quantizers (params=%d-bit, activations=%d-bit), "
        "disabled %d scale quantizers",
        count,
        param_bw,
        activation_bw,
        scale_count,
    )
    return count


def _iter_lora_quantizers(sim, lora_tensors, param_bw, activation_bw):
    """Yield (name, bitwidth) for each LoRA quantizer found in sim."""
    for name in lora_tensors.params:
        if name in sim.qc_quantize_op_dict:
            yield name, param_bw
    for name in lora_tensors.activations:
        if name in sim.qc_quantize_op_dict:
            yield name, activation_bw


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------


def freeze_base_param_quantizers(sim, lora_tensors: LoraTensorNames) -> int:
    """Freeze all base (non-LoRA) parameter quantizers.

    :param sim: QuantizationSimModel instance
    :param lora_tensors: LoRA tensor names from ``get_lora_tensor_names``
    :return: Number of quantizers frozen
    """
    lora_set = set(lora_tensors.params)
    count = 0
    for name in sim.param_names:
        if name not in lora_set and name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            if qtzr.enabled:
                qtzr.freeze_encodings()
                count += 1

    _logger.info("Froze %d base param quantizers", count)
    return count


def freeze_base_activation_quantizers(sim, lora_tensors: LoraTensorNames) -> int:
    """Freeze all base (non-LoRA) activation quantizers.

    :param sim: QuantizationSimModel instance
    :param lora_tensors: LoRA tensor names. The scales count as LoRA tensors too: QuantSim
        gives them activation quantizers. So do the A and B weights of a
        ``load_lora_adapters_as_inputs`` model, which are graph inputs.
    :return: Number of quantizers frozen
    """
    lora_set = set(lora_tensors.activations) | set(lora_tensors.scales)
    lora_set |= set(lora_tensors.params)
    count = 0
    for name in sim.activation_names:
        if name not in lora_set and name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            if qtzr.enabled:
                qtzr.freeze_encodings()
                count += 1

    _logger.info("Froze %d base activation quantizers", count)
    return count


def freeze_base_model(sim, lora_tensors: LoraTensorNames) -> int:
    """Freeze all base quantizers (params + activations).

    :param sim: QuantizationSimModel instance
    :param lora_tensors: LoRA tensor names from ``get_lora_tensor_names``
    :return: Total number of quantizers frozen
    """
    count = freeze_base_param_quantizers(sim, lora_tensors)
    count += freeze_base_activation_quantizers(sim, lora_tensors)
    _logger.info("Froze %d total base quantizers", count)
    return count


def unfreeze_lora_quantizers(sim, lora_tensors: LoraTensorNames) -> int:
    """Unfreeze all LoRA quantizers (params + activations) and reset stats.

    :param sim: QuantizationSimModel instance
    :param lora_tensors: LoRA tensor names from ``get_lora_tensor_names``
    :return: Number of quantizers unfrozen
    """
    all_lora = lora_tensors.quantizer_names
    count = 0
    for name in all_lora:
        if name in sim.qc_quantize_op_dict:
            qtzr = sim.qc_quantize_op_dict[name]
            qtzr._is_encoding_frozen = False  # pylint: disable=protected-access
            qtzr.reset_encoding_stats()
            count += 1

    if count == 0:
        raise ValueError(
            f"No LoRA quantizers found in sim for {len(all_lora)} LoRA names. "
            f"First few names: {all_lora[:3]}"
        )

    _logger.info("Unfroze %d LoRA quantizers", count)
    return count


# ---------------------------------------------------------------------------
# Encoding snapshot / restore
# ---------------------------------------------------------------------------


def get_lora_encodings(
    sim,
    lora_tensors: LoraTensorNames,
    encoding_version: str = "1.0.0",
) -> dict[str, dict]:
    """Capture current LoRA quantizer encodings (params + activations).

    :param sim: QuantizationSimModel instance
    :param lora_tensors: LoRA tensor names from ``get_lora_tensor_names``
    :param encoding_version: Encoding format version (default ``"1.0.0"``)
    :return: Dict mapping LoRA quantizer name to encoding dict
    :raises RuntimeError: If any LoRA quantizer has no encoding (uncalibrated)
    """
    all_lora = lora_tensors.quantizer_names
    encodings = {}
    for name in all_lora:
        if name not in sim.qc_quantize_op_dict:
            continue
        qtzr = sim.qc_quantize_op_dict[name]
        encoding = qtzr.export_encodings(encoding_version)
        if encoding is None:
            raise RuntimeError(
                f"LoRA quantizer '{name}' has no encoding. "
                f"Call sim.compute_encodings() before get_lora_encodings()."
            )
        encodings[name] = encoding

    if not encodings:
        raise ValueError(
            f"No LoRA quantizers found in sim for {len(all_lora)} LoRA names. "
            f"First few names: {all_lora[:3]}"
        )

    _logger.info("Captured encodings for %d LoRA quantizers", len(encodings))
    return encodings


def set_lora_encodings(sim, encodings: dict[str, dict]) -> int:
    """Restore previously captured LoRA quantizer encodings and freeze them.

    :param sim: QuantizationSimModel instance
    :param encodings: Dict from ``get_lora_encodings()``
    :return: Number of quantizers updated
    :raises ValueError: If any encoding name is not found in the sim
    """
    missing = [name for name in encodings if name not in sim.qc_quantize_op_dict]
    if missing:
        raise ValueError(
            f"{len(missing)} encoding names not found in sim: {missing[:5]}"
        )

    count = 0
    for name, encoding_dict in encodings.items():
        qtzr = sim.qc_quantize_op_dict[name]
        qtzr._is_encoding_frozen = False  # pylint: disable=protected-access
        qtzr._load_encodings_dict(encoding_dict)  # pylint: disable=protected-access
        qtzr.freeze_encodings()
        count += 1

    _logger.info("Restored and froze encodings for %d LoRA quantizers", count)
    return count


# ---------------------------------------------------------------------------
# Recipe exclusion helpers
# ---------------------------------------------------------------------------


def get_lora_node_names_for_exclusion(
    model: "onnx.ModelProto",
    lora_tensors: LoraTensorNames,
) -> list[str]:
    """Return ONNX node names in LoRA branches for recipe exclusion.

    Traces downstream from LoRA weight initializers and scale constants,
    collecting all nodes that are exclusively part of LoRA computation.
    Stops at Add/Sum merge points where the LoRA branch rejoins the base
    path (one input from LoRA, one from base) — those are NOT excluded
    since they carry the base computation forward.

    The returned names match ConnectedGraph op names (which are ONNX node
    names), suitable for passing as ``nodes_to_exclude`` to SeqMSE and
    AdaScale.

    :param model: ONNX ModelProto (raw or from QuantSim ``sim.model.model``).
    :param lora_tensors: LoRA tensor names from ``get_lora_tensor_names``
    :return: List of ONNX node names in LoRA branches.
    """
    lora_param_set = set(lora_tensors.params)
    scale_set = set(lora_tensors.scales)
    # Non-LoRA initializers are treated as constants (not traced through).
    # LoRA params and scales are excluded from the const set so the trace
    # propagates through them and their QcQuantizeOp wrappers.
    all_init_names = {init.name for init in model.graph.initializer}
    const_names = all_init_names - lora_param_set - scale_set
    branch_tensors = lora_param_set | scale_set

    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    lora_node_names: set[str] = set()
    frontier = list(branch_tensors)
    visited: set[str] = set()

    while frontier:
        next_frontier = []
        for tensor_name in frontier:
            for node in consumers.get(tensor_name, []):
                if node.name in visited:
                    continue
                visited.add(node.name)

                non_const_inputs = [
                    inp for inp in node.input if inp and inp not in const_names
                ]
                any_lora = any(inp in branch_tensors for inp in non_const_inputs)
                if not any_lora:
                    continue

                all_lora = all(inp in branch_tensors for inp in non_const_inputs)

                # Stop at merge points: Add/Sum where base and LoRA paths meet
                if node.op_type in ("Add", "Sum") and not all_lora:
                    continue

                lora_node_names.add(node.name)
                for out in node.output:
                    if out:
                        branch_tensors.add(out)
                        next_frontier.append(out)
        frontier = next_frontier

    _logger.info(
        "Found %d LoRA branch node names for recipe exclusion",
        len(lora_node_names),
    )
    return sorted(lora_node_names)


# ---------------------------------------------------------------------------
# Phase lifecycle helpers
# ---------------------------------------------------------------------------


def disable_lora_quantizers(sim, lora_tensors: LoraTensorNames) -> int:
    """Disable all LoRA quantizers, e.g. while a recipe runs on the LoRA sim.

    During recipe application (AdaScale, SeqMSE, LPBQ), LoRA quantizers
    should be disabled so they don't interfere with base-model optimization.

    :param sim: QuantizationSimModel instance
    :param lora_tensors: LoRA tensor names from ``get_lora_tensor_names``
    :return: Number of quantizers disabled
    """
    all_lora = lora_tensors.quantizer_names
    count = 0
    for name in all_lora:
        if name in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[name].enabled = False
            count += 1

    # Scales are already disabled by set_lora_bitwidth, but ensure it
    for name in lora_tensors.scales:
        if name in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[name].enabled = False

    _logger.info("Disabled %d LoRA quantizers for recipe compatibility", count)
    return count


def enable_lora_calibration(
    sim,
    lora_tensors: LoraTensorNames,
    param_type: str | qtype,
    activation_type: str | qtype,
) -> int:
    """Enable LoRA quantizers and set bitwidth for calibration.

    Enables all LoRA quantizers and sets their bitwidth. Undoes ``disable_lora_quantizers``.

    :param sim: QuantizationSimModel instance.
    :param lora_tensors: LoRA tensor names from ``get_lora_tensor_names``
    :param param_type: Quantization type for LoRA weights (e.g., ``"int16"``).
    :param activation_type: Quantization type for LoRA activations (e.g., ``"int8"``).
    :return: Number of quantizers enabled.
    """
    all_lora = lora_tensors.quantizer_names
    count = 0
    for name in all_lora:
        if name in sim.qc_quantize_op_dict:
            sim.qc_quantize_op_dict[name].enabled = True
            count += 1

    set_lora_bitwidth(sim, lora_tensors, param_type, activation_type)

    _logger.info("Enabled %d LoRA quantizers for calibration", count)
    return count

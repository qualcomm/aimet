# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ExportableMoE adaptation: predicated experts.

Replaces the transformers 5.x ``<Model>Experts`` module (fused 3D parameters
consumed by a data-dependent loop over ``nonzero()``) with per-expert
``nn.Linear`` leaves, executed under one of three output-equivalent policies:

``sparse``
    The stock gather path. Not exportable, fastest; the default, for eval.
``dense``
    Every expert on every token, weighted by a routing weight of exactly 0 where
    unrouted. Exportable, no dynamism.
``predicated``
    One ``torch.cond`` per expert, lowering to an ONNX ``If`` that ORT skips.

Unfusing is what makes per-expert encodings addressable: an indexed slice of an
``nn.Parameter`` is not a leaf module, so aimet-torch attaches no weight
quantizer, and ONNX lowers it to ``Gather`` -> ``MatMul``, making the weight
arrive as an activation.

``force_all`` is a buffer rather than a Python flag so it survives export as an
initializer and can be flipped post-export (:func:`set_onnx_force_all`): one
graph and one set of encodings serve both calibration and eval.

The graph passes that deploy these experts efficiently on target are specified
at the end of ``expert_subselection.py``.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Iterator

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.transforms.base import rgetattr, rsetattr

#: ``sparse`` is eval-only (not exportable); the others are the exportable realizers.
EXECUTION_MODES = ("sparse", "dense", "predicated")

#: What each expert observes at its input; equivalently, ``force_all`` 0 or 1.
SELECTION_MODES = ("routed", "all")


class ExpertMLP(nn.Module):
    """One expert as ordinary ``nn.Linear`` leaves.

    Named to match ``Qwen3MoeMLP`` so name-based role classification
    (LlmTopology's ``gate_up`` group, AdaScale boundaries) still works.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn, dtype, device):
        super().__init__()
        kw = {"bias": False, "dtype": dtype, "device": device}
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, **kw)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, **kw)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, **kw)
        self.act_fn = act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


def is_fused_experts(module: nn.Module) -> bool:
    """A module holding 3D ``gate_up_proj``/``down_proj``, whatever its family."""
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    return (
        isinstance(gate_up, torch.Tensor)
        and isinstance(down, torch.Tensor)
        and gate_up.dim() == 3
        and down.dim() == 3
    )


def _unfuse(fused: nn.Module) -> tuple[nn.ModuleList, int, int]:
    """Split fused 3D expert parameters into per-expert :class:`ExpertMLP` leaves.

    Qwen layout only: ``gate_up_proj`` is ``[E, 2I, H]`` (rows ``:I`` gate,
    ``I:`` up), ``down_proj`` is ``[E, H, I]``. gpt_oss's transposed, interleaved,
    biased layout is rejected rather than silently mis-split.
    """
    gate_up, down = fused.gate_up_proj, fused.down_proj
    num_experts, two_i, hidden_dim = gate_up.shape
    inter = down.shape[-1]

    if two_i != 2 * inter or down.shape[1] != hidden_dim:
        raise NotImplementedError(
            f"Unsupported fused-expert layout on {type(fused).__name__}: "
            f"gate_up_proj={tuple(gate_up.shape)}, down_proj={tuple(down.shape)}. "
            f"Expected the Qwen layout gate_up_proj=[E, 2I, H], down_proj=[E, H, I]. "
            f"A transposed layout (e.g. gpt_oss's [E, H, 2I] with interleaved "
            f"gate/up and biases) needs its own ExpertMLP body."
        )
    for extra in ("gate_up_proj_bias", "down_proj_bias"):
        if getattr(fused, extra, None) is not None:
            raise NotImplementedError(
                f"{type(fused).__name__} has {extra}; biased experts are not "
                f"supported yet (ExpertMLP is built bias-free)."
            )

    act_fn = getattr(fused, "act_fn", None)
    if act_fn is None:
        raise NotImplementedError(
            f"{type(fused).__name__} has no act_fn; cannot build per-expert MLPs."
        )

    experts = nn.ModuleList()
    for e in range(num_experts):
        expert = ExpertMLP(
            hidden_dim, inter, act_fn, dtype=gate_up.dtype, device=gate_up.device
        )
        with torch.no_grad():
            # Clone, not view: SpinQuant/AdaScale rewrite weights in place.
            expert.gate_proj.weight.copy_(gate_up[e, :inter])
            expert.up_proj.weight.copy_(gate_up[e, inter:])
            expert.down_proj.weight.copy_(down[e])
        experts.append(expert)
    return experts, hidden_dim, inter


class PredicatedExperts(nn.Module):
    """Drop-in replacement for a fused ``<Model>Experts`` module.

    Keeps the stock ``(hidden_states, top_k_index, top_k_weights)`` signature, so
    the router, shared expert and enclosing block are untouched.
    """

    def __init__(
        self,
        fused: nn.Module,
        *,
        execution: str = "sparse",
        selection: str = "routed",
        calibration_execution: str = "dense",
        export_execution: str = "predicated",
    ):
        super().__init__()
        if execution not in EXECUTION_MODES:
            raise ValueError(f"execution must be one of {EXECUTION_MODES}")
        if selection not in SELECTION_MODES:
            raise ValueError(f"selection must be one of {SELECTION_MODES}")
        if calibration_execution not in ("dense", "predicated"):
            raise ValueError("calibration_execution must be 'dense' or 'predicated'")
        if export_execution not in ("dense", "predicated"):
            raise ValueError("export_execution must be 'dense' or 'predicated'")

        self.experts, self.hidden_dim, self.intermediate_dim = _unfuse(fused)
        self.num_experts = len(self.experts)
        self.execution = execution
        #: Policies used by :func:`forced_expert_activation`; recorded here so the
        #: recipe/export hooks need no knowledge of MoE internals.
        self.calibration_execution = calibration_execution
        #: Realizer baked into the exported graph. ``predicated`` is the
        #: faithful representation; ``dense`` is a stopgap for getting numbers
        #: while aimet-onnx cannot quantize inside If bodies.
        self.export_execution = export_execution
        self.selection = selection
        # Shape [1], not 0-d: ONNX initializer deduplication merges a 0-d
        # float32 0.0 with any comparison's 0.0 constant, silently rewiring the
        # predicate.
        # On the experts' device: this replaces a module inside an
        # already-placed model, so nothing calls .to() on it afterwards.
        self.register_buffer(
            "force_all",
            torch.tensor(
                [1.0 if selection == "all" else 0.0],
                dtype=torch.float32,
                device=self.experts[0].gate_proj.weight.device,
            ),
            persistent=False,
        )

    # -- selection -----------------------------------------------------------

    def _selection_weights(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter top-k weights into a dense ``[T, E]``, 0 off-selection."""
        num_tokens = hidden_states.shape[0]
        index = top_k_index
        weights = top_k_weights.to(hidden_states.dtype)

        # qwen4_exp uses index E as a dropped-token sentinel: send it to column
        # 0 with weight 0 so scatter stays in bounds and it stays inert.
        if index.dtype != torch.int64:
            index = index.to(torch.int64)
        in_range = index < self.num_experts
        index = torch.where(in_range, index, torch.zeros_like(index))
        weights = weights * in_range.to(weights.dtype)

        dense = hidden_states.new_zeros((num_tokens, self.num_experts))
        return dense.scatter(-1, index, weights)

    def _input_mask(self, weight_e: torch.Tensor) -> torch.Tensor:
        """``[T, 1]`` multiplier on an expert's input: routed rows, or all rows.

        ``sign(abs(w))`` rather than ``w > 0`` because the latter needs a literal
        0 constant, which ONNX deduplication merges with ``force_all``, turning a
        later ``force_all = 1`` patch into ``w > 1``: every expert skipped,
        silently.
        """
        routed = weight_e.abs().sign()
        return torch.maximum(routed, self.force_all.to(weight_e.dtype))

    def _is_active(self, in_mask: torch.Tensor) -> torch.Tensor:
        """Scalar bool: does this expert run?

        Derived from the mask so the two cannot disagree. ``amax`` needs an
        explicit ``dim`` in the ONNX lowering; flatten keeps the result 0-d, as
        ``torch.cond`` requires of a predicate.
        """
        return in_mask.flatten().amax(dim=0).to(torch.bool)

    # -- realizers -----------------------------------------------------------

    def _forward_sparse(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Stock gather path. Not exportable; fastest; eval default."""
        out = torch.zeros_like(hidden_states)
        expert_mask = nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        ).permute(2, 1, 0)
        hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            e = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current = hidden_states[token_idx]
            weighted = self.experts[e](current) * top_k_weights[
                token_idx, top_k_pos, None
            ].to(hidden_states.dtype)
            out.index_add_(0, token_idx, weighted.to(out.dtype))
        return out

    def _forward_dense(
        self, hidden_states: torch.Tensor, selected: torch.Tensor
    ) -> torch.Tensor:
        """Realizer (C): every expert on every token, weighted by ``selected``."""
        out = torch.zeros_like(hidden_states)
        for e, expert in enumerate(self.experts):
            weight_e = selected[:, e : e + 1]
            out = out + expert(hidden_states * self._input_mask(weight_e)) * weight_e
        return out

    def _is_quantized(self) -> bool:
        """True when the expert leaves have been replaced by quantized modules."""
        return hasattr(self.experts[0].gate_proj, "param_quantizers")

    def _forward_predicated(
        self, hidden_states: torch.Tensor, selected: torch.Tensor
    ) -> torch.Tensor:
        """Realizer (A): one ``torch.cond`` per expert, accumulator threaded."""
        if self._is_quantized():
            # torch.cond needs dynamo capture; a quantized forward enters a
            # contextlib.ExitStack that dynamo cannot trace.
            raise RuntimeError(
                "The predicated realizer cannot run inside an aimet-torch "
                "QuantizationSimModel: torch.cond needs to capture its branches "
                "with dynamo, and a quantized module's forward uses "
                "contextlib.ExitStack, which dynamo cannot trace.\n"
                "Use execution='dense' for torch quantsim work (the default for "
                "calibration) -- it is output-equivalent and gives the same "
                "per-expert encodings. The predicated realizer is for ONNX "
                "export, which happens from the float model."
            )

        out = torch.zeros_like(hidden_states)
        for e, expert in enumerate(self.experts):
            weight_e = selected[:, e : e + 1]
            in_mask = self._input_mask(weight_e)
            active = self._is_active(in_mask)

            def taken(acc, hs, mask, w, _expert=expert):
                return acc + _expert(hs * mask) * w

            def skipped(acc, hs, mask, w):
                # Not `return acc`: torch.cond rejects input-to-output aliasing.
                # Lowers to ONNX Identity.
                return acc.clone()

            out = torch.cond(
                active, taken, skipped, (out, hidden_states, in_mask, weight_e)
            )
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.execution == "sparse":
            if float(self.force_all) > 0:
                raise RuntimeError(
                    "execution='sparse' cannot honour force_all: the gather path "
                    "visits only routed experts by construction. Use 'dense' or "
                    "'predicated' when forcing all experts active."
                )
            return self._forward_sparse(hidden_states, top_k_index, top_k_weights)

        selected = self._selection_weights(hidden_states, top_k_index, top_k_weights)
        if self.execution == "dense":
            return self._forward_dense(hidden_states, selected)
        return self._forward_predicated(hidden_states, selected)


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------


def replace_fused_experts(
    model: nn.Module,
    *,
    execution: str = "sparse",
    selection: str = "routed",
    calibration_execution: str = "dense",
    export_execution: str = "predicated",
) -> list[str]:
    """Swap every fused-experts module for :class:`PredicatedExperts`.

    :return: qualified names of the modules replaced.
    :raises RuntimeError: if the model has no fused-experts module -- silently
        doing nothing here would produce an unexportable model with no warning.
    """
    targets = [
        name for name, module in model.named_modules() if is_fused_experts(module)
    ]
    if not targets:
        raise RuntimeError(
            "ExportableMoE adaptation found no fused-experts module (a module with 3D "
            "gate_up_proj/down_proj parameters). Either this is not a "
            "transformers>=5 MoE model, or the expert layout has changed."
        )
    for name in targets:
        fused = rgetattr(model, name)
        rsetattr(
            model,
            name,
            PredicatedExperts(
                fused,
                execution=execution,
                selection=selection,
                calibration_execution=calibration_execution,
                export_execution=export_execution,
            ),
        )
    return targets


def predicated_expert_modules(obj) -> list[PredicatedExperts]:
    """Every :class:`PredicatedExperts` reachable from ``obj``.

    Non-torch objects (e.g. an ONNX quantsim) yield an empty list, so callers can
    bracket code unconditionally.
    """
    candidates = [obj, getattr(obj, "model", None)]
    for candidate in candidates:
        if isinstance(candidate, nn.Module):
            return [m for m in candidate.modules() if isinstance(m, PredicatedExperts)]
    return []


@contextlib.contextmanager
def forced_expert_activation(
    obj,
    *,
    execution: str | None = None,
    force_all: bool | None = None,
    phase: str = "calibration",
) -> Iterator[list[PredicatedExperts]]:
    """Put MoE experts into their calibration/export policy for the duration.

    A no-op for non-MoE models, so framework-agnostic callers need no knowledge
    of the model. Defaults come from each module's own configuration, selected by
    ``phase`` (``"calibration"`` or ``"export"``).
    """
    if phase not in ("calibration", "export"):
        raise ValueError("phase must be 'calibration' or 'export'")
    modules = predicated_expert_modules(obj)
    saved = [(m, m.execution, m.force_all.clone()) for m in modules]
    try:
        for module in modules:
            default = (
                module.export_execution
                if phase == "export"
                else module.calibration_execution
            )
            module.execution = execution or default
            want_force = (
                module.selection == "all" if force_all is None else bool(force_all)
            )
            module.force_all.fill_(1.0 if want_force else 0.0)
        yield modules
    finally:
        for module, execution_was, force_was in saved:
            module.execution = execution_was
            module.force_all.copy_(force_was)


def set_onnx_force_all(model_path: str, value: bool) -> int:
    """Flip the exported graph's ``force_all`` initializer(s) in place.

    External data is deliberately not loaded: a real export keeps weights in a
    sidecar, and re-saving them inline to change 4 bytes would rewrite gigabytes
    and breach protobuf's 2GB ceiling.

    :return: number of initializers updated.
    """
    import numpy as np
    import onnx

    model = onnx.load(model_path, load_external_data=False)

    # If ONNX merged force_all with another constant of the same value, patching
    # it would rewrite that operator instead, skipping every expert silently.
    illegal = sorted(_force_all_consumers(model.graph) - {"Max"})
    if illegal:
        raise RuntimeError(
            f"'force_all' in {model_path} is consumed by {illegal}, not only by "
            f"Max. ONNX initializer deduplication has merged it with another "
            f"constant of the same value, so patching it would corrupt that "
            f"operator (typically turning an expert predicate into 'w > 1', "
            f"silently skipping every expert). Do not use this graph."
        )

    raw = np.array([1.0 if value else 0.0], dtype=np.float32).tobytes()
    updated = 0

    def _patch(graph):
        nonlocal updated
        for init in graph.initializer:
            if "force_all" not in init.name:
                continue
            if list(init.dims) != [1]:
                raise RuntimeError(
                    f"Initializer '{init.name}' has dims {list(init.dims)}, "
                    f"expected [1]. Refusing to patch an unexpected tensor."
                )
            # Inline raw data, dropping any external reference.
            init.ClearField("external_data")
            init.data_location = onnx.TensorProto.DEFAULT
            for field in ("float_data", "int32_data", "int64_data", "double_data"):
                init.ClearField(field)
            init.data_type = onnx.TensorProto.FLOAT
            init.raw_data = raw
            updated += 1
        for node in graph.node:
            for attr in node.attribute:
                if attr.g.ByteSize():
                    _patch(attr.g)
                for sub in attr.graphs:
                    _patch(sub)

    _patch(model.graph)
    if not updated:
        raise RuntimeError(
            f"No 'force_all' initializer found in {model_path}. It was likely "
            f"constant-folded during export; re-export with folding disabled or "
            f"plumb force_all as a graph input instead."
        )

    onnx.save(model, model_path)
    return updated


def _force_all_consumers(graph) -> set[str]:
    """Op types that read a ``force_all`` initializer, subgraphs included."""
    consumers: set[str] = set()
    for node in graph.node:
        if any("force_all" in name for name in node.input):
            consumers.add(node.op_type)
        for attr in node.attribute:
            if attr.g.ByteSize():
                consumers |= _force_all_consumers(attr.g)
            for sub in attr.graphs:
                consumers |= _force_all_consumers(sub)
    return consumers


def unquantized_subgraph_ops(graph, op_types=("MatMul", "Gemm", "Conv")) -> int:
    """Count compute ops in control-flow subgraphs that hold no ``QcQuantizeOp``.

    Such ops run in float whatever the sim reports for the enclosing graph.
    """
    count = 0

    def walk(g):
        nonlocal count
        for node in g.node:
            for attr in node.attribute:
                subgraphs = list(attr.graphs)
                if attr.g.ByteSize():
                    subgraphs.append(attr.g)
                for sub in subgraphs:
                    has_quantizer = any(n.op_type == "QcQuantizeOp" for n in sub.node)
                    if not has_quantizer:
                        count += sum(1 for n in sub.node if n.op_type in op_types)
                    walk(sub)

    walk(graph)
    return count


def assert_experts_quantized(quantsim) -> None:
    """Refuse a sim whose expert bodies are unquantized.

    aimet-onnx does not descend into control-flow subgraphs, so a predicated MoE
    sim runs its experts in float while reporting a quantized model. Fail rather
    than fall back to ``dense``: the graph is right, the quantizer needs to catch
    up. No-op for graphs without control flow.
    """
    model = getattr(quantsim, "model", quantsim)
    graph = getattr(getattr(model, "model", model), "graph", None)
    if graph is None:
        return
    unquantized = unquantized_subgraph_ops(graph)
    if unquantized:
        raise RuntimeError(
            f"{unquantized} compute ops inside ONNX control-flow subgraphs have no "
            f"quantizer. For a predicated MoE export this means every expert GEMM "
            f"is running in FLOAT while the sim reports a quantized model, so any "
            f"accuracy number would be optimistic and wrong.\n"
            f"aimet-onnx does not yet insert quantizers inside If/Loop bodies; "
            f"that is the feature this graph needs (quantizer insertion, "
            f"calibration and encoding export for subgraph bodies).\n"
            f"To get numbers before that lands, set the ExportableMoE adaptation's "
            f"`export_execution: dense` -- a flat, fully quantizable graph with "
            f"the same outputs and the same per-expert encodings, losing only the "
            f"runtime expert skip."
        )


# ---------------------------------------------------------------------------
# Adaptation registration
# ---------------------------------------------------------------------------

_MOE_MODEL_TYPES = ("qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text")


class ExportableMoEMixin:
    """Mixin that unfuses experts at instantiation.

    Configurable via YAML adaptation kwargs (``selection``,
    ``calibration_execution``, ``export_execution``). ``execution`` is not a
    knob: eval always runs ``sparse``, and calibration and export switch policy
    through :func:`forced_expert_activation`.
    """

    selection: str = "routed"
    calibration_execution: str = "dense"
    export_execution: str = "predicated"

    @staticmethod
    def use_dynamo_export() -> bool:
        """torch.jit.trace would bake in whichever branch the sample input took."""
        return True

    @classmethod
    def instantiate_model(cls, *args, **kwargs) -> PreTrainedModel:
        model = super().instantiate_model(*args, **kwargs)
        replace_fused_experts(
            model,
            execution="sparse",
            selection=cls.selection,
            calibration_execution=cls.calibration_execution,
            export_execution=cls.export_execution,
        )
        return model


class ExportableMoEAdaptation(ExportableMoEMixin):
    """Predicated-experts ExportableMoE adaptation."""


def register_adaptations() -> None:
    """Register the ExportableMoE adaptation for every MoE model type.

    Per model_type rather than ``"*"``: ``required_for_export`` matches model_type
    exactly, so ``"*"`` would demand this adaptation for every non-MoE export.
    Also called by tests, which run under a fixture that wipes the registry.
    """
    for model_type in _MOE_MODEL_TYPES:
        YAMLConfigParser.register_adaptation(
            "ExportableMoE", model_type=model_type, required_for_export=True
        )(ExportableMoEAdaptation)


register_adaptations()

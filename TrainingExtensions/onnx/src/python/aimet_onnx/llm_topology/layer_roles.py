# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Name-based classification of decoder linear ops into fine-grained roles.

:func:`get_llm_topology` groups a block's weighted linears coarsely into two
``LinearGroup``\\ s (the ``qkv`` group reading through the input norm, the
``gate_up`` group reading through the post-attention norm). To rotate or
analyze an individual projection (e.g. R2 rotates only V) each group is split
into the individual roles ``q_proj`` / ``k_proj`` / ``v_proj`` and
``gate_proj`` / ``up_proj``.

The split is done by matching the originating ``nn.Module`` attribute name. A node
name from an HF/optimum ONNX export looks like
``/model/.../self_attn/v_proj/MatMul``; the second-to-last ``/``-segment
(``v_proj``) is the attribute name. We match it against a small allow-list of
conventional names per role.

This module is the single seam for future manual registration: pass a custom
``role_patterns`` mapping to override the built-in table for exotic exports.
"""

import re
from enum import Enum
from typing import Dict, Optional, Pattern


class LinearRole(Enum):
    """Fine-grained role of a weighted linear op inside a decoder block."""

    Q_PROJ = "q_proj"
    K_PROJ = "k_proj"
    V_PROJ = "v_proj"
    O_PROJ = "o_proj"
    GATE_PROJ = "gate_proj"
    UP_PROJ = "up_proj"
    DOWN_PROJ = "down_proj"
    FUSED_QKV = "fused_qkv"
    FUSED_GATE_UP = "fused_gate_up"
    MIXER_IN_PROJ = "mixer_in_proj"
    MIXER_OUT_PROJ = "mixer_out_proj"
    UNKNOWN = "unknown"


# Default module-name allow-list per role. Each pattern is matched against the
# second-to-last ``/``-segment of ``Op.name`` (the originating module name),
# optionally suffixed with ``_sha`` (per-head split exports) and/or a per-head
# index (``.0``). Longest / most-specific alternatives are listed first so a
# fused name (``qkv_proj``) is tested before the single-projection names.
#
# NOTE: order matters. FUSED_* are checked before the single Q/K/V roles so a
# fused ``qkv_proj`` is not mis-classified as ``q``/``q_proj``.
_DEFAULT_ROLE_PATTERNS: Dict[LinearRole, Pattern] = {
    # No ``_sha`` on the fused variants: SHA is a per-head *split*, the opposite
    # of a fused projection, so a fused name never carries that suffix.
    LinearRole.FUSED_QKV: re.compile(r"^(qkv_proj|qkv|c_attn|Wqkv|in_proj)(\.\d+)?$"),
    LinearRole.FUSED_GATE_UP: re.compile(
        r"^(gate_up_proj|gateup_proj|gate_up)(\.\d+)?$"
    ),
    LinearRole.Q_PROJ: re.compile(r"^(q_proj|query|q)(_sha)?(\.\d+)?$"),
    LinearRole.K_PROJ: re.compile(r"^(k_proj|key|k)(_sha)?(\.\d+)?$"),
    LinearRole.V_PROJ: re.compile(r"^(v_proj|value|v)(_sha)?(\.\d+)?$"),
    LinearRole.O_PROJ: re.compile(r"^(o_proj|out_proj|dense|wo|o)(_sha)?(\.\d+)?$"),
    LinearRole.GATE_PROJ: re.compile(r"^(gate_proj|w1|gate)(_sha)?(\.\d+)?$"),
    LinearRole.UP_PROJ: re.compile(r"^(up_proj|w3|up)(_sha)?(\.\d+)?$"),
    LinearRole.DOWN_PROJ: re.compile(r"^(down_proj|w2|down)(_sha)?(\.\d+)?$"),
}

# Deterministic priority for resolving a classification: fused variants first,
# then the individual projections. Iteration order of a dict is insertion order,
# but we pin it explicitly so a future edit to the table above cannot silently
# reorder the checks.
_ROLE_PRIORITY = (
    LinearRole.FUSED_QKV,
    LinearRole.FUSED_GATE_UP,
    LinearRole.Q_PROJ,
    LinearRole.K_PROJ,
    LinearRole.V_PROJ,
    LinearRole.O_PROJ,
    LinearRole.GATE_PROJ,
    LinearRole.UP_PROJ,
    LinearRole.DOWN_PROJ,
)


def module_name_of(node_name: str) -> Optional[str]:
    """Return the originating ``nn.Module`` attribute name for ``node_name``.

    A node name from an HF/optimum export looks like
    ``/model/.../self_attn/v_proj/MatMul``; the second-to-last ``/``-segment is
    the module attribute name. Returns ``None`` when the name has too few
    segments to carry a module name.
    """
    parts = node_name.rsplit("/", 2)
    if len(parts) < 2:
        return None
    return parts[-2]


def classify_linear_role(
    node_name: str,
    role_patterns: Optional[Dict[LinearRole, Pattern]] = None,
) -> LinearRole:
    """Classify a weighted linear into a :class:`LinearRole` by its module name.

    :param node_name: Name of a weighted linear node (MatMul / Gemm / Conv).
    :param role_patterns: Optional override of the default role→pattern table.
        Only the roles present in the mapping are tested; roles absent from a
        supplied mapping are skipped. This is the hook for manually registering
        name prefixes for non-standard exports.
    :return: The matched :class:`LinearRole`, or :attr:`LinearRole.UNKNOWN`.
    """
    patterns = role_patterns if role_patterns is not None else _DEFAULT_ROLE_PATTERNS
    module_name = module_name_of(node_name)
    if module_name is None:
        return LinearRole.UNKNOWN
    for role in _ROLE_PRIORITY:
        pattern = patterns.get(role)
        if pattern is not None and pattern.match(module_name):
            return role
    return LinearRole.UNKNOWN


__all__ = [
    "LinearRole",
    "classify_linear_role",
    "module_name_of",
]

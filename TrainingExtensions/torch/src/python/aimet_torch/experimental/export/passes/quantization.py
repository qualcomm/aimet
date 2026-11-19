# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""
Collection of quantization-related graph passes
"""

import itertools
import torch
from aimet_torch.quantization.affine import QuantizeDequantize
from .._utils import _get_aimet_qdq_module


__all__ = [
    "cleanup_duplicate_aimet_qdq_nodes",
    "resolve_aimet_qdq_constraints",
]


def cleanup_duplicate_aimet_qdq_nodes(gm: torch.fx.GraphModule):
    """
    Clean up adjacent QDQ nodes associated with identical QuantizeDequantize object

    Before:
        ... ---> q1 ---> q2 -> ...
               (qdq1)  (qdq1)
    After:
        ... ---> q1 --> ...
               (qdq1)

    where  q1  denotes qdq1 node "q" associated with QuantizeDequantize nn.Module "qdq1"
         (qdq1)
    """
    q1: torch.fx.Node
    q2: torch.fx.Node
    qdq1: QuantizeDequantize | None
    qdq2: QuantizeDequantize | None

    for q2 in gm.graph.nodes:
        if not (qdq2 := _get_aimet_qdq_module(q2)):
            continue

        q1, *_ = q2.all_input_nodes

        if not (qdq1 := _get_aimet_qdq_module(q1)):
            continue

        if qdq1 is qdq2:
            q2.replace_all_uses_with(q1)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    gm.delete_all_unused_submodules()


def resolve_aimet_qdq_constraints(gm: torch.fx.GraphModule, reverse: bool = False):
    """
    Resolve constraints of two adjacent QuantizeDequantize nodes.

    When two adjacent QDQ nodes are both constrained, they can't be deduplicated
    simply by dropping one of them.
    Instead, the two constrained QDQ nodes should be properly "resolved"
    to minimize unnecessary requantization while preserving both constraints

    Example 1. qdq1 tied / qdq2 tied

        Before:
            qdq1 -> maxpool -> qdq1 -> qdq2 -+
                                             +-> cat -> qdq2 ->
                                ... -> qdq2 -+

        After:
            qdq2 -> maxpool -> qdq2 -> qdq2 -+
                                             +-> cat -> qdq2 ->
                                ... -> qdq2 -+

        Replaced qdq1 with qdq2.
        This removes unnecessary requantization from qdq1 to qdq2,
        while preserving the tie constraints of both maxpool and cat

    Example 2. qdq1 fixed / qdq2 tied

        Before:
                      [0, 1]
            softmax -> qdq1 -> qdq2 -+
                                     +-> cat -> qdq2 ->
            sigmoid -> qdq3 -> qdq2 -+
                      [0, 1]

        After:
                      [0, 1]  [0, 1]
            softmax -> qdq1 -> qdq1 -+
                                     +-> cat -> qdq1 ->
            sigmoid -> qdq1 -> qdq1 -+         [0, 1]
                      [0, 1]  [0, 1]

        Replaced qdq2 with qdq1.
        This removes unnecessary requantization from qdq1 to qdq2,
        while preserving all the constraints of softmax, sigmoid, and cat


    Args:
        gm: The GraphModule to be modified in-place
        reverse: If True, the graph is traversed in reverse topological order.

    Preconditions:
        Every qdq node should be in the following form:
            input_provider -> qdq -> user

        where exactly one of the following is true:
            - `input_provider` is graph input or another qdq node
            - `user` is a graph output or another qdq node

        e.g. (graph_in) -> qdq1 -> conv2d -> qdq2 -> qdq3 -> softmax -> qdq4 -> (graph_out)
    """
    # Build mapping table from nn.Module to its associated fx.Node
    module_to_nodes: dict[QuantizeDequantize, set[torch.fx.Node]] = {}

    for node in gm.graph.nodes:
        if qdq := _get_aimet_qdq_module(node):
            module_to_nodes.setdefault(qdq, set()).add(node)

    node_qdq1: torch.fx.Node
    node_qdq2: torch.fx.Node
    qdq1: QuantizeDequantize | None
    qdq2: QuantizeDequantize | None

    all_nodes = gm.graph.nodes

    if reverse:
        all_nodes = reversed(all_nodes)

    for node_qdq2 in all_nodes:
        if not (qdq2 := _get_aimet_qdq_module(node_qdq2)):
            continue

        node_qdq1, *_ = node_qdq2.all_input_nodes

        if not (qdq1 := _get_aimet_qdq_module(node_qdq1)):
            continue

        if qdq1 is qdq2:
            continue

        # Can't resolve if qdq1 and qdq2 have different settings
        if _get_setting(qdq1) != _get_setting(qdq2):
            continue

        # Users of qdq1. String indicates graph output
        qdq1_users: set[QuantizeDequantize | str] = {
            user_module or user_node.name
            for qdq1_node in module_to_nodes[qdq1]
            for user_node in qdq1_node.users
            if (user_module := _get_aimet_qdq_module(user_node))
            or user_node is gm.graph.output_node()
        }
        # Input providers of qdq2. String indicates graph input
        qdq2_input_providers: set[QuantizeDequantize | str] = {
            provider_module or provider_node.name
            for qdq2_node in module_to_nodes[qdq2]
            for provider_node in qdq2_node.all_input_nodes[:1]
            if (provider_module := _get_aimet_qdq_module(provider_node))
            or provider_node.op == "placeholder"
        }
        qdq1_is_fixed = _is_fixed(qdq1)
        qdq2_is_fixed = _is_fixed(qdq2)
        qdq1_is_tied = len(module_to_nodes[qdq1]) > 1
        qdq2_is_tied = len(module_to_nodes[qdq2]) > 1

        match qdq1_is_fixed, qdq2_is_fixed, qdq1_is_tied, qdq2_is_tied:
            case False, False, _, _:
                # Case 0-3. qdq1 not fixed / qdq2 not fixed
                if qdq2_input_providers == {qdq1}:
                    # qdq1 is the only input provider of qdq2.
                    # Replace qdq2 with qdq1
                    #
                    # Case 0a. qdq1 unconstrained / qdq2 unconstrained
                    #
                    #   conv2d -> qdq1 -> qdq2 -> softmax
                    #
                    # Case 1a. qdq1 unconstrained / qdq2 tied
                    #
                    #                   +-> ...
                    #                   |                  +-> qdq2
                    #   conv2d -> qdq1 -+-> qdq2 -> split -+
                    #                                      +-> qdq2
                    #
                    # Case 2a. qdq1 tied / qdq2 unconstrained
                    #
                    #                  +-> qdq1 -> ...
                    #   qdq1 -> split -+
                    #                  +-> qdq1 -> qdq2 -> conv2d
                    #
                    # Case 3a. qdq1 tied / qdq2 tied
                    #
                    #                                             +-> qdq2
                    #   qdq1 -> maxpool -> qdq1 -> qdq2 -> split -+
                    #                                             +-> qdq2
                    winner = qdq1
                elif qdq1_users == {qdq2}:
                    # qdq2 is the only user of qdq1.
                    # Replace qdq1 with qdq2
                    #
                    # Case 0a. qdq1 unconstrained / qdq2 unconstrained
                    #
                    #   conv2d -> qdq1 -> qdq2 -> softmax
                    #
                    # Case 1b. qdq1 unconstrained / qdq2 tied
                    #
                    #   conv2d -> qdq1 -> qdq2 -+
                    #                           +-> cat -> qdq2
                    #              ... -> qdq2 -+
                    #
                    # Case 2b. qdq1 tied / qdq2 unconstrained
                    #
                    #   qdq1 -+
                    #         +-> cat -> qdq1 -> qdq2 -> conv2d
                    #   qdq1 -+
                    #
                    # Case 3b. qdq1 tied / qdq2 tied
                    #   qdq1 -> maxpool -> qdq1 -> qdq2 -+
                    #                                    +-> cat -> qdq2 ->
                    #                       ... -> qdq2 -+
                    winner = qdq2
                else:
                    # Otherwise, can't merge
                    #
                    # Case 0a. qdq1 unconstrained / qdq2 unconstrained
                    #
                    #   (impossible)
                    #
                    # Case 1c. qdq1 unconstrained / qdq2 tied
                    #
                    #                   +-> qdq3 -> softmax
                    #                   |
                    #   conv2d -> qdq1 -+-> qdq2 -+
                    #                             +-> cat -> qdq2
                    #                ... -> qdq2 -+
                    #
                    # Case 2c. qdq1 tied / qdq2 unconstrained
                    #
                    #   (impossible)
                    #
                    # Case 3c. qdq1 tied / qdq2 tied
                    #
                    #                  +-> qdq1 -> ...
                    #   qdq1 -> split -+
                    #                  +-> qdq1 -> qdq2 -+
                    #                                    +-> cat -> qdq2 ->
                    #                       ... -> qdq2 -+
                    winner = None
            case False, True, _, _:
                # Case 4-7. qdq1 not fixed / qdq2 fixed
                if all(
                    isinstance(user, QuantizeDequantize) and _are_equal(user, qdq2)
                    for user in qdq1_users
                ):
                    # All users of qdq1 are fixed to the same range as qdq2.
                    # Replace qdq1 with qdq2
                    #
                    # Case 4a. qdq1 unconstrained / qdq2 fixed
                    #
                    #                      [-1, 1]
                    #                   +-> qdq3 -> asin -> ...
                    #   conv2d -> qdq1 -+-> qdq2 -> asin -> ...
                    #                      [-1, 1]
                    #
                    # Case 5a. qdq1 unconstrained / qdq2 fixed & tied
                    #
                    #                      [0, ?]
                    #                   +-> qdq3 -> relu -> ...
                    #   conv2d -> qdq1 -+-> qdq2 -> relu -> qdq2 -> ...
                    #                      [0, ?]          [0, ?]
                    #
                    # Case 6a. qdq1 tied / qdq2 fixed
                    #
                    #                               [-1, 1]
                    #   qdq1 -+                   +-> qdq3 -> asin
                    #         +-> concat -> qdq1 -+-> qdq2 -> asin
                    #   qdq1 -+                     [-1, 1]
                    #
                    # Case 7a. qdq1 tied / qdq2 fixed & tied
                    #
                    #                                [0, ?]
                    #   qdq1 -+                   +-> qdq3 -> relu
                    #         +-> concat -> qdq1 -+-> qdq2 -> relu -> qdq2
                    #   qdq1 -+                      [0, ?]          [0, ?]
                    winner = qdq2
                else:
                    # Otherwise, can't merge
                    #
                    # Case 4b. qdq1 unconstrained / qdq2 fixed
                    #
                    #                   +-> qdq3 -> softmax
                    #                   |
                    #   conv2d -> qdq1 -+-> qdq2 -> asin
                    #                      [-1, 1]
                    #
                    # Case 5b. qdq1 unconstrained / qdq2 fixed & tied
                    #
                    #                   +-> qdq3 -> softmax
                    #                   |
                    #   conv2d -> qdq1 -+-> qdq2 -> relu -> qdq2 -> ...
                    #                      [0, ?]          [0, ?]
                    #
                    # Case 6b. qdq1 tied / qdq2 fixed
                    #
                    #   qdq1 -+                   +-> qdq3 -> softmax
                    #         +-> concat -> qdq1 -+-> qdq2 -> asin
                    #   qdq1 -+                      [-1, 1]
                    #
                    # Case 7b. qdq1 tied / qdq2 fixed & tied
                    #
                    #   qdq1 -+                   +-> qdq3 -> softmax
                    #         +-> concat -> qdq1 -+-> qdq2 -> relu -> qdq2
                    #   qdq1 -+                      [0, ?]          [0, ?]
                    winner = None
            case True, False, _, _:
                # Case 8-11. qdq1 fixed / qdq2 not fixed
                if all(
                    isinstance(provider, QuantizeDequantize)
                    and _are_equal(provider, qdq1)
                    for provider in qdq2_input_providers
                ):
                    # All input providers of qdq2 are fixed to the same range as qdq1.
                    # Replace qdq2 with qdq1
                    #
                    # Case 8a. qdq1 fixed / qdq2 unconstrained
                    #
                    #   softmax -> qdq1 -> qdq2 -> conv
                    #             [0, 1]
                    #
                    # Case 9a. qdq1 fixed / qdq2 tied
                    #
                    #             [0, 1]
                    #   softmax -> qdq1 -> qdq2 -+
                    #                            +-> cat -> qdq2 -> ...
                    #   sigmoid -> qdq3 -> qdq2 -+
                    #             [0, 1]
                    #
                    # Case 10a. qdq1 fixed & tied / qdq2 unconstrained
                    #
                    #    qdq1 -> relu -> qdq1 -> qdq2 -> conv
                    #   [0, ?]          [0, ?]
                    #
                    # Case 11a. qdq1 fixed & tied / qdq2 tied
                    #
                    #  [0, 1]             [0, 1]
                    #   qdq1 -> maxpool -> qdq1 -> qdq2 -+
                    #                                    +-> cat -> qdq2 -> ...
                    #   qdq3 -> maxpool -> qdq3 -> qdq2 -+
                    #  [0, 1]             [0, 1]
                    winner = qdq1
                else:
                    # Otherwise, can't merge
                    #
                    # Case 8b. qdq1 fixed / qdq2 unconstrained
                    #
                    #   (impossible)
                    #
                    # Case 9b. qdq1 fixed / qdq2 tied
                    #
                    #             [0, 1]
                    #   softmax -> qdq1 -> qdq2 -+
                    #                            +-> cat -> qdq2 -> ...
                    #    conv2d -> qdq3 -> qdq2 -+
                    #
                    # Case 10b. qdq1 fixed & tied / qdq2 unconstrained
                    #
                    #  (impossible)
                    #
                    # Case 11b. qdq1 fixed & tied / qdq2 tied
                    #
                    #   [0, ?]          [0, ?]
                    #    qdq1 -> relu -> qdq1 -> qdq2 -+
                    #                                  +-> cat -> qdq2 -> ...
                    #    qdq3 -> relu -> qdq3 -> qdq2 -+
                    #   [0, ?]          [0, ?]
                    winner = None
            case True, True, _, _:
                # Case 12-15. Both qdq1 and qdq2 are fixed.
                #
                #   softmax -> qdq1 --> qdq2 -> asin -> ...
                #             [0, 1]  [-1, 1]
                #
                # qdq1 and qdq2 can be merged if and only if
                # they are fixed to the same encoding range
                winner = qdq1 if _are_equal(qdq1, qdq2) else None
            case _:
                raise AssertionError("This case is unreachable")

        if not winner:
            continue

        target_module_name = node_qdq1.target if winner is qdq1 else node_qdq2.target
        qdq1_nodes = module_to_nodes.pop(qdq1)
        qdq2_nodes = module_to_nodes.pop(qdq2)

        for node in itertools.chain(qdq1_nodes, qdq2_nodes):
            node.target = target_module_name

        module_to_nodes[winner] = qdq1_nodes | qdq2_nodes

    gm.graph.eliminate_dead_code()
    gm.recompile()
    gm.delete_all_unused_submodules()


def _is_fixed(qdq: QuantizeDequantize) -> bool:
    """
    Return True if qdq is fixed to a specific encoding range.
    """
    min_fixed = qdq.is_initialized("min") and not qdq.is_overwrite_allowed("min")
    max_fixed = qdq.is_initialized("max") and not qdq.is_overwrite_allowed("max")
    return min_fixed or max_fixed


def _get_setting(qdq: QuantizeDequantize):
    """
    Return a tuple of settings that determine the encoding of qdq
    """
    # TODO: Treat symmetry as a constraint, not as quantizer setting.
    #
    # Technically, symmetry is just a constraint on zero_point, where
    #   zero_point == 0                ... if signed
    #   zero_point == 2^(bitwidth-1)   ... if unsigned
    #
    # This should be properly handled by resolve_aimet_qdq_constraints.
    # For example:
    #
    #     Before:
    #                              ... ->
    #                                    matmul
    #         conv2d -> qdq1 ---> qdq2 ->
    #                           [zp=128]
    #
    #     After:
    #                              ... ->
    #                                    matmul
    #         conv2d -> qdq2 ---> qdq2 ->
    #                 [zp=128]  [zp=128]
    return (
        qdq.shape,
        qdq.qmin,
        qdq.qmax,
        qdq.block_size,
        qdq.zero_point_shift,
        qdq.symmetric,
    )


def _are_equal(qdq1: QuantizeDequantize, qdq2: QuantizeDequantize) -> bool:
    """
    Return True if qdq1 and qdq2 have the same encoding.
    """
    # Edge case: if qdq1 and qdq2 are the same object, they are equal
    # even if they are not initialized yet
    if qdq1 is qdq2:
        return True

    # Encodings are only comparable if qdq1 and qdq2 have the same settings
    if _get_setting(qdq1) != _get_setting(qdq2):
        return False

    return bool(
        (qdq1_encoding := qdq1.get_encodings())
        and (qdq2_encoding := qdq2.get_encodings())
        and torch.allclose(qdq1_encoding.scale, qdq2_encoding.scale)
        and torch.equal(qdq1_encoding.offset, qdq2_encoding.offset)
    )

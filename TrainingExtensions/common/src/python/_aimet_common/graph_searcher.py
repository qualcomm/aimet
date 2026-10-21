# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Main class for pattern match based graph searcher"""

from typing import Callable, Optional, List, Set
from .connected_graph.connectedgraph import ConnectedGraph
from .graph_pattern_matcher import PatternType
from .utils import AimetLogger
from .connected_graph.operation import Op

__all__ = [
    "GraphSearcher",
    "_check_if_conv3d_or_depthwise_conv",
]


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


# TODO: #5597: Remove Conv3d and Depthwise Conv supergroup once HTP support is added
def _check_if_conv3d(op: Op) -> bool:
    if op.type != "Conv" and op.type != "ConvTranspose":
        return False

    if op.inputs[0].shape is not None and len(op.inputs[0].shape) == 5:
        return True

    # Additional check on weights shape if available.
    if op.inputs[1].shape is not None and len(op.inputs[1].shape) == 5:
        return True

    # Not a conv3d op
    return False


# TODO: #5597: Remove Conv3d and Depthwise Conv supergroup once HTP support is added
def _check_if_depthwise_conv(op: Op) -> bool:
    if op.type not in ("Conv", "ConvTranspose"):
        return False

    groups = getattr(op, "groups", None) or 1

    if groups == 1:
        return False

    if len(op.inputs) < 2:
        raise RuntimeError("Expecting at least two inputs to Conv op.")

    weight_shape = op.inputs[1].shape

    if weight_shape is None:
        # Dynamic weight; cannot determine if depthwise conv or not.
        return False

    # Check if Cin == groups
    if op.type == "Conv":
        # Conv weight layout: [Cout, Cin / groups, kH, kW]
        _, in_channels_per_group, *_ = weight_shape
        return in_channels_per_group == 1
    else:
        # ConvTranspose weight layout: [Cin, Cout / groups, kH, kW]
        in_channels, *_ = weight_shape
        return in_channels == groups


# TODO: #5597: Remove Conv3d and Depthwise Conv supergroup once HTP support is added
def _check_if_conv3d_or_depthwise_conv(op: Op) -> bool:
    if _check_if_conv3d(op):
        return True
    if _check_if_depthwise_conv(op):
        return True
    return False


class GraphSearcher:
    """
    Graph searcher class performs graph search on connected graph.
    It uses SlidingWindow to maintain the search window and PatternMatcher to match sub graph patterns.
    """

    def __init__(
        self, conn_graph: ConnectedGraph, patterns_with_callback: List[PatternType]
    ):
        """
        initializes params required for pattern matching
        :param patterns_with_callback: patterns with corresponding call back functions
        """
        self._connected_graph = conn_graph
        self._patterns_with_callbacks = patterns_with_callback
        self.type_to_op_dict = {}
        for op in conn_graph.get_all_ops().values():
            if op.type in self.type_to_op_dict:
                self.type_to_op_dict[op.type].append(op)
            else:
                self.type_to_op_dict[op.type] = [op]

        self._already_matched: Set[Op] = set()

    # pylint: disable=too-many-nested-blocks
    def find_all_patterns_in_graph_apply_actions(
        self,
        ignore: Optional[Op] = None,
        op_pattern_to_reject: Callable[[Op], bool] = None,
        disjoint: bool = False,
    ):
        """
        Find corresponding op sequences and apply actions.
        :param ignore: List of operations to ignore during searching
        :param op_pattern_to_reject: Callable to perform additional checks on Op to reject pattern match.
            This is useful to express intent on patterns that should not be matched.
            Since GraphSearcher performs high level pattern match, this enables to provide override for aggressive rejection for a given op config.
        :param disjoint: If True, ensures that matched patterns do not share any ops.
        """

        if ignore is None:
            ignore = []

        # Search patterns starting with longer patterns first
        for pattern_type in sorted(
            self._patterns_with_callbacks, key=lambda l: len(l.pattern), reverse=True
        ):
            if pattern_type.pattern[0] in self.type_to_op_dict:
                # One or more ops in the graph correspond to the current pattern's starting op type
                for op in self.type_to_op_dict[pattern_type.pattern[0]]:
                    matched_ops = self._match_pattern(
                        op, pattern_type.pattern, ignore, op_pattern_to_reject
                    )
                    if not matched_ops:
                        continue
                    for matched_ops_list in matched_ops:
                        if disjoint and any(
                            matched_op in self._already_matched
                            for matched_op in matched_ops_list
                        ):
                            # This pattern has already been matched as part of a longer pattern
                            continue
                        else:
                            self._already_matched |= set(matched_ops_list)

                        pattern_type.action(pattern_type, matched_ops_list)
                        logger.debug("found match: %s", matched_ops_list)

    # pylint: disable=too-many-branches, too-many-return-statements
    def _match_pattern(
        self,
        op: Op,
        pattern: List[str],
        ignored_ops: List[Op],
        op_pattern_to_reject: Optional[Callable[[Op], bool]] = None,
    ) -> Optional[List[List[Op]]]:
        if not pattern:
            return []

        matched_ops = None
        if op in ignored_ops:
            if not op.outputs:
                return None
            for child_op in op.output_ops:
                matched_child_ops = self._match_pattern(child_op, pattern, ignored_ops)
                if matched_child_ops is not None:
                    if matched_ops is None:
                        matched_ops = []
                    matched_ops.extend(matched_child_ops)
            return matched_ops

        if op.type != pattern[0]:
            return None

        # If additional op pattern checks are provided and matches for rejection, early exit.
        # This is useful to provide more aggresive checks e.g. depthwise conv or 3d conv, ...

        if op_pattern_to_reject is not None and op_pattern_to_reject(op):
            return False

        if len(pattern) > 1:
            # Still more to match
            if not op.outputs:
                return None
            if len(op.output_ops) > 1:  # Can't match patterns with branches
                return None
            for child_op in op.output_ops:
                matched_child_ops = self._match_pattern(
                    child_op, pattern[1:], ignored_ops
                )
                if matched_child_ops:
                    if matched_ops is None:
                        matched_ops = []
                    for matched_child_op_list in matched_child_ops:
                        matched_ops.append([op] + matched_child_op_list)
        else:
            matched_ops = [[op]]

        return matched_ops

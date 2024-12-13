# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" Main class for pattern match based graph searcher"""
from typing import Optional
from aimet_common.utils import AimetLogger
from aimet_common.connected_graph.operation import Op

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

class GraphSearcher:
    """
    Graph searcher class performs graph search on connected graph.
    It uses SlidingWindow to maintain the search window and PatternMatcher to match sub graph patterns.
    """

    def __init__(self, conn_graph, patterns_with_callback):
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

    # pylint: disable=too-many-nested-blocks
    def find_all_patterns_in_graph_apply_actions(self, ignore: Optional[Op]=None):
        """
        Find corresponding op sequences and apply actions.
        :param ignore: List of operations to ignore during searching
        """

        if ignore is None:
            ignore = []

        # Search patterns starting with longer patterns first
        for pattern_type in sorted(self._patterns_with_callbacks, key=lambda l: len(l.pattern), reverse=True):
            if pattern_type.pattern[0] in self.type_to_op_dict:
                # One or more ops in the graph correspond to the current pattern's starting op type
                for op in self.type_to_op_dict[pattern_type.pattern[0]]:
                    matched_ops = self._match_pattern(op, pattern_type.pattern, ignore)
                    if matched_ops:
                        for matched_ops_list in matched_ops:
                            pattern_type.action(pattern_type, matched_ops_list)
                            logger.debug('found match: %s', matched_ops_list)

    # pylint: disable=too-many-branches, too-many-return-statements
    def _match_pattern(self, op, pattern, ignored_ops):
        if not pattern:
            return []

        matched_ops = None
        if op in ignored_ops:
            if not op.output:
                return None
            for child_op in op.output.consumers:
                matched_child_ops = self._match_pattern(child_op, pattern, ignored_ops)
                if matched_child_ops is not None:
                    if matched_ops is None:
                        matched_ops = []
                    matched_ops.extend(matched_child_ops)
            return matched_ops

        if op.type != pattern[0]:
            return None

        if len(pattern) > 1:
            # Still more to match
            if not op.output:
                return None
            if len(op.output_ops) > 1: # Can't match patterns with branches
                return None
            for child_op in op.output.consumers:
                matched_child_ops = self._match_pattern(child_op, pattern[1:], ignored_ops)
                if matched_child_ops:
                    if matched_ops is None:
                        matched_ops = []
                    for matched_child_op_list in matched_child_ops:
                        matched_ops.append([op] + matched_child_op_list)
        else:
            matched_ops = [[op]]

        return matched_ops

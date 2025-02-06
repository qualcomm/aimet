# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" QuantizerGroup interface """

import abc
from typing import Dict, List, Tuple, Set
from collections import deque
from aimet_common.connected_graph.connectedgraph_utils import get_all_input_ops
from aimet_common.amp.utils import CANDIDATE_WITH_DTYPE

class QuantizerGroupBase(abc.ABC):
    """ QuantizerGroup interface """

    @abc.abstractmethod
    def get_candidate(self, name_to_quantizer_dict: Dict) -> CANDIDATE_WITH_DTYPE:
        """ Gets Activation & parameter bitwidth """

    @abc.abstractmethod
    def set_quantizers_to_candidate(
            self,
            name_to_quantizer_dict: Dict,
            candidate: CANDIDATE_WITH_DTYPE
    ) -> None:
        """ Sets a quantizer group to a given candidate bitwidth """

    @abc.abstractmethod
    def to_list(self) -> List[Tuple[str, str]]:
        """ Converts quantizer group to a list """

    @abc.abstractmethod
    def get_active_quantizers(self, name_to_quantizer_dict: Dict) -> List:
        """ Find all active tensor quantizers associated with this quantizer group """


def reformat_supported_kernels(supported_kernels: Dict) -> Dict[str, List[CANDIDATE_WITH_DTYPE]]:
    """
    reformat the supported kernels dict to match the internal representation of it ->
    ((activation bitwidth, activation data type), (param bitwidth, param data type))
    :param supported_kernels: Dict with module name and its candidates
    """

    ret_dict = {}

    for op_name, op_supported_kernels in supported_kernels.items():
        candidates = []
        for supported_kernel in op_supported_kernels:
            if "param" in supported_kernel:
                candidate = ((supported_kernel['activation']['bitwidth'], supported_kernel['activation']['dtype']),
                             (supported_kernel['param']['bitwidth'], supported_kernel['param']['dtype']))
            else:
                candidate = ((supported_kernel['activation']['bitwidth'], supported_kernel['activation']['dtype']), )
            candidates.append(candidate)

        ret_dict[op_name] = candidates

    return ret_dict

def store_candidates_for_quantizer(supported_kernels: dict, op: str, amp_candidates_set: set, act_bw_set: set,
                                   act_and_param_set: list, act_only_set: list, null_intersection_ops: list):
    '''
    Store candidates for quantizer

    :param supported_kernels: Dictionary containing list of supported kernels for op_types
    :param op: op
    :param amp_candidates_set: list of candidates passed by the user for the AMP algorithm
    :param act_bw_set: Activation bitwidths extracted from given amp candidates
    :param act_and_param_set: List to store activation and param bitwidths for a given op
    :param act_only_set: List to store activation bitwidths for a given op
    '''
    # Ops containing only act bw should be tuple of len = 1. For Eg: [((8, 'int'),), ((16, 'int'),)]
    if len(supported_kernels[op][0]) == 1:
        candidates_for_quantizer = set(supported_kernels[op]).intersection(act_bw_set)
        # Intersection is NULL. Error is raised at later step
        if not candidates_for_quantizer:
            null_intersection_ops.append(op)
            return
        act_only_set.append(candidates_for_quantizer)
    else:
        candidates_for_quantizer = set(supported_kernels[op]).intersection(amp_candidates_set)
        # Intersection is NULL. Error is raised at later step
        if not candidates_for_quantizer:
            null_intersection_ops.append(op)
            return
        act_and_param_set.append(candidates_for_quantizer)

def order_candidates(act_and_param_set: set, act_only_set: set) -> List[CANDIDATE_WITH_DTYPE]:
    '''
    Order the candidate list (with priority: Activation BW > Params BW) in non-increasing order

    :param act_and_param_set: Set with (Activation BW, Param BW)
    :param act_only_set: Set with (Activation BW, )
    :return: List of candidates in order of their priority
    '''
    supported_candidates_for_quantizers = []

    # If both (act_bw, param_bw) and (act_bw, ) are present in a single quantizer group then using only those
    # combinations of (act_bw, param_bw) where the act_bw is common in both
    if act_and_param_set and act_only_set:
        for candidate in act_and_param_set:
            if (candidate[0],) in act_only_set:
                supported_candidates_for_quantizers.append(candidate)
        supported_candidates_for_quantizers = sorted(supported_candidates_for_quantizers,
                                                     key=lambda cand: (cand[0][0], cand[1][0]), reverse=True)
    elif act_and_param_set:
        supported_candidates_for_quantizers = list(act_and_param_set)
        supported_candidates_for_quantizers = sorted(supported_candidates_for_quantizers,
                                                     key=lambda cand: (cand[0][0], cand[1][0]), reverse=True)
    elif act_only_set:
        supported_candidates_for_quantizers = list(act_only_set)
        supported_candidates_for_quantizers = sorted(supported_candidates_for_quantizers,
                                                     key=lambda cand: cand[0][0], reverse=True)

    return supported_candidates_for_quantizers

def get_supported_candidates_for_quantizers(quantizers: List,
                                            onnx_ops: dict,
                                            supported_kernels: dict,
                                            amp_candidates: List[CANDIDATE_WITH_DTYPE],
                                            use_all_amp_candidates: bool) -> List[CANDIDATE_WITH_DTYPE]:
    """
    find the intersection of supported kernels for all the quantizers present in the quantizers list
    :param quantizers: List of quantizer names present in a quantizer group
    :param onnx_ops: Dict which specifies the mapping between quantizers and their ONNX types
    :param supported_kernels: Dictionary containing list of supported kernels for op_types
    :param amp_candidates: list of candidates passed by the user for the AMP algorithm
    :param use_all_amp_candidates:  Using the “supported_kernels” field in the config file (under defaults
                and op_type sections), a list of supported candidates can be specified. All the AMP candidates which are
                passed through the “candidates” field may not be supported based on the data passed through
                “supported_kernels”. When the field “use_all_amp_candidates” is set to True, the AMP algorithm
                will ignore the "supported_kernels" in the config file and continue to use all AMP candidates.
    """

    # pylint: disable=too-many-locals
    # if use_all_amp_candidates is set to True, use all the candidates present in the AMP candidate list
    # if supported_kernels is empty, use all the candidates (this is by design)
    if use_all_amp_candidates or not supported_kernels:
        return amp_candidates

    if "defaults" not in supported_kernels.keys():
        raise ValueError('Aborting AMP, supported_kernels expects defaults to be present')

    if len(supported_kernels) == 1 and not supported_kernels.get("defaults"):
        # defaults section is empty. Return all the amp_candidates
        return amp_candidates

    amp_candidates_set = set(amp_candidates)
    act_bw_set = {(candidate[0],) for candidate in amp_candidates_set}

    act_only_set = []
    act_and_param_set = []
    null_intersection_ops = []

    for quantizer in quantizers:
        ops = onnx_ops[quantizer]

        # By default assign "defaults" candidates for the given quantizer. But, if there is a specialized entry for
        # this quantizer, then assign the new set of candidates
        candidates_for_quantizer = supported_kernels['defaults']
        ops_found = False

        for op in ops:
            if op in supported_kernels:
                ops_found = True
                # Store candidates for quantizer
                store_candidates_for_quantizer(supported_kernels, op, amp_candidates_set, act_bw_set, act_and_param_set,
                                               act_only_set, null_intersection_ops)

        # Default candidate selected if op not found in supported kernels
        if not ops_found:
            if not candidates_for_quantizer:
                raise ValueError("'defaults' are empty in supported kernels")
            default_intersection_set = set(candidates_for_quantizer).intersection(amp_candidates_set)
            if not default_intersection_set:
                raise ValueError("Given AMP candidates has no common candidates with default candidates")
            act_and_param_set.append(default_intersection_set)

    # If intersection between user given candidates and supported candidates is empty, then raise an error
    if null_intersection_ops:
        error_msg = ""
        for op in null_intersection_ops:
            error_msg += f"Given AMP candidates ({amp_candidates_set}) has no intersection with supported " \
                         f"candidates for {op}. Consider adding candidates supported by this op. Supported " \
                         f"candidates are: {supported_kernels[op]}\n\n"
        raise ValueError(error_msg)

    act_and_param_set = set.intersection(*act_and_param_set) if act_and_param_set else set()
    act_only_set = set.intersection(*act_only_set) if act_only_set else set()

    # Ordering the candidates on op level so that if a candidate is not supported by some op then the candidate
    # chosen will be the first index of the op
    supported_candidates_for_quantizers = order_candidates(act_and_param_set, act_only_set)

    if not supported_candidates_for_quantizers:
        raise ValueError(
            'Provided combination of supported_kernels does not yield any candidates for the quantizer group:',
            [onnx_ops[quantizer] for quantizer in quantizers], 'AMP candidates passed:', amp_candidates,
            'supported_kernels read from the config file:', supported_kernels)

    return supported_candidates_for_quantizers

def compute_baseline_candidate_options(quantizers_with_supported_candidates: Dict,
                                       amp_candidates: List[CANDIDATE_WITH_DTYPE],
                                       use_all_amp_candidates: bool) -> List[CANDIDATE_WITH_DTYPE]:
    """
    Computes and returns a list of candidate options for calculating max candidate in the AMP algorithm
    :param quantizers_with_supported_candidates: Dict of quantizers with a list of supported candidates as value. This
                is used along with amp_candidates to determine the max candidate options
    :param amp_candidates: list of candidates passed by the user for the AMP algorithm
    :param use_all_amp_candidates: Using the “supported_kernels” field in the config file (under defaults
                and op_type sections), a list of supported candidates can be specified. All the AMP candidates which are
                passed through the “candidates” field may not be supported based on the data passed through
                “supported_kernels”. When the field “use_all_amp_candidates” is set to True, the AMP algorithm
                will ignore the "supported_kernels" in the config file and continue to use all AMP candidates.
    """
    if use_all_amp_candidates:
        return amp_candidates

    candidates_shortlisted = []
    # Considering all quantizer candidates having both activation bw & param bw as candidate options
    for candidates_per_quantizer_group in quantizers_with_supported_candidates.values():
        # Baseline candidate shouldn't be the one having ONLY activation bw
        if len(candidates_per_quantizer_group[0]) != 1:
            candidates_shortlisted.extend(candidates_per_quantizer_group)

    candidates_shortlisted = list(set(candidates_shortlisted))

    if not candidates_shortlisted:
        raise ValueError(
            'No candidate is supported by all the quantizer_groups and is also present in amp_candidates')

    # candidates_shortlisted is the correct list which can be used as max_candidate_options. But since we use "set"
    # to compute the max_candidate_options, the result is in an unpredictable order. This is okay for production since
    # order does not matter for the AMP algorithm. But for some of the unit tests, the candidates are expected to be in
    # the same order as  amp_candidates, ie if amp_candidates originally has [a, b, c, d, e] and if only "a", "b" and
    # "c" are the valid options, then the expected output for the tests is [a, b, c]
    max_candidate_options = []
    for candidate in amp_candidates:
        if candidate in candidates_shortlisted and candidate not in max_candidate_options:
            max_candidate_options.append(candidate)

    return max_candidate_options


def find_valid_ops(connected_graph, op_not_to_traverse: List) -> Set:
    """
    Finds valid ops for creating quantizer groups

    :param connected_graph: ConnectedGraph
    :param op_not_to_traverse: List of ops not to traverse
    :return: Set of valid ops
    """

    input_ops = get_all_input_ops(connected_graph)
    input_ops = [op for op in input_ops if op.type not in op_not_to_traverse]
    valid_ops = set()

    q = deque(input_ops)
    while q:
        op = q.pop()
        valid_ops.add(op.dotted_name)
        for consumer in op.output_ops:
            if consumer.type not in op_not_to_traverse and consumer.dotted_name not in valid_ops:
                q.append(consumer)

    return valid_ops

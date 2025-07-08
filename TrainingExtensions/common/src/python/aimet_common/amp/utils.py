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

"""Utilities for mixed precision feature"""

from enum import IntEnum
import math
import os
import json
from typing import Callable, Tuple, List, Dict, Sequence
import numpy as np
from bokeh.plotting import ColumnDataSource, figure, output_file, save
from bokeh.models import HoverTool
from bokeh.transform import factor_cmap
from bokeh.colors import groups as color_groups
import pandas as pd
from aimet_common.defs import QuantizationDataType
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)
CANDIDATE_WITH_DTYPE = Tuple[
    Tuple[int, QuantizationDataType], Tuple[int, QuantizationDataType]
]
CANDIDATE_WITHOUT_DTYPE = Tuple[int, int]

ACCURACY_LIST_DATAPOINT = Tuple["QuantizerGroup", CANDIDATE_WITH_DTYPE, float, int]
ACCURACY_LIST = List[ACCURACY_LIST_DATAPOINT]
FLOATING_POINT_MULTIPLIER = math.sqrt(1.5)


class AmpCandidate(CANDIDATE_WITH_DTYPE):
    """
    Convenience class that extends CANDIDATE_WITH_DTYPE
    """

    @property
    def param_bw(self):
        """Returns parameter bitwidth"""
        # pylint: disable=unsubscriptable-object
        return self[CandAttr.parameter][CandParam.bitwdith]

    @property
    def param_dtype(self):
        """Returns parameter data type"""
        # pylint: disable=unsubscriptable-object
        return self[CandAttr.parameter][CandParam.data_type]

    @property
    def output_bw(self):
        """Returns output bitwidth"""
        # pylint: disable=unsubscriptable-object
        return self[CandAttr.activation][CandParam.bitwdith]

    @property
    def output_dtype(self):
        """Returns output data type"""
        # pylint: disable=unsubscriptable-object
        return self[CandAttr.activation][CandParam.data_type]


class AMPSearchAlgo(IntEnum):
    """
    Defines the available search algorithm to be used for the Phase 2 of Mixed Precison.
    """

    BruteForce = 1
    Interpolation = 2
    Binary = 3


def enable_quantizers(quantizers):
    """
    Enable quantizers.

    Note: This is very anti-OOP implementation.
          Needs to be refactored in the future.
    """
    for quantizer in quantizers:
        if hasattr(quantizer, "enabled"):
            if not quantizer.enabled:
                quantizer.enabled = True
        else:
            quantizer.enable()


def disable_quantizers(quantizers):
    """
    Disable quantizers.

    Note: This is very anti-OOP implementation.
          Needs to be refactored in the future.
    """
    for quantizer in quantizers:
        if hasattr(quantizer, "enabled"):
            if quantizer.enabled:
                quantizer.enabled = False
        else:
            quantizer.disable()


class CandAttr(IntEnum):
    """Enum Class to index candidate attributes -> activation or parameter"""

    activation = 0
    parameter = 1


class CandParam(IntEnum):
    """Enum Class to index the candidate parameter"""

    bitwdith = 0
    data_type = 1


def sort_accuracy_list(
    accuracy_list: List, index_of_quantizer_group: Dict["QuantizerGroup", int]
):
    """
    Sort accuracy list

    :param accuracy_list: Accuracy list
    :param index_of_quantizer_group: Mapping from QuantizerGroup to index
    :return: Sorted accuracy list
    """

    def _find_higher_bitwidth_sum(cand: CANDIDATE_WITH_DTYPE) -> int:
        """
        Function for sorting accuracy list based on higher sum of BW

        :param cand: candidate containing Activation BW, Param BW
        :return: sum of BW
        """
        if len(cand) == 1:
            ((activation_bw, _),) = cand
            param_bw, _ = None, None
        else:
            (activation_bw, _), (param_bw, _) = cand
        return (activation_bw + param_bw) if param_bw is not None else activation_bw

    def _find_index_of_quantizer_group(quant_group) -> int:
        """
        Function for sorting accuracy list based on position of quantizer group in model

        :param quant_group: Quantizer group
        :return: Negative of index of quantizer group because we are sorting in reversed order
        """
        return -1 * index_of_quantizer_group[quant_group]

    def key(elem):
        quantizer_group, candidate, eval_score, bitops = elem
        return (
            eval_score,
            _find_higher_bitwidth_sum(candidate),
            bitops,
            _find_index_of_quantizer_group(quantizer_group),
        )

    return sorted(accuracy_list, key=key, reverse=True)


def calculate_starting_bit_ops(
    mac_dict: Dict, default_candidate: CANDIDATE_WITH_DTYPE
) -> int:
    """
    Calculate sum of bit ops of compressible layers of the model, assuming every compressible layer uses
    default_bitwidth

    :param mac_dict: Dictionary mapping modules to mac counts
    :param default_candidate: Default starting bitwidth (or bitwidth and data type)
    """
    starting_bit_ops = 0
    if len(default_candidate) == 1:
        ((activation_bw, activation_dtype),) = default_candidate
        param_bw, param_dtype = None, None
    else:
        (activation_bw, activation_dtype), (param_bw, param_dtype) = default_candidate

    activation_effective_bw = get_effective_bitwidth(activation_dtype, activation_bw)
    if param_bw is not None:
        param_effective_bw = get_effective_bitwidth(param_dtype, param_bw)

    # Sum up mac counts
    for macs in mac_dict.values():
        starting_bit_ops += macs

    # Multiply sum by default_bitwidth squared
    starting_bit_ops *= (
        (activation_effective_bw * param_effective_bw)
        if param_bw is not None
        else activation_effective_bw
    )
    return starting_bit_ops


_dtype_to_str = {
    QuantizationDataType.int: "int",
    QuantizationDataType.float: "float",
}


def _candidate_to_str(candidate: CANDIDATE_WITH_DTYPE) -> str:
    if len(candidate) == 1:
        ((activation_bw, activation_dtype),) = candidate
        param_bw, param_dtype = None, None
    else:
        (activation_bw, activation_dtype), (param_bw, param_dtype) = candidate
    activation_dtype = _dtype_to_str[activation_dtype]
    if param_dtype is not None:
        param_dtype = _dtype_to_str[param_dtype]
    return f"W: {param_dtype}{param_bw} / A: {activation_dtype}{activation_bw}"


def visualize_quantizer_group_sensitivity(
    accuracy_list: ACCURACY_LIST,
    baseline_candidate: CANDIDATE_WITH_DTYPE,
    fp32_accuracy: float,
    results_dir: str,
) -> figure:
    """
    Creates & returns an interactive bokeh plot. The plot is saved under results_dir
    as quantizer_group_sensitivity.html

    :param accuracy_list: List of Quantizer Group, bitwidth, accuracy, and bitops.
    :param baseline_candidate: The candidate that performs the best in terms of accuracy.
    :param fp32_accuracy: The accuracy of baseline_candidate.
    :param results_dir: The directory under which plot will be saved
    :return: Bokeh Plot
    """
    file_path = os.path.join(results_dir, "quantizer_group_sensitivity.html")
    output_file(file_path)
    plot = create_sensitivity_plot(accuracy_list, baseline_candidate, fp32_accuracy)
    save(plot)
    return plot


DEFAULT_BOKEH_FIGURE_WIDTH = 600
MIN_WIDTH_PER_BAR = 40
MIN_WIDTH_PER_CHAR = 6


def create_sensitivity_plot(
    accuracy_list: ACCURACY_LIST,
    baseline_candidate: CANDIDATE_WITH_DTYPE,
    fp32_accuracy: float,
) -> figure:
    """
    Return a Bokeh plot that visualizes each QuantizerGroup's sensitivity to quantization.

    :param accuracy_list: List of Quantizer Group, bitwidth, accuracy, and bitops.
    :param baseline_candidate: The candidate that performs the best in terms of accuracy.
    :param fp32_accuracy: The accuracy of baseline_candidate.
    :return: Bokeh Plot
    """
    df = pd.DataFrame(
        {
            "QuantizerGroup": str(quantizer_group),
            "Bitwidth": candidate,
            "Accuracy": accuracy,
            "BitwidthWithEnumVals": (
                (candidate[0][0], candidate[0][1].value),
                (candidate[1][0], candidate[1][1].value),
            )
            if len(candidate) != 1
            else (
                (candidate[0][0], candidate[0][1].value),
                (candidate[0][0], candidate[0][1].value),
            ),
        }
        for quantizer_group, candidate, accuracy, _ in accuracy_list
    )

    df = df.sort_values(by=["BitwidthWithEnumVals", "QuantizerGroup"]).drop(
        ["BitwidthWithEnumVals"], axis=1
    )
    df["Bitwidth"] = df["Bitwidth"].apply(_candidate_to_str)

    group = df.groupby(by=["QuantizerGroup", "Bitwidth"])

    # Create plot
    plot_width = max(
        # Minimum plot width
        DEFAULT_BOKEH_FIGURE_WIDTH,
        # Fit the major label
        MIN_WIDTH_PER_BAR * len(df),
        # Fit the group label
        MIN_WIDTH_PER_CHAR
        * df["QuantizerGroup"].nunique()
        * df["QuantizerGroup"].map(len).max(),
    )
    plot = figure(
        x_range=group,
        x_axis_label="QuantizerGroup",
        y_axis_label="Accuracy",
        width=plot_width,
        title="QuantizerGroup Sensitivity",
    )

    # Plot accuracy
    plot.vbar(
        x="QuantizerGroup_Bitwidth",
        top="Accuracy_mean",
        width=0.9,
        source=group,
        line_color=None,
        fill_color=factor_cmap(
            "QuantizerGroup_Bitwidth",
            palette=list(color_groups.blue),
            factors=df["Bitwidth"].unique(),
            start=1,
            end=2,
        ),
    )

    # Plot baseline accuracy
    plot.ray(
        x=0,
        y=fp32_accuracy,
        length=0,
        angle=0,
        line_color="green",
        line_width=2,
        legend_label=f"baseline ({_candidate_to_str(baseline_candidate)})",
    )

    hover = HoverTool(
        tooltips=[
            ("QuantizerGroup", "@QuantizerGroup_Bitwidth"),
            ("Accuracy", "@Accuracy_mean"),
        ],
        mode="mouse",
    )
    plot.add_tools(hover)

    plot.legend.location = "top_left"
    plot.y_range.start = 0
    plot.x_range.range_padding = 0.1
    plot.xaxis.major_label_orientation = 1
    plot.xgrid.grid_line_color = None

    return plot


def visualize_pareto_curve(pareto_front_list: List, results_dir: str) -> figure:
    """
    Creates & returns an interactive bokeh plot. The plot is saved under results_dir as pareto_curve.html

    :param pareto_front_list: List of Relative bit ops, acc, Quantizer Group, bitwidth
    :param results_dir: The directory under which plot will be saved
    :return: Bokeh Plot
    """
    file_path = os.path.join(results_dir, "pareto_curve.html")
    output_file(file_path)
    plot = create_pareto_curve(pareto_front_list)
    save(plot)
    return plot


def create_pareto_curve(pareto_front_list: List) -> figure:
    """
    Creates & returns an interactive bokeh plot. The plot is saved under results_dir as pareto_curve.html

    :param pareto_front_list: List of Relative bit ops, acc, Quantizer Group, bitwidth
    :param results_dir: The directory under which plot will be saved
    :return: Bokeh Plot
    """
    bits_ops = []
    acc_list = []
    for relative_bit_ops, acc, _, _ in pareto_front_list:
        bits_ops.append(relative_bit_ops)
        acc_list.append(acc)

    source = ColumnDataSource(data={"BitOps": bits_ops, "Accuracy": acc_list})

    plot = figure(
        x_axis_label="BitOps",
        y_axis_label="Accuracy",
        width=800,
        height=800,
        title="Accuracy vs BitOps",
    )

    plot.line(
        "BitOps",
        "Accuracy",
        line_width=2,
        line_color="#2171b5",
        line_dash="dotted",
        source=source,
        name="name",
    )

    hover1 = HoverTool(
        tooltips=[("BitOps", "@BitOps"), ("Accuracy", "@Accuracy")], mode="mouse"
    )
    plot.add_tools(hover1)

    return plot


def get_effective_bitwidth(data_type: QuantizationDataType, bitwidth: int) -> float:
    """
    Returns a float multiplier multiplied by bitwidth to get an effective bitwidth based on the datatype

    :param data_type: Either int or float data type
    """
    return (
        FLOATING_POINT_MULTIPLIER * bitwidth
        if data_type == QuantizationDataType.float
        else bitwidth
    )


def create_quant_group_to_candidate_dict(accuracy_list_reverse):
    """
    This will create the dictionary where key will be quantizer group and value will be the
    order of candidates as occured in accuracy_list_reverse for the particular quantizer group

    :param accuracy_list_reverse: phase1 accuracy list in reversed form.
    :return: dictionary where key will be quantizer group and value will be the
    order of candidates as occured in accuracy_list_reverse for the particular quantizer group
    """
    res_dict = {}
    for elem in accuracy_list_reverse:
        quantizer_group, candidate, _, _ = elem
        if quantizer_group in res_dict:
            res_dict[quantizer_group].append(candidate)
        else:
            res_dict[quantizer_group] = [candidate]

    return res_dict


def modify_candidate_in_accuracy_list(
    accuracy_list_reverse, quant_group_to_candidate_dict, max_candidate
):
    """
    candidate in accuracy_list_reverse give information about the sensitivity of the quantizer group when turning
    the quantizer group into that particular candidate, modify_candidate_in_accuracy_list function will replace
    the candidate by the target candidate, that the quantizer group should be converted to, which is the next
    higher deserving candidate

    :param accuracy_list_reverse: phase1 accuracy list in reversed form.
    :param quant_group_to_candidate_dict: dictionary where key will be quantizer group and value will be the
    order of candidates as occured in accuracy_list_reverse for the particular quantizer group
    :param:max_candidate: The maximum [bitwidth, dtype] candidate
    :return: modified accuracy_list_reverse
    """
    result_list = []
    for elem in accuracy_list_reverse:
        quantizer_group, candidate, eval_score, bit_ops_reduction = elem
        cand_sens_order = quant_group_to_candidate_dict[quantizer_group]
        # If candidate position in cand_sens_order is the last index or if candidate is not found in cand_sens_order, then
        # set index to the length of the list cand_sens_order, otherwise get the next better candidate.
        if candidate in cand_sens_order:
            index = cand_sens_order.index(candidate) + 1
        else:
            index = len(cand_sens_order)
        # replacing the candidate with the next better candidate and the max candidate.
        if index == len(cand_sens_order):
            result_list.append(
                (quantizer_group, max_candidate, eval_score, bit_ops_reduction)
            )
        else:
            result_list.append(
                (quantizer_group, cand_sens_order[index], eval_score, bit_ops_reduction)
            )
    return result_list


def export_list(lst: list, results_dir: str, file_name: str = "amp_info_list"):
    """
    Exports info of execution as a json file, like time taken by phase1, phase2, percentage of quantizers in target precision.
    This function can be used to log the ad-hoc information.

    :param lst: Information in List of tuples about execution.
    :param results_dir: Path to save list.
    :param: file_name: File name to save the list.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    file_path_json = os.path.join(results_dir, file_name + ".json")
    with open(file_path_json, "w", encoding="utf-8") as f:
        json.dump(lst, f, indent=1)


def binary_search_ascending(values: Sequence[Callable[[], float]], target: float):
    """
    Find value `target` from an array `values`.
    If the exact target value may not be present in the array,
    find the element which is closest to the target

    :param values: List of functions whose return values are **assumed** to in ascending order.
    :param target: Target value to find from `values`.
    :returns: Index of the element that is equal to or higher than the target value.
    """
    binary_execution_info = ""
    left = 0
    right = len(values) - 1

    binary_execution_info += (
        "starting left and right index as " + str(left) + "," + str(right) + "\n"
    )
    if target <= values[left]():
        return left  # Smallest elem is already larger than target value
    if target >= values[right]():
        return right  # Largest elem is smaller than target value

    while right - left > 1:
        percentile = 0.5
        mid = left + (right - left) * percentile
        mid = int(np.clip(mid, left + 1, right - 1))  # Make sure left < mid < right
        binary_execution_info += "mid index calculated as " + str(mid) + "\n"

        if values[mid]() < target:
            left = mid
        elif values[mid]() > target:
            right = mid
        else:
            left = mid
            right = mid
        binary_execution_info += (
            "left and right calculated as " + str(left) + "," + str(right) + "\n"
        )

    if values[left]() >= target:
        binary_execution_info += "final index returned " + str(left) + "\n"
        return left
    binary_execution_info += "final index returned " + str(right) + "\n"
    logger.info("binary search trace: %s", binary_execution_info)
    return right


def binary_search(
    values: Sequence[Callable[[], float]], target: float, phase2_reverse=False
) -> int:
    """
    Find value `target` from an array `values`.
    If the exact target value may not be present in the array,
    find the element which is closest to the target

    :param values: List of functions whose return values are **assumed** to in descending order.
    :param target: Target value to find from `values`.
    :param phase2_reverse: If user will set this parameter to True, then values parameter are **assumed** in increasing order.
    :param result_dir: Path to save binary execution logs.
    :returns: Index of the element that is equal to or higher than the target value.
    """
    if phase2_reverse:
        i = binary_search_ascending(values, target)
    else:
        values = list(reversed(values))
        i = binary_search_ascending(values, target)
        i = len(values) - 1 - i
    return i


def interpolation_search_ascending(
    values: Sequence[Callable[[], float]], target: float
) -> int:
    """
    Find value `target` from an array `values`.
    If the exact target value may not be present in the array,
    find the element which is closest to the target

    :param values: List of functions whose return values are **assumed** to in ascending order.
    :param target: Target value to find from `values`.
    :param result_dir: Path to save binary execution logs.
    :returns: Index of the element that is equal to or higher than the target value.
    """
    left = 0
    right = len(values) - 1

    if target <= values[left]():
        return left  # Smallest elem is already larger than target value
    if target >= values[right]():
        return right  # Largest elem is smaller than target value

    while right - left > 1:
        percentile = (target - values[left]()) / (values[right]() - values[left]())
        mid = left + (right - left) * percentile
        mid = int(np.clip(mid, left + 1, right - 1))  # Make sure left < mid < right

        if values[mid]() < target:
            left = mid
        elif values[mid]() > target:
            right = mid
        else:
            left = mid
            right = mid

    if values[left]() >= target:
        return left
    return right


def interpolation_search(
    values: Sequence[Callable[[], float]], target: float, phase2_reverse=False
) -> int:
    """
    Find value `target` from an array `values`.
    If the exact target value may not be present in the array,
    find the element which is closest to the target

    :param values: List of functions whose return values are **assumed** to in descending order if phase2_reverse is set to
    False otherwise increasing order.
    :param target: Target value to find from `values`.
    :param phase2_reverse: If user will set this parameter to True, then values parameter are **assumed** in increasing order.
    :param result_dir: Path to save binary execution logs.
    :returns: Index of the element that is equal to or higher than the target value.
    """
    if phase2_reverse:
        i = interpolation_search_ascending(values, target)
    else:
        values = list(reversed(values))
        i = interpolation_search_ascending(values, target)
        i = len(values) - 1 - i

    return i


def brute_force_search(
    values: Sequence[Callable[[], float]], target: float, phase2_reverse=False
) -> int:
    """
    Find value `target` from an array `values`.
    If the exact target value may not be present in the array,
    find the element which is closest to the target

    :param values: List of functions whose return values are **assumed** to in descending order if phase2_reverse is set to
    False otherwise increasing order.
    :param target: Target value to find from `values`.
    :param phase2_reverse: If user will set this parameter to True, then values parameter are **assumed** in increasing order.
    :param result_dir: Path to save binary execution logs.
    :returns: Index of the element that is equal to or higher than the target value.
    """
    if phase2_reverse:
        for i, fn in enumerate(values):
            if fn() >= target:
                return i

    else:
        for i, fn in enumerate(values):
            if fn() == target:
                return i
            if fn() < target:
                if i > 0:
                    return (
                        i - 1
                    )  # (i-1)-th elem is the smallest elem that satisfies the target score
                return i  # Even 0-th elem can't satisfy the target score

    # target is smaller than the last element
    return len(values) - 1


def candidate_cost(
    activation_candidate: Tuple[int, QuantizationDataType],
    param_candidate: Tuple[int, QuantizationDataType],
) -> float:
    """
    Approximates the computational cost factor of a given mixed precision candidate. The raw value itself is meaningless
    and is only to be used for comparison between candidates.
    """
    factor = 1
    if activation_candidate[0]:
        factor *= get_effective_bitwidth(
            activation_candidate[1], activation_candidate[0]
        )
    if param_candidate[0]:
        factor *= get_effective_bitwidth(param_candidate[1], param_candidate[0])
    return factor

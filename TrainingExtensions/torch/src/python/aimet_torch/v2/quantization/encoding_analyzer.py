# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=redefined-builtin
# pylint: disable=missing-docstring

""" Computes statistics and encodings """

from abc import ABC, abstractmethod
import math
import numpy as np
import warnings
from dataclasses import dataclass
from typing import TypeVar, Generic, Tuple, Optional, List
import itertools
import torch
from aimet_common.quantsim import _get_minimum_scale
from aimet_torch.v2.utils import reduce, StatisticsNotFoundError, _is_expandable


@dataclass
class _MinMaxRange:
    min: Optional[torch.Tensor] = None
    max: Optional[torch.Tensor] = None

@dataclass
class _Histogram:
    histogram: torch.Tensor = None
    bin_edges: torch.Tensor = None
    min: Optional[torch.Tensor] = None
    max: Optional[torch.Tensor] = None

_Statistics = TypeVar('_Statistics', _MinMaxRange, _Histogram)

class _Observer(Generic[_Statistics], ABC):
    """
    Observes and gathers statistics
    """
    def __init__(self, shape: tuple):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)

    @abstractmethod
    def collect_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        pass

    @abstractmethod
    def merge_stats(self, stats: _Statistics):
        pass

    @abstractmethod
    def reset_stats(self):
        pass

    @abstractmethod
    def get_stats(self) -> _Statistics:
        pass


class _MinMaxObserver(_Observer[_MinMaxRange]):
    """
    Observer for Min-Max calibration technique
    """
    def __init__(self, shape: tuple):
        super().__init__(shape)
        self.stats = _MinMaxRange()

    @torch.no_grad()
    def collect_stats(self, input_tensor: torch.Tensor) -> _MinMaxRange:
        new_min = reduce(input_tensor, shape=self.shape, reduce_op=torch.min).values
        new_max = reduce(input_tensor, shape=self.shape, reduce_op=torch.max).values
        return _MinMaxRange(new_min, new_max)

    @torch.no_grad()
    def merge_stats(self, stats: _MinMaxRange):
        updated_min = self.stats.min
        if stats.min is not None:
            if updated_min is None:
                updated_min = stats.min.clone()
            else:
                updated_min = torch.minimum(updated_min, stats.min)

        updated_max = self.stats.max
        if stats.max is not None:
            if updated_max is None:
                updated_max = stats.max.clone()
            else:
                updated_max = torch.maximum(updated_max, stats.max)

        self.stats = _MinMaxRange(updated_min, updated_max)

    def reset_stats(self):
        self.stats = _MinMaxRange()

    def get_stats(self) -> _MinMaxRange:
        return self.stats


class _HistogramObserver(_Observer[_Histogram]):
    """
    Observer for Histogram based calibration techniques (percentile, MSE)
    """
    def __init__(self, shape: tuple, num_bins: int):
        super().__init__(shape)
        self.num_bins = num_bins
        self.stats = [_Histogram() for _ in range(np.prod(self.shape, dtype=np.int32))]

    # pylint: disable=too-many-locals
    @torch.no_grad()
    def collect_stats(self, input_tensor: torch.Tensor) -> List[_Histogram]:
        if not _is_expandable(self.shape, input_tensor.shape):
            raise RuntimeError(f"Shape {self.shape} is incompatible with input of shape {input_tensor.shape}")

        hist_stats = []
        input_shape = tuple(input_tensor.shape)
        histogram_shape = self.shape

        padded_histogram_shape = (
            *itertools.repeat(1, len(input_shape) - len(histogram_shape)),
            *histogram_shape
        )

        for i, _ in enumerate(self.stats):
            hist_input = input_tensor

            for axis, dim in enumerate(padded_histogram_shape):
                if dim == 1:
                    continue
                # elements in current axis, ex: could be W*C, C, or 1 for input_shape [H, W, C]
                numel = np.prod(padded_histogram_shape[axis+1:], dtype=np.int32)
                # index where hist_input at current dimension will be sliced at
                index = (i // numel) % dim
                hist_input = torch.unsqueeze(torch.select(hist_input, axis, index), axis)

            hist_min, hist_max = self._get_min_max(hist_input)

            bin_edges = self._create_bin_edges(min_val=hist_min, max_val=hist_max, device=input_tensor.device)
            histogram = torch.histc(hist_input.to(torch.float), bins=self.num_bins, min=bin_edges[0], max=bin_edges[-1])

            # clip inf values to hist_min and hist_max and adjust for any fp errors
            histogram[0] += torch.sum(hist_input < bin_edges[0])
            histogram[-1] += torch.sum(hist_input > bin_edges[-1])

            hist_stats.append(_Histogram(histogram, bin_edges, hist_min, hist_max))

        return hist_stats

    def _get_min_max(self, hist_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        isfinite = hist_input.isfinite()
        min = torch.where(isfinite, hist_input, float('inf')).min()
        max = torch.where(isfinite, hist_input, -float('inf')).max()
        return min, max

    def _create_bin_edges(self, min_val, max_val, device):
        # Adjust min/max values to be in line with PyTorch's torch.histc implementation
        min_val, max_val = torch.where(min_val == max_val, min_val - 0.5, min_val).float(), \
                           torch.where(min_val == max_val, max_val + 0.5, max_val).float()

        step = (max_val - min_val) / self.num_bins
        return torch.arange(0, self.num_bins + 1, device=device) * step + min_val

    def _get_bin_num(self, bin_width, curr_min, data):
        bin_tensor = torch.full(data.shape, self.num_bins - 1, device=data.device)
        index_tensor = (data - curr_min) / bin_width
        return torch.minimum(index_tensor.to(torch.int32), bin_tensor)

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-locals
    @torch.no_grad()
    def merge_stats(self, new_stats_list: List[_Histogram], input_tensor: torch.Tensor):
        if self.stats[0].histogram is None:
            self.stats = new_stats_list
            return

        hist_inputs = torch.reshape(input_tensor, (len(new_stats_list), -1))
        for index, new_stats in enumerate(new_stats_list):
            curr_stats = self.stats[index]
            curr_input = hist_inputs[index]

            updated_min = min(new_stats.min, curr_stats.min)
            updated_max = max(new_stats.max, curr_stats.max)

            # if the current histogram can capture new_stats within in its range
            if updated_min == curr_stats.min and updated_max == curr_stats.max:
                histogram_updates = curr_stats.histogram
            else:
                dest_bin_width = (updated_max - updated_min) / self.num_bins
                src_bin_width = (curr_stats.max - curr_stats.min) / self.num_bins
                histogram_updates = torch.zeros(self.num_bins, device=input_tensor.device)

                src_bin_start = curr_stats.min + (src_bin_width * torch.arange(0, self.num_bins, device=input_tensor.device))
                dest_bin_index = self._get_bin_num(dest_bin_width, updated_min, src_bin_start)
                dest_bin_end = updated_min + dest_bin_width * (dest_bin_index + 1)

                # split curr_hist if values in source bin cannot neatly fold into dest bin
                split_hist_value = torch.round(((dest_bin_end - src_bin_start) / src_bin_width) * curr_stats.histogram)
                dest_bin_updates = torch.minimum(split_hist_value, curr_stats.histogram)

                # update appropriate bin with either the full or split curr_hist value
                histogram_updates.index_add_(dim=0, index=dest_bin_index, source=dest_bin_updates)

                # if curr_hist is split, update other bin that the remaining values fall into
                other_bins = (dest_bin_updates < curr_stats.histogram).nonzero().flatten()
                other_bin_index = self._get_bin_num(dest_bin_width, updated_min, src_bin_start + dest_bin_width)
                other_bin_updates = curr_stats.histogram - dest_bin_updates
                histogram_updates.index_add_(dim=0,
                                             index=other_bin_index[other_bins],
                                             source=other_bin_updates[other_bins])

            # create histogram given input tensor and full range
            expanded_histogram = torch.histc(curr_input.to(torch.float), bins=self.num_bins, min=updated_min, max=updated_max)
            expanded_histogram += histogram_updates.to(expanded_histogram.device)

            # clip inf values to hist_min and hist_max
            expanded_histogram[0] += torch.sum(curr_input < updated_min)
            expanded_histogram[-1] += torch.sum(curr_input > updated_max)

            expanded_bin_edges = self._create_bin_edges(min_val=updated_min, max_val=updated_max, device=expanded_histogram.device)
            self.stats[index] = _Histogram(expanded_histogram, expanded_bin_edges, updated_min, updated_max)

    def reset_stats(self):
        self.stats = [_Histogram() for _ in self.stats]

    def get_stats(self) -> List[_Histogram]:
        return self.stats

class EncodingAnalyzer(Generic[_Statistics], ABC):
    '''
    Base class that gathers statistics of input data and computes encodings
    '''
    def __init__(self, observer: _Observer):
        self.observer = observer

    @torch.no_grad()
    def update_stats(self, input_tensor: torch.Tensor):
        r"""
        Updates the internal statistics given the input data

        Args:
            input_tensor (torch.Tensor): Input data
        """
        new_stats = self.observer.collect_stats(input_tensor)
        self.observer.merge_stats(new_stats)
        return new_stats

    def reset_stats(self):
        """
        Resets the internal stats
        """
        self.observer.reset_stats()

    def compute_encodings(self, num_steps: int, is_symmetric: bool):
        r"""
        Computes encodings based on the input data & calibration scheme and returns the encoding minimum and maximum value

        Args:
            num_steps (int): Number of steps used in quantization.
            is_symmetric (bool): True if encodings are symmetric

        Returns:
            Encoding min and max as a tuple

        """
        return self.compute_encodings_from_stats(self.observer.get_stats(), num_steps, is_symmetric)

    def compute_dynamic_encodings(self, input_tensor: torch.Tensor, num_steps: int,
                                  is_symmetric: bool)-> Tuple[torch.Tensor, torch.Tensor]:
        return self.compute_encodings_from_stats(
            self.observer.collect_stats(input_tensor), num_steps, is_symmetric)

    @abstractmethod
    def compute_encodings_from_stats(self, stats: _Statistics, num_steps: int, is_symmetric: bool)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class MinMaxEncodingAnalyzer(EncodingAnalyzer[_MinMaxRange]):
    r"""
    EncodingAnalyzer subclass which uses min-max calibration. This involves tracking the minimum and maximum observed values and computing the min-max range as :math:`[min(input), max(input)]`

    Args:
        shape (tuple): Shape of calculated encoding

    Example:

        >>> from aimet_torch.v2.quantization.encoding_analyzer import MinMaxEncodingAnalyzer
        >>> encoding_analyzer = MinMaxEncodingAnalyzer(shape=(1,))
        >>> encoding_analyzer.update_stats(torch.randn(100))
        >>> encoding_analyzer.compute_encodings(num_steps=math.pow(2, 8), is_symmetric=False)
        (tensor([-2.0991]), tensor([2.3696]))
        >>> encoding_analyzer.reset_stats()
        >>> encoding_analyzer.update_stats(torch.randn(100))
        _MinMaxRange(min=tensor([-2.1721]), max=tensor([2.2592]))
        >>> encoding_analyzer.compute_encodings(num_steps=math.pow(2, 8), is_symmetric=False)
        (tensor([-2.1721]), tensor([2.2592]))
    """
    def __init__(self, shape: tuple):
        observer = _MinMaxObserver(shape)
        super().__init__(observer)

    #pylint: disable=too-many-locals
    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: _MinMaxRange, num_steps: int, is_symmetric: bool)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        if num_steps <= 0:
            raise ValueError('The number of quantization bins cannot be less than or equal to 0.')

        if stats.min is None or stats.max is None:
            raise StatisticsNotFoundError('No statistics present to compute encodings.')

        minimum_scale = _get_minimum_scale(num_steps)

        # enforces that 0 is within the min/max
        min_with_zero = torch.clamp(stats.min, max=0)
        max_with_zero = torch.clamp(stats.max, min=0)

         # adjusts any min/max pairing that are too close
        tensor_diff = (max_with_zero - min_with_zero) / num_steps
        adjustment_step = minimum_scale * (tensor_diff < minimum_scale)

        if is_symmetric:
            updated_max = max_with_zero + math.floor(num_steps / 2) * adjustment_step
            updated_min = min_with_zero - math.ceil(num_steps / 2) * adjustment_step
        else:
            updated_max = max_with_zero + num_steps * adjustment_step
            updated_min = min_with_zero

        if is_symmetric:
            num_pos_steps = math.floor(num_steps / 2)
            num_neg_steps = math.ceil(num_steps / 2)
            delta = torch.maximum(updated_max / num_pos_steps, -updated_min / num_neg_steps)
            offset = -1 * num_neg_steps
            updated_min = offset * delta
            updated_max = num_pos_steps * delta

        # replace pos and neg inf respectively
        updated_max = torch.clamp(updated_max, max=torch.finfo(stats.min.dtype).max).to(stats.min.dtype)
        updated_min = torch.clamp(updated_min, min=torch.finfo(stats.max.dtype).min).to(stats.max.dtype)

        return updated_min, updated_max

def _flag_extreme_min_max(curr_min, curr_max):
    extreme_val = torch.full_like(curr_min, torch.finfo(curr_min.dtype).max)
    if not (torch.all(torch.isfinite(curr_min)) and torch.all(torch.isfinite(curr_max))):
        warnings.warn('Infinite or NaN values detected within input! This may skew the associated encodings')

    if torch.any(torch.isclose(torch.abs(curr_min), extreme_val, rtol=0.05)) or torch.any(torch.isclose(torch.abs(curr_max), extreme_val, rtol=0.05)):
        warnings.warn('Extreme values detected within input! This may skew the associated encodings')

def adjust_min_max(curr_min, curr_max, num_steps, is_symmetric):
    minimum_scale = _get_minimum_scale(num_steps)

    # ensure that 0 is in the range
    curr_min = torch.minimum(curr_min, torch.zeros_like(curr_min))
    curr_max = torch.maximum(curr_max, torch.zeros_like(curr_max))

    # ensure that min/max are finite
    curr_min.clamp_(min=torch.finfo(curr_max.dtype).min, max=0)
    curr_max.clamp_(min=0, max=torch.finfo(curr_max.dtype).max)

    # ensure that min/max aren't too close
    tensor_threshold = (curr_max - curr_min) / num_steps

    if is_symmetric:
        curr_min[tensor_threshold < minimum_scale] -= minimum_scale * math.ceil(num_steps / 2)
        curr_max[tensor_threshold < minimum_scale] += minimum_scale * math.floor(num_steps / 2)
    else:
        curr_max[tensor_threshold < minimum_scale] += minimum_scale * num_steps

    if is_symmetric:
        num_pos_steps = math.floor(num_steps / 2)
        num_neg_steps = math.ceil(num_steps / 2)
        delta = max(curr_max / num_pos_steps, -curr_min / num_neg_steps)
        offset = -1 * num_neg_steps

        curr_min = offset * delta
        curr_max = num_pos_steps * delta



    return curr_min, curr_max

# pylint: disable=arguments-differ
class PercentileEncodingAnalyzer(EncodingAnalyzer[_Histogram]):
    r"""
    EncodingAnalyzer subclass which uses percentile calibration. This involves recording values in a histogram and computing the min-max range given a percentile value :math:`p`. The range would be computed after clipping (100 - :math:`p`)% of the largest and smallest observed values.

    Args:
        shape (tuple): Shape of calculated encoding
        num_bins (int): Number of bins used to create the histogram
        percentile (float): Percentile value which is used to clip values

    Example:

        >>> from aimet_torch.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer
        >>> encoding_analyzer = PercentileEncodingAnalyzer(shape=(1,), num_bins = 10, percentile = 80)
        >>> encoding_analyzer.update_stats(torch.randn(100))
        >>> encoding_analyzer.compute_encodings(num_steps = math.pow(2, 8), is_symmetric = False)
        (tensor([-1.1188]), tensor([0.3368]))
        >>> encoding_analyzer.reset_stats()
        >>> encoding_analyzer.update_stats(torch.randn(100))
        [_Histogram(histogram=tensor([ 1.,  1.,  8., 13., 19., 27., 16., 10.,  3.,  2.]), bin_edges=tensor([-2.5710, -2.0989, -1.6269, -1.1548, -0.6827, -0.2106,  0.2614,  0.7335, 1.2056,  1.6776,  2.1497]), min=tensor(-2.5710), max=tensor(2.1497))]
        >>> encoding_analyzer.compute_encodings(num_steps = math.pow(2, 8), is_symmetric = False)
        (tensor([-1.1548]), tensor([0.2614]))
    """
    def __init__(self, shape: tuple, num_bins: int = 2048, percentile: float = 100):
        if num_bins <= 0:
            raise ValueError('Number of bins cannot be less than or equal to 0.')

        observer = _HistogramObserver(shape=shape, num_bins=num_bins)
        super().__init__(observer)
        self.set_percentile(percentile)

    def set_percentile(self, percentile):
        r"""
        Set the clipping percentile of the encoding analyzer. The encoding analyzer will clip the (100 - :math:`p`)% largest and smallest observed values from the encoding range when computing encodings.

        Args:
            percentile (float): Percentile value which is used to clip values

        """
        if percentile < 50 or percentile > 100:
            raise ValueError('Percentile value must be within 50-100 range')

        self.percentile = percentile

    @torch.no_grad()
    def update_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        """
        :meta private:
        """
        new_stats = self.observer.collect_stats(input_tensor)
        self.observer.merge_stats(new_stats, input_tensor)
        return new_stats

    # pylint: disable=too-many-locals
    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: List[_Histogram], num_steps: int, is_symmetric: bool)\
            -> Tuple[torch.Tensor, torch.Tensor]:

        if num_steps <= 0:

            raise ValueError('The number of quantization bins cannot be less than or equal to 0.')

        if stats[0].histogram is None:
            raise StatisticsNotFoundError('No statistics present to compute encodings.')

        encoding_min_list = []
        encoding_max_list = []

        for list_elem in stats:
            cum_sum = torch.cumsum(list_elem.histogram, dim=0)
            # trim percentile value from min and max
            max_index = torch.searchsorted(cum_sum, cum_sum[-1] * self.percentile/100)
            min_index = torch.searchsorted(cum_sum, cum_sum[-1] * (1 - self.percentile/100))

            if self.percentile == 100:
                min_index = 0
                max_index = -1

            curr_min = list_elem.bin_edges[min_index]
            curr_max = list_elem.bin_edges[max_index]
            # adjust min/max
            updated_min, updated_max = adjust_min_max(curr_min, curr_max, num_steps, is_symmetric)
            encoding_min_list.append(updated_min)
            encoding_max_list.append(updated_max)

        encoding_min = torch.tensor(encoding_min_list, device=stats[0].histogram.device, dtype = stats[0].min.dtype)
        encoding_min = torch.reshape(encoding_min, self.observer.shape)

        encoding_max = torch.tensor(encoding_max_list, device=stats[0].histogram.device, dtype = stats[0].max.dtype)
        encoding_max = torch.reshape(encoding_max, self.observer.shape)

        return encoding_min, encoding_max


class SqnrEncodingAnalyzer(EncodingAnalyzer[_Histogram]):
    r"""
    EncodingAnalyzer subclass which uses SQNR calibration. This involves recording values in a histogram and computing the min-max range based on values that produce the lowest expected SQNR.

    Args:
        shape (tuple): Shape of calculated encoding
        num_bins (int): Number of bins used to create the histogram
        asymmetric_delta_candidates (int): Number of delta values to search over in asymmetric mode
        symmetric_delta_candidates (int): Number of delta values to search over in symmetric mode
        offset_candidates (int): Number of offset values to search over in asymmetric mode
        max_parallelism (int): Maximum number of encodings to process in parallel (higher number results in higher memory usage but faster computation)
        gamma (float): Weighting factor on clipping noise (higher value results in less clipping noise)
        percentile (float): Percentile value which is used to clip values

    Example:

        >>> from aimet_torch.v2.quantization.encoding_analyzer import SqnrEncodingAnalyzer
        >>> encoding_analyzer = SqnrEncodingAnalyzer(shape=(1,), num_bins = 10, gamma = 1)
        >>> encoding_analyzer.update_stats(torch.randn(100))
        >>> encoding_analyzer.compute_encodings(num_steps = math.pow(2, 8), is_symmetric = False)
        (tensor([-2.3612]), tensor([2.8497]))
        >>> encoding_analyzer.reset_stats()
        >>> encoding_analyzer.update_stats(torch.randn(100))
        [_Histogram(histogram=tensor([ 2.,  0.,  8.,  8., 16., 22., 23., 12.,  6.,  3.]), bin_edges=tensor([-2.8907, -2.3625, -1.8343, -1.3061, -0.7779, -0.2497,  0.2784,  0.8066, 1.3348,  1.8630,  2.3912]), min=tensor(-2.8907), max=tensor(2.3912))]
        >>> encoding_analyzer.compute_encodings(num_steps = math.pow(2, 8), is_symmetric = False)
        (tensor([-2.7080]), tensor([2.2438]))
    """
    def __init__(self,
                 shape: tuple,
                 num_bins: int = 2048, *,
                 asymmetric_delta_candidates=17,
                 symmetric_delta_candidates=101,
                 offset_candidates=21,
                 max_parallelism=64,
                 gamma=3.0):
        if num_bins <= 0:
            raise ValueError('Number of bins cannot be less than or equal to 0.')
        observer = _HistogramObserver(shape=shape, num_bins=num_bins)
        super().__init__(observer)
        self.asym_delta_candidates = asymmetric_delta_candidates
        self.sym_delta_candidates = symmetric_delta_candidates
        self.num_offset_candidates = offset_candidates
        self.gamma = gamma
        self.max_parallelism = max_parallelism

    @torch.no_grad()
    def update_stats(self, input_tensor: torch.Tensor) -> _Statistics:
        """
        :meta private:
        """
        new_stats = self.observer.collect_stats(input_tensor)
        self.observer.merge_stats(new_stats, input_tensor)
        return new_stats

    # pylint: disable=too-many-locals
    @torch.no_grad()
    def compute_encodings_from_stats(self, stats: List[_Histogram], num_steps: int, is_symmetric: bool)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        if stats[0].histogram is None:
            raise StatisticsNotFoundError('No statistics present to compute encodings.')
        if num_steps <= 0:
            raise ValueError('The number of quantization bins cannot be less than or equal to 0.')
        chunked_stats = [stats[i:min(i+self.max_parallelism, len(stats))] for i in range(0, len(stats), self.max_parallelism)]
        best_deltas, best_offsets = [], []
        for stats_ in chunked_stats:
            test_deltas, test_offsets = self._pick_test_candidates(stats_, num_steps, is_symmetric)
            best_delta, best_offset = self._select_best_candidates(test_deltas, test_offsets, stats_, num_steps)
            best_deltas.append(best_delta)
            best_offsets.append(best_offset)
        best_offset = best_offsets[0] if is_symmetric else torch.cat(best_offsets)
        best_delta = torch.cat(best_deltas)
        min_enc = best_offset * best_delta
        max_enc = min_enc + num_steps * best_delta

        min_enc = min_enc.to(stats[0].min.dtype)
        max_enc = max_enc.to(stats[0].max.dtype)

        return min_enc.view(self.observer.shape), \
               max_enc.view(self.observer.shape)

    def _pick_test_candidates(self, stats, num_steps, symmetric):
        minimum_scale = _get_minimum_scale(num_steps)
        # min/max.shape = (num_histograms, )
        min_vals = torch.stack([stat.min for stat in stats])
        max_vals = torch.stack([stat.max for stat in stats])
        min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
        max_vals = torch.max(max_vals, torch.zeros_like(max_vals))
        max_vals = torch.max(max_vals, min_vals + minimum_scale * num_steps)
        if symmetric:
            return self._pick_test_candidates_symmetric(min_vals, max_vals, num_steps)
        return self._pick_test_candidates_asymmetric(min_vals, max_vals, num_steps)

    def _pick_test_candidates_asymmetric(self, min_vals, max_vals, num_steps):
        """
        Selects the set of deltas and offsets over which to search for the optimal encodings
        """
        # Note: casting to float32 for two reason:
        #       1) float16 on CPU is not well-supported in pytorch
        #       2) Computing int16 encodings using f16 can result in inf (2 ** 16 - 1 == inf in fp16)
        tensor_kwargs = {"device": min_vals.device, "dtype": torch.float32}
        max_delta = (max_vals - min_vals).to(torch.float32) / num_steps
        observed_offset = torch.round(min_vals / max_delta)
        observed_min = max_delta * observed_offset
        observed_max = observed_min + max_delta * num_steps
        num_deltas = self.asym_delta_candidates
        search_space = torch.arange(start=1, end=(1 + num_deltas), step=1, **tensor_kwargs)
        # test_deltas.shape = (num_histograms, num_tests)
        test_deltas = max_delta[:, None] * search_space[None, :] / (num_deltas - 1)
        # test_offsets.shape = (num_offsets)
        num_offsets = min(num_steps + 2, self.num_offset_candidates)
        test_offset_step = num_steps / (num_offsets - 2) # subtract 2 because we add the observed offset
        test_offsets = torch.round(torch.arange(start=-num_steps, end=test_offset_step, step=test_offset_step, **tensor_kwargs))
        test_offsets = test_offsets[None, :].expand(min_vals.shape[0], -1)
        # Add in the observed offset as a candidate, test_offsets.shape = (num_histograms, num_offsets + 1)
        test_offsets = torch.concat((test_offsets, observed_offset[:, None]), dim=1)
        return self._clamp_delta_offset_values(observed_min, observed_max, num_steps, test_deltas, test_offsets)

    def _pick_test_candidates_symmetric(self, min_vals, max_vals, num_steps):
        """
        Selects the set of deltas over which to search for the optimal symmetric encodings
        """
        minimum_scale = _get_minimum_scale(num_steps)
        tensor_kwargs = {"device": min_vals.device, "dtype": torch.float32}
        max_delta = 2 * torch.max(max_vals, -min_vals).to(torch.float32) / num_steps
        test_offsets = torch.full((1, ), (-num_steps) // 2, **tensor_kwargs)
        num_deltas = self.sym_delta_candidates
        search_space = torch.arange(start=1, end=(1 + num_deltas), step=1, **tensor_kwargs)
        test_deltas = max_delta[:, None] * search_space[None, :] / (num_deltas - 1)
        # test_deltas.shape = (num_histograms, num_deltas, 1)
        # test_offsets.shape = (1, 1, 1)
        min_delta = torch.Tensor([minimum_scale]).to(**tensor_kwargs)
        test_deltas = torch.max(test_deltas, min_delta)
        return test_deltas[:, :, None], test_offsets[:, None, None]

    @staticmethod
    def _clamp_delta_offset_values(min_vals, max_vals, num_steps, test_deltas, test_offsets):
        """
        Clamps delta/offset encodings such that represented range falls within the observed min/max range of inputs
        """
        minimum_scale = _get_minimum_scale(num_steps)

        # test_min shape = (num_histograms, num_deltas, num_offsets)
        test_min = test_deltas[:, :, None] * test_offsets[:, None, :]
        test_max = test_min + test_deltas[:, :, None] * num_steps
        # Clamp min/max to observed min/max
        test_min = torch.max(min_vals[:, None, None], test_min)
        test_max = torch.min(max_vals[:, None, None], test_max)
        # Recompute delta/offset with clamped min/max
        # Returned delta/offset shapes = (num_histograms, num_deltas, num_offsets)
        test_deltas = (test_max - test_min) / num_steps
        min_delta = torch.tensor([minimum_scale], device=test_deltas.device, dtype=test_deltas.dtype)
        test_deltas = torch.max(test_deltas, min_delta)
        test_offsets = torch.round(test_min / test_deltas)
        return test_deltas, test_offsets

    def _select_best_candidates(self, test_deltas, test_offsets, stats, num_steps):
        """
        Searches all pairs of (delta, offset) in test_deltas, test_offsets to find the set with the lowest expected SQNR
        """
        noise = self._estimate_clip_and_quant_noise(stats, test_deltas, test_offsets, num_steps, self.gamma)
        _, min_idx = torch.min(noise.flatten(start_dim=1), dim=1)
        best_delta = torch.gather(test_deltas.flatten(start_dim=1), dim=1, index=min_idx[:, None])
        if test_offsets.numel() == 1:
            best_offset = test_offsets
        else:
            best_offset = torch.gather(test_offsets.flatten(start_dim=1), dim=1, index=min_idx[:, None])
        return best_delta, best_offset

    # pylint: disable=too-many-locals
    @staticmethod
    def _estimate_clip_and_quant_noise(stats: List[_Histogram],
                                       test_deltas: torch.Tensor,
                                       test_offsets: torch.Tensor,
                                       num_steps: int,
                                       gamma: float = 1.0):
        """
        Calculates the error from quantization for each delta, offset pair in test_deltas, test_offsets.
        We approximately reconstruct x from hists by assuming all elements within a given bin fall exactly on the
        midpoint of that bin.

        Args:
            stats (List): A list of _Histogram objects with length equal to the number of encodings to compute
            test_deltas (torch.Tensor): Tensor holding the values of all deltas to search with shape (num_hists, num_deltas, num_offsets)
            test_offsets (torch.Tensor):Tensor holding values of all offsets to search with shape (num_hists, num_deltas, num_offsets)
            num_steps (int): Number of quantization steps, i.e., (2 ** bitwidth) - 1
            gamma (float): Fudge factor to trade off between saturation cost and quantization cost. When gamma=1.0, this approximates the MSE of the quantization function
        """
        tensor_kwargs = {"device": test_deltas.device, "dtype": test_deltas.dtype}
        hists = torch.stack([stat.histogram for stat in stats])
        bin_edges = torch.stack([stat.bin_edges for stat in stats])
        hist_delta = bin_edges[:, 1] - bin_edges[:, 0]
        # hist_midpoints is shape (hists, num_bins)
        hist_offsets = hist_delta[:, None] * torch.arange(0, bin_edges.shape[1] - 1, **tensor_kwargs)[None, :]
        hist_midpoints = (bin_edges[:, 0] + hist_delta/2)[:, None] + hist_offsets
        # hists_midpoints_qdq is shape (hists, num_deltas, num_offsets, num_bins)
        test_offsets_bcast = test_offsets[:, :, :, None]
        test_deltas_bcast = test_deltas[:, :, :, None]
        hist_midpoints_qdq = hist_midpoints[:, None, None, :].div(test_deltas_bcast).sub(test_offsets_bcast).round()
        if gamma != 1.0:
            clipped = torch.logical_or(hist_midpoints_qdq < 0,
                                       hist_midpoints_qdq > num_steps)
        hist_midpoints_qdq = hist_midpoints_qdq.clamp_(0, num_steps).add_(test_offsets_bcast).mul_(test_deltas_bcast)
        square_error = hist_midpoints_qdq.sub_(hist_midpoints[:, None, None, :]).pow_(2).mul_(hists[:, None, None, :])
        if gamma != 1.0:
            # Apply the gamma "fudge factor" to the clipped errors
            square_error = torch.where(clipped, square_error * gamma, square_error)
        return torch.sum(square_error, dim=-1)

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
import torch
import math
import pytest
import numpy as np
import random
from aimet_torch.v2.quantization.affine import quantize_dequantize
from aimet_torch.v2.quantization.encoding_analyzer import (
    SqnrEncodingAnalyzer,
    PercentileEncodingAnalyzer,
    MinMaxEncodingAnalyzer,
    _HistogramObserver,
    _get_minimum_scale,
)

@pytest.fixture(autouse=True)
def set_seed():
    random.seed(999)
    torch.random.manual_seed(999)
    np.random.seed(999)

class TestEncodingAnalyzer():
    @pytest.fixture
    def encoding_analyzers(self):
        min_max_encoding_analyzer = MinMaxEncodingAnalyzer(())
        percentile_encoding_analyzer = PercentileEncodingAnalyzer((), num_bins=3, percentile=99)
        sqnr_encoding_analyzer = SqnrEncodingAnalyzer((1, ))
        encoding_analyzer_list = [min_max_encoding_analyzer, percentile_encoding_analyzer, sqnr_encoding_analyzer]
        yield encoding_analyzer_list

    def test_compute_encodings_with_invalid_num_steps(self, encoding_analyzers):
        for encoding_analyzer in encoding_analyzers:
            encoding_analyzer.update_stats(torch.randn(3, 4))
            with pytest.raises(ValueError):
                encoding_analyzer.compute_encodings(num_steps=0, is_symmetric=False)

    def test_reset_stats(self, encoding_analyzers):
        for encoding_analyzer in encoding_analyzers:
            encoding_analyzer.update_stats(torch.randn(3, 4))
            if isinstance(encoding_analyzer, MinMaxEncodingAnalyzer):
                assert encoding_analyzer.observer.stats.min
                assert encoding_analyzer.observer.stats.max
            else:
                assert all(x.min is not None for x in encoding_analyzer.observer.stats)
                assert all(x.max is not None for x in encoding_analyzer.observer.stats)

            encoding_analyzer.reset_stats()
            if isinstance(encoding_analyzer, MinMaxEncodingAnalyzer):
                assert not encoding_analyzer.observer.stats.min
                assert not encoding_analyzer.observer.stats.max
            else:
                assert all(x.min is None for x in encoding_analyzer.observer.stats)
                assert all(x.max is None for x in encoding_analyzer.observer.stats)

    def test_compute_encodings_with_no_stats(self, encoding_analyzers):
        for encoding_analyzer in encoding_analyzers:
            with pytest.raises(RuntimeError):
               encoding_analyzer.compute_encodings(num_steps=8, is_symmetric=False)

    @pytest.mark.parametrize('dtype', [torch.float, torch.half])
    @pytest.mark.parametrize('symmetric', [True, False])
    @pytest.mark.cuda
    def test_continuity(self, symmetric, dtype, encoding_analyzers):
        for encoding_analyzer in encoding_analyzers:
            normal_range = torch.arange(-128, 128).to(dtype).cuda() / 256
            eps = torch.finfo(dtype).eps

            num_steps = 2**8 - 2
            min_1, max_1 = encoding_analyzer.compute_dynamic_encodings(normal_range * (1 - eps),
                                                                       num_steps=num_steps,
                                                                       is_symmetric=symmetric)
            min_2, max_2 = encoding_analyzer.compute_dynamic_encodings(normal_range,
                                                                       num_steps=num_steps,
                                                                       is_symmetric=symmetric)
            min_3, max_3 = encoding_analyzer.compute_dynamic_encodings(normal_range * (1 + eps),
                                                                       num_steps=num_steps,
                                                                       is_symmetric=symmetric)

            assert min_3 <= min_2 <= min_1 <= max_1 <= max_2 <= max_3
            assert torch.allclose(max_1, max_2, atol=eps)
            assert torch.allclose(min_1, min_2, atol=eps)
            assert torch.allclose(max_2, max_3, atol=eps)
            assert torch.allclose(min_2, min_3, atol=eps)

class TestMinMaxEncodingAnalyzer():
    def test_compute_encodings_asymmetric(self):
        encoding_analyzer = MinMaxEncodingAnalyzer(())
        input_tensor =  torch.arange(start=0, end=26, step=0.5)
        encoding_analyzer.update_stats(input_tensor)

        num_steps = 2**8 - 1
        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                             is_symmetric=False)
        expected_min = torch.zeros_like(asymmetric_min)
        expected_max = torch.full_like(asymmetric_max, 25.5)
        assert torch.allclose(asymmetric_min, expected_min)
        assert torch.allclose(asymmetric_max, expected_max)

    def test_compute_encodings_strict_symmetric(self):
        encoding_analyzer = MinMaxEncodingAnalyzer(())
        input_tensor =  torch.arange(start=0, end=26, step=0.5)
        encoding_analyzer.update_stats(input_tensor)

        num_steps = 2**8 - 2
        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                           is_symmetric=True)
        expected_min = torch.full_like(symmetric_min, -25.5)
        expected_max = torch.full_like(symmetric_max,  25.5)
        assert torch.allclose(symmetric_min, expected_min)
        assert torch.allclose(symmetric_max, expected_max)

    def test_compute_encodings_non_strict_symmetric(self):
        encoding_analyzer = MinMaxEncodingAnalyzer(())
        input_tensor =  torch.arange(start=-2, end=10, step=1, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)
        num_bins = 2**3 - 1
        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(num_steps=num_bins,
                                                                           is_symmetric=True)
        expected_min = torch.full_like(symmetric_min, -12.)
        expected_max = torch.full_like(symmetric_max,  9.)
        assert torch.allclose(symmetric_min, expected_min)
        assert torch.allclose(symmetric_max, expected_max)

    @pytest.mark.parametrize("min_max_size", [[3,4], [2, 3, 1], [4], [1]])
    def test_update_stats_with_different_dimensions(self,min_max_size):
        for _ in range(4):
            encoding_analyzer = MinMaxEncodingAnalyzer(min_max_size)
            encoding_analyzer.update_stats(torch.randn(2, 3, 4))
            assert list(encoding_analyzer.observer.stats.min.shape) == min_max_size
            assert list(encoding_analyzer.observer.stats.max.shape) == min_max_size

    def test_update_stats_incompatible_dimension(self):
        encoding_analyzer = MinMaxEncodingAnalyzer([3, 4])
        with pytest.raises(RuntimeError):
            encoding_analyzer.update_stats(torch.randn(2, 3, 5))

    def test_compute_encodings_with_same_nonzero_tensor(self):
        encoding_analyzer = MinMaxEncodingAnalyzer(())
        encoding_analyzer.update_stats(torch.full((), 3.))

        num_steps = 2**8 - 1
        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                             is_symmetric=False)
        expected_min = torch.zeros_like(asymmetric_min)
        expected_max = torch.full_like(asymmetric_max, 3.)
        assert torch.allclose(asymmetric_min, expected_min)
        assert torch.allclose(asymmetric_max, expected_max)

        symmetric_min , symmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                            is_symmetric=True)
        # Symmetric encodings have one more bins on the negative side
        expected_min = torch.full_like(symmetric_min, -3.) - (symmetric_max / (num_steps // 2))
        expected_max = torch.full_like(symmetric_max,  3.)
        assert torch.allclose(symmetric_min, expected_min)
        assert torch.allclose(symmetric_max, expected_max)

    def test_compute_encodings_with_only_zero_tensor(self):
        encoding_analyzer = MinMaxEncodingAnalyzer(())
        encoding_analyzer.update_stats(torch.zeros(()))

        num_steps = 2**8 - 1
        minimum_scale = torch.tensor(_get_minimum_scale(num_steps))

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                             is_symmetric=False)
        expected_min = torch.zeros_like(asymmetric_min)
        expected_max = torch.full_like(asymmetric_max, minimum_scale * num_steps)
        assert torch.allclose(asymmetric_min, expected_min)
        assert torch.allclose(asymmetric_max, expected_max)

        symmetric_min , symmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                            is_symmetric=True)
        num_pos_bins = math.floor(num_steps / 2)
        num_neg_bins = math.ceil(num_steps / 2)
        expected_min = torch.full_like(symmetric_min, -minimum_scale * num_neg_bins)
        expected_max = torch.full_like(symmetric_max,  minimum_scale * num_pos_bins)
        assert torch.allclose(symmetric_min, expected_min)
        assert torch.allclose(symmetric_max, expected_max)

    @pytest.mark.parametrize('symmetric', [True, False])
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
    def test_overflow(self, symmetric, dtype):
        encoding_analyzer = MinMaxEncodingAnalyzer(())
        input = (torch.arange(10) * torch.finfo(torch.float).tiny)
        encoding_analyzer.update_stats(input)
        num_steps = 2**8 - 2
        minimum_scale = torch.tensor(_get_minimum_scale(num_steps))
        min, max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                       is_symmetric=symmetric)
        scale = (max - min) / 255

        # Scale should be at least as large as torch.min
        assert scale != 0
        assert torch.allclose(scale, minimum_scale, rtol=0.01)

        float_input_max = (torch.arange(10) * torch.finfo(torch.float).max)
        encoding_analyzer.update_stats(float_input_max)
        min_1, max_1 = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                           is_symmetric=symmetric)
        assert torch.all(torch.isfinite(min_1))
        assert torch.all(torch.isfinite(max_1))

class TestHistogramEncodingAnalyzer:
    @pytest.mark.parametrize('num_bins', [-1, 0])
    def test_invalid_bin_value(self, num_bins):
        min_max_shape = ()

        with pytest.raises(ValueError):
            PercentileEncodingAnalyzer(num_bins = num_bins, shape=min_max_shape, percentile  = 99)

        with pytest.raises(ValueError):
            SqnrEncodingAnalyzer(min_max_shape, num_bins)

    @pytest.mark.cuda
    @pytest.mark.parametrize('symmetric', [True, False])
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    @pytest.mark.parametrize('shape', [(4,), (3, 1), (2, 1, 1), (3, 4), (2, 3, 1), (2, 1, 4), (2, 3, 4)])
    def test_compute_encodings_multidimensional(self, symmetric, device, shape):
        x = torch.arange(24, dtype=torch.float).view(2, 3, 4).to(device)
        encoding_analyzer = PercentileEncodingAnalyzer(shape=shape, percentile = 99)
        encoding_analyzer.update_stats(x)
        num_steps = 2**8 - 1
        encoding_min, encoding_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                         is_symmetric=symmetric)
        assert encoding_min.shape == shape
        assert encoding_max.shape == shape
        if device == 'cuda':
            assert encoding_min.is_cuda
            assert encoding_max.is_cuda
        else:
            assert not encoding_min.is_cuda
            assert not encoding_max.is_cuda

    def test_collect_stats_multidimensional(self):
        x = torch.arange(24, dtype=torch.float).view(2, 3, 4)
        shape = (4,)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(4):
            assert torch.equal(histograms[i].min,  x[:,:,i].min())
            assert torch.equal(histograms[i].max,  x[:,:,i].max())

        shape = (3, 1)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(3):
            assert torch.equal(histograms[i].min,  x[:,i,:].min())
            assert torch.equal(histograms[i].max,  x[:,i,:].max())

        shape = (2, 1, 1)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(2):
            assert torch.equal(histograms[i].min,  x[i,:,:].min())
            assert torch.equal(histograms[i].max,  x[i,:,:].max())

        shape = (3, 4)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(12):
            j = i // 4
            k = i % 4
            assert torch.equal(histograms[i].min,  x[:,j,k].min())
            assert torch.equal(histograms[i].max,  x[:,j,k].max())

        shape = (2, 3, 1)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(6):
            j = i // 3
            k = i % 3
            assert torch.equal(histograms[i].min,  x[j,k,:].min())
            assert torch.equal(histograms[i].max,  x[j,k,:].max())

        shape = (2, 1, 4)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(8):
            j = i // 4
            k = i % 4
            assert torch.equal(histograms[i].min,  x[j,:,k].min())
            assert torch.equal(histograms[i].max,  x[j,:,k].max())

        shape = (2, 3, 4)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(24):
            j = i // 12
            k = (i // 4) % 3
            m = i % 4
            assert torch.equal(histograms[i].min,  x[j,k,m].min())
            assert torch.equal(histograms[i].max,  x[j,k,m].max())

    def test_histogram_during_merging(self):
        observer = _HistogramObserver((), num_bins=10)
        input = torch.arange(-50, 51, dtype=torch.float)
        old_histogram = observer.collect_stats(input)
        observer.merge_stats(old_histogram, input)

        input = torch.arange(-50, 51, dtype=torch.float) * 1.5
        new_histogram = observer.collect_stats(input)
        observer.merge_stats(new_histogram, input)

        merged_histogram = observer.stats[0]
        assert list(merged_histogram.histogram) == [10, 15, 25, 25, 25, 25, 25, 26, 15, 11]
        assert list(merged_histogram.bin_edges) == [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]

        #                                       (old_histogram)
        #
        #                   10    10    10    10    10    10    10    10    10    11
        #                 |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
        #                -50 | -40   -30   -20 | -10    0     10 |  20    30    40 |  50
        #                    |        |        |        |        |        |        |
        #                    |        |        |        |        |        |        |
        #                    |        |        |        |        |        |        |
        #              (+5)  | (+15)  | (+15)  | (+15)  | (+15)  | (+15)  | (+16)  |  (+5)
        #      10       10   |   10   |   10   |   10   |   10   |   10   |   10   |   10       11
        #  |--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
        # -75      -60      -45      -30      -15       0        15       30       45       60       75
        #
        #                                       (new_histogram)


class TestPercentileEncodingAnalyzer():
    @pytest.fixture
    def percentile_encoding_analyzer(self):
        return PercentileEncodingAnalyzer(shape=(), num_bins=3, percentile=99)

    def test_compute_encodings_with_same_nonzero_tensor(self, percentile_encoding_analyzer):
        percentile_encoding_analyzer.update_stats(torch.full((), 3.))

        stats = percentile_encoding_analyzer.observer.stats[0]
        cum_sum = torch.cumsum(stats.histogram, dim=0)
        index = torch.searchsorted(cum_sum, torch.quantile(cum_sum, 99/100))
        max_val = stats.bin_edges[index]

        num_steps = 2**8 - 1
        asymmetric_min, asymmetric_max = percentile_encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                                        is_symmetric=False)
        expected_min = torch.zeros_like(asymmetric_min)
        expected_max = torch.full_like(asymmetric_max, max_val)
        assert torch.allclose(asymmetric_min, expected_min)
        assert torch.allclose(asymmetric_max, expected_max)

        symmetric_min , symmetric_max = percentile_encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                                       is_symmetric=True)
        # Symmetric encodings have one more bins on the negative side
        expected_min = torch.full_like(symmetric_min, -max_val) - (symmetric_max / (num_steps // 2))
        expected_max = torch.full_like(symmetric_max,  max_val)
        assert torch.allclose(symmetric_min, expected_min)
        assert torch.allclose(symmetric_max, expected_max)

    def test_compute_encodings_with_only_zero_tensor(self, percentile_encoding_analyzer):
        percentile_encoding_analyzer.update_stats(torch.zeros(()))

        stats = percentile_encoding_analyzer.observer.stats[0]
        cum_sum = torch.cumsum(stats.histogram, dim=0)
        index = torch.searchsorted(cum_sum, torch.quantile(cum_sum, 99/100))
        min_value = stats.bin_edges[index]

        num_steps = 2**8 - 1
        asymmetric_min, asymmetric_max = percentile_encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                                        is_symmetric=False)
        expected_min = torch.full_like(asymmetric_min, min_value)
        expected_max = torch.zeros_like(asymmetric_max)
        assert torch.allclose(asymmetric_min, expected_min)
        assert torch.allclose(asymmetric_max, expected_max)

        symmetric_min , symmetric_max = percentile_encoding_analyzer.compute_encodings(num_steps=num_steps - 1,
                                                                                       is_symmetric=True)
        expected_min = torch.full_like(symmetric_min, min_value)
        expected_max = torch.full_like(symmetric_max, -min_value)
        assert torch.allclose(symmetric_min, expected_min)
        assert torch.allclose(symmetric_max, expected_max)

    @pytest.mark.cuda
    @pytest.mark.parametrize('symmetric', [True, False])
    def test_cuda_inputs(self, percentile_encoding_analyzer, symmetric):
        input_tensor = torch.tensor([2., 3.5, 4.2, 5.])
        percentile_encoding_analyzer.update_stats(input_tensor.cuda())
        num_steps = 2**8 - 1
        percentile_encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                       is_symmetric=symmetric)
        percentile_encoding_analyzer.update_stats(input_tensor.cuda())

        input_tensor_2 = input_tensor * 1.1 - 0.1
        percentile_encoding_analyzer.update_stats(input_tensor_2.cuda())
        percentile_encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                       is_symmetric=symmetric)

        assert percentile_encoding_analyzer.observer.stats[0].histogram.is_cuda
        assert percentile_encoding_analyzer.observer.stats[0].bin_edges.is_cuda

    def test_merge_stats_resize_histogram(self, percentile_encoding_analyzer):
        input_tensor_1 = torch.tensor([2., 3.5, 4.2, 5.])
        percentile_encoding_analyzer.update_stats(input_tensor_1)
        assert len(percentile_encoding_analyzer.observer.stats) == 1
        assert percentile_encoding_analyzer.observer.stats[0].min == 2
        assert percentile_encoding_analyzer.observer.stats[0].max == 5
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].histogram, torch.tensor([1., 1., 2.]))
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].bin_edges, torch.tensor([2., 3., 4., 5.]))

        # update max
        input_tensor_2 = torch.tensor([5.3, 6.4, 7., 8.])
        percentile_encoding_analyzer.update_stats(input_tensor_2)
        assert len(percentile_encoding_analyzer.observer.stats) == 1
        assert percentile_encoding_analyzer.observer.stats[0].min == 2
        assert percentile_encoding_analyzer.observer.stats[0].max == 8
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].histogram, torch.tensor([2., 3., 3.]))
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].bin_edges, torch.tensor([2., 4., 6., 8.]))

        # update min
        input_tensor_3 = torch.tensor([-4.2, 0, 2.3, 4.5])
        percentile_encoding_analyzer.update_stats(input_tensor_3)
        assert len(percentile_encoding_analyzer.observer.stats) == 1
        assert percentile_encoding_analyzer.observer.stats[0].min == -4.2
        assert percentile_encoding_analyzer.observer.stats[0].max == 8
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].histogram, torch.tensor([1, 4, 7]))
        assert torch.allclose(percentile_encoding_analyzer.observer.stats[0].bin_edges, torch.tensor([-4.2, -0.133333, 3.933333, 8.]))

    def test_merge_stats_resize_histogram_with_ambiguous_bins(self, percentile_encoding_analyzer):
        input_tensor_1 = torch.tensor([-4.2, 2.4, 7., 8.])
        percentile_encoding_analyzer.update_stats(input_tensor_1)
        assert len(percentile_encoding_analyzer.observer.stats) == 1
        assert percentile_encoding_analyzer.observer.stats[0].min == -4.2
        assert percentile_encoding_analyzer.observer.stats[0].max == 8
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].histogram, torch.tensor([1., 1., 2.]))
        assert torch.allclose(percentile_encoding_analyzer.observer.stats[0].bin_edges, torch.tensor([-4.2, -0.133333, 3.933333, 8.]))

        input_tensor_2 = torch.tensor([-6.7, -2.5, 7.2, 10.3])
        # hist is [2, 0, 2] for this tensor only
        percentile_encoding_analyzer.update_stats(input_tensor_2)
        assert len(percentile_encoding_analyzer.observer.stats) == 1
        assert percentile_encoding_analyzer.observer.stats[0].min == -6.7
        assert percentile_encoding_analyzer.observer.stats[0].max == 10.3
        '''
        Ambiguity lies when mapping 1st and 3rd num_steps ex: values in [-4.2, -0.133) could map to [-6.7, -1.033) or [-1.033, 4.633)
        '''
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].histogram, torch.tensor([3., 1., 4.]))
        assert torch.allclose(percentile_encoding_analyzer.observer.stats[0].bin_edges, torch.tensor([-6.7, -1.03333,  4.63333, 10.3]))

    def test_merge_stats_resize_histogram_with_bin_splitting(self, percentile_encoding_analyzer):
        input_tensor_1 = torch.tensor([1, 7, 5.3, 6, 5.7, 6.8, 6.2, 2.8, 3.9])
        percentile_encoding_analyzer.update_stats(input_tensor_1)
        assert len(percentile_encoding_analyzer.observer.stats) == 1
        assert percentile_encoding_analyzer.observer.stats[0].min == 1
        assert percentile_encoding_analyzer.observer.stats[0].max == 7
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].histogram, torch.tensor([2., 1., 6.]))
        assert torch.allclose(percentile_encoding_analyzer.observer.stats[0].bin_edges, torch.tensor([1., 3., 5., 7.]))

        input_tensor_2 = torch.tensor([0, 9, 7.8, 2.5, 4.6, 6.2, 8.8])
        percentile_encoding_analyzer.update_stats(input_tensor_2)
        assert len(percentile_encoding_analyzer.observer.stats) == 1
        assert percentile_encoding_analyzer.observer.stats[0].min == 0
        assert percentile_encoding_analyzer.observer.stats[0].max == 9
        # 6 values from the source's histograms 3rd bucket are split in half into the destination's 2nd and 3rd bucket
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].histogram, torch.tensor([4., 5., 7.]))
        assert torch.allclose(percentile_encoding_analyzer.observer.stats[0].bin_edges, torch.tensor([0., 3., 6., 9.]))

    def test_histogram_with_one_bin(self, percentile_encoding_analyzer):
        input_tensor_1 = torch.tensor([1, 7, 5.3, 6, 5.7, 6.8, 6.2, 2.8, 3.9])
        percentile_encoding_analyzer.update_stats(input_tensor_1)
        assert percentile_encoding_analyzer.observer.stats[0].min == 1
        assert percentile_encoding_analyzer.observer.stats[0].max == 7

    def test_handle_inf_inputs(self):
        percentile_encoding_analyzer = PercentileEncodingAnalyzer(shape=(), num_bins=10, percentile=99)
        input_tensor_1 = torch.tensor([1, 7, 5.3, 6, float('inf'), float('inf')])
        percentile_encoding_analyzer.update_stats(input_tensor_1)
        assert percentile_encoding_analyzer.observer.stats[0].min == 1
        assert percentile_encoding_analyzer.observer.stats[0].max == 7
        # 2 inf values are clipped to 7
        assert percentile_encoding_analyzer.observer.stats[0].histogram[9] == 3
        assert torch.allclose(percentile_encoding_analyzer.observer.stats[0].bin_edges,
                                  torch.tensor([1., 1.6, 2.2, 2.8, 3.4, 4., 4.6, 5.2, 5.8, 6.4, 7.]))

        input_tensor_2 = torch.tensor([0, 17, -5, 3, -float('inf'), float('inf'), -float('inf')])
        percentile_encoding_analyzer.update_stats(input_tensor_2)
        assert percentile_encoding_analyzer.observer.stats[0].min == -5
        assert percentile_encoding_analyzer.observer.stats[0].max == 17
        # 1 inf value clipped to 17
        assert percentile_encoding_analyzer.observer.stats[0].histogram[9] == 2
        # 2 neg inf value clipped to -5
        assert percentile_encoding_analyzer.observer.stats[0].histogram[0] == 3
        old_histogram = percentile_encoding_analyzer.observer.stats[0].histogram
        assert torch.allclose(percentile_encoding_analyzer.observer.stats[0].bin_edges,
                                  torch.tensor([-5., -2.8, -0.6,  1.6,  3.8,  6.,  8.2, 10.4, 12.6, 14.8, 17.]))

        input_tensor_3 = torch.tensor([10, -float('inf'), float('inf'), -float('inf')])
        percentile_encoding_analyzer.update_stats(input_tensor_3)
        assert percentile_encoding_analyzer.observer.stats[0].histogram[0] == old_histogram[0] + 2
        assert percentile_encoding_analyzer.observer.stats[0].histogram[-1] == old_histogram[-1] + 1

    def test_merge_stats_without_resizing(self, percentile_encoding_analyzer):
        input_tensor_1 = torch.tensor([2., 3.5, 4.2, 5.])
        percentile_encoding_analyzer.update_stats(input_tensor_1)
        assert len(percentile_encoding_analyzer.observer.stats) == 1
        assert percentile_encoding_analyzer.observer.stats[0].min == 2
        assert percentile_encoding_analyzer.observer.stats[0].max == 5
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].histogram, torch.tensor([1, 1, 2]))
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].bin_edges, torch.tensor([2, 3, 4, 5]))

        # same min, max values
        input_tensor_2 = torch.tensor([2., 3.3, 4.8, 5])
        percentile_encoding_analyzer.update_stats(input_tensor_2)
        assert len(percentile_encoding_analyzer.observer.stats) == 1
        assert percentile_encoding_analyzer.observer.stats[0].min == 2
        assert percentile_encoding_analyzer.observer.stats[0].max == 5
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].histogram, torch.tensor([2, 2, 4]))
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].bin_edges, torch.tensor([2, 3, 4, 5]))

        # min and max within current range
        input_tensor_3 = torch.tensor([3.1, 3.3, 3.7, 3.9])
        percentile_encoding_analyzer.update_stats(input_tensor_3)
        assert len(percentile_encoding_analyzer.observer.stats) == 1
        assert percentile_encoding_analyzer.observer.stats[0].min == 2
        assert percentile_encoding_analyzer.observer.stats[0].max == 5
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].histogram, torch.tensor([2, 6, 4]))
        assert torch.equal(percentile_encoding_analyzer.observer.stats[0].bin_edges, torch.tensor([2, 3, 4, 5]))

    @pytest.mark.parametrize("percentile_value", [-1, 49, 5, 101])
    def test_invalid_percentile_value(self, percentile_value):
        with pytest.raises(ValueError):
            PercentileEncodingAnalyzer((), percentile = percentile_value, num_bins = 3)

    def test_compute_encodings_asymmetric_normalized(self):
        encoding_analyzer = PercentileEncodingAnalyzer((), percentile = 99)
        mean = std_dev = 2
        input_tensor = np.random.normal(mean, std_dev, size=(100000))
        encoding_analyzer.update_stats(torch.from_numpy(input_tensor))
        num_steps = 2**8 - 1
        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                             is_symmetric=False)

        # 99% of the population is within 2 1/2 standard deviations of the mean
        assert asymmetric_min > mean - std_dev * 2.5
        assert asymmetric_max < mean + std_dev * 2.5

    def test_compute_encodings_asymmetric_sequential(self):
        encoding_analyzer = PercentileEncodingAnalyzer((), percentile = 99, num_bins = 500)
        input_tensor =  torch.arange(start=0, end=1001, step=1, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)

        num_steps = 2**8 - 1
        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                             is_symmetric=False)

        # encoding max is the histogram bin edge which contains 99% percentile (990.02)
        assert asymmetric_min == 0
        assert asymmetric_max == 990.0

    def test_compute_encodings_signed_symmetric_normalized(self):
        encoding_analyzer = PercentileEncodingAnalyzer((), percentile = 99, num_bins = 3)
        mean = std_dev = 2
        input_tensor = np.random.normal(mean, std_dev, size=(10000))
        encoding_analyzer.update_stats(torch.from_numpy(input_tensor))

        num_steps = 2**8 - 1
        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                           is_symmetric=True)
        largest_absolute_value = max(abs(element) for element in input_tensor)
        assert symmetric_min > -largest_absolute_value
        assert symmetric_max < largest_absolute_value

    def test_compute_encodings_signed_symmetric_sequential(self):
        encoding_analyzer = PercentileEncodingAnalyzer((), percentile = 99, num_bins = 500)
        input_tensor =  torch.arange(start=0, end=1001, step=1, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)

        num_steps = 2**8 - 2
        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                           is_symmetric=True)
        assert symmetric_min == -990.0
        assert symmetric_max == 990.0

    def test_compute_encodings_non_strict_symmetric(self):
        encoding_analyzer = PercentileEncodingAnalyzer((), percentile = 100, num_bins = 1)
        input_tensor =  torch.arange(start=-2, end=10, step=1, dtype=torch.float)

        encoding_analyzer.update_stats(input_tensor)

        num_steps = 2**3 - 1
        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                                           is_symmetric=True)
        expected_min = torch.full_like(symmetric_min, -12.)
        expected_max = torch.full_like(symmetric_max,  9.)
        assert torch.allclose(symmetric_min, expected_min)
        assert torch.allclose(symmetric_max, expected_max)

    def test_compute_encodings_100_percentile(self):
        encoding_analyzer = PercentileEncodingAnalyzer((), percentile = 100, num_bins = 3)
        mean = std_dev = 2
        input_tensor = torch.randn(10000) * std_dev + mean
        encoding_analyzer.update_stats(input_tensor)

        num_steps = 2**8 - 1
        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps - 1,
                                                                           is_symmetric=True)
        absmax = input_tensor.abs().max()
        assert torch.allclose(symmetric_min, -absmax)
        assert torch.allclose(symmetric_max,  absmax)

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps, is_symmetric=False)
        assert torch.allclose(asymmetric_min, input_tensor.min())
        assert torch.allclose(asymmetric_max, input_tensor.max())

    def test_compute_encodings_50_percentile(self):
        encoding_analyzer = PercentileEncodingAnalyzer((), percentile = 50, num_bins = 3)
        input_tensor =  torch.arange(start=0, end=1001, step=1, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)

        num_steps = 2**8 - 1
        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps - 1, is_symmetric=True)

        stats = encoding_analyzer.observer.stats[0]
        cum_sum = torch.cumsum(stats.histogram, dim=0)
        index = torch.searchsorted(cum_sum, torch.quantile(cum_sum, 50/100))
        mid_value = stats.bin_edges[index]

        assert symmetric_min == -1 * mid_value
        assert symmetric_max == mid_value

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps, is_symmetric=False)
        assert asymmetric_min == 0
        assert asymmetric_max == mid_value

    def test_compute_percentile_encodings_with_outliers(self):
        encoding_analyzer = PercentileEncodingAnalyzer(num_bins=2048, shape=(1, ), percentile=90.)
        input_tensor = torch.arange(start=0, end=101, step=1, dtype=torch.float)
        input_tensor[-1] = 1000
        input_tensor[-2] = 800
        encoding_analyzer.update_stats(input_tensor)
        num_steps = 2**8 - 1
        _, asymmetric_max = encoding_analyzer.compute_encodings(num_steps=num_steps, is_symmetric=False)
        assert 89 < asymmetric_max < 91


class TestSqnrEncodingAnalyzer:

    def test_computed_encodings_uniform_dist(self):
        """
        Given: Update stats on an equally spaced input (no outliers)
        When: Compute encodings
        Then: computed encodings should cover then entire input range
        """
        x = torch.arange(-100, 101) / 100
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(1, ), gamma=1)
        # Expected min/max isn't exactly -1, 1 due to offset rounding
        observed_delta = torch.tensor(2 / 255.)
        observed_offset = torch.round(-1 / observed_delta)
        observed_min = observed_offset * observed_delta
        observed_max = observed_min + 255 * observed_delta
        encoding_analyzer.update_stats(x)
        num_steps = 2**8 - 1
        _min, _max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                         is_symmetric=False)
        assert torch.equal(_min, observed_min.view((1, )))
        assert torch.equal(_max, observed_max.view((1, )))

    def test_computed_encodings_with_outliers(self):
        """
        Given: Update stats on input with a severe outliers
        When: Compute encodings
        Then: Min/max range should be less than the observed min/max
        """
        # Aim for an optimal max_encoding of 3/4 the outlier value
        expected_delta = torch.tensor([0.1])
        outlier_val = (255 * expected_delta) / 0.75
        """
        Some math required to determine the number of non-outliers to make expected_delta optimal:
            Let n = number of non outliers
            Let m = the value of the outlier
            Assume uniform distribution
            MSE_non_outlier = Var(Uniform(0, 0.5 delta)) = delta^2 / 12
            MSE_outlier = (m - 255 * delta)^2 (in unsigned symmetric)
            MSE = n * delta ^ 2 / 12 + (m - 255 * delta)^2
            d_MSE / d_delta = n * delta / 6  + 2 * 255^2 * delta - 510 * m
        Optimal delta at d_MSE / d_delta = 0, solve for n
            n = 12 * 255 * m / delta - 12 * 255 ^ 2
        """
        # In this case, n should evaluate to 210,100
        n = round(12 * 255 * outlier_val.item() / expected_delta.item() - 12 * (255 ** 2))
        encoding_analyzer = SqnrEncodingAnalyzer((1, ), gamma=1.)
        x = torch.rand((1, n)) * 255 * expected_delta
        encoding_analyzer.update_stats(x)
        outlier = torch.tensor([outlier_val]).view(1, 1)
        encoding_analyzer.update_stats(outlier)
        num_steps = 2**8 - 1
        _min, _max = encoding_analyzer.compute_encodings(num_steps=num_steps,
                                                         is_symmetric=False)
        expected_min = torch.zeros_like(_min)
        expected_max = expected_delta * 255
        assert torch.allclose(_min, expected_min)
        assert torch.allclose(_max, expected_max)

    def test_shape(self):
        """
        Given: Encoding analyzer with an arbitrary shape
        When: Compute encodings
        Then: Encodings have shape == encoding_analyzer.shape
        """
        torch.manual_seed(10)
        x = torch.randn(5*3*5*100, dtype=torch.float).view(5, 3, 5, 100)
        shape = (5, 1, 5, 1)

        encoding_analyzer = SqnrEncodingAnalyzer(shape=shape)
        histograms = encoding_analyzer.observer.collect_stats(x)

        for i in range(25):
            assert histograms[i].min == x[i//5, :, i%5, :].min().item()
            assert histograms[i].max == x[i//5, :, i%5, :].max().item()

        best_min, best_max = encoding_analyzer.compute_encodings_from_stats(histograms, 8, False)
        assert best_min.shape == shape
        assert best_max.shape == shape

        encoding_analyzer.update_stats(x/2)
        merged_histograms = encoding_analyzer.observer.get_stats()
        for i in range(25):
            assert merged_histograms[i].min == x[i//5, :, i%5, :].min().item() / 2
            assert merged_histograms[i].max == x[i//5, :, i%5, :].max().item() / 2

        encoding_analyzer.update_stats(x)
        merged_histograms = encoding_analyzer.observer.get_stats()
        for i in range(25):
            assert merged_histograms[i].min == x[i//5, :, i%5, :].min().item()
            assert merged_histograms[i].max == x[i//5, :, i%5, :].max().item()

    def test_clamp_delta_offset_candidates(self):
        """
        Given: A set of delta offset candidates
        When: Some combinations of delta/offset extend beyond the observed min/max ranged
        Then: Clamp the delta values such that the min/max encodings fall within the observed min/max range
        """
        encoding_analyzer = SqnrEncodingAnalyzer(shape=())
        num_steps = 255
        deltas = torch.tensor([[1 / 4, 3 / 4, 5 / 4]])
        offsets = torch.tensor([-255, -128, 0])[None, :]
        should_clamp = torch.tensor([
            [False, False, False],
            [True,  False, True],
            [True,  True,  True],
        ]).to(torch.bool)
        deltas_clamp, offsets_clamp = encoding_analyzer._clamp_delta_offset_values(torch.tensor([-128]),
                                                                                   torch.tensor([127]),
                                                                                   num_steps,
                                                                                   deltas,
                                                                                   offsets)
        assert torch.all(torch.where(should_clamp,
                                     deltas_clamp < deltas[:, :, None],
                                     deltas_clamp == deltas[:, :, None]))
        min_after_clamp = deltas_clamp * offsets_clamp
        max_after_clamp = min_after_clamp + deltas_clamp * num_steps
        assert torch.all(min_after_clamp > -128.5)
        assert torch.all(max_after_clamp < 127.5)

    def test_pick_test_candidates_asymmetric(self):
        """
        Given: An initialized encoding analyzer and min/max observations
        When: Encoder selects the search space for asymmetric delta/offset combinations
        """
        min_obs = torch.tensor([-128])
        max_obs = torch.tensor([127])
        observed_offset = -128
        observed_scale = 1.0
        num_steps = 255
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(), asymmetric_delta_candidates=5, offset_candidates=7)
        deltas, offsets = encoding_analyzer._pick_test_candidates_asymmetric(min_obs,
                                                                             max_obs,
                                                                             num_steps)
        """
        Then: 1) The number of candidates should be equal to num_encodings * num_delta_candidates * num_offset_candidates
        """
        assert deltas.shape == (1, 5, 7)
        assert offsets.shape == (1, 5, 7)
        """
        Then: 2) Unclamped delta values should be equally spaced in the range max_delta * [ 1/(n-1)... n/(n-1) ]
              3) Offset candidates should contain evenly spaced steps between -num_steps and 0
              4) Offset candidates should contain the observed offset
        """
        # Steps of 1 / (5 - 1) = 1 / 4
        for value in [1/4, 1/2, 3/4, 1]:
            assert value in deltas
        # Steps of 255/5 = 51
        for value in [-255, -204, -153, -102, -51, 0]:
            assert value in offsets
        assert observed_offset in offsets
        assert observed_scale in deltas
        """
        Then: 5) None of the candidates should represent values outside of [observed_min, observed_max]
        """
        min_val = torch.round(offsets * deltas)
        max_val = min_val + deltas * num_steps
        # Allow some room for offset rounding
        assert torch.all(min_val > -128.5)
        assert torch.all(max_val < 127.5)

    def test_pick_test_candidates_symmetric(self):
        """
        Given: An initialized encoding analyzer and min/max observations
        When: Encoder selects the search space for symmetric delta/offset combinations
        """
        min_obs = torch.tensor([-100])
        max_obs = torch.tensor([127.5])
        num_steps = 255
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(), symmetric_delta_candidates=5)
        deltas, offsets = encoding_analyzer._pick_test_candidates_symmetric(min_obs, max_obs, num_steps)
        """
        Then: 1) The number of candidates should be equal to num_encodings * num_delta_candidates * num_offset_candidates
        """
        assert deltas.shape == (1, 5, 1)
        """
        Then: 2) Delta values should be equally spaced in the range max_delta * [ 1/(n-1)... n/(n-1) ]
              3) Only offset candidate should be -128
        """
        # Steps of 1 / (5 - 1) = 1 / 4
        for value in [1/4, 1/2, 3/4, 1]:
            assert value in deltas
        assert torch.all(offsets == -128)
        assert offsets.numel() == 1

    def test_estimate_quantization_noise(self):
        """
        Given: 1) A histogram with sufficient granularity
               2) A set of candidate delta/offset encodings
        When: Estimate the total quantization noise for each delta/offset candidate
        Then: The estimated quantization noise should be very close to the actual quantization noise
        """
        # Create random input
        input = torch.randn(2, 1000)
        # Add some outliers
        input[0][0] = 100; input[0][1] = -50
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(2, 1))
        # Get Histogram inputs
        histograms = encoding_analyzer.observer.collect_stats(input)
        # Create a wide range of delta/offsets to search
        deltas = torch.tensor([0.001, 0.1, 0.5, 1.])[None, :, None].expand(2, 4, 6)
        offsets = torch.tensor([-255, -204, -153, -102, -51, 0]).expand_as(deltas)
        # Compute the actual MSE for each encoding candidate:
        delta_bc = deltas[:, :, :, None]
        offset_bc = offsets[:, :, :, None]

        # Broadcast & expand the input to match the shape of delta and offset
        input = input[:, None, None, :].expand(2, 4, 6, -1)

        input_qdq = quantize_dequantize(input, delta_bc, offset_bc, qmin=0, qmax=255)
        q_error_exp = (input - input_qdq).square().sum(dim=-1)
        # Estimate the MSE from the observer histogram
        q_error_est = encoding_analyzer._estimate_clip_and_quant_noise(histograms, deltas, offsets, 255, gamma=1.)
        # Estimated and measured errors should be very close
        assert torch.allclose(q_error_exp, q_error_est, rtol=0.01)

    def test_select_best_delta_offset(self):
        """
        Given: A set of candidate delta offsets and observed histograms
        When: Call encoding_analyzer._select_best_candidates
        Then: Return the (delta, offset) pair which results in the lowest SQNR of the candidates
        """
        # Channels are in range ([-0.5, 0.5], [0, 2])
        input = (torch.rand(2, 1000) - torch.tensor([[0.5], [0.]])) * torch.tensor([[1.], [2.]])
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(2, 1), gamma=1.)
        # Get Histogram inputs
        histograms = encoding_analyzer.observer.collect_stats(input)
        # best delta offset: delta=[1 / 255., 2 / 255.], offset=[-128, 0]
        deltas = torch.tensor([1/510., 1/255., 2/255.])[None, :, None].expand(2, 3, 3)
        offsets = torch.tensor([-200, -128, 0]).expand_as(deltas)
        # Find the best delta/offsets based on the candidates
        best_delta, best_offset = encoding_analyzer._select_best_candidates(deltas, offsets, histograms, 255)
        assert torch.equal(best_delta, torch.tensor([1/255., 2/255.]).view(2, 1))
        assert torch.equal(best_offset, torch.tensor([-128, 0]).view(2, 1))

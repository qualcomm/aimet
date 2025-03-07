# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Common type definitions that are used across aimet """

import io
from enum import Enum
from typing import Union, Callable, Any, Optional, Dict, List
from decimal import Decimal

from aimet_common.layer_database import Layer
from aimet_common import libpymo


class QuantScheme(Enum):
    """Quantization schemes"""
    min_max = 1
    post_training_tf = min_max
    post_training_tf_enhanced = 2

    ### Below are deprecated ###
    training_range_learning_with_tf_init = 3
    training_range_learning_with_tf_enhanced_init = 4
    training_range_learning = 5
    post_training_percentile = 6
    #############################

    @classmethod
    def from_str(cls, alias: str) -> "QuantScheme":
        """
        Returns QuantScheme object from string alias
        """
        try:
            return _quant_scheme_aliases[alias]
        except KeyError as e:
            raise ValueError(
                f"Invalid string literal {alias}"
                f"Expected one of {list(_quant_scheme_aliases.keys())}"
            ) from e


_quant_scheme_aliases = {
    "min_max": QuantScheme.min_max,
    "tf": QuantScheme.min_max,
    "tf_enhanced": QuantScheme.post_training_tf_enhanced,
    "percentile": QuantScheme.post_training_percentile,
}


MAP_QUANT_SCHEME_TO_PYMO = {QuantScheme.post_training_tf: libpymo.QuantizationMode.QUANTIZATION_TF,
                            QuantScheme.post_training_tf_enhanced:
                                libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                            QuantScheme.training_range_learning_with_tf_init:
                                libpymo.QuantizationMode.QUANTIZATION_TF,
                            QuantScheme.training_range_learning_with_tf_enhanced_init:
                                libpymo.QuantizationMode.QUANTIZATION_TF_ENHANCED,
                            QuantScheme.post_training_percentile:
                                libpymo.QuantizationMode.QUANTIZATION_PERCENTILE}
MAP_ROUND_MODE_TO_PYMO = {'nearest': libpymo.RoundingMode.ROUND_NEAREST,
                          'stochastic': libpymo.RoundingMode.ROUND_STOCHASTIC}

RANGE_LEARNING_SCHEMES = {QuantScheme.training_range_learning_with_tf_init,
                          QuantScheme.training_range_learning_with_tf_enhanced_init}


class ActivationType(Enum):
    """ Enums to identify activation type"""
    no_activation = 0
    """ No activation """

    relu = 1
    """ ReLU activation """

    relu6 = 2
    """ ReLU6 activation """

    def __eq__(self, other: "ActivationType"):
        return self.value == other.value and self.name == other.name # pylint: disable=comparison-with-callable


class CostMetric(Enum):
    """ Enumeration of metrics to measure cost of a model/layer """

    mac = 1
    """ MAC: Cost modeled for compute requirements """

    memory = 2
    """ Memory: Cost modeled for space requirements """


class CompressionScheme(Enum):
    """ Enumeration of compression schemes supported in aimet """

    weight_svd = 1
    """ Weight SVD """

    spatial_svd = 2
    """ Spatial SVD """

    channel_pruning = 3
    """ Channel Pruning """


class RankSelectScheme(Enum):
    """ Enumeration of rank selection schemes supported in aimet """

    greedy = 1
    """ Greedy scheme"""


class LayerCompRatioPair:
    """
    Models a pair of (layer: nn.Module, CompRatio: Decimal)
    """

    def __init__(self, layer: Layer, comp_ratio: Union[Decimal, None]):
        """
        Constructor
        :param layer: Reference to layer
        :param comp_ratio: Comp-ratio as a floating point number between 0 and 1
        """
        self.layer = layer
        self.comp_ratio = comp_ratio

    def __str__(self):
        return 'LayerCompRatioPair: layer={}, comp-ratio={}'.format(self.layer.name, self.comp_ratio)


class LayerCompRatioEvalScore:
    """
    Models data element with (layer: nn.Module, CompRatio: Decimal, EvalScore: Decimal) attributes
    """

    def __init__(self, layer: Layer, comp_ratio: Union[Decimal, None], eval_score: Optional[Union[Decimal, None]]):
        """
        Constructor
        :param layer: Reference to layer
        :param comp_ratio: Comp-ratio as a floating point number between 0 and 1
        :param eval_score: Eval score as floating point number
        """
        self.layer = layer
        self.comp_ratio = comp_ratio
        self.eval_score = eval_score

    def __str__(self):
        return 'LayerCompRatioEvalScore: layer={}, comp-ratio={}, eval_score={}'. \
            format(self.layer.name, self.comp_ratio, self.eval_score)


EvalFunction = Callable[[Any, Optional[int], bool], float]


class GreedySelectionParameters:
    """
    Configuration parameters for the Greedy compression-ratio selection algorithm

    :ivar target_comp_ratio: Target compression ratio. Expressed as value between 0 and 1.
            Compression ratio is the ratio of cost of compressed model to cost of the original model.
    :ivar num_comp_ratio_candidates: Number of comp-ratio candidates to analyze per-layer
            More candidates allows more granular distribution of compression at the cost
            of increased run-time during analysis. Default value=10. Value should be greater than 1.
    :ivar use_monotonic_fit: If True, eval scores in the eval dictionary are fitted to a monotonically increasing
            function. This is useful if you see the eval dict scores for some layers are not monotonically increasing.
            By default, this option is set to False.
    :ivar saved_eval_scores_dict: Path to the eval_scores dictionary pickle file that was
            saved in a previous run. This is useful to speed-up experiments when trying
            different target compression-ratios for example. aimet will save eval_scores
            dictionary pickle file automatically in a ./data directory relative to the
            current path. num_comp_ratio_candidates parameter will be ignored when this option is used.
    """

    def __init__(self,
                 target_comp_ratio: float,
                 num_comp_ratio_candidates: int = 10,
                 use_monotonic_fit: bool = False,
                 saved_eval_scores_dict: Optional[str] = None):

        self.target_comp_ratio = target_comp_ratio

        # Sanity check
        if num_comp_ratio_candidates < 2:
            raise ValueError("Error: num_comp_ratio_candidates={}. Need more than 1 candidate for "
                             "Greedy compression-ratio selection".format(num_comp_ratio_candidates))

        self.num_comp_ratio_candidates = num_comp_ratio_candidates
        self.use_monotonic_fit = use_monotonic_fit
        self.saved_eval_scores_dict = saved_eval_scores_dict


class GreedyCompressionRatioSelectionStats:
    """ Statistics for the greedy compression-ratio selection algorithm """

    def __init__(self, eval_scores_dict: Dict[str, Dict[Decimal, float]]):
        """
        Constructor
        :param eval_scores_dict: Dictionary of {layer_name: {compression_ratio: eval_score}}
        """
        self.eval_scores_dictionary = eval_scores_dict

    def __str__(self):
        stream = io.StringIO(newline='\n')
        stream.write('\nGreedy Eval Dict\n')
        layer_dict = self.eval_scores_dictionary
        for layer in layer_dict:
            stream.write('    Layer: {}\n'.format(layer))

            for ratio in sorted(layer_dict[layer]):
                stream.write('        Ratio={}, Eval score={}\n'.format(ratio, layer_dict[layer][ratio]))

        return stream.getvalue()


class TarCompressionRatioSelectionStats:
    """ Statistics for the TAR compression-ratio selection algorithm """

    def __init__(self, layers_comp_ratio_eval_score_per_rank_index):
        """
        Constructor
        :param  layers_comp_ratio_eval_score_per_rank_index: List of [layer_name: compression_ratio: eval_score] params
        """
        self.layers_comp_ratio_eval_score_per_rank_index = layers_comp_ratio_eval_score_per_rank_index

    def __str__(self):
        stream = io.StringIO(newline='\n')
        stream.write('\nTar Eval table\n')
        for data_to_print in self.layers_comp_ratio_eval_score_per_rank_index:
            stream.write('    Layer: {}\n'.format(data_to_print.layer))
            stream.write('        Ratio={}, Eval score={}\n'.format((data_to_print.comp_ratio),
                                                                    (data_to_print.eval_score)))

        return stream.getvalue()


class CompressionStats:
    """ Statistics generated during model compression """

    class LayerStats:
        """ Statistics for every layer in the model that was compressed """

        def __init__(self, name: str, comp_ratio: Decimal):
            self.name = name
            self.compression_ratio = comp_ratio

    def __init__(self, base_accuracy: float, comp_accuracy: float,
                 mem_comp_ratio: Decimal, mac_comp_ratio: Decimal,
                 per_layer_stats: List[LayerStats],
                 comp_ratio_select_stats: Union[GreedyCompressionRatioSelectionStats, None]):

        self.baseline_model_accuracy = format(base_accuracy, '.6f')
        self.compressed_model_accuracy = format(comp_accuracy, '.6f')
        self.memory_compression_ratio = format(mem_comp_ratio, '.6f')
        self.mac_compression_ratio = format(mac_comp_ratio, '.6f')
        self.per_layer_stats = per_layer_stats
        self.compression_ratio_selection_stats = comp_ratio_select_stats

    def __str__(self):

        stream = io.StringIO(newline='\n')
        stream.write('**********************************************************************************************\n')
        stream.write('Compressed Model Statistics\n')
        stream.write('Baseline model accuracy: {}, Compressed model accuracy: {}\n'
                     .format(self.baseline_model_accuracy,
                             self.compressed_model_accuracy))
        stream.write('Compression ratio for memory={}, mac={}\n'.format(self.memory_compression_ratio,
                                                                        self.mac_compression_ratio))
        stream.write('\n')
        stream.write('**********************************************************************************************\n')

        stream.write('\nPer-layer Stats\n')
        for layer in self.per_layer_stats:
            stream.write('    Name:{}, compression-ratio: {}\n'.format(layer.name,
                                                                       layer.compression_ratio))
        stream.write('\n')
        stream.write('**********************************************************************************************\n')

        stream.write('{}\n'.format(self.compression_ratio_selection_stats))
        stream.write('**********************************************************************************************\n')

        return stream.getvalue()


class AdaroundConstants:
    """ Constants used for Adarounding """

    GAMMA = -0.1
    ZETA = 1.1


class QuantizationDataType(Enum):
    """ Enumeration of tensor quantizer data types supported """
    undefined = 0
    int = 1
    float = 2

class SupportedKernelsAction(Enum):
    """ Enumeration to specify the action to apply during supported_kernels validation"""
    allow_error = 1
    warn_on_error = 2
    assert_on_error = 3


class QuantDtypeBwInfo:
    """
    QuantDtypeBwInfo holds activation dtype/bw and param dtype/bw
    """


    def __init__(self, act_dtype: QuantizationDataType, act_bw: int,
                 param_dtype: QuantizationDataType = QuantizationDataType.undefined, param_bw: int = 0):
        """
        Data class to hold dtype and bw info
        :param act_dtype: Activation datatype of type QuantizationDataType
        :param act_bw: Activation bitwidth of type int
        :param param_dtype: Param datatype of type QuantizationDataType
        :param param_bw: Param bitwidth of type int
        """
        self.act_dtype = act_dtype
        self.act_bw = act_bw
        self.param_dtype = param_dtype
        self.param_bw = param_bw
        self._validate_inputs()

    def __repr__(self):
        return f'(activation:({self.act_dtype}, {self.act_bw}) param:({self.param_dtype}, {self.param_bw})'

    def __str__(self):
        return f'activation:({self.act_dtype}, {self.act_bw}) param:({self.param_dtype}, {self.param_bw})'

    def __eq__(self, other):
        return self.act_dtype == other.act_dtype and self.act_bw == other.act_bw and \
               self.param_dtype == other.param_dtype and self.param_bw == other.param_bw

    def _validate_inputs(self):
        """
        Validate inputs
        """
        if self.param_dtype and self.param_bw:
            if self.param_dtype == QuantizationDataType.float and self.param_bw not in [16, 32]:
                raise ValueError(
                    'float param_dtype can only be used when param_bw is set to 16, not ' + str(self.param_bw))

        if self.act_dtype == QuantizationDataType.float and self.act_bw not in [16, 32]:
            raise ValueError(
                'float act_dtype can only be used when act_bw is set to 16, not ' + str(self.act_bw))

    def is_same_activation(self, dtype: QuantizationDataType, bw: int):
        """
        helper function to check if activation of the object is same as input
        :param bw: bitwidth to verify against
        :param dtype: dtype to verify against
        """
        return bw == self.act_bw and dtype == self.act_dtype

    def is_same_param(self, dtype: QuantizationDataType, bw: int):
        """
        helper function to check if param of the object is same as input
        :param bw: bitwidth to verify against
        :param dtype: dtype to verify against
        """
        return bw == self.param_bw and dtype == self.param_dtype

    def get_activation(self) -> tuple:
        """ getter method for activation candidate"""
        return self.act_dtype, self.act_bw

    def get_param(self) -> tuple:
        """ getter method for param candidate"""
        return self.param_dtype, self.param_bw


class CallbackFunc:
    """
    Class encapsulating call back function and it's arguments
    """
    def __init__(self, func: Callable, func_callback_args=None):
        """
        :param func: Callable Function
        :param func_callback_args: Arguments passed to the callable function
        """
        self.func = func
        self.args = func_callback_args

class EncodingType(Enum):
    """ Encoding type """
    PER_TENSOR = 0
    PER_CHANNEL = 1
    PER_BLOCK = 2
    LPBQ = 3
    VECTOR = 4

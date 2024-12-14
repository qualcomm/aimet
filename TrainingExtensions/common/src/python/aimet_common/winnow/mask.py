#  =============================================================================
#
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
#
#  =============================================================================

""" This file contains classes and functions related to input, output channel masks associated with all modules in a
model. The InternalConnectivity and its derived classes encapsulate the internal connectivity of the modules. """

from typing import List, Tuple
from enum import Enum
import abc

from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_common.connected_graph.operation import Op
from aimet_common.utils import AimetLogger, api_channel_index_dict, ModelApi
from aimet_common.winnow.winnow_utils import get_zero_positions_in_binary_mask, OpConnectivity, ConnectivityType, \
    get_conv_ops_for_api, get_linear_ops_for_api


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


class InternalConnectivity(abc.ABC):
    """
    Models the Internal Connectivity for an Op.
    For an Op, the Internal Connectivity defines how masks are propagated
    in the forward (input to output) and
    in the backward (output to input) directions.
    """

    def __init__(self, input_mask_and_length_tuple: List[Tuple[List, int]],
                 output_mask_and_length_tuple: List[Tuple[List, int]]):
        """
        :param input_mask_and_length_tuple: List of Tuples. Each Tuple contains a list of input masks and the mask
            length
        :param output_mask_and_length_tuple: List of Tuples. Each Tuple contains a list of output masks and the mask
            length
        """
        self.initialize_masks(input_mask_and_length_tuple, output_mask_and_length_tuple)

    @staticmethod
    def initialize_masks(input_mask_and_length_tuple: List[Tuple[List, int]],
                         output_mask_and_length_tuple: List[Tuple[List, int]]):
        """
        Initialize input and output masks given lengths of each.

        :param input_mask_and_length_tuple: List of Tuples. Each Tuple contains a list of input masks and the mask
            length.
        :param output_mask_and_length_tuple: List of Tuples. Each Tuple contains a list of output masks and the mask
            length.
        """

        if not input_mask_and_length_tuple and not output_mask_and_length_tuple:
            # SKIP internal Connectivity. Nothing to do.
            return

        num_in_masks = len(input_mask_and_length_tuple)
        for i in range(num_in_masks):
            in_masks, in_mask_length = input_mask_and_length_tuple[i]
            assert in_mask_length > 0
            for _ in range(in_mask_length):
                in_masks.append(1)

        num_out_masks = len(output_mask_and_length_tuple)
        for k in range(num_out_masks):
            out_masks, out_mask_length = output_mask_and_length_tuple[k]
            if out_mask_length > 0:
                for _ in range(out_mask_length):
                    out_masks.append(1)

    @abc.abstractmethod
    def forward_propagate_the_masks(self, input_mask_list: List[List[int]], output_mask_list: List[List[int]]):
        """
        Based on the internal connectivity and input mask(s), updates the output mask(s).

        :param input_mask_list: The input mask(s) to be propagated
        :param output_mask_list: The output mask(s) to be updated based on the Op's Internal Connectivity
        """

    @abc.abstractmethod
    def backward_propagate_the_masks(self, output_mask_list: List[List[int]], input_mask_list: List[List[int]]):
        """
        Based on the internal connectivity and output mask(s), updates the input mask(s).

        :param output_mask_list: The output mask(s) to be propagated
        :param input_mask_list: The input mask(s) to be updated based on the Op's Internal Connectivity
        """


class SkipInternalConnectivity(InternalConnectivity):
    """ Models SKIP internal connectivity for an Op.
    There are many Functional operators used in a model's forward() function.
    These operators have inputs and outputs but may not have in channels and out channels.
    These operators need to be present in the ConnectedGraph so that they can be connected
    with the operators present in the model. During Mask Propagation, these operators are
    not a factor and are skipped over."""

    def forward_propagate_the_masks(self, input_mask_list: List[List[int]], output_mask_list: List[List[int]]):
        """
        Based on the internal connectivity and input mask(s), updates the output mask(s).

        :param input_mask_list: The input mask(s) to be propagated
        :param output_mask_list: The output mask(s) to be updated based on the Op's Internal Connectivity
        """
        # Since internal connectivity is SKIP, nothing needs to be done.

    def backward_propagate_the_masks(self, output_mask_list: List[List[int]], input_mask_list: List[List[int]]):
        """
        Based on the internal connectivity and output mask(s), updates the input mask(s).

        :param output_mask_list: The output mask(s) to be propagated
        :param input_mask_list: The input mask(s) to be updated based on the Op's Internal Connectivity
        """
        # Since internal connectivity is SKIP, nothing needs to be done.


class NullInternalConnectivity(InternalConnectivity):
    """ Models NULL internal connectivity for an Op. """

    def __init__(self, input_mask_and_length_tuple: List[Tuple[List, int]],
                 output_mask_and_length_tuple: List[Tuple[List, int]]):
        """
        :param input_mask_and_length_tuple: List of Tuples. Each Tuple contains a list of input masks and the mask
            length.
        :param output_mask_and_length_tuple: List of Tuples. Each Tuple contains a list of output masks and the mask
            length.
        """

        # The first Conv2d of the model doesn't have the input mask but its output channels can be pruned.
        # For all other Conv2d modules, there is one input mask and one output mask.
        if input_mask_and_length_tuple and output_mask_and_length_tuple:
            assert len(input_mask_and_length_tuple) == len(output_mask_and_length_tuple) == 1

        super().__init__(input_mask_and_length_tuple, output_mask_and_length_tuple)

    def forward_propagate_the_masks(self, input_mask_list: List[List[int]], output_mask_list: List[List[int]]) -> bool:
        """
        Based on the internal connectivity and input mask(s), updates the output mask(s).

        :param input_mask_list: The input mask(s) to be propagated
        :param output_mask_list: The output mask(s) to be updated based on the Op's Internal Connectivity
        """
        # Since internal connectivity is NULL, nothing needs to be done.
        mask_changed = False
        return mask_changed

    def backward_propagate_the_masks(self, output_mask_list: List[List[int]], input_mask_list: List[List[int]]) -> bool:
        """
        Based on the internal connectivity and output mask(s), updates the input mask(s).

        :param output_mask_list: The output mask(s) to be propagated
        :param input_mask_list: The input mask(s) to be updated based on the Op's Internal Connectivity
        """
        # Since internal connectivity is NULL, nothing needs to be done.
        mask_changed = False
        return mask_changed


class DirectInternalConnectivity(InternalConnectivity):
    """ Models DIRECT internal connectivity for an Op. """

    def forward_propagate_the_masks(self, input_mask_list: List[List[int]], output_mask_list: List[List[int]]) -> bool:
        """
        Based on the internal connectivity and input mask(s), updates the output mask(s).

        :param input_mask_list: The input mask(s) to be propagated
        :param output_mask_list: The output mask(s) to be updated based on the Op's Internal Connectivity
        """

        mask_changed = False
        zero_positions = get_zero_positions_in_binary_mask(input_mask_list[0])
        if zero_positions:
            original_out_mask = output_mask_list[0]
            output_mask_list[0] = input_mask_list[0]
            if output_mask_list[0] != original_out_mask:
                mask_changed = True
                logger.debug("Direct Connectivity: Output mask changed from %s to %s.",
                             get_zero_positions_in_binary_mask(original_out_mask),
                             get_zero_positions_in_binary_mask(output_mask_list[0]))
        return mask_changed

    def backward_propagate_the_masks(self, output_mask_list: List[List[int]], input_mask_list: List[List[int]]) -> bool:
        """
        Based on the internal connectivity and output mask(s), updates the input mask(s).

        :param output_mask_list: The output mask(s) to be propagated
        :param input_mask_list: The input mask(s) to be updated based on the Op's Internal Connectivity
        """
        mask_changed = False
        original_in_mask = input_mask_list[0]
        input_mask_list[0] = output_mask_list[0]
        if input_mask_list[0] != original_in_mask:
            mask_changed = True
            logger.debug("Direct Connectivity: Input mask changed from %s to %s.",
                         get_zero_positions_in_binary_mask(original_in_mask),
                         get_zero_positions_in_binary_mask(input_mask_list[0]))
        return mask_changed


class SplitInternalConnectivity(InternalConnectivity):
    """ Models SPLIT internal connectivity for an Op. """

    def forward_propagate_the_masks(self, input_mask_list: List[List[int]], output_mask_list: List[List[int]]) -> bool:
        """
        Based on the internal connectivity and input mask(s), updates the output mask(s).

        :param input_mask_list: The input mask(s) to be propagated
        :param output_mask_list: The output mask(s) to be updated based on the Op's Internal Connectivity
        """
        # Split has one input and multiple outputs.
        # All the output masks are set to the same value as the input mask.

        mask_changed = False
        input_mask = input_mask_list[0]
        # zero_positions = get_zero_positions_in_binary_mask(input_mask)
        # if zero_positions:
        # Propgate the masks only if there are 0 masked channels.
        num_out_masks = len(output_mask_list)
        for i in range(num_out_masks):
            output_mask_list[i] = input_mask
        mask_changed = True
        return mask_changed

    def backward_propagate_the_masks(self, output_mask_list: List[List[int]], input_mask_list: List[List[int]]) -> bool:
        """
        Based on the internal connectivity and output mask(s), updates the input mask(s).

        :param output_mask_list: The output mask(s) to be propagated
        :param input_mask_list: The input mask(s) to be updated based on the Op's Internal Connectivity
        """

        mask_changed = False
        saved_input_mask = input_mask_list[0]
        num_in_masks = len(input_mask_list)
        num_out_masks = len(output_mask_list)
        assert num_in_masks == 1
        combined_mask = None
        for i in range(num_out_masks):
            if combined_mask is None:
                combined_mask = output_mask_list[i]
            else:
                combined_mask = [a or b for a, b in zip(combined_mask, output_mask_list[i])]
        input_mask_list[0] = combined_mask
        if input_mask_list[0] != saved_input_mask:
            mask_changed = True

        return mask_changed


class AddInternalConnectivity(InternalConnectivity):
    """ Models ADD internal connectivity for an Op. """

    def __init__(self, input_mask_and_length_tuple: List[Tuple[List, int]],
                 output_mask_and_length_tuple: List[Tuple[List, int]]):
        """
        :param input_mask_and_length_tuple: List of Tuples. Each Tuple contains a list of input masks and the mask
            length.
        :param output_mask_and_length_tuple: List of Tuples. Each Tuple contains a list of output masks and the mask
            length.
        """

        if len(input_mask_and_length_tuple) < 2:
            # As an intended design strategy, winnower ignores certain types of Ops in ConnectedGraph's parse_xnode().
            # Because of this, in ConnectedGraph's connect_xnode(), an Add Op could have only one input.
            # This happens with teh Inception V3 models.
            # In this scenarios, the Add Op doesn't have an impact on winnowing and is ignored.
            # The Add Op can not be ignored during graph construction since there could be a legitimate Add Op in the
            # model (e.g., ResNet18) where both the inputs are NOT ignored.
            # The Add op is ignored only when it has only one input as detected in this code block.
            return
        assert len(output_mask_and_length_tuple) == 1
        super().__init__(input_mask_and_length_tuple, output_mask_and_length_tuple)

    def forward_propagate_the_masks(self, input_mask_list: List[List[int]], output_mask_list: List[List[int]]) -> bool:
        """
        Based on the internal connectivity and input mask(s), updates the output mask(s).

        :param input_mask_list: The input mask(s) to be propagated
        :param output_mask_list: The output mask(s) to be updated based on the Op's Internal Connectivity
        """

        # The Add Op has multiple inputs and a single output.

        mask_changed = False
        saved_output_mask = output_mask_list[0]
        num_in_masks = len(input_mask_list)
        num_out_masks = len(output_mask_list)
        assert num_out_masks == 1
        combined_mask = None
        for i in range(num_in_masks):
            if combined_mask is None:
                combined_mask = input_mask_list[i]
            else:
                combined_mask = [a or b for a, b in zip(combined_mask, input_mask_list[i])]
        output_mask_list[0] = combined_mask
        if output_mask_list[0] != saved_output_mask:
            mask_changed = True

        return mask_changed

    def backward_propagate_the_masks(self, output_mask_list: List[List[int]], input_mask_list: List[List[int]]) -> bool:
        """
        Based on the internal connectivity and output mask(s), updates the input mask(s).

        :param output_mask_list: The output mask(s) to be propagated
        :param input_mask_list: The input mask(s) to be updated based on the Op's Internal Connectivity
        """

        mask_changed = False
        zero_positions = get_zero_positions_in_binary_mask(output_mask_list[0])
        if zero_positions:
            num_in_masks = len(input_mask_list)
            num_out_masks = len(output_mask_list)
            assert num_out_masks == 1
            for i in range(num_in_masks):
                input_mask_list[i] = output_mask_list[0]
            mask_changed = True

        return mask_changed


class ConcatInternalConnectivity(InternalConnectivity):
    """ Models CONCAT internal connectivity for an Op. """

    def __init__(self, input_mask_and_length_tuple: List[Tuple[List, int]],
                 output_mask_and_length_tuple: List[Tuple[List, int]]):
        """
        :param input_mask_and_length_tuple: List of Tuples. Each Tuple contains a list of input masks and the mask
            length.
        :param output_mask_and_length_tuple: List of Tuples. Each Tuple contains a list of output masks and the mask
            length.
        """
        assert len(input_mask_and_length_tuple) > 1
        assert len(output_mask_and_length_tuple) == 1
        super().__init__(input_mask_and_length_tuple, output_mask_and_length_tuple)

    def forward_propagate_the_masks(self, input_mask_list: List[List[int]], output_mask_list: List[List[int]]):
        """
        Based on the internal connectivity and input mask(s), updates the output mask(s).

        :param input_mask_list: The input mask(s) to be propagated
        :param output_mask_list: The output mask(s) to be updated based on the Op's Internal Connectivity
        """
        assert len(input_mask_list) > 1
        assert len(output_mask_list) == 1

        output_mask_list[0] = [item for input_mask in input_mask_list for item in input_mask]

    def backward_propagate_the_masks(self, output_mask_list: List[List[int]], input_mask_list: List[List[int]]):
        """
        Based on the internal connectivity and output mask(s), updates the input mask(s).

        :param output_mask_list: The output mask(s) to be propagated
        :param input_mask_list: The input mask(s) to be updated based on the Op's Internal Connectivity
        """

        output_mask = output_mask_list[0]
        number_of_zeros_in_output_mask = get_zero_positions_in_binary_mask(output_mask)

        if number_of_zeros_in_output_mask:
            start_pos = 0
            end_pos = 0
            index = 0
            for input_mask in input_mask_list:
                segmented_mask = []
                in_mask_length = len(input_mask)
                end_pos = start_pos + in_mask_length
                for i in range(start_pos, end_pos):
                    segmented_mask.append(output_mask[i])
                start_pos += in_mask_length
                assert len(input_mask) == len(segmented_mask)
                input_mask_list[index] = segmented_mask
                index += 1


class StopInternalConnectivity(InternalConnectivity):
    """ Models STOP internal connectivity for an Op. """

    def forward_propagate_the_masks(self, input_mask_list: List[List[int]], output_mask_list: List[List[int]]) -> bool:
        """
        Based on the internal connectivity and input mask(s), updates the output mask(s).

        :param input_mask_list: The input mask(s) to be propagated
        :param output_mask_list: The output mask(s) to be updated based on the Op's Internal Connectivity
        """
        # Since internal connectivity is STOP, nothing needs to be done.
        mask_changed = False
        return mask_changed

    def backward_propagate_the_masks(self, output_mask_list: List[List[int]], input_mask_list: List[List[int]]) -> bool:
        """
        Based on the internal connectivity and output mask(s), updates the input mask(s).

        :param output_mask_list: The output mask(s) to be propagated
        :param input_mask_list: The input mask(s) to be updated based on the Op's Internal Connectivity
        """
        # Since internal connectivity is STOP, nothing needs to be done.
        mask_changed = False
        return mask_changed


# pylint: disable=too-many-instance-attributes
class Mask:
    """ The Mask class contains properties and functions related to input channel mask
    and output channel mask propagation. """

    class ChannelType(Enum):
        """ Defines the channel types"""

        INPUT = 1   # input channel
        OUTPUT = 2  # output channel

    def __init__(self, op: Op, model_api: ModelApi):
        """
        :param op: The op for which this mask corresponds to
        :param model_api: Either pytorch or tensorflow
        """

        self._num_in_channels = op.num_in_channels
        self._num_out_channels = op.num_out_channels
        self._groups = op.groups
        self._op_type = op.type
        self._dotted_name = op.dotted_name
        self._op_input_ops = op.input_ops
        self._op_output = op.output
        self._model_api = model_api
        self._input_channel_masks = [[] for _ in range(len(self._op_input_ops))]
        if self._op_output:
            self._output_channel_masks = [[] for _ in range(len(self._op_output.consumers))]
        else:
            self._output_channel_masks = None
        self._internal_connectivity = None

        self._set_default_input_output_masks(self._num_in_channels, self._num_out_channels)

    @property
    def internal_connectivity(self) -> InternalConnectivity:
        """ Returns the internal connectivity """
        return self._internal_connectivity

    @property
    def input_channel_masks(self) -> List[List[int]]:
        """ Returns the input channel_mask. """
        return self._input_channel_masks

    def set_input_channel_mask(self, index: int, in_channel_mask: List[int]):
        """ Sets the input channel mask. """
        self._input_channel_masks[index] = in_channel_mask

    @property
    def output_channel_masks(self) -> List[List[int]]:
        """ Returns the output channel_mask. """
        return self._output_channel_masks

    def set_output_channel_mask(self, index: int, out_channel_mask: List[int]):
        """ Sets the output channel mask. """
        original_mask = self._output_channel_masks[index]
        self._output_channel_masks[index] = out_channel_mask
        new_mask = self._output_channel_masks[index]
        if original_mask != new_mask:
            # NOTE: aimet_pytorch uses "Split" to CG_SPLIT to distinguish with aimet_torch._base.nn.modules.custom.Split
            #   After migrating to use CG_SPLIT across other frameworks, we can clean up below line
            split_type = CG_SPLIT if self._model_api == ModelApi.pytorch else "Split"
            if self._op_type == split_type:
                logger.debug("For %s, for output mask index: %s mask changed from %s to %s", self._dotted_name, index,
                             get_zero_positions_in_binary_mask(original_mask), get_zero_positions_in_binary_mask(new_mask))

    def _create_input_output_mask_and_length_tuples(self, num_input_masks: int, input_mask_length: int,
                                                    num_output_masks: int, output_mask_length: int) \
            -> Tuple[List[Tuple[List[int], int]], List[Tuple[List[int], int]]]:
        """
        Create a tuple of input and output mask lists and lengths for each mask list.

        :param num_input_masks: Number of input masks
        :param input_mask_length: Length of input masks
        :param num_output_masks: Number of output masks
        :param output_mask_length: Length of output masks
        :return: Tuple of lists of tuples of input and output masks and their lengths
        """

        input_mask_and_length_tuple_list = []
        output_mask_and_length_tuple_list = []

        for i in range(num_input_masks):
            input_mask_and_length_tuple = (self._input_channel_masks[i], input_mask_length)
            input_mask_and_length_tuple_list.append(input_mask_and_length_tuple)

        for i in range(num_output_masks):
            output_mask_and_length_tuple = (self._output_channel_masks[i], output_mask_length)
            output_mask_and_length_tuple_list.append(output_mask_and_length_tuple)

        return input_mask_and_length_tuple_list, output_mask_and_length_tuple_list

    def _create_masks_list_for_single_input_multi_output_ops(self, in_channels: int) \
            -> Tuple[List[Tuple[List[int], int]], List[Tuple[List[int], int]]]:

        """
        For Ops that have single input and multiple outputs (e.g., Split), set the default mask value
        (all 1s) for the output masks (multiple) based on the output consumer Ops' output shapes.
        Set the default mask value (all 1s) for the input mask (single) based on the current Op's input shape.

        :param in_channels: Number of input channels
        :return: Tuple of lists of tuples of input and output masks and their lengths
        """

        input_masks_list = []
        output_masks_list = []

        # Input masks
        input_mask_length = in_channels
        input_masks_length_tuple = (self._input_channel_masks[0], input_mask_length)
        input_masks_list.append(input_masks_length_tuple)

        # Output masks
        # For Split, the shape of the output masks should be the same as the shape of the input masks.
        # Split Op, broadcasts the input to all the outputs.
        num_output_masks = len(self._op_output.consumers)
        for i in range(num_output_masks):
            output_mask_length = self._op_output.shape[api_channel_index_dict[self._model_api]]
            output_masks_length_tuple = (self._output_channel_masks[i], output_mask_length)
            output_masks_list.append(output_masks_length_tuple)

        return input_masks_list, output_masks_list

    def _create_masks_list_for_multi_input_single_output_ops(self, out_channels: int) \
            -> Tuple[List[Tuple[List[int], int]], List[Tuple[List[int], int]]]:

        """
        For Ops that have multiple inputs and single output (e.g., Add, Concat), set the default mask value
        (all 1s) for the input masks (multiple) based on the input Op's output shapes. Set the default mask value
        (all 1s) for the output mask (single) based on the current Op's output shape.

        :param out_channels: Number of output channels
        :return: Tuple of lists of tuples of input and output masks and their lengths
        """

        input_masks_list = []
        output_masks_list = []

        # Input masks
        for index, input_op in enumerate(self._op_input_ops):
            input_mask_length = input_op.output_shape[api_channel_index_dict[self._model_api]]
            input_masks_length_tuple = (self._input_channel_masks[index], input_mask_length)
            input_masks_list.append(input_masks_length_tuple)

        # Output masks
        if out_channels:
            output_mask_length = out_channels
        else:
            output_mask_length = 0
        output_masks_length_tuple = (self._output_channel_masks[0], output_mask_length)
        output_masks_list.append(output_masks_length_tuple)

        return input_masks_list, output_masks_list

    def _set_default_masks_for_conv_and_linear(self):
        """
        Set the default input and output masks for Conv and Linear modules.
        """
        if self._op_type in get_conv_ops_for_api(self._model_api):
            num_input_masks = len(self._op_input_ops)
            input_mask_length = self._num_in_channels
            if self._op_output:
                num_output_masks = len(self._op_output.consumers)
                output_mask_length = self._num_out_channels
            else:
                num_output_masks = 0
                output_mask_length = 0

            in_mask_length_list, out_mask_length_list = self._create_input_output_mask_and_length_tuples(
                num_input_masks, input_mask_length, num_output_masks, output_mask_length)
            # Group value of 1 represents a normal Conv2d Op which will have Null Connectivity.
            # Group value of anything else represents a depthwise convolution which will have Direct Connectivity
            if self._groups == 1:
                self._internal_connectivity = NullInternalConnectivity(in_mask_length_list, out_mask_length_list)
            else:
                self._internal_connectivity = DirectInternalConnectivity(in_mask_length_list, out_mask_length_list)

        else:
            num_input_masks = len(self._op_input_ops)
            input_mask_length = self._num_in_channels
            if self._op_output:
                num_output_masks = len(self._op_output.consumers)
            else:
                num_output_masks = 0
            output_mask_length = self._num_out_channels

            in_mask_length_list, out_mask_length_list = self._create_input_output_mask_and_length_tuples(
                num_input_masks, input_mask_length, num_output_masks, output_mask_length)
            self._internal_connectivity = NullInternalConnectivity(in_mask_length_list, out_mask_length_list)

    def _set_default_masks_for_direct_connectivity_ops(self, in_channels: int, out_channels: int):
        """
        Set the default input and output masks for Modules that have "Direct" internal connectivity.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """

        num_input_masks = len(self._op_input_ops)

        if not in_channels:
            # If AveragePool is used in the forward function, an additional node called onnx::Pad is generated.
            # onnx::Pad is not considered an Op and it is not added to the Ops list.
            # In this case, the num_input_masks will be zero.
            # Since this is DIRECT internal connectivity, the number of output masks is set to the number of input
            # masks.
            in_channels = out_channels
        input_mask_length = in_channels
        if out_channels:
            output_mask_length = out_channels
            num_output_masks = len(self._op_output.consumers)
        else:
            # out_channels can be none for the last layer
            output_mask_length = 0
            num_output_masks = 0

        in_mask_length_list, out_mask_length_list = self._create_input_output_mask_and_length_tuples(
            num_input_masks, input_mask_length, num_output_masks, output_mask_length)

        self._internal_connectivity = DirectInternalConnectivity(in_mask_length_list, out_mask_length_list)

    def _set_default_masks_for_null_and_stop_connectivity_ops(self, in_channels: int, out_channels: int,
                                                              is_null_connectivity: bool):
        """
        Set the default input and output masks for Modules that have "Null" internal connectivity.

        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        :param is_null_connectivity: True if op is null connectivity, False otherwise
        """

        num_input_masks = len(self._op_input_ops)
        num_output_masks = len(self._op_output.consumers)

        input_mask_length = in_channels
        if out_channels:
            output_mask_length = out_channels
        else:
            output_mask_length = 0

        in_mask_length_list, out_mask_length_list = self._create_input_output_mask_and_length_tuples(
            num_input_masks, input_mask_length, num_output_masks, output_mask_length)

        if is_null_connectivity:
            self._internal_connectivity = NullInternalConnectivity(in_mask_length_list, out_mask_length_list)
        else:
            self._internal_connectivity = StopInternalConnectivity(in_mask_length_list, out_mask_length_list)

    # pylint: disable=too-many-branches
    def _set_default_input_output_masks(self, in_channels: int, out_channels: int):
        """
        Based on the Op type, sets default input and output channel masks.

        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        """

        op_connectivity = OpConnectivity.get_op_connectivity(self._model_api, self._op_type)

        if op_connectivity == ConnectivityType.null:
            if self._op_type in get_conv_ops_for_api(self._model_api) or \
                    self._op_type in get_linear_ops_for_api(self._model_api):
                self._set_default_masks_for_conv_and_linear()
            else:
                self._set_default_masks_for_null_and_stop_connectivity_ops(in_channels, out_channels,
                                                                           is_null_connectivity=True)
        elif op_connectivity == ConnectivityType.direct:
            # Necessary to switch connectivity of padding to null when adjusting channel size since staying at direct
            # connectivity will cause input and output channel sizes to become equal
            if self._model_api == ModelApi.tensorflow and self._op_type in ["Pad", "PadV2", "MirrorPad"] and \
                    in_channels != out_channels:
                self._set_default_masks_for_null_and_stop_connectivity_ops(in_channels, out_channels,
                                                                           is_null_connectivity=True)
            else:
                self._set_default_masks_for_direct_connectivity_ops(in_channels, out_channels)
        elif op_connectivity == ConnectivityType.add:
            in_masks_list, out_masks_list = self._create_masks_list_for_multi_input_single_output_ops(out_channels)
            self._internal_connectivity = AddInternalConnectivity(in_masks_list, out_masks_list)
        elif op_connectivity == ConnectivityType.concat:
            in_masks_list, out_masks_list = self._create_masks_list_for_multi_input_single_output_ops(out_channels)
            self._internal_connectivity = ConcatInternalConnectivity(in_masks_list, out_masks_list)
        elif op_connectivity == ConnectivityType.split:
            in_masks_list, out_masks_list = self._create_masks_list_for_single_input_multi_output_ops(in_channels)
            self._internal_connectivity = SplitInternalConnectivity(in_masks_list, out_masks_list)
        elif op_connectivity == ConnectivityType.skip:
            in_masks_list = []
            out_masks_list = []
            self._internal_connectivity = SkipInternalConnectivity(in_masks_list, out_masks_list)
        elif op_connectivity == ConnectivityType.stop:
            self._set_default_masks_for_null_and_stop_connectivity_ops(in_channels, out_channels,
                                                                       is_null_connectivity=False)
        else:
            error_msg = (f'Unsupported op_type {self._op_type}, dotted {self._dotted_name}, input_ops: '
                         f'{self._op_input_ops}')
            logger.error(error_msg)
            raise NotImplementedError(error_msg)

    def _update_input_output_channels_to_winnow(self, channel_type: ChannelType, total_num_channels: int,
                                                winnow_channels: List[int]):
        """
        Set winnowed channels for the input or output mask.

        :param channel_type: Determines whether mask to be winnowed is input or output
        :param total_num_channels: Total number of channels
        :param winnow_channels: Channel indices to be winnowed
        """

        if winnow_channels:
            if max(winnow_channels) < total_num_channels:
                for k in winnow_channels:
                    if channel_type == Mask.ChannelType.INPUT:
                        self._input_channel_masks[0][k] = 0
                    else:
                        self._output_channel_masks[0][k] = 0
            else:
                logger.error("Max channel number to winnow: %s exceeds the module's max channels: %s",
                             max(winnow_channels), total_num_channels)

    def _update_conv_linear_channels_to_winnow(self, in_channels_total_and_winnow: Tuple[int, List[int]],
                                               out_channels_total_and_winnow: Tuple[int, List[int]]):
        """
        For Conv2d and Linear modules, sets the input and output channel masks.

        :param in_channels_total_and_winnow: Tuple of number of total input channels and list of input channels to
            winnow.
        :param out_channels_total_and_winnow: Tuple of number of total output channels and list of output channels to
            winnow.
        """

        if in_channels_total_and_winnow:
            in_total, in_winnow = in_channels_total_and_winnow
            self._update_input_output_channels_to_winnow(Mask.ChannelType.INPUT, in_total, in_winnow)

        if out_channels_total_and_winnow:
            out_total, out_winnow = out_channels_total_and_winnow
            self._update_input_output_channels_to_winnow(Mask.ChannelType.OUTPUT, out_total, out_winnow)

    def are_masks_unchanged(self):
        """ Return true if all input and output masks are unchanged """
        if self.input_channel_masks:
            for input_mask in self.input_channel_masks:
                if 0 in input_mask:
                    return False

        if self.output_channel_masks:
            for output_mask in self.output_channel_masks:
                if 0 in output_mask:
                    return False

        return True

    def update_channels_to_winnow(self, list_of_zero_in_channels: List[int], list_of_zero_out_channels: List[int]):
        """
        Sets the parameters associated with Mask Propagation

        :param list_of_zero_in_channels: List of in channels to winnow
        :param list_of_zero_out_channels: List of out channels to winnow
        """

        if self._op_type not in get_conv_ops_for_api(self._model_api) and \
                self._op_type not in get_linear_ops_for_api(self._model_api):
            raise ValueError(" Module type %s is not allowed to be winnowed" % self._op_type)

        if self._op_type in get_conv_ops_for_api(self._model_api):
            num_in_channels = self._num_in_channels
            in_channels_total_and_winnow = (num_in_channels, list_of_zero_in_channels)
            num_out_channels = self._num_out_channels
            out_channels_total_and_winnow = (num_out_channels, list_of_zero_out_channels)
        else:
            num_in_channels = self._num_in_channels
            in_channels_total_and_winnow = (num_in_channels, list_of_zero_in_channels)
            num_out_channels = self._num_out_channels
            out_channels_total_and_winnow = (num_out_channels, list_of_zero_out_channels)

        self._update_conv_linear_channels_to_winnow(in_channels_total_and_winnow, out_channels_total_and_winnow)

    def propagate_internal_connectivity_in_channels_to_out_channels(self):
        """ Based on the internal connectivity, propagates the input channel masks to output channel masks"""

        # The first module doesn't have input channel mask
        if self._input_channel_masks:
            if self._internal_connectivity is not None:
                self._internal_connectivity.forward_propagate_the_masks(self._input_channel_masks,
                                                                        self._output_channel_masks)

    def propagate_internal_connectivity_out_channels_to_in_channels(self):
        """ Based on the internal connectivity, propagates the output channel masks to  input channel masks"""

        # The last module doesn't have output channel mask
        if self._output_channel_masks:
            if self._internal_connectivity is not None:
                self._internal_connectivity.backward_propagate_the_masks(self._output_channel_masks,
                                                                         self._input_channel_masks)

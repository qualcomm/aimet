# -*- mode: python -*-
#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2020, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Contains functionality related  to all aspects of propagating the masks. """

from typing import List, Union, Dict

from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_common.connected_graph.operation import Op, determine_preceding_op_input_product_index_in_multi_input_op, \
    determine_succeeding_op_output_product_index_in_multi_output_op
from aimet_common.connected_graph.connectedgraph import ConnectedGraph
from aimet_common.connected_graph.product import Product
from aimet_common.winnow.mask import Mask, NullInternalConnectivity, DirectInternalConnectivity, \
    SplitInternalConnectivity, SkipInternalConnectivity, AddInternalConnectivity, StopInternalConnectivity, \
    ConcatInternalConnectivity
from aimet_common.winnow.winnow_utils import get_zero_positions_in_binary_mask, get_conv_ops_for_api
from aimet_common.utils import AimetLogger, ModelApi, api_channel_index_dict


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


class MaskPropagator:
    """ The MaskPropagator class encapsulates the mask propagation functionality.
     It is responsible for Forward and Backward mask propagation within a module
     as well as inter module mask propagation."""

    def __init__(self, conn_graph: ConnectedGraph, model_api: ModelApi):
        """
        Initialize the MaskPropagator

        :param conn_graph: The computing graph that represents the model.
        """

        self._graph = conn_graph
        self._ops = self._graph.get_all_ops()
        self._products = self._graph.get_all_products()
        self._mask_changed = False
        self._model_api = model_api
        self._op_to_mask_dict = {}

        self._create_masks()

    @property
    def op_to_mask_dict(self) -> Dict[Op, Mask]:
        """ Return the op_to_mask_dict """
        return self._op_to_mask_dict

    def _create_masks(self):
        """ Create masks for each op in the connected graph that leads to a conv op """
        for op in self._ops.values():
            # TODO: Only creating masks for ops that lead to conv ops was only tested for TF. See if the same can be
            # done for pytorch, where we traditionally created masks for all ops.
            if self._model_api == ModelApi.tensorflow:
                if op.type in get_conv_ops_for_api(self._model_api):
                    self._create_masks_for_op_and_all_ancestors(op)
            else:
                self._create_masks_for_op_and_all_ancestors(op)

    def _create_masks_for_op_and_all_ancestors(self, op: Op):
        """
        Create mask for the current op, as well as all ancestors of the op

        :param op: Op to create mask for as well as all ancestor ops
        """
        dfs_queue = [op]

        while dfs_queue:
            current_op = dfs_queue.pop()
            # If current_op already has a mask, it means we have already created masks for it and all ancestors from a
            # conv descendant.
            if current_op in self._op_to_mask_dict:
                continue

            if current_op.inputs:
                input_shape = current_op.inputs[0].shape
                if input_shape:
                    current_op.num_in_channels = input_shape[api_channel_index_dict[self._model_api]]
            if current_op.outputs:
                output_shape = current_op.outputs[0].shape
                if output_shape:
                    current_op.num_out_channels = output_shape[api_channel_index_dict[self._model_api]]
            self._op_to_mask_dict[current_op] = Mask(current_op, self._model_api)
            input_products = current_op.get_input_products()
            if input_products:
                for inp in input_products:
                    # Pytorch input products don't have Ops attached to them, while TF ones do
                    # Check that the input product does indeed have a producer
                    if inp.producer:
                        dfs_queue.append(inp.producer)

    def propagate_masks(self):
        """ Propagate the masks within the module and between the modules. """

        # Print the masks before mask propagation starts.
        self._print_all_ip_op_masks_zero_indices()

        # Increasing the iterations to make sure masks are propagated.
        for n in range(20):

            self._propagate_intra_module_masks()
            logger.debug("After Intra: %s", n)
            self._print_all_ip_op_masks_zero_indices()

            self._propagate_inter_module_masks()
            logger.debug("After Inter: %s", n)
            self._print_all_ip_op_masks_zero_indices()

        # Mask propagation has been completed.
        # Validate and adjust the multi-input and multi-output Ops.
        self._validate_and_adjust_masks_for_multi_input_multi_output_ops()

        logger.debug("After Validating and adjusting masks.")
        self._print_all_ip_op_masks_zero_indices()

    def _propagate_intra_module_masks(self):
        """ Propagate the output channel masks to input channel masks, followed by
        propagating the input channel masks to output channel masks. """

        for op_mask in self._op_to_mask_dict.values():
            op_mask.propagate_internal_connectivity_out_channels_to_in_channels()
            op_mask.propagate_internal_connectivity_in_channels_to_out_channels()

    def _propagate_inter_module_masks(self):
        """ Propagate masks between Ops. In the case of Ops with multiple inputs and/or outputs, masks must be
        propagated through all the branches. """

        for a_product in self._products.values():

            # The Product class represents the following entities in a model.
            # 1) a Tensor between two modules (Ops)
            # 2) an input Tensor
            # 3) a constant
            # 4) a parameter
            # For inter module mask propagation, only Products between two Ops are considered.

            inter_module = a_product.is_inter_module()
            if inter_module and a_product.producer in self._op_to_mask_dict:
                # This Product is between two Ops
                producer = a_product.producer
                # If parent op is stop connectivity, do not propagate mask up
                if isinstance(self._op_to_mask_dict[producer].internal_connectivity, StopInternalConnectivity):
                    continue
                # Look at the Producer Op and the consumer Op of the product and propagate the masks between them.
                consumers = a_product.consumers

                for consumer in consumers:
                    if consumer in self._op_to_mask_dict:
                        consumer_connectivity = self._op_to_mask_dict[consumer].internal_connectivity
                        # If consumer op is stop connectivity, do not propagate mask up
                        if isinstance(consumer_connectivity, StopInternalConnectivity):
                            continue
                        if isinstance(consumer_connectivity, ConcatInternalConnectivity):
                            self._propagate_up_concat_inter_module_masks(consumer, a_product)
                        elif isinstance(consumer_connectivity, AddInternalConnectivity):
                            self._propagate_up_add_masks(consumer, a_product)
                        elif isinstance(consumer_connectivity, SkipInternalConnectivity):
                            # Get the Op's output product's consumer and propagate up that consumer's mask.
                            self._propagate_up_skip_masks(consumer, a_product)
                        else:
                            # Consumers that are not Add or Concat
                            assert isinstance(consumer_connectivity, (DirectInternalConnectivity,
                                                                      NullInternalConnectivity,
                                                                      SplitInternalConnectivity))
                            self._set_inter_module_producer_output_and_consumer_input_mask(consumer, a_product)

    def _validate_and_adjust_masks_for_multi_input_multi_output_ops(self):
        """ For Split, Add and Concat Ops, validate the integrity of the input and output masks.
        Some of the masks might have to be adjusted. """

        for op in (op for op, _ in self._op_to_mask_dict.items()):
            internal_connectivity = self._op_to_mask_dict[op].internal_connectivity
            if isinstance(internal_connectivity, SplitInternalConnectivity):
                self._validate_and_adjust_split_op_masks(op)

            elif isinstance(internal_connectivity, AddInternalConnectivity):
                self._validate_and_adjust_add_op_masks(op, self._model_api)

            elif isinstance(internal_connectivity, ConcatInternalConnectivity):
                self._validate_and_adjust_concat_op_masks(op)

    def _adjust_masks_for_upsample_ops(self):
        """ For tensorflow mask propagation, if any upsample mask has changed, reset it to default mask.
        This will cause downsample/upsample ops to again be inserted before or after the current upsample op.
        Naive inefficient method for dealing with the situation, can look into alternative methods for actually
        winnowing upsample op """
        for op in (op for op, _ in self._op_to_mask_dict.items()):
            if op.type == 'Upsample':
                op_mask = self._op_to_mask_dict[op]
                in_masks, out_masks = op_mask.input_channel_masks, op_mask.output_channel_masks
                # Adjust all input masks
                in_mask = in_masks[0]
                in_mask_length = len(in_mask)
                modified_mask = [1 for _ in range(in_mask_length)]
                op_mask.set_input_channel_mask(0, modified_mask)

                # Adjust the single output mask
                output_mask = out_masks[0]
                out_mask_length = len(output_mask)
                out_modified_mask = [1 for _ in range(out_mask_length)]
                op_mask.set_output_channel_mask(0, out_modified_mask)

    def _print_all_ip_op_masks_zero_indices(self):
        """ Print the input and output channel masks of the Ops.
        Only mask indices for masked channels are printed.
        If a module has a mask with default value (all 1s), it is printed as []
        indicating no channels are masked. """

        for op, op_mask in self._op_to_mask_dict.items():
            ip_mask_zero_positions_list = []
            op_mask_zero_positions_list = []

            ip_masks = op_mask.input_channel_masks
            if ip_masks:
                for num in range(len(ip_masks)):
                    ip_mask_zero_positions = [i for i in range(len(ip_masks[num])) if ip_masks[num][i] == 0]
                    # TODO: remove 'Add', 'Concat' when old CG is gone
                    if (op.type in ('Add', 'Concat', 'Split', 'add', 'cat', CG_SPLIT) and
                            self._model_api == ModelApi.pytorch) or \
                            (op.type in ('Add', 'ConcatV2', 'branch') and self._model_api == ModelApi.tensorflow):
                        ip_mask_zero_positions_list.append(ip_mask_zero_positions)
                    else:
                        if ip_mask_zero_positions:
                            ip_mask_zero_positions_list.append(ip_mask_zero_positions)

            op_masks = op_mask.output_channel_masks
            if op_masks:
                for num in range(len(op_masks)):
                    op_mask_zero_positions = [i for i in range(len(op_masks[num])) if op_masks[num][i] == 0]
                    # TODO: remove 'Add', 'Concat' when old CG is gone
                    if (op.type in ('Add', 'Concat', 'Split', 'add', 'cat', CG_SPLIT) and
                            self._model_api == ModelApi.pytorch) or \
                            (op.type in ('Add', 'ConcatV2', 'branch') and self._model_api == ModelApi.tensorflow):
                        op_mask_zero_positions_list.append(op_mask_zero_positions)
                    else:
                        if op_mask_zero_positions:
                            op_mask_zero_positions_list.append(op_mask_zero_positions)

            # Log only if either input_masks or output masks are non-empty
            if ip_mask_zero_positions_list or op_mask_zero_positions_list:
                logger.debug("Op: %s ip mask zero indices: %s, op mask zero indices: %s",
                             op.dotted_name, ip_mask_zero_positions_list, op_mask_zero_positions_list)

    def get_ops_with_non_default_ip_op_masks(self) -> List[Op]:
        """ Returns a list of Ops whose input and/or output channel default masks have been modified. """

        list_of_mask_modified_ops = []

        for op, op_mask in self._op_to_mask_dict.items():
            check_op = False
            if self._model_api == ModelApi.pytorch and op.type in ('Dropout', 'Relu', 'ReLU', 'MaxPool', 'MaxPool2d',
                                                                   'AveragePool', 'Neg', 'BatchNorm2d',
                                                                   'Conv', 'Conv2d', 'Conv2D', 'ConvTranspose',
                                                                   'BatchNormalization'):
                check_op = True
            elif self._model_api == ModelApi.tensorflow:
                # marking any changed op as a modified op for tensorflow
                check_op = True

            if check_op:
                ip_masks, op_masks = op_mask.input_channel_masks, op_mask.output_channel_masks
                modified = False
                for ip_mask in ip_masks:
                    in_zero_channels = get_zero_positions_in_binary_mask(ip_mask)
                    if in_zero_channels:
                        modified = True
                        continue

                # None of the input masks have been modified. Check the output masks.
                if op_masks:
                    for _op_mask in op_masks:
                        out_zero_channels = get_zero_positions_in_binary_mask(_op_mask)
                        if out_zero_channels:
                            modified = True
                            continue
                if modified:
                    list_of_mask_modified_ops.append(op)

        return list_of_mask_modified_ops

    def _is_module_reshape_needed(self, op: Op):
        """
        Tells whether the module requires reshaping during winnowing.

        :param op: Determine if this Op is in reshape scenario
        :return: True, if the module requires reshaping.
                 False, if the module doesn't require reshaping.
        """

        # Look at the Op's input Op. If any of them have multiple inputs and/or outputs,
        # then module requires reshaping. If the previous Op is a single input, single output Op,
        # check if it is a Conv Op. If Conv Op, then module doesn't require reshaping. If it is
        # not a Conv, Op keep looking up the next input Op, until a Conv module or a multi-input/output Op
        # is reached.

        # Conv module has one input Op.
        input_product = op.inputs[0]
        input_op = input_product.producer
        # TODO: remove 'Add', 'Concat' when old CG is gone
        if (input_op.type in ('Add', 'Concat', 'Split', 'add', 'cat', CG_SPLIT) and self._model_api == ModelApi.pytorch) or \
            (input_op.type in ('Add', 'ConcatV2', 'branch', 'Upsample', 'Downsample') and self._model_api ==
             ModelApi.tensorflow) or isinstance(self._op_to_mask_dict[input_op].internal_connectivity,
                                                StopInternalConnectivity):
            logger.debug("Op: %s, below: %s", op.dotted_name, input_op.dotted_name)
            return True
        # TODO: remove 'Conv2d' when old CG is gone
        if (input_op.type in ['Conv', 'ConvTranspose'] and self._model_api == ModelApi.pytorch) or \
           (input_op.type in 'Conv2D' and self._model_api == ModelApi.tensorflow):
            logger.debug("Op: %s, below: %s", op.dotted_name, input_op.dotted_name)
            return False
        return self._is_module_reshape_needed(input_op)

    def _set_inter_module_producer_output_and_consumer_input_mask(self, consumer_op: Op, input_product: Product):
        """
        Set the product's producer op's output mask and the product's consumer op's output mask.

        :param consumer_op: Consumer op whose input mask will be set
        :param input_product: Product with consumer op as output
        """

        producer = input_product.producer
        producer_mask = self._op_to_mask_dict[producer]
        producer_out_masks = producer_mask.output_channel_masks
        consumer_op_mask = self._op_to_mask_dict[consumer_op]
        consumer_in_masks = consumer_op_mask.input_channel_masks

        consumer_mask_index = None
        producer_mask_index = None
        # Determine the consumer mask index
        num_consumer_in_masks = len(consumer_in_masks)
        if num_consumer_in_masks == 1:
            consumer_mask_index = 0
        elif num_consumer_in_masks > 1:
            consumer_mask_index = determine_preceding_op_input_product_index_in_multi_input_op(producer, consumer_op)
        else:
            logger.error("Number of input masks for Op: %s is None", consumer_op.dotted_name)

        # Determine the producer mask index
        num_producer_out_masks = len(producer_out_masks)
        if num_producer_out_masks == 1:
            producer_mask_index = 0
        elif num_producer_out_masks > 1:
            producer_mask_index = determine_succeeding_op_output_product_index_in_multi_output_op(consumer_op, producer)
        else:
            logger.error("Number of output masks for Product: %s is None", input_product.name)

        # Create the connection mask and set the Producer Op's output mask and the Consumer Op's input mask.
        connection_mask = producer_out_masks[producer_mask_index] and consumer_in_masks[consumer_mask_index]
        producer_mask.set_output_channel_mask(producer_mask_index, connection_mask)
        logger.debug("Connection propagation: Op: %s, Product: %s, number of producer masks: %s, "
                     "number of consumer masks: %s, Connection mask: %s",
                     consumer_op.dotted_name,
                     input_product.name,
                     len(producer_out_masks),
                     len(consumer_in_masks),
                     get_zero_positions_in_binary_mask(connection_mask))

    def _propagate_up_concat_inter_module_masks(self, concat_op: Op, input_product: Product):
        """
        Concat Op has multiple inputs. The input index is maintained in the same order in which the inputs were
        mentioned in the torch.cat operation in the forward() function of the model. Look at the number of input
        channels associated with each input and propagate up only the masks for those channels.

        :param concat_op: The Concat Op for which the mask associated with the input_product must be propagated up.
        :param input_product:  One of the products for which the Concat Op is the consumer. The corresponding mask must
        be propagated through this product to it's Producer Op.
        """

        logger.debug("Propagate up concat: For Concat Op: %s, all input product names: %s", concat_op.dotted_name,
                     [input_product.name for input_product in concat_op.inputs])
        concat_op_mask = self._op_to_mask_dict[concat_op]

        # Need only input masks for propagating up to the previous op. Ignore the output mask.
        concat_in_masks = concat_op_mask.input_channel_masks

        logger.debug("Propagate up concat: Processing input product: %s. Concat's input mask lengths: %s",
                     input_product.name, [len(concat_mask) for concat_mask in concat_in_masks])

        # For the Concat Op, look at all the input Ops and find the input Op that matches with this specific input
        # product.
        for concat_input_op_index, input_op in enumerate(concat_op.input_ops):
            logger.debug("Propagate up concat: input Op: %s, Concat Op's index for this input op: %s, mask length: %s",
                         input_op.dotted_name, concat_input_op_index, len(concat_in_masks[concat_input_op_index]))

            if input_product.producer.dotted_name == input_op.dotted_name:
                logger.debug("Propagate up concat: Matching Product: %s with input_op: %s", input_product.name,
                             input_op.dotted_name)

                for product_consumer_index, consumer in enumerate(input_product.consumers):
                    if consumer.dotted_name == concat_op.dotted_name:
                        logger.debug("Propagate up concat: Input op's index for the Concat Op: %s",
                                     product_consumer_index)

                        # For the input Op, look at only the Output mask. That is the one going to be over written
                        # during this inter module mask propagation.
                        input_op_mask = self._op_to_mask_dict[input_product.producer]

                        input_producer_out_masks = input_op_mask.output_channel_masks
                        connection_mask = input_producer_out_masks[product_consumer_index] and \
                                          concat_in_masks[concat_input_op_index]

                        if input_product.producer.type in 'Split' or input_product.producer.type in CG_SPLIT:
                            logger.debug("Not propagating masks from Concat: %s to Split: %s", concat_op.dotted_name,
                                         input_product.producer.dotted_name)
                            mask_length = len(concat_in_masks[concat_input_op_index])
                            modified_mask = [1 for _ in range(mask_length)]
                            # concat_op_mask.set_input_channel_mask(product_consumer_index, modified_mask)
                            concat_op_mask.set_input_channel_mask(concat_input_op_index, modified_mask)

                        else:
                            input_op_mask.set_output_channel_mask(product_consumer_index, connection_mask)
                            concat_op_mask.set_input_channel_mask(concat_input_op_index, connection_mask)
                        # No need to check other consumers.
                        break

        logger.debug("Propagate up concat: Completed processing input product: %s, input mask lengths: %s",
                     input_product.name, [len(concat_mask) for concat_mask in concat_in_masks])

    def _propagate_up_add_masks(self, add_op: Op, product: Product):
        """
        Add has multiple inputs (i.e., input Products). If an input Product is originating from a Split Op,
        do not propagate the mask up through that Product. This function is being called once for each one
        of the Add's input Products.

        :param add_op: The Add op for which masks are getting propagated.
        :param product: The product through which masks are considered to be propagated.
        """
        logger.debug("propagate_up_add_masks: Add's inputs: %s", [product.name for product in add_op.inputs])

        for index, a_product in enumerate(add_op.inputs):
            # get the product.
            # look at the product shape[1]
            # Propagate only those channel masks up.
            if a_product.producer is not None and a_product.producer.dotted_name == product.producer.dotted_name:
                if isinstance(self._op_to_mask_dict[a_product.producer].internal_connectivity,
                              SplitInternalConnectivity):
                    add_op_mask = self._op_to_mask_dict[add_op]
                    logger.debug("Not propagating to Split. Restoring mask to default value.")
                    input_masks = add_op_mask.input_channel_masks
                    mask_length = len(input_masks[index])
                    modified_mask = [1 for _ in range(mask_length)]
                    add_op_mask.set_input_channel_mask(index, modified_mask)
                else:
                    self._set_inter_module_producer_output_and_consumer_input_mask(add_op, product)

    def _propagate_up_skip_masks(self, skip_op: Op, product: Product):
        """
        Propagate up mask from skip op's child op to skip op's parent op.

        :param skip_op: The Skip op for which masks are getting propagated.
        :param product: The product through which masks are considered to be propagated.
        """

        if skip_op.outputs:
            skip_consumer_op = skip_op.output_ops[0]

            producer = product.producer
            producer_mask = self._op_to_mask_dict[producer]
            producer_out_masks = producer_mask.output_channel_masks
            consumer_op_mask = self._op_to_mask_dict[skip_consumer_op]
            consumer_in_masks = consumer_op_mask.input_channel_masks

            # The producer could be a multi-output producer. Determine which output mask (index) should be used.
            producer_mask_index = determine_succeeding_op_output_product_index_in_multi_output_op(skip_op, producer)

            # In the case of Skip Op, there is only one consumer.
            consumer_mask_index = 0

            # Create the connection mask and set the Producer Op's output mask and the Consumer Op's input mask.
            connection_mask = producer_out_masks[producer_mask_index] and consumer_in_masks[consumer_mask_index]
            producer_mask.set_output_channel_mask(producer_mask_index, connection_mask)

    def _validate_and_adjust_concat_op_masks(self, op: Op):
        """
        Check if the concatenation of the multiple input masks yield the single output mask.
        If the check fails, adjust the masks.

        :param op: The Concat Op for which masks are validated and adjusted.
        """

        op_mask = self._op_to_mask_dict[op]
        in_masks, out_masks = op_mask.input_channel_masks, op_mask.output_channel_masks

        # Adjust all input masks
        index = 0
        for in_mask in in_masks:
            in_mask_length = len(in_mask)
            modified_mask = [1 for _ in range(in_mask_length)]
            op_mask.set_input_channel_mask(index, modified_mask)
            index += 1

        # Adjust the single output mask
        output_mask = out_masks[0]
        out_mask_length = len(output_mask)
        out_modified_mask = [1 for _ in range(out_mask_length)]
        op_mask.set_output_channel_mask(0, out_modified_mask)

    def _validate_and_adjust_split_op_masks(self, op: Op):
        """
        This function is called as a final step during mask propagation.
        Make sure Split Op's input mask and output masks are the same.
        Propagate the masks downstream so that Ops downstream have teh updated mask.

        :param op: the Split Op for which masks are validated and adjusted.
        """

        op_mask = self._op_to_mask_dict[op]
        in_masks, out_masks = op_mask.input_channel_masks, op_mask.output_channel_masks

        # Split Op has one input and multiple output masks
        input_mask = in_masks[0]

        # set all the output masks to the same value as the input mask.
        for index in range(len(out_masks)):
            op_mask.set_output_channel_mask(index, input_mask)

        # The output masks of the split have been adjusted. Now this new mask must be propagated down
        # to Ops further down. This is done so that while reducing Conv modules a local decision
        # could be made based the module above the Conv Op. For this reason, we shouldn't adjust
        # the Conv Op's masks. From Add and Concat Ops, the masks are not propagated to Split Op
        # as this considered as a special-Op to special-Op.
        for consumer in op.output_ops:
            while consumer in self._op_to_mask_dict and \
                    isinstance(self._op_to_mask_dict[consumer].internal_connectivity, DirectInternalConnectivity):
                self._op_to_mask_dict[consumer].set_input_channel_mask(0, input_mask)
                if not consumer.outputs:
                    break
                self._op_to_mask_dict[consumer].set_output_channel_mask(0, input_mask)
                logger.debug("Masks adjusted for: %s, %s", consumer.dotted_name, consumer.type)
                consumer = consumer.output_ops[0]

    def _validate_and_adjust_add_op_masks(self, op: Op, model_api: ModelApi):
        """
        Check if the Add Op's all input masks are the same.
        If not, adjust the masks to default values.

        :param op: the Add Op for which masks are validated and adjusted.
        :param model_api: either tensorflow or pytorch
        """

        op_mask = self._op_to_mask_dict[op]
        in_masks, out_masks = op_mask.input_channel_masks, op_mask.output_channel_masks
        mask_length = out_masks[0]

        # The Add Op has multiple inputs and a single output.

        # Check for number of zero positions in the masks? Not necessary.
        # For Add Op, during Intra-Module back propagation of the masks,
        # the output mask is propagated to all the input masks.
        # In the ideal case, the input masks must be the same.
        # When the Add is connected to Split, during Inter-module mask propagation,
        # the Add's mask is NOT propagated to Split and the corresponding mask is set
        # to default value of all 1s (no masking).

        if all(mask == in_masks[0] for mask in in_masks):
            logger.debug("Valid masks for Add Op: %s", op.dotted_name)
            return

        # Reset the all the input masks and the output mask to default value.
        modified_mask = [1 for _ in range(len(mask_length))]

        # Set the input channel masks
        for index in range(len(in_masks)):
            op_mask.set_input_channel_mask(index, modified_mask)

        # Set the output channel mask. For Add, there is only one output mask.
        out_index = 0
        op_mask.set_output_channel_mask(out_index, modified_mask)
        logger.debug("Invalid masks for Add Op: %s", op.dotted_name)

        # Update downstream Ops' masks as long the Op is not a Conv.
        # This step is essential so that for the Conv Op, only the previous Op's
        # output mask is checked to make the local decision (whether a DownSampleLayer
        # need to be prepended to the Conv.
        downstream_op = op.output_ops[0]
        self._adjust_downstream_op_masks(downstream_op, modified_mask, model_api)

    def _adjust_downstream_op_masks(self, downstream_op: Op, modified_mask: List[int], model_api: ModelApi):
        """
        Starting with the downstream_op, adjust the input and output masks for the Ops until a Conv Op is reached.
        :param downstream_op: the starting downstream op
        :param modified_mask: the mask to be set for the downstream Ops
        :param model_api: either tensorflow or pytorch
        """
        if downstream_op.type not in get_conv_ops_for_api(model_api):
            downstream_op_mask = self._op_to_mask_dict[downstream_op]
            if isinstance(self._op_to_mask_dict[downstream_op].internal_connectivity, SplitInternalConnectivity):
                # Downstream Op has single input and multiple outputs.
                downstream_op_mask.set_input_channel_mask(0, modified_mask)
                downstream_out_masks = downstream_op_mask.output_channel_masks
                num_out_masks = len(downstream_out_masks)
                for index in range(num_out_masks):
                    downstream_op_mask.set_output_channel_mask(index, modified_mask)
                    self._adjust_downstream_op_masks(downstream_op.output_ops[index], modified_mask, model_api)
            elif not isinstance(self._op_to_mask_dict[downstream_op].internal_connectivity,
                                StopInternalConnectivity):
                # Downstream Op has single input and single output.
                downstream_op_mask.set_input_channel_mask(0, modified_mask)
                downstream_op_mask.set_output_channel_mask(0, modified_mask)
                logger.debug("Masks adjusted for: %s", downstream_op.dotted_name)
                if downstream_op.output_ops:
                    self._adjust_downstream_op_masks(downstream_op.output_ops[0], modified_mask, model_api)
            else:
                # Stop propagating downstream if we hit a stop connectivity op
                return

    def update_channels_to_winnow(self, name: str, reshape: bool, input_channels_to_winnow: Union[None, List[int]],
                                  output_channels_to_winnow: Union[None, List[int]]):
        """ For the Given Op, update the channels to be winnowed.

        :param name: Name of module to winnow to search in ConnectedGraph
        :param reshape: If set to False, UpSampleLayers and DownSampleLayers will not be used in the winnowed model.
                        If set to True, UpSampleLayers and DownSampleLayers will be used in the winnowed model.
        :param input_channels_to_winnow: List of input channels to winnow
        :param output_channels_to_winnow: List of output channels to winnow (currently not supported)
        """

        module_op = self._graph.get_op_from_module_name(name)
        if module_op:
            if reshape:
                # DownSampleLayers and UpSampleLayers can be added as needed.
                self._op_to_mask_dict[module_op].update_channels_to_winnow(input_channels_to_winnow,
                                                                           output_channels_to_winnow)
            else:
                # Determine if the OP is right below a Split, Add or Concat.
                # If yes, do not update the channels to winnow.
                reshape_needed = self._is_module_reshape_needed(module_op)
                if reshape_needed:
                    logger.debug("Reshape flag set to False. Module :%s will not be winnowed.",
                                 module_op.dotted_name)
                else:
                    self._op_to_mask_dict[module_op].update_channels_to_winnow(input_channels_to_winnow,
                                                                               output_channels_to_winnow)
        else:
            logger.error(" Update channels to winnow: module_op is None for: %s", name)
            raise RuntimeError("For the module, an Op was not found in the ConnectedGraph:", name)

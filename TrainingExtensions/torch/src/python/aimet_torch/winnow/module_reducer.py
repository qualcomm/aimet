# -*- mode: python -*-
#  =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Contains functionality related to reducing a module.  """

from typing import List
import torch
from aimet_common.utils import AimetLogger, ModelApi
from aimet_common.winnow.winnow_utils import get_zero_positions_in_binary_mask, get_conv_ops_for_api, get_indices_among_ones_of_overlapping_ones
from aimet_common.winnow.module_reducer import ModuleReducer as AimetCommonModuleReducer
from aimet_common.connected_graph.operation import Op as Operation
from aimet_common.connected_graph.operation import determine_preceding_op_input_product_index_in_multi_input_op, \
    determine_succeeding_op_output_product_index_in_multi_output_op
from aimet_common.polyslice import PolySlice
from aimet_torch.winnow.winnow_utils import UpsampleLayer, DownsampleLayer
from aimet_torch.utils import get_one_positions_in_binary_mask, is_leaf_module
from aimet_torch.winnow.winnow_utils import reduce_tensor

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Winnow)


class ModuleReducer(AimetCommonModuleReducer):
    """ Class responsible for reducing PyTorch modules. """

    def __init__(self, model, using_cuda, reshape, op_to_mask_dict: dict):
        """
        ModuleReducer initialization.

        :param model: The model for which modules are being reduced.
        :param using_cuda: if True, indicates that the model is on GPU.
        :param reshape: If set to True, add DownSampleLayer/UpSampleLayer  module as needed.
                        If set to False, do not add ownSampleLayer/UpsampleLayer module.
        :param op_to_mask_dict: Dictionary mapping Op to mask
        """

        super().__init__(using_cuda, reshape, op_to_mask_dict)
        self._model = model
        self._parent_module_ref = {}

        self._generate_parent_ref(self._model)

    def _generate_parent_ref(self, module: torch.nn.Module):
        """
        For all the modules in a model, generate the parent reference.

        :param module: Create parent references for all the modules in this model.
        :return:
        """
        for module_name, module_ref in module.named_children():
            # first check if the module is leaf module or not
            if is_leaf_module(module_ref):
                # iterate over all the layer attributes and if the match is found
                # then set the parent class and module name for that module
                self._parent_module_ref[module_ref] = module, module_name
            # if module is not leaf, call recursively
            else:
                self._generate_parent_ref(module_ref)

    def reduce_modules(self, list_of_ops_to_reduce: List):
        """
        For the Ops in the list,  reduce he corresponding modules.
        Reduce includes reducing the parameter tensors associated with the module
        as well as prepending/appending DownSample/UpSample layer to the module.

        :param list_of_ops_to_reduce: list of Ops whose associated modules need to be reduced.
        :return: dictionary mapping names of reduced modules to the modules themselves
        """

        modified_modules = {}

        for an_op in list_of_ops_to_reduce:

            if an_op.type in get_conv_ops_for_api(ModelApi.pytorch):
                a_conv_module = self._reduce_conv_module(an_op)
                modified_modules[an_op.dotted_name] = a_conv_module
            elif an_op.type in ['BatchNorm2d', 'BatchNormalization']:
                a_bn_module = self._reduce_batchnorm_module(an_op)
                modified_modules[an_op.dotted_name] = a_bn_module
            else:
                logger.debug("reduce_modules(): skipping: %s", an_op.dotted_name)

        return modified_modules

    def _reduce_batchnorm_module(self, an_op: Operation):
        """ Winnow a BatchNorm2d Module """

        # BatchNorm2d Op's inputs provide following:
        # 1) input
        # 2) two parameters 'weight' and 'bias'
        # 3) two register buffers 'running_mean' and 'running_var'
        assert len(an_op.inputs) == 5

        # We need to verify ordering, even though doesn't matter if weight, bias, running_mean and
        # running_var are interchanged, because they are of same size (1D array)
        x_input, weight, bias, running_mean, running_var = an_op.inputs

        # Set the parm_name for inputs and set out_channels to True
        x_input.parm_name, \
        weight.parm_name, \
        bias.parm_name, \
        running_mean.parm_name, \
        running_var.parm_name = 'input', 'weight', 'bias', 'running_mean', 'running_var'

        weight.impacts_out_channels = True
        assert x_input.shape == an_op.output_shape

        op_mask = self._op_to_mask_dict[an_op]
        input_ch_masks, output_ch_masks = op_mask.input_channel_masks, op_mask.output_channel_masks
        assert len(input_ch_masks) == 1 and len(output_ch_masks) == 1

        output_ch_indices_to_reduce = get_zero_positions_in_binary_mask(output_ch_masks[0])

        # create PolySlice object with dim = 0 and indices_to_reduce
        # weight, bias, running_mean and running_var are 1D arrays so we need to prune in dimension 0
        output_reduction = PolySlice(dim=0, index=output_ch_indices_to_reduce)

        named_module = an_op.get_module()
        reduced_module = self._reduce_batchnorm_parameters(an_op, named_module, output_reduction)
        logger.info("Winnowed BatchNorm module: %s", an_op.dotted_name)

        if reduced_module:
            seq = self._check_bn_output_mask_and_add_upsample_layer(an_op)
            if seq:
                return seq

        return reduced_module

    def _check_bn_output_mask_and_add_upsample_layer(self, bn_op: Operation):
        """

        Check the BatchNorm module's output mask with the next Op's input mask.
        If BatchNorm has more channels winnowed, add an UpSampleLayer.

        :param bn_op: BatchNorm op to check.
        :return: Return None if no UpSampleLayer is added.
                 Return the Sequential, if an UpSampleLayer is added to the BatchNorm module.
        """

        bn_op_mask = self._op_to_mask_dict[bn_op]
        output_ch_masks = bn_op_mask.output_channel_masks

        # BatchNorm Op has a single output mask.
        output_mask_index = 0
        bn_op_out_mask = output_ch_masks[output_mask_index]

        # Get the next downstream Op's input mask
        downstream_op = bn_op.output_ops[0]
        downstream_op_mask = self._op_to_mask_dict[downstream_op]
        downstream_op_input_masks = downstream_op_mask.input_channel_masks
        downstream_op_mask_index = determine_preceding_op_input_product_index_in_multi_input_op(bn_op,
                                                                                                downstream_op)
        downstream_op_input_mask = downstream_op_input_masks[downstream_op_mask_index]
        bn_op_out_mask_zero_positions = get_zero_positions_in_binary_mask(bn_op_out_mask)
        downstream_op_input_mask_zero_positions = get_zero_positions_in_binary_mask(downstream_op_input_mask)
        logger.debug("BatchNorm output mask: %s, downstream input mask: %s", bn_op_out_mask_zero_positions,
                     downstream_op_input_mask_zero_positions)

        if len(bn_op_out_mask_zero_positions) > len(downstream_op_input_mask_zero_positions):
            # More channels have been reduced from NatchNorm. Append an UpSamle layer to the BatchNorm.
            seq = self._append_upsample_layer_to_module(bn_op, bn_op_out_mask)
            return seq

        return None

    def _reduce_batchnorm_parameters(self, bn_op, named_module, output_reduction):
        """ Reduce the BatchNorm module's parameters. """

        for op_input_product in bn_op.inputs:

            if op_input_product.parm_name == "input":
                continue
            cur_parm = getattr(named_module, op_input_product.parm_name)
            if op_input_product.parm_name in ("running_mean", "running_var"):
                bn_running_parm = torch.nn.Parameter(reduce_tensor(cur_parm, output_reduction), requires_grad=False)
                named_module.register_buffer(op_input_product.parm_name, bn_running_parm)
            else:
                if self._using_cuda:
                    cur_parm = torch.nn.Parameter(cur_parm.cuda())
                weight_parm = torch.nn.Parameter(reduce_tensor(cur_parm, output_reduction), requires_grad=True)
                setattr(named_module, op_input_product.parm_name, weight_parm)

                reduction_dim = output_reduction.get_dims()[0]
                slices_decr = len(output_reduction.get_slices(reduction_dim))

                if op_input_product.impacts_out_channels:
                    named_module.num_features -= slices_decr

        return named_module

    def _reduce_conv_module(self, an_op: Operation):
        """

        Reduce the Conv2d module.

        :param an_op: Reduce this Conv2d module.
        :return:
        """
        logger.info("winnow_conv_module(): winnowing %s", an_op.dotted_name)
        op_input_product = an_op.inputs[1]
        op_input_product.parm_name = 'weight'

        op_mask = self._op_to_mask_dict[an_op]
        input_ch_masks, output_ch_masks = op_mask.input_channel_masks, op_mask.output_channel_masks

        if len(input_ch_masks) == 1:
            input_ch_indices_to_reduce = get_zero_positions_in_binary_mask(input_ch_masks[0])
        else:
            input_ch_indices_to_reduce = []
        if output_ch_masks:
            output_ch_indices_to_reduce = get_zero_positions_in_binary_mask(output_ch_masks[0])
        else:
            output_ch_indices_to_reduce = []
        logger.debug("winnow_conv_module(): ip indices to reduce: %s, op indices to reduce: %s",
                     input_ch_indices_to_reduce, output_ch_indices_to_reduce)

        named_module = an_op.get_module()

        if named_module.groups > 1:
            op_input_product.impacts_groups = True

        if input_ch_indices_to_reduce and named_module.groups == 1:
            parm_dim_to_reduce = 1
            ip_reduction = PolySlice(parm_dim_to_reduce, input_ch_indices_to_reduce)
            op_input_product.impacts_in_channels = True
            reduction = ip_reduction
            reduced_module = reduce_conv_module_weight_parameter(an_op, named_module, reduction, self._using_cuda)

        if output_ch_indices_to_reduce:
            parm_dim_to_reduce = 0
            op_reduction = PolySlice(parm_dim_to_reduce, output_ch_indices_to_reduce)
            op_input_product.impacts_out_channels = True
            reduction = op_reduction
            reduced_module = reduce_conv_module_weight_parameter(an_op, named_module, reduction, self._using_cuda)

        if len(an_op.inputs) > 2:
            # Bias Parameter
            if output_ch_indices_to_reduce:
                # Has output channel reduction
                logger.debug("Op: %s, has Bias parameter with output channels to reduce: %s",
                             an_op.dotted_name, output_ch_indices_to_reduce)
                reduce_conv_module_bias_parameter(an_op, named_module, output_ch_indices_to_reduce, self._using_cuda)

        logger.info("Winnowed Conv module: %s", an_op.dotted_name)

        # If a Conv module's inputs channels were reduced, also check if the previous Op's output mask is the same
        # as the conv module's input mask. If not, it may be necessary to prepend a DownSampleLayer to the
        # Conv modules.
        if input_ch_indices_to_reduce:
            sequential = self._check_conv_input_mask_and_add_downsample_layer(an_op)
            if sequential:
                return sequential

        return reduced_module


    def _check_conv_input_mask_and_add_downsample_layer(self, conv_op: Operation):
        """

        Check the Conv module's input mask with the previous Op's output mask.
        If Conv has more channels winnowed prepend a DownSampleLayer to the Conv module.

        :param conv_op: Conv op to check.
        :return: Return None if no DownSampleLayer is added.
                 Return the Sequential, if a DownSampleLayer is prepended to the Conv module.
        """

        conv_op_mask = self._op_to_mask_dict[conv_op]
        input_ch_masks = conv_op_mask.input_channel_masks

        # Conv module has a single input.
        mask_index = 0

        input_ch_indices_to_reduce = get_zero_positions_in_binary_mask(input_ch_masks[mask_index])
        input_producer_op = get_previous_op(conv_op)
        logger.debug("Input producer Op output shape: %s", input_producer_op.output_shape[1])
        input_producer_op_mask = self._op_to_mask_dict[input_producer_op]
        input_producer_op_out_masks = input_producer_op_mask.output_channel_masks
        input_producer_op_out_mask = None
        if len(input_producer_op.output_ops) == 1:
            input_producer_op_out_mask = input_producer_op_out_masks[0]
        elif len(input_producer_op.output_ops) > 1:
            input_producer_op_output_mask_index = \
                determine_succeeding_op_output_product_index_in_multi_output_op(conv_op, input_producer_op)
            input_producer_op_out_mask = input_producer_op_out_masks[input_producer_op_output_mask_index]
        else:
            logger.error("Number of consumer is zero for: %s", input_producer_op.dotted_name)

        input_producer_op_out_mask_zero_positions = get_zero_positions_in_binary_mask(input_producer_op_out_mask)
        input_producer_op_out_mask_length = len(get_one_positions_in_binary_mask(input_producer_op_out_mask))
        if len(input_ch_indices_to_reduce) > len(input_producer_op_out_mask_zero_positions):
            logger.debug("Conv module in mask zero positions: %s, Previous Op out mask zero positions: %s, "
                         "Previous Op out mask length: %s", input_ch_indices_to_reduce,
                         input_producer_op_out_mask_zero_positions, input_producer_op_out_mask_length)
            sequential = self._prepend_downsample_layer_to_module(conv_op, input_producer_op_out_mask)
            return sequential

        return None

    def _prepend_downsample_layer_to_module(self, op: Operation, input_producer_op_out_mask: List[int]):
        """Creates a Sequential by prepending a Downsample layer to the module associated with the Op.
        Replaces the module with the Sequential at the module's parent.


        :param op: The Op to which a Downsample layer is prepended.
        :param input_producer_op_out_mask: List of channels for the input op; 0 to winnow and 1 to keep channel
        :return:
        """

        logger.debug("Prepend Downsample: Op dotted name: %s, Op type: %s next down module dotted name: %s, type: %s",
                     op.dotted_name, op.type, op.output_ops[0].dotted_name, op.output_ops[0].type)

        if op.type in get_conv_ops_for_api(ModelApi.pytorch):
            conv_op = op
        else:
            conv_op = get_next_conv_op_for_op_with_single_consumer(op)

        module = conv_op.get_module()

        parent_module_ref, var_name = self._parent_module_ref[module]
        op_mask = self._op_to_mask_dict[op]
        op_in_masks = op_mask.input_channel_masks

        keep_indices = get_indices_among_ones_of_overlapping_ones(input_producer_op_out_mask, op_in_masks[0])
        keep_indices_tensor = torch.tensor(keep_indices)        # pylint: disable=not-callable
        if self._using_cuda:
            keep_indices_tensor = keep_indices_tensor.cuda()

        down_sample = DownsampleLayer(keep_indices_tensor)

        # Create a sequential of the  Downsample layer and the module
        seq = torch.nn.Sequential(down_sample, module)

        # Set the Sequential as the child for parent module.
        setattr(parent_module_ref, var_name, seq)

        logger.info("Prepended Downsample Layer to %s", op.dotted_name)

        return seq

    def _append_upsample_layer_to_module(self, op: Operation, input_mask):
        """ Creates a Sequential by appending an Upsample layer to the given module.
            Replaces the module with the Sequential at the module's parent. """

        module = op.get_module()
        parent_module_ref, var_name = self._parent_module_ref[module]

        input_mask_tensor = torch.tensor(input_mask) # pylint: disable=not-callable
        if self._using_cuda:
            input_mask_tensor = input_mask_tensor.cuda()

        up_sample = UpsampleLayer(input_mask_tensor)

        # Create a sequential of the module and the UpsampleLayer
        seq = torch.nn.Sequential(module, up_sample)

        # Set the Sequential as the child for parent module.
        setattr(parent_module_ref, var_name, seq)

        logger.info("Appended Upsample Layer to %s", op.dotted_name)
        return seq


def reduce_conv_module_bias_parameter(an_op, named_module, indices_to_reduce, using_cuda):
    """
    Reduces the bias parameter of a Conv module.

    :param an_op:
    :param named_module:
    :param reduction:
    :param using_cuda:
    :return:
    """

    # An Op could have the following input Products:
    #                                       0: true input,
    #                                       1: weights,
    #                                       2: bias
    bias_parameter_product = an_op.inputs[2]
    bias_parameter_product.parm_name = 'bias'
    assert len(bias_parameter_product.shape) == 1
    assert bias_parameter_product.shape[0] == an_op.output_shape[1]
    cur_parm = getattr(named_module, bias_parameter_product.parm_name)
    bias_reduction = PolySlice(0, indices_to_reduce)

    if using_cuda:
        cur_parm = torch.nn.Parameter(cur_parm.cuda())

    bias_parameter = torch.nn.Parameter(reduce_tensor(cur_parm, bias_reduction), requires_grad=True)
    setattr(named_module, 'bias', bias_parameter)

    logger.debug("For module %s, Bias Parameter reduced.", an_op.dotted_name)


def reduce_conv_module_weight_parameter(an_op, named_module, reduction, using_cuda):
    """
    Reduces the weight parameter of a Conv module.

    :param an_op:
    :param named_module:
    :param using_cuda:
    :return:
    """

    # An Op could have the following inputs:
    #                                       0: true input,
    #                                       1: weights,
    #                                       2: bias
    op_input_product = an_op.inputs[1]
    cur_parm = getattr(named_module, op_input_product.parm_name)

    if using_cuda:
        cur_parm = torch.nn.Parameter(cur_parm.cuda())

    weight_parm = torch.nn.Parameter(reduce_tensor(cur_parm, reduction), requires_grad=True)
    setattr(named_module, 'weight', weight_parm)

    reduction_dim = reduction.get_dims()[0]
    slices_decr = len(reduction.get_slices(reduction_dim))

    if op_input_product.impacts_in_channels and reduction_dim == 1:
        named_module.in_channels -= slices_decr
        op_input_product.impacts_in_channels = False

    if op_input_product.impacts_out_channels and reduction_dim == 0:
        named_module.out_channels -= slices_decr
        op_input_product.impacts_out_channels = False

    if op_input_product.impacts_groups:
        named_module.groups -= slices_decr
        named_module.in_channels -= slices_decr
        logger.debug("For module %s reduced input channels to %d and groups to %d", an_op.dotted_name,
                     named_module.in_channels, named_module.groups)

    return named_module


def get_next_conv_op_for_op_with_single_consumer(op: Operation):
    """
    For the given single consumer Op, return the next downstream Conv Op

    :param op: For this Op, find the next downstream Conv Op
    :return: The Conv Op downstream to the given Op
    """

    single_consumer_op_index = 0
    while op.type not in ['Conv2d', 'convolution']:     # TODO: remove 'Conv2d' when old CG is gone
        op = op.output_ops[single_consumer_op_index]
    return op


def get_previous_op(op: Operation):
    """
    Return the previous Op for the given Op.
    If the previous op is ReLU, return the next previous op.

    :param op: The op for which to return the previous op.
    :return:
    """

    previous_op = op.inputs[0].producer
    while previous_op.type in ('Relu', 'ReLU', 'relu'):     # TODO: remove first two after old CG is gone
        previous_op = previous_op.inputs[0].producer
    return previous_op

# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

# pylint: disable=too-many-lines

""" Utilities to load and save onnx models """
from dataclasses import dataclass
from typing import Union, List, Tuple, Dict, Set, Optional, Any
import os
import copy
from collections import defaultdict, deque
from enum import IntEnum
import torch
import torch.nn as nn
import torch.onnx.symbolic_caffe2
import torchvision
import onnx
import yaml
from onnx import GraphProto
from packaging import version  # pylint: disable=wrong-import-order

from aimet_common.utils import AimetLogger
import aimet_torch.utils
import aimet_torch._base.nn.modules.custom as aimet_modules
from aimet_torch.defs import OpToIOTensors

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

# Controls whether quantsim export uses OnnxSaver to set exported ONNX graph node names or not.
# From torch 1.13 onwards, ONNX export automatically names graph nodes using PyTorch's naming scheme, so no renaming is
# required.
# TODO: Corner cases exist when ONNX names model nodes with reused modules, or models with recurrent nodes. Do not use
# this flag in such circumstances.
EXPORT_TO_ONNX_DIRECT = False

# runs the second pass of markers for non-leaf torch module and updates names of onnx ops belonging to
# non-leaf pytorch module
update_all_onnx_nodes_name = True

# Flag to adjust ONNX node output to have unique name
MAKE_NODE_OUTPUT_NAME_UNIQUE = True


recurrent_onnx_optypes = ['LSTM', 'GRU', 'RNN']

# This is a dict that maps a PyTorch module type to the corresponding ONNX op type (as a string)
map_torch_types_to_onnx = {
    nn.AdaptiveAvgPool2d: ['GlobalAveragePool', 'AveragePool'],
    nn.AvgPool2d: ['AveragePool'],
    nn.BatchNorm1d: ['BatchNormalization'],
    nn.BatchNorm2d: ['BatchNormalization'],
    nn.ChannelShuffle: ['ChannelShuffle'],
    nn.Conv1d: ['Conv'],
    nn.Conv2d: ['Conv'],
    nn.ConvTranspose2d: ['ConvTranspose'],
    nn.Dropout: ['Dropout'],
    nn.Dropout2d: ['Dropout'],
    nn.Embedding: ['Gather'],
    nn.Flatten: ['Flatten'],
    nn.GELU: ['GELU'],                  # Not a supported op in ONNX, adding this entry for usage by Connected Graph
    nn.GroupNorm: ['GroupNorm'],
    nn.GRU: ['GRU'],
    nn.Hardswish: ['HardSwish'],
    nn.MaxPool2d: ['MaxPool'],
    nn.MaxPool3d: ['MaxPool'],
    nn.LayerNorm: ['LayerNorm'],        # Not a supported op in ONNX, adding this entry for usage by Connected Graph
    nn.InstanceNorm2d: ['InstanceNormalization'],
    nn.InstanceNorm1d: ['InstanceNormalization'],
    nn.LeakyReLU: ['LeakyRelu'],
    nn.Linear: ['Gemm', 'MatMul'],
    nn.LogSoftmax: ['LogSoftmax'],
    nn.LSTM: ['LSTM'],
    nn.PixelShuffle: ['DepthToSpace'],
    nn.PixelUnshuffle: ['SpaceToDepth'],
    nn.PReLU: ['PRelu'],
    nn.ReLU: ['Relu'],
    nn.ReLU6: ['Clip'],
    nn.RNN:  ['RNN'],
    nn.Sigmoid: ['Sigmoid'],
    nn.Softmax: ['Softmax'],
    nn.Softplus: ['Softplus'],
    nn.Tanh: ['Tanh'],
    nn.Upsample: ['Upsample'],
    nn.UpsamplingNearest2d: ['Upsample'],
    aimet_modules.Add: ['Add'],
    aimet_modules.Cast: ['Cast'],
    aimet_modules.ChannelShuffle: ['ChannelShuffle'],
    aimet_modules.Concat: ['Concat'],
    aimet_modules.CustomGather: ['Gather'],
    aimet_modules.DepthToSpaceDCRMode: ['DepthToSpace'],
    aimet_modules.Divide: ['Div'],
    aimet_modules.Expand: ['Expand'],
    aimet_modules.Gather: ['Gather'],
    aimet_modules.GatherNd: ['GatherND'],
    aimet_modules.IndexSelect: ['Gather'],
    aimet_modules.Max: ['ReduceMax'],
    aimet_modules.MaxPool2d: ['MaxPool'],
    aimet_modules.Min: ['ReduceMin'],
    aimet_modules.Multiply: ['Mul'],
    aimet_modules.NonZero: ['NonZero'],
    aimet_modules.Pad: ['Pad'],
    aimet_modules.Permute: ['Transpose'],
    aimet_modules.Reshape: ['Reshape'],
    aimet_modules.RoiAlign: ['RoiAlign'],
    aimet_modules.ScatterElements: ['ScatterElements'],
    aimet_modules.Select: ['Gather'],
    aimet_modules.Split: ['Split'],
    aimet_modules.StridedSlice: ['Slice'],
    aimet_modules.Subtract: ['Sub'],
    aimet_modules.Tile: ['Tile'],
    aimet_modules.TopK: ['TopK'],
    torchvision.ops.RoIPool: ['MaxRoiPool'],
    aimet_modules.Mean: ['ReduceMean'],
    aimet_modules.NonMaxSuppression: ['NonMaxSuppression'],
}

# Maps pytorch functional op string names to corresponding onnx types.
pytorch_functional_name_to_onnx_dict = {
    'add': 'Add',
    'cat': 'Concat',
    'mul': 'Mul',
    'div': 'Div'
}

if version.parse(torch.__version__) >= version.parse("1.9"):
    onnx_subgraph_op_to_pytorch_module_param_name = {
        torch.nn.GroupNorm:
            {
                # '#depth', 'op_type': {input_index: torch module parameter name}
                ('#3', 'Mul'): {1: 'weight'},
                ('#4.end', 'Add'): {1: 'bias'}
            },
        torch.nn.Linear:
            {
                ('', 'MatMul'): {1: 'weight'},
                ('#1.end', 'Add'): {0: 'bias'}
            },
        torch.nn.PReLU:
            {
                ('', 'PRelu'): {1: 'weight'}
            }
    }
else:
    onnx_subgraph_op_to_pytorch_module_param_name = {
        torch.nn.GroupNorm:
            {
                # '#depth', 'op_type': {input_index: torch module parameter name}
                ('#3', 'Mul'): {1: 'weight'},
                ('#4.end', 'Add'): {1: 'bias'}
            },
        torch.nn.Linear:
            {
                ('', 'MatMul'): {1: 'weight'},
                ('#1.end', 'Add'): {1: 'bias'}
            },
        torch.nn.PReLU:
            {
                ('', 'PRelu'): {1: 'weight'}
            }
    }

onnx_subgraph_op_to_pytorch_module_param_name_index_based = {
    torch.nn.GroupNorm:
        {
            (7, 'Mul'): {1: 'weight'},
            (8, 'Add'): {1: 'bias'}
        },
    torch.nn.Linear:
        {
            (0, 'MatMul'): {1: 'weight'},
            (1, 'Add'): {0: 'bias'}
        },
    torch.nn.PReLU:
        {
            (0, 'PRelu'): {1: 'weight'}
        }
}


def get_pytorch_name_from_onnx_name(onnx_name: str) -> str:
    """
    Extract the PyTorch name from ONNX name, for ONNX names obtained from torch 1.13 ONNX export.

    NOTE: submodule layer2.0.downsample.0 has a following node name in exported
     ONNX graph: /layer2/layer2.0/downsample/downsample.0/Conv
    This utility tries to remove redundant information from the node name
     and returns pytorch name. => layer2.0.downsample.0

    :param onnx_name: ONNX name to parse
    :return: PyTorch name as a string preserving scope information.
    """
    tokens = onnx_name.split('/')[1:-1]
    updated_tokens = []

    # 1) If the next token has "." and if the next token contains current token then don't add current token
    # to the updated_tokens list.
    # 2) If the next token don't have "." then simply add current token to the updated_tokens list.
    for i in range(len(tokens) - 1):
        if "." in tokens[i + 1]:
            if not tokens[i] in tokens[i + 1]:
                updated_tokens.append(tokens[i])
        else:
            updated_tokens.append(tokens[i])
    # Add the last token to the updated_token list.
    updated_tokens.extend(tokens[-1:])
    pytorch_name = '.'.join([token for token in updated_tokens if token])
    return pytorch_name


@dataclass(frozen=True)
class OnnxExportApiArgs:
    """
    Configuration for torch onnx export api invocation
    Refer torch documentation https://pytorch.org/docs/1.7.1/onnx.html?highlight=onnx%20export#torch.onnx.export

    :param opset_version: onnx opset version to use to export the model
    :param input_names:  names to assign to the input nodes of the onnx graph, in order
    :param output_names: names to assign to the output nodes of the graph, in order
    """

    opset_version: Optional[int] = None
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None

    @property
    def kwargs(self):
        """
        formats all override options into kwarg format to appended to onnx export call
        """
        return {'opset_version': self.opset_version,
                'input_names': self.input_names,
                'output_names': self.output_names}


@dataclass(frozen=True)
class PrunedInitializerInfo:
    """
    Data carrier containing initializer to be added and identity node to be removed in ONNX graph
    """
    initializer: onnx.TensorProto
    identity_node: onnx.NodeProto


class MarkerAttr(IntEnum):
    """ Enumeration for the custom marker attribute to index into the onnx node """
    NAME = 0
    IS_LEAF = 1
    IS_START = 2

class CustomMarkerFunc(torch.autograd.Function):
    """
    This function helps add a custom layer when exporting to ONNX
    Note the input tensor has a trivial operation performed on it (clamp). This is needed to force
    pytorch trace to not ignore the function.
    """

    @staticmethod
    def symbolic(g, inp, identifier, start, is_leaf):
        """
        Magic method that helps with exporting a custom ONNX node
        """
        # Note: the attribute are listed alphabetically in onnx.NodeProto
        return g.op('aimet_torch::CustomMarker', inp, id_s=identifier, leaf_s=is_leaf, start_s=start)\
                .setType(inp.type())

    @staticmethod
    def forward(ctx, inp, _identifier, _start, _is_leaf):     # pylint: disable=arguments-differ
        return inp.clone().detach() # clone prevents export tracing to avoid optimizing out the operation.

    @staticmethod
    def backward(ctx, _grad):                       # pylint: disable=arguments-differ
        raise NotImplementedError()


class CustomMarker(torch.nn.Module):
    """
    This is a temporary layer that in inserted next to a real layer to distinguish the real layer in the
    exported ONNX format
    """

    def __init__(self, module, identifier, is_leaf):
        super().__init__()
        self.marked_module = module
        self.identifier = identifier
        self.is_leaf = is_leaf

    def forward(self, *inputs, **kwargs):
        """
        Forward method for this CustomMarker layer
        """
        marked_tensor_map = dict()
        marked_inputs = self._apply_markers_to_tuple(inputs, 'True', marked_tensor_map)

        if kwargs:
            kwargs = self._apply_marker_to_dict(kwargs, 'True', marked_tensor_map)

        x = self.marked_module(*marked_inputs, **kwargs)

        marked_tensor_map.clear() # TODO should input/output be decoupled?
        if isinstance(x, dict):
            output = self._apply_marker_to_dict(x, 'False', marked_tensor_map)
        else:
            was_output_tuple = isinstance(x, tuple)
            if isinstance(x, torch.Tensor):
                x = [x]

            output = self._apply_markers_to_tuple(x, 'False', marked_tensor_map)

            # retain the tuple as output if marked module generates tuple.
            if len(output) == 1 and not was_output_tuple:
                output = output[0]
            else:
                output = tuple(output)

        return output

    def _apply_markers_to_tuple(
            self, inputs, is_start_marker, marked_tensor_map) -> List[Union[torch.Tensor, Dict]]:
        """
        method to apply marker to every tensor or dictionary of tensor in the tuple
        :param is_start_marker: set to 'True' or 'False' based on if called for input or output dict of tensors
        :param marked_tensor_map: contains a map of id(tensor) to updated tensor i.e. after applying marker function.
        """
        marked_inputs = []
        for t in inputs:
            if id(t) in marked_tensor_map: # if tensor is already seen before map to the previous tensor
                t = marked_tensor_map[id(t)]
            else:
                key = id(t)
                if isinstance(t, torch.Tensor):
                    t = CustomMarkerFunc.apply(t, self.identifier, is_start_marker, self.is_leaf)
                elif isinstance(t, dict):
                    t = self._apply_marker_to_dict(t, is_start_marker, marked_tensor_map)
                marked_tensor_map[key] = t

            marked_inputs.append(t)

        return marked_inputs

    def _apply_marker_to_dict(
            self, tensors_dict: Dict, is_start_marker: str, marked_tensor_map) -> Dict:
        """
        method to apply marker to every tensor value in the dictionary
        :param tensors_dict: dictionary that may contain tensor values, currently not enabled for recursion
            i.e. dict of dict
        :param is_start_marker: set to 'True' or 'False' based on if called for input or output dict of tensors
        :param marked_tensor_map: contains a map of id(tensor) to updated tensor i.e. after applying marker function.
        """
        marked_dict_inputs = dict()
        for k, t in tensors_dict.items():
            if id(t) in marked_tensor_map: # if tensor is already seen before map to the previous tensor
                t = marked_tensor_map[id(t)]
            else:
                key = id(t)
                if isinstance(t, torch.Tensor):
                    t = CustomMarkerFunc.apply(t, self.identifier, is_start_marker, self.is_leaf)
                marked_tensor_map[key] = t
            marked_dict_inputs[k] = t
        return marked_dict_inputs

    def __getitem__(self, item):
        """
        method to allow forwarding request to the marked module
        """
        return self.marked_module[item]

    def __getattr__(self, name):
        """
        method to allow forwarding getattr request to the marked module
        """
        try:

            if name == 'marked_module':
                return self.__dict__['_modules']['marked_module']

            return self.__dict__[name]
        except KeyError:
            return getattr(self.__dict__['_modules']['marked_module'], name)


class OnnxSaver:
    """
    Utilities to save/load onnx models
    """

    @classmethod
    def create_onnx_model_with_pytorch_layer_names(cls, onnx_model_path: str, pytorch_model: torch.nn.Module,
                                                   dummy_input: Union[torch.Tensor, Tuple, List], is_conditional=False,
                                                   module_marker_map=None, onnx_export_args: Optional[Union[OnnxExportApiArgs, dict]] = None):
        """
        This utility does some pre-processing on the pytorch model and then uses it to obtain an equivalent onnx model
        with node names same as that in pytorch model. Whatever pre-processing/post-processing steps to be done on
        the resultant onnx model must be done here.

        :param onnx_model_path: Path to the ONNX model file
        :param pytorch_model: Equivalent PyTorch model instance
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param is_conditional: True if model is a conditional model, False otherwise
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        :param onnx_export_args:  override options for torch.onnx.export call
        :return:
        """
        # Pre-processing pytorch model
        for dropout_type in aimet_torch.utils.DROPOUT_TYPES:
            aimet_torch.utils.replace_modules_of_type1_with_type2(pytorch_model, dropout_type, torch.nn.Identity)

        if EXPORT_TO_ONNX_DIRECT:
            cls._export_model_to_onnx(pytorch_model, dummy_input, onnx_model_path, is_conditional, onnx_export_args)
            onnx_model = cls.load_simply_onnx_model(onnx_model_path)

            cls._fix_initializer_names_for_export_to_onnx_direct(onnx_model, pytorch_model)
            save_as_external_data = onnx_model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF
            onnx.save(onnx_model, onnx_model_path, save_as_external_data=save_as_external_data)
        else:
            # Obtaining equivalent onnx model
            cls.set_node_names(onnx_model_path, pytorch_model, dummy_input, is_conditional, module_marker_map,
                               onnx_export_args)

    @classmethod
    def set_node_names(cls, onnx_model_path: str, pytorch_model: torch.nn.Module,
                       dummy_input: Union[torch.Tensor, Tuple], is_conditional=False, module_marker_map=None,
                       onnx_export_args: Optional[Union[OnnxExportApiArgs, dict]] = None):
        """
        This utility loads a given onnx model file and set the names of all the nodes (ops) to equivalent
        pytorch module names given the corresponding pytorch model.
        :param onnx_model_path: Path to the ONNX model file
        :param pytorch_model: Equivalent PyTorch model instance
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param is_conditional: True if model is a conditional model, False otherwise
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        :param onnx_export_args:  override options for torch.onnx.export call
        :return:
        """
        if module_marker_map is None:
            module_marker_map = {}
        if onnx_export_args is None:
            onnx_export_args = OnnxExportApiArgs()

        onnx_model = cls._map_onnx_nodes_to_pytorch_modules(pytorch_model, dummy_input,
                                                            onnx_model_path, onnx_export_args, is_conditional,
                                                            module_marker_map)

        cls.check_onnx_node_names(onnx_model, pytorch_model)

        save_as_external_data = onnx_model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF

        onnx.save(onnx_model, onnx_model_path, save_as_external_data=save_as_external_data)

    @classmethod
    def check_onnx_node_names(cls, onnx_model: onnx.ModelProto, pytorch_model: torch.nn.Module):
        """
        This utility check the onnx node names for module names from  pytorch model
        :param onnx_model: ONNX model object
        :param pytorch_model: Equivalent PyTorch model instance
        """
        # pylint: disable=consider-using-generator
        root_module_names = tuple([local_module_name for local_module_name, _ in pytorch_model.named_children()])
        node_names = [node.name  for node in onnx_model.graph.node if not node.name.startswith('Constant')]
        num_nodes = len(node_names)
        node_names = set(node_names)
        if num_nodes != len(node_names):
            _logger.warning('%d nodes do not have unique names', num_nodes - len(node_names))

        num_named_nodes = sum(name.startswith(root_module_names) for name in node_names)
        _logger.info("successfully created onnx model with %d/%d node names updated", num_named_nodes, num_nodes)

    @staticmethod
    def _create_map_of_tensor_to_node(onnx_model: onnx.ModelProto) -> Tuple[Dict[str, List[onnx.NodeProto]],
                                                                            Dict[str, onnx.NodeProto]]:
        """
        Create and return two dicts
            1. Tensor -> list of nodes that consume this tensor
            2. Tensor -> node that produces this tensor
        :param onnx_model: ONNX model object
        :return: The two dicts described above

        Note: The list in #1 is ordered exactly in the order that pytorch trace reaches these nodes. This is important
        because later on we will use pytorch layer hooks to match these nodes with the equivalent PyTorch modules.
        The expectation is that PyTorch trace and PyTorch hooks follow the same execution sequence
        """
        map_input_tensor_to_node = {}
        map_output_tensor_to_node = {}
        for node in onnx_model.graph.node:
            OnnxSaver._populate_input_output_tensor_maps(map_input_tensor_to_node, map_output_tensor_to_node, node)
        return map_output_tensor_to_node, map_input_tensor_to_node

    @staticmethod
    def _populate_input_output_tensor_maps(map_input_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                                           map_output_tensor_to_node: Dict[str, onnx.NodeProto], node: onnx.NodeProto):
        """
        Populate input tensor to node and output tensor to node maps given a node. If the node has a graph with nodes,
        recursively populate the maps for subgraph nodes.
        :param map_input_tensor_to_node: Dictionary mapping onnx tensor to nodes that consume it
        :param map_output_tensor_to_node: Dictionary mapping onnx tensor to node that generated it
        :param node: Node to extract input and output tensors for to populate maps
        """
        for in_tensor in node.input:
            if in_tensor in map_input_tensor_to_node:
                map_input_tensor_to_node[in_tensor].append(node)
            else:
                map_input_tensor_to_node[in_tensor] = [node]

        for output in node.output:
            assert output not in map_output_tensor_to_node, 'More than one node produces the same tensor'
            map_output_tensor_to_node[output] = node

        for attribute in node.attribute:
            if getattr(attribute, 'g').name != '':
                for subnode in getattr(attribute, 'g').node:
                    OnnxSaver._populate_input_output_tensor_maps(map_input_tensor_to_node, map_output_tensor_to_node,
                                                                 subnode)

    @classmethod
    def _add_markers(cls, starting_module, module_name_map, module_marker_map, use_trace, add_all_marker: bool):
        """Recursively add marker layers
        :param add_all_marker: if True add marker for non-leaf modules along with leaf module.
        """
        for local_module_name, module_ref in starting_module.named_children():
            if aimet_torch.utils.is_leaf_module(module_ref):
                # local module name only contains the module's attribute name directly under the parent module
                # need to pick up the full name starting with top level module name from module_inputs_map
                full_module_name = module_name_map[module_ref]
                if use_trace:
                    assert full_module_name in module_marker_map
                    marker_layer = module_marker_map[full_module_name]
                else:
                    marker_layer = CustomMarker(module_ref, full_module_name, 'True')
                setattr(starting_module, local_module_name, marker_layer)

            # recursively call children modules
            else:
                # nn.ModuleList: does not have forward() method so should be ignored
                if add_all_marker and not isinstance(module_ref, torch.nn.ModuleList):
                    if not isinstance(module_ref, CustomMarker):
                        full_module_name = module_name_map[module_ref]
                        marker_layer = CustomMarker(module_ref, full_module_name, 'False')
                        setattr(starting_module, local_module_name, marker_layer)
                    else:
                        module_name = f'{module_name_map.get(starting_module, "<..>")}.{local_module_name}'

                        # check if it is the case of containing module having multiple reference.
                        if module_name != module_ref.identifier:
                            _logger.warning("layer=%s already marked as '%s', skipping",
                                            module_name, module_ref.identifier)
                cls._add_markers(module_ref, module_name_map, module_marker_map, use_trace, add_all_marker)

    @classmethod
    def _map_onnx_nodes_to_pytorch_modules(cls, pt_model, dummy_input, onnx_model_path, onnx_export_args,
                                           is_conditional, module_marker_map):
        """
        Exports an onnx model, maps the nodes in the onnx model to corresponding pytorch modules and names
        them accordingly
        :param pt_model: PyTorch model
        :param dummy_input: Dummy input to run a fwd pass on @pt_model
        :param onnx_model_path: Path to the saved ONNX model
        :param onnx_export_args:  override options for torch.onnx.export call
        :param is_conditional: True if model is a conditional model, False otherwise
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        """
        # pylint: disable=too-many-locals
        working_dir = os.path.dirname(onnx_model_path)

        onnx_model, onnx_model_all_marker = cls._create_onnx_model(dummy_input, is_conditional, module_marker_map,
                                                                   onnx_export_args, pt_model, working_dir,
                                                                   update_all_onnx_nodes_name)

        graphs_list, output_names_list = OnnxSaver._get_graph_and_output_names_lists(onnx_model)

        # Parse the ONNX model and create mapping from input and output tensors to corresponding nodes
        map_output_tensor_to_node, map_input_tensor_to_node = cls._create_map_of_tensor_to_node(onnx_model)

        # Find all marker nodes
        end_marker_map, start_marker_map = cls._create_map_of_marker_nodes(onnx_model)

        # Set names
        cls._set_onnx_node_names(map_input_tensor_to_node, start_marker_map)

        # set names for onnx ops belonging to non-leaf torch module
        if update_all_onnx_nodes_name:
            cls._update_non_leaf_pytorch_modules_onnx_nodes_names(
                pt_model, dummy_input, working_dir, onnx_export_args, is_conditional, module_marker_map,
                onnx_model_all_marker, onnx_model)

        cls._remove_redundant_end_suffix(onnx_model)

        # Remove markers
        cls._detach_start_and_end_markers(map_input_tensor_to_node, map_output_tensor_to_node, start_marker_map,
                                          end_marker_map, graphs_list, output_names_list)

        # Make sure we rename the model outputs to original names
        cls._set_output_names(onnx_model, graphs_list, output_names_list, map_output_tensor_to_node,
                              map_input_tensor_to_node)

        # Clean up the detached nodes
        # pylint: disable=no-member
        cls._remove_detached_nodes_from_onnx_graph(onnx_model.graph)

        cls._fix_param_names(onnx_model)
        cls._fix_initializer_names(onnx_model, pt_model)

        return onnx_model

    @classmethod
    def _create_onnx_model(cls, dummy_input, is_conditional: bool, module_marker_map,
                           onnx_export_args: Union[OnnxExportApiArgs, dict], pt_model: torch.nn.Module,
                           working_dir: str, add_all_markers: bool) -> Tuple[onnx.NodeProto, Optional[onnx.NodeProto]]:
        """
        creates an onnx model with markers at all module-levels if not successful falls back to marker at leaf only.
        :param dummy_input: Dummy input to run a fwd pass on @pt_model
        :param is_conditional: True if model is a conditional model, False otherwise
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        :param onnx_export_args:  override options for torch.onnx.export call
        :param pt_model: PyTorch model
        :param working_dir: working directory to save intermediate files
        :return onnx model w/ leaf level markers and when feasible at non-leaf level as well.
        """
        try:
            onnx_model = cls._create_onnx_model_with_markers(dummy_input, pt_model, working_dir, onnx_export_args,
                                                             is_conditional, module_marker_map, add_all_markers)
            if add_all_markers:
                return onnx_model, copy.deepcopy(onnx_model)

        except (IndexError, AttributeError, TypeError):
            onnx_model = cls._create_onnx_model_with_markers(dummy_input, pt_model, working_dir, onnx_export_args,
                                                             is_conditional, module_marker_map, False)
        return onnx_model, None

    @classmethod
    def _update_non_leaf_pytorch_modules_onnx_nodes_names(cls, pt_model: torch.nn.Module,
                                                          dummy_input,
                                                          working_dir: str,
                                                          onnx_export_args: Union[OnnxExportApiArgs, dict],
                                                          is_conditional: bool,
                                                          module_marker_map,
                                                          onnx_model_all_marker: Optional[onnx.ModelProto],
                                                          onnx_model: Optional[onnx.ModelProto]):
        # pylint: disable=too-many-arguments
        """
        updates the names of onnx ops belonging to non-leaf pytorch module with parent pytorch module context.
        :param pt_model: PyTorch model
        :param dummy_input: Dummy input to run a fwd pass on @pt_model
        :param working_dir: Path to the saved ONNX model
        :param onnx_export_args:  override options for torch.onnx.export call
        :param is_conditional: True if model is a conditional model, False otherwise
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        :param onnx_model_all_marker: onnx_model with marker attached at every module level before import, optionally provided.
        :param onnx_model: onnx_model with updated names for onnx ops belonging to leaf pytorch module
        """
        try:

            if onnx_model_all_marker is None:
                onnx_model_all_marker = cls._create_onnx_model_with_markers(
                    dummy_input, pt_model, working_dir, onnx_export_args, is_conditional, module_marker_map, True)

            cls._update_non_leaf_onnx_nodes_names(
                node_list_from_leaf_markers=cls._get_topological_sorted_nodes_list(onnx_model),
                node_list_from_all_markers=cls._get_topological_sorted_nodes_list(onnx_model_all_marker),
                working_dir=working_dir
            )

        except (KeyError, AttributeError, TypeError):
            _logger.error('failed with exception when naming of onnx op at non-leaf modules', exc_info=True)
            _logger.warning('naming of onnx op at non-leaf modules failed, skipping naming of non-leaf')

    @classmethod
    def _update_non_leaf_onnx_nodes_names(cls,
                                          node_list_from_leaf_markers: List[Tuple[onnx.NodeProto, str]],
                                          node_list_from_all_markers: List[Tuple[onnx.NodeProto, str]],
                                          working_dir: str):
        """
        update the names of onnx ops list belonging to onnx model with markers at leaf-level with names from
         node list generated with markers at all level.
        :param node_list_from_leaf_markers: onnx nodes list obtained with makers for leaf modules only
        :param node_list_from_all_markers: onnx nodes lisr obtained with marker for all pytorch modules
        :param working_dir: Path to the saved ONNX model
        """
        i = 0
        for (node, pt_module_name), (leaf_only_node, _) in zip(node_list_from_all_markers, node_list_from_leaf_markers):
            if node.op_type != leaf_only_node.op_type:
                _logger.warning('leaf (%s:%s) and non-leaf (%s:%s) sequence did not match at %d',
                                node.name, node.op_type, leaf_only_node.name, leaf_only_node.op_type, i)
                break

            if pt_module_name is not None and '#' not in leaf_only_node.name and leaf_only_node.name != pt_module_name:
                if 'marked_module' in leaf_only_node.name:
                    leaf_only_node.name = cls._get_updated_name(leaf_only_node.name)
                else:
                    leaf_only_node.name = f'{pt_module_name}.{leaf_only_node.name}'

            i += 1

        if i < len(node_list_from_leaf_markers) - 1:
            _logger.warning('partially (%d/%d) named onnx op at non-leaf modules', i, len(node_list_from_leaf_markers))
            cls.save_mismatch_sequence(i, node_list_from_all_markers, node_list_from_leaf_markers, working_dir)

    @classmethod
    def save_mismatch_sequence(cls, mismatch_index: int,
                               node_list_from_leaf_markers: List[Tuple[onnx.NodeProto, str]],
                               node_list_from_all_markers: List[Tuple[onnx.NodeProto, str]],
                               working_dir: str):
        """
        saves the mismatch sequence as a YAML file.
        :param mismatch_index: index at which the first mismatch occurs.
        :param node_list_from_leaf_markers: onnx nodes list obtained with makers for leaf modules only
        :param node_list_from_all_markers: onnx nodes lisr obtained with marker for all pytorch modules
        :param working_dir: Path to the saved ONNX model
        """
        sequence = []
        for index, ((node, pt_module_name), (leaf_only_node, _)) in enumerate(zip(node_list_from_all_markers,
                                                                                  node_list_from_leaf_markers)):
            sequence.append({'index': index,
                             'leaf': {'name': leaf_only_node.name, 'op_type': leaf_only_node.op_type},
                             'non-leaf': {'name': node.name, 'op_type': node.op_type, 'module_name': pt_module_name}})

        filename = os.path.join(working_dir, 'mismatch_onnx_mapping.yaml')
        with open(filename, 'w') as f:
            yaml.dump({'mismatch': mismatch_index, 'sequence': sequence}, f, default_flow_style=False, allow_unicode=True)
        _logger.info('saved node sequence to %s', filename)

    @classmethod
    def _get_topological_sorted_nodes_list(cls, onnx_model) -> List[Tuple[onnx.NodeProto, Optional[str]]]:
        """
        gather nodes in topological order along with the innermost marker layer context if available.
        :return: List of onnx nodes in topological order along with parent module context if available.
        """

        # Parse the ONNX model and create mapping from input and output tensors to corresponding nodes
        map_output_tensor_to_node, map_input_tensor_to_node = cls._create_map_of_tensor_to_node(onnx_model)
        visited = set()
        marker_context = {}
        nodes_list = []
        pending_nodes_list = deque()

        def gather_nodes_in_topological_order():
            """
            Implement a depth-first traversal of onnx nodes based on the input tensor map.
            """
            # pylint: disable=too-many-branches
            while pending_nodes_list:

                node, parent_module_name = pending_nodes_list.popleft()
                if id(node) in visited:
                    continue

                if node.op_type == 'CustomMarker':
                    identifier = node.attribute[MarkerAttr.NAME].s.decode()
                    is_start_marker = node.attribute[MarkerAttr.IS_START].s.decode()
                    if is_start_marker == 'True':
                        marker_context[identifier] = parent_module_name
                        parent_module_name = identifier
                    else:
                        if identifier in marker_context:
                            parent_module_name = marker_context[identifier]
                        else:
                            parent_context = parent_module_name if parent_module_name is not None else '<model>'
                            _logger.warning("end-marker seen without passing start-marker for '%s', continue to "
                                            "use parent context '%s'", identifier, parent_context)
                else:
                    # ignoring Constants since they might vary depending on where the markers were placed.
                    if node.op_type != 'Constant':
                        nodes_list.append((node, parent_module_name))

                visited.add(id(node))
                for attribute in node.attribute:
                    if getattr(attribute, 'g').name != '':
                        # traversing the list in reverse, see 'NOTE1'
                        for subnode in reversed(getattr(attribute, 'g').node):
                            pending_nodes_list.appendleft((subnode, parent_module_name))

                for tensor in node.output:
                    downstream_nodes = map_input_tensor_to_node.get(tensor, [])
                    # NOTE1: To preserve the order in which the trace was generated, traversing the list in reverse to
                    # ensure DFS is executed left to right when popping from the pending nodes queue.
                    for dnode in reversed(downstream_nodes):

                        # continue only if all nodes associated with the input tensor(s) have been visited.
                        skip = any(
                            input_tensor in map_output_tensor_to_node \
                            and map_output_tensor_to_node[input_tensor].op_type not in ['Constant'] \
                            and id(map_output_tensor_to_node[input_tensor]) not in visited
                            for input_tensor in dnode.input
                        )

                        if not skip:
                            pending_nodes_list.appendleft((dnode, parent_module_name))

        for n in onnx_model.graph.node:
            # Ideally traversing the graph here might not be required with DFS but might be
            # required to name ops in dangling sub-graph.
            if id(n) not in visited:
                pending_nodes_list.append((n, None))
                gather_nodes_in_topological_order()

        return nodes_list

    @staticmethod
    def _get_graph_and_output_names_lists(onnx_model: onnx.ModelProto) -> Tuple[List[onnx.GraphProto], List[List[str]]]:
        """
        Get a list of graphs in the model with outputs, and a list of lists of output names for the corresponding
        graphs.
        Length of the two lists should be identical. A dictionary would be the preferred structure, however
        onnx.GraphProto objects are unhashable.
        :param onnx_model: Onnx model to get list of graphs and output names for
        :return: A list of graphs in the model with outputs, and a list of lists of output names for the corresponding
        graphs.
        """
        graphs_list = []
        output_names_list = []
        OnnxSaver._populate_graph_and_output_names_lists(onnx_model.graph, graphs_list, output_names_list)
        return graphs_list, output_names_list

    @classmethod
    def _remove_redundant_end_suffix(cls, onnx_model: onnx.GraphProto):
        """
        Helper function to remove redundant  `.end` name suffix for module which generates a single onnx op e.g.
         torch.nn.Conv2d
        :param onnx_model: Onnx Model with onnx ops in leaf module named.
        """

        root_name_to_nodes_map = defaultdict(list)
        for node in onnx_model.graph.node:
            if node.op_type != 'CustomMarker':
                root_name = node.name.split('#')[0]
                root_name_to_nodes_map[root_name].append(node)

        for root_name, nodes_list in root_name_to_nodes_map.items():
            if len(nodes_list) == 1:
                nodes_list[0].name = root_name

    @staticmethod
    def _populate_graph_and_output_names_lists(onnx_graph: onnx.GraphProto, graphs_list: List[onnx.GraphProto],
                                               output_names_list: List[List[str]]):
        """
        Helper function to populate lists of graph and output names for a particular onnx graph. Also recursively
        populates the lists for subgraphs of the given graph.
        :param onnx_graph: Onnx graph to populate graphs list and output names list with
        :param graphs_list: List of graphs with outputs
        :param output_names_list: List of list of output names for corresponding graphs
        """
        output_names_list_for_graph = []
        for output in onnx_graph.output:
            output_names_list_for_graph.append(output.name)
        if output_names_list_for_graph:
            graphs_list.append(onnx_graph)
            output_names_list.append(output_names_list_for_graph)

        for node in onnx_graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g').name != '':
                    OnnxSaver._populate_graph_and_output_names_lists(attribute.g, graphs_list, output_names_list)

    @staticmethod
    def _get_onnx_node_map(onnx_graph: onnx.GraphProto, onnx_node_map: Dict[Tuple[str, str], onnx.NodeProto] = None) \
            -> Dict[Tuple[str, str], onnx.NodeProto]:
        """
        Get a mapping between tuples of onnx node names and op types, and the node itself.
        :param onnx_graph: Onnx graph used to populate map
        :param onnx_node_map: Map between tuples of onnx node names and op types, and the node itself
        """
        if onnx_node_map is None:
            onnx_node_map = {}

        for node in onnx_graph.node:
            onnx_node_map[(node.name, node.op_type)] = node

        for node in onnx_graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g').name != '':
                    OnnxSaver._get_onnx_node_map(attribute.g, onnx_node_map)
        return onnx_node_map

    @classmethod
    def _fix_initializer_names(cls, onnx_model: onnx.NodeProto, pt_model: torch.nn.Module):
        """
        Parameter names in some case do not have reflect the torch param names. This method updates the onnx model
        with param names using a custom mapping.
        When exporting a scripted model, all modules with params will need to update their initializer names.
        :param onnx_model: Onnx Model
        :param pt_model: PyTorch Model
        """

        initializers = OnnxSaver._get_all_initializers(onnx_model.graph)
        initializer_name_to_index = {initializer.name: idx for idx, initializer in enumerate(initializers)}
        onnx_node_map = OnnxSaver._get_onnx_node_map(onnx_model.graph)

        for module_name, module_ref in pt_model.named_modules():

            # Not using isinstance since we want to check for the exact type and not any subclasses
            # pylint: disable=unidiomatic-typecheck
            if type(module_ref) in onnx_subgraph_op_to_pytorch_module_param_name:
                for (node_suffix, op_type), replace_pairs in \
                        onnx_subgraph_op_to_pytorch_module_param_name[type(module_ref)].items():
                    # Some modules like linear can take on various forms (e.g. Gemm versus MatMul and Add)
                    if (module_name + node_suffix, op_type) in onnx_node_map:
                        node = onnx_node_map[module_name + node_suffix, op_type]

                        cls._replace_param_name(initializers, initializer_name_to_index, module_name, node,
                                                replace_pairs)

    @classmethod
    def _fix_initializer_names_for_export_to_onnx_direct(cls, onnx_model: onnx.NodeProto, pt_model: torch.nn.Module):
        """
        Parameter names in some case do not reflect the torch param names. This method updates the onnx model
        with param names using a custom mapping.
        When exporting a scripted model, all modules with params will need to update their initializer names.

        :param onnx_model: Onnx Model
        :param pt_model: PyTorch Model
        """
        initializers = OnnxSaver._get_all_initializers(onnx_model.graph)
        initializer_name_to_index = {initializer.name: idx for idx, initializer in enumerate(initializers)}

        # Create a dictionary mapping a pytorch module name to one or more ONNX nodes which are associated with that
        # pytorch module.
        # The assumption is that all ONNX nodes which are associated with a single pytorch module will share
        # commonalities in node naming, leading to the same pytorch name extracted by get_pytorch_name_from_onnx_name().
        pt_name_to_onnx_nodes = {}
        for node in onnx_model.graph.node:
            pt_name = get_pytorch_name_from_onnx_name(node.name)
            if pt_name not in pt_name_to_onnx_nodes:
                pt_name_to_onnx_nodes[pt_name] = [node]
            else:
                pt_name_to_onnx_nodes[pt_name].append(node)

        for module_name, module_ref in pt_model.named_modules():
            if type(module_ref) in onnx_subgraph_op_to_pytorch_module_param_name:
                if module_name not in pt_name_to_onnx_nodes:
                    error_str = f'{module_name} of type {module_ref} not found in pt_name_to_onnx_nodes dictionary.' \
                                f'Unable to rename parameters of {module_name}'
                    _logger.error(error_str)
                else:
                    onnx_nodes = pt_name_to_onnx_nodes[module_name]
                    for (idx, op_type), replace_pairs in \
                            onnx_subgraph_op_to_pytorch_module_param_name_index_based[type(module_ref)].items():
                        if len(onnx_nodes) > idx and onnx_nodes[idx].op_type == op_type:
                            cls._replace_param_name(initializers, initializer_name_to_index, module_name,
                                                    onnx_nodes[idx], replace_pairs)

    @classmethod
    def _replace_param_name(cls, initializers: List[onnx.TensorProto], initializer_name_to_index: Dict[str, int],
                            module_name: str, node: onnx.NodeProto, replace_pairs: Dict[int, str]):
        """
        helper method to replace parameter names at the corresponding input tensor index
        :param initializers: List of initializers in the onnx model
        :param initializer_name_to_index: Dictionary mapping initializer names to indices in initializers list
        :param module_name: PyTorch module name
        :param node: Onnx node part of sub-graph that maps to the torch module
        :param replace_pairs: dictionary of input tensor indices and param names
        """
        for input_index, param_name in replace_pairs.items():
            # If bias is not present for example, skip processing for the index
            if len(node.input) > input_index:
                new_param_name = module_name + '.' + param_name
                old_param_name = node.input[input_index]
                # Check if inp_tensor name is already named as a parameter (name.param_name, instead of a number). If so,
                # sanity check that the name we want to replace it with is the same as the existing name, then continue.
                if '.' in old_param_name:
                    if old_param_name != new_param_name:
                        print(f'{old_param_name} != {new_param_name} expected param name set')
                    continue
                node.input.remove(old_param_name)
                node.input.insert(input_index, new_param_name)

                # Find the index of the old initializer name and use it to update the corresponding initializer's name
                # in the actual initializers array
                initializer_index = initializer_name_to_index[old_param_name]
                initializers[initializer_index].name = new_param_name

    @classmethod
    def _remove_marked_module_string_from_initializer_names(cls, onnx_model):
        """
        Remove 'marked_module' from all initializer names.
        :param onnx_model: Onnx model containing initializers to update
        """
        for ini in OnnxSaver._get_all_initializers(onnx_model.graph):
            if 'marked_module' in ini.name:
                name = ini.name
                name = name.replace('marked_module.', '')
                ini.name = name

    @classmethod
    def _remove_marked_module_string_from_node_inp_out_names(
        cls,
        onnx_graph: GraphProto,
        node_output_name_counter: Optional[Dict[str, int]] = None,
        param_name_to_updated_name: Optional[Dict[str, str]] = None,
    ):
        """
        Remove 'marked_module' from all node's input and output names. Also, recursively updates subgraph node's
        input and output names.

        Examples:
        PyTorch version 1.9:
        1) 'layer1.marked_module.0.marked_module.conv2.marked_module.weight' --> 'layer1.0.conv2.weight'

        PyTorch version 1.9 onwards:
        1) '/layer1/marked_module/0/marked_module/conv2/marked_module/Conv_output_0' --> /layer1/0/conv2/Conv_output_0
        2) '/layer1/0/relu/marked_module_1/Relu_output_0' --> '/layer1/0/relu_1/Relu_output_0'

        :param onnx_graph: Onnx graph containing nodes and subgraph to modify
        :param node_output_name_counter: Dictionary holding base node name to its frequency
        :param param_name_to_updated_name: Dictionary holding param_name to updated_name
        """
        if node_output_name_counter is None:
            node_output_name_counter = {}
        if param_name_to_updated_name is None:
            param_name_to_updated_name = {}

        # Remove 'marked_module' string from input and output field of onnx.GraphProto
        for inp in onnx_graph.input:
            updated_name = cls._get_updated_name(inp.name)
            inp.name = updated_name
        for out in onnx_graph.output:
            updated_name = cls._get_updated_name(out.name)
            out.name = updated_name

        # Remove 'marked_module' string from all node's input and output names.
        for node in onnx_graph.node:
            for index, param_name in enumerate(node.output):
                updated_name = cls._get_updated_name(param_name)

                if MAKE_NODE_OUTPUT_NAME_UNIQUE:
                    updated_name = cls._get_unique_node_output_name(
                        updated_name,
                        param_name,
                        node_output_name_counter,
                        param_name_to_updated_name,
                    )
                node.output[index] = updated_name

            for index, param_name in enumerate(node.input):
                updated_name = cls._get_updated_name(param_name)
                if MAKE_NODE_OUTPUT_NAME_UNIQUE:
                    updated_name = param_name_to_updated_name.get(param_name, updated_name)
                node.input[index] = updated_name

        # Recursively updates subgraph node's input and output names.
        for node in onnx_graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g').name != '':
                    cls._remove_marked_module_string_from_node_inp_out_names(
                        attribute.g, node_output_name_counter, param_name_to_updated_name
                    )

    @staticmethod
    def _get_unique_node_output_name(
        updated_name: str,
        param_name: str,
        node_output_name_counter: Dict[str, int],
        param_name_to_updated_name: Dict[str, str],
    ) -> str:
        """
        Modify updated_name to make sure it will be unique node output name in ONNX graph

        :param updated_name: Updated name from cls._get_updated_name
        :param param_name: Original param_name to be key of param_name_to_updated_name
        :param node_output_name_counter: Dictionary holding node output name to its frequency
        :param param_name_to_updated_name: Dictionary holding param_name to updated_name
        :return: Unique node output name
        """
        if updated_name in node_output_name_counter:
            node_output_name_counter[updated_name] += 1
            name_count = node_output_name_counter[updated_name]
            deduplicated_name = f'{updated_name}_dup{name_count}'
            _logger.info(
                '%s has been updated to %s for name deduplication',
                updated_name,
                deduplicated_name,
            )
            updated_name = deduplicated_name
        else:
            node_output_name_counter[updated_name] = 0

        param_name_to_updated_name[param_name] = updated_name
        return updated_name

    @classmethod
    def _fix_param_names(cls, onnx_model):
        """
        Parameter names have an additional level due to the name of the Marker module itself. This method removes that.
        :param onnx_model: Onnx Model
        """
        # Rename initializers
        cls._remove_marked_module_string_from_initializer_names(onnx_model)

        # Change the references to initializers in each node
        cls._remove_marked_module_string_from_node_inp_out_names(onnx_model.graph)

    @classmethod
    def _remove_detached_nodes_from_onnx_graph(cls, onnx_graph: onnx.GraphProto):
        """
        Given a ONNX model, remove any detached nodes from the graph. Recursively removes detached nodes from subgraphs.
        """
        marker_nodes = [node for node in onnx_graph.node if node.op_type == 'CustomMarker']

        for node in marker_nodes:
            onnx_graph.node.remove(node)

        for node in onnx_graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g').name != '':
                    OnnxSaver._remove_detached_nodes_from_onnx_graph(attribute.g)

    @classmethod
    def _set_onnx_node_names(cls, map_input_tensor_to_node, start_marker_map):
        """
        Set names of the ONNX nodes using the identifier fields in the marker layers
        :param map_input_tensor_to_node: Map of tensor to node consuming that tensor
        :param start_marker_map: Map of start marker nodes in the ONNX graph
        :return:
        """
        node_name_count_map = dict()
        visited = set()
        leaf_only_start_marker = {n:m for n, m in start_marker_map.items()
                                  if m[0].attribute[MarkerAttr.IS_LEAF].s.decode() == 'True'}

        def set_name_for_downstream_nodes(starting_nodes, name, depth):
            for node in starting_nodes:

                if id(node) in visited:
                    continue

                if node.op_type == 'CustomMarker':      # Recursion end condition
                    return

                if depth == 0:
                    node.name = name
                else:
                    node.name = name + "#" + str(depth)

                if node.name in node_name_count_map:
                    reuse_count = node_name_count_map[node.name]
                    node_name_count_map[node.name] = reuse_count + 1
                    node.name = f'{node.name}-{reuse_count}' if '#' in node.name else f'{node.name}#0-{reuse_count}'
                else:
                    node_name_count_map[node.name] = 1

                for tensor in node.output:
                    downstream_nodes = map_input_tensor_to_node.get(tensor, [])
                    set_name_for_downstream_nodes(downstream_nodes, name, depth + 1)

                    for dnode in downstream_nodes:
                        if dnode.op_type == 'CustomMarker':  #end marker
                            node.name += '.end' if '#' in node.name else '#0.0.end'
                            break
                visited.add(id(node))

        for node_name, markers in leaf_only_start_marker.items():
            for marker in markers:
                out_tensor = marker.output[0]
                downstream_nodes = map_input_tensor_to_node.get(out_tensor, [])
                set_name_for_downstream_nodes(downstream_nodes, node_name, 0)

    @classmethod
    def _create_map_of_marker_nodes(cls, onnx_model):
        """
        Creates and returns maps of start and end marker nodes
        :param onnx_model: Onnx model
        :return: Map of end marker node, Map of start marker nodes
        """
        start_marker_map = defaultdict(list)
        end_marker_map = defaultdict(list)
        for node in onnx_model.graph.node:
            OnnxSaver._populate_start_and_end_marker_maps(start_marker_map, end_marker_map, node)
        return end_marker_map, start_marker_map

    @staticmethod
    def _populate_start_and_end_marker_maps(start_marker_map: Dict[str, onnx.NodeProto],
                                            end_marker_map: Dict[str, onnx.NodeProto], node: onnx.NodeProto):
        """
        Populate dictionaries mapping identifier names to start and end markers. Recursively populates dictionaries for
        subgraphs of the given node.
        :param start_marker_map: Dictionary mapping identifier names to start markers
        :param end_marker_map: Dictionary mapping identifier names to end markers
        :param node: Onnx node to potentially include in a map
        """
        if node.op_type == 'CustomMarker':
            identifier = node.attribute[MarkerAttr.NAME].s.decode()
            is_start_marker = node.attribute[MarkerAttr.IS_START].s.decode()

            if is_start_marker == 'True':
                start_marker_map[identifier].append(node)
            else:
                end_marker_map[identifier].append(node)

        for attribute in node.attribute:
            if getattr(attribute, 'g').name != '':
                for subnode in getattr(attribute, 'g').node:
                    OnnxSaver._populate_start_and_end_marker_maps(start_marker_map, end_marker_map, subnode)

    @classmethod
    # pylint: disable=too-many-locals
    def _create_onnx_model_with_markers(cls, dummy_input, pt_model, working_dir, onnx_export_args, is_conditional,
                                        module_marker_map, add_all_markers) -> \
            onnx.ModelProto:
        """
        Exports an onnx model with marker nodes inserted
        :param dummy_input: Dummy input
        :param pt_model: PyTorch model
        :param working_dir: Working directory for storing the exported onnx model
        :param onnx_export_args:  override options for torch.onnx.export call
        :param is_conditional: True if model is a conditional model, False otherwise
        :param module_marker_map: Maps module names to traced custom markers (only used for conditional models)
        :return: Onnx model with marker layers
        """
        model = copy.deepcopy(pt_model).cpu()
        module_name_map = {}
        for module_name, module_ref in model.named_modules():
            if add_all_markers or aimet_torch.utils.is_leaf_module(module_ref):
                module_name_map[module_ref] = module_name
        cls._add_markers(model, module_name_map, module_marker_map, is_conditional, add_all_markers)
        temp_file = os.path.join(working_dir,
                                 'temp_onnx_model_with_markers.onnx' if not add_all_markers else
                                 'temp_onnx_model_with_all_markers.onnx')

        cls._export_model_to_onnx(model, dummy_input, temp_file, is_conditional, onnx_export_args)
        return cls.load_simply_onnx_model(temp_file)

    @classmethod
    def load_simply_onnx_model(cls, filepath) -> onnx.ModelProto:
        """
         load the save onnx model and applies simply pass if enabled.
        :param filepath: file path of saved onnx model
        :return: Onnx model with optional simply pass
        """
        onnx_model = onnx.load(filepath)
        onnx_model = restore_onnx_graph_initializers(onnx_model, inplace=True)
        return onnx_model

    @classmethod
    def _detach_start_and_end_markers(cls, map_input_tensor_to_node: Dict[str, onnx.NodeProto],
                                      map_output_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                                      start_marker_map: Dict[str, onnx.NodeProto],
                                      end_marker_map: Dict[str, onnx.NodeProto], graphs_list: List[onnx.GraphProto],
                                      output_names_list: List[List[str]]):
        """
        Detach start and end marker nodes from the onnx graph
        :param map_input_tensor_to_node: Map of tensor to node consuming the tensor
        :param map_output_tensor_to_node: Map of tensor to node producing the tensor
        :param start_marker_map: Map of start marker identifiers to start markers
        :param end_marker_map: Map of end marker identifiers to end markers
        :param graphs_list: List of graphs with outputs
        :param output_names_list: List of list of output names for corresponding graphs
        """
        for markers in start_marker_map.values():
            for marker in markers:
                cls._detach_start_marker_node(map_input_tensor_to_node, map_output_tensor_to_node, marker)

        output_names_list = copy.deepcopy(output_names_list)
        for markers in end_marker_map.values():
            for marker in markers:
                cls._detach_end_marker_node(map_input_tensor_to_node, map_output_tensor_to_node,
                                            graphs_list, output_names_list, marker)

    @classmethod
    def _detach_start_marker_node(cls, map_input_tensor_to_node: Dict[str, onnx.NodeProto],
                                  map_output_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                                  start_marker: onnx.NodeProto):
        """
        Given a ONNX start_marker node, detach it from the graph
        :param map_input_tensor_to_node: Map of tensor to node consuming the tensor
        :param map_output_tensor_to_node: Map of tensor to node producing the tensor
        :param start_marker: Reference to the ONNX node to detach
        """
        assert len(start_marker.input) == 1
        assert len(start_marker.output) == 1

        input_tensor = start_marker.input[0]
        output_tensor = start_marker.output[0]

        for next_node in map_input_tensor_to_node[output_tensor]:

            index = list(next_node.input).index(output_tensor)
            next_node.input.remove(output_tensor)
            next_node.input.insert(index, input_tensor)
            map_input_tensor_to_node[input_tensor].append(next_node)

        map_input_tensor_to_node[input_tensor].remove(start_marker)
        del map_output_tensor_to_node[output_tensor]        # No node should produce output tensor anymore
        del map_input_tensor_to_node[output_tensor]         # No node should consume output tensor anymore

        start_marker.input.pop()
        start_marker.output.pop()

    @classmethod
    def _detach_end_marker_node(cls, map_input_tensor_to_node: Dict[str, onnx.NodeProto],
                                map_output_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                                graphs_list: List[onnx.GraphProto], output_names_for_all_graphs: List[List[str]],
                                end_marker: onnx.NodeProto):
        """
        Given a ONNX end_marker node, detach it from the graph
        :param onnx_model: ONNX model instance
        :param map_input_tensor_to_node: Map of tensor to node consuming the tensor
        :param map_output_tensor_to_node: Map of tensor to node producing the tensor
        :param end_marker: Reference to the ONNX node to detach
        """
        assert len(end_marker.input) == 1
        assert len(end_marker.output) == 1

        input_tensor = end_marker.input[0]
        output_tensor = end_marker.output[0]

        for idx, output_names_list in enumerate(output_names_for_all_graphs):
            if output_tensor in output_names_list:
                graph = graphs_list[idx]
                # Degenerate case: somebody did a "return y, y" at the end of the model or something similar
                for index, model_output in enumerate(output_names_list):
                    if model_output == output_tensor:
                        graph.output[index].name = input_tensor
                        output_names_for_all_graphs[idx][index] = input_tensor
                break

        if output_tensor in map_input_tensor_to_node:
            for next_node in map_input_tensor_to_node[output_tensor]:
                index = list(next_node.input).index(output_tensor)
                next_node.input.remove(output_tensor)
                next_node.input.insert(index, input_tensor)
                map_input_tensor_to_node[input_tensor].append(next_node)

        map_input_tensor_to_node[input_tensor].remove(end_marker)
        if not map_input_tensor_to_node[input_tensor]:
            del map_input_tensor_to_node[input_tensor]

        del map_output_tensor_to_node[output_tensor]        # No node should produce output tensor anymore
        if output_tensor in map_input_tensor_to_node:
            del map_input_tensor_to_node[output_tensor]     # No node should consume output tensor anymore

        end_marker.input.pop()
        end_marker.output.pop()

    @staticmethod
    def _set_output_names(onnx_model: onnx.ModelProto, graphs_list: List[onnx.GraphProto],
                          output_names_for_all_graphs: List[List[str]],
                          map_output_tensor_to_node: Dict[str, List[onnx.NodeProto]],
                          map_input_tensor_to_node: Dict[str, onnx.NodeProto]):
        """
        Set output names for outputs of the onnx model. Also recursively sets output names for subgraphs in the model.
        :param onnx_model: Model to set output names for
        :param graphs_list: List of graphs with outputs
        :param output_names_for_all_graphs: List of list of output names for corresponding graphs
        :param map_output_tensor_to_node: Dictionary mapping onnx tensor to node that generated it
        :param map_input_tensor_to_node: Dictionary mapping onnx tensor to nodes that consume it
        """

        OnnxSaver._set_output_names_for_graph(onnx_model.graph, graphs_list, output_names_for_all_graphs,
                                              map_output_tensor_to_node, map_input_tensor_to_node)

        for node in onnx_model.graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g').name != '':
                    OnnxSaver._set_output_names_for_graph(attribute.g, graphs_list, output_names_for_all_graphs,
                                                          map_output_tensor_to_node, map_input_tensor_to_node)

    @staticmethod
    def _set_output_names_for_graph(onnx_graph: onnx.ModelProto, graphs_list, output_names_for_all_graphs,
                                    map_output_tensor_to_node, map_input_tensor_to_node):
        """
        Set output names for the given onnx graph
        :param onnx_graph: Graph to set output names for
        :param graphs_list: List of graphs with outputs
        :param output_names_for_all_graphs: List of list of output names for corresponding graphs
        :param map_output_tensor_to_node: Dictionary mapping onnx tensor to node that generated it
        :param map_input_tensor_to_node: Dictionary mapping onnx tensor to nodes that consume it
        """
        try:
            graph_index = graphs_list.index(onnx_graph)
        except ValueError:
            return

        desired_model_output_names = output_names_for_all_graphs[graph_index]
        # Iterate over the model outputs
        for index, output in enumerate(onnx_graph.output):
            new_tensor = desired_model_output_names[index]
            old_tensor = output.name

            if old_tensor == new_tensor:  # Nothing to do
                continue

            if old_tensor in map_input_tensor_to_node:
                # Degenerate case: model output tensor also is an intermediate tensor that inputs into other nodes
                for consumer in map_input_tensor_to_node[old_tensor]:
                    index = list(consumer.input).index(old_tensor)
                    consumer.input.remove(old_tensor)
                    consumer.input.insert(index, new_tensor)
                    if new_tensor not in map_input_tensor_to_node:
                        map_input_tensor_to_node[new_tensor] = []
                    map_input_tensor_to_node[new_tensor].append(consumer)

                del map_input_tensor_to_node[old_tensor]  # No node should consume old tensor anymore

            producer = map_output_tensor_to_node[old_tensor]

            output.name = new_tensor
            index = list(producer.output).index(old_tensor)
            producer.output.remove(old_tensor)
            producer.output.insert(index, new_tensor)

            del map_output_tensor_to_node[old_tensor]
            map_output_tensor_to_node[new_tensor] = producer

            # If there were duplicate outputs with the same name, they need to be updated
            for output_node in onnx_graph.output:
                # Ugly double loop - cannot avoid
                if output_node.name == old_tensor:
                    output_node.name = new_tensor

    @staticmethod
    def _get_all_nodes(onnx_graph: onnx.GraphProto, all_nodes: Union[List[onnx.NodeProto], None] = None) -> \
            List[onnx.NodeProto]:
        """
        Get all nodes in an onnx graph, including nodes in subgraphs.
        :param onnx_graph: Onnx graph to get nodes for
        :param all_nodes: List of nodes in the graph
        :return: List of nodes in the graph
        """
        if all_nodes is None:
            all_nodes = []
        for node in onnx_graph.node:
            all_nodes.append(node)
            for attribute in node.attribute:
                if getattr(attribute, 'g').name != '':
                    OnnxSaver._get_all_nodes(attribute.g, all_nodes)
        return all_nodes

    @staticmethod
    def _collate_io_tensors_for_multi_layer_recurrent_nodes(onnx_model: onnx.NodeProto,
                                                            node_to_io_tensor_name_map: Dict):
        """
        Given an ONNX model and corresponding node-tensor map, consolidate multi-layer recurrent nodes
        into single map entries
        """
        recurrent_nodes = []
        for node in onnx_model.graph.node:
            if node.op_type in recurrent_onnx_optypes:
                recurrent_nodes.append(node.name)

        # Collection of recurrent nodes that includes only the first layer nodes
        root_nodes = dict()
        # onnx graph is maintained in topological order, the first occurrence of the onnx node with the module name will
        # be the root node of the recurrent module
        for node_name in recurrent_nodes:
            root_name = node_name.split('#')[0]
            if root_name not in root_nodes:
                root_nodes[root_name] = node_name

        for root_node in root_nodes.values():
            # Find nodes corresponding to all other layers of the recurrent node
            other_layers = [node for node in recurrent_nodes if node.startswith(root_node.split('#')[0])]

            # Remove the root node from other layers
            other_layers.remove(root_node)

            # sort the other layers using the depth value following the '#'
            def get_depth(layer_name):
                right_of_pound = layer_name.split('#')[1]
                but_ignore_dash = right_of_pound.split('-')[0]
                return int(but_ignore_dash)

            other_layers = sorted(other_layers, key=get_depth)

            # Append the io_tensors for all layers for the current root recurrent node, in order
            io_tensor_list = [node_to_io_tensor_name_map[root_node]]
            for layer in other_layers:
                io_tensor_list.append(node_to_io_tensor_name_map[layer])
                del node_to_io_tensor_name_map[layer]

            # Let's rename the root node, so we can identify it later when building encoding file e.g.
            del node_to_io_tensor_name_map[root_node]
            root_node = root_node.split('#')[0] + '#root_node'

            node_to_io_tensor_name_map[root_node] = io_tensor_list

    @staticmethod
    def _get_all_initializers(onnx_graph: onnx.GraphProto, initializers: Union[List[onnx.TensorProto], None] = None) \
            -> List[onnx.TensorProto]:
        """
        Get all initializer names in the onnx graph. Also recursively gets initializer names for subgraphs of the graph.
        :param onnx_graph: Onnx graph to get initializer names for
        :param initializers: List of initializer names in the graph
        :return List of initializer names in the graph
        """
        if initializers is None:
            initializers = []
        for initializer in onnx_graph.initializer:
            initializers.append(initializer)
        for node in onnx_graph.node:
            for attribute in node.attribute:
                if getattr(attribute, 'g').name != '':
                    OnnxSaver._get_all_initializers(attribute.g, initializers)
        return initializers

    @staticmethod
    def _populate_node_to_io_tensor_map_and_valid_param_set(
            onnx_graph: onnx.GraphProto, initializer_names: List[str],
            node_to_io_tensor_name_map: Dict[str, OpToIOTensors],
            valid_param_set: Set[str]):
        """
        Populate the node name to io tensor map and the valid param set.
        :param onnx_graph: Onnx graph used to populate maps
        :param initializer_names: List of initializer names in the graph
        :param node_to_io_tensor_name_map: Map of onnx node names to input and output tensors
        :param valid_param_set: Set containing valid parameter names in the graph
        """
        for node in onnx_graph.node:
            if node.name:
                onnx_node_io_tensors = OpToIOTensors(list(node.input), list(node.output))
                if (node.name not in node_to_io_tensor_name_map) or node.op_type in recurrent_onnx_optypes:
                    node_to_io_tensor_name_map[node.name] = onnx_node_io_tensors

            # update valid params list
            for input_tensor in list(node.input):
                if input_tensor in initializer_names:
                    valid_param_set.add(input_tensor)

            for attribute in node.attribute:
                if getattr(attribute, 'g').name != '':
                    OnnxSaver._populate_node_to_io_tensor_map_and_valid_param_set(attribute.g, initializer_names,
                                                                                  node_to_io_tensor_name_map,
                                                                                  valid_param_set)

    @classmethod
    def get_onnx_node_to_io_tensor_names_map(cls, onnx_model: onnx.NodeProto) -> \
            (Dict[str, Union[OpToIOTensors, List[OpToIOTensors]]], Set[str]):
        """
        Given an ONNX model, gets the inputs and output tensor names for each node in the model.
        if multiple onnx nodes have the same name then the nodes are provided as a list of inputs and output tensor
         names, one for each onnx node.
        :param onnx_model: The ONNX model instance
        :return: Dictionary of ONNX node name and corresponding input and output tensor names and a set with all valid
        param names in model
        """

        node_to_io_tensor_name_map = {}
        valid_param_set = set()
        initializer_names = [ini.name for ini in OnnxSaver._get_all_initializers(onnx_model.graph)]
        OnnxSaver._populate_node_to_io_tensor_map_and_valid_param_set(onnx_model.graph, initializer_names,
                                                                      node_to_io_tensor_name_map, valid_param_set)

        if version.parse(torch.__version__) < version.parse("1.13.0") or not EXPORT_TO_ONNX_DIRECT:
            cls._collate_io_tensors_for_multi_layer_recurrent_nodes(onnx_model, node_to_io_tensor_name_map)

        return node_to_io_tensor_name_map, valid_param_set

    @staticmethod
    def _get_updated_name(name: str) -> str:
        """
        Remove 'marked_module' from given name.
        :param name: Name.
        :return: Updated name.
        """
        if 'marked_module.' in name:
            name = name.replace('marked_module.', '')
        if 'marked_module/' in name:
            name = name.replace('marked_module/', '')
        if '/marked_module' in name:
            name = name.replace('/marked_module', '')
        return name

    @staticmethod
    def _export_model_to_onnx(model: Union[torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction],
                              dummy_input: Union[Tuple[Any, ...], torch.Tensor], temp_file: str, is_conditional: bool,
                              onnx_export_args: Union[OnnxExportApiArgs, dict]):
        """
        Export model to ONNX format.

        NOTE:
        1) the ONNX checker is enabled by default in torch version 1.11 onwards and
        'enabled_onnx_checker' argument is removed. thus, added try - except block to avoid onnx checker related
         errors if any.

        2) when torch.onnx.export() is called with ScriptModule/ScriptFunction, 'example_outputs' arg is required
        to set so that the type and shape of the outputs can be captured without executing the model.
        In torch version 1.11 onwards, 'example_outputs' is also removed and determined internally inside the export
        function.

        :param model: model to be exported.
        :param dummy_input: dummy inputs to model.
        :param temp_file: A string containing file name.
        :param is_conditional: True if model is a conditional model, False otherwise
        :param onnx_export_args: Additional kwargs.
        """
        # TODO: remove logic to support for older versions once we upgrade.
        # pylint: disable=no-member
        if isinstance(onnx_export_args, OnnxExportApiArgs):
            kwargs = onnx_export_args.kwargs
        else:
            kwargs = onnx_export_args

        if is_conditional:
            with aimet_torch.utils.in_eval_mode(model), torch.no_grad():
                dummy_output = model(*dummy_input)
            model = torch.jit.script(model)
            kwargs.update({'example_outputs': dummy_output})

        if version.parse(torch.__version__) < version.parse('1.11.0'):
            kwargs.update({'enable_onnx_checker': False})
            torch.onnx.export(model, dummy_input, temp_file, **kwargs)
        else:
            try:
                remove_kwargs = ['enable_onnx_checker', 'example_outputs', 'use_external_data_format']
                for key in remove_kwargs:
                    kwargs.pop(key, None)
                torch.onnx.export(model, dummy_input, temp_file, **kwargs)
            except torch.onnx.CheckerError:
                _logger.warning("ONNX Checker has failed but ONNX graph is still generated.")

def save_initializer_restored_onnx_graph(original_model_path: str,
                                         restored_model_path: str):
    """
    Load original ONNX model path and save restored ONNX model to specific path

    :param original_model_path: Path where the original ONNX artifact was stored
    :param restored_model_path: Path to store restored ONNX artifact
    """
    model = onnx.load(original_model_path)
    restored_model = restore_onnx_graph_initializers(model, inplace=True)
    save_as_external_data = model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF
    onnx.save(restored_model, restored_model_path, save_as_external_data=save_as_external_data)


def restore_onnx_graph_initializers(model: onnx.ModelProto,
                                    inplace: bool = False) -> onnx.ModelProto:
    """
    Copy original model and restore its pruned initializers

    :param model: Original ONNX ModelProto
    :param inplace: Whether to modify ModelProto by inplace manner or not
    :return: Initializer restored ONNX ModelProto
    """
    # pylint: disable=protected-access, no-member
    if not inplace:
        model = copy.deepcopy(model)

    onnx_graph = model.graph

    initializers = OnnxSaver._get_all_initializers(onnx_graph)
    initializer_names = [initializer.name for initializer in initializers]
    pruned_initializer_map = _get_pruned_initializer_map(
        onnx_graph, initializers, initializer_names
    )

    for node in onnx_graph.node:
        for input_tensor in node.input:
            _restore_pruned_initializer(
                onnx_graph, input_tensor, pruned_initializer_map
            )

    # Remove all the detached "Identity" type nodes
    for pruned_initializer_info in pruned_initializer_map.values():
        onnx_graph.node.remove(pruned_initializer_info.identity_node)
        _logger.debug(
            "Added new Initializer `%s` and removing existing Identity node `%s`",
            pruned_initializer_info.initializer.name,
            pruned_initializer_info.identity_node.name,
        )
    return model


def _get_pruned_initializer_map(onnx_graph: onnx.GraphProto,
                                initializers: List[onnx.TensorProto],
                                initializer_names: List[str]) -> Dict[str, PrunedInitializerInfo]:
    """
    Find pruned ONNX initializers by iterating Identity nodes

    :param onnx_graph: ONNX graph
    :param initializers: List of ONNX initializers
    :param initializer_names: List of model initializer names
    :return: Dictionary with output of identity node as key and PrunedInitializerInfo as value
    """
    pruned_initializer_map = {}
    for node in onnx_graph.node:
        if node.op_type == "Identity" and node.input[0] in initializer_names:
            index = initializer_names.index(node.input[0])
            initializer = copy.deepcopy(initializers[index])
            pruned_initializer_map[node.output[0]] = PrunedInitializerInfo(
                initializer, node
            )

    return pruned_initializer_map


def _restore_pruned_initializer(onnx_graph: onnx.GraphProto,
                                input_tensor: str,
                                pruned_initializer_map: Dict[str, PrunedInitializerInfo],
                                new_initializer_name: Optional[str] = None):
    """
    Create new Initializer to restore pruned Initializer

    :param onnx_graph: ONNX graph
    :param input_tensor: Input tensor name
    :param pruned_initializer_map: Dictionary with output of identity node as key and PrunedInitializerInfo as value
    :param new_initializer_name: Name for new initializer
    """
    if result := pruned_initializer_map.get(input_tensor):
        new_initializer = result.initializer
        new_initializer.name = new_initializer_name or input_tensor
        onnx_graph.initializer.append(new_initializer)


def get_tensor_to_consumer_map(op_to_io_tensor_map: Dict[str, Dict]) -> Dict[str, str]:
    """
    Get a dictionary mapping tensor names to names of ops consuming that tensor.

    :param op_to_io_tensor_map: Dictionary mapping op names to IO Tensors
    :return: Dictionary mapping tensor names to names of ops consuming that tensor
    """
    tensor_to_consumer_map = {}
    if version.parse(torch.__version__) >= version.parse("1.13.0") and EXPORT_TO_ONNX_DIRECT:
        for op_name, io_tensors in op_to_io_tensor_map.items():
            for inp in io_tensors.inputs:
                if inp not in tensor_to_consumer_map:
                    tensor_to_consumer_map[inp] = [op_name]
                else:
                    tensor_to_consumer_map[inp].append(op_name)
            for output in io_tensors.outputs:
                if output not in tensor_to_consumer_map:
                    tensor_to_consumer_map[output] = []
    return tensor_to_consumer_map


def get_layers_in_io_tensor_map(op_to_io_tensor_map: Dict) -> Dict[str, str]:
    """
    extract root(layer) names of onnx op names in tensor map
    :param op_to_io_tensor_map: ONNX or Torch Script map of layer name to it's input/output tensors
    :return: a set containing layer names present in io tensor map.
    """
    layers_to_onnx_op_names = {}
    if version.parse(torch.__version__) < version.parse("1.13.0") or not EXPORT_TO_ONNX_DIRECT:
        for name in op_to_io_tensor_map.keys():
            modified_name = name
            if modified_name.endswith('.end'):
                modified_name = modified_name[:-4]
            if name in layers_to_onnx_op_names.keys():
                layers_to_onnx_op_names[modified_name.split('#')[0]].append(name)
            else:
                layers_to_onnx_op_names[modified_name.split('#')[0]] = [name]
    else:
        for name in op_to_io_tensor_map.keys():
            pytorch_name = get_pytorch_name_from_onnx_name(name)
            if pytorch_name in layers_to_onnx_op_names.keys():
                layers_to_onnx_op_names[pytorch_name].append(name)
            else:
                layers_to_onnx_op_names[pytorch_name] = [name]
    return layers_to_onnx_op_names


__deleted_atttributes__ = {
    'simplify_onnx_model': 'aimet_torch==2.0'
}

def __getattr__(name: str):
    try:
        return globals()[name]
    except KeyError as e:
        if name in __deleted_atttributes__:
            since = __deleted_atttributes__[name]
            msg = f"Attribute '{name}' was deleted from aimet_torch.onnx_utils since {since}"
        else:
            msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg) from e

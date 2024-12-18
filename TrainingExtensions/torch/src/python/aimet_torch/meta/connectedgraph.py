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
"""For constructing a uniform representation of the computational graph for a PyTorch model,
that is easy to navigate and stores information for the purpose of winnowing.
The representation graph consists of nodes that are either 'operation' or 'product';
operations represent a module or a function that generates a tensor, while products represent
the tensors that are either input to the model (input, constant or parameter) or the
result of an operation. Furthermore the graph representation is bi-directional."""

import copy
from collections import defaultdict
from types import SimpleNamespace
import inspect
from typing import Tuple, Union, List, Dict, Type, Optional
import torch

from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_common.connected_graph.connectedgraph import ConnectedGraph as AimetCommonConnectedGraph
from aimet_common.connected_graph.product import Product
from aimet_common.connected_graph.operation import determine_preceding_op_input_product_index_in_multi_input_op
from aimet_common.model_module import PytorchModelModule
from aimet_common.utils import AimetLogger
import aimet_torch._base.nn.modules.custom as aimet_modules
from aimet_torch.meta.operation import Op
from aimet_torch.utils import is_leaf_module, run_hook_for_layers_with_given_input, in_eval_mode, \
    is_torch_nn_leaf_module, is_custom_leaf_module, get_torch_tensortype_shape
from aimet_torch import onnx_utils
import aimet_torch.utils

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ConnectedGraph)

# Global dictionary holding arguments to be used in ConnectedGraph's torch.jit.trace. Can be imported and modified by
# users as needed.
jit_trace_args = {'check_trace': False}

# Dictionary mapping connected graph op types to a tuple consisting of:
# index 0: number of expected inputs
# index 1: whether in the order of inputs to the op, if the expected inputs come at the front of the list (True) or the
# back of the list (False).
# This is needed for removing certain constant inputs to ops which we don't want to exist, for example mangled Conv and
# BN ops which expose parameter inputs as constant inputs to the op.
op_inputs_dict = {
    'Conv': (1, False),
    'ConvTranspose': (1, False),
    'BatchNormalization': (1, False),
    'Gemm': (1, False),
    'Add': (2, True)
}

# When traced, the leaf level module for these ops may have ordering of inputs flipped. We trace into these modules in
# order to determine the true input ordering.
MULTI_INPUT_OPS_TO_PARSE = [aimet_modules.Add, aimet_modules.Multiply, aimet_modules.Subtract,
                            aimet_modules.Divide, aimet_modules.Pow, aimet_modules.IndexSelect, aimet_modules.MatMul,
                            aimet_modules.Concat, aimet_modules.Minimum, aimet_modules.DynamicConv2d]

# We want to consider following operations as leaf nodes while creating op for connected graph.
SKIP_LIST_FOR_SUBGRAPH_TRACE = [aimet_modules.StridedSlice, aimet_modules.GatherNd, aimet_modules.ScatterND,
                                aimet_modules.CustomGather, aimet_modules.DepthToSpaceDCRMode,
                                aimet_modules.RoiAlign, aimet_modules.ChannelShuffle, aimet_modules.NonMaxSuppression,
                                aimet_modules.DepthToSpaceCRDMode, ]


# pylint: disable=too-many-lines
# pylint: disable=protected-access
class OpWithMultipleOutputs(Op):
    """
    Op with multiple outputs to be used temporarily during CG construction. This helps to keep track of multi output
    nodes like TuplePack, Unpack, ListConstruct, etc., and also gives a truer representation of connectivity than
    connected graph can provide. At the end, prior to finishing connected graph construction, there will be a pass to
    consolidate multiple output products into one for each op, and replace OpWithMultipleOutputs with connected graph
    Ops.
    """
    def __init__(self, name: str, dotted_name: str, output_shape,
                 is_anonymous: bool, op_type: str, residing_module: Union[torch.nn.Module, None]):
        super().__init__(name, dotted_name, output_shape, is_anonymous, op_type, residing_module)
        self.output_products = []

class GetAttrNodeInfo:
    """
    Holds information for GetAttr node types.
    """
    def __init__(self, node_alias: str, node_name: str, node_input: Union[None, str]):
        """
        Ex. GetAttr node: %1 : ... prim::GetAttr[name="model"](%self.1)
        node_alias: %1, node_name: model, node_input: %self.1
        :param node_alias: String representing an alias for this node (used as input in other nodes)
        :param node_name: Name of this node as it is referred to in trace attributes
        :param node_input: Inputs to this node. Will be node aliases of other nodes. If this node is a direct descendant
            of the current trace graph being parsed, node_input will point to the current module. Otherwise, it points
            to the direct parent of this node in the trace (following parents upwards will eventually lead to the
            current module).
        """
        self.node_alias = node_alias
        self.node_name = node_name
        self.node_input = node_input


ModuleTensorShapeMapType = Dict[torch.nn.Module, Tuple[List[Union[List, torch.Size]], List[Union[List, torch.Size]]]]


class ConnectedGraph(AimetCommonConnectedGraph):
    """
    For construction of a graph that connects operations together (
        either module or functional) as producers and consumers of tensors.
        Note that the graph has two kinds of nodes: operations and products."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, model: torch.nn.Module, model_input: Union[torch.Tensor, Tuple]):
        """
        Init function for connected graph.

        :param model: Pytorch model to create connected graph from
        :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
        """
        super().__init__()
        self._model_name = type(model).__name__
        # Maps pytorch module names to modules
        self._name_to_module = {}
        # Maps pytorch modules to module names
        self._module_to_name = {}
        self._op_counter = 0
        self._split_count = 0  # Use it in the name of split Ops getting added to the connected graph.
        # Counts number of constant inputs there are in the graph
        self._constant_count = 0

        self._input_structure = None
        self._output_structure = None

        self._generate_module_lookup_table(model)
        with in_eval_mode(model), torch.no_grad():
            self._aimet_defined_modules = \
                tuple(classtype for _, classtype in inspect.getmembers(aimet_modules,
                                                                       lambda m: inspect.isclass(m) and issubclass(m,
                                                                                                                   torch.nn.Module)))
            self._construct_graph(model, model_input)
            del self._aimet_defined_modules

        # List of ops in the order they are traversed using the forward function
        self.ordered_ops = self._get_ordered_ops()
        # Maps pytorch modules to connected graph ops
        self._module_to_op_dict = _create_module_to_op_dict(self.ordered_ops)

    # List of math invariant op types which can remain as functional in pytorch model definition without affecting the
    # outcome of AIMET features.
    math_invariant_types = {
        'size',
        'NumToTensor',
        'view',
        'narrow',
        'reshape',
        'reshape_as',
        'mean',
        'index_select',
        'slice',
        'select',
        'unsqueeze',
        'randn',
        'flatten',
        'permute',
        'squeeze',
        'contiguous',
        'copy',
        'clone',
        'index',
        'ScalarImplicit',
        'transpose',
        'expand',
        'expand_as',
        'DictConstruct',
        'eq',
        'ne',
        'split',
        'chunk'
    }

    # Graph nodes for which we will treat as passthrough and not represent with an Op
    passthrough_graph_nodes = [
        "Int",       # aten::Int
        "t",         # aten::t
        "to",        # aten::to
        "detach",    # aten::detach
        "values",    # aten::values
        "Identity"
    ]

    # Input graph nodes to ignore
    input_graph_nodes_to_ignore = [
        "Constant",     # prim::Constant
    ]

    def __del__(self):
        """
        Destructor of ConnectedGraph class
        break the dependencies of Ops with Product
        """
        for product in self._products.values():
            product.producer = None
            product.set_consumers_to_null()

    def get_op_from_module_name(self, name: str) -> Union[Op, None]:
        """
        Given the name of a operation/module, return the corresponding op in ops dict
        :param name: Pytorch module name
        :return: Connected graph operation corresponding to pytorch module name.  Returns None if not found
        """
        module = self._name_to_module.get(name, None)
        if module:
            return self._module_to_op_dict.get(module, None)
        return None

    def _generate_module_lookup_table(self, model: torch.nn.Module):
        """
        Generates a look up dictionary for getting modules from their names.
        :param model: Pytorch model
        """
        for name, module in model.named_modules(prefix=self._model_name):
            self._name_to_module[name] = module
            self._module_to_name[module] = name

    @staticmethod
    def _generate_module_tensor_shapes_lookup_table(model: torch.nn.Module,
                                                    model_input: Union[torch.Tensor, Tuple]) -> \
            ModuleTensorShapeMapType:
        """
        Generates a look up dictionary for getting module input and output shapes from the module.
        :param model: Pytorch model
        :param model_input: Input to run through forward pass
        :return: Map of modules and input and output tensor shapes obtained from a forward pass
        """
        module_tensor_shapes_map = {}

        def forward_hook(curr_module: torch.nn.Module,
                         inputs: Union[torch.Tensor, List, Dict, None],
                         outputs: Union[torch.Tensor, List, Dict, None]):
            """
            Custom forward hook function to add every module to module-to-tensor shapes dict.
            :param curr_module: Current module being traversed during forward pass.
            :param inputs: tuple of input tensors to the current module
            :param outputs: tuple of output tensors of the current module
            """
            input_shapes = _get_module_tensor_shapes_entry(inputs)
            output_shapes = _get_module_tensor_shapes_entry(outputs)
            if not isinstance(input_shapes, List):
                input_shapes = [input_shapes]
            if not isinstance(output_shapes, List):
                output_shapes = [output_shapes]
            module_tensor_shapes_map[curr_module] = (input_shapes, output_shapes)

        run_hook_for_layers_with_given_input(model, model_input, forward_hook, leaf_node_only=False)

        return module_tensor_shapes_map

    def _construct_graph(self, model: torch.nn.Module, model_input: Union[torch.Tensor, Tuple]):
        """
        Construct connected graph from model and example inputs.

        :param model: Pytorch model to create connected graph from
        :param model_input: Example input to model.  Can be a single tensor or a list/tuple of input tensors
        """
        module_tensor_shapes_map = ConnectedGraph._generate_module_tensor_shapes_lookup_table(model, model_input)
        trace = torch.jit.trace(model, model_input, **jit_trace_args)
        self._parse_top_level_trace(trace, model)
        self._recover_input_output_structure()
        self._optimize_connected_graph()
        self._transform_ops_and_products_to_connected_graph_convention()
        self._fill_op_and_product_properties(module_tensor_shapes_map)

        # In certain models, a 'mangled' version of nodes like Conv or BN appears in the trace which exposes parameters
        # as constant inputs to the node. In such cases, remove constant inputs since param products will be created
        # to track them.
        self._remove_inputs_for_ops()
        # Create parameters for ops such as conv, batchnorm, etc.
        self._create_param_products()

        # For each split in the model, insert a corresponding split Op in the connected graph.
        # pylint: disable=unnecessary-comprehension
        ops_list = [op for op in self._ops.values()]
        producer_to_product_name_map = defaultdict(list)
        for product in self._products.values():
            if product.producer:
                producer_to_product_name_map[product.producer.dotted_name].append(product.name)

        for op in ops_list:
            self._determine_split_behavior_for_op_and_insert_split_op_in_connected_graph(op,
                                                                                         producer_to_product_name_map)

    def _parse_top_level_trace(self, trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule],
                               model: torch.nn.Module):
        """
        Parse the top level trace.
        :param trace: Pytorch JIT trace for model or a submodule
        :param model: Pytorch model to create connected graph from
        """
        module_to_jit_trace = self._generate_trace_lookup_table(model, trace)
        # pylint: disable=unnecessary-comprehension
        top_level_inputs = [inp for inp in trace.graph.inputs()][1:]
        output_map = {}
        for idx, inp in enumerate(top_level_inputs):
            shape = get_torch_tensortype_shape(inp)
            product = self._add_product(f'input_{idx}', shape=shape)
            product.is_model_input = True
            output_map[inp] = product

        _ = self._parse_trace_graph(trace, model, output_map, top_level_inputs, module_to_jit_trace=module_to_jit_trace)

    # pylint: disable=too-many-branches
    def _parse_trace_graph(self, # pylint: disable=too-many-locals
                           trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule],
                           model: torch.nn.Module,
                           output_map: Dict[torch._C.TensorType, Product],
                           higher_level_inputs: List[torch._C.TensorType],
                           module_to_jit_trace: Dict[torch.nn.Module, torch.jit.TracedModule],
                           elementwise_info: Optional[Tuple[torch.nn.Module, torch.nn.Module]] = None):
        """
        Implements a depth-first graph extraction to create ops and products.
        Depth-first extraction is realized using recursion.

        :param trace: Pytorch JIT trace for model or a submodule
        :param model: Pytorch model to create connected graph from
        :param output_map: Dictionary mapping output tensors to products
        :param higher_level_inputs: Corresponding inputs from a higher graph level
        :param module_to_jit_trace: Dictionary mapping torch modules to their traces
        :param elementwise_info: If not None, contains a tuple with the residing module and module of the elementwise
            node currently being processed
        :return: the outputs of the traced module
        """
        # pylint: disable=unnecessary-comprehension
        curr_inputs = [inp for inp in trace.graph.inputs()]

        # curr_inputs[0] corresponds to an identifier for the current graph node.
        assert len(curr_inputs) == len(higher_level_inputs) + 1

        # Inputs to the current trace level will correspond to higher level inputs passed in from an upper trace level.
        # Replace entries of higher level inputs in the output_map with corresponding current level inputs.
        for idx, higher_level_inp in enumerate(higher_level_inputs):
            if higher_level_inp in output_map.keys():
                temp_product = output_map[higher_level_inp]
                del output_map[higher_level_inp]
                output_map[curr_inputs[idx + 1]] = temp_product

        # A map of sub-graph models and node name that requires recursive parsing
        node_name_to_subgraph_model = {}
        # modules that are being referenced within the sub-graph
        node_name_to_module = {curr_inputs[0].debugName(): model}
        # Keep track of output tensors generated from this current trace level. After parsing all nodes, remove all
        # entries in output_map that are contained in this list, except for tensors that are outputted from this graph.
        curr_level_tensors = []

        for node in trace.graph.nodes():
            # pylint: disable=unnecessary-comprehension
            outputs = [output for output in node.outputs()]

            # retrieving a module reference
            if 'GetAttr' in node.kind():
                self._parse_getattr_node(node, curr_inputs, outputs, node_name_to_module, node_name_to_subgraph_model,
                                         module_to_jit_trace, model, output_map, curr_level_tensors)

            # invoking forward method
            elif 'CallMethod' in node.kind():
                # In the case of parsing the interior of an elementwise op, we never expect to see any CallMethod nodes
                assert not elementwise_info
                submodule_outputs = self._parse_callmethod_node(node, trace, node_name_to_module,
                                                                node_name_to_subgraph_model, output_map, model,
                                                                module_to_jit_trace)
                assert len(submodule_outputs) == len(outputs)

                # Output map contains outputs from the parsed callmethod node, which can be different than the outputs
                # listed in the current trace level, if the callmethod node is not a leaf module. Replace the entries
                # for the callmethod's internal outputs with the outputs for the callmethod shown at the current trace
                # level.
                for idx, submodule_output in enumerate(submodule_outputs):
                    temp_product = output_map[submodule_output]
                    del output_map[submodule_output]
                    output_map[outputs[idx]] = temp_product
                    curr_level_tensors.append(outputs[idx])

            # functional operations e.g. cat, size etc
            else:
                aten_nodes = self._find_aten_nodes_in_forward_pass(trace)
                if elementwise_info and node == aten_nodes[0]:
                    # Aten op that corresponds to the elementwise op
                    op_type = self.get_op_type(type(elementwise_info[1]))
                    op = self._create_new_multi_output_op(op_type,
                                                          residing_module=elementwise_info[0],
                                                          op_module=elementwise_info[1])
                else:
                    op_type = self._get_functional_node_type(node)
                    op = self._create_new_multi_output_op(op_type, residing_module=model)
                # For prim and aten nodes, inputs[0] is a regular input to the module, so no need to take inputs[1:]
                # pylint: disable=unnecessary-comprehension
                self._add_products_for_op(op, [inp for inp in node.inputs()], outputs, output_map)
                for output in outputs:
                    curr_level_tensors.append(output)


        # replace entries in output_map for current level inputs back to the higher_level_inputs entries, so that other
        # callmethod nodes in the higher level trace graph can make use of those inputs.
        for idx, higher_level_inp in enumerate(higher_level_inputs):
            if curr_inputs[idx + 1] in output_map.keys():
                temp_product = output_map[curr_inputs[idx + 1]]
                del output_map[curr_inputs[idx + 1]]
                output_map[higher_level_inp] = temp_product

        # Any entries in the output_map which don't show up in the returned tensors for the current graph level will not
        # be used again. Remove them from output_map. Not removing entries will cause us to run into issues if a
        # duplicated non leaf module is seen later on during trace parsing.
        curr_level_outputs = set(trace.graph.return_node().inputs())
        for output in curr_level_tensors:
            if output not in curr_level_outputs:
                del output_map[output]

        return list(trace.graph.return_node().inputs())

    @staticmethod
    def _parse_op_type(node: torch._C.Node) -> str:
        """
        Helper method to extract op type from node info
        :param node: trace graph node
        :return: Op Type string
        """
        # extracting Op type from node.kind string e.g. aten::relu_, aten::size etc
        op_type = node.kind().split("::")[-1].lstrip('_').rstrip('_')
        return op_type

    # pylint: disable=too-many-arguments
    def _parse_getattr_node(self, node: torch._C.Node, inputs: List[torch._C.TensorType],
                            outputs: List[torch._C.TensorType], node_name_to_module: Dict[str, torch.nn.Module],
                            node_name_to_subgraph_model: Dict[str, Tuple[torch.jit.TracedModule, torch._C.Node]],
                            module_to_jit_trace: Dict[torch.nn.Module, torch.jit.TracedModule],
                            residing_module: torch.nn.Module,
                            output_map: Dict[torch._C.TensorType, Product],
                            curr_level_tensors: List[torch._C.TensorType]):
        """
        Parse getattr nodes in the trace graph, adding name to module references and name to subgraph model references
        if it references a subgraph.

        :param node: trace graph node i.e. 'GetAttr' node
        :param inputs: Inputs to the node
        :param outputs: Outputs of the node
        :param node_name_to_module: dictionary of module indexed by output_name referenced in the sub-graph
        :param node_name_to_subgraph_model: dictionary of torch graph nodes index of output_name that have not been
            resolved
        :param module_to_jit_trace: Dictionary mapping torch modules to their traces
        :param residing_module: Module which the node belongs to
        :param output_map: Dictionary mapping high recursion level outputs to lower level equivalent outputs
        :param curr_level_tensors: output tensors generated from the current trace level
        """
        # For GetAttr lines, the output name will be referring to the module, and not the module's output(s)
        assert len(outputs) == 1
        getattr_node_info = ConnectedGraph._get_getattr_node_info(node)
        if getattr_node_info.node_input == inputs[0].debugName():
            getattr_node_info.node_input = None

        subgraph_model = ConnectedGraph._get_module_instance(node, node_name_to_module)
        if isinstance(subgraph_model, torch.Tensor):
            op = self._create_new_multi_output_op('Constant', residing_module=residing_module)
            # pylint: disable=unnecessary-comprehension
            self._add_products_for_op(op, [inp for inp in node.inputs()], outputs, output_map)
            if isinstance(subgraph_model, torch.nn.Parameter):
                op.output_products[0].is_parm = True
            for output in outputs:
                curr_level_tensors.append(output)
            return
        if getattr_node_info.node_alias not in node_name_to_module:
            node_name_to_module[getattr_node_info.node_alias] = subgraph_model
        else:
            raise ValueError("duplicate model for {0} -> {1} and {2}".format(
                getattr_node_info.node_alias, node_name_to_module[getattr_node_info.node_alias],
                subgraph_model))

        # Recursive parsing is not needed 1) if the module is leaf module and
        # module is from torch.nn (Conv2d, Linear etc.) 2) if the module is leaf module and
        # custom module whose forward method has only one functional operation (aimet_modules.Add()).
        # Recursive parsing is needed 1) if the module is not leaf module.
        # 2) If the module is leaf module but has multiple functional operations in
        # forward method.
        # In the case of elementwise ops, we want to express the elementwise op as an Operation, however sometimes
        # it is necessary to parse the interior of the elementwise op's CallMethod to extract information about
        # constants being passed in, or the order in which operands are used.
        if self._is_recursive_parsing_needed(subgraph_model, module_to_jit_trace[subgraph_model]) or \
                self._is_multi_input_op_to_parse(subgraph_model):
            node_name_to_subgraph_model[getattr_node_info.node_alias] = (subgraph_model, getattr_node_info)

    # pylint: disable=too-many-arguments
    def _parse_callmethod_node(self, node: torch._C.Node,
                               trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule],
                               node_name_to_module: Dict[str, torch.nn.Module],
                               node_name_to_subgraph_model: Dict[str, Tuple[torch.jit.TracedModule, torch._C.Node]],
                               output_map: Dict[torch._C.TensorType, Product],
                               residing_module: torch.nn.Module,
                               module_to_jit_trace: Dict[torch.nn.Module, torch.jit.TracedModule]):
        # pylint: disable=too-many-locals
        """
        The call method node signifies invocation of the forward method. Typically the node has the following construct:
            %output_N : Tensor = prim::CallMethod[name="forward"](%output_L, %output_M)
        :param node: trace graph node i.e. 'CallMethod' node
        :param trace: trace of model or submodule
        :param node_name_to_module: dictionary of module indexed by output_name referenced in the sub-graph
        :param node_name_to_subgraph_model: dictionary of torch graph nodes index of output_name that have not been
            resolved
        :param inputs_map: Dictionary mapping low recursion level inputs to higher level equivalent inputs
        :param output_map: Dictionary mapping high recursion level outputs to lower level equivalent outputs
        :param residing_module: Torch module in which the current node is situated
        :param module_to_jit_trace: Dictionary mapping torch modules to their traces
        """
        # pylint: disable=unnecessary-comprehension
        inputs = [inp for inp in node.inputs()]
        # 1st input is a reference on which the call method is being invoked.
        input_name: str = inputs[0].debugName()
        # pylint: disable=unnecessary-comprehension
        outputs = [output for output in node.outputs()]

        # We don't want to further trace some custom implementation from aimet_modules
        if input_name in node_name_to_subgraph_model and \
                not isinstance(node_name_to_subgraph_model[input_name][0], tuple(SKIP_LIST_FOR_SUBGRAPH_TRACE)):
            elementwise_info = None
            subgraph_model, getattr_node_info = node_name_to_subgraph_model[input_name]
            # For elementwise ops, we need to parse the callmethod interior, but want to retain information about the
            # elementwise op's residing module and torch module
            if self._is_multi_input_op_to_parse(subgraph_model):
                elementwise_info = (residing_module, node_name_to_module[input_name])
            trace_levels = [getattr_node_info.node_name]
            # If node_input (input to the current GetAttr node) is None, we are at the topmost level, and can call
            # trace.<current node name> to get the trace for the subgraph. Otherwise, compile a list of node names to
            # call into by following the subgraph_input entries up the chain.
            while getattr_node_info.node_input is not None:
                _, getattr_node_info = node_name_to_subgraph_model[getattr_node_info.node_input]
                # Later on trace levels will be processed with most recent level first, so insert each one at the front.
                trace_levels.insert(0, getattr_node_info.node_name)
            subgraph_trace = trace
            for level in trace_levels:
                subgraph_trace = getattr(subgraph_trace, level)

            # For elementwise ops, we need to parse the callmethod interior, but want to retain information about the
            # elementwise op's residing module and torch module
            if self._is_multi_input_op_to_parse(subgraph_model):
                aten_nodes = self._find_aten_nodes_in_forward_pass(subgraph_trace)
                assert len(aten_nodes) <= 1
                if len(aten_nodes) == 1:
                    elementwise_info = (residing_module, node_name_to_module[input_name])
                else:
                    # This is an elementwise op that does not actually perform aten operations inside.
                    # We see this in case of elementwise Add when it is concatenating two lists for example.
                    # In this case, simply treat the op as a leaf level op without parsing the interior.
                    op_type = self.get_op_type(type(node_name_to_module[input_name]))
                    op = self._create_new_multi_output_op(op_type, residing_module, node_name_to_module[input_name])
                    self._add_products_for_op(op, inputs[1:], outputs, output_map)
                    return outputs

            submodule_outputs = self._parse_trace_graph(subgraph_trace, subgraph_model, output_map, inputs[1:],
                                                        module_to_jit_trace=module_to_jit_trace,
                                                        elementwise_info=elementwise_info)
            return submodule_outputs

        # Op is a leaf level module
        op_type = self.get_op_type(type(node_name_to_module[input_name]))
        op = self._create_new_multi_output_op(op_type, residing_module, node_name_to_module[input_name])
        self._add_products_for_op(op, inputs[1:], outputs, output_map)
        return outputs

    def _add_products_for_op(self, op: OpWithMultipleOutputs, inputs: List[torch._C.TensorType],
                             outputs: List[torch._C.TensorType], output_map: Dict[torch._C.TensorType, Product]):
        """
        Create output products for the op, and link to input products if existing.
        :param op: Op to link input products and create output products for
        :param inputs: List of torch graph inputs
        :param outputs: List of torch graph outputs
        :param output_map: Dictionary mapping torch tensors to Products
        """
        for idx, output in enumerate(outputs):
            shape = get_torch_tensortype_shape(output)
            product = self._add_product(f'{op.name}#{idx}', shape)
            op.output_products.append(product)
            product.producer = op
            output_map[output] = product

        for inp in inputs:
            if inp in output_map:
                inp_product = output_map[inp]
                inp_product.add_consumer(op)
                op.add_input(inp_product)

    @staticmethod
    def _get_attribute_name(node: torch._C.Node) -> Dict[str, str]:
        """
        Retrieve the attributes associated with the graph node
        :param node: trace graph node
        :return: a dictionary of attributes associated with the node
        """
        attributes = {}
        # node description has pseudo-code of the form  '... torch_mangle_2.Module = prim::GetAttr[name="fc"](%self.1)'
        # for the above example attributeNames() iterator should return a string 'name'
        node_desc = str(node)
        for attribute_name in node.attributeNames():
            pattern = attribute_name + '="'
            if pattern in node_desc:
                attributes[attribute_name] = node_desc.split(pattern)[1].split('"')[0]
        return attributes

    @staticmethod
    def _get_module_instance(node: torch._C.Node,
                             node_name_to_module: Dict[str, torch.nn.Module]) -> torch.nn.Module:
        """
        Get the torch.nn.Module referenced by the node.
        :param node: trace graph node
        :param node_name_to_module: dictionary of module index by output_name referenced in the sub-graph
        :return: torch module corresponding to the node
        """
        input_name: str = node.input().debugName()
        attributes = ConnectedGraph._get_attribute_name(node)
        model = node_name_to_module[input_name]
        sub_model = getattr(model, attributes['name'])
        return sub_model

    @staticmethod
    def _get_getattr_node_info(node: torch._C.Node) -> GetAttrNodeInfo:
        """
        Obtain node_alias, node_name, and node_input for the given node.
        :param node: GetAttr node to obtain information for
        :return: GetAttrNodeInfo object containing node information
        """
        # Obtain the input node to the GetAttr node. This will usually refer to the current level module.
        # However, in certain cases such as Sequentials, the index level will also show up as a separate GetAttr
        # node.
        # Ex.
        # %1 : ... prim::GetAttr[name="model"](%self.1)
        # %2 : ... prim::GetAttr[name="_layer0"](%1)
        # Here, to call into %2 from the current trace, we must call .model._layer0. Tracking inputs to
        # the GetAttr nodes tells us this path (%2 comes from %1 which comes from %self.1, the current module)
        # pylint: disable=unnecessary-comprehension
        node_input = [inp for inp in node.inputs()][0].debugName()
        # pylint: disable=unnecessary-comprehension
        node_alias = [output for output in node.outputs()][0].debugName()
        node_name = ConnectedGraph._get_attribute_name(node).get('name')
        return GetAttrNodeInfo(node_alias, node_name, node_input)

    @staticmethod
    def get_op_type(model_cls: Type[torch.nn.Module]) -> str:
        """
        Get connected graph op type for a pytorch module
        :param model_cls: Pytorch module class to get op type for
        :return: Connected graph op type
        """
        # use nominal Op type if its a known type else use torch defined Module name
        if model_cls in onnx_utils.map_torch_types_to_onnx:
            # Currently always taking first element in the list. Check whether we need extra logic for using other
            # elements in the list.
            op_type = onnx_utils.map_torch_types_to_onnx[model_cls][0]
        else:
            op_type = model_cls.__name__
            logger.debug("unknown op_type -- defaulting to class name %s", op_type)
        return op_type

    def _create_new_multi_output_op(self, op_type: str, residing_module: torch.nn.Module,
                                    op_module: Union[None, torch.nn.Module] = None):
        """
        Create a new multi output operation and add it to self._ops
        """
        op_name = op_type + f'_{self._op_counter}'
        self._op_counter += 1
        op = OpWithMultipleOutputs(name=op_name, dotted_name=op_name, output_shape=None, is_anonymous=False,
                                   op_type=op_type, residing_module=residing_module)
        self._ops[op.name] = op
        if op_module:
            op.model_module = PytorchModelModule(op_module)
        return op

    def _recover_input_output_structure(self):
        """
        pass through constructed graph to recover structure of input and output tensors. Creates a map of each input
        tensor to its first consumer op, and each output tensor to its last producer op.
        """
        def get_index_in_op_inputs(op, product):
            if not op:
                return None
            return SimpleNamespace(op=op, index=op.inputs.index(product))

        def get_index_in_op_outputs(op, product):
            if not op:
                return None
            return SimpleNamespace(op=op, index=op.output_products.index(product))

        def flatten_unpack_consumers(construct_op: Op):
            collapsed_outputs = []
            for product in construct_op.output_products:
                consumers = [flatten_unpack_consumers(consumer) \
                                 if consumer and consumer.type in ['TupleUnpack', 'ListUnpack'] \
                                 else get_index_in_op_inputs(consumer, product) for consumer in product.consumers]
                collapsed_outputs.append(consumers if len(consumers) > 0 else None)
            return collapsed_outputs

        def flatten_pack_producers(deconstruct_op: Op):
            return [flatten_pack_producers(product.producer) \
                        if product.producer and product.producer.type in ['TupleConstruct', 'ListConstruct'] \
                        else get_index_in_op_outputs(product.producer, product) for product in deconstruct_op.inputs]

        input_structure = {}
        output_structure = {}
        for name, op in self.get_all_ops().items():
            # If there is an unpack op that takes in a model input, then "flatten" that op to find all downstream
            # consumers
            if op.type in ['TupleUnpack', 'ListUnpack']:
                if len(op.inputs) == 1 and not op.inputs[0].producer:
                    input_structure[op.inputs[0].name] = flatten_unpack_consumers(op)
            # If there is a pack op that produces a model outputs, then "flatten" that op to find all the upstream
            # nodes that produced the components
            if op.type in ['TupleConstruct', 'ListConstruct']:
                if len(op.output_products) == 1 and not op.output_products[0].consumers:
                    output_structure[name] = flatten_pack_producers(op)

        for name, product in self.get_all_products().items():
            # If there is a model input that has not already been captured, then store its consumer ops
            if product.is_model_input and name not in input_structure:
                input_structure[name] = list(filter(
                    lambda consumer: consumer and consumer.type not in ['TupleConstruct', 'ListConstruct'],
                    product.consumers))
                input_structure[name] = [get_index_in_op_inputs(consumer, product) for consumer in input_structure[name]] if len(input_structure[name]) > 0 else [None]
            # If there is a model outputs that has not already been captured, then store its producer ops
            elif not product.consumers and product.producer.name not in output_structure:
                output_structure[product.producer.name] = get_index_in_op_outputs(product.producer, product )

        # graph should only have one model output (could be a tuple or single item). If a single model output cannot
        # be isolated then we do not know which one is correct.
        if len(output_structure) == 1:
            self._output_structure = next(iter(output_structure.values()))
        else:
            logger.warning("Unable to isolate model outputs.")
            self._output_structure = None

        # Remove inputs called "input_i" and populate their contents into a list based on their index
        self._input_structure = [input_structure.pop(f'input_{i}') for i in range(len(input_structure))]
        self._input_structure = self._input_structure[0] if len(self._input_structure) == 1 else self._input_structure
        if len(input_structure) != 0:
            logger.warning("Unable to isolate model inputs.")
            self._input_structure = None

    @property
    def input_structure(self):
        """
        Getter function for input structure
        """
        return self._input_structure

    @property
    def output_structure(self):
        """
        Getter function for output structure
        """
        return self._output_structure

    def _optimize_connected_graph(self):
        """
        Optimization passes through the constructed graph to remove unnecessary nodes
        """
        self._handle_ops_to_ignore()
        self._handle_tuple_and_list_construct_ops()
        self._handle_tuple_and_list_unpack_ops()

    def _transform_ops_and_products_to_connected_graph_convention(self):
        """
        Prior to this point, the ops and products exhibit full connectivity information including allowing ops to have
        multiple distinct products. This step tweaks the ops and products to conform to existing connected graph
        conventions, losing some information along the way.
        """
        self._consolidate_multi_output_op_output_products()
        self._replace_multi_output_ops_with_single_input_ops()
        self._create_products_in_connected_graph_convention()

    def _handle_ops_to_ignore(self):
        """
        Remove passthrough and Constant input ops
        """
        ops_to_remove = []
        for op in self.get_all_ops().values():
            if op.type in self.passthrough_graph_nodes or op.type in self.input_graph_nodes_to_ignore:
                assert len(op.output_products) == 1
                # pylint: disable=unnecessary-comprehension
                consumers = [consumer for consumer in op.output_products[0].consumers]

                if not op.inputs:
                    # Op has no inputs. Simply delete the op, its output product, and the output product from the inputs
                    # of consumer ops.
                    for consumer in consumers:
                        # Check if consumer is not a passthrough or ignore op type. If so, create a constant input into
                        # the consumer.
                        if consumer.type in self.passthrough_graph_nodes or consumer.type in \
                                self.input_graph_nodes_to_ignore:
                            consumer.inputs.remove(op.output_products[0])
                        else:
                            product_index = consumer.inputs.index(op.output_products[0])
                            constant_product = self._add_product(f'constant_{self._constant_count}',
                                                                 op.output_products[0].shape)
                            constant_product._is_const = True
                            constant_product._is_parm = op.output_products[0].is_parm
                            self._constant_count += 1
                            constant_product.add_consumer(consumer)
                            consumer.inputs[product_index] = constant_product

                else:
                    assert len(op.inputs) == 1
                    for consumer in consumers:
                        # Index of consumer's input list corresponding to this op's output product
                        consumer_input_index = consumer.inputs.index(op.output_products[0])
                        # Replace this op's output product in consumer's input with this op's input product
                        consumer.inputs[consumer_input_index] = op.inputs[0]

                    # Get index of op's input product consumer list corresponding to this op
                    op_index = op.inputs[0].consumers.index(op)
                    # Replace this op in the input product consumers list with all consumers of this op's output product
                    op.inputs[0]._consumers[op_index] = consumers
                    op.inputs[0]._consumers = _flatten_lists(op.inputs[0].consumers)

                ops_to_remove.append(op)
                del self._products[op.output_products[0].name]

        for op in ops_to_remove:
            del self._ops[op.name]

    def _handle_tuple_and_list_construct_ops(self):
        """
        Remove tuple and list construct ops, updating parent and children ops and products along the way.
        """
        ops_to_remove = []
        for op in self.get_all_ops().values():
            if op.type in ['TupleConstruct', 'ListConstruct']:
                assert len(op.output_products) == 1
                output = op.output_products[0]
                # pylint: disable=unnecessary-comprehension
                consumers = [consumer for consumer in output.consumers]

                # For each consumer, update their inputs by replacing the connection from Tuple/ListConstruct to the
                # inputs of Tuple/ListConstruct instead
                for consumer in consumers:
                    consumer_input_index = consumer.inputs.index(output)
                    consumer.inputs[consumer_input_index] = op.inputs

                    # Flatten consumer's inputs
                    consumer.inputs = _flatten_lists(consumer.inputs)

                for inp in op.inputs:
                    # For each of op's inputs, replace the input consumers with op's consumers
                    op_index = inp.consumers.index(op)
                    inp._consumers[op_index] = consumers
                    inp._consumers = _flatten_lists(inp.consumers)


                # Remove all trace of op and its output products
                ops_to_remove.append(op)
                del self._products[output.name]

        for op in ops_to_remove:
            del self._ops[op.name]

    def _handle_tuple_and_list_unpack_ops(self):
        """
        Remove tuple and list unpack ops, updating parent and children ops and products along the way.
        """
        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-branches
        # For tuple unpack, either input came from tuple pack, or an op that had multiple outputs
        ops_to_remove = []
        for op in self.get_all_ops().values():
            if op.type in ['TupleUnpack', 'ListUnpack']:
                # Two possibilities: 1) The op comes after a module with multiple outputs, given as one torch graph
                # tensor, or 2) The op op comes after a Tuple/List construct op that was already processed, so this op's
                # inputs had already been replaced with multiple constituent products.
                assert len(op.inputs) == len(op.output_products) or len(op.inputs) == 1
                # The op sits after a module that has multiple outputs
                if len(op.inputs) == 1:
                    inp_op = op.inputs[0].producer
                    if not inp_op:
                        # TupleUnpack is taking in a tuple model input and unpacking it, no parent op exists.
                        assert op.inputs[0].is_model_input
                        inp_name = op.inputs[0].name
                    else:
                        assert len(inp_op.output_products) == 1, 'TupleUnpack with one input product has parent op ' \
                                                                 'with multiple output products. This is currently ' \
                                                                 'unhandled.'
                        inp_op.output_products = []
                        inp_name = inp_op.name

                        # Delete the singular input product to the op, and replace it with multiple distinct products
                        # for each output product of the op.
                    del self._products[op.inputs[0].name]
                    for idx, output_product in enumerate(op.output_products):
                        # Create a product for each consumer of tuple unpack, to represent a distinct tensor feeding
                        # into each consumer.
                        new_product = self._add_product(f'{inp_name}_#{idx}', shape=output_product.shape)
                        for consumer in output_product.consumers:
                            # Replace consumer's input product with the input product of this op
                            consumer_input_idx = consumer.inputs.index(output_product)
                            consumer.inputs[consumer_input_idx] = new_product
                            # Add consumer into this op's input product consumers if not already present
                            if consumer not in op.inputs[0].consumers:
                                new_product.add_consumer(consumer)
                        if inp_op is not None:
                            new_product.producer = inp_op
                            inp_op.output_products.append(new_product)
                        else:
                            new_product.is_model_input = True
                        del self._products[output_product.name]
                # The op sits after a Tuple/List construct op that was already processed
                else:
                    for idx, inp in enumerate(op.inputs):
                        # Get output corresponding to the input (1:1 mapping)
                        output_product = op.output_products[idx]
                        for consumer in output_product.consumers:
                            consumer_input_idx = consumer.inputs.index(output_product)
                            consumer.inputs[consumer_input_idx] = inp

                        # Replace current op in input consumers with list of consumers of the corresponding output
                        # product
                        op_index = inp.consumers.index(op)
                        inp._consumers[op_index] = output_product.consumers
                        inp._consumers = _flatten_lists(inp.consumers)

                        del self._products[output_product.name]

                ops_to_remove.append(op)

        for op in ops_to_remove:
            del self._ops[op.name]

    def _consolidate_multi_output_op_output_products(self):
        """
        Combine products of ops with multiple outputs into a single output product.
        """
        for op in self.get_all_ops().values():
            if len(op.output_products) > 1:
                error_message = f'Op {op.name} with multiple outputs detected. Currently, AIMET connected graph does ' \
                                f'not distinguish between different outputs of the same op.'
                logger.debug(error_message)
                products_to_remove = []
                consumers_of_first_output_product = set(op.output_products[0].consumers)
                for output in op.output_products[1:]:
                    for consumer in output.consumers:
                        # Replace the output product entry in consumer's inputs with the op's first output product
                        consumer_input_index = consumer.inputs.index(output)
                        consumer.inputs[consumer_input_index] = op.output_products[0]
                        # Update op's first output product consumer list if it doesn't already contain the consumer
                        if consumer not in consumers_of_first_output_product:
                            op.output_products[0].add_consumer(consumer)
                            consumers_of_first_output_product.add(consumer)
                    products_to_remove.append(output)
                for product in products_to_remove:
                    del self._products[product.name]

    def _replace_multi_output_ops_with_single_input_ops(self):
        """
        Replace all ops in self._ops() to be regular Ops.
        """
        new_ops_dict = {}
        for op in self.get_all_ops().values():
            new_op = Op(op.name, op.dotted_name, op.output_shape, op.is_anonymous, op.type, op.residing_module)
            new_op.model_module = op.model_module
            new_op._inputs = op._inputs
            new_op._op_info = op._op_info

            for input_idx, inp in enumerate(new_op.inputs):
                for consumer_idx, consumer in enumerate(inp.consumers):
                    if consumer == op:
                        new_op.inputs[input_idx]._consumers[consumer_idx] = new_op
            # Op will not have output products if it is a terminating op in the model
            if op.output_products:
                new_op._output = op.output_products[0]
                new_op.output.producer = new_op
            new_ops_dict[new_op.name] = new_op
        self._ops = new_ops_dict

    def _create_products_in_connected_graph_convention(self):
        """
        Existing products may have multiple inputs. Create a set of new products where each product has exactly one
        producer and one consumer, as per existing connected graph construction (only split ops will have multiple
        consumers, which will be added later)
        """
        new_product_dict = {}
        for product in self.get_all_products().values():
            producer = product.producer
            # Input products have no producer
            if producer:
                producer.output = None
                producer_name = producer.name
            else:
                # Input products don't have the #x in their name so we can directly take the product name
                producer_name = product.name
            for consumer in product.consumers:
                new_product = Product(f'{producer_name}_to_{consumer.name}', shape=product.shape)
                new_product.producer = product.producer
                new_product.is_model_input = product.is_model_input
                new_product.is_const = product.is_const
                new_product.is_parm = product.is_parm
                new_product._consumers = [consumer]
                new_product_dict[new_product.name] = new_product
                if producer and not producer.output:
                    producer.output = new_product
                consumer_input_index = consumer.inputs.index(product)
                consumer.inputs[consumer_input_index] = new_product

        self._products = new_product_dict

    def _create_param_products(self):
        """
        For certain ops like convolution, batch norm, and linear, create products for their parameters if they don't
        exist yet.
        """
        for op in self._ops.values():
            module = op.get_module()
            if module is not None:
                name = self._module_to_name.get(module, None)
                if isinstance(op.get_module(), tuple(aimet_torch.utils.modules_to_treat_as_leaf)):
                    for child_name, child_module in op.get_module().named_children():
                        self._create_param_products_helper(op, child_module, name + "." + child_name,
                                                           self.get_op_type(type(child_module)))
                else:
                    self._create_param_products_helper(op, module, name, op.type)

    def _create_param_products_helper(self, conn_graph_op: Op, module: torch.nn.Module, module_name: str, op_type: str):
        """
        Helper to create param products for ops like convolution, batch norm, and linear if they don't exist yet
        :param conn_graph_op: Connected graph whose param products need to be added
        :param module: module of type torch.nn.Module to be used to create the param products.
        There can be several modules in a single connected graph op.
        :param module_name: name of the module.
        :param op_type: type (str) of the op which is stored in the connected graph
        """
        if op_type in ['Conv', 'ConvTranspose', 'BatchNormalization', 'Gemm']:
            if module.weight is not None:
                product_name = module_name + '.weight'
                self._create_and_add_param_product_if_not_exists(conn_graph_op, product_name, list(module.weight.shape))
            if module.bias is not None:
                product_name = module_name + '.bias'
                self._create_and_add_param_product_if_not_exists(conn_graph_op, product_name, list(module.bias.shape))
        if op_type == 'BatchNormalization':
            # If batch_norm, fill in rest of bn params
            if module.running_mean is not None:
                product_name = module_name + '.running_mean'
                self._create_and_add_param_product_if_not_exists(conn_graph_op, product_name,
                                                                 list(module.running_mean.shape))
            if module.running_var is not None:
                product_name = module_name + '.running_var'
                self._create_and_add_param_product_if_not_exists(conn_graph_op, product_name,
                                                                 list(module.running_var.shape))

    def _remove_inputs_for_ops(self):
        """
        Remove certain constant inputs for certain ops.
        """

        for op in self._ops.values():
            num_expected_inputs, start_from_front = op_inputs_dict.get(op.type, (None, None))
            if num_expected_inputs is not None and len(op.inputs) > num_expected_inputs:
                if start_from_front:
                    # Remove trailing inputs so that only num_expected_inputs remains
                    inputs_to_remove = op.inputs[num_expected_inputs:]
                    inputs_to_keep = op.inputs[:num_expected_inputs]
                else:
                    # Remove leading inputs so that only num_expected_inputs remains
                    inputs_to_remove = op.inputs[:-num_expected_inputs]
                    inputs_to_keep = op.inputs[-num_expected_inputs:]

                for inp in inputs_to_remove:
                    del self._products[inp.name]
                op.inputs = inputs_to_keep

    def _create_and_add_param_product_if_not_exists(self, op: Op, product_name: str, shape: List[int]):
        """
        Given a name of a product, create it if it doesn't exist, and attach it to the specified op as a parameter.
        :param op: Op to connect the parameter product to.
        :param product_name: Name of the product to create.
        :param shape: Shape of the product to create.
        """
        if product_name not in self._products.keys():
            product = Product(product_name, shape)
            product.is_parm = True
            product.add_consumer(op)
            op.add_input(product)
            self._products[product_name] = product

    def _fill_op_and_product_properties(self, module_tensor_shapes_map):
        """
        Fill in op properties like output shape, dotted name, groups info. Also fill in product shape information.
        """
        for op in self.get_all_ops().values():
            op_module = op.get_module()
            if op_module:
                assert op_module in module_tensor_shapes_map
                _, output_tensor_shapes = module_tensor_shapes_map[op_module]
                # Temporarily not handling dict types for output tensors.
                if isinstance(output_tensor_shapes, Dict):
                    output_tensor_shapes = None
                # For now, treat only the first output shape as the shape of the node if there is more than one
                # entry
                flattened_shapes = _flatten_lists(output_tensor_shapes)
                op.output_shape = flattened_shapes[0]
                op.dotted_name = self._module_to_name[op_module]
                _fill_groups_info(op, op_module)

            if op.output:
                if op.output.shape is None:
                    op.output.shape = op.output_shape
                elif op.output_shape is None:
                    op.output_shape = op.output.shape
                elif op.output.shape != op.output_shape:
                    logger.debug('Mismatch between existing shape %s for product %s and output shape %s for '
                                 'output of op %s', op.output.shape, op.output.name, op.output_shape,
                                 op.name)

    def _determine_split_behavior_for_op_and_insert_split_op_in_connected_graph(self, op: Op,
                                                                                producer_to_product_name_map: Dict[str, List]):
        """
        Determine if an Op's output is used as an input to more than one Op. If it is, create a Split Op and
        insert it in the connected graph, below this Op.
        Note that the split is done in the forward() function of a model and is NOT a PyTorch OP.
        :param op: Op to check if output is used as an input to more than one op.
        :param producer_to_product_name_map: Dictionary mapping op names to product names which the op produces.
        """

        name = op.name
        dotted_name = op.dotted_name

        # Get the output product names.
        output_product_names = producer_to_product_name_map[dotted_name]

        name_list = []
        for prod_name in output_product_names:
            to_pos = prod_name.find('to')
            first_name = prod_name[:to_pos]
            name_list.append(first_name)

        # Split ops have 2 or more output products
        if len(output_product_names) > 1:
            name_list = [+1 for prod in name_list if name in prod]
            if len(name_list) > 1:
                # Create a Split Op
                split_op = self._create_split_op(op)

                # Insert the Split Op in the connected graph.
                self._insert_split_op_in_connected_graph(op, split_op, producer_to_product_name_map)

    def _create_split_op(self, op: Op) -> Op:
        """
        The op's output is split in the forward function. To model it correctly, create a Split Op.
        :param op: Op to create split op after
        :return: Split op that was created
        """
        split_name_parts = [f"{CG_SPLIT}_", str(self._split_count)]
        split_name = ''.join(split_name_parts)
        self._split_count += 1
        split_dotted_name_parts = [self._model_name, split_name]
        split_dotted_name = '.'.join(split_dotted_name_parts)
        is_anonymous = True
        split_op = Op(name=split_name, dotted_name=split_dotted_name, output_shape=op.output_shape,
                      is_anonymous=is_anonymous, op_type=CG_SPLIT, residing_module=None)
        self._ops[split_name] = split_op
        return split_op

    def _insert_split_op_in_connected_graph(self, preceding_op: Op, split_op: Op,
                                            producer_to_product_name_map: Dict[str, List]):
        """
        Insert a Split Op below the preceding Op in the connected graph.
        :param preceding_op: Op prior to split op
        :param split_op: Split op to insert
        :param producer_to_product_name_map: Dictionary mapping op names to product names which the op produces.
        """

        # Important Notes
        # Op:
        # An Op class represents a module in a model.
        #
        # Product:
        # In this version of the Winnower, the Product class represents the following entities in a model.
        # 1) a Tensor between two modules (in Winnower, 2 Ops).
        # 2) an input
        # 3) a constant
        # 4) a parameter
        #
        # Considering only the definition #1) above, i.e., Product is a Tensor between 2 Ops,
        # an Op's inputs and output are Products.
        # That means, an Op could have multiple input Products and one output Product.
        # Examples of Op with multiple input products: add, cat (Concat)
        # A Product's Producer and Consumer are Ops.
        # A Product could have only one Producer but could have multiple consumers.
        # For example, a Split Op has one output.  The Split Op's output is a Product.
        # That product's single Producer is the Split Op and multiple consumers are the 2 Ops in the 2 branches of
        # the Split, that receive the Split output.

        # Steps:
        # 1. Create a new Product for Split Op's output.
        # 2.This product has multiple consumers. Add the consumers to the Product.
        #   Get the consumers from the op's multiple products.
        # 3. Set the the current Op's output Product's consumer to Split Op. The output product's name must be changed.
        # 4. Set the Split Op's input to point to current Op's output. Its name must be changed.

        # 1. Create a new Product for Split Op's output.
        split_op_product = self._create_split_op_output_product(preceding_op, split_op)
        producer_to_product_name_map[split_op_product.name].append(split_op_product)
        split_op.output = split_op_product

        # 2.This product has multiple consumers. Add the consumers to the Product.
        # Get the consumers from the op's multiple products.

        self._add_consumers_to_split_op_product(preceding_op, split_op_product, producer_to_product_name_map)

        # 3. Create a new product to connect the preceding Op to the Split Op.
        # Set the the preceding Op's output Product's consumer to Split Op.

        # The preceding Op's output products (products, since it was behaving like a Split) are going to be deleted,
        # since a Split is being inserted in the connected graph.
        # Save the preceding Op's output Product shape.
        # This is needed to create the new product from the preceding Op to the newly being inserted Split Op.
        new_product_shape = preceding_op.output.shape

        # Since the preceding Op was behaving like a Split Op, it  would have 2 products with the preceding Op as the
        # producer. Delete these products from the product dictionary.
        preceding_op_product_names = copy.deepcopy(producer_to_product_name_map[preceding_op.dotted_name])
        for name in preceding_op_product_names:
            # Important Notes
            # The following check is needed since ResNet uses the same Relu twice in BasicBlock's forward()
            # Please read the details comments in _add_consumers_to_split_op_product()
            if preceding_op.name in name:
                deleted_product = self._products.pop(name)
                try:
                    producer_to_product_name_map[preceding_op.dotted_name].remove(deleted_product.name)
                except ValueError as e:
                    raise AssertionError(f'Product {deleted_product.name} not found in producer_to_product_name_map') from e

        new_product_name = preceding_op.name + '__to__' + split_op.name
        new_product = self._add_product(new_product_name, new_product_shape)
        new_product.producer = preceding_op
        preceding_op.output = new_product
        preceding_op.output.consumers.append(split_op)

        # 4. Set the Split Op's input to point to current Op's output.
        # new_name = preceding_op.name + '__to__' + split_op.name
        split_op.inputs.append(preceding_op.output)

    def _create_split_op_output_product(self, preceding_op: Op, split_op: Op) -> Product:
        """
        Create output product of the split op and connected it to the split op
        :param preceding_op: Op prior to split op
        :param split_op: Split op to create output product for
        :return: Output product of the split op
        """
        split_op_product_name = split_op.name + '__to__' + 'multiple_ops'
        split_op_product_shape = preceding_op.output.shape
        split_op_product = self._add_product(split_op_product_name, split_op_product_shape)
        split_op_product.producer = split_op
        return split_op_product

    def _add_product(self, name: str, shape: Union[List[int], None]) -> Product:
        """
        Add product to self._products dictionary
        :param name: Name of product
        :param shape: Shape of product
        :return: Product that was created
        """
        assert name not in self._products
        product = Product(name, shape)
        self._products[name] = product
        return product

    def _add_consumers_to_split_op_product(self, preceding_op: Op, split_op_product: Product,
                                           producer_to_product_name_map: Dict[str, List]):
        """
        A Split Op's output product has multiple consumers. Add them to the product.
        :param preceding_op: Op prior to split op
        :param split_op_product: Output product of split op
        :param producer_to_product_name_map: Dictionary mapping op names to product names which the op produces.
        """

        dotted_name = preceding_op.dotted_name
        output_product_names = producer_to_product_name_map[dotted_name]

        # Important Notes
        # ResNet model uses the same Relu twice in the forward function of ResNet's BasicBlock.
        # The first Relu feeds in to the BasicBlock's Conv2.
        # The second Relu's output is split with one branch feeding the next BasicBlock's conv1 and the other
        # branch feeding in to the next BasicBlock's Add.
        # The following line filters out the Relu whose output is NOT split :(
        out_product_names = [name for name in output_product_names if preceding_op.name in name]

        num_products = len(out_product_names)
        consumer_index = 0
        for a_product_index in range(num_products):
            a_product = self.get_product(out_product_names[a_product_index])
            a_consumer = a_product.consumers[0]
            split_op_product.consumers.append(a_consumer)
            # Need to insert the newly created split_op product in the correct input index of the op
            input_product_index = determine_preceding_op_input_product_index_in_multi_input_op(preceding_op,
                                                                                               a_consumer)
            a_consumer.inputs[input_product_index] = split_op_product
            consumer_index += 1

    def _is_recursive_parsing_needed(self, module: torch.nn.Module,
                                     trace: torch.jit.TracedModule) -> bool:
        """
        Utility to decide whether recursive parsing is needed for given module and it's jit trace.
        Recursive parsing is not needed
        1) if the module is leaf module and from torch.nn class (nn.Conv2d, nn.ReLU, nn.rNN etc.)
        2) if the module is leaf module and has only one aten node inside forward method (aimet_modules.Add etc.)

        :param module: PyTorch module.
        :param trace: torch.jit trace of the module
        :return: Boolean whether recursive parsing needed or not. If needed returns True, False otherwise.
        """
        # pylint: disable=import-outside-toplevel, cyclic-import
        from aimet_torch.v2.nn import BaseQuantizationMixin
        if isinstance(module, BaseQuantizationMixin):
            return self._is_recursive_parsing_needed(module.get_original_module(), trace)

        recursive_parsing_needed = True
        if is_torch_nn_leaf_module(module) or \
                is_custom_leaf_module(module, self._find_aten_nodes_in_forward_pass(trace)) or \
                isinstance(module, (self._aimet_defined_modules, tuple(aimet_torch.utils.modules_to_treat_as_leaf))):
            recursive_parsing_needed = False

        return recursive_parsing_needed

    @staticmethod
    def _is_multi_input_op_to_parse(module: torch.nn.Module):
        """
        Check whether the module is an multi input op to parse.

        :param module: module to check
        :return: True if module is an multi input op to parse.
        """

        return isinstance(module, tuple(MULTI_INPUT_OPS_TO_PARSE))

    def _generate_trace_lookup_table(self,
                                     model: torch.nn.Module,
                                     trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule]):
        """
        Generate pytorch module names to corresponding JIT trace dictionary. There will be always one to one
        mapping between pytorch module and corresponding JIT trace.

        :param model: PyTorch model.
        :param trace: PyTorch JIT trace.
        """
        def _add_jit_trace(model: torch.nn.Module,
                           trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule]):
            """
            Recursively add jit trace for all the modules to dictionary.
            :param model: PyTorch model or submodule.
            :param trace: PyTorch JIT trace of model.
            """
            for name, module in model.named_children():
                sub_trace = getattr(trace, name)
                module_to_jit_trace[module] = sub_trace

                # recursively call children modules.
                if not is_leaf_module(module):
                    _add_jit_trace(module, sub_trace)

        # Add top level model and corresponding JIT trace.
        module_to_jit_trace = {model: trace}
        # Recursively add children modules and corresponding JIT traces.
        _add_jit_trace(model, trace)

        missing_modules = self._module_to_name.keys() - module_to_jit_trace.keys()

        for m in module_to_jit_trace:
            if is_leaf_module(m):
                missing_modules -= set(m.modules())

        if missing_modules:
            missing_modules = ', '.join([
                self._module_to_name[module] for module in missing_modules
            ])
            raise RuntimeError(f"Couldn't find corresponding JIT trace for modules: {missing_modules}")

        return module_to_jit_trace

    @staticmethod
    def _find_aten_nodes_in_forward_pass(trace: Union[torch.jit.TopLevelTracedModule, torch.jit.TracedModule]) \
            -> List[torch._C.Node]:
        """
        Find all the valid nodes in forward pass for given trace of model or submodule.
        Three possible outcomes:
        1) When the forward method has multiple functional operations and valid nodes are returned.
        2) When the forward method is not defined which causes RuntimeError and empty list is returned.
        3) When the module is unused, empty list is returned.

        :param trace: trace of model or submodule.
        :return: List of trace graph nodes if node.kind() starts with "aten::".
        """
        # pylint: disable=protected-access
        nodes = []
        try:
            nodes = [node for node in trace.graph.nodes() if "aten::" in node.kind() and
                     ConnectedGraph._parse_op_type(node) not in ConnectedGraph.passthrough_graph_nodes]
        except RuntimeError:
            pass
        return nodes

    @staticmethod
    def _get_functional_node_type(node: torch._C.Node) -> str:
        """
        Get the type of a functional node.
        :param node: trace graph node
        :return: The type of the functional node
        """
        op_type = ConnectedGraph._parse_op_type(node)
        # If there is a known mapping to an onnx type for this functional, use the onnx type as the op type
        if op_type in onnx_utils.pytorch_functional_name_to_onnx_dict.keys():
            op_type = onnx_utils.pytorch_functional_name_to_onnx_dict[op_type]
        return op_type

    def _get_ordered_ops(self):
        op_num_dict = {}
        for op in self.get_all_ops().values():
            if op.type != CG_SPLIT:
                last_underscore_idx = op.name.rfind('_')
                op_num = int(op.name[last_underscore_idx + 1:])
                op_num_dict[op_num] = op

        return [op for (num, op) in sorted(op_num_dict.items(), key=lambda x: x[0])]

def _create_module_to_op_dict(ops: List[Op]) -> Dict[torch.nn.Module, Op]:
    """
    Utility to create dictionary mapping pytorch modules to connected graph Ops
    :param ops: List of connected graph Ops
    :return: Dictionary mapping pytorch modules to connected graph Ops
    """
    module_to_op_dict = {}
    for op in ops:
        if op.get_module():
            module_to_op_dict[op.get_module()] = op
    return module_to_op_dict

def _fill_groups_info(op: Op, module: torch.nn.Module):
    """
    Fill in groups info for convolution ops. If op is not a conv op, groups is unchanged.
    :param op: Connected graph op to fill groups info for
    :param module: Pytorch module to check for groups
    """

    if op.type in 'Conv':
        op.groups = module.groups

def _flatten_lists(nested_list: List[List]) -> List:
    """
    Given a list which may contain either an item or a further list of items, generate a list that only contains items
    by bringing all elements of nested lists into the top level list.
    :param nested_list: List containing items or lists of items
    :return: List containing only items (no nested lists)
    """
    flattened_list = []
    for item in nested_list:
        if not isinstance(item, List):
            flattened_list.append(item)
        else:
            flattened_list.extend(_flatten_lists(item))
    return flattened_list

def _get_module_tensor_shapes_entry(tensors: Union[torch.Tensor, List, Dict, None]):
    """
    Given the tensor input or output for a module, extract their shapes and return the shapes in the same list structure
    as the given tensor.
    :param tensors: Input or output tensors for a module
    :return: Shapes of the constituent tensors in tensors
    """
    if tensors is None:
        return None
    if isinstance(tensors, torch.Tensor):
        return tensors.shape
    # Tensors must then be a nested list of tensors, so recursively get the shapes for each item in the list
    if isinstance(tensors, (List, Tuple)):
        return [_get_module_tensor_shapes_entry(entry) for entry in tensors]
    if isinstance(tensors, Dict):
        shapes_dict = {}
        for k, v in tensors.items():
            shapes_dict[k] = _get_module_tensor_shapes_entry(v)
        return shapes_dict
    logger.debug('Unexpected data type for tensor %s. Supported types include tensors, or Lists, Tuples, and Dicts of '
                 'tensors.', type(tensors))
    return None

# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  From PyTorch:
#
#  Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
#  Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
#  Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
#  Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
#  Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
#  Copyright (c) 2011-2013 NYU                      (Clement Farabet)
#  Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
#  Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
#  Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
#  From Caffe2:
#
#  Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
#  All contributions by Facebook:
#  Copyright (c) 2016 Facebook Inc.
#
#  All contributions by Google:
#  Copyright (c) 2015 Google Inc.
#  All rights reserved.
#
#  All contributions by Yangqing Jia:
#  Copyright (c) 2015 Yangqing Jia
#  All rights reserved.
#
#  All contributions by Kakao Brain:
#  Copyright 2019-2020 Kakao Brain
#
#  All contributions by Cruise LLC:
#  Copyright (c) 2022 Cruise LLC.
#  All rights reserved.
#
#  All contributions from Caffe:
#  Copyright(c) 2013, 2014, 2015, the respective contributors
#  All rights reserved.
#
#  All other contributions:
#  Copyright(c) 2015, 2016 the respective contributors
#  All rights reserved.
#
#  Caffe2 uses a copyright model similar to Caffe: each contributor holds
#  copyright over their contributions to Caffe2. The project versioning records
#  all such contribution and copyright details. If a contributor wants to further
#  mark their specific copyright on a particular contribution, they should
#  indicate their copyright solely in the commit message of the change when it is
#  committed.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#     and IDIAP Research Institute nor the names of its contributors may be
#     used to endorse or promote products derived from this software without
#     specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" Implementation to automatically prepare pytorch models for AIMET features """

# --------------------------------------------------------------------------------------------------------
# Reference : https://github.com/pytorch/pytorch/blob/main/torch/fx/proxy.py#L26
#             https://github.com/pytorch/pytorch/blob/main/torch/fx/proxy.py#L57

# Above PyTorch code is used to get node_name_to_scope information by overriding call_module and create_node methods
# of torch.fx.Tracer base class:
# TODO: node_name_to_scope should be removed and instead use node.meta[] after upgrading to torch 2.0
# ----------------------------------------------------------------------------------------------------------

import copy
import re
from typing import Any, Optional, Dict, Union, List, Callable, Tuple
import torch
import torch.fx
from aimet_common.utils import AimetLogger
from aimet_torch.utils import in_eval_mode
from aimet_torch.utils import replace_modules_of_type1_with_type2
import aimet_torch._base.nn.modules.custom as aimet_modules

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.ModelPreparer)

# this is a map of torch.nn.functional type to corresponding module type
functional_op_to_module_map = {
    torch.nn.functional.relu: torch.nn.ReLU,
    torch.nn.functional.gelu: torch.nn.GELU
}

# In this functional --> module map, corresponding model is of type torch.nn and stateful.
functional_with_stateful_api = {
    'relu'          : torch.nn.ReLU,
    'relu6'         : torch.nn.ReLU6,
    'hardtanh'      : torch.nn.Hardtanh,
    'hardwish'      : torch.nn.Hardswish,
    'elu'           : torch.nn.ELU,
    'selu'          : torch.nn.SELU,
    'celu'          : torch.nn.CELU,
    'leaky_relu'    : torch.nn.LeakyReLU,
    'prelu'         : torch.nn.PReLU,
    'rrelu'         : torch.nn.RReLU,
    'glu'           : torch.nn.GLU,
    'gelu'          : torch.nn.GELU,
    'logsigmoid'    : torch.nn.LogSigmoid,
    'hardshrink'    : torch.nn.Hardshrink,
    'tanhshrink'    : torch.nn.Tanhshrink,
    'softsign'      : torch.nn.Softsign,
    'softplus'      : torch.nn.Softplus,
    'softmin'       : torch.nn.Softmin,
    'softmax'       : torch.nn.Softmax,
    'softshrink'    : torch.nn.Softshrink,
    'log_softmax'   : torch.nn.LogSoftmax,
    'tanh'          : torch.nn.Tanh,
    'sigmoid'       : torch.nn.Sigmoid,
    'hardsigmoid'   : torch.nn.Hardsigmoid,
    'silu'          : torch.nn.SiLU,
    'scaled_dot_product_attention': aimet_modules.ScaledDotProductAttention,
}


# Function that requires special transformation.
functional_with_special_handling = {
    'cat'           : aimet_modules.Concat,
    'conv2d'        : torch.nn.Conv2d
}

# In this functional --> module map, corresponding custom module is of type torch.nn and uses stateless API.
functional_with_stateless_api = {
    '_pad'                      : aimet_modules.Pad,
    'pad'                       : aimet_modules.Pad,
    'sum'                       : aimet_modules.Sum,
    'add'                       : aimet_modules.Add,
    'subtract'                  : aimet_modules.Subtract,
    'sub'                       : aimet_modules.Subtract,
    'mul'                       : aimet_modules.Multiply,
    'div'                       : aimet_modules.Divide,
    'truediv'                   : aimet_modules.Divide,
    'floordiv'                  : aimet_modules.FloorDivide,
    'matmul'                    : aimet_modules.MatMul,
    'exp'                       : aimet_modules.Exponential,
    'interpolate'               : aimet_modules.Interpolate,
    'max_pool2d'                : aimet_modules.MaxPool2d,
    'max_pool2d_with_indices'   : aimet_modules.MaxPool2d,
    'adaptive_avg_pool2d'       : aimet_modules.AdaptiveAvgPool2d,
    'avg_pool2d'                : aimet_modules.AvgPool2d,
    'norm'                      : aimet_modules.Norm,
    'batch_norm'                : aimet_modules.BatchNorm,
    'group_norm'                : aimet_modules.GroupNorm,
    'mean'                      : aimet_modules.Mean,
    'pow'                       : aimet_modules.Pow,
    'where'                     : aimet_modules.Where,
    'addmm'                     : aimet_modules.Addmm,
    'bmm'                       : aimet_modules.Bmm,
    'baddbmm'                   : aimet_modules.Baddbmm,
    'cumsum'                    : aimet_modules.CumSum,
    'masked_fill'               : aimet_modules.MaskedFill,
    'square'                    : aimet_modules.Square,
    'rsqrt'                     : aimet_modules.RSqrt,
}


class Scope:
    """
    Code adapted from: https://github.com/pytorch/pytorch/blob/main/torch/fx/proxy.py#L26

    Scope object that records the module path and the module type of module.
    Scope is used to track the information of the module that contains a Node
    in a Graph of GraphModule.
    """
    def __init__(self, module_path: str, module_type: Any):
        super().__init__()
        self.module_path = module_path
        self.module_type = module_type


class ScopeContextManager:
    """
    Code adapted from: https://github.com/pytorch/pytorch/blob/main/torch/fx/proxy.py#L57

    A context manager to track the Scope of Node during symbolic tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.
    """
    def __init__(self, scope: Scope, current_scope: Scope):
        super().__init__()
        # Keep a copy of prev scope.
        self._prev_scope = copy.copy(scope)
        # Update scope to current scope
        scope.module_path = current_scope.module_path
        scope.module_type = current_scope.module_type
        # Save a reference so, we can restore tracer.scope with prev scope on exit.
        self._scope = scope

    def __enter__(self):
        return

    def __exit__(self, *args):
        self._scope.module_path = self._prev_scope.module_path
        self._scope.module_type = self._prev_scope.module_type


def conv2d_create_node(traced_model: torch.fx.GraphModule, module_name: str, node: torch.fx.node) \
        -> torch.fx.node:
    """
    Create the node to be inserted in the graph model.

    :param traced_model: Symbolically traced model
    :param module_name: Qualified module name in symbolic_traced_model hierarchy corresponding to new node
    :param node: Current node in the graph after which new node will be inserted
    :return: torch.fx.node to be inserted in the graph
    """

    n_args = len(node.args)
    # input tensors must be passed as args, not kwargs for QcQuantizeWrapper
    input_tensor = []
    # input and weight is guaranteed to exist, but bias can be None
    # Since None cannot be passed as args in QcQuantizeWrapper, do not add it to input_tensor
    for index, key in [[0, 'input'], [1, 'weight'], [2, ' bias']]:
        value = None
        if n_args > index:
            value = node.args[index]
        elif key in node.kwargs:
            value = node.kwargs[key]

        if value is not None:
            input_tensor.append(value)
        else:
            break

    with traced_model.graph.inserting_after(node):
        if check_dynamic_conv2d(traced_model, module_name):
            new_node = traced_model.graph.call_module(module_name, args=tuple(input_tensor))
        else:
            new_node = traced_model.graph.call_module(module_name, args=tuple([input_tensor[0]]))
        return new_node


def check_dynamic_conv2d(traced_model: torch.fx.GraphModule, module_name: str) -> bool:
    """
    return True if the module is dynamic conv2d.
    """
    m = traced_model
    for name in module_name.split('.'):
        m = getattr(m, name)

    return isinstance(m, aimet_modules.DynamicConv2d)


def conv2d_create_module(node: torch.fx.node) -> torch.nn.Module:
    """
    Create the replacement module.

    :param node: Current node in the graph after which new node will be inserted
    :return: New module.
    """

    # Get weight and bias from argument
    params = merge_args_and_kwargs(node, {1: 'weight', 2: 'bias'})

    # Convert F.Conv2D arguments to nn.Conv2D arguments
    kwargs = merge_args_and_kwargs(node, {3: 'stride', 4: 'padding', 5: 'dilation', 6: 'groups'})

    # If weight or bias is from activation of another layer, use dynamic_conv2d
    use_dynamic_conv2d = False
    for key, param in params.items():
        if param.op != 'get_attr':
            use_dynamic_conv2d = True
            break

    if use_dynamic_conv2d:
        module = aimet_modules.DynamicConv2d(**kwargs)
    else:
        for key, param_node in params.items():
            params[key] = get_node_attr(param_node)

        # Fetch additional info using parameters
        out_channels, in_channels, kernel_size, _ = params['weight'].shape
        bias = 'bias' in params

        # For Depthwise Conv, multiply in_channels by number of groups
        # if groups is not passed as arg, use its default value 1
        kwargs['in_channels'] = in_channels * kwargs.get('groups', 1)
        kwargs['out_channels'] = out_channels
        kwargs['kernel_size'] = kernel_size
        kwargs['bias'] = bias

        module = torch.nn.Conv2d(**kwargs)
        # Replace nn.Conv2D params using F.Conv2D arguments
        module.weight = torch.nn.Parameter(params['weight'])
        if bias:
            module.bias = torch.nn.Parameter(params['bias'])
    return module


def merge_args_and_kwargs(node: torch.fx.node, arguments_to_fetch: Dict) -> Dict:
    """
    Merge args and kwargs into a single kwargs and return it
    :param node: node to fetch args and kwargs from
    :param arguments_to_fetch: dictionary containing arguments' indices in args and keys in kwargs
    :return: single merged kwargs
    """
    n_args = len(node.args)
    kwargs = {}
    for index, key in arguments_to_fetch.items():
        value = None
        if n_args > index:
            value = node.args[index]
        elif key in node.kwargs:
            value = node.kwargs[key]

        if value is not None:
            kwargs[key] = value
    return kwargs


def get_node_attr(node: torch.fx.node):
    """
    Codes modified from https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern

    :param node: node to fetch data from
    :return: value returned from node
    """
    def fetch_attr(target: str):
        target_atoms = target.split('.')
        attr_itr = node.graph.owning_module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    assert node.op == 'get_attr'

    return fetch_attr(node.target)


def concat_create_node(traced_model: torch.fx.GraphModule, module_name: str, node: torch.fx.node) \
        -> torch.fx.node:
    """
    Create the node to be inserted in the graph model.

    :param traced_model: Symbolically traced model
    :param module_name: Qualified module name in symbolic_traced_model hierarchy corresponding to new node
    :param node: Current node in the graph after which new node will be inserted
    :return: torch.fx.node to be inserted in the graph
    """

    with traced_model.graph.inserting_after(node):
        # call_module only accepts tuple as args but node.args[0] can be a list. Convert it into a tuple
        # If node.args[0] is already a tuple, tuple() will do nothing
        new_node = traced_model.graph.call_module(module_name, args=tuple(node.args[0]))
        return new_node


def concat_create_module(node: torch.fx.node) -> torch.nn.Module:
    """
    Create the replacement module.

    :param node: Current node in the graph after which new node will be inserted
    :return: New module.
    """

    num_args = len(node.args)
    if num_args == 1 and 'dim' not in node.kwargs:
        # Handle torch.cat being called with default parameter dim
        kwargs = node.kwargs
        module = aimet_modules.Concat()
    else:
        axis = node.args[1] if num_args > 1 else node.kwargs['dim']
        module = aimet_modules.Concat(axis)
        kwargs = {'axis': axis}

    for key, value in kwargs.items():
        setattr(module, key, value)

    return module

special_handler_functions = {
    # Special handling functions for creating node and module
    'cat': {'node_fn': concat_create_node, 'module_fn': concat_create_module},
    'conv2d': {'node_fn': conv2d_create_node, 'module_fn': conv2d_create_module}
}


def prepare_model(model: torch.nn.Module,
                  modules_to_exclude: List[torch.nn.Module] = None,
                  module_classes_to_exclude: List[Callable] = None,
                  concrete_args: Optional[Dict[str, Any]] = None) -> torch.fx.GraphModule:
    """
    Prepare and modify the pytorch model for AIMET features using torch.FX symbolic tracing API.

    1. Replace torch.nn.functional by module of type torch.nn.Module
    2. Create new independent torch.nn.Module instances for reused/duplicate module

    :param model: pytorch Model to be modified.
    :param modules_to_exclude: List of modules to exclude when tracing.
    :param module_classes_to_exclude: List of module classes to exclude when tracing.
    :param concrete_args: Allows you to partially specialize your function, whether it's to remove control flow or
     data structures. If the model has control flow, torch.fx won't be able to trace the model. Check
     torch.fx.symbolic_trace API in detail.
    :return: Modified pytorch Model
    """
    with in_eval_mode(model):
        traced_model, node_name_to_scope = \
            _trace_model(model, modules_to_exclude, module_classes_to_exclude, concrete_args)

    # Prepare model and perform checks to make sure the graph is well-formed.
    _prepare_traced_model(traced_model, node_name_to_scope)
    return traced_model


def _trace_model(model: torch.nn.Module,
                 modules_to_exclude: Optional[List[torch.nn.Module]],
                 module_classes_to_exclude: Optional[List[Callable]],
                 concrete_args: Optional[Dict[str, Any]]) -> [torch.fx.GraphModule, Dict]:
    """
    Returns traced model and dictionary of node name to the scope of module which contains the node.

    :param model: pytorch Model to be modified.
    :param modules_to_exclude: List of modules to exclude when tracing.
    :param module_classes_to_exclude: List of module classes to exclude when tracing.
    :param concrete_args: Concrete arguments that should not be treated as Proxies.
    :return: (Traced model, node_name_to_scope)
    """
    class Tracer(torch.fx.Tracer):
        """
        Override is_leaf_module(), call_module() and create_node() methods of parent class.
        """
        def __init__(self):
            super().__init__()
            self.scope = Scope("", None)
            self.node_name_to_scope = {}

        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
            return (
                modules_to_exclude and m in modules_to_exclude
                or module_classes_to_exclude and type(m) in module_classes_to_exclude # pylint: disable=unidiomatic-typecheck
                or super().is_leaf_module(m, module_qualified_name)
            )

        def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...],
                        kwargs: Dict[str, Any]) -> Any:
            module_qualified_name = self.path_of_module(m)
            with ScopeContextManager(self.scope, Scope(module_qualified_name, type(m))):
                return super().call_module(m, forward, args, kwargs)

        def create_node(self, kind: str, target, args, kwargs, name: Optional[str] = None,
                        type_expr: Optional[Any] = None) -> torch.fx.Node:
            node = super().create_node(kind, target, args, kwargs, name, type_expr)
            self.node_name_to_scope[node.name] = (self.scope.module_path, self.scope.module_type)
            return node

    # Symbolic tracing frontend - captures the semantics of the module
    tracer = Tracer()
    graph = tracer.trace(model, concrete_args=concrete_args)
    traced_model = torch.fx.GraphModule(tracer.root, graph)
    return traced_model, tracer.node_name_to_scope


def _prepare_traced_model(traced_model: torch.fx.GraphModule,
                          node_name_to_scope: Dict[str, Tuple[str, type]] = None):
    """
    Helper for prepare_model(). This prepares the given traced_model in-place.

    :param traced_model: Symbolically traced model.
    :param node_name_to_scope: Mapping from node name to the scope of module which contains the node.
    """
    unique_nodes = set()

    # Modify the symbolically traced model by iterating over all the nodes
    for node in traced_model.graph.nodes:

        # Create new module for functional nodes
        if node.op in ['call_function', 'call_method']:
            functional_name = _find_functional_name_for_node(node.name)
            if functional_name:
                # Instantiate new module for functional node
                new_module = _create_module_for_functional_node(node, functional_name)
                parent_module, new_module_name, new_module_qualified_name = \
                    _get_info_for_functional_node(traced_model, node, node_name_to_scope)
                setattr(parent_module, new_module_name, new_module)
                # Insert the node for new module in the graph
                _insert_node_for_new_module(traced_model, node, new_module_qualified_name, functional_name)
                logger.info("Functional         : Adding new module for node: {%s} ", new_module_qualified_name)

        # Create new module for reused/duplicate nodes
        elif node.target in unique_nodes:
            if node.op == 'call_module':
                # Instantiate new module for reused node
                new_module = _create_module_for_reused_node(node, traced_model)
                parent_module, new_module_name, new_module_qualified_name = \
                    _get_info_for_reused_node(traced_model, node, node_name_to_scope)
                setattr(parent_module, new_module_name, new_module)
                # Insert the node for new module in the graph
                _insert_node_for_new_module(traced_model, node, new_module_qualified_name)
                logger.info("Reused/Duplicate   : Adding new module for node: {%s} ", new_module_qualified_name)
        else:
            unique_nodes.add(node.target)

    _verify_traced_model(traced_model)

    # Replace SiLU with CustomSiLU
    replace_modules_of_type1_with_type2(traced_model, torch.nn.SiLU, aimet_modules.CustomSiLU)


def _verify_traced_model(traced_model: torch.fx.GraphModule):
    """
    Does some checks to make sure the graph is well-formed and recompile the forward() method of symbolic_traced
    model from its graph

    :param traced_model: Symbolically traced model
    """
    traced_model.graph.lint()
    traced_model.recompile()


def _insert_node_for_new_module(traced_model: torch.fx.GraphModule,
                                node: torch.fx.node,
                                module_qualified_name: str,
                                functional_name: str = None):
    """
    Insert 'call module' node into graph and replace all the uses of 'node' with newly added node and erase the
    old node from graph
    :param traced_model: Symbolically traced model
    :param node: Current node in the graph after which new node will be inserted
    :param module_qualified_name: Qualified module name in symbolic_traced_model hierarchy corresponding to new node
    :param functional_name: Original functional name
    """
    with traced_model.graph.inserting_after(node):
        if functional_name:
            if functional_name in functional_with_special_handling:
                new_node = special_handler_functions[functional_name]['node_fn'](traced_model, module_qualified_name, node)
            elif functional_name in functional_with_stateless_api:
                new_node = traced_model.graph.call_module(module_qualified_name, args=node.args, kwargs=node.kwargs)
            elif functional_name in functional_with_stateful_api:
                new_node = traced_model.graph.call_module(module_qualified_name, args=node.args)
            else:
                raise ValueError("Unsupported module: {}".format(functional_name))
        else:
            new_node = traced_model.graph.call_module(module_qualified_name, args=node.args)

        node.replace_all_uses_with(new_node)
    traced_model.graph.erase_node(node)


def _find_functional_name_for_node(node_name: str) -> Union[str, None]:
    """
    For given node name, find corresponding functional name from combined lookup

    :param node_name: torch.fx Node name
    :return: corresponding functional name if found, else None
    """
    combined_lookup = {**functional_with_stateful_api, **functional_with_special_handling, **functional_with_stateless_api}

    # Functional operations with similar names are differentiated using "_count" suffix
    # when symbolically traced. For example, two add operations will have name 'add' and 'add_1'.
    # Split given node name by occurrence of pattern. \d is used to match [0-9] followed by '_'.
    strings = re.split(pattern=r'_\d', string=node_name)
    for string in strings:
        if string in combined_lookup.keys():
            return string

    logger.debug("Couldn't find functional: %s in the lookup. If functional op isn't math invariant,"
                 " add an entry in the lookup.", node_name)
    return None


def _create_module_for_functional_node(node: torch.fx.node, functional_name: str) -> torch.nn.Module:
    """
    For given node and functional name, create torch.nn.Module with same parameters as functional node parameters
    :param node: torch.fx Node
    :param functional_name: Functional name for given node
    :return: New module
    """
    # Instantiate new module from lookup
    if functional_name in functional_with_stateful_api:
        module = functional_with_stateful_api[functional_name]()
        # Set the parameters for module from node.kwargs
        for key, value in node.kwargs.items():
            setattr(module, key, value)
    elif functional_name in functional_with_special_handling:
        module = special_handler_functions[functional_name]['module_fn'](node)
    elif functional_name in functional_with_stateless_api:
        module = functional_with_stateless_api[functional_name]()
    else:
        raise ValueError("Unsupported module: {}".format(functional_name))
    return module


def _create_module_for_reused_node(node: torch.fx.node, symbolic_traced_model: torch.fx.GraphModule) ->\
        torch.nn.Module:
    """
    For given reused/Duplicate node in symbolically traced model, create new module with same parameters as
    original module
    :param node: Reused/Duplicate torch.fx Node
    :param symbolic_traced_model: Symbolically traced model
    :return: New module
    """
    # Get the original module and return newly deep copied module
    module = _get_module_for_dotted_name(symbolic_traced_model, node.target)
    new_module = copy.deepcopy(module)

    return new_module


def _get_module_for_dotted_name(module: torch.fx.GraphModule, dotted_name: str) -> torch.nn.Module:
    """
    For given dotted name, find the module
    :param module: module to be found
    :param dotted_name: dotted name of module
    :return: module
    """
    if '.' in dotted_name:
        module_name, _, remainder = dotted_name.partition('.')
        return _get_module_for_dotted_name(module._modules[module_name], remainder) # pylint: disable=protected-access

    return getattr(module, dotted_name)


def get_module_for_activation_fn(act_fn: torch.nn.functional):
    """
    returns module instance for functional tyoe handled within PT transformers for activation functions
    :param act_fn: activation function implemented as a functional.
    :return: module equivalent for the activation function.
    """

    if act_fn not in functional_op_to_module_map:
        logger.error("Unsupported activation function {%s}", act_fn)
        return None
    module = functional_op_to_module_map[act_fn]()
    return module


def prepare_pt_transformer_for_quantsim(transformer_model: torch.nn.Module):
    """
    Replaces functionals with modules for activation function, updates model in-place
    :param transformer_model: model with PyTorch nn.Transformer layer
    :return: updated model with modules for activation function.
    """

    for module in transformer_model.modules():

        # encoder layer or decoder layer type is the leaf level node to be updated within nn.transformer layer
        if isinstance(module, torch.nn.TransformerEncoderLayer) and not isinstance(module.activation, torch.nn.Module):
            module.activation = get_module_for_activation_fn(module.activation)

        if isinstance(module, torch.nn.TransformerDecoderLayer) and not isinstance(module.activation, torch.nn.Module):
            module.activation = get_module_for_activation_fn(module.activation)


def _get_info_for_functional_node(traced_model: torch.fx.GraphModule,
                                  node: torch.fx.Node,
                                  node_name_to_scope: Dict[str, Tuple[str, type]])\
        -> Tuple[torch.fx.GraphModule, str, str]:
    """
    For functional node, get module which contains the node, corresponding new module's name and fully qualified name.
    This information will be used to add new module at either module-level scope or model-level scope.

    NOTE: If node_name_to_scope is not provided, then the corresponding new module will be added at model-level scope.
    Also, if exception is raised, new module will be added at model-level scope.

    :param traced_model: Traced model
    :param node: torch.fx Node
    :param node_name_to_scope: Mapping from node name to the scope of module which contains the node.
    :return: (parent_module, new_module_name, new_module_qualified_name)
    """
    parent_module = traced_model
    new_module_name = "module_" + node.name
    new_module_qualified_name = new_module_name

    if node_name_to_scope:
        try:
            module_path, _ = node_name_to_scope[node.name]
            parent_module = traced_model.get_submodule(module_path)
            if module_path == "":
                new_module_qualified_name = new_module_name
            else:
                new_module_qualified_name = module_path + "." + new_module_name
        except (KeyError, AttributeError):
            pass

    return parent_module, new_module_name, new_module_qualified_name


def _get_info_for_reused_node(traced_model: torch.fx.GraphModule,
                              node: torch.fx.Node,
                              node_name_to_scope: Dict[str, Tuple[str, type]])\
        -> Tuple[torch.fx.GraphModule, str, str]:
    """
    For reused node, get module which contains the node, corresponding new module's name and fully qualified name.
    This information will be used to add new module at either module-level scope or model-level scope.

    NOTE: If node_name_to_scope is not provided, then the corresponding new module will be added at model-level scope.
    Also, if exception is raised, new module will be added at model-level scope.

    :param traced_model: Traced model
    :param node: torch.fx Node
    :param node_name_to_scope: Mapping from node name to the scope of module which contains the node.
    :return: (parent_module, new_module_name, new_module_qualified_name)
    """
    parent_module = traced_model
    new_module_name = "module_" + node.name
    new_module_qualified_name = new_module_name

    if node_name_to_scope:
        try:
            module_path, _ = node_name_to_scope[node.name]
            if "." in module_path:
                parent_name, child_name = module_path.rsplit(".", maxsplit=1)
            else:
                parent_name, child_name = "", module_path
            parent_module = traced_model.get_submodule(parent_name)
            new_module_name = "module_" + child_name + "_" + node.name.rsplit("_", maxsplit=1)[1]
            if parent_name == "":
                new_module_qualified_name = new_module_name
            else:
                new_module_qualified_name = parent_name + "." + new_module_name
        except (KeyError, AttributeError):
            pass

    return parent_module, new_module_name, new_module_qualified_name

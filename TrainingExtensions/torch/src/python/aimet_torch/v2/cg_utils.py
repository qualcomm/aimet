# /usr/bin/env python
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
"""Utilities to traverse model graph"""

from typing import Dict, Optional, Generator, Tuple, Union
from dataclasses import dataclass
import functools

import torch

from aimet_common.connected_graph.connectedgraph_utils import CG_SPLIT
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.meta.operation import Op as CG_Op
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.utils import rgetattr, flatten_list, has_no_quantizers, apply_fn_recursively_to_all_elems

@dataclass
class ModuleProduct:
    """ Data structure to store a particular input product to or output product from a torch module"""
    module: torch.nn.Module
    index: int

class ConnectedGraphTraverser:
    """
    GraphTraverser class provides APIs for traversing a model graph
    """
    def __init__(self, sim: QuantizationSimModel):
        self._sim = sim

    def get_leaf_modules(self, torch_module: torch.nn.Module) -> Generator[Tuple[str, torch.nn.Module], None, None]:
        """ Get all the leaf modules in the given module """
        for name, module in torch_module.named_modules():
            if module not in self._sim.model.modules():
                raise ValueError(f"Specified module {module} is not part of the sim object")
            if isinstance(module, BaseQuantizationMixin):
                yield name, module

    def get_modules_of_type(self, module_type):
        """ Get all the modules of given type"""
        for name, module in self._sim.model.named_modules():
            if isinstance(module, BaseQuantizationMixin) and isinstance(module.get_original_module(), module_type):
                yield name, module

    @functools.lru_cache
    def get_module_name(self, inp_module):
        """ Find the name of the provided module"""
        for name, module in self._sim.model.named_modules():
            if inp_module == module:
                return name
        raise RuntimeError("Provided module is not part of the sim object.")

    def get_module_from_cg_op(self, cg_op: CG_Op) -> Optional[torch.nn.Module]:
        """ Find the torch.nn.Module corresponding to the given CG_Op """
        if cg_op is None:
            return None

        module = cg_op.get_module()

        if module is None:
            return None

        fully_qualified_name = self._sim.connected_graph._module_to_name[module] # pylint: disable=protected-access
        _, name = fully_qualified_name.split('.', maxsplit=1)
        quant_module = rgetattr(self._sim.model, name)
        return quant_module

    @functools.cached_property
    def model_inputs(self):
        """
        Returns input structure of the underlying connected graph, with CG_Ops converted to a tuple of the corresponding
        torch.nn.Modules, and the index of the input to the module
        """
        # pylint: disable=protected-access
        return apply_fn_recursively_to_all_elems(
            lambda model_input: ModuleProduct(module=self.get_module_from_cg_op(model_input.op), index=model_input.index),
            self._sim.connected_graph._input_structure)

    @functools.cached_property
    def model_outputs(self):
        """
        Returns output structure of the underlying connected graph, with CG_Ops converted to a tuple of the corresponding
        torch.nn.Modules, and the index of the output from the module
        """
        # pylint: disable=protected-access
        return apply_fn_recursively_to_all_elems(
            lambda model_output: ModuleProduct(module=self.get_module_from_cg_op(model_output.op), index=model_output.index),
            self._sim.connected_graph._output_structure)

    @functools.cached_property
    def model_input_modules(self):
        """
        Returns input structure of the underlying connected graph, with CG_Ops converted to the corresponding
        torch.nn.Modules
        """
        # pylint: disable=protected-access
        return apply_fn_recursively_to_all_elems(lambda model_input: self.get_module_from_cg_op(model_input.op),
                                                  self._sim.connected_graph._input_structure)

    @functools.cached_property
    def model_output_modules(self):
        """
        Returns output structure of the underlying connected graph, with CG_Ops converted to the corresponding
        torch.nn.Modules
        """
        # pylint: disable=protected-access
        return apply_fn_recursively_to_all_elems(lambda model_output: self.get_module_from_cg_op(model_output.op),
                                                  self._sim.connected_graph._output_structure)

    @functools.cached_property
    def module_to_cg_op_mapping(self) -> Dict[torch.nn.Module, CG_Op]:
        """
        Class property that maintains a mapping between torch.nn.Module objects and the corresponding CG_Op
        """
        module_to_op_dict = {}
        for cg_op in self._sim.connected_graph.ordered_ops:
            module = self.get_module_from_cg_op(cg_op)
            if module is not None:
                module_to_op_dict[module] = cg_op
        return module_to_op_dict

    def get_cg_op_from_module(self, module):
        """ Helper functions to lookup CG_Op corresponding to the given module """
        return self.module_to_cg_op_mapping[module]

    def get_valid_parent_module_at_input_idx(self, module, input_idx) -> Union[torch.nn.Module, None]:
        """
        Traverses upstream to determine the parent module provided input idx.
        This method errors out if a functional is encountered which is not a data movement op.

        :param module: torch.nn.Module contained within the QuantSim object
        :param input_idx: input idx to determine the parent module
        :return: parent torch.nn.Module providing input idx
        """
        cg_op = self.get_cg_op_from_module(module)
        parent_cg_op = cg_op.inputs[input_idx].producer

        while parent_cg_op:
            if parent_cg_op.get_module():
                return parent_cg_op.get_module()

            if parent_cg_op.type in ConnectedGraph.math_invariant_types or parent_cg_op.type == CG_SPLIT:
                # Split op or "functional data movement" op is encountered. Query its parent.
                parent_cg_op = parent_cg_op.inputs[0].producer
            else:
                raise RuntimeError(f"Parent of {cg_op.dotted_name} is a functional which is not a data movement op"
                                   f"CG name of the op:{parent_cg_op.dotted_name}. Considering removing this functional "
                                   f"to process")
        return None

    def get_child_module_at_output(self, module):
        """
        Traverses downstream to determine the child modules consuming output

        :param module: torch.nn.Module contained within the QuantSim object
        :return: List of (child torch.nn.Module consuming output, input idx that it is consuming output at)
        """

        def _get_child_modules_from_cg_op(cg_op: CG_Op):
            output_ops = []
            for output_op in cg_op.output_ops:
                output_tensor_name = cg_op.output.name
                output_module = self.get_module_from_cg_op(output_op)

                # this means that the output is being consumed by an implicit op (if output_module is None) OR
                # an op that has no quantizers because it is a data movement or is in a supergroup
                if output_module is None or \
                        has_no_quantizers(output_module) and output_module not in flatten_list([self.model_output_modules]):
                    output_ops.extend(_get_child_modules_from_cg_op(output_op))
                else:
                    for idx, input_tensor in enumerate(output_op.inputs):
                        if input_tensor.name == output_tensor_name:
                            output_ops.append((output_module, idx))
                            break
                    else:
                        # condition is triggered if break statement in loop is not encountered
                        assert False, "Could not match inputs and outputs at adjacent ops. Indicates CG is broken."
            return output_ops

        cg_op = self.get_cg_op_from_module(module)
        return _get_child_modules_from_cg_op(cg_op)

    def topographically_ordered_modules(self) -> Generator[torch.nn.Module, None, None]:
        """
        Generator function to yield all layers in the graph in topographical order
        """
        for cg_op in self._sim.connected_graph.ordered_ops:
            module = self.get_module_from_cg_op(cg_op)
            if module is not None:
                yield module

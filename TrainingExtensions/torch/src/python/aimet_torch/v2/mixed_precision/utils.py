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
"""Utilities to achieve mixed precision"""

# pylint: disable=logging-fstring-interpolation

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Type, List, Literal, Optional, Union, Generator
import functools

import torch

from aimet_common.defs import QuantizationDataType, QuantScheme
from aimet_common.utils import AimetLogger
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantization.float.quantizer import FloatQuantizeDequantize
from aimet_torch.meta.operation import Op as CG_Op
from aimet_torch.quantsim_config.builder import LazyQuantizer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

SupportedDType = Literal['Int16', 'Int8', 'Int4', 'Fp16']

@dataclass
class Precision:
    """ Internal data structure to represent quantization data type and bitwidth """
    data_type: QuantizationDataType
    bitwidth: int

    def __lt__(self, other):
        if self == other:
            return False
        if self.bitwidth != other.bitwidth:
            return self.bitwidth < other.bitwidth
        return self.data_type == QuantizationDataType.int and other.data_type != QuantizationDataType.int


TranslateUserDtypes = {
    'Int16': Precision(QuantizationDataType.int, 16),
    'Int8': Precision(QuantizationDataType.int, 8),
    'Int4': Precision(QuantizationDataType.int, 4),
    'Fp16': Precision(QuantizationDataType.float, 16),
}


@dataclass
class MpRequest:
    """ Internal data structure to save the request to act upon"""
    id: int = None  # original request ID
    input_candidates: List[Precision] = None
    output_candidates: List[Precision] = None
    param_candidate: Dict[str, Precision] = None


class RequestType(Enum):
    """Enum to represent the type of request made by the user"""
    set_precision_by_module = 1
    set_precision_by_module_type = 2
    set_model_input_precision = 3
    set_model_output_precision = 4


@dataclass
class UserRequest:
    """ Data structure to store user requests"""
    request_type: RequestType
    module: Union[torch.nn.Module, Type, None] = None
    activation: Union[List[SupportedDType], SupportedDType, None] = None
    param: Optional[Dict[str, SupportedDType]] = None


def _has_no_quantizers(module: BaseQuantizationMixin, ignore_params: bool = False) -> bool:
    """
    Helper function to check if a module has any quantizers enabled
    """
    return (all(inp_qtzr is None for inp_qtzr in module.input_quantizers) and
            all(out_qtzr is None for out_qtzr in module.output_quantizers) and
            (ignore_params or all(param_qtzr is None for param_qtzr in module.param_quantizers.values())))

def _is_qtzr_higher_precision_than_candidate(qtzr: BaseQuantizationMixin, candidate: Precision) -> bool:
    """ Helper function to determine if qtzr is higher precision than candidate """
    qtzr_dtype = QuantizationDataType.float if isinstance(qtzr, FloatQuantizeDequantize) else QuantizationDataType.int
    generated_candidate = Precision(qtzr_dtype, qtzr.bitwidth)
    return generated_candidate > candidate

# getattr replacement that can handle dotted strings
def _rgetattr(obj, attr):
    return functools.reduce(getattr, [obj] + attr.split('.'))

# setattr replacement that can handle dotted strings
def _rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    pre_obj = _rgetattr(obj, pre) if pre else obj
    return setattr(pre_obj, post, val)

class MpHandler:
    """
    Mixed Precision handler provides the functionalities to generate the Mixed Precision profile from the user provided
    requests and apply to the sim

    """
    def __init__(self, sim: QuantizationSimModel):
        self._sim = sim
        self.mp_requests = {}

    @staticmethod
    def _get_candidate_from_user_dtype(user_dtype: Union[List[SupportedDType], SupportedDType, None] = None):
        """
        Converts user dtype to internal representation in AIMET (QuantizationDataType, Int)

        :param user_dtype: user input for an activation/param
        """
        candidate = None
        if user_dtype:
            if isinstance(user_dtype, List):
                candidate = []
                for dtype in user_dtype:
                    candidate.append(TranslateUserDtypes.get(dtype))
            else:
                candidate = TranslateUserDtypes.get(user_dtype)
        return candidate

    def _get_leaf_modules(self, torch_module: torch.nn.Module) -> List:
        """ Get all the leaf modules in the given module """
        for name, module in torch_module.named_modules():
            if module not in self._sim.model.modules():
                raise ValueError(f"Specified module {module} is not part of the sim object")
            if isinstance(module, BaseQuantizationMixin):
                yield name, module

    def _get_modules_of_type(self, module_type):
        """ Get all the modules of given type"""
        for name, module in self._sim.model.named_modules():
            if isinstance(module, BaseQuantizationMixin) and isinstance(module.get_original_module(), module_type):
                yield name, module

    def _process_user_requests(self, user_requests: Dict[int, UserRequest]):

        def create_mp_request(torch_module: BaseQuantizationMixin, module_name: str, idx: int,
                              activation: Union[List[SupportedDType], SupportedDType, None] = None,
                              param: Optional[Dict[str, SupportedDType]] = None):
            """ For a given leaf module, and the specified activation and param candidates, convert to MpRequest"""
            # TODO fill missing inputs
            if torch_module in mp_requests:
                prev_request = mp_requests[torch_module]
                logger.info(f"{module_name} was already encountered with request_id {prev_request.id} and request "
                            f"{user_requests[prev_request.id]}. This would be replaced with the new request "
                            f"{user_requests[idx]}")

            # multi-inputs would be wrong here
            input_candidates = self._get_candidate_from_user_dtype(activation)
            output_candidates = self._get_candidate_from_user_dtype(activation[0]) \
                if isinstance(activation, List) else self._get_candidate_from_user_dtype(activation)

            # Expectation is that input_candidates and output_candidates are either None or a list with the same number
            # of elements as input/output quantizers (note that each of these list elements could either be a candidate
            # object or None)
            if not isinstance(input_candidates, List):
                input_candidates = [input_candidates] * len(torch_module.input_quantizers)
            if not isinstance(output_candidates, List):
                output_candidates = [output_candidates] * len(torch_module.output_quantizers)

            if len(input_candidates) != len(torch_module.input_quantizers):
                raise RuntimeError(f"Invalid number of activation candidates for module {module_name} provided.")

            param_candidate = \
                {param_name: self._get_candidate_from_user_dtype(dtype) for param_name, dtype in param.items()} \
                if param is not None else None

            mp_requests[torch_module] = MpRequest(id=idx, input_candidates=input_candidates,
                                                  output_candidates=output_candidates,
                                                  param_candidate=param_candidate)

        mp_requests = {}
        for request_id, user_request in user_requests.items():
            if user_request.request_type == RequestType.set_precision_by_module_type:
                for name, module in self._get_modules_of_type(user_request.module):
                    create_mp_request(module, name, request_id, user_request.activation,
                                      user_request.param)
            elif user_request.request_type == RequestType.set_precision_by_module:
                for name, module in self._get_leaf_modules(user_request.module):
                    create_mp_request(module, name, request_id, user_request.activation,
                                      user_request.param)
            elif user_request.request_type == RequestType.set_model_input_precision:
                ...
            elif user_request.request_type == RequestType.set_model_output_precision:
                ...
            else:
                raise RuntimeError(f"Unsupported request type {user_request.request_type} encountered")
        return mp_requests

    # pylint: disable=unused-argument, no-self-use
    def _apply_backend_awareness(self, mp_requests: Dict, config: str = "", strict: bool = True) -> Dict:
        """
        Apply backend awareness to the requests from the user

        :param mp_requests: MP requests generated after processing user requests
        :param config: Config file to be used for backend awareness. If empty no backend awareness would be checked
        :param strict: Boolean flag to indicate whether to fail (strict=True) on incorrect/conflicting inputs made by
        the user or (strict=False) take a best-effort approach to realize the MP settings
        """
        return mp_requests

    @staticmethod
    def _apply_request_to_quantizer(quantizer: QuantizerBase, candidate: Precision):
        """
        Helper function to apply mixed precision candidate to a quantizer
        :param quantizer: quantizer object
        :param candidate: mixed precision candidate
        """
        if candidate.data_type == QuantizationDataType.float:
            if not isinstance(quantizer, FloatQuantizeDequantize):
                # convert to float QDQ
                quantizer = LazyQuantizer(candidate.bitwidth,
                                          'nearest',
                                          QuantScheme.post_training_tf,
                                          quantizer.symmetric,
                                          enabled_by_default=True,
                                          data_type=QuantizationDataType.float
                                          ).realize()

            if candidate.bitwidth == 16:
                quantizer.exponent_bits = 5
                quantizer.mantissa_bits = 10
            elif candidate.bitwidth == 8:
                quantizer.exponent_bits = 4
                quantizer.mantissa_bits = 3
            else:
                assert False, "FP16 and FP8 are the only supported float quantization types."
        else:
            if isinstance(quantizer, FloatQuantizeDequantize):
                # convert to int QDQ
                quantizer = LazyQuantizer(candidate.bitwidth,
                                          'nearest',
                                          QuantScheme.post_training_tf,
                                          quantizer.symmetric,
                                          enabled_by_default=True,
                                          data_type=QuantizationDataType.int
                                          ).realize()

            quantizer.bitwidth = candidate.bitwidth

        return quantizer

    def _get_module_from_cg_op(self, cg_op: CG_Op) -> Optional[torch.nn.Module]:
        if cg_op is None:
            return None

        module = cg_op.get_module()

        if module is None:
            return None

        fully_qualified_name = self._sim.connected_graph._module_to_name[module] # pylint: disable=protected-access
        _, name = fully_qualified_name.split('.', maxsplit=1)
        quant_module = _rgetattr(self._sim.model, name)
        return quant_module

    @functools.cached_property
    def _module_to_cg_op_mapping(self) -> Dict[torch.nn.Module, CG_Op]:
        module_to_op_dict = {}
        for cg_op in self._sim.connected_graph.ordered_ops:
            module = self._get_module_from_cg_op(cg_op)
            if module is not None:
                module_to_op_dict[module] = cg_op
        return module_to_op_dict

    def _get_cg_op_from_module(self, module):
        return self._module_to_cg_op_mapping[module]

    def _get_parent_module_at_input_idx(self, module, input_idx) -> torch.nn.Module:
        """
        Traverses upstream to determine the parent module provided input idx
        :param module: torch.nn.Module contained within the QuantSim object
        :param input_idx: input idx to determine the parent module
        :return: parent torch.nn.Module providing input idx
        """
        cg_op = self._get_cg_op_from_module(module)
        parent_cg_op = cg_op.inputs[input_idx].producer
        parent_module = self._get_module_from_cg_op(parent_cg_op)

        while parent_module is None and parent_cg_op is not None:
            parent_cg_op = parent_cg_op.inputs[0].producer
            parent_module = self._get_module_from_cg_op(parent_cg_op)

        return parent_module

    def _get_child_module_at_output(self, module):
        """
        Traverses downstream to determine the child modules consuming output
        :param module: torch.nn.Module contained within the QuantSim object
        :return: List of (child torch.nn.Module consuming output, input idx that it is consuming output at)
        """

        def _get_child_modules_from_cg_op(cg_op: CG_Op):
            output_ops = []
            for output_op in cg_op.output_ops:
                output_tensor_name = cg_op.output.name
                output_module = self._get_module_from_cg_op(output_op)

                # this means that the output is being consumed by an implicit op (if output_module is None) OR
                # an op that has no quantizers because it is a data movement or is in a supergroup
                if output_module is None or _has_no_quantizers(output_module):
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

        cg_op = self._get_cg_op_from_module(module)
        return _get_child_modules_from_cg_op(cg_op)

    def _topographically_ordered_modules(self) -> Generator[torch.nn.Module, None, None]:
        """
        Generator function to yield all layers in the graph in topographical order
        """
        for cg_op in self._sim.connected_graph.ordered_ops:
            module = self._get_module_from_cg_op(cg_op)
            if module is not None:
                yield module

    @staticmethod
    def _update_request_at_module(mp_requests, module, input_candidates=None, param_candidate=None,
                                  output_candidates=None, strict=False):
        """
        Helper function to update MpRequest for the provided module. If there is already a request for this module,
        it will be updated with the provided fields. Otherwise, a new request will be created
        :param module: torch.nn.Module contained within the QuantSim object
        :param input_candidates: List of tuples containing the input candidates for the module
        :param param_candidate: Dict of tuples containing the param candidates for the module
        :param output_candidates: Tuple containing the output candidate for the module
        """

        def _check_for_overwrites(existing_requests, new_requests):
            """ Helper function to check if new requests are overwriting existing requests"""
            # overwrite not possible if one or both parameters are None
            if existing_requests is None or new_requests is None:
                return False

            if isinstance(existing_requests, dict):
                assert existing_requests.keys() == new_requests.keys()
                for key, candidate in existing_requests.items():
                    # f there are distinct non-None candidates with the same key then there is overwrite
                    if candidate is not None and new_requests[key] is not None:
                        if new_requests[key] != candidate:
                            return True
            elif isinstance(existing_requests, list):
                assert len(existing_requests) == len(new_requests)
                for new_candidate, existing_candidate in zip(new_requests, existing_requests):
                    if new_candidate is not None and existing_candidate is not None:
                        # if there are distinct non-None candidates at the same position then there is overwrite
                        if new_candidate != existing_candidate:
                            return True

            return False

        # create a new request for this module if one does not already exist
        if module not in mp_requests:
            mp_requests[module] = MpRequest(param_candidate=None, input_candidates=None, output_candidates=None)

        if input_candidates is not None:
            if isinstance(input_candidates, Precision):
                input_candidates = [input_candidates] * len(module.input_quantizers)
            assert len(input_candidates) == len(module.input_quantizers)
            if strict and _check_for_overwrites(mp_requests[module].input_candidates, input_candidates):
                raise RuntimeError("Overlapping requests not permitted in strict mode.")
            mp_requests[module].input_candidates = input_candidates

        if param_candidate is not None:
            assert isinstance(param_candidate, dict)
            for key in module.param_quantizers.keys():
                if key not in param_candidate:
                    param_candidate[key] = None
            assert param_candidate.keys() == module.param_quantizers.keys()
            if strict and _check_for_overwrites(mp_requests[module].param_candidates, param_candidate):
                raise RuntimeError("Overlapping requests not permitted in strict mode.")
            param_candidate = {k:v for (k,v) in param_candidate.items() if v}
            mp_requests[module].param_candidate = mp_requests[module].param_candidate | param_candidate

        if output_candidates is not None:
            if isinstance(output_candidates, Precision):
                output_candidates = [output_candidates] * len(module.output_quantizers)
            assert len(output_candidates) == len(module.output_quantizers)
            if strict and _check_for_overwrites(mp_requests[module].output_candidates, output_candidates):
                raise RuntimeError("Overlapping requests not permitted in strict mode.")
            mp_requests[module].output_candidates = output_candidates

    def _propagate_requests_upstream(self, mp_requests: Dict, strict: bool = True):
        """
        Propagate requests to parent modules to achieve precision at given module

        :param mp_requests: MP requests generated after processing user requests
        :param strict: Boolean flag to indicate whether to fail (strict=True) on incorrect/conflicting inputs made by
        the user or (strict=False) take a best-effort approach to realize the MP settings
        """
        def _propagate_request_upstream_helper(module):
            request = mp_requests.get(module)
            if request is None:
                return

            for in_idx, input_candidate in enumerate(request.input_candidates):
                # Do not traverse upward if there is no candidate for this input
                if input_candidate is None:
                    continue

                # Do not traverse upward if this input already has an input quantizer at this module
                if module.input_quantizers[in_idx] is not None:
                    continue

                parent_module = self._get_parent_module_at_input_idx(module, in_idx)
                if parent_module is None:
                    logger.warning(f"Warning: unable to propagate request at {module} upward. "
                                    "Parent module could not be found.")
                    continue

                # TODO: remove this once ops with multiple outputs are supported
                if len(parent_module.output_quantizers) > 1:
                    raise RuntimeError(f"Unable to propagate request at {module} upward. "
                                       f"Parent module has more than one output quantizer.")

                if any(out_qtzr is not None for out_qtzr in parent_module.output_quantizers):
                    # If the parent layer has output quantizers, then we only need to propagate the request until there
                    self._update_request_at_module(mp_requests,
                                                   parent_module,
                                                   output_candidates=input_candidate,
                                                   strict=strict)
                else:
                    # If the parent layer does not have an output quantizer, then we need to propagate the request up to
                    # that layer's inputs
                    self._update_request_at_module(mp_requests,
                                                   parent_module,
                                                   input_candidates=input_candidate,
                                                   output_candidates=input_candidate,
                                                   strict=strict)

                # If the parent layer has no input or output quantizers, then propagate this request further upstream
                # This should only happen if the parent layer is a data movement op
                if _has_no_quantizers(parent_module, ignore_params=True):
                    _propagate_request_upstream_helper(parent_module)

        for module in self._topographically_ordered_modules():
            _propagate_request_upstream_helper(module)
        return mp_requests

    def _resolve_request_outputs(self, mp_requests):
        """
        Determine if output candidates from request at the provided module should be applied or discarded
        :param module: torch.nn.Module contained within the QuantSim object
        """
        def _resolve_request_outputs_helper(module):
            request = mp_requests.get(module)
            if request is None or request.output_candidates is None or module.output_quantizers[0] is None:
                return

            # If the output request at this module came from a downstream consumer, return without changing candidate
            child_modules_and_idxs = self._get_child_module_at_output(module)
            for child_module, input_idx in child_modules_and_idxs:
                child_request = mp_requests.get(child_module)
                if child_request is not None and \
                        child_request.input_candidates[input_idx] == request.output_candidates[0]:
                    return

            # If this output is a model output, return without changing output candidate
            # TODO in subsequent PR (once details of setting model input/output precisions has been resolved)

            # If the output quantizer at this module has a higher precision than the output candidate, return without
            # changing output candidate
            if _is_qtzr_higher_precision_than_candidate(module.output_quantizers[0], request.output_candidates[0]):
                return

            # None of above conditions were met, so discard output_candidate at this module
            request.output_candidates = None

        for module in self._topographically_ordered_modules():
            _resolve_request_outputs_helper(module)

        return mp_requests

    def _apply_requests_to_sim(self, mp_requests: Dict):
        """
        Apply MP configuration to the sim object

        :param mp_requests: MP requests after preprocessing, applying backend awareness(if present), propagating to
        parent modules
        """
        for module, request in mp_requests.items():
            if request.input_candidates is not None:
                assert len(module.input_quantizers) == len(request.input_candidates)
                for idx in range(len(module.input_quantizers)):
                    if request.input_candidates[idx] is not None and module.input_quantizers[idx] is not None:
                        module.input_quantizers[idx] = self._apply_request_to_quantizer(module.input_quantizers[idx],
                                                                                        request.input_candidates[idx])

            if request.param_candidate is not None:
                assert all(param_key in module.param_quantizers for param_key in request.param_candidate.keys())
                for param_key, param_candidate in request.param_candidate.items():
                    if param_candidate is not None and module.param_quantizers[param_key] is not None:
                        module.param_quantizers[param_key] = \
                            self._apply_request_to_quantizer(module.param_quantizers[param_key], param_candidate)

            if request.output_candidates is not None:
                assert len(module.output_quantizers) == len(request.output_candidates)
                for idx in range(len(module.output_quantizers)):
                    if request.output_candidates[idx] is not None and module.output_quantizers[idx] is not None:
                        module.output_quantizers[idx] = self._apply_request_to_quantizer(module.output_quantizers[idx],
                                                                                        request.output_candidates[idx])

    def apply(self, user_requests: Dict[int, UserRequest], config: str = "", strict: bool = True,
              log_file: str = './mmp_log.txt'): # pylint: disable=unused-argument
        """
        Apply the mp settings specified through the set_precision/set_model_input_precision/set_model_output_precision
        calls to the QuantSim object

        :param user_requests: Dict of request id and user request to apply to sim
        :param config: Config file to be used for backend awareness. If empty no backend awareness would be checked
        :param strict: Boolean flag to indicate whether to fail (strict=True) on incorrect/conflicting inputs made by
        the user or (strict=False) take a best-effort approach to realize the MP settings
        :param log_file: Log file to store the logs
        """
        mp_requests = self._process_user_requests(user_requests)
        mp_requests = self._apply_backend_awareness(mp_requests, config, strict)
        mp_requests = self._propagate_requests_upstream(mp_requests, strict)
        mp_requests = self._resolve_request_outputs(mp_requests)
        self._apply_requests_to_sim(mp_requests)

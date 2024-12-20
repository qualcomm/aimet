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
"""Mixed precision handler class"""
# pylint: disable=logging-fstring-interpolation

import copy
from typing import Dict, List, Tuple, Optional, Union, IO

import torch.nn

from aimet_common.defs import QuantizationDataType, QuantScheme
from aimet_common.utils import AimetLogger
from aimet_torch.v2.nn.modules.custom import QuantizedConcat
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantization.float.quantizer import FloatQuantizeDequantize
from aimet_torch.v2._builder import _V2LazyQuantizer

from aimet_torch.v2.utils import flatten_list, has_no_quantizers
from aimet_torch.v2.cg_utils import ConnectedGraphTraverser
from aimet_torch.v2.mixed_precision.utils import Precision, MpRequest, RequestType, SupportedDType, TranslateUserDtypes
from aimet_torch.v2.mixed_precision.utils import _is_qtzr_higher_precision_than_candidate

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

class MpHandler:
    """
    Mixed Precision handler provides the functionalities to generate the Mixed Precision profile from the user provided
    requests and apply to the sim

    """
    def __init__(self, sim: QuantizationSimModel):
        self.cg_traverser = ConnectedGraphTraverser(sim)
        self.mp_requests = {}

    @staticmethod
    def _get_candidate_from_user_dtype(user_dtype: Union[List[SupportedDType], SupportedDType, None] = None):
        """
        Converts user dtype to internal representation in AIMET (QuantizationDataType, Int)

        :param user_dtype: user input for an activation/param
        """
        candidate = None
        if user_dtype:
            if isinstance(user_dtype, (List, Tuple)):
                candidate = []
                for dtype in user_dtype:
                    candidate.append(TranslateUserDtypes.get(dtype))
            else:
                candidate = TranslateUserDtypes.get(user_dtype)
        return candidate

    def _log_mp_requests(self, mp_requests: dict, heading_str: str, log_file: IO):
        """
        Logs MP requests to log file

        :param mp_requests: MP requests to log
        :param heading_str: string representing the step
        :param log_file: log file to log
        """
        def pretty_print(precision: Optional[Union[List[Precision], Dict[str, Precision], Precision]]):
            if isinstance(precision, List):
                ret = ", ".join([str(p) if p else "-" for p in precision])
            elif isinstance(precision, Dict):
                ret = ", ".join([f"{k}: {str(v)}" for k, v in precision.items()])
            elif isinstance(precision, Precision):
                ret = str(precision)
            else:
                ret = "-"
            return ret

        log_file.write("-" * 150 + "\n")
        log_file.write(f"{heading_str}\n")
        log_file.write("-" * 150 + "\n")

        log_file.write("\n{:<60} {:<10} {:<20} {:<20} {:<25}".format("Layer", "ID", "Inputs", "Outputs", "Params"))
        for module, request in mp_requests.items():
            log_file.write(
                "\n{:<60} {:<10} {:<20} {:<20} {:<25}".format(str(self.cg_traverser.get_cg_op_from_module(module).dotted_name),
                                                              str(request.id),
                                                              pretty_print(request.input_candidates),
                                                              pretty_print(request.output_candidates),
                                                              pretty_print(request.param_candidate)))

        log_file.write("\n\n\n")

    def _process_user_requests(self, user_requests: List, log_file: IO, strict: bool):
        """
        Process user requests and convert them into internal format

        :param user_requests: List of user requests to process
        :param log_file: log file to report layers MP settings
        :param strict: Used only for backend awareness in this method. strict==true would check whether the user input
        is supported through the backed options available for the layer
        """
        # pylint: disable=too-many-statements
        def create_mp_request(torch_module: BaseQuantizationMixin, module_name: str, request_id: int,
                              activation: Union[List[SupportedDType], SupportedDType, None] = None,
                              param: Optional[Dict[str, SupportedDType]] = None):
            """ For a given leaf module, and the specified activation and param candidates, convert to MpRequest"""
            if torch_module in mp_requests:
                prev_request = mp_requests[torch_module]
                logger.info(f"{module_name} was already encountered with request_id {prev_request.id} and request "
                            f"{user_requests[prev_request.id]}. This would be replaced with the new request "
                            f"{user_requests[request_id]}")

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

            param_candidate = {}
            if param:
                for param_name, dtype in param.items():
                    if param_name in torch_module.param_quantizers:
                        param_candidate[param_name] = self._get_candidate_from_user_dtype(dtype)

            mp_requests[torch_module] = MpRequest(id=request_id, input_candidates=input_candidates,
                                                  output_candidates=output_candidates,
                                                  param_candidate=param_candidate)

        def create_mp_io_request(torch_module: BaseQuantizationMixin, io_idx: int, module_name: str, request_id: int,
                                 activation: Union[SupportedDType, None], request_type: RequestType):
            """ For a given module and input/output index, create an MpRequest at the specified input/output"""
            if torch_module in mp_requests:
                prev_request = mp_requests[torch_module]
                logger.info(f"{module_name} was already encountered with request_id {prev_request.id} and request "
                            f"{user_requests[prev_request.id]}. The output activation field of this request"
                            f" would be updated with the new request {user_requests[request_id]}")

            request = MpRequest()
            request.id = request_id

            if request_type == RequestType.set_model_output_precision:
                candidate = self._get_candidate_from_user_dtype(activation)
                assert not isinstance(candidate, List), "Only one candidate can be supplied to create_mp_io_request."

                output_candidates = [None] * len(torch_module.output_quantizers)
                output_candidates[io_idx] = candidate
                request.output_candidates = output_candidates

                # If there are no output qtzrs at this module, then we will need to propagate this request upward
                # (For example, if the last layer in the model is a data movement op)
                # By also adding this request at the input of the module, the upward propagation logic will take effect.
                if all(out_qtzr is None for out_qtzr in torch_module.output_quantizers):
                    input_candidates = [request.output_candidates[io_idx]] * len(torch_module.input_quantizers)
                    request.input_candidates = input_candidates
            elif request_type == RequestType.set_model_input_precision:
                if all(in_qtzr is None for in_qtzr in torch_module.input_quantizers):
                    raise RuntimeError(f"No input quantizers detected at module {module_name}. "
                                       f"Input precision request at this module cannot be realized.")

                candidate = self._get_candidate_from_user_dtype(activation)
                assert not isinstance(candidate, List), "Only one candidate can be supplied to create_mp_io_request."

                input_candidates = [None] * len(torch_module.input_quantizers)
                input_candidates[io_idx] = candidate
                if len(input_candidates) != len(torch_module.input_quantizers):
                    raise RuntimeError(f"Invalid number of activation candidates for module {module_name} provided.")
                request.input_candidates = input_candidates

            mp_requests[torch_module] = request.fuse(mp_requests.get(torch_module))

        mp_requests = {}
        for request_id, user_request in enumerate(user_requests):
            if user_request.request_type == RequestType.set_precision_by_module_type:
                for name, module in self.cg_traverser.get_modules_of_type(user_request.module):
                    create_mp_request(module, name, request_id, user_request.activation,
                                      user_request.param)
            elif user_request.request_type == RequestType.set_precision_by_module:
                for name, module in self.cg_traverser.get_leaf_modules(user_request.module):
                    create_mp_request(module, name, request_id, user_request.activation,
                                      user_request.param)
            elif user_request.request_type == RequestType.set_model_input_precision:
                name = self.cg_traverser.get_module_name(user_request.module.module)
                create_mp_io_request(user_request.module.module, user_request.module.index,  name, request_id,
                                     user_request.activation, RequestType.set_model_input_precision)
            elif user_request.request_type == RequestType.set_model_output_precision:
                name = self.cg_traverser.get_module_name(user_request.module.module)
                create_mp_io_request(user_request.module.module, user_request.module.index, name, request_id,
                                     user_request.activation, RequestType.set_model_output_precision)
            else:
                raise RuntimeError(f"Unsupported request type {user_request.request_type} encountered")

        self._log_mp_requests(mp_requests, "Mixed Precision Requests Before Preprocessing", log_file)
        self._apply_backend_awareness(mp_requests, strict)
        return mp_requests

    def _apply_backend_awareness(self, mp_requests: Dict, strict: bool = True) -> Dict:
        """
        Apply backend awareness to the requests from the user

        :param mp_requests: MP requests generated after processing user requests
        :param strict: Boolean flag to indicate whether to fail (strict=True) on incorrect/conflicting inputs made by
        the user or (strict=False) take a best-effort approach to realize the MP settings
        """

        def validate_supported_kernels_for_module(module, input_activation, param) -> bool:
            # supported_kernels has just one entry for activation and param(and both are required fields).
            # Choosing input activation's first entry and param's weight entry to validate.
            # TODO enhance the logic when supported_kernels schema is improved
            act = input_activation[0] if input_activation and len(input_activation) else None
            weight = param['weight'] if param and 'weight' in param else None

            if not module.supported_kernels:
                return True

            if act and weight:
                input_kernel = ((act.bitwidth, act.data_type), (weight.bitwidth, weight.data_type))
                return input_kernel in module.supported_kernels
            return False

        for m, request in mp_requests.items():
            if not validate_supported_kernels_for_module(m, request.input_candidates, request.param_candidate):
                error_str = (
                    f"For module {self.cg_traverser.get_module_name(m)}, input_candidates {request.input_candidates} and"
                    f" {request.param_candidate} are not valid combination supported by backend. Supported combinations: {m.supported_kernels}")
                if strict:
                    raise RuntimeError(error_str)
                logger.warning(error_str)
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
                quantizer = _V2LazyQuantizer(candidate.bitwidth,
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
                quantizer = _V2LazyQuantizer(candidate.bitwidth,
                                             'nearest',
                                             QuantScheme.post_training_tf,
                                             quantizer.symmetric,
                                             enabled_by_default=True,
                                             data_type=QuantizationDataType.int
                                             ).realize()

            quantizer.bitwidth = candidate.bitwidth

        return quantizer

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
                    if candidate and key in new_requests:
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
            mp_requests[module] = MpRequest()

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
            if strict and _check_for_overwrites(mp_requests[module].param_candidate, param_candidate):
                raise RuntimeError("Overlapping requests not permitted in strict mode.")
            param_candidate = {k:v for (k,v) in param_candidate.items() if v}
            if not mp_requests[module].param_candidate:
                mp_requests[module].param_candidate = param_candidate

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
            if request is None or request.input_candidates is None:
                return

            for in_idx, input_candidate in enumerate(request.input_candidates):
                # Do not traverse upward if there is no candidate for this input
                if input_candidate is None:
                    continue

                # Do not traverse upward if this input already has an input quantizer at this module
                if module.input_quantizers[in_idx] is not None:
                    continue

                parent_module = self.cg_traverser.get_valid_parent_module_at_input_idx(module, in_idx)
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
                if has_no_quantizers(parent_module, ignore_params=True):
                    _propagate_request_upstream_helper(parent_module)

        for module in self.cg_traverser.topographically_ordered_modules():
            _propagate_request_upstream_helper(module)
        return mp_requests

    def _get_child_module_and_idx(self, module: torch.nn.Module):
        """
        Helper to get the child module and their input idxes consistent with QuantSim interpretation

        :param module: module to return the child modules and their idxes
        """
        child_module_idxs = self.cg_traverser.get_child_module_at_output(module)
        # Even if concat op has more than one input, in QuantSim there is only one quantizer added.
        # This check always returns idx=0 for those modules
        updated_child_module_idxs = []
        for child_module, input_idx in child_module_idxs:
            if isinstance(child_module, QuantizedConcat):
                input_idx = 0
            updated_child_module_idxs.append((child_module, input_idx))
        return updated_child_module_idxs


    def _resolve_request_outputs(self, mp_requests, log_file: IO):
        """
        Determine if output candidates from request at the provided module should be applied or discarded

        :param mp_requests: MP requests dict with module as key and its request as value
        :param log_file: log file to write the logs into
        """
        def _resolve_request_outputs_helper(module):
            request = mp_requests.get(module)
            if request is None or request.output_candidates is None or module.output_quantizers[0] is None:
                return

            # If the output request at this module came from a downstream consumer, return without changing candidate
            child_modules_and_idxs = self._get_child_module_and_idx(module)
            for child_module, input_idx in child_modules_and_idxs:
                child_request = mp_requests.get(child_module)
                if child_request and child_request.input_candidates and \
                        child_request.input_candidates[input_idx] == request.output_candidates[0]:
                    return

            # If this output is a model output, return without changing output candidate
            if module in flatten_list([self.cg_traverser.model_output_modules]):
                return

            # If the output quantizer at this module has a higher precision than the output candidate, return without
            # changing output candidate
            if _is_qtzr_higher_precision_than_candidate(module.output_quantizers[0], request.output_candidates[0]):
                return

            # None of above conditions were met, so discard output_candidate at this module
            request.output_candidates = None

        for module in self.cg_traverser.topographically_ordered_modules():
            _resolve_request_outputs_helper(module)

        self._log_mp_requests(mp_requests, "Mixed Precision Requests After Propagation", log_file)
        return mp_requests

    def _apply_requests_to_sim(self, mp_requests: Dict):
        """
        Apply MP configuration to the sim object
        :param mp_requests: MP requests after preprocessing, applying backend awareness(if present), propagating to
        parent modules
        """
        for module, request in mp_requests.items():
            if request.input_candidates:
                assert len(module.input_quantizers) == len(request.input_candidates)
                for idx in range(len(module.input_quantizers)):
                    if request.input_candidates[idx] and module.input_quantizers[idx]:
                        module.input_quantizers[idx] = self._apply_request_to_quantizer(module.input_quantizers[idx],
                                                                                        request.input_candidates[idx])

            if request.param_candidate:
                assert all(param_key in module.param_quantizers for param_key in request.param_candidate.keys())
                for param_key, param_candidate in request.param_candidate.items():
                    if param_candidate and param_key in module.param_quantizers.keys() and module.param_quantizers[param_key]:
                        module.param_quantizers[param_key] = \
                            self._apply_request_to_quantizer(module.param_quantizers[param_key], param_candidate)

            if request.output_candidates:
                assert len(module.output_quantizers) == len(request.output_candidates)
                for idx in range(len(module.output_quantizers)):
                    if request.output_candidates[idx] and module.output_quantizers[idx]:
                        module.output_quantizers[idx] = self._apply_request_to_quantizer(module.output_quantizers[idx],
                                                                                        request.output_candidates[idx])

    def _resolve_contentions_at_module(self, current_module, mp_request, visited_modules, mp_requests: Dict,
                                       strict: bool = True):
        """
        Helper function to resolve contention at the specified module
        :param current_module: module to resolve contention at
        :param mp_request: request to be applied at current module
        :param visited_modules:  modules already visited during the current request ID traversal
        :param mp_requests: MP requests to resolve the contentions
        :param strict: Resolve contentions in a strict manner
        """
        # pylint: disable=too-many-branches

        def _apply_new_request_for_module(module, request) -> bool:
            """ output is True if request is updated or created afresh"""
            curr_request = mp_requests.get(module)

            request_modified = True
            if not curr_request:
                # module does not have a request. Create a new one based on the request input
                self._update_request_at_module(mp_requests,
                                               module,
                                               request.input_candidates[0] if request.output_candidates and len(request.output_candidates) > 0 else None,
                                               copy.deepcopy(request.param_candidate) if len(module.param_quantizers.keys()) else None,
                                               request.output_candidates[0] if request.output_candidates and len(request.output_candidates) > 0 else None,
                                               strict=strict)
                mp_requests[module].id = request.id

            elif curr_request.id == request.id:
                request_modified = False

            elif curr_request.id < request.id and not curr_request.is_same_precision(request):
                logger.info(
                    f"For module{module}, new request with id:{request.id} overlaps a previous request with "
                    f"id:{curr_request.id} and would be overwritten")
                if strict:
                    raise RuntimeError(f"Conflicting requests with request IDs:{curr_request.id} and {request.id}")
                mp_requests[module] = request
            return request_modified

        if current_module in visited_modules:
            return # already serviced for given id

        current_request = mp_requests.get(current_module)
        visited_modules.add(current_module)

        if current_request and current_request.id > mp_request.id:
            # there is a request already with a higher request ID. No need to resolve contention at the module
            return

        if mp_request.input_candidates:
            # resolve contention at the inputs using input candidates
            for in_idx, input_candidate in enumerate(mp_request.input_candidates):
                parent_module = self.cg_traverser.get_valid_parent_module_at_input_idx(current_module, in_idx)

                # if input candidate is not present in the request (say, the request is from set_model_output_precision) or
                # if input quantizer is present for the layer(along with input candidate) then no need to resolve any contention
                if not input_candidate or (current_module.input_quantizers[in_idx] and not parent_module):
                    continue

                # if parent has output quantizer, propagate this request to all other children
                if any(parent_module.output_quantizers):
                    child_modules_and_idxs = self._get_child_module_and_idx(parent_module)
                    for child_module, _ in child_modules_and_idxs:
                        new_request = MpRequest(id=mp_request.id,
                                                input_candidates=[input_candidate] * len(child_module.input_quantizers),
                                                output_candidates=[input_candidate])
                        if _apply_new_request_for_module(child_module, new_request):
                            self._resolve_contentions_at_module(child_module, new_request, visited_modules, mp_requests,
                                                                strict)
                else:
                    # parent does not have any output quantizers, apply the request to it.
                    new_request = MpRequest(id=mp_request.id,
                                            input_candidates=[input_candidate] * len(parent_module.input_quantizers),
                                            output_candidates=[input_candidate])
                    if _apply_new_request_for_module(parent_module, new_request):
                        self._resolve_contentions_at_module(parent_module, new_request, visited_modules, mp_requests,
                                                            strict)

        if mp_request.output_candidates:
            # resolve at output using output candidate, if the module has output quantizer, then no need to resolve at output
            if not any(current_module.output_quantizers):
                child_modules_and_idxs = self._get_child_module_and_idx(current_module)
                for child_module, _ in child_modules_and_idxs:
                    new_request = MpRequest(id=mp_request.id,
                                            input_candidates=mp_request.output_candidates * len(child_module.input_quantizers),
                                            output_candidates=mp_request.output_candidates)
                    if _apply_new_request_for_module(child_module, new_request):
                        self._resolve_contentions_at_module(child_module, new_request, visited_modules, mp_requests,
                                                            strict)

    def _resolve_contentions(self, mp_requests: Dict, strict: bool, log_file: IO) -> Dict:
        """
        This step resolves the contentions in user requests. Some examples of contentions are below
        1. One tensor leading into two ops
        Op1 -> Op2
            -> Op3
        Here, Op2 and Op3 cannot be simulated in different precisions because both the ops have Op1's output quantizer
        as their input quantizer
        2. Data movement ops
        Regular1 -> Data2 -> Data3 -> Regular4
        Here, {Data2, Data3, Regular4} all need to operate at the same input precision because they can only be
        simulated with the output quantizer of Regular1
        3. Super groups
        Suppose Conv+Relu is a super group
        Op1 -> Conv2 -> Relu3 -> Op4
        In case of super groups, output of Conv2 is disabled since by definition {Conv2, Relu3} operate at one precision.

        :param mp_requests: MP requests to resolve contentions
        :param strict: Boolean flag to indicate whether to fail (strict=True) on contentions or (strict=False) to
        overwrite an older request with the newer one
        """

        requests = []
        for torch_module, mp_request in mp_requests.items():
            requests.append((torch_module, mp_request))
        requests = sorted(requests, key=lambda r : r[1].id)

        prev_id = None
        # maintain a list of visited_modules for a given request_id to not repeat the checks unnecessarily.
        # this needs to be per request_id and not per module because the user might have called set_precision on a non-leaf
        # module where in all the leaf modules would have the same request_id
        visited_modules = set()
        for (torch_module, mp_request) in requests:
            if prev_id != mp_request.id:
                visited_modules = set()
                prev_id = mp_request.id

            self._resolve_contentions_at_module(torch_module, mp_request, visited_modules, mp_requests, strict)

        self._log_mp_requests(mp_requests,"Mixed Precision Requests After Preprocessing", log_file)
        return mp_requests


    def apply(self, log_file: IO, user_requests: List, strict: bool = True):
        """
        Apply the mp settings specified through the set_precision/set_model_input_precision/set_model_output_precision
        calls to the QuantSim object

        :param log_file: Log file to store the logs
        :param user_requests: List of user requests to apply to sim
        :param strict: Boolean flag to indicate whether to fail (strict=True) on incorrect/conflicting inputs made by
        the user or (strict=False) take a best-effort approach to realize the MP settings

        Note: If a layer has multiple requests through different set_precision(...) calls, the latest one would be honored
        In strict==True mode
            - the apply() call would error out if the precision specified is not supported based on the backend config file
            - the apply() call would error out if the parent request needs to be changed to honor a child op's request
        """
        mp_requests = self._process_user_requests(user_requests, log_file, strict)
        mp_requests = self._resolve_contentions(mp_requests, strict, log_file)
        mp_requests = self._propagate_requests_upstream(mp_requests, strict)
        mp_requests = self._resolve_request_outputs(mp_requests, log_file)
        self._apply_requests_to_sim(mp_requests)

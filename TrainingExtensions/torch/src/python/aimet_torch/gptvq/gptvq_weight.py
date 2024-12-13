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
"""Top level API for GPTVQ - Post-Training Quantization (PTQ)"""
import collections
import contextlib
import itertools
import json
import os
import time
from typing import Union, Tuple, Optional, Dict, List, Set, Iterable

import torch
from torch import nn

from aimet_common.defs import QuantScheme
from aimet_common.utils import Spinner, AimetLogger
from aimet_torch import utils
from aimet_torch.gptvq.defs import GPTVQSupportedModules, GPTVQParameters
from aimet_torch.gptvq.gptvq_optimizer import GPTVQOptimizer
from aimet_torch.gptvq.utils import get_module_name_to_hessian_tensor
from aimet_torch._base.quantsim import _QuantizedModuleProtocol
from aimet_torch.save_utils import SaveUtils
from aimet_torch.utils import get_named_module
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantization.affine.quantizer import QuantizeDequantize
from aimet_torch.v2.quantization.tensor import QuantizedTensorBase
from aimet_torch.v2.quantsim import QuantizationSimModel


_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


# pylint: disable=protected-access, too-many-arguments
class GPTVQ:
    """
    Weight rounding mechanism for GPTVQ
    """
    @classmethod
    def apply_gptvq(
        cls,
        model: nn.Module,
        dummy_input: Union[torch.Tensor, Tuple],
        gptvq_params: GPTVQParameters,
        param_encoding_path: str,
        module_names_to_exclude: Optional[List[str]] = None,
        block_level_module_names: Optional[List[List[str]]] = None,
        file_name_prefix: str = "gptvq",
        config_file_path: Optional[str] = None,
    ) -> nn.Module:
        """
        Returns model with optimized weight rounding of GPTVQ supportable modules
        and saves the corresponding parameter quantization encodings to a separate JSON file
        that can be imported by QuantizationSimModel for inference or QAT

        :param model: PyTorch model to GPTVQ
        :param dummy_input: Dummy input to the model. Used to parse model graph. If the model has more than one input,
                            pass a tuple. User is expected to place the tensors on the appropriate device
        :param gptvq_params: Dataclass holding GPTVQ parameters
        :param param_encoding_path: Path where to store parameter encodings
        :param module_names_to_exclude: Module names which are excluded during GPTVQ optimization
        :param block_level_module_names: List of module name lists to optimize block level GPTVQ optimization instead of leaf module level
        :param file_name_prefix: Prefix to use for filename of the encodings file
        :param config_file_path: Configuration file path for model quantizers
        :return: QuantizationSimModel with GPTVQ applied weights and saves corresponding parameter encodings JSON file at provided path
        """
        _logger.info(gptvq_params)

        if module_names_to_exclude is not None:
            cls._validate_module_names(model, module_names_to_exclude, "module_names_to_exclude")

        if block_level_module_names is not None:
            cls._validate_module_names(model, itertools.chain.from_iterable(block_level_module_names), "block_level_module_names")

        module_name_set = cls._get_candidate_module_name_set(model, module_names_to_exclude)
        sim = cls._get_quantsim(model, dummy_input, gptvq_params, config_file_path, module_name_set)
        if module_names_to_exclude is None:
            module_names_to_exclude = []

        with cls._disable_quantizers_for_gptvq_optimization(sim, module_name_set):
            cls._apply_gptvq(model, sim, dummy_input, gptvq_params, set(module_names_to_exclude), block_level_module_names)

        cls._export_encodings_to_json(param_encoding_path, file_name_prefix, sim, gptvq_params.rows_per_block)
        # Restore all nn.Parameters holding DequantizedTensors to hold plain torch.Tensor
        # so as to keep the output as a pure pytorch model
        for qmodule in sim.qmodules():
            for name, param in qmodule.named_parameters():
                if isinstance(param, QuantizedTensorBase):
                    setattr(qmodule, name, nn.Parameter(param.as_subclass(torch.Tensor)))
        SaveUtils.remove_quantization_wrappers(sim.model)
        return sim.model

    @staticmethod
    def _validate_module_names(model: nn.Module, module_names: Iterable[str], parameter_name: str):
        """
        Validate user provided parameter containing module names
        Each module should exist in the model and be a GPTVQ supportable module

        :param model: torch Model
        :param module_names: Iterable of module names
        :param parameter_name: Name of parameter to validate
        :raise ValueError: If module names are not valid
        """
        name_to_module = dict(model.named_modules())
        invalid_module_names = []
        for name in module_names:
            if name in name_to_module and isinstance(name_to_module[name], GPTVQSupportedModules):
                continue
            invalid_module_names.append(name)

        if invalid_module_names:
            msg = (f"Parameter `{parameter_name}` contains invalid module names ({', '.join(invalid_module_names)}) "
                   f"that don't exist in model or aren't GPTVQ supportable")
            raise ValueError(msg)

    @staticmethod
    def _get_candidate_module_name_set(model: nn.Module,
                                       module_names_to_exclude: Optional[List[str]]) -> Set[str]:
        """
        Return module name set considering module_names_to_exclude and block_level_module_names

        :param model: Original model
        :param module_names_to_exclude: Module names which are excluded during GPTVQ optimization
        :return: Module name set considering module_names_to_exclude and block_level_module_names
        """
        possible_module_names = {name for name, module in model.named_modules() if isinstance(module, GPTVQSupportedModules)}
        module_names_to_exclude = set(module_names_to_exclude) if module_names_to_exclude else set()
        return possible_module_names.difference(module_names_to_exclude)

    @classmethod
    def _get_quantsim(cls,
                      model: nn.Module,
                      dummy_input: Union[torch.Tensor, Tuple],
                      gptvq_params: GPTVQParameters,
                      config_file_path: Optional[str],
                      module_name_set: Set[str]) -> QuantizationSimModel:
        """
        Instantiate QuantizationSimModel object and
        replace param quantizers to be compatible with vector quantization

        :param model: Original PyTorch model
        :param dummy_input: Dummy input to be passed in QuantizationSimModel initialization
        :param gptvq_params: Dataclass holding GPTVQ parameters
        :param config_file_path: Config file path to be passed in QuantizationSimModel initialization
        :param module_name_set: Module name set containing candidates of GPTVQ optimization
        :return: QuantizationSimModel with replaced param quantizers
        """
        sim = QuantizationSimModel(
            model,
            dummy_input=dummy_input,
            quant_scheme=QuantScheme.post_training_tf,
            default_param_bw=gptvq_params.vector_bw,
            config_file=config_file_path,
        )
        cls._replace_param_quantizers(sim, gptvq_params.rows_per_block, module_name_set)

        # TODO: Remove this line after fixing root cause of GC block and memory leak in connected graph
        del sim.connected_graph

        return sim

    @staticmethod
    def _replace_param_quantizers(sim: QuantizationSimModel, rows_per_block: int, module_name_set: Set[str]):
        """
        Replace param quantizers to be compatible with vector quantization
        if modules are GPTVQ supportable modules

        :param sim: QuantizationSimModel object
        :param rows_per_block: The number of rows per block
        :param module_name_set: Module name set containing candidates of GPTVQ optimization
        """
        for module_name in module_name_set:
            module = get_named_module(sim.model, module_name)
            assert isinstance(module, BaseQuantizationMixin)

            param_quantizer = module.param_quantizers["weight"]
            num_rows, *remaining_shapes = module.weight.shape
            assert num_rows % rows_per_block == 0, f"The number of rows in weight (#: {num_rows}) should be divided by rows per block (#: {rows_per_block})"
            q = QuantizeDequantize(
                shape=(num_rows // rows_per_block, *[1 for _ in remaining_shapes]),
                bitwidth=param_quantizer.bitwidth,
                symmetric=param_quantizer.symmetric,
                block_size=(rows_per_block, *remaining_shapes),
            ).to(module.weight.device)
            module.param_quantizers["weight"] = q

    @staticmethod
    def _disable_quantizers_for_gptvq_optimization(sim: QuantizationSimModel, module_name_set) -> contextlib.ExitStack:
        """
        Get context managers to disable quantizers temporarily

        :param sim: QuantizationSimModel object
        :return: List of context managers to disable quantizers
        """
        exit_stack = contextlib.ExitStack()
        for name, module in sim.model.named_modules():
            if not isinstance(module, BaseQuantizationMixin):
                continue

            if "weight" in module.param_quantizers and name in module_name_set:
                exit_stack.enter_context(module._remove_activation_quantizers())
            else:
                exit_stack.enter_context(module._remove_all_quantizers())

        return exit_stack

    @classmethod
    def _get_block_level_module_names(
        cls,
        original_model: nn.Module,
        dummy_input: Union[torch.Tensor, Tuple],
        block_level_modules_names: Optional[List[List[str]]],
        module_names_to_exclude: Set[str],
    ) -> List[List[str]]:
        """
        Return block level module name list

        :param original_model: Original torch model
        :param dummy_input: Dummy input
        :param block_level_modules_names: User provided block level module names
        :param module_names_to_exclude: Module names which are excluded during GPTVQ optimization
        :return: Block level module name list
        """
        ordered_module_names = [
            name
            for name, module in utils.get_ordered_list_of_modules(original_model, dummy_input)
            if isinstance(module, GPTVQSupportedModules) and name not in module_names_to_exclude
        ]
        if block_level_modules_names:
            leaf_level_module_names = set(ordered_module_names).difference(
                set(itertools.chain.from_iterable(block_level_modules_names))
            )
            leaf_level_module_names = [[name] for name in leaf_level_module_names]

            name_to_index = {name: idx for idx, name in enumerate(ordered_module_names)}
            for module_block in block_level_modules_names:
                module_block.sort(key=lambda x: name_to_index.get(x, float("inf")))

            block_level_modules_names.extend(leaf_level_module_names)
            ordered_block_level_modules = sorted(
                block_level_modules_names,
                key=lambda x: name_to_index.get(x[0], float("inf")),
            )
        else:
            ordered_block_level_modules = [[name] for name in ordered_module_names]

        msg = "\n".join([str(x) for x in ordered_block_level_modules])
        _logger.info("GPTVQ Hessian sampling and optimization will be applied in the following order and granularity\n%s", msg)
        return ordered_block_level_modules

    @classmethod
    def _apply_gptvq( # pylint: disable=too-many-locals
            cls,
            original_model: nn.Module,
            sim: QuantizationSimModel,
            dummy_input: Union[torch.Tensor, Tuple],
            gptvq_params: GPTVQParameters,
            module_names_to_exclude: Set[str],
            block_level_module_names: Optional[List[List[str]]],
    ):
        """
        Apply GPTVQ algorithm to optimize weights

        :param original_model: Original PyTorch model
        :param sim: QuantizationSimModel object to optimize weight
        :param dummy_input: Dummy input to model to be used to parse model graph
        :param gptvq_params: Dataclass holding GPTVQ parameters
        :param module_names_to_exclude: Module names which are excluded during GPTVQ optimization
        :param block_level_module_names: List of module name lists to optimize block level GPTVQ optimization instead of leaf module level
        """
        block_level_module_names = cls._get_block_level_module_names(
            original_model, dummy_input, block_level_module_names, module_names_to_exclude
        )

        total_sampling_time = 0
        total_optimization_time = 0
        for module_names in block_level_module_names:
            name_to_quant_module = cls._get_applicable_name_to_module_dict(
                module_names, sim, module_names_to_exclude
            )

            start_sampling_time = time.perf_counter()
            with Spinner(f"Sampling Hessian tensor of {', '.join(module_names)}"):
                name_to_hessian = get_module_name_to_hessian_tensor(gptvq_params, sim, module_names)
            sampling_time = time.perf_counter() - start_sampling_time
            _logger.info('Took %.4f seconds for Hessian sampling of %s', sampling_time, ', '.join(module_names))
            total_sampling_time += sampling_time

            for name, quant_module in name_to_quant_module.items():
                assert isinstance(quant_module, BaseQuantizationMixin), "%s is not BaseQuantizationMixin" % quant_module
                assert quant_module.param_quantizers["weight"], "%s does not have weight quantizer" % quant_module

                start_optimization_time = time.perf_counter()
                with Spinner(f"Started GPTVQ optimization of {name}"), torch.no_grad():
                    GPTVQOptimizer.weight_update(
                        module=quant_module,
                        gptvq_params=gptvq_params,
                        hessian=name_to_hessian[name],
                    )
                optimization_time = time.perf_counter() - start_optimization_time
                _logger.info('Took %.4f seconds for GPTVQ optimization of %s', optimization_time, name)
                total_optimization_time += optimization_time

        _logger.info('Total %.4f seconds for total Hessian sampling', total_sampling_time)
        _logger.info('Total %.4f seconds for total GPTVQ optimization', total_optimization_time)

    @staticmethod
    def _get_applicable_name_to_module_dict(
        module_names: List[str],
        sim: QuantizationSimModel,
        module_names_to_exclude: Set[str],
    ) -> Dict[str, BaseQuantizationMixin]:
        """
        Generate GPTVQ applicable name to module dictionary

        :param module_names: Module name list
        :param sim: QuantizationSimModel object
        :param module_names_to_exclude: Module names to exclude GPTVQ optimization
        :return: Module name to Quantization module dictionary
        """
        name_to_quant_module = collections.OrderedDict()
        for name in module_names:
            quant_module = get_named_module(sim.model, name)
            if name not in module_names_to_exclude:
                name_to_quant_module[name] = quant_module
        return name_to_quant_module

    @classmethod
    def _export_encodings_to_json(cls,
                                  path: str,
                                  filename_prefix: str,
                                  sim: QuantizationSimModel,
                                  rows_per_block: int):
        """
        Save GPTVQ applied parameter encodings to JSON file

        :param path: path where to store param encodings
        :param filename_prefix: filename to store exported weight encodings in JSON format
        :param sim: QuantizationSimModel object
        :param rows_per_block: The number of rows per block
        """
        # Create a dictionary to export to JSON file
        param_encodings = {}

        for name, quant_module in sim.model.named_modules():
            if isinstance(quant_module, GPTVQSupportedModules):
                if isinstance(quant_module.weight, QuantizedTensorBase):
                    cls._update_param_encodings_dict(
                        quant_module, name, param_encodings, rows_per_block
                    )

        # Unify the encoding format to be same as that of full encoding export file
        encoding = {"param_encodings": param_encodings}
        # export encodings to JSON file
        os.makedirs(os.path.abspath(path), exist_ok=True)
        encoding_file_path = os.path.join(path, f"{filename_prefix}.encodings")
        with open(encoding_file_path, "w") as encoding_fp:
            json.dump(encoding, encoding_fp, sort_keys=True, indent=4)

    @staticmethod
    def _update_param_encodings_dict(quant_module: _QuantizedModuleProtocol,
                                     name: str,
                                     param_encodings: Dict,
                                     rows_per_block: int):
        """
        Update block level encodings to per-channel encodings

        :param quant_module: quant module
        :param name: name of module
        :param param_encodings: Dictionary of param encodings
        :param rows_per_block: The number of rows per block
        """
        for orig_param_name, encodings in quant_module.export_param_encodings(encoding_version='0.6.1').items():
            if orig_param_name == "weight" and encodings:
                per_channel_encodings = []
                # Transform block encodings to per-channel encodings
                # blocks_per_column x 1 --> (blocks_per_column x rows_per_block) x 1
                for encoding in encodings:
                    per_channel_encodings.extend([encoding for _ in range(rows_per_block)])
                param_encodings[f"{name}.{orig_param_name}"] = per_channel_encodings

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

"""Sequential MSE implementation"""

# pylint: disable=no-name-in-module, ungrouped-imports, too-many-lines

import copy
from typing import List, Dict, Collection
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager
import itertools
import numpy as np
import onnx
import onnxruntime
import torch
from onnx.utils import Extractor
from aimet_common.libpymo import TensorQuantizerOpMode
from aimet_common.defs import QuantScheme
from aimet_common.utils import AimetLogger, deprecated
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.sequential_mse.dependency_graph import (
    DependencyGraph,
    SUPPORTED_MODULES,
)
from aimet_onnx.utils import disable_quantizers
from aimet_onnx.sequential_mse.dependency_graph import DependencyNode

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)


def apply_seq_mse(
    sim: QuantizationSimModel,
    inputs: Collection[Dict[str, np.ndarray]],
    num_candidates: int = 20,
):
    """
    Sequentially optimizes the QuantizationSimModel's weight encodings to reduce MSE loss at layer outputs.

    Args:
        sim (QuantizationSimModel): Calibrated QuantizationSimModel instance to optimize
        inputs (Collection[Dict[str, np.ndarray]]): The set of input samples to use during optimization
        num_candidates (int): Number of encoding candidates to sweep for each weight. Decreasing this can reduce
            runtime but may lead to lower accuracy.
    """
    seq_mse_params = SeqMseParams(num_batches=None, num_candidates=num_candidates)
    seq_mse = SequentialMse(None, sim, seq_mse_params, inputs)
    seq_mse.apply_seq_mse_algo()


@dataclass
class SeqMseParams:
    """
    Sequential MSE parameters

    :param num_batches: Number of batches.
    :param num_candidates: Number of candidates to perform grid search. Default 20.
    :param inp_symmetry: Input symmetry. Available options are 'asym', 'symfp' and 'symqt'. Default 'symqt'.
    :param loss_fn: Loss function. Available options are 'mse', 'l1' and 'sqnr'. Default 'mse'.
    """

    num_batches: int
    num_candidates: int = 20
    inp_symmetry: str = "symqt"
    loss_fn: str = "mse"


# pylint: disable=too-many-instance-attributes
class SequentialMse:
    """
    Sequentially minimizing activation MSE loss in layer-wise way to decide optimal param quantization encodings.
    """

    def __init__(
        self,
        model: onnx.ModelProto,
        sim: QuantizationSimModel,
        params: SeqMseParams,
        data_loader: Collection[Dict[str, np.ndarray]],
    ):
        """
        Initialize the sequential mse object

        :param model: float model
        :param sim: QuantizationSimModel object
        :param params: Sequential MSE parameters
        :param data_loader: The set of input samples to use during optimization
        """
        # pylint: disable=protected-access
        assert sim._quant_scheme in (
            QuantScheme.post_training_tf,
            QuantScheme.training_range_learning_with_tf_init,
        ), "Use TF quant-scheme with sequential MSE."

        self.sim = sim
        self.params = params
        data_loader = itertools.islice(data_loader, params.num_batches)

        # Hacky way to get around onnx.shape_inference.infer_shapes call as it doesn't work for model >2GB
        raw_data = {}
        # Store and clear raw_data from initializers
        for initializer in self.sim.model.model.graph.initializer:
            if initializer.HasField("raw_data"):
                raw_data[initializer.name] = initializer.raw_data
                initializer.ClearField("raw_data")

        # Copy the model without weight data and remove quantizer to allow shape inference
        model = copy.deepcopy(sim.model)
        model = sim.remove_quantizers(model).model

        self._extractor = Extractor(model)

        # Restore raw_data to initializers
        for initializer in self.sim.model.model.graph.initializer:
            if initializer.name in raw_data:
                initializer.raw_data = raw_data[initializer.name]
        for initializer in self._extractor.wmap.values():
            if initializer.name in raw_data:
                initializer.raw_data = raw_data[initializer.name]
        del raw_data

        self._update_value_info()

        self.dependency_graph = DependencyGraph(self._extractor.model, data_loader)
        self._extractor.model = self.sim.model.model
        self._extractor.graph = self.sim.model.model.graph
        self.data_loader = data_loader

    def _update_value_info_for_output(self, node):
        """
        Updates the value info for output of a node in sim model.
        Value info for QcQuantizeOp is not present in _sim_extractor

        :param node: onnx node
        """

        input_name = node.input[0]
        output_name = node.output[0]
        if (
            input_name in self._extractor.vimap
            and output_name not in self._extractor.vimap
        ):
            value_info_for_output = copy.deepcopy(self._extractor.vimap[input_name])
            value_info_for_output.name = node.output[0]
            self._extractor.vimap[node.output[0]] = value_info_for_output

    def _update_value_info_for_input(self, node):
        """
        Updates the value info for input of a node in sim model.
        Value info for QcQuantizeOp is not present in _sim_extractor

        :param node: onnx node
        """

        input_name = node.input[0]
        output_name = node.output[0]
        if (
            output_name in self._extractor.vimap
            and input_name not in self._extractor.vimap
        ):
            value_info_for_input = copy.deepcopy(self._extractor.vimap[output_name])
            value_info_for_input.name = node.input[0]
            self._extractor.vimap[node.input[0]] = value_info_for_input

    def _update_value_info_for_graph_output(self):
        """
        Updates the value info for input of a node in sim model.
        Value info for QcQuantizeOp is not present in _sim_extractor

        :param node: onnx node
        """
        for value_info in self.sim.model.model.graph.output:
            self._extractor.vimap[value_info.name] = value_info

    def _update_value_info(self):
        """
        Updates the value info for sim model.
        Value info for QcQuantizeOp is not present in _sim_extractor
        """

        self._update_value_info_for_graph_output()

        for node in self.sim.model.nodes():
            if node.op_type == "QcQuantizeOp":
                self._update_value_info_for_output(node)
                self._update_value_info_for_input(node)

    @deprecated("Use aimet_onnx.apply_seq_mse instead")
    @staticmethod
    def apply_seq_mse(
        model: onnx.ModelProto,
        sim: QuantizationSimModel,
        params: SeqMseParams,
        data_loader: Collection[Dict[str, np.ndarray]],
    ):
        """
        It performs following steps:
        1) creates seq_mse object
        2) call apply_seq_algo() member function

        :param model: float model
        :param sim: QuantizationSimModel object
        :param params: Sequential MSE parameters
        :param data_loader: The set of input samples to use during optimization
        """
        seq_mse = SequentialMse(model, sim, params, data_loader)
        seq_mse.apply_seq_mse_algo()

    def apply_seq_mse_algo(self):
        """
        It performs following steps:
        1) disable the quantizer for unsupported modules
        2) create the dependency graph
        3) run the onnx graph and compute encoding using seq mse algorithm
        4) re-enable the quantizer disabled in first step
        """
        with (
            disable_quantizers(self.sim, self._get_quantizers_to_be_disabled()),
            _remove_session(self.sim),
        ):
            self._topological_traversal()

    def _get_quantizers_to_be_disabled(self) -> List[str]:
        """
        Get list of quantizer names to be disabled in sim model before applying seq mse.

        NOTE: Disable all activation quantizers and param quantizers of non-supported modules

        :return Returns the quantizer names to be disabled in sim model.
        """
        enabled_quantizer_names = []

        # Get list of all the enabled activation + param quantizer names
        for name, quantizer in self.sim.qc_quantize_op_dict.items():
            if quantizer.enabled:
                enabled_quantizer_names.append(name)

        # Get list of all the enabled param quantizers of supported ops
        param_quantizer_names = []
        for cg_op in self.dependency_graph.conn_graph.ordered_ops:
            if cg_op.type in SUPPORTED_MODULES:
                for param_name in cg_op.parameters:
                    if (
                        param_name in self.sim.qc_quantize_op_dict
                        and self.sim.qc_quantize_op_dict[param_name].enabled
                    ):
                        param_quantizer_names.append(param_name)

        # Get list of all the quantizers that are not part of param quantizers of supported ops
        quantizers_to_be_disabled = []
        for name in enabled_quantizer_names:
            if name not in param_quantizer_names:
                quantizers_to_be_disabled.append(name)

        return quantizers_to_be_disabled

    @contextmanager
    def _disable_subgraph_quantizers(self, model: onnx.ModelProto):
        quantizer_keys = [
            node.input[0] for node in model.graph.node if node.op_type == "QcQuantizeOp"
        ]
        enabled = {
            name: self.sim.qc_quantize_op_dict[name].enabled for name in quantizer_keys
        }
        try:
            for name in quantizer_keys:
                self.sim.qc_quantize_op_dict[name].enabled = False

            yield

        finally:
            for name in quantizer_keys:
                self.sim.qc_quantize_op_dict[name].enabled = enabled[name]

    def _get_min_max_from_weights(self, dependency_node: DependencyNode):
        """
        Get per channel min/max values across output channel.

        :param dependency_node: Dependevy node which is to be optimized
        :return: per_channel_min and per_channel_max
        """
        # pylint: disable=protected-access
        channel_axis = QuantizationSimModel._get_quantization_axes(
            dependency_node.cg_op
        )[0]

        weight_data = self.dependency_graph.get_param_value(dependency_node)
        # Handle negative indexing
        if channel_axis < 0:
            channel_axis += len(weight_data.shape)

        axis = tuple(i for i in range(len(weight_data.shape)) if i != channel_axis)
        per_channel_max = np.max(abs(weight_data), axis=axis)
        return [-per_channel_max, per_channel_max]

    def _get_candidates(self, per_channel_max, per_channel_min):
        """
        Perform grid search.
        :param per_channel_max: Per channel max values
        :param per_channel_min: Per channel min values
        :return: candidates
        """
        candidates = []
        num_candidates = self.params.num_candidates
        for i in range(num_candidates):
            cand_max = per_channel_max / num_candidates * (i + 1)
            cand_min = per_channel_min / num_candidates * (i + 1)
            candidates.append((cand_max, cand_min))
        return candidates

    def _compute_encoding_from_candidate(
        self, candidate, dependency_node: DependencyNode
    ):
        """
        computes the encoding using candidate min and candidate max

        :param candidate: list containing min and max value
        :param dependency_node: Corresponding Dependency node
        :return: encoding
        """

        cand_max = candidate[0]
        cand_min = candidate[1]
        cand = np.stack((cand_max, cand_min), axis=-1)

        weight_name = self.dependency_graph.get_param_name(dependency_node)
        quantize_op = self.sim.qc_quantize_op_dict[weight_name]
        quantize_op.reset_encoding_stats()

        # pylint: disable=protected-access
        quantizer_shape = quantize_op._encoding_shape()
        num_encodings = np.prod(quantizer_shape)

        if num_encodings != len(cand) and num_encodings != 1:
            raise ValueError(
                weight_name,
                " should be per-tensor or number of "
                "quantizer must match with number of channels",
            )

        if quantizer_shape:
            quantize_op.update_encoding_stats(
                np.reshape(cand, (*quantizer_shape[0:-1], 2 * quantizer_shape[-1]))
            )
        else:
            quantize_op.update_encoding_stats(cand)

        quantize_op.compute_encodings()

        quantize_op.op_mode = TensorQuantizerOpMode.quantizeDequantize

    def _freeze_encodings(self, dependency_node: DependencyNode):
        """
        Freezes the encoding after the node is optimized
        :param dependency_node: Optimized dependency node
        """
        weight_name = self.dependency_graph.get_param_name(dependency_node)
        quantize_op = self.sim.qc_quantize_op_dict[weight_name]
        quantize_op.freeze_encodings()

    @staticmethod
    def neg_sqnr(pred: torch.Tensor, target: torch.Tensor, eps=1e-10, reduction="none"):
        """
        Loss function to minimize negative SQNR which is equivalent to maximizing SQNR.

        :param pred: X^Q^ quantized-dequantized values
        :param target: XW FP32 values
        :param eps: epsilon
        :param reduction: unused arg added only to have the same signature as that of functional losses of pytorch library
        :return: Negative SQNR
        """
        # pylint: disable=unused-argument
        quant_error = target - pred
        exp_noise = torch.mean(quant_error**2, 0, keepdim=True) + eps
        exp_signal = torch.mean(target**2, 0, keepdim=True)
        sqnr = exp_signal / exp_noise
        sqnr_db = 10 * torch.log10(sqnr)
        return -sqnr_db

    def _compute_recon_loss(self, sim_output, float_output, dependency_node):
        """
        Compute reconstruction loss and return the sum by reducing over all the dimensions except last channel dimension.

        :param xqwq: X^Q^ quantized-dequantized values
        :param xw: XW FP32 values
        :param params: Sequential MSE parameters
        :return: loss
        """

        xqwq = torch.from_numpy(sim_output)
        xw = torch.from_numpy(float_output)

        if dependency_node.cg_op.type == "Conv":
            permute_order = [0] + list(range(2, xw.dim())) + [1]
            xqwq = xqwq.permute(permute_order)
            xw = xw.permute(permute_order)

        if self.params.loss_fn == "mse":
            loss_fn = torch.nn.functional.mse_loss
        elif self.params.loss_fn == "l1":
            loss_fn = torch.nn.functional.l1_loss
        elif self.params.loss_fn == "sqnr":
            loss_fn = SequentialMse.neg_sqnr
        else:
            raise ValueError(f"Invalid loss function: {self.params.loss_fn}")

        channel_dim = xqwq.shape[-1]
        xqwq = xqwq.reshape(-1, channel_dim)
        xw = xw.reshape(-1, channel_dim)
        loss = loss_fn(xqwq, xw, reduction="none").sum(0)
        assert loss.size() == torch.Size([channel_dim])
        return np.array(loss)

    def _run_seq_mse(self, dep_nodes_to_parallelize: List[DependencyNode]):
        """
        Run Sequential MSE for all the dep_nodes_to_parallelize at same level.

        :param dep_nodes_to_parallelize: Dependency nodes to be parallelized.
        """

        def _set_candidates(index: int):
            """
            Helper function to set candidate based on index for ops at same level.
            Internally computes the encoding using candidate min and candidate max

            :param index: Index of candidate
            """
            for dep_node in dep_nodes_to_parallelize:
                candidate = min_max_candidates[dep_node.cg_op.name]
                self._compute_encoding_from_candidate(candidate[index], dep_node)

        def _compute_loss(all_fp_outputs: List, all_sim_outputs: List):
            """
            Helper function to compute reconstruction loss for ops at same level.

            :param all_fp_outputs: FP Outputs of all ops at same level
            :param all_sim_outputs: Sim Outputs of all ops at same level
            """
            for i, dep_node in enumerate(dep_nodes_to_parallelize):
                fp_output = np.concatenate(all_fp_outputs[i], axis=0)
                sim_output = np.concatenate(all_sim_outputs[i], axis=0)
                loss = self._compute_recon_loss(fp_output, sim_output, dep_node)
                total_loss[dep_node.cg_op.name].append(loss)

        def _get_dep_node_io_names(dep_nodes: List[DependencyNode]):
            """
            Helper function to get the input and output names of subgraph.

            :param dep_nodes: List of dependency nodes to be parallelized.
            :return: Subgraph input and output names.
            """
            subgraph_inputs = []
            subgraph_outputs = []
            for dep_node in dep_nodes:
                subgraph_inputs.append(dep_node.op_input_names[0])
                subgraph_outputs.append(dep_node.op_output_names[0])

            subgraph_inputs = list(set(subgraph_inputs))
            return subgraph_inputs, subgraph_outputs

        total_loss = defaultdict(list)
        min_max_candidates = {}

        # Perform grid search
        for dep_node in dep_nodes_to_parallelize:
            per_channel_min, per_channel_max = self._get_min_max_from_weights(dep_node)
            min_max_candidates[dep_node.cg_op.name] = self._get_candidates(
                per_channel_max, per_channel_min
            )

        subgraph_inp_names, subgraph_outs_names = _get_dep_node_io_names(
            dep_nodes_to_parallelize
        )

        # For now, we only expose "symqt" input symmetry.
        assert self.params.inp_symmetry == "symqt", (
            "Only symmetric quantsim inputs ('symqt') are supported."
        )
        sim_inputs = self.dependency_graph.get_sim_data(dep_nodes_to_parallelize)

        # Create inference session for subgraph from float model
        subgraph_model = self._split_onnx_graph(
            self._extractor, subgraph_inp_names, subgraph_outs_names
        )
        with self._create_session(subgraph_model) as session:
            with self._disable_subgraph_quantizers(subgraph_model):
                fp_outputs = self._run_onnx_graph(session, sim_inputs)

            for i in range(self.params.num_candidates):
                _set_candidates(i)
                sim_outputs = self._run_onnx_graph(session, sim_inputs)
                _compute_loss(fp_outputs, sim_outputs)
                _logger.debug(f"Finished candidate: {i}")

        del fp_outputs, sim_outputs, sim_inputs

        # Postprocessing (not vectorized)
        for dep_node in dep_nodes_to_parallelize:
            loss = total_loss[dep_node.cg_op.name]
            stacked_loss = np.stack(loss, axis=0)
            arg_min_ = np.argmin(stacked_loss, axis=0, keepdims=True)
            best_max = torch.stack(
                [
                    torch.tensor(cand_max)
                    for cand_max, _ in min_max_candidates[dep_node.cg_op.name]
                ]
            ).gather(0, torch.tensor(arg_min_))[0]
            best_min = torch.stack(
                [
                    torch.tensor(cand_min)
                    for _, cand_min in min_max_candidates[dep_node.cg_op.name]
                ]
            ).gather(0, torch.tensor(arg_min_))[0]
            best_candidate = (best_max, best_min)
            self._compute_encoding_from_candidate(best_candidate, dep_node)
            self._freeze_encodings(dep_node)

        dep_node_names = [dep_node.cg_op.name for dep_node in dep_nodes_to_parallelize]
        _logger.info(
            f"Computed optimal parameter encodings for ops: {', '.join(dep_node_names)}"
        )

    @staticmethod
    def _split_onnx_graph(
        extractor: Extractor, input_names: List[str], output_names: List[str]
    ) -> onnx.ModelProto:
        """
        Splits the onnx graph from input names to output names using extractor

        :param input_names: input names of split graph
        :param output_names: output names of split graph
        :return: float split model and sim split model
        """
        return extractor.extract_model(list(input_names), list(output_names))

    def _run_onnx_graph(
        self, session: onnxruntime.InferenceSession, inputs: Dict
    ) -> List[List[np.ndarray]]:
        """
        Run the onnx graph using onnx runtime

        :param session: Onnxruntime session
        :param inputs: inputs to the model
        :return: outputs
        """
        outputs = []
        dataset_len = len(next(iter(inputs.values())))
        for i in range(dataset_len):
            input_batch = {}
            for name, data in inputs.items():
                input_batch[name] = data[i]
            output = session.run(None, input_batch)
            if len(outputs) == 0:
                outputs = [[] for _ in range(len(output))]
            for idx, out in enumerate(output):
                outputs[idx].append(out)

        return outputs

    def _cache_subgraph_input_data(self, dep_nodes: List[DependencyNode]):
        """
        For given dependency nodes at the same level, cache intermediate activation data

        - Extract a subgraph using the parent nodes,
        - Collect the intermediate activations by executing the subgraph,
        - Cache these data to provide them to the next subgraph.

        :param dep_nodes: List of dependency nodes at same level
        """
        dep_node_names = [dep_node.cg_op.name for dep_node in dep_nodes]
        _logger.debug(
            f"Started caching inputs for dep nodes: {', '.join(dep_node_names)}"
        )

        subgraph_inp_names, subgraph_out_names = (
            self.dependency_graph.get_subgraph_inp_out_names(dep_nodes)
        )
        subgraph_inps = self.dependency_graph.dependency_node_inputs(dep_nodes)
        assert len(subgraph_inp_names) == len(subgraph_inps)

        _logger.debug(
            f"Subgraph input names: {subgraph_inp_names}, Subgraph output names: {subgraph_out_names}"
        )
        sim_split_model = self._split_onnx_graph(
            self._extractor, subgraph_inp_names, subgraph_out_names
        )
        with self._create_session(sim_split_model) as session:
            subgraph_outs = self._run_onnx_graph(session, subgraph_inps)
        self.dependency_graph.update_sim_data(subgraph_out_names, subgraph_outs)
        _logger.debug(
            f"Collected intermediate data for output names: {subgraph_out_names}"
        )
        del subgraph_inps, subgraph_outs

        # Decrease the reference count for the input data.
        for dep_node in dep_nodes:
            for inward_node in dep_node.inward_nodes:
                inward_node.out_degree = inward_node.out_degree - 1
                if inward_node.out_degree == 0:
                    self.dependency_graph.dec_ref_count(inward_node)

    def _topological_traversal(self):
        """
        Start the topo sort from the starting ops i.e. ops having in_degree equal to zero
        Flow:
            - Cache intermediate activations input data before applying Seq MSE at a given level in topological order.
            - Use cached intermediate activations and run Seq MSE in parallel.

        NOTE: For the first iteration, no need to cache subgraph input data since model graph inputs are already saved.
        """
        sorted_order = self.dependency_graph.get_topologically_sorted_nodes()

        for i, sorted_nodes in sorted_order.items():
            if i != 0:
                self._cache_subgraph_input_data([node for node in sorted_nodes])

            dep_nodes_to_parallelize = [
                node for node in sorted_nodes if node.cg_op.type in SUPPORTED_MODULES
            ]
            if dep_nodes_to_parallelize:
                self._run_seq_mse(dep_nodes_to_parallelize)

    @contextmanager
    def _create_session(self, model: onnx.ModelProto):
        """
        Build and return onnxruntime inference session

        :param model: onnx model
        :return: Session
        """
        try:
            session = QuantizationSimModel.build_session(
                model,
                self.sim.providers,
                user_onnx_libs=self.sim._user_onnx_libs,
                path=self.sim._path,
            )
            yield session
        finally:
            del session


@contextmanager
def _remove_session(sim: QuantizationSimModel):
    """
    Deletes sim.session for the duration of the context to save GPU memory. Rebuilds the session upon exiting.
    """
    try:
        del sim.session
        yield
    finally:
        sim._rebuild_session()  # pylint:disable = protected-access

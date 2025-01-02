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

""" Sequential MSE implementation """

# pylint: disable=no-name-in-module, ungrouped-imports, too-many-lines

import copy
from typing import List
import numpy as np
import torch
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from onnx import numpy_helper
from onnx.utils import Extractor

# pylint: disable=wrong-import-order
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.sequential_mse.dependency_graph_utils import DependencyGraphUtils
from aimet_onnx.sequential_mse.dependency_graph import DependencyGraph
from aimet_onnx.sequential_mse.dependency_graph import DependencyNode
from aimet_common.libpymo import TensorQuantizerOpMode
from aimet_common.defs import QuantScheme
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_common.utils import AimetLogger
from dataclasses import dataclass

from onnx import TensorProto


SUPPORTED_MODULES = ("Conv", "Gemm", "MatMul")

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)

@dataclass
class SeqMseParams:
    """
    Sequential MSE parameters

    :param num_batches: Number of batches.
    :param num_candidates: Number of candidates to perform grid search. Default 20.
    :param inp_symmetry: Input symmetry. Available options are 'asym', 'symfp' and 'symqt'. Default 'symqt'.
    :param loss_fn: Loss function. Available options are 'mse', 'l1' and 'sqnr'. Default 'mse'.
    """
    num_batches: int = 4
    num_candidates: int = 20
    inp_symmetry: str = 'symqt'
    loss_fn: str = 'mse'

# pylint: disable=too-many-instance-attributes
class SequentialMse:
    """
    Sequentially minimizing activation MSE loss in layer-wise way to decide optimal param quantization encodings.
    """

    def __init__(self,
                 model,
                 sim: QuantizationSimModel,
                 params: SeqMseParams,
                 data_loader):

        """
        Initialize the sequential mse object

        :param model: float model
        :param sim: QuantizationSimModel object
        :param data_loader: Torch Dataloader
        :param params: Sequential MSE parameters
        """

        # pylint: disable=protected-access
        assert sim._quant_scheme in (QuantScheme.post_training_tf, QuantScheme.training_range_learning_with_tf_init), \
            "Use TF quant-scheme with sequential MSE."

        self.sim = sim
        self.model = model
        self.params = params
        self.node_name_to_input_names = dict()
        self.static_tensor_name_to_proto = dict()

        if not isinstance(self.model, ONNXModel):
            self.model = ONNXModel(self.model)

        self._fill_node_name_to_input_names()
        self._fill_static_tensor_name_to_proto()

        raw_data = dict()
        for initializer in self.model.model.graph.initializer:
            if initializer.HasField('raw_data'):
                raw_data[initializer.name] = initializer.raw_data
                initializer.ClearField("raw_data")

        self._float_extractor = Extractor(self.model.model)

        for initializer in self.model.model.graph.initializer:
            if initializer.name in raw_data:
                initializer.raw_data = raw_data[initializer.name]

        for initializer in self._float_extractor.model.graph.initializer:
            if initializer.name in raw_data:
                initializer.raw_data = raw_data[initializer.name]

        self._sim_extractor = copy.deepcopy(self._float_extractor)

        self._update_value_info()

        self._sim_extractor.model = self.sim.model.model
        self._sim_extractor.graph = self.sim.model.model.graph

        self.connected_graph = ConnectedGraph(self.model)

        self.data_loader = data_loader

        self.dependency_graph = DependencyGraph()
        self.dependency_graph_utils = DependencyGraphUtils(self.connected_graph, self.node_name_to_input_names,
                                                           self.static_tensor_name_to_proto)
        self.quantizers_to_be_disabled = self._get_quantizers_to_be_disabled() # check this

    def _update_value_info_for_output(self, node):
        """
        Updates the value info for output of a node in sim model.
        Value info for QcQuantizeOp is not present in _sim_extractor

        :param node: onnx node
        """

        input_name = node.input[0]
        output_name = node.output[0]
        if input_name in self._sim_extractor.vimap and output_name not in self._sim_extractor.vimap:
            value_info_for_output = copy.deepcopy(self._sim_extractor.vimap[input_name])
            value_info_for_output.name = node.output[0]
            self._sim_extractor.vimap[node.output[0]] = value_info_for_output

    def _update_value_info_for_input(self, node):
        """
        Updates the value info for input of a node in sim model.
        Value info for QcQuantizeOp is not present in _sim_extractor

        :param node: onnx node
        """

        input_name = node.input[0]
        output_name = node.output[0]
        if output_name in self._sim_extractor.vimap and input_name not in self._sim_extractor.vimap:
            value_info_for_input = copy.deepcopy(self._sim_extractor.vimap[output_name])
            value_info_for_input.name = node.input[0]
            self._sim_extractor.vimap[node.input[0]] = value_info_for_input

    def _update_value_info_for_graph_output(self):
        """
        Updates the value info for input of a node in sim model.
        Value info for QcQuantizeOp is not present in _sim_extractor

        :param node: onnx node
        """

        for value_info in self.model.model.graph.output:
            self._float_extractor.vimap[value_info.name] = value_info

        for value_info in self.sim.model.model.graph.output:
            self._sim_extractor.vimap[value_info.name] = value_info

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

    def _fill_static_tensor_name_to_proto(self):
        """
        Fills the mapping from static tensor name to the prop buf
        """
        for initializer in self.model.model.graph.initializer:
            self.static_tensor_name_to_proto[initializer.name] = initializer

        for node in self.model.model.graph.node:
            if node.op_type == "Constant":
                self.static_tensor_name_to_proto[node.output[0]] = node

    # pylint: disable=inconsistent-return-statements
    def _extract_float_data_from_proto(self, name):
        """
        returns the tensor value of the given name using static_tensor_name_to_proto dictionary
        :param name: name of the static tensor
        :return tensor value
        """
        if name in self.static_tensor_name_to_proto:
            proto_buf = self.static_tensor_name_to_proto[name]
            if isinstance(proto_buf, TensorProto):
                return numpy_helper.to_array(proto_buf)

            for attr in proto_buf.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
        else:
            raise ValueError(name, " is neither constant or initializer")

    def _fill_node_name_to_input_names(self):
        """
        Fills the mapping from node name to input names
        """
        for node in self.model.nodes():
            self.node_name_to_input_names[node.name] = node.input

    @staticmethod
    def apply_seq_mse(model, sim: QuantizationSimModel, params: SeqMseParams, data_loader):
        """
        It performs following steps:
        1) creates seq_mse object
        2) call apply_seq_algo() member function

        :param model: float model
        :param sim: QuantizationSimModel object
        :param data_loader: Data loader
        :param params: Sequential MSE parameters
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

        try:
            self.temporarily_disable_quantizers()
            self.dependency_graph = self.dependency_graph_utils.create_dependency_graph(self.data_loader,
                                                                                        self.params.num_batches)
            self._run_onnx_graph_dependency_graph_order()
        finally:
            self.re_enable_quantizers()

    def _get_quantizers_to_be_disabled(self) -> List[QcQuantizeOp]:
        """
        :return Returns the quantizers of unsupported modules
        """

        quantizer_to_disable_name = list()
        quantizer_to_not_disable_name = list()

        for name, qc_quantize_op in self.sim.qc_quantize_op_dict.items():
            if qc_quantize_op.enabled:
                quantizer_to_disable_name.append(name)

        for node in self.model.nodes():
            if self.dependency_graph_utils.is_supported_module(node):
                weight_node_name = node.input[1]
                quantizer_to_not_disable_name.append(weight_node_name)

        quantizer_to_disable_name = [name for name in quantizer_to_disable_name
                                     if name not in quantizer_to_not_disable_name]

        quantizer_to_disable = list()

        for name in quantizer_to_disable_name:
            if name in self.sim.qc_quantize_op_dict:
                quantizer_to_disable.append(self.sim.qc_quantize_op_dict[name])

        return quantizer_to_disable

    def temporarily_disable_quantizers(self):
        """
        Disable quantizers needed to be disabled before applying sequential MSE.
        """
        for quantizer in self.quantizers_to_be_disabled:
            quantizer.enabled = False

    def re_enable_quantizers(self):
        """
        Re-enable quantizers that were disabled by temporarily_disable_quantizers method
        """
        for quantizer in self.quantizers_to_be_disabled:
            quantizer.enabled = True

    def _get_min_max_from_weights(self, dependency_node: DependencyNode):
        """
        Get per channel min/max values across output channel.

        :param dependency_node: Dependevy node which is to be optimized
        :return: per_channel_min and per_channel_max
        """

        weight_name = self.node_name_to_input_names[dependency_node.op_name][1]
        weight_data = self._extract_float_data_from_proto(weight_name)

        connected_op = self.connected_graph.get_op_from_module_name(dependency_node.op_name)
        # pylint: disable=protected-access
        channel_axis = QuantizationSimModel._get_quantization_axes(connected_op)[0]
        # pylint: disable=consider-using-generator, use-a-generator
        axis = tuple([i for i in range(len(weight_data.shape)) if i != channel_axis])

        per_channel_max = np.max(abs(weight_data), axis=axis)

        return [-per_channel_max, per_channel_max]

    def _get_candidates(self, per_channel_max, per_channel_min):
        """
        Perform grid search.
        :param per_channel_max: Per channel max values
        :param per_channel_min: Per channel min values
        :return: candidates
        """
        candidates = list()
        num_candidates = self.params.num_candidates
        for i in range(num_candidates):
            cand_max = per_channel_max / num_candidates * (i + 1)
            cand_min = per_channel_min / num_candidates * (i + 1)
            candidates.append((cand_max, cand_min))
        return candidates

    def _compute_encoding_from_candidate(self, candidate, dependency_node: DependencyNode):
        """
        computes the encoding using candidate min and candidate max

        :param candidate: list containing min and max value
        :param dependency_node: Corresponding Dependency node
        :return: encoding
        """

        cand_max = candidate[0]
        cand_min = candidate[1]

        cand = np.stack((cand_max, cand_min), axis=-1)

        weight_name = self.node_name_to_input_names[dependency_node.op_name][1]

        quantize_op = self.sim.qc_quantize_op_dict[weight_name]

        quantize_op.reset_encoding_stats()

        # pylint: disable=protected-access
        tensor_quantizers = quantize_op._tensor_quantizer

        if len(tensor_quantizers) != len(cand) and len(tensor_quantizers) != 1:
            raise ValueError(weight_name, " should be per-tensor or number of "
                                          "quantizer must match with number of channels")

        # pylint: disable=protected-access
        if len(tensor_quantizers) == 1:
            tensor_quantizer = tensor_quantizers[0]
            tensor_quantizer.updateStats(cand, False)
        else:
            for i, tensor_quantizer in enumerate(tensor_quantizers):
                tensor_quantizer.updateStats(cand[i], False)

        quantize_op.compute_encodings()

        quantize_op.op_mode = TensorQuantizerOpMode.quantizeDequantize

    def _freeze_encodings(self, dependency_node: DependencyNode):
        """
        Freezes the encoding after the node is optimized
        :param dependency_node: Optimized dependency node
        """
        weight_name = self.node_name_to_input_names[dependency_node.op_name][1]
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
        exp_noise = torch.mean(quant_error ** 2, 0, keepdim=True) + eps
        exp_signal = torch.mean(target ** 2, 0, keepdim=True)
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

        if dependency_node.op_type == "Conv":
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

    # pylint: disable-msg=too-many-locals
    def _do_seq_mse(self, dependency_node: DependencyNode):
        """
        Find and freeze optimal parameter encodings candidate for given dependency node.
        :param dependency_node: Corresponding Dependency node
        """
        per_channel_min, per_channel_max = self._get_min_max_from_weights(dependency_node)

        candidates = self._get_candidates(per_channel_max, per_channel_min)

        total_loss = list()

        float_split_model, sim_split_model = self._split_onnx_graph(dependency_node.op_input_names,
                                                                    dependency_node.op_output_names)

        _logger.info("Finding and freezing optimal param encodings candidate of op: %s", dependency_node.op_name)
        # for different modes only inputs will change
        if self.params.inp_symmetry == "asym":
            float_inputs = self.dependency_graph.get_float_data(dependency_node)
            sim_inputs = self.dependency_graph.get_sim_data(dependency_node)
        elif self.params.inp_symmetry == "symfp":
            float_inputs = self.dependency_graph.get_float_data(dependency_node)
            sim_inputs = self.dependency_graph.get_float_data(dependency_node)
        elif self.params.inp_symmetry == "symqt":
            float_inputs = self.dependency_graph.get_sim_data(dependency_node)
            sim_inputs = self.dependency_graph.get_sim_data(dependency_node)
        else:
            raise ValueError(f"Invalid inp_symmetry: {self.params.inp_symmetry}")

        float_outputs = self._run_onnx_graph(float_split_model, float_inputs)
        float_outputs = np.concatenate(float_outputs[0], axis=0)

        for candidate in candidates:

            self._compute_encoding_from_candidate(candidate, dependency_node)

            sim_outputs = self._run_onnx_graph(sim_split_model, sim_inputs)
            sim_outputs = np.concatenate(sim_outputs[0], axis=0)

            loss = self._compute_recon_loss(sim_outputs, float_outputs, dependency_node)

            total_loss.append(loss)

        stacked_loss = np.stack(total_loss, axis=0)
        arg_min_ = np.argmin(stacked_loss, axis=0, keepdims=True)

        best_max = torch.stack([torch.tensor(cand_max) for cand_max, _ in candidates]).gather(0, torch.tensor(arg_min_))[0]
        best_min = torch.stack([torch.tensor(cand_min) for _, cand_min in candidates]).gather(0, torch.tensor(arg_min_))[0]

        best_candidate = (best_max, best_min)

        self._compute_encoding_from_candidate(best_candidate, dependency_node)
        self._freeze_encodings(dependency_node)

    # pylint: disable=no-self-use
    def _get_input_names_from_dependencies(self, dependency_node: DependencyNode):
        """
        Returns the input names for the op corresponding to dependency node

        :param dependency_node: Corresponding Dependency node
        :return: input names for the op corresponding to dependency node
        """

        input_names = set()

        for inward_node in dependency_node.inward_nodes:
            input_names.update(inward_node.op_input_names)

        return list(input_names)

    def _get_inputs_from_dependencies(self, dependency_node: DependencyNode):
        """
        Returns the input needed for the op corresponding to the dependency node

        :param dependency_node: Corresponding dependency node
        :return: float inputs and sim inputs
        """

        float_inputs = dict()
        sim_inputs = dict()

        for inward_node in dependency_node.inward_nodes:
            float_inputs.update(self.dependency_graph.get_float_data(inward_node))
            sim_inputs.update(self.dependency_graph.get_sim_data(inward_node))

        return float_inputs, sim_inputs

    def _split_onnx_graph(self, input_names, output_names):
        """
        Splits the onnx graph from input names to output names using extractor

        :param input_names: input names of split graph
        :param output_names: output names of split graph
        :return: float split model and sim split model
        """
        float_split_model = self._float_extractor.extract_model(list(input_names), list(output_names))
        sim_split_model = self._sim_extractor.extract_model(list(input_names), list(output_names))
        return float_split_model, sim_split_model

    def _run_onnx_graph(self, model, inputs):
        """
        Run the onnx graph using onnx runtime

        :param model: Corresponding model
        :param inputs: inputs to the model
        :return: outputs
        """
        # pylint: disable=protected-access
        session = QuantizationSimModel.build_session(model, self.sim.providers,
                                                     user_onnx_libs=self.sim._user_onnx_libs, path=self.sim._path)

        outputs = list()

        num_batches = min(self.params.num_batches, len(self.data_loader.dataset) // self.data_loader.batch_size)

        for i in range(num_batches):
            input_batch = dict()
            for name, data in inputs.items():
                input_batch[name] = data[i]
            output = session.run(None, input_batch)
            if len(outputs) == 0:
                outputs = [list() for _ in range(len(output))]
            for idx, out in enumerate(output):
                outputs[idx].append(out)

        return outputs

    def _process_dependency_nodes(self, dependency_node: DependencyNode):
        """
        1) Get input names, output names using dependency graph
        2) Split the graph using input names and output names
        3) Run the split graph
        4) Decrease the out-degree of the inward nodes by -1, if outdegree becomes zero, then delete the data
        5) Optimize the dependency node

        :param dependency_node: Corresponding dependency node
        """
        # get input names and output names, split and run and do_seq_mse
        # then make out_degree of the inward nodes -1, if that becomes zero delete the data

        input_names = self._get_input_names_from_dependencies(dependency_node=dependency_node)

        graph_inputs = [node.name for node in self.model.model.graph.input]
        output_names = [name for name in dependency_node.op_input_names if name not in graph_inputs]

        float_split_model, sim_split_model = self._split_onnx_graph(input_names=input_names, output_names=output_names)
        float_inputs, sim_inputs = self._get_inputs_from_dependencies(dependency_node=dependency_node)

        float_outputs = self._run_onnx_graph(model=float_split_model, inputs=float_inputs)
        self.dependency_graph.update_float_data(output_names, float_outputs)

        sim_outputs = self._run_onnx_graph(model=sim_split_model, inputs=sim_inputs)
        self.dependency_graph.update_sim_data(output_names, sim_outputs)

        for inward_node in dependency_node.inward_nodes:
            inward_node.outdegree = inward_node.outdegree - 1
            if inward_node.outdegree == 0:
                self.dependency_graph.dec_ref_count(inward_node)

        if dependency_node.op_type in SUPPORTED_MODULES:
            self._do_seq_mse(dependency_node=dependency_node)

    def _do_topo_sort_helper(self, dependency_node: DependencyNode):
        """
        1) Decrease indegree of the child ops by -1, if the indegree becomes zero, then process the node
        2) run _do_topo_sort_helper for the child node

        :param dependency_node: Corresponding dependency node
        """
        # make indegree of the child ops -1, if the indegree becomes zero split and run and do_seq_mse
        # then make out_degree of the inward nodes -1, if that becomes zero delete the data
        # then call _do_topo_sort_helper for that node

        for child_node in dependency_node.outward_nodes:
            child_node.indegree = child_node.indegree - 1
            if child_node.indegree == 0:
                self._process_dependency_nodes(dependency_node=child_node)
                self._do_topo_sort_helper(dependency_node=child_node)

    def _run_onnx_graph_dependency_graph_order(self):
        """
        Start the topo sort from the starting ops i.e. ops having indegree equal to zero
        """

        for start_op in self.dependency_graph.starting_ops:
            if start_op.op_type in SUPPORTED_MODULES:
                self._do_seq_mse(dependency_node=start_op)
            self._do_topo_sort_helper(dependency_node=start_op)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""Sequential MSE implementation"""

# pylint: disable=no-name-in-module, ungrouped-imports, too-many-lines

from typing import List, Dict, Collection, Iterable, Union, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager
import functools
import itertools
import os
import tempfile
import numpy as np
import onnx
import onnx_ir
import onnxruntime
import torch
from onnxruntime.quantization.onnx_quantizer import ONNXModel

from aimet_onnx.common.libpymo import TensorQuantizerOpMode
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.common.utils import AimetLogger, deprecated
from aimet_onnx.common.onnx._utils import (
    _iterate_graphs_recursive,
    _iterate_graphs_recursive_ir,
)
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import (
    disable_quantizers,
    OrtInferenceSession,
    get_torch_device,
    map_np_dtype_to_torch,
    _BFLOAT16,
    _onnx_element_type,
    _run_session_to_numpy,
)
from aimet_onnx.blockwise_sampler import BlockwiseSampler, SamplingPlan
from aimet_onnx.sequential_mse.frontier import (
    FrontierPlan,
    SeqMseLayer,
    find_layers,
    get_weight_name,
    is_supported_op,
    plan_frontier,
    resolve_nodes_to_exclude,
)
from aimet_onnx.sequential_mse.progress import (
    SeqMseProgress,
    display_name,
    weight_result,
)
from aimet_onnx.sequential_mse.transform import (
    modify_graph_with_grouped_conv,
    modify_graph_with_grouped_linear,
    prepare_linear_inputs,
)

_GPU_ALLOC_MARGIN = 0.8

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.SeqMse)


def apply_seq_mse(
    sim: QuantizationSimModel,
    inputs: Collection[Dict[str, np.ndarray]],
    num_candidates: int = 20,
    nodes_to_exclude: Optional[List[str]] = None,
):
    """
    Sequentially optimizes the QuantizationSimModel's weight encodings to reduce MSE loss at layer outputs.

    Args:
        sim (QuantizationSimModel): QuantizationSimModel instance to optimize
        inputs (Collection[Dict[str, np.ndarray]]): The set of input samples to use during optimization
        num_candidates (int): Number of encoding candidates to sweep for each weight. Decreasing this can reduce
            runtime but may lead to lower accuracy.
        nodes_to_exclude (Optional[List[str]]): List of supported node name(s) to exclude from sequential MSE optimization
    """
    seq_mse_params = SeqMseParams(num_batches=None, num_candidates=num_candidates)
    seq_mse = SequentialMse(None, sim, seq_mse_params, inputs, nodes_to_exclude)
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
        nodes_to_exclude: Optional[List[str]] = None,
    ):
        """
        Initialize the sequential mse object

        :param model: float model
        :param sim: QuantizationSimModel object
        :param params: Sequential MSE parameters
        :param data_loader: The set of input samples to use during optimization
        nodes_to_exclude: List of supported node name(s) to exclude from sequential MSE optimization
        """
        self.sim = sim
        self.params = params

        data_loader = itertools.islice(data_loader, params.num_batches)

        # Save the model once with external data and load it lazily, so that subgraphs can be
        # extracted without loading weights.
        self._model_dir = tempfile.TemporaryDirectory()
        model_path = os.path.join(self._model_dir.name, "model.onnx")

        # onnx_ir.save leaves sim.model.model unchanged.
        onnx_ir.save(
            onnx_ir.from_proto(sim.model.model),
            model_path,
            external_data="model.onnx.data",
            # Prevents Split's split_sizes input from being externalized
            size_threshold_bytes=1024**2,
        )
        self._ir_model = onnx_ir.load(model_path)

        # The user-supplied exclusions plus auto-excluded ops (e.g. SpinQuant R3 online
        # rotations), so quantizer handling stays in sync.
        self._nodes_to_exclude = resolve_nodes_to_exclude(
            _sim_nodes(sim), nodes_to_exclude
        )
        self._inputs = _collect_inputs(
            data_loader, [inp.name for inp in sim.model.model.graph.input]
        )

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
        2) plan the levels of layers to optimize together
        3) run the onnx graph and compute encoding using seq mse algorithm
        4) re-enable the quantizer disabled in first step
        """
        plan = self._frontier_plan()
        names_to_optimize = self._get_quantizer_names_to_optimize(plan)
        quantizers_to_optimize = [
            self.sim.qc_quantize_op_dict[name] for name in names_to_optimize
        ]

        with (
            _temporarily_disable_block_grouping(self.sim),
            _use_min_max_quant_scheme(quantizers_to_optimize),
        ):
            self.sim._compute_param_encodings(overwrite=False)

            with (
                disable_quantizers(
                    self.sim, self._get_quantizers_to_be_disabled(names_to_optimize)
                ),
                _remove_session(self.sim),
            ):
                self._frontier_traversal(plan, names_to_optimize)

    def _get_quantizers_to_be_disabled(self, names_to_optimize: List[str]) -> List[str]:
        """
        Get list of quantizer names to be disabled in sim model before applying seq mse.

        NOTE: Disable all activation quantizers and param quantizers of non-supported modules

        :param names_to_optimize: Param quantizer names optimized by seq mse
        :return Returns the quantizer names to be disabled in sim model.
        """
        enabled_quantizer_names = []

        # Get list of all the enabled activation + param quantizer names
        for name, quantizer in self.sim.qc_quantize_op_dict.items():
            if quantizer.enabled:
                enabled_quantizer_names.append(name)

        # Get list of all the quantizers that are not part of param quantizers of supported ops
        quantizers_to_be_disabled = []
        for name in enabled_quantizer_names:
            if name not in names_to_optimize:
                quantizers_to_be_disabled.append(name)

        return quantizers_to_be_disabled

    def _get_quantizer_names_to_optimize(self, plan: FrontierPlan) -> List[str]:
        """
        Get list of enabled param quantizer names of the layers the plan optimizes
        (top-level only). Layers inside control-flow bodies are not optimized, so their
        param quantizers are not listed here.

        :param plan: Frontier plan whose levels give the layers to optimize
        :return Returns the param quantizer names of the layers the plan optimizes.
        """
        param_quantizer_names = []
        for layer in (layer for level in plan.levels for layer in level):
            name = get_weight_name(layer.node)
            quantizer = self.sim.qc_quantize_op_dict.get(name)
            if quantizer and quantizer.enabled:
                param_quantizer_names.append(name)

        return param_quantizer_names

    @contextmanager
    def _disable_subgraph_quantizers(self, model: onnx.ModelProto):
        """
        Temporarily disables the quantizers of the given subgraph, including the ones inside
        control-flow bodies, to run it in full precision. Restores them on exit.

        :param model: Model containing subgraph
        """
        quantizer_keys = [
            node.input[0]
            for graph in _iterate_graphs_recursive(model.graph)
            for node in graph.node
            if node.op_type == "QcQuantizeOp"
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

    def _get_min_and_max_for_candidate_selection(
        self,
        layer: SeqMseLayer,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get recalibrated min/max values for candidate selection.

        :param layer: Layer to optimize
        :return: Tuple of min and max values for candidate selection.
        """
        # pylint: disable=protected-access

        weight_name = get_weight_name(layer.node)

        # Retrieve the quantizer and its encodings
        quantizer = self.sim.qc_quantize_op_dict[weight_name]
        encodings = quantizer.get_encodings()
        encoding_shape = quantizer._encoding_shape()

        min_tensor = np.array([enc.min for enc in encodings], dtype=np.float32)
        max_tensor = np.array([enc.max for enc in encodings], dtype=np.float32)

        # Reshape if encoding_shape is not scalar
        if encoding_shape:
            min_tensor = min_tensor.reshape(encoding_shape)
            max_tensor = max_tensor.reshape(encoding_shape)

        return min_tensor, max_tensor

    def _get_candidate(
        self,
        candidate_idx: Union[int, np.ndarray],
        min_tensor: np.ndarray,
        max_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get candidate min and max tensors on the fly.

        NOTE: Divides `min_tensor` and `max_tensor` into `num_candidates` equal parts and
              select the values corresponding to the `candidate_idx + 1`

        :param min_tensor: Min tensor
        :param max_tensor: Max tensor
        :return: candidates
        """
        num_candidates = self.params.num_candidates

        cand_min = min_tensor / num_candidates * (candidate_idx + 1)
        cand_max = max_tensor / num_candidates * (candidate_idx + 1)

        return cand_min, cand_max

    def _compute_encoding_from_candidate(
        self,
        layer: SeqMseLayer,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ):
        """
        Computes the encoding using candidate min and candidate max values.

        :param layer: Layer the encoding is for
        :param x_min: min values
        :param x_max: max values
        """
        # Get the parameter name and corresponding quantizer
        weight_name = get_weight_name(layer.node)
        quantizer = self.sim.qc_quantize_op_dict[weight_name]

        # Extract quantization configuration
        channel_axis = quantizer.quant_info.channelAxis
        block_size = quantizer.quant_info.blockSize

        if np.isscalar(x_min) and np.isscalar(x_max):
            x_min = np.asarray([x_min], dtype=np.float32)
            x_max = np.asarray([x_max], dtype=np.float32)

        # Stack x_min and x_max to form candidate tensor
        if block_size > 0:
            # For per-block quantization, always stack along axis=0 -> shape: (2, ...)
            cand = np.stack((x_min, x_max), axis=0)

        # For per-tensor/per-channel, stack based on channel axis
        else:
            if channel_axis == 0:
                # Channels along axis 0 -> shape: (num_channels, 2)
                cand = np.stack((x_min, x_max), axis=1)
            elif channel_axis == 1:
                # Channels along axis 1 -> shape: (2, num_channels)
                cand = np.stack((x_min, x_max), axis=0)
            else:
                raise ValueError(f"Unsupported channel_axis: {channel_axis}")

        # Reset and update encodings statistics with the candidate tensor
        quantizer.reset_encoding_stats()
        quantizer.update_encoding_stats(cand)

        # Compute final encodings and set QDQ mode
        quantizer.compute_encodings()
        quantizer.op_mode = TensorQuantizerOpMode.quantizeDequantize

    def _freeze_encodings(self, layer: SeqMseLayer):
        """
        Freezes the encoding after the node is optimized
        :param layer: Optimized layer
        """
        weight_name = get_weight_name(layer.node)
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

    def _compute_recon_loss(
        self,
        fp_outputs: torch.Tensor,
        sim_outputs: torch.Tensor,
        layer: SeqMseLayer,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss and return the sum by reducing over all the dimensions except last channel dimension.

        :param fp_outputs: fp_outputs (X^W)
        :param sim_outputs: sim_outputs (X^W^)
        :param layer: Layer the outputs are of
        :return: loss
        """
        xqwq = sim_outputs
        xqw = fp_outputs

        loss_fn_map = {
            "mse": torch.nn.functional.mse_loss,
            "l1": torch.nn.functional.l1_loss,
            "sqnr": SequentialMse.neg_sqnr,
        }
        if self.params.loss_fn not in loss_fn_map:
            raise ValueError(f"Invalid loss function: {self.params.loss_fn}")

        loss_fn = loss_fn_map[self.params.loss_fn]

        weight_name = get_weight_name(layer.node)
        quantizer = self.sim.qc_quantize_op_dict[weight_name]

        if quantizer is None:
            raise KeyError(f"Quantizer not found for {weight_name}")

        tensor_shape = quantizer.tensor_quantizer_params.tensor_shape
        block_size = quantizer.quant_info.blockSize
        channel_axis = quantizer.quant_info.channelAxis
        block_axis = quantizer.quant_info.blockAxis

        # Handle block-wise quantization
        if block_size > 0:
            if layer.node.op_type == "Conv":
                loss = loss_fn(xqwq, xqw, reduction="none").sum(dim=(0, 2, 3))
                loss = loss.reshape(tensor_shape[block_axis] // block_size, -1)
                loss = loss.permute(1, 0)
            elif layer.node.op_type in ["Gemm", "MatMul"]:
                loss = loss_fn(xqwq, xqw, reduction="none").sum(dim=1)
                if block_axis > channel_axis:
                    loss = loss.permute(1, 0)  # For transposed form of Gemm
            else:
                raise NotImplementedError(
                    f"Unsupported op type with block quantization: {layer.node.op_type}"
                )
            return loss

        # Handle per-tensor and per-channel case
        if layer.node.op_type == "Conv":
            # Permute to move channel dimension to the end
            xqwq = xqwq.transpose(1, xqwq.dim() - 1)
            xqw = xqw.transpose(1, xqw.dim() - 1)

        # Flatten all dimensions except channel
        channel_dim = xqwq.shape[-1]
        xqwq = xqwq.reshape(-1, channel_dim)
        xqw = xqw.reshape(-1, channel_dim)

        # Compute channel-wise loss
        loss = loss_fn(xqwq, xqw, reduction="none").sum(0)
        return loss

    def _run_seq_mse(
        self,
        layers: List[SeqMseLayer],
        progress: SeqMseProgress,
        sim_inputs: Dict[str, List[np.ndarray]],
    ):
        """
        Run Sequential MSE for all the layers of one level.

        :param layers: Layers of the level.
        :param progress: Progress report of the run this group belongs to.
        :param sim_inputs: Input name -> its array per calibration sample, for every input of
            the nodes.
        """

        def _set_candidates(candidate_index: int):
            """
            Helper function to set candidate based on index for ops at same level.
            Internally computes the encoding using candidate min and candidate max

            :param candidate_index: Index of candidate
            """
            for layer in layers:
                init_min_val, init_max_val = initial_min_max[layer.node.name]
                cand_min, cand_max = self._get_candidate(
                    candidate_index, init_min_val, init_max_val
                )
                self._compute_encoding_from_candidate(layer, cand_min, cand_max)

        def _compute_loss(
            all_fp_outputs: List[torch.Tensor], all_sim_outputs: List[torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            """
            Helper function to compute reconstruction loss for ops at same level.

            :param all_fp_outputs: FP Outputs of all ops at same level
            :param all_sim_outputs: Sim Outputs of all ops at same level
            """
            candidate_loss = {}

            for i, layer in enumerate(layers):
                loss = self._compute_recon_loss(
                    all_fp_outputs[i], all_sim_outputs[i], layer
                )
                candidate_loss[layer.node.name] = loss

            return candidate_loss

        def _get_layer_io_names(layers: List[SeqMseLayer]):
            """
            Helper function to get the input and output names of subgraph.

            :param layers: Layers of the level.
            :return: Subgraph input and output names.
            """
            subgraph_inputs = []
            subgraph_outputs = []
            for layer in layers:
                subgraph_inputs.append(layer.op_input_names[0])
                subgraph_outputs.append(layer.op_output_names[0])

            subgraph_inputs = list(set(subgraph_inputs))
            return subgraph_inputs, subgraph_outputs

        progress.start_phase("build")

        # Cache the initial min and max values
        initial_min_max = {}
        for layer in layers:
            init_min_val, init_max_val = self._get_min_and_max_for_candidate_selection(
                layer
            )
            initial_min_max[layer.node.name] = (init_min_val, init_max_val)

        subgraph_inp_names, subgraph_outs_names = _get_layer_io_names(layers)

        # For now, we only expose "symqt" input symmetry.
        assert self.params.inp_symmetry == "symqt", (
            "Only symmetric quantsim inputs ('symqt') are supported."
        )
        subgraph_path = self._split_onnx_graph(subgraph_inp_names, subgraph_outs_names)
        subgraph_model = onnx.load_model(subgraph_path, load_external_data=False)

        # Transform graph for bq/lpbq quantizers
        subgraph_model, sim_inputs, num_blocks = (
            self._transform_graph_for_block_quantization(subgraph_model, sim_inputs)
        )
        # Re-save transformed graph
        onnx.save_model(subgraph_model, subgraph_path)

        # Determine which axis carries the sequence dimension:
        #   3-D inputs (num_blocks, seq_len, block_size)  -> Linear BQ, chunk along dim 1
        #   4-D inputs (batch, c_in, 1,       seq_len)    -> Conv BQ,   chunk along dim 3
        first_input = next(iter(sim_inputs.values()))[0]
        chunk_dim = 3 if first_input.ndim == 4 else 1
        dataset_len = len(next(iter(sim_inputs.values())))

        with self._create_session(subgraph_path) as session:
            all_out_shapes, all_out_dtypes = self._infer_bq_output_shapes(
                session, subgraph_model, sim_inputs, num_blocks, chunk_dim, dataset_len
            )

            torch_device = get_torch_device(session)
            device_type = torch_device.type
            device_id = torch_device.index if torch_device.index is not None else 0

            output_names = [out.name for out in session.get_outputs()]

            # Sequence chunking splits sim_inputs along the sequence dimension so each
            # "batch" is a smaller chunk.
            chunk_size = _compute_seq_chunk_size(
                num_blocks,
                sim_inputs,
                all_out_shapes,
                all_out_dtypes,
                torch_device,
                chunk_dim,
            )
            if chunk_size is not None:
                progress.notice_once(
                    "sequence chunking",
                    "Large block-quantized outputs: chunking the sequence (first chunk_size=%d) "
                    "to fit both fp and quantized block outputs on the GPU.",
                    chunk_size,
                )
                _logger.debug("Sequence chunking with chunk_size=%d", chunk_size)
                sim_inputs, all_out_shapes, all_out_dtypes = _rechunk_seq_inputs(
                    sim_inputs, chunk_size, all_out_shapes, all_out_dtypes, chunk_dim
                )
                dataset_len = len(next(iter(sim_inputs.values())))

            # Store accumulated loss per layer across candidates
            total_loss = defaultdict(list)

            progress.start_phase("search")
            for candidate_index in range(self.params.num_candidates):
                _set_candidates(candidate_index)

                # Initialize per layer loss accumulator
                accumulated_loss = defaultdict(lambda: 0)

                for batch_idx in range(dataset_len):
                    # Create OrtValues per batch and bind them immediately
                    # to avoid memory reuse issues in ORT
                    binding = session.io_binding()

                    # Bind all inputs
                    for name, data in sim_inputs.items():
                        inp_ort_value = _ortvalue_from_numpy(
                            data[batch_idx], device_type, device_id
                        )
                        binding.bind_input(
                            name=name,
                            device_type=device_type,
                            device_id=device_id,
                            element_type=_onnx_element_type(data[batch_idx].dtype),
                            shape=data[batch_idx].shape,
                            buffer_ptr=inp_ort_value.data_ptr(),
                        )

                    # Bind outputs dynamically
                    shared_outputs = []
                    for out_name, out_shape, out_dtype in zip(
                        output_names,
                        all_out_shapes[batch_idx],
                        all_out_dtypes[batch_idx],
                    ):
                        torch_dtype = map_np_dtype_to_torch(out_dtype)
                        shared_tensor = torch.empty(
                            out_shape, dtype=torch_dtype, device=torch_device
                        )
                        shared_outputs.append(shared_tensor)
                        binding.bind_output(
                            name=out_name,
                            device_type=device_type,
                            device_id=device_id,
                            element_type=_onnx_element_type(out_dtype),
                            shape=out_shape,
                            buffer_ptr=shared_tensor.data_ptr(),
                        )
                    # Run fp inference
                    with self._disable_subgraph_quantizers(subgraph_model):
                        session.run_with_iobinding(binding)
                    fp_outputs = [out.clone() for out in shared_outputs]

                    # Run sim inference
                    session.run_with_iobinding(binding)
                    sim_outputs = [out for out in shared_outputs]

                    batched_loss = _compute_loss(fp_outputs, sim_outputs)
                    for node_name, b_loss in batched_loss.items():
                        accumulated_loss[node_name] += b_loss

                    del binding, fp_outputs, sim_outputs, shared_outputs

                torch.cuda.empty_cache()

                # After all batches, append the accumulated loss for this candidate
                for node_name, loss in accumulated_loss.items():
                    total_loss[node_name].append(loss)

                _logger.debug(f"Finished candidate {candidate_index}")

        del sim_inputs

        progress.start_phase("finalize")
        results = []
        # Postprocessing (not vectorized)
        for layer in layers:
            loss = total_loss[layer.node.name]
            stacked_loss = torch.stack(
                loss, dim=0
            )  # Resulting shape depends on granularity: per-tensor and per-channel: (num_candidates, num_channels) and per-block: (num_candidates, num_channels, num_blocks)

            # Find the index of the minimum loss along the candidate axis (axis=0)
            best_indices = stacked_loss.min(0)[1]
            results.append(
                weight_result(
                    display_name(layer.node),
                    stacked_loss,
                    best_indices,
                    self.params.num_candidates,
                )
            )
            init_min, init_max = initial_min_max[layer.node.name]

            # Unsqueeze best_indices until it matches dim length of max_val
            while best_indices.ndim < init_max.ndim:
                best_indices = best_indices[..., None]
            best_indices = best_indices.cpu().numpy()

            # Use the best indices to compute corresponding best candidate min and max tensors
            best_min, best_max = self._get_candidate(best_indices, init_min, init_max)

            # Compute and freeze parameter encodings using best candidate
            self._compute_encoding_from_candidate(layer, best_min, best_max)
            self._freeze_encodings(layer)

        layer_names = [layer.node.name for layer in layers]
        _logger.debug(
            "Computed optimal parameter encodings for ops: %s",
            ", ".join(layer_names),
        )
        progress.end_group(results)

    @staticmethod
    def _infer_bq_output_shapes(
        session: OrtInferenceSession,
        subgraph_model: onnx.ModelProto,
        sim_inputs: Dict[str, List[np.ndarray]],
        num_blocks: int,
        chunk_dim: int,
        dataset_len: int,
    ):
        """
        Infer output shapes and dtypes for a block-quantized subgraph.

        For BQ subgraphs, uses a single-token probe along chunk_dim to avoid ORT's CUDA arena retaining a large allocation equal to the full
        grouped output size, then scales the probed shapes back to the full sequence length.

        :param session: ORT inference session for the subgraph.
        :param subgraph_model: ONNX model proto of the subgraph.
        :param sim_inputs: Dict mapping input name -> list of numpy arrays.
        :param num_blocks: Number of BQ blocks (0 means no block quantization).
        :param chunk_dim: Axis carrying the sequence dimension (1 for Linear BQ, 3 for Conv BQ).
        :param dataset_len: Number of dataset samples.
        :return: Tuple of (all_out_shapes, all_out_dtypes) each a list of length dataset_len.
        """
        first_input = next(iter(sim_inputs.values()))[0]

        if num_blocks > 0 and _has_unit_kernel_only(subgraph_model):
            slices = [slice(None)] * first_input.ndim
            slices[chunk_dim] = slice(None, 1)
            probe_input = {
                name: [arr_list[0][tuple(slices)]]
                for name, arr_list in sim_inputs.items()
            }
            probe_shapes, probe_dtypes = _infer_out_shapes_and_dtypes(
                session, probe_input
            )

            full_seq_len = first_input.shape[chunk_dim]
            scaled_shapes = [
                s[:chunk_dim] + (full_seq_len,) + s[chunk_dim + 1 :]
                for s in probe_shapes[0]
            ]
            all_out_shapes = [scaled_shapes] * dataset_len
            all_out_dtypes = [probe_dtypes[0]] * dataset_len
        else:
            all_out_shapes, all_out_dtypes = _infer_out_shapes_and_dtypes(
                session, sim_inputs
            )

        return all_out_shapes, all_out_dtypes

    def _split_onnx_graph(self, input_names: List[str], output_names: List[str]) -> str:
        """
        Splits the onnx graph from input names to output names. The subgraph references the
        external data of the saved model.

        NOTE: Unlike onnx.utils.Extractor, onnx_ir also keeps the producers of values that
        subgraph attributes (e.g. Scan bodies) read from the outer scope.

        :param input_names: input names of split graph
        :param output_names: output names of split graph
        :return: path to the subgraph
        """
        graph = onnx_ir.convenience.extract(
            self._ir_model.graph, list(input_names), list(output_names)
        )
        # ORT requires every graph input and output to have a type.
        for value in itertools.chain(graph.inputs, graph.outputs):
            if value.dtype is None:
                value.dtype = onnx_ir.DataType.from_numpy(
                    self.sim.activation_dtypes[value.name]
                )
        model = onnx_ir.Model(
            graph,
            ir_version=self._ir_model.ir_version,
            functions=list(self._ir_model.functions.values()),
        )
        with tempfile.NamedTemporaryFile(
            suffix=".onnx", dir=self._model_dir.name, delete=False
        ) as f:
            subgraph_path = f.name

        # self._ir_model's initializers reference model.onnx.data, so saving without
        # external_data= writes only those references; keep splits in this directory.
        onnx_ir.save(model, subgraph_path)
        return subgraph_path

    def _frontier_traversal(self, plan: FrontierPlan, names_to_optimize: List[str]):
        """
        Optimize the layers level by level, sampling each level's inputs with a
        BlockwiseSampler that advances a cache of the frontier tensors from one level to the
        next. A level is optimized on inputs computed with every earlier level frozen.

        :param plan: Frontier plan giving the levels to optimize and the tensors to sample
        :param names_to_optimize: Param quantizer names optimized by seq mse
        """
        _logger.debug(
            "Frontier sampling plan: %d levels, block inputs %s",
            len(plan.levels),
            plan.block_inputs,
        )
        sampler = BlockwiseSampler(
            self._ir_model,
            SamplingPlan(self._ir_model.graph, plan.block_inputs, plan.block_outputs),
            self._inputs,
            self.sim.providers,
            may_update=functools.partial(
                _is_unfrozen, set(names_to_optimize), self.sim.qc_quantize_op_dict
            ),
            session_options=self.sim._ort_session_options,  # pylint: disable=protected-access
        )
        progress = SeqMseProgress(
            [[layer.node for layer in level] for level in plan.levels],
            num_candidates=self.params.num_candidates,
            num_batches=len(self._inputs),
            loss_fn=self.params.loss_fn,
            excluded=self._excluded_weight_names(),
        )

        # The summary still reports what was done if a group fails
        try:
            for idx, level in enumerate(plan.levels):
                progress.start_phase("cache inputs")
                samples = sampler.sample_block(idx)
                position = {name: i for i, name in enumerate(plan.block_inputs[idx])}
                # Each layer's subgraph is split on its first activation input
                sim_inputs = {
                    layer.op_input_names[0]: [
                        sample[position[layer.op_input_names[0]]] for sample in samples
                    ]
                    for layer in level
                }
                del samples
                self._run_seq_mse(level, progress, sim_inputs)
                # Free this level's inputs so the next level's sampling doesn't hold two levels
                del sim_inputs
        finally:
            progress.finish()

    def _frontier_plan(self) -> FrontierPlan:
        """Levels of the layers to optimize, and the tensors to sample for each level."""
        layers = find_layers(_sim_nodes(self.sim), self._nodes_to_exclude)
        layer_names = {layer.node.name for layer in layers}
        # SamplingPlan needs a sorted graph, and the sim appends its quantizers to the end
        onnx_ir.passes.common.TopologicalSortPass().call(self._ir_model)
        return plan_frontier(
            self._ir_model.graph,
            layers,
            weighted_op_names=[
                node.name
                for node in _sim_nodes(self.sim)
                if is_supported_op(node) and node.name not in layer_names
            ],
        )

    def _excluded_weight_names(self) -> List[str]:
        """Names of the excluded ops sequential MSE would otherwise optimize."""
        excluded = set(self._nodes_to_exclude)
        return [
            node.name
            for node in _sim_nodes(self.sim)
            if node.name in excluded and is_supported_op(node)
        ]

    @contextmanager
    def _create_session(self, model: Union[onnx.ModelProto, str]):
        """
        Build and return onnxruntime inference session

        :param model: onnx model or path to onnx model
        :return: Session
        """
        try:
            session = OrtInferenceSession(
                model,
                self.sim.providers,
                session_options=self.sim._ort_session_options,
                path=self.sim._path,
                save_as_external_data=self.sim._use_external_data,
            )
            yield session
        finally:
            del session

    def _transform_graph_for_block_quantization(
        self, model: onnx.ModelProto, sim_inputs: Dict
    ):
        """
        Identifies BQ/LPBQ quantizers in the given subgraph and modifies the graph
        to compute block-wise reconstruction loss based on the detected quantization configurations.

        NOTE: If no BQ/LPBQ quantizers are found, the model is returned unchanged.

        Assumptions:
         - If block-wise quantizer(s) are found, all of them should be block-wise quantizers.
         - All block-wise quantizers in the subgraph must have the same `block_size` and `block_axis`.
         - The subgraph must either contain Conv ops or Gemm/MatMul ops, but not both.
         - With the above assumptions satisfied, graph transformation can be performed in a single pass
          across all the relevant ops in the subgraph.

        :param model: Model containing subgraph
        :return: Tuple of (modified model, sim_inputs, num_blocks).
                 num_blocks == 0  -> no BQ quantizers found; skip sequence chunking.
                 num_blocks > 0 for linear BQ: equals the first dimension of the transformed
                 sim_inputs (num_blocks, seq_len, block_size).
                 num_blocks == 1 for Conv BQ: sim_inputs are not reshaped, but
                 the grouped-conv output is large enough to require sequence chunking.
        """
        param_names = set(self.sim.param_names)

        quantizer_keys = [
            node.input[0]
            for node in model.graph.node
            if node.op_type == "QcQuantizeOp" and node.input[0] in param_names
        ]

        # Filter only weight quantizers
        quantizers = {
            name: self.sim.qc_quantize_op_dict[name]
            for name in quantizer_keys
            if self.sim.qc_quantize_op_dict[name].enabled
        }

        bq_quantizers = [
            quantizer
            for quantizer in quantizers.values()
            if quantizer.quant_info.blockSize > 0
        ]

        # Early exit if no block quantizers found
        if not bq_quantizers:
            return model, sim_inputs, 0

        # If some quantizers are block-wise and non block-wise raise an error
        # Add support later
        if len(quantizers) != len(bq_quantizers):
            raise NotImplementedError(
                f"Mixed usage of block-wise and non block-wise quantizers is not supported"
            )

        ref_block_size = bq_quantizers[0].quant_info.blockSize
        ref_block_axis = bq_quantizers[0].quant_info.blockAxis

        for quantizer in bq_quantizers[1:]:
            q_info = quantizer.quant_info
            if q_info.blockAxis != ref_block_axis or q_info.blockSize != ref_block_size:
                raise RuntimeError(
                    f"All block quantizers in the subgraph should have the same quantization configuration"
                )

        conv_ops = {"Conv"}
        linear_ops = {"MatMul", "Gemm"}

        op_types = {node.op_type for node in model.graph.node}
        has_conv = bool(op_types.intersection(conv_ops))
        has_linear = bool(op_types.intersection(linear_ops))

        if has_conv and has_linear:
            raise RuntimeError(f"Subgraph contains both Conv and linear ops.")

        # Transform graph only for block quantization
        if has_conv:
            modify_graph_with_grouped_conv(model, ref_block_size, ref_block_axis)
            # sim_inputs are not reshaped for Conv (no per-block input decomposition needed).
            # Return num_blocks=1 to trigger sequence chunking in _compute_seq_chunk_size;
            # the grouped-conv output shape (1, num_blocks*c_out, 1, seq_len)
            return model, sim_inputs, 1
        elif has_linear:
            sim_inputs = prepare_linear_inputs(sim_inputs, block_size=ref_block_size)
            modify_graph_with_grouped_linear(model, ref_block_size, ref_block_axis)
            # num_blocks is the first dimension of the transformed inputs
            # (num_blocks, seq_len, block_size).
            num_blocks = next(iter(sim_inputs.values()))[0].shape[0]
            return model, sim_inputs, num_blocks
        else:
            raise RuntimeError(f"Subgraph contains no conv or linear ops.")


def _sim_nodes(sim: QuantizationSimModel) -> List[onnx_ir.Node]:
    """
    The nodes of the sim's model from before quantizers were inserted, which are named like its
    quantizers: the nodes the sim quantizes, so those of If branches only if it quantizes them.
    """
    # pylint: disable=protected-access
    graphs = _iterate_graphs_recursive_ir(
        sim._ir_model.graph, include_if=sim._quantize_subgraphs
    )
    return [node for graph in graphs for node in graph]


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


@contextmanager
def _temporarily_disable_block_grouping(sim: QuantizationSimModel):
    """
    Set all grouped block quantizers to regular block-wise quantization for the duration of the context manager.

    NOTE: disabling scale quantization is equivalent to standard block-wise quantization, as each block has its own encodings.

    :param sim: QuantizationSimModel object
    """
    quantizers = [
        sim.qc_quantize_op_dict[name]
        for name in sim.param_names
        if sim.qc_quantize_op_dict[name].enabled
        and sim.qc_quantize_op_dict[name]._scale_quantizer is not None
    ]

    original_scale_quantizers = {
        quantizer: quantizer._scale_quantizer for quantizer in quantizers
    }

    try:
        for quantizer in quantizers:
            quantizer._scale_quantizer = None
        yield
    finally:
        for quantizer, original_scale_quantizer in original_scale_quantizers.items():
            quantizer._scale_quantizer = original_scale_quantizer


@contextmanager
def _use_min_max_quant_scheme(quantizers: List[QcQuantizeOp]):
    """
    Set the given quantizers to the min-max quant scheme for the duration of the context manager.

    NOTE: Sequential MSE derives encodings by passing exact candidate min/max values through the
          quantizer, which histogram-based schemes such as TF-Enhanced would distort.

    :param quantizers: Quantizers to be set to min-max
    """
    original_quant_schemes = {
        quantizer: quantizer.quant_scheme
        for quantizer in quantizers
        if quantizer.quant_scheme != QuantScheme.post_training_tf
        and not quantizer.is_encoding_frozen()
    }

    try:
        for quantizer in original_quant_schemes:
            quantizer.set_quant_scheme(QuantScheme.post_training_tf)
        yield

    finally:
        for quantizer, quant_scheme in original_quant_schemes.items():
            # set_quant_scheme discards the encodings, so carry over the ones computed under min-max.
            encodings = quantizer.get_encodings()
            was_frozen = quantizer.is_encoding_frozen()
            quantizer._is_encoding_frozen = False  # pylint: disable=protected-access
            quantizer.set_quant_scheme(quant_scheme)
            if encodings:
                quantizer.load_encodings(encodings)
            if was_frozen:
                quantizer.freeze_encodings()


def _has_unit_kernel_only(model: onnx.ModelProto) -> bool:
    """
    Return True if every Conv node in the model has all-ones kernel_shape (e.g. 1×1).

    The single-token probe used to infer output shapes assumes output_W == input_W, which
    only holds when the kernel does not reduce the spatial dimension.  Non-unit kernels
    (e.g. 3×3) change the output size, making the probe's scaled shapes incorrect.

    :param model: Model containing subgraph
    :return: True if every Conv has an all-ones kernel, False otherwise.
    """
    initializer_shapes = {
        init.name: list(init.dims) for init in model.graph.initializer
    }

    for node in model.graph.node:
        if node.op_type != "Conv":
            continue

        # Prefer the explicit kernel_shape attribute.
        kernel_shape = None
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
                break

        if kernel_shape is None:
            # Fall back to the weight initializer: input[1] is the weight tensor,
            # whose shape is [C_out, C_in/group, *spatial_dims].
            if len(node.input) >= 2:
                weight_name = node.input[1]
                weight_dims = initializer_shapes.get(weight_name)
                if weight_dims is not None and len(weight_dims) >= 3:
                    kernel_shape = weight_dims[2:]

        if kernel_shape is None:
            # Cannot determine kernel shape — assume non-unit.
            return False

        if any(k != 1 for k in kernel_shape):
            return False

    return True


def _compute_seq_chunk_size(
    num_blocks: int,
    sim_inputs: Dict[str, List[np.ndarray]],
    all_out_shapes: List,
    all_out_dtypes: List,
    torch_device: torch.device,
    chunk_dim: int = 1,
) -> Optional[int]:
    """
    Compute the sequence chunk size needed to hold both fp and quantized block outputs for
    (BQ/LPBQ) subgraphs.

    :param num_blocks: Number of blocks.
                       0 means no block transformation was applied (chunking skipped).
    :param sim_inputs: Inputs dict.
    :param all_out_shapes: Per-batch list of output shapes.
    :param all_out_dtypes: Per-batch list of output dtypes.
    :param torch_device: PyTorch device.
    :param chunk_dim: Axis that represents the sequence dimension.
                      1 for 3-D Linear inputs (num_blocks, seq_len, block_size);
                      3 for 4-D Conv inputs (batch, c_in, 1, seq_len).
    :return: Chunk size (number of tokens per chunk), or None if chunking is not needed.
    """
    if num_blocks == 0 or torch_device.type != "cuda":
        return None

    # Bytes per token across all outputs.
    bytes_per_token = sum(
        int(np.prod(shape)) // shape[chunk_dim] * np.dtype(dtype).itemsize
        for shape, dtype in zip(all_out_shapes[0], all_out_dtypes[0])
    )
    if bytes_per_token == 0:
        return None

    free_mem, _ = torch.cuda.mem_get_info(torch_device)
    avail_mem = max(0, free_mem)

    # fp_outputs + sim_outputs + mse_loss(reduction='none') output are all alive simultaneously.
    # Unlike PyTorch SeqMSE (which processes one block at a time and uses factor 2)
    max_tokens = max(int(avail_mem * _GPU_ALLOC_MARGIN / (3 * bytes_per_token)), 1)

    # cuDNN uses int32 indexing, so each output tensor must contain at most
    # 2**31 - 1 elements.
    elems_per_token = max(
        int(np.prod(shape)) // shape[chunk_dim] for shape in all_out_shapes[0]
    )
    max_tokens = min(
        max_tokens,
        max(np.iinfo(np.int32).max // elems_per_token, 1),
    )

    first_input = next(iter(sim_inputs.values()))[0]
    seq_len = first_input.shape[chunk_dim]

    # Full sequence fits
    if max_tokens >= seq_len:
        return None

    return max_tokens


def _rechunk_seq_inputs(
    sim_inputs: Dict[str, List[np.ndarray]],
    chunk_size: int,
    all_out_shapes: List,
    all_out_dtypes: List,
    chunk_dim: int = 1,
) -> Tuple[Dict, List, List]:
    """
    Splits inputs along the sequence dimension into chunks of at most chunk_size tokens.
    Returns updated sim_inputs, all_out_shapes, and all_out_dtypes sized to match the chunks.

    :param sim_inputs: Inputs dict.
    :param chunk_size: Maximum number of tokens (seq_len steps) per chunk.
    :param all_out_shapes: Per-batch list of output shapes.
    :param all_out_dtypes: Per-batch list of output dtypes.
    :param chunk_dim: Axis that represents the sequence dimension.
                      1 for 3-D Linear inputs (num_blocks, seq_len, block_size);
                      3 for 4-D Conv inputs (batch, c_in, 1, seq_len).
    :return: Tuple of (new sim_inputs, new all_out_shapes, new all_out_dtypes).
    """
    new_sim_inputs: Dict[str, List[np.ndarray]] = {}
    for key, arr_list in sim_inputs.items():
        new_arr_list = []
        for arr in arr_list:
            splits = list(range(chunk_size, arr.shape[chunk_dim], chunk_size))
            new_arr_list.extend(np.array_split(arr, splits, axis=chunk_dim))
        new_sim_inputs[key] = new_arr_list

    # Derive per-chunk output shapes by replacing shape[chunk_dim] with the chunk seq_len.
    template_shapes = all_out_shapes[0]
    template_dtypes = all_out_dtypes[0]
    first_key = next(iter(new_sim_inputs))
    new_all_out_shapes = []
    new_all_out_dtypes = []
    for chunk_arr in new_sim_inputs[first_key]:
        chunk_seq_len = chunk_arr.shape[chunk_dim]
        new_all_out_shapes.append(
            [
                s[:chunk_dim] + (chunk_seq_len,) + s[chunk_dim + 1 :]
                for s in template_shapes
            ]
        )
        new_all_out_dtypes.append(template_dtypes)

    return new_sim_inputs, new_all_out_shapes, new_all_out_dtypes


def _ortvalue_from_numpy(
    array: np.ndarray, device_type: str, device_id: int
) -> onnxruntime.OrtValue:
    """
    Copy ``array`` to an OrtValue on the given device, as a buffer to bind by pointer.

    ``OrtValue.ortvalue_from_numpy`` rejects bfloat16 arrays, so those are copied as their
    raw bits; the bfloat16 element type is given when the buffer is bound.
    """
    if array.dtype == _BFLOAT16:
        array = np.ascontiguousarray(array).view(np.uint16)
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        array, device_type=device_type, device_id=device_id
    )


def _infer_out_shapes_and_dtypes(
    session: OrtInferenceSession, data: Dict[str, List[torch.Tensor]]
) -> Tuple[List, List]:
    """

    Infers output shapes and data types for each batch of inputs using the provided ONNX Runtime session.

    This function runs the model on each batch of input tensors
    and collects the output shapes and dtypes for later use in output buffer allocation.

    TODO: Instead of running inference, just consider inspecting the graph for dynamic shapes.

    :param session: ORT inference session.
    :param data: Dictionary of input tensors.
    :return: List of output shapes and dtypes per batch of inputs.
    """
    dataset_len = len(next(iter(data.values())))
    all_out_shapes = []
    all_out_dtypes = []

    for batch_idx in range(dataset_len):
        input_batch = {name: array_list[batch_idx] for name, array_list in data.items()}
        outputs = _run_session_to_numpy(session, input_batch)
        all_out_shapes.append([out.shape for out in outputs])
        all_out_dtypes.append([out.dtype for out in outputs])
        del outputs

    return all_out_shapes, all_out_dtypes


def _collect_inputs(
    inputs: Iterable[Dict[str, np.ndarray]], input_names: List[str]
) -> List[Dict[str, np.ndarray]]:
    """
    The calibration samples, each a dict of every graph input.

    :param inputs: Calibration samples
    :param input_names: Names of the graph inputs
    :return: The samples, restricted to the graph inputs
    """
    samples = []
    for sample in inputs:
        if not isinstance(sample, dict):
            raise TypeError(
                f"Expected each input sample to be type `Dict[str, np.ndarray]` but got type {type(sample)}. "
                "To resolve this, ensure that `inputs` argument is type `Iterable[Dict[str, np.ndarray]]`"
            )
        for name in input_names:
            if name not in sample:
                raise ValueError(
                    f"All inputs to the graph must be present in the dataloader. {name} is missing in the dataloader"
                )
        samples.append({name: sample[name] for name in input_names})
    return samples


def _is_unfrozen(
    names_to_optimize: Collection[str],
    quantizer_dict: Dict[str, QcQuantizeOp],
    name: str,
) -> bool:
    """Whether sequential MSE may still change the encodings of the quantizer of ``name``."""
    return name in names_to_optimize and not quantizer_dict[name].is_encoding_frozen()

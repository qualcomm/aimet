# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=protected-access
import operator
from contextlib import contextmanager
from typing import Optional, Callable
import torch
from torch._C import _fx_map_arg
from torch.export.graph_signature import TensorArgument
import aimet_torch
from aimet_torch.quantization.affine import QuantizeDequantize
from aimet_torch.experimental.export.export import _post_process
from aimet_torch.experimental.export._utils import (
    _is_grid_preserving_op,
    _remove_dangling_nodes,
    _eval_node,
    _insert_placeholder,
    _is_multi_output_op,
    _refresh_output_specs,
    _refresh_output_specs_ctx,
    _is_torch_ao_qdq_node,
    _is_aimet_qdq_node,
    _get_aimet_qdq_module,
)
from aimet_torch.experimental.export.qnn import _qnn_decompositions
from aimet_torch.experimental.backends import BackendDescription


class ExportedProgram(torch.export.ExportedProgram):
    """
    Quantization-friendly ExportedProgram subclass for QNN.

    Conceptually,

                                 Export   (`torch.export`)
                                   |
                          ATen ... |
                                   |
                                   V
                                 Lower    (`from_torch_exported_program`)
                                   |
    quantization-friendly ATen ... |
                                   |
                                   V
                                 Quantize (`compute_missing_encodings`)
                                   |
                quantized ATen ... |
                                   |
                                   V
                                 Lower
                                   |
           quantized core ATen ... |
                                   |
                                   V
                                 (...)
    """

    @classmethod
    def from_torch_exported_program(
        cls, ep: torch.export.ExportedProgram
    ) -> "ExportedProgram":
        """
        Convert a PyTorch ExportedProgram to AIMET ExportedProgram,
        lowering ATen IR to quantization-friendly IR for QNN
        """
        # Run QNN decompositions to perform post-export quantization based on QNN-friendly IR
        ep = cls._shallow_copy_as_subclass(ep)
        return ep.run_decompositions(_qnn_decompositions())

    def run_decompositions(
        self,
        decomp_table: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
        decompose_custom_triton_ops: bool = False,
    ) -> "ExportedProgram":
        ep = super().run_decompositions(decomp_table, decompose_custom_triton_ops)

        # super().run_decompositions() can ambiguate metadata of some constant nodes
        # that is useful for determining whether the constant originated from
        # Tensor or non-Tensor constant (numpy array, python scalar, etc) in the
        # original model. To avoid such ambiguity, restore the original metadata
        # of placeholder nodes from the original ExportedProgram.
        original_node_metadata = {
            node.name: node.meta
            for node in self.graph_module.graph.nodes
            if node.op == "placeholder"
        }
        for node in ep.graph_module.graph.nodes:
            orig_metadata = original_node_metadata.get(node.name)
            if orig_metadata is not None:
                node.meta.clear()
                node.meta.update(original_node_metadata[node.name])

        # super().run_decompositions returns the base ExportedProgram classs.
        # Cast it back to subclass to access additional methods and properties
        return self._shallow_copy_as_subclass(ep)

    @classmethod
    def _shallow_copy_as_subclass(cls, ep: torch.export.ExportedProgram):
        """
        Cast a shallow-copy of PyTorch ExprotedProgram to aimet subclass
        """
        if type(ep) == cls:
            return ep

        new_ep = cls.__new__(cls)  # pylint: disable=no-value-for-parameter
        new_ep.__dict__ = ep.__dict__
        return new_ep

    @contextmanager
    def compute_missing_encodings(
        self,
        param_bw: int,
        activation_bw: int,
        target_backend: str | None = None,
    ):
        """
        Add missing output quantizers and compute encodings.

        Args:
            param_bw: Bitwidth for weight quantization
            activation_bw: Bitwidth for activation quantization
            target_backends: Target backend(s) for the quantized model.
                The model will be configured for the specified backend(s).

        Example:

            >>> # *Partially* quantized aten graph
            >>> ep = aimet_torch.experimental.export.export(sim.model, (dummy_input,))
            >>> torch.export.save(ep, "partially_quantized_model.pt2")
            >>> # Add missing quantizers and compute encodings
            >>> with ep.compute_missing_encodings(param_bw=8, activation_bw=16):
            ...     for inp in calibration_dataloader:
            ...         ep.module()(inp)
            ...
            >>> # *Fully* quantized aten graph
            >>> torch.export.save(ep, "fully_quantized_model.pt2")
        """
        # Step 1. Add missing quantizers
        # Add QuantizeDequantize module after
        # every floating-point op doesn't have quantized outputs.
        if target_backend:
            backend_description = BackendDescription.from_backend_id(target_backend)
            with _refresh_output_specs_ctx(self):
                backend_description.configure(
                    self.graph_module, param_bw, activation_bw
                )
        else:
            # If target backend is not specified, fall back to legacy behavior
            # This is only to keep CI happy. In the future, we will require target backend to be specified.
            self._add_missing_quantizers(param_bw, activation_bw)

        # Step 2. Enter compute_encodings mode and yield control back to user to
        # calibrate encodings of the newly added quantizers. While the control
        # flow is yielded, the user is expected to run forward passes with
        # calibration dataset
        with aimet_torch.nn.compute_encodings(self.module()):
            yield

        newly_added_qtzrs = [
            node.name
            for node in self.graph_module.graph.nodes
            if _is_aimet_qdq_node(node)
        ]

        # Step 3. Inline QuantizeDequantize modules into torch.ao Q/DQ operators
        # (torch.ops.quantized_decomposed.(de)quantize_per_*)
        self._inline_qdq()

        # Step 4. Post-processing
        _post_process(self)
        self._fold_param_qantizers()
        _remove_dangling_nodes(self)

        print(
            "Added and calibrated encodings for the following missing quantizers:"
            f"\n{newly_added_qtzrs}"
        )

    @_refresh_output_specs
    def _add_missing_quantizers(self, param_bw: int, activation_bw: int):
        graph_module = self.graph_module
        graph = graph_module.graph

        def replace(args, old: torch.fx.Node, new: torch.fx.Node):
            return _fx_map_arg(args, lambda arg: new if arg is old else arg)

        for node in graph.nodes:
            # Exclude multi-output ops which returns tuple of tensors.
            # QDQ for each output will be attached to the the succeeding getitem node
            #
            #                                      ┌─ getitem -> y0 -> QDQ
            # ... -> QDQ -> split -> (y0, y1, y2) ─┼─ getitem -> y1 -> QDQ
            #                                      └─ getitem -> y2 -> QDQ
            if _is_multi_output_op(node):
                continue

            # getitem output will inherit the previous node's input encoding
            # if the previous node is a grid-preserving op (aka data movement op)
            if node.target is operator.getitem and _is_grid_preserving_op(
                node.all_input_nodes[0]
            ):
                continue

            # Exclude grid-preserving ops (aka data movement ops) if its input is already quantized.
            # This is to avoid redundant quantize-dequantize pairs around data movement ops.
            if _is_grid_preserving_op(node) and node.target.overloadpacket not in (
                torch.ops.aten.relu,
                torch.ops.aten.relu_,
            ):
                continue

            # Nodes followed by relu will inherit output encoding from relu
            if all(
                isinstance(user.target, torch._ops.OpOverload)
                and user.target.overloadpacket
                in (torch.ops.aten.relu, torch.ops.aten.relu_)
                for user in node.users
            ):
                continue

            val = node.meta.get("val", None)

            # Exclude non-tensors from quantization
            if not isinstance(val, torch.Tensor):
                continue

            # Exclude non-floating point tensors from quantization
            if not val.dtype.is_floating_point:
                continue

            if _is_aimet_qdq_node(node):
                continue

            if _is_torch_ao_qdq_node(node):
                continue

            if all(_is_torch_ao_qdq_node(user) for user in node.users):
                continue

            if self._is_static_tensor(node):
                # TODO: Skip bias encoding calibration and derive it analytically
                #       from weight and activation encodings
                if _is_bias(node):  # bias
                    bitwidth = 32
                    symmetric = True
                else:  # weight
                    bitwidth = param_bw
                    symmetric = True
            else:  # activation
                bitwidth = activation_bw
                symmetric = False

            output_qdq = QuantizeDequantize((), bitwidth=bitwidth, symmetric=symmetric)
            output_qdq.to(node.meta["val"].device)
            graph_module.add_module(f"{node.name}_qdq", output_qdq)

            with graph.inserting_after(node):
                output_qdq_node = graph.call_module(f"{node.name}_qdq", args=(node,))

            for user in list(node.users):
                if user is output_qdq_node:
                    continue

                if _is_torch_ao_qdq_node(user):
                    continue

                user.args = replace(user.args, old=node, new=output_qdq_node)
                user.kwargs = replace(user.kwargs, old=node, new=output_qdq_node)

        # Recompile the graph
        graph_module.recompile()

    @torch.no_grad()
    @_refresh_output_specs
    def _inline_qdq(self):
        """
        Inline AIMET QuantizeDequantize nodes into torch.ops.quantized_decomposed.(de)quantize_per_*
        """
        uncalibrated_qtzrs = [
            name
            for name, qtzr in self.graph_module.named_modules()
            if isinstance(qtzr, QuantizeDequantize) and not qtzr.is_initialized()
        ]

        if uncalibrated_qtzrs:
            raise RuntimeError(
                f"Following quantizers are not calibrated:\n{uncalibrated_qtzrs}\n"
                "Please make sure to run forward passes with "
                "calibration dataset inside the context manager."
            )

        from torch.fx.passes.shape_prop import _extract_tensor_metadata
        from aimet_torch.quantization.affine.backends.torch_builtins import _get_dtype

        def _inline_qdq(
            qdq_node: torch.fx.Node,
            scale: torch.Tensor,
            zero_point: torch.Tensor,
            qmin: int,
            qmax: int,
        ):
            dtype = _get_dtype(qmin, qmax)

            if scale.dim() == zero_point.dim() == 0:
                scale = scale.item()
                zero_point = zero_point.item()
            else:
                raise NotImplementedError(
                    "Per-channel quantization is not supported yet"
                )

            (input_node,) = qdq_node.all_input_nodes

            with self.graph.inserting_after(qdq_node):
                q_node = self.graph.create_node(
                    op="call_function",
                    target=torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    args=(*qdq_node.args, scale, zero_point, qmin, qmax, dtype),
                    kwargs=qdq_node.kwargs.copy(),
                    name=f"{input_node.name}_q",
                )
                q_node.meta["val"] = input_node.meta["val"].to(dtype, copy=True)
                q_node.meta["tensor_meta"] = _extract_tensor_metadata(
                    q_node.meta["val"]
                )

            with self.graph.inserting_after(q_node):
                dq_node = self.graph.create_node(
                    op="call_function",
                    target=torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    args=(q_node, scale, zero_point, qmin, qmax, dtype),
                    kwargs=qdq_node.kwargs.copy(),
                    name=f"{input_node.name}_dq",
                )
                dq_node.meta["val"] = input_node.meta["val"].clone()
                dq_node.meta["tensor_meta"] = _extract_tensor_metadata(
                    dq_node.meta["val"]
                )

            qdq_node.replace_all_uses_with(dq_node)
            self.graph.erase_node(qdq_node)

        for qdq_node in self.graph_module.graph.nodes:
            qtzr = _get_aimet_qdq_module(qdq_node)
            if not qtzr:
                continue
            encoding = qtzr.get_encodings()
            scale = encoding.scale
            zero_point = -encoding.offset.to(torch.int32)
            _inline_qdq(qdq_node, scale, zero_point, encoding.qmin, encoding.qmax)

    def _is_static_tensor(self, node: torch.fx.Node) -> bool:
        if node.op != "placeholder":
            return False

        try:
            param_name = next(
                input_spec.target
                for input_spec in self.graph_signature.input_specs
                if isinstance(input_spec.arg, TensorArgument)
                and input_spec.arg.name == node.name
            )
        except StopIteration:
            return False

        return param_name in self.state_dict

    def _fold_param_qantizers(self):
        for node in self.graph_module.graph.nodes:
            if self._is_static_tensor(node):
                self._do_fold_param_qantizers(node)

    def _do_fold_param_qantizers(self, node: torch.fx.Node):
        if not self._is_static_tensor(node):
            raise RuntimeError

        if len(node.users) != 1:
            return

        (q,) = node.users

        if not _is_torch_ao_qdq_node(q):
            return

        if len(q.users) != 1:
            return

        (dq,) = q.users

        if not _is_torch_ao_qdq_node(dq):
            return

        Wq: torch.Tensor = _eval_node(dq.all_input_nodes[0], self)
        _, dtype_str = str(Wq.dtype).split(".")

        Wq_placeholder: torch.fx.Node = _insert_placeholder(
            self,
            val=Wq,
            node_name=f"{node.name}_{dtype_str}",
            tensor_name=f"{node.name}_{dtype_str}",
            consumers=[dq],
        )
        q.replace_all_uses_with(Wq_placeholder)


def _is_bias(node: torch.fx.Node) -> bool:
    return node.name.endswith("bias")  # TODO

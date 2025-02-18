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
# pylint: disable=redefined-outer-name
"""Utility methods for working with GPTVQ"""
import math
from typing import Optional, List, Tuple, Dict

import torch
from torch import nn

import aimet_torch.v2.quantization as Q
from aimet_torch.gptvq.activation_sampler import ActivationSampler
from aimet_torch.gptvq.defs import GPTVQParameters
from aimet_torch.v2.nn import BaseQuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel

HESSIAN_WEIGHTED_LOOKUP = False
DO_CODEBOOK_FINE_TUNING = False


def generate_codebook(weight_block: torch.Tensor,
                      num_of_centroids: int,
                      inverse_hessian_diagonal: Optional[torch.Tensor] = None,
                      assignment_chunk_size: Optional[int] = None,
                      kmeans_iteration: int = 100):
    """
    Generate and optimize codebook using K-means and return it

    :param weight_block: Weight block
    :param num_of_centroids: Number of centroids
    :param inverse_hessian_diagonal: Diagonal of inverse Hessian tensor
    :param assignment_chunk_size: Chunk size for better memory management
    :param kmeans_iteration: Number of K-means iterations
    :return: Optimized codebook
    """
    codebook = hacky_mahalanobis_init(weight_block, num_of_centroids)
    for _ in range(kmeans_iteration):
        # Expectation step
        assignments = get_assignments(weight_block, codebook, inverse_hessian_diagonal, assignment_chunk_size)

        # Maximization step
        codebook = do_kmeans_maximization(weight_block, codebook, assignments, inverse_hessian_diagonal)
    return codebook


def hacky_mahalanobis_init(tensor: torch.Tensor, num_of_centroids: int) -> torch.Tensor:
    """
    Initialize centroids using hacky Mahalanobis

    :param tensor: num_blocks_per_column x N x vector_dim weight tensor
    :param num_of_centroids: Number of centroids
    :return: Initialized codebook
    """
    vector_dim = tensor.shape[-1]
    mu = tensor.mean(1).unsqueeze(1)
    x_centered = tensor - mu

    dists = x_centered.pow(2).sum(-1)
    sorted_dists = torch.argsort(dists, dim=1)  # num_blocks_per_column x N
    idx = torch.round(torch.linspace(0, x_centered.shape[1] - 1, num_of_centroids)).long()  # num_of_centroids

    # num_blocks_per_column x num_of_centroids --> num_blocks_per_column x num_of_centroids x 1 --> num_blocks_per_column x num_of_centroids x vector_dim
    idx = (sorted_dists[:, idx].unsqueeze(-1).expand(-1, -1, vector_dim))
    return torch.gather(tensor, dim=1, index=idx)


def manipulate_inverse_hessian_diagonal(tensor: torch.Tensor,
                                        inverse_hessian_diagonal: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Manipulate diagonal of inverse Hessian tensor if needed

    :param tensor: Tensor corresponding to diagonal of inverse Hessian tensor
    :param inverse_hessian_diagonal: Diagonal of inverse Hessian tensor
    :return: Manipulated Hessian tensor
    """
    if inverse_hessian_diagonal is None:
        return torch.ones(tensor.shape[-1], device=tensor.device)

    if inverse_hessian_diagonal.ndim > 2:  # should then be 1 x N x vector_dim
        assert (
                inverse_hessian_diagonal.shape[0] == 1
                and inverse_hessian_diagonal.shape[1] == tensor.shape[1]
                and inverse_hessian_diagonal.shape[2] == tensor.shape[2]
        ), f"{inverse_hessian_diagonal.shape, tensor.shape}"
        return inverse_hessian_diagonal.unsqueeze(2)  # 1 x N x 1 x vector_dim

    return inverse_hessian_diagonal


def generate_tensor_chunks(tensor: torch.Tensor,
                           chunk_size: Optional[int]) -> List[torch.Tensor]:
    """
    Generate chunks of torch.Tensor

    :param tensor: torch.Tensor
    :param chunk_size: Chunk size
    :return: Tensor chunks
    """
    if chunk_size is None:
        return [tensor]

    return torch.split(tensor, chunk_size, dim=1)


def generate_hessian_chunks(hessian: torch.Tensor,
                            num_of_tensor_chunks: int,
                            chunk_size: Optional[int]) -> List[torch.Tensor]:
    """
    Generate chunks of diagonal of inverse Hessian tensor

    :param hessian: Diagonal of inverse Hessian tensor
    :param num_of_tensor_chunks: Number of corresponding tensor chunks
    :param chunk_size: Chunk size
    :return: Hessian tensor chunks
    """
    if chunk_size is None:
        return [hessian]

    if hessian.ndim > 1:
        return torch.split(hessian, chunk_size, dim=1)

    return [hessian] * num_of_tensor_chunks


def prepare_tensor_and_hessian_chunks(tensor: torch.Tensor,
                                      hessian: torch.Tensor,
                                      chunk_size: Optional[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Use chunking for better memory management and return tensor and hessian chunks

    :param tensor: Tensor corresponding to diagonal of inverse Hessian tensor
    :param hessian: Diagonal of inverse Hessian tensor
    :param chunk_size: Chunk size
    :return: Tuple of tensor chunks and hessian chunks
    """
    tensor_chunks = generate_tensor_chunks(tensor, chunk_size)
    hessian_chunks = generate_hessian_chunks(hessian, len(tensor_chunks), chunk_size)

    return tensor_chunks, hessian_chunks


def get_assignments(tensor: torch.Tensor,
                    centroids: torch.Tensor,
                    inverse_hessian_diagonal: Optional[torch.Tensor] = None,
                    chunk_size: Optional[int] = None) -> torch.Tensor:
    """
    Calculate nearest centroid index tensor

    :param tensor: num_blocks_per_column x N x vector_dim
    :param centroids: num_blocks_per_column x num_centroids x vector_dim
    :param inverse_hessian_diagonal: Diagonal of inverse Hessian tensor
    :param chunk_size: Chunk size for better memory management
    :return: nearest centroid index tensor
    """
    manipulated_hessian = manipulate_inverse_hessian_diagonal(tensor, inverse_hessian_diagonal)
    tensor_chunks, hessian_chunks = prepare_tensor_and_hessian_chunks(tensor, manipulated_hessian, chunk_size)

    centroids = centroids.unsqueeze(1) # num_blocks_per_column x 1 x num_centroids x vector_dim
    assignments = []
    for tensor_chunk, hessian_chunk in zip(tensor_chunks, hessian_chunks):
        tensor_chunk = tensor_chunk.unsqueeze(2) # num_blocks_per_column x N x 1 x vector_dim
        distance = ((tensor_chunk - centroids).pow(2) * hessian_chunk).sum(-1)
        assignments.append(distance.argmin(-1))

    return torch.concat(assignments, dim=1)  # num_blocks_per_column x N


def do_kmeans_maximization(tensor: torch.Tensor,
                           centroids: torch.Tensor,
                           assignments: torch.Tensor,
                           inverse_hessian_diagonal: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Do K-means maximization step

    :param tensor: torch.Tensor (num_blocks_per_column x N x vector_dim)
    :param centroids: Codebook including centroids (num_blocks_per_column x num_centroids x vector_dim)
    :param assignments: Assignment result from expectation step (num_blocks_per_column x N)
    :param inverse_hessian_diagonal: Diagonal of inverse Hessian (1 x N x vector_dim)
    :return: Updated codebook after maximization step
    """
    centroid_range = torch.arange(centroids.shape[1], device=centroids.device)
    expanded_assignments = (
        assignments.unsqueeze(-1) == centroid_range.view(1, 1, -1)
    ).to(tensor.dtype)

    if inverse_hessian_diagonal is None:
        norm = 1.0 / torch.clip(expanded_assignments.sum(1), min=1)
        new_centroids = torch.einsum(
            "gnd,gnk,gk->gkd", tensor, expanded_assignments, norm
        )
    else:
        norm = 1.0 / torch.clip(
            torch.einsum(
                "gnk,nd->gkd", expanded_assignments, inverse_hessian_diagonal[0]
            ),
            min=1e-10,
        )
        new_centroids = torch.einsum(
            "gnd,nd,gnk,gkd->gkd",
            tensor,
            inverse_hessian_diagonal[0],
            expanded_assignments,
            norm,
        )

    return new_centroids


def quad_loss_2(
        weight_tensor: torch.Tensor,
        quantized_weight_tensor: torch.Tensor,
        hessian_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Compute quad loss

    :param weight_tensor: Weight tensor
    :param quantized_weight_tensor: Quantized weight tensor
    :param hessian_tensor: Hessian tensor
    :return: Quad loss
    """
    weight_error = weight_tensor - quantized_weight_tensor
    return (weight_error.mm(hessian_tensor) * weight_error).sum()


# pylint: disable=too-many-locals
def fine_tune_codebook(
        original_weight: torch.Tensor,
        original_hessian: torch.Tensor,
        codebooks: List,
        original_assignments: List,
        gptvq_params: GPTVQParameters,
        quantizer: nn.Module,
):
    """
    Update codebook by fine-tuning

    :param original_weight: Original weight tensor of target module
    :param original_hessian: Original Hessian tensor of target module
    :param codebooks: List of codebooks
    :param original_assignments: List of corresponding assignments
    :param gptvq_params: GPTVQ parameters
    :param quantizer: Weight quantizer
    :return: Fine-tuned weight tensor
    """
    with torch.enable_grad():
        weight_tensor = original_weight.float()

        if len(weight_tensor.shape) > 2:
            weight_tensor = weight_tensor.flatten(1)

        rows, cols = weight_tensor.shape
        weight_tensor = (
            weight_tensor.reshape(rows, -1, gptvq_params.vector_stride)
            .transpose(1, 2)
            .reshape(rows, cols)
        )
        all_centroids = codebooks
        all_assignments = original_assignments

        def make_quantized_weight(centroids, assignments):
            all_values = []
            for c, a in zip(centroids, assignments):
                for a_ in a:
                    values = torch.gather(
                        c, dim=1, index=a_.unsqueeze(-1).expand(-1, -1, gptvq_params.vector_dim)
                    )
                    all_values.append(values.view(weight_tensor.shape[0], -1))
            Q = torch.concat(all_values, dim=1)
            return Q

        with torch.no_grad():
            Q = make_quantized_weight(all_centroids, all_assignments)
            orig_loss = quad_loss_2(weight_tensor, Q, original_hessian)
            if orig_loss.item() == 0:
                return Q

        must_restart = True
        lr = 1e-3
        while must_restart:
            orig_centroids = [c.data.clone() for c in all_centroids]
            for c in all_centroids:
                c.requires_grad_()

            param_list = list(all_centroids)
            o = torch.optim.Adam(param_list, lr=lr)
            for _ in range(25):
                must_restart = False
                o.zero_grad()
                Q = make_quantized_weight(all_centroids, all_assignments)
                loss = quad_loss_2(weight_tensor, Q, original_hessian)
                if loss > orig_loss or torch.isnan(loss):
                    lr *= 1e-1
                    must_restart = True
                    all_centroids = orig_centroids
                    break
                loss.backward()
                o.step()

            if not must_restart:
                new_all_centroids = [
                    quantize_dequantize_codebook(
                        c.requires_grad_(False),
                        quantizer,
                        rows // gptvq_params.rows_per_block,
                    )
                    for c in all_centroids
                ]
                Q = make_quantized_weight(new_all_centroids, all_assignments)
                loss = quad_loss_2(weight_tensor, Q, original_hessian)
                if torch.isnan(loss):
                    lr *= 1e-1
                    must_restart = True
                    all_centroids = orig_centroids
                    continue

    return Q


def quantize_dequantize_codebook(codebook: torch.Tensor,
                                 quantizer: nn.Module,
                                 num_blocks_per_column: int) -> torch.Tensor:
    """
    Quantize-Dequantize codebook

    :param codebook: Codebook
    :param quantizer: Quantizer
    :param num_blocks_per_column: Number of blocks per column
    :return: Quantize-Dequantized codebook
    """
    qdq_codebook = Q.affine.quantize_dequantize(
        codebook.reshape(num_blocks_per_column, -1),
        quantizer.get_scale(),
        quantizer.get_offset(),
        quantizer.bitwidth,
        quantizer.symmetric,
    )
    return qdq_codebook.reshape(codebook.shape)


def get_module_name_to_hessian_tensor(gptvq_params: GPTVQParameters,
                                      sim: QuantizationSimModel,
                                      module_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    Get module name to Hessian tensor dictionary

    :param gptvq_params: Data carrier holding GPTVQ parameters
    :param sim: QuantizationSimModel object
    :param module_names: Topologically ordered module names
    :return: Module name to Hessian tensor dictionary
    """
    name_to_hessian = {}
    for module_name in module_names:
        quant_module = sim.model.get_submodule(module_name)
        _, num_cols = get_2d_tensor_shape(quant_module)
        device = quant_module.weight.device
        name_to_hessian[module_name] = torch.zeros((num_cols, num_cols), device=device)

    act_sampler = ActivationSampler(sim.model, gptvq_params.forward_fn, module_names)
    n_samples = 0
    for current_data in gptvq_params.data_loader:
        module_name_to_input_tensor = act_sampler.sample_activation_tensors(current_data)

        curr_batch_size = 0
        for name, inp_data in module_name_to_input_tensor.items():
            if len(inp_data.shape) == 2:
                inp_data = inp_data.unsqueeze(0)
            curr_batch_size = inp_data.shape[0]

            quant_module = sim.model.get_submodule(name)
            update_hessian(quant_module, inp_data, n_samples, curr_batch_size, name_to_hessian[name])

        n_samples += curr_batch_size

    return name_to_hessian


def get_2d_tensor_shape(quant_module: BaseQuantizationMixin) -> torch.Size:
    """
    Compute num of columns from quantization module

    :param quant_module: Quantization module
    :return: Num of columns
    """
    if isinstance(quant_module, nn.Linear):
        tensor_shape = quant_module.weight.shape
    elif isinstance(quant_module, nn.Conv2d):
        tensor_shape = quant_module.weight.flatten(1).shape
    else:
        raise ValueError(f"Unsupported module type {type(quant_module)}")

    return tensor_shape

def update_hessian(
    quant_module: BaseQuantizationMixin,
    inp: torch.Tensor,
    n_samples: int,
    curr_batch_size: int,
    hessian: torch.Tensor,
):
    """
    Updates the hessian matrix using the passed input data to the module and applies scaling

    :param quant_module: Quantization module
    :param inp: activation input passed to the given module
    :param hessian: hessian for the module used to do weight update
    :param n_samples: samples seen so far for hessian computation
    :param curr_batch_size: batch size of current input
    """
    # The hessian is of the shape [weight.shape[1], weight.shape[1]], i.e C*C columns
    # THIS SHOULD WORK FOR ALL THE DIMENSIONS, it makes the last dimension match the weight's column dimension, and first one as the reshaped/ adjusted sample size
    if isinstance(quant_module, nn.Linear):
        if len(inp.shape) >= 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        ## we calculate the transpose of input to compute the Hessian in accordance with the weight shape
        inp = inp.T
    elif isinstance(quant_module, nn.Conv2d):
        unfold = nn.Unfold(
            quant_module.kernel_size,
            dilation=quant_module.dilation,
            padding=quant_module.padding,
            stride=quant_module.stride,
        )
        inp = unfold(inp)
        inp = inp.permute([1, 0, 2])
        inp = inp.flatten(1)
    else:
        raise ValueError(f"Unsupported module type {type(quant_module)}")

    # scale the hessian matrix in place
    hessian *= n_samples / (n_samples + curr_batch_size)
    inp = math.sqrt(2 / (n_samples + curr_batch_size)) * inp.float()
    # update the hessian in place
    hessian += inp.matmul(inp.T)

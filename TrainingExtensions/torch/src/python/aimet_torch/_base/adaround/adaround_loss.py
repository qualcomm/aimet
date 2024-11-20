# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Loss function for Adaround """

from typing import Tuple
import numpy as np
import torch

# Import AIMET specific modules
from aimet_common.defs import AdaroundConstants


class AdaroundHyperParameters:
    """
    Hyper parameters for Adaround
    """
    def __init__(self, num_iterations: int, reg_param: float, beta_range: Tuple, warm_start: float):
        """
        :param num_iterations: Number of maximum iterations to adaround layer
        :param reg_param: Regularization parameter, trading off between rounding loss vs reconstruction loss
        :param beta_range: Start and stop parameters for annealing of rounding loss (start_beta, end_beta)
        :param warm_start: Warm up period, during which rounding loss has zero effect
        """
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.beta_range = beta_range
        self.warm_start = warm_start


class AdaroundLoss:
    """
    Calculates the Reconstruction loss and Rounding loss - needed for Adaround optimization to
    learn weight rounding
    """
    @staticmethod
    def compute_recon_loss(ada_quantized_output: torch.Tensor, orig_output: torch.Tensor) -> torch.Tensor:
        """
        Compute Reconstruction Loss using Squared Frobenius Norm - first part of Combined Loss
        :param ada_quantized_output: Activation output from quantized wrapper module
        :param orig_output: Activation output from original module
        :return: reconstruction loss
        """
        recon_loss = (torch.norm(ada_quantized_output - orig_output, p="fro", dim=1) ** 2).mean()

        return recon_loss

    @classmethod
    def compute_round_loss(cls, alpha: torch.Tensor, opt_params: AdaroundHyperParameters,
                           cur_iter: int) -> torch.Tensor:
        """
        Compute Rounding Loss - second part of Combined Loss
        :param alpha: parameter 'alpha' to be optimized, float32 tensor same shape as weight tensor
        :param opt_params: Optimization parameters for Adaround
        :param cur_iter: current iteration
        :return: rounding loss
        """
        if cur_iter < opt_params.num_iterations * opt_params.warm_start:
            # Warm Start duration
            round_loss = 0

        else:
            # compute rectified sigmoid of parameter 'alpha' which maps it between zero and one
            h_alpha = torch.clamp(torch.sigmoid(alpha) * (AdaroundConstants.ZETA - AdaroundConstants.GAMMA) +
                                  AdaroundConstants.GAMMA, 0, 1)

            # compute beta parameter to anneal the rounding loss
            beta = cls._compute_beta(opt_params.num_iterations, cur_iter, opt_params.beta_range, opt_params.warm_start)

            # calculate regularization term - which ensures parameter to converge to exactly zeros and ones
            # at the end of optimization
            reg_term = torch.add(1, -(torch.add(2 * h_alpha, -1).abs()).pow(beta)).sum()

            # calculate the rounding loss
            round_loss = opt_params.reg_param * reg_term

        return round_loss

    @staticmethod
    def _compute_beta(max_iter: int, cur_iter: int, beta_range: Tuple, warm_start: float) -> float:
        """
        Compute beta parameter used in regularization function using cosine decay
        :param max_iter: total maximum number of iterations
        :param cur_iter: current iteration
        :param beta_range: range for beta decay (start_beta, end_beta)
        :param warm_start: warm up period, during which rounding loss has zero effect
        :return: parameter beta
        """
        assert cur_iter < max_iter, 'Current iteration should be less than total maximum number of iterations.'

        #  Start and stop beta for annealing of rounding loss (start_beta, end_beta)
        start_beta, end_beta = beta_range

        # iteration at end of warm start period, which is 20% of max iterations
        warm_start_end_iter = warm_start * max_iter

        # compute relative iteration of current iteration
        rel_iter = (cur_iter - warm_start_end_iter) / (max_iter - warm_start_end_iter)
        beta = end_beta + 0.5 * (start_beta - end_beta) * (1 + np.cos(rel_iter * np.pi))

        return beta

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

""" Test AdaroundLoss """

import numpy as np
import unittest.mock

import torch
import torch.nn.functional as functional
from aimet_torch._base.adaround.adaround_loss import AdaroundLoss, AdaroundHyperParameters

class TestAdaroundLoss(unittest.TestCase):
    """ Test AdaroundLoss """

    def test_compute_recon_loss(self):
        """ test compute reconstruction loss using dummy input and target tensors """
        np.random.seed(0)
        inp = np.random.rand(32, 3, 12, 12)
        target = np.random.rand(32, 3, 12, 12)
        inp_tensor = torch.from_numpy(inp)
        target_tensor = torch.from_numpy(target)

        recons_loss = AdaroundLoss.compute_recon_loss(inp_tensor, target_tensor)
        print(recons_loss.item())
        self.assertAlmostEqual(recons_loss.item(),
                               functional.mse_loss(inp_tensor, target_tensor, reduction='none').sum(1).mean().item(),
                               places=5)

        # Linear layer
        inp = np.random.rand(32, 10)
        target = np.random.rand(32, 10)
        inp_tensor = torch.from_numpy(inp)
        target_tensor = torch.from_numpy(target)

        recons_loss = AdaroundLoss.compute_recon_loss(inp_tensor, target_tensor)
        print(recons_loss.item())
        self.assertAlmostEqual(recons_loss.item(),
                               functional.mse_loss(inp_tensor, target_tensor, reduction='none').sum(1).mean().item(),
                               places=5)

    def test_compute_round_loss(self):
        """ test compute rounding loss """
        np.random.seed(0)
        alpha = np.random.rand(1, 3, 12, 12)
        alpha_tensor = torch.from_numpy(alpha)

        # Since warm start is 0.2 (20%), cut iter < 2000 (20% of 10000 iterations) will have rounding loss = 0
        opt_params = AdaroundHyperParameters(num_iterations=10000, reg_param=0.01, beta_range=(20, 2),
                                             warm_start=0.2)
        cur_iter = 10
        round_loss_1 = AdaroundLoss.compute_round_loss(alpha_tensor, opt_params, cur_iter)
        self.assertEqual(round_loss_1, 0)

        cur_iter = 8000
        round_loss_2 = AdaroundLoss.compute_round_loss(alpha_tensor, opt_params, cur_iter)
        self.assertAlmostEqual(round_loss_2.item(), 4.266156963161077, places=5)

    def test_compute_beta(self):
        """ test compute beta """
        num_iterations = 10000
        cur_iter = 8000
        beta_range = (20, 2)
        warm_start = 0.2
        self.assertEqual(AdaroundLoss._compute_beta(num_iterations, cur_iter, beta_range, warm_start),
                         4.636038969321072)

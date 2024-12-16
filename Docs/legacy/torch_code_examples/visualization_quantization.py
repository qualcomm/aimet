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
""" Code examples for visualization APIs """
# Quantization visualization imports
import copy
import torch

from torchvision import models

from aimet_torch.cross_layer_equalization import equalize_model

from aimet_torch import batch_norm_fold
from aimet_torch import visualize_model
# End of import statements


def visualize_changes_in_model_after_and_before_cle():
    """
    Code example for visualizating model before and after Cross Layer Equalization optimization
    """
    model = models.resnet18(pretrained=True).to(torch.device('cpu'))
    model = model.eval()
    # Create a copy of the model to visualize the before and after optimization changes
    model_copy = copy.deepcopy(model)

    # Specify a folder in which the plots will be saved
    results_dir = './visualization'

    batch_norm_fold.fold_all_batch_norms(model_copy, (1, 3, 224, 224))

    equalize_model(model, (1, 3, 224, 224))
    visualize_model.visualize_changes_after_optimization(model_copy, model, results_dir)


def visualize_weight_ranges_model():
    """
    Code example for model visualization
    """
    model = models.resnet18(pretrained=True).to(torch.device('cpu'))
    model = model.eval()

    # Specify a folder in which the plots will be saved
    results_dir = './visualization'

    batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))

    # Usually it is observed that if we do BatchNorm fold the layer's weight range increases.
    # This helps in visualizing layer's weight
    visualize_model.visualize_weight_ranges(model, results_dir)


def visualize_relative_weight_ranges_model():
    """
    Code example for model visualization
    """
    model = models.resnet18(pretrained=True).to(torch.device('cpu'))
    model = model.eval()

    # Specify a folder in which the plots will be saved
    results_dir = './visualization'

    batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))

    # Usually it is observed that if we do BatchNorm fold the layer's weight range increases.
    # This helps in finding layers which can be equalized to get better performance on hardware
    visualize_model.visualize_relative_weight_ranges_to_identify_problematic_layers(model, results_dir)


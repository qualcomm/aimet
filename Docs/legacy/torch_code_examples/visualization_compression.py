# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2017-2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

# Visualization imports
from decimal import Decimal
import torch
from torchvision import models
import aimet_common.defs
import aimet_torch.defs
import aimet_torch.utils
from aimet_common.utils import start_bokeh_server_session
from aimet_torch.compress import ModelCompressor
from aimet_torch.visualize_serialized_data import VisualizeCompression

# End of import statements


def model_compression_with_visualization(eval_func):
    """
    Code example for compressing a model with a visualization url provided.
    """
    process = None
    try:
        visualization_url, process = start_bokeh_server_session()

        input_shape = (1, 3, 224, 224)
        model = models.resnet18(pretrained=True).to(torch.device('cuda'))

        modules_to_ignore = [model.conv1]

        greedy_params = aimet_common.defs.GreedySelectionParameters(target_comp_ratio=Decimal(0.65),
                                                                    num_comp_ratio_candidates=10,
                                                                    saved_eval_scores_dict=
                                                                    '../data/resnet18_eval_scores.pkl')

        auto_params = aimet_torch.defs.SpatialSvdParameters.AutoModeParams(greedy_params,
                                                                           modules_to_ignore=modules_to_ignore)

        params = aimet_torch.defs.SpatialSvdParameters(aimet_torch.defs.SpatialSvdParameters.Mode.auto, auto_params,
                                                       multiplicity=8)

        # If no visualization URL is provided, during model compression execution no visualizations will be published.
        ModelCompressor.compress_model(model=model, eval_callback=eval_func, eval_iterations=5,
                                       input_shape=input_shape,
                                       compress_scheme=aimet_common.defs.CompressionScheme.spatial_svd,
                                       cost_metric=aimet_common.defs.CostMetric.mac, parameters=params,
                                       visualization_url=None)

        comp_ratios_file_path = './data/greedy_selection_comp_ratios_list.pkl'
        eval_scores_path = '../data/resnet18_eval_scores.pkl'

        # A user can visualize the eval scores dictionary and optimal compression ratios by executing the following code.
        compression_visualizations = VisualizeCompression(visualization_url)
        compression_visualizations.display_eval_scores(eval_scores_path)
        compression_visualizations.display_comp_ratio_plot(comp_ratios_file_path)
    finally:
        if process:
            process.terminate()
            process.join()

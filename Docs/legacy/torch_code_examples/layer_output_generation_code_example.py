# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

""" Code example to generate intermediate layer outputs of a model """

# Step 0. Import statements
import torch

from aimet_torch.v1.quantsim import QuantizationSimModel, load_encodings_to_sim

from aimet_torch.layer_output_utils import LayerOutputUtil, NamingScheme
from aimet_torch.onnx_utils import OnnxExportApiArgs
# End step 0


# Step 1. Obtain original or quantsim model
# Load the model on CPU device. Ensure model definition is present in the PYTHONPATH to successfully load the model.
# If exported on CPU, load this way.
model = torch.load('path/to/aimet_export_artifacts/model.pth')
# Or
# If exported on GPU, load this way.
# model = torch.load('path/to/aimet_export_artifacts/model.pth', map_location=torch.device('cpu'))

dummy_input = torch.rand(1, 3, 224, 224)

# Use same arguments as that were used for the exported QuantSim model. For sake of simplicity only mandatory arguments are passed below.
quantsim = QuantizationSimModel(model=model, dummy_input=dummy_input)

# Load exported encodings into quantsim object
load_encodings_to_sim(quantsim, 'path/to/aimet_export_artifacts/model_torch.encodings')

# Check whether constructed original and quantsim model are running properly before using Layer Output Generation API.
_ = model(dummy_input)
_ = quantsim.model(dummy_input)
# End step 1


# Step 2. Obtain pre-processed inputs
# Use same input pre-processing pipeline as was used for computing the quantization encodings.
input_batches = get_pre_processed_inputs()
# End step 2


# Step 3. Generate outputs
# Use original model to get fp32 layer-outputs
fp32_layer_output_util = LayerOutputUtil(model=model, dir_path='./fp32_layer_outputs', naming_scheme=NamingScheme.ONNX,
                                         dummy_input=dummy_input, onnx_export_args=OnnxExportApiArgs())
# Use quantsim model to get quantsim layer-outputs
quantsim_layer_output_util = LayerOutputUtil(model=quantsim.model, dir_path='./quantsim_layer_outputs', naming_scheme=NamingScheme.ONNX,
                                             dummy_input=dummy_input, onnx_export_args=OnnxExportApiArgs())
for input_batch in input_batches:
    fp32_layer_output_util.generate_layer_outputs(input_batch)
    quantsim_layer_output_util.generate_layer_outputs(input_batch)
# End step 3

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

""" Code example for SparseConv3D """

# Step 0. Import statements
import torch
import spconv.pytorch as spconv
import aimet_torch
from aimet_torch.v1.quantsim import QuantizationSimModel, QuantScheme
from aimet_torch.pro.model_preparer import prepare_model
# End step 0

import tempfile

# Step 1. Create or load model with SpConv3D module(s)
class SpConvModel(torch.nn.Module):
    def __init__(self):
        super(SpConvModel, self).__init__()

        # "SparseTensorWrapper" needs to be used to convert a dense tensor to a sparse tensor
        self.spconv_tensor = aimet_torch._base.nn.modules.custom.SparseTensorWrapper()

        # First SparseConv3D layer
        self.spconv1 = spconv.SparseConv3d(in_channels=3, out_channels=9, kernel_size=2,
                                           bias=False)

        # Second SparseConv3D layer
        self.spconv2 = spconv.SparseConv3d(in_channels=9, out_channels=5, kernel_size=3, bias=False)

        # Normal Conv3D layer
        self.normal_conv3d = torch.nn.Conv3d(in_channels=5, out_channels=3, kernel_size=3, bias=True)

        # "ScatterDense" needs to be used to convert a sparse tensor to a dense tensor
        self.spconv_scatter_dense = aimet_torch._base.nn.modules.custom.ScatterDense()

        # Adding ReLU activation
        self.relu1 = torch.nn.ReLU()

    def forward(self, coords, voxels):
        '''
        Forward function for the test SpConvModel
        :param coords: Dense indices
        :param voxels: Dense features
        :return: SpConvModel output (dense tensor)
        '''

        # Convert dense indices and features to sparse tensor
        sp_tensor = self.spconv_tensor(coords, voxels)

        # Output from SparseConv3D layer 1
        sp_outputs1 = self.spconv1(sp_tensor)

        # Output from SparseConv3D layer 2
        sp_outputs2 = self.spconv2(sp_outputs1)

        # Convert Sparse tensor output to a dense tensor output
        sp_outputs2_dense = self.spconv_scatter_dense(sp_outputs2)

        # Output from Normal Conv3D layer
        sp_outputs = self.normal_conv3d(sp_outputs2_dense)

        # Output from ReLU
        sp_outputs_relu = self.relu1(sp_outputs)

        return sp_outputs_relu
# End Step 1

model = SpConvModel()

# Step 2. Obtain model inputs
dense_tensor_sp_inputs = torch.randn(1, 3, 10, 10, 10) # generate a random NCDHW tensor
dense_tensor_sp_inputs = dense_tensor_sp_inputs.permute(0, 2, 3, 4, 1) # convert NCDHW to NDHWC

# Creating dense indices
indices = torch.stack(torch.meshgrid(torch.arange(dense_tensor_sp_inputs.shape[0]), torch.arange(dense_tensor_sp_inputs.shape[1]),
                                     torch.arange(dense_tensor_sp_inputs.shape[2]), torch.arange(dense_tensor_sp_inputs.shape[3]),
                                     indexing='ij'), dim=-1).reshape(-1, 4).int()

# Creating dense features
features = dense_tensor_sp_inputs.view(-1, dense_tensor_sp_inputs.shape[4])
# End Step 2

# FP32 model inference
with torch.no_grad():
    orig_output = model(indices, features)

with tempfile.TemporaryDirectory() as dir:
    # Step 3. Apply model preparer pro
    prepared_model = prepare_model(model, dummy_input=(indices, features), path=dir,
                                   onnx_export_args=dict(operator_export_type=
                                                         torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                                                         opset_version=16),
                                   converter_args=['--input_dtype', "indices.1", "int32", '--input_dtype',
                                                   "features.1", "float32", '--expand_sparse_op_structure',
                                                   '--preserve_io', 'datatype', 'indices.1'])
    # End Step 3

# Prepared model inference
with torch.no_grad():
    prep_output = prepared_model(indices, features)

# Step 4. Apply QuantSim
qsim = QuantizationSimModel(prepared_model, dummy_input=(indices, features),
                            quant_scheme=QuantScheme.post_training_tf)
# End Step 4

# Dummy forward pass
def dummy_forward_pass(model, inp):
    with torch.no_grad():
        _ = model(*inp)

# Step 5. Compute encodings
qsim.compute_encodings(dummy_forward_pass, (indices, features))
# End Step 5

# Qsim model inference
with torch.no_grad():
    qsim_output = qsim.model(indices, features)

# Qsim model export
with tempfile.TemporaryDirectory() as dir:
    # Step 6. QuantSim export
    qsim.export(dir, "exported_sp_conv_model", dummy_input=(indices, features),
                onnx_export_args=dict(operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK))
    # End Step 6

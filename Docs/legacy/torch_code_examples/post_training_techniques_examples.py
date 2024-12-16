# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: skip-file
""" These are code examples to be used when generating AIMET documentation via Sphinx """

import torch
from torchvision import models

# Cross layer Equalization related imports
from aimet_torch import batch_norm_fold
from aimet_torch import cross_layer_equalization
from aimet_torch.cross_layer_equalization import equalize_model

from aimet_torch import utils

# Bias Correction related imports


def cross_layer_equalization_manual():
    model = models.resnet18(pretrained=True)

    model = model.eval()

    # Batch Norm Fold
    # Create a list of conv/linear and BN layers for folding forward or backward
    layer_list = [(model.conv1, model.bn1),
                  (model.layer1[0].conv1, model.layer1[0].bn1)]

    # Save the corresponding BN layers (needed only for high bias folding)
    bn_dict = {}
    for conv_bn in layer_list:
        bn_dict[conv_bn[0]] = conv_bn[1]

    batch_norm_fold.fold_given_batch_norms(model, layer_list)

    # Replace any ReLU6 layers with ReLU
    utils.replace_modules_of_type1_with_type2(model, torch.nn.ReLU6, torch.nn.ReLU)

    # Cross Layer Scaling
    # Create a list of consecutive conv layers to be equalized
    consecutive_layer_list = [(model.conv1, model.layer1[0].conv1),
                              (model.layer1[0].conv1, model.layer1[0].conv2)]

    scaling_factor_list = cross_layer_equalization.CrossLayerScaling.scale_cls_sets(consecutive_layer_list)

    # High Bias Fold
    # Create a list of consecutive conv layers whose previous layers bias has to be folded to next layers bias
    ClsSetInfo = cross_layer_equalization.ClsSetInfo
    ClsPairInfo = cross_layer_equalization.ClsSetInfo.ClsSetLayerPairInfo
    cls_set_info_list = [ClsSetInfo(ClsPairInfo(model.conv1, model.layer1[0].conv1, scaling_factor_list[0], True)),
                         ClsSetInfo(ClsPairInfo(model.layer1[0].conv1, model.layer1[0].conv2, scaling_factor_list[1], True))]

    cross_layer_equalization.HighBiasFold.bias_fold(cls_set_info_list, bn_dict)


def cross_layer_equalization_auto():
    model = models.resnet18(pretrained=True)

    input_shape = (1, 3, 224, 224)

    model = model.eval()

    # Performs BatchNorm fold, Cross layer scaling and High bias folding
    equalize_model(model, input_shape)


def cross_layer_equalization_depthwise_layers():
    model = MobileNetV2().to(torch.device('cpu'))
    model.eval()
    # Batch Norm Fold
    # Create a list of conv/linear and BN layers for folding forward or backward
    layer_list = [(model.features[0][0], model.features[0][1]),
                  (model.features[1].conv[0], model.features[1].conv[1]),
                  (model.features[1].conv[3], model.features[1].conv[4])]

    # Save the corresponding BN layers (needed only for high bias folding)
    bn_dict = {}
    for conv_bn in layer_list:
        bn_dict[conv_bn[0]] = conv_bn[1]

    batch_norm_fold.fold_given_batch_norms(model, layer_list)

    # Replace any ReLU6 layers with ReLU
    utils.replace_modules_of_type1_with_type2(model, torch.nn.ReLU6, torch.nn.ReLU)

    # Cross Layer Scaling
    # Create a list of consecutive conv layers to be equalized
    consecutive_layer_list = [(model.features[0][0], model.features[1].conv[0], model.features[1].conv[3])]
    scaling_factor_list = cross_layer_equalization.CrossLayerScaling.scale_cls_sets(consecutive_layer_list)

    # High Bias Fold
    # Create a list of consecutive conv layers whose previous layers bias has to be folded to next layers bias
    ClsSetInfo = cross_layer_equalization.ClsSetInfo
    ClsPairInfo = cross_layer_equalization.ClsSetInfo.ClsSetLayerPairInfo
    cls_set_info_list = [ClsSetInfo(ClsPairInfo(model.features[0][0], model.features[1].conv[0], scaling_factor_list[0][0], True)),
                         ClsSetInfo(ClsPairInfo(model.features[1].conv[0], model.features[1].conv[3], scaling_factor_list[0][1], True))]

    cross_layer_equalization.HighBiasFold.bias_fold(cls_set_info_list, bn_dict)


def cross_layer_equalization_auto_step_by_step():
    model = models.resnet18(pretrained=True)

    model = model.eval()
    input_shape = (1, 3, 224, 224)
    # Fold BatchNorm layers
    folded_pairs = batch_norm_fold.fold_all_batch_norms(model, input_shape)
    bn_dict = {}
    for conv_bn in folded_pairs:
        bn_dict[conv_bn[0]] = conv_bn[1]

    # Replace any ReLU6 layers with ReLU
    utils.replace_modules_of_type1_with_type2(model, torch.nn.ReLU6, torch.nn.ReLU)

    # Perform cross-layer scaling on applicable layer sets
    cls_set_info_list = cross_layer_equalization.CrossLayerScaling.scale_model(model, input_shape)

    # Perform high-bias fold
    cross_layer_equalization.HighBiasFold.bias_fold(cls_set_info_list, bn_dict)


def bias_correction_empirical():
    # Load the model empirical
    model = MobileNetV2()
    model.eval()

    # Apply Empirical Bias Correction
    from aimet_torch import bias_correction
    from aimet_torch.v1.quantsim import QuantParams

    params = QuantParams(weight_bw=4, act_bw=4, round_mode="nearest", quant_scheme='tf_enhanced')

    # User action required
    # The following line of code is an example of how to use the ImageNet data's validation data loader.
    # Replace the following line with your own dataset's validation data loader.
    data_loader = ImageNetDataPipeline.get_val_dataloader()

    # Perform Empirical Bias Correction
    bias_correction.correct_bias(model.to(device="cuda"), params, num_quant_samples=1000,
                                 data_loader=data_loader, num_bias_correct_samples=512)

    # End of example empirical


def bias_correction_analytical_and_empirical():

    dataset_size = 2000
    batch_size = 64

    # Load the model analytical_empirical
    model = MobileNetV2()
    model.eval()

    # Find BNs
    module_prop_dict = bias_correction.find_all_conv_bn_with_activation(model, input_shape=(1, 3, 224, 224))

    # Apply Analytical and Empirical Bias Correction
    from aimet_torch import bias_correction
    from aimet_torch.v1.quantsim import QuantParams

    params = QuantParams(weight_bw=4, act_bw=4, round_mode="nearest", quant_scheme='tf_enhanced')

    # User action required
    # The following line of code is an example of how to use the ImageNet data's validation data loader.
    # Replace the following line with your own dataset's validation data loader.
    data_loader = ImageNetDataPipeline.get_val_dataloader()

    # Perform Bias Correction
    bias_correction.correct_bias(model.to(device="cuda"), params, num_quant_samples=1000,
                                 data_loader=data_loader, num_bias_correct_samples=512,
                                 conv_bn_dict=module_prop_dict, perform_only_empirical_bias_corr=False)

    # End of example analytical_empirical

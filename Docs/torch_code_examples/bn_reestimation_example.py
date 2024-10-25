# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" PyTorch code example to be used for bn_reestimation API document generation """
import json
import os




default_config_per_channel = {
        "defaults":
            {
                "ops":
                    {
                        "is_output_quantized": "True"
                    },
                "params":
                    {
                        "is_quantized": "True",
                        "is_symmetric": "True"
                    },
                "strict_symmetric": "False",
                "unsigned_symmetric": "True",
                "per_channel_quantization": "True"
            },

        "params":
            {
                "bias":
                    {
                        "is_quantized": "False"
                    }
            },

        "op_type":
            {
                "Squeeze":
                    {
                        "is_output_quantized": "False"
                    },
                "Pad":
                    {
                        "is_output_quantized": "False"
                    },
                "Mean":
                    {
                        "is_output_quantized": "False"
                    }
            },

        "supergroups":
            [
                {
                    "op_list": ["Conv", "Relu"]
                },
                {
                    "op_list": ["Conv", "Clip"]
                },
                {
                    "op_list": ["Conv", "BatchNormalization", "Relu"]
                },
                {
                    "op_list": ["Add", "Relu"]
                },
                {
                    "op_list": ["Gemm", "Relu"]
                }
            ],

        "model_input":
            {
                "is_input_quantized": "True"
            },

        "model_output":
            {}
    }

with open("/tmp/default_config_per_channel.json", "w") as f:
    json.dump(default_config_per_channel, f)


# ##################   PyTorch Example #############################


# PyTorch imports

import torch

import torch.cuda

# End of PyTorch imports

# Load FP32 model and prepare it

def load_fp32_model():

    import torchvision
    from torchvision.models import resnet18
    from aimet_torch.model_preparer import prepare_model

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = resnet18(pretrained=True).to(device)
    model = prepare_model(model)

    return model, use_cuda

    # End of Load FP32 model and prepare it


def pass_calibration_data(sim_model):
    """
    The User of the QuantizationSimModel API is expected to write this function based on their data set.
    This is not a working function and is provided only as a guideline.

    :param sim_model:
    :return:
    """

    # User action required
    # The following line is an example of how to use the ImageNet data's validation data loader.
    # Replace the following line with your own dataset's validation data loader.
    data_loader = ImageNetDataPipeline.get_val_dataloader()

    # User action required
    # For computing the activation encodings, around 1000 unlabelled data samples are required.
    # Edit the following 2 lines based on your batch size.
    # batch_size * max_batch_counter should be 1024
    batch_size = 64
    max_batch_counter = 16

    sim_model.eval()

    current_batch_counter = 0
    with torch.no_grad():
        for input_data, target_data in data_loader:

            inputs_batch = input_data  # labels are ignored
            sim_model(inputs_batch)

            current_batch_counter += 1
            if current_batch_counter == max_batch_counter:
                break


def forward_fn(model, inputs):
        input_data, target_data = inputs
        model(input_data)


def create_quant_sim(model, dummy_input, use_cuda):

    # Create QuantSim
    from aimet_common.defs import QuantScheme
    from aimet_torch.v1.quantsim import QuantizationSimModel

    sim = QuantizationSimModel(model=model,
                               quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                               dummy_input=dummy_input,
                               default_output_bw=8,
                               default_param_bw=8,
                               config_file=config_file_path)

    sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                          forward_pass_callback_args=use_cuda)

    return sim
    # End of Create QuantSim


def perform_qat(sim, use_cuda):

    # Perform QAT

    # User action required
    # The following line of code is an example of how to use an example ImageNetPipeline's train function.
    # Replace the following line with your own pipeline's  train function.
    ImageNetPipeline.train(sim.model, epochs=1, learning_rate=5e-7, learning_rate_schedule=[5, 10], use_cuda=use_cuda)

    # End of Perform QAT


def call_bn_reestimation_apis(quant_sim, use_cuda):

    # Call reestimate_bn_stats
    from aimet_torch.bn_reestimation import reestimate_bn_stats

    # User action required
    # The following line of code is an example of how to use the ImageNet data's training data loader.
    # Replace the following line with your own dataset's training data loader.
    train_loader = ImageNetDataPipeline.get_train_dataloader()

    reestimate_bn_stats(quant_sim.model, train_loader, forward_fn=forward_fn)

    # End of Call reestimate_bn_stats

    # Call fold_all_batch_norms_to_scale

    from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms_to_scale

    fold_all_batch_norms_to_scale(quant_sim)

    # End of Call fold_all_batch_norms_to_scale


def export_the_model(quant_sim, dummy_input):
    os.makedirs('./output/', exist_ok=True)
    dummy_input = dummy_input.cpu()
    quant_sim.export(path='./output/', filename_prefix='resnet18_after_qat_bnreestimation', dummy_input=dummy_input)


def bn_reestimation_example():

    model, use_cuda = load_fp32_model()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dummy_input = torch.rand(1, 3, 224, 224, device=device)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)

    quant_sim = create_quant_sim(model, dummy_input, use_cuda)

    perform_qat(quant_sim, use_cuda)

    call_bn_reestimation_apis(quant_sim, device)

    export_the_model(quant_sim, dummy_input)








































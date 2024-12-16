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
# pylint: skip-file

""" AdaRound code example to be used for documentation generation. """


# PyTorch imports

import torch
import torch.cuda

# End of PyTorch imports


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


def apply_adaround_example():

    # Load the model
    import torch
    from torchvision import models

    model = models.resnet18(pretrained=True).eval()

    # Prepare the model
    from aimet_torch.model_preparer import prepare_model
    prepared_model = prepare_model(model)

    # Apply AdaRound
    from aimet_common.defs import QuantScheme
    from aimet_torch.v1.quantsim import QuantizationSimModel
    from aimet_torch.v1.adaround.adaround_weight import Adaround, AdaroundParameters

    # User action required
    # The following line of code is an example of how to use the ImageNet data's training data loader.
    # Replace the following line with your own dataset's training data loader.
    data_loader = ImageNetDataPipeline.get_train_dataloader()

    params = AdaroundParameters(data_loader=data_loader, num_batches=4, default_num_iterations=32,
                                default_reg_param=0.01, default_beta_range=(20, 2))

    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape)

    # Returns model with adarounded weights and their corresponding encodings
    adarounded_model = Adaround.apply_adaround(prepared_model, dummy_input, params, path='./',
                                               filename_prefix='resnet18', default_param_bw=4,
                                               default_quant_scheme=QuantScheme.post_training_tf_enhanced,
                                               default_config_file=None)

    # Create Quantization Simulation using the adarounded_model
    sim = QuantizationSimModel(adarounded_model, quant_scheme=quant_scheme, default_param_bw=param_bw,
                               default_output_bw=output_bw, dummy_input=dummy_input)

    # Set and freeze encodings to use same quantization grid and then invoke compute encodings
    sim.set_and_freeze_param_encodings(encoding_path='./resnet18.encodings')


    # Compute encodings
    sim.compute_encodings(pass_calibration_data, forward_pass_callback_args=None)

    # Determine simulated accuracy
    accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda)
    print(accuracy)

    # Export the model
    # Export the model which saves pytorch model without any simulation nodes and saves encodings file for both
    # activations and parameters in JSON format
    sim.export(path='./', filename_prefix='quantized_resnet18', dummy_input=dummy_input.cpu())

    # End of example


if __name__ == '__main__':
    apply_adaround_example()

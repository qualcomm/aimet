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
# pylint: disable=missing-docstring
from aimet_common.defs import QuantScheme
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_tensorflow.keras.adaround_weight import Adaround, AdaroundParameters
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from tensorflow.keras import applications, losses, metrics, preprocessing
from tensorflow.keras.applications import mobilenet_v2

model = applications.MobileNetV2()
print(model.summary())
# End of model

# Set up dataset
BATCH_SIZE = 32
imagenet_dataset = preprocessing.image_dataset_from_directory(
    directory='<your_imagenet_validation_data_path>',
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

imagenet_dataset = imagenet_dataset.map(
    lambda x, y: (mobilenet_v2.preprocess_input(x), y)
)

NUM_CALIBRATION_SAMPLES = 2048
calibration_dataset = imagenet_dataset.take(NUM_CALIBRATION_SAMPLES // BATCH_SIZE)
unlabeled_dataset = calibration_dataset.map(lambda x, _: x)
# End of dataset


# Step 1
def pass_calibration_data(model, _):
    for inputs, _ in calibration_dataset:
        model(inputs)


PARAM_BITWIDTH = 4
ACTIVATION_BITWIDTH = 8
QUANT_SCHEME = QuantScheme.post_training_tf
params = AdaroundParameters(
    data_set=unlabeled_dataset,
    num_batches=NUM_CALIBRATION_SAMPLES // BATCH_SIZE,
    default_num_iterations=1,
)

ada_rounded_model = Adaround.apply_adaround(
    model,
    params,
    path='/tmp',
    filename_prefix='mobilenet_v2',
    default_param_bw=PARAM_BITWIDTH,
    default_quant_scheme=QUANT_SCHEME,
    config_file=get_path_for_per_channel_config(),
)
# End of step 1

# Step 2
sim = QuantizationSimModel(
    ada_rounded_model,
    quant_scheme=QUANT_SCHEME,
    default_param_bw=PARAM_BITWIDTH,
    default_output_bw=ACTIVATION_BITWIDTH,
    config_file=get_path_for_per_channel_config(),
)

# AdaRound optimizes the rounding of weight quantizers only. These values are preserved through set_and_freeze_param_encodings()
sim.set_and_freeze_param_encodings(encoding_path='/tmp/mobilenet_v2.encodings')

# The activation quantizers remain uninitialized and derived through compute_encodings()
sim.compute_encodings(pass_calibration_data, None)
# End of step 2

# Step 3
eval_dataset = imagenet_dataset.skip(NUM_CALIBRATION_SAMPLES // BATCH_SIZE)
sim.model.compile(
    loss=[losses.CategoricalCrossentropy()],
    metrics=[metrics.CategoricalAccuracy()],
)
result = sim.model.evaluate(eval_dataset)
print(result)
# End of step 3

# Step 4
sim.export(path='/tmp', filename_prefix='quantized_mobilenet_v2')
# End of step 4

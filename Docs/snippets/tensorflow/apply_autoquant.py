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
import tensorflow as tf
from aimet_common.defs import QuantizationDataType, QuantScheme
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_tensorflow.keras.adaround_weight import AdaroundParameters
from aimet_tensorflow.keras.auto_quant_v2 import AutoQuantWithAutoMixedPrecision
from tensorflow.keras import applications, losses, metrics, preprocessing
from tensorflow.keras.applications import mobilenet_v2

model = applications.MobileNetV2()
# End of step 1

# Step 2
BATCH_SIZE = 32
imagenet_dataset = preprocessing.image_dataset_from_directory(
    directory='<your_imagenet_validation_data_path>',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

imagenet_dataset = imagenet_dataset.map(
    lambda x, y: (mobilenet_v2.preprocess_input(x), y)
)

NUM_CALIBRATION_SAMPLES = 2048
unlabeled_dataset = imagenet_dataset.take(NUM_CALIBRATION_SAMPLES // BATCH_SIZE).map(
    lambda x, _: x
)
eval_dataset = imagenet_dataset.skip(NUM_CALIBRATION_SAMPLES // BATCH_SIZE)
# End of step 2


# Step 3
def eval_callback(model: tf.keras.Model, _) -> float:
    # Model should be compiled before evaluation
    model.compile(
        loss=losses.CategoricalCrossentropy(), metrics=metrics.CategoricalAccuracy()
    )
    _, acc = model.evaluate(eval_dataset)

    return acc
# End of step 3

# Step 4. Create AutoQuant object
auto_quant = AutoQuantWithAutoMixedPrecision(
    model,
    eval_callback,
    unlabeled_dataset,
    param_bw=4,
    output_bw=8,
    quant_scheme=QuantScheme.post_training_tf,
    config_file=get_path_for_per_channel_config(),
)
# End of step 4

# Step 5. Set AdaRound params
adaround_params = AdaroundParameters(
    unlabeled_dataset, num_batches=NUM_CALIBRATION_SAMPLES // BATCH_SIZE
)
auto_quant.set_adaround_params(adaround_params)
# End of step 5

# Step 6. Set AMP params
W4A8 = (
    (8, QuantizationDataType.int),  # A: int8
    (4, QuantizationDataType.int),  # W: int4
)
W8A8 = (
    (8, QuantizationDataType.int),  # A: int8
    (8, QuantizationDataType.int),  # W: int8
)
auto_quant.set_mixed_precision_params(candidates=[W4A8, W8A8])
# End of step 6

# Step 7. Run AutoQuant
sim, initial_accuracy = auto_quant.run_inference()
model, optimized_accuracy, encoding_path, pareto_front = auto_quant.optimize(
    allowed_accuracy_drop=0.01
)

print(f'- Quantized Accuracy (before optimization): {initial_accuracy:.4f}')
print(f'- Quantized Accuracy (after optimization):  {optimized_accuracy:.4f}')
# End of step 7

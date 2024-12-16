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
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms_to_scale
from aimet_tensorflow.keras.bn_reestimation import reestimate_bn_stats
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from tensorflow.keras import applications, losses, metrics, optimizers, preprocessing
from tensorflow.keras.applications import mobilenet_v2

model = applications.MobileNetV2()

# Set up dataset
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
calibration_dataset = imagenet_dataset.take(NUM_CALIBRATION_SAMPLES // BATCH_SIZE)
eval_dataset = imagenet_dataset.skip(NUM_CALIBRATION_SAMPLES // BATCH_SIZE)
# End of set up


# Step 1
def pass_calibration_data(model, _):
    for inputs, _ in calibration_dataset:
        _ = model(inputs)


sim = QuantizationSimModel(
    model,
    quant_scheme=QuantScheme.training_range_learning_with_tf_init,
    default_param_bw=4,
    default_output_bw=8,
    config_file=get_path_for_per_channel_config(),
)
sim.compute_encodings(pass_calibration_data, None)
# End of step 1

# Step 2
sim.model.compile(
    optimizer=optimizers.SGD(learning_rate=1e-5),
    loss=[losses.CategoricalCrossentropy()],
    metrics=[metrics.CategoricalAccuracy()],
)

sim.model.fit(calibration_dataset, epochs=10)
_, accuracy = sim.model.evaluate(eval_dataset)
print(f'Model accuracy before BN re-estimation: {accuracy:.4f}')
# End of step 2

# Step 3
unlabeled_dataset = calibration_dataset.map(lambda x, _: x)
reestimate_bn_stats(
    sim.model, unlabeled_dataset, bn_num_batches=NUM_CALIBRATION_SAMPLES // BATCH_SIZE
)
_, accuracy = sim.model.evaluate(eval_dataset)
print(f'Model accuracy after BN re-estimation: {accuracy:.4f}')

fold_all_batch_norms_to_scale(sim)
# End of step 3

# Step 4
sim.export(path='/tmp', filename_prefix='quantized_mobilenet_v2')
# End of step 4

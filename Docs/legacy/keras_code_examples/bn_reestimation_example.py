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
""" Keras code example for bn_reestimation """
import json
import os
import tensorflow as tf

from aimet_common.defs import QuantScheme
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.bn_reestimation import reestimate_bn_stats
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms_to_scale


def evaluate(model: tf.keras.Model, forward_pass_callback_args):
    """
    This is intended to be the user-defined model evaluation function. AIMET requires the above signature. So if the
    user's eval function does not match this signature, please create a simple wrapper.
    Use representative dataset that covers diversity in training data to compute optimal encodings.
    :param model: Model to evaluate
    :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
           the user to determine the type of this parameter. E.g. could be simply an integer representing the number
           of data samples to use. Or could be a tuple of parameters or an object representing something more
           complex.
           If set to None, forward_pass_callback will be invoked with no parameters.
    """
    dummy_x = forward_pass_callback_args
    model(dummy_x)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


tf.keras.backend.clear_session()
inputs = tf.keras.Input(shape=(28, 28, 1,), name="inputs")
conv = tf.keras.layers.Conv2D(16, (3, 3), name ='conv1')(inputs)
bn = tf.keras.layers.BatchNormalization(fused=True)(conv)
relu = tf.keras.layers.ReLU()(bn)
pool = tf.keras.layers.MaxPooling2D()(relu)
conv2 = tf.keras.layers.Conv2D(8, (3, 3), name ='conv2')(pool)
flatten = tf.keras.layers.Flatten()(conv2)
dense  = tf.keras.layers.Dense(10)(flatten)
functional_model = tf.keras.Model(inputs=inputs, outputs=dense)

functional_model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

functional_model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

functional_model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test data using `evaluate`
print("Evaluate quantized model (post QAT) on test data")
results =  functional_model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)



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



qsim = QuantizationSimModel(functional_model, quant_scheme=QuantScheme.training_range_learning_with_tf_init, config_file="/tmp/default_config_per_channel.json")
qsim.compute_encodings(evaluate, forward_pass_callback_args=(x_test[0:100]))


qsim.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError())
print("Evaluate quantized model on test data")
results = qsim.model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)


quantized_callback = tf.keras.callbacks.TensorBoard(log_dir="./log/quantized")
history = qsim.model.fit(
        x_train[0:1024], y_train[0:1024], batch_size=32, epochs=1, validation_data=(x_test, y_test),
        callbacks=[quantized_callback]
    )


print("Evaluate quantized model (post QAT) on test data")
results =  qsim.model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)


# preparing dataset start
batch_size = 4
dataset = tf.data.Dataset.from_tensor_slices(x_train[0:100])
dataset = dataset.batch(batch_size=batch_size)
dummy_inputs = x_train[0:4]
# preparing dataset end

# start BatchNorm Re-estimation
reestimate_bn_stats(qsim.model, dataset, 1)
# end BatchNorm Re-estimation
# start BatchNorm fold to scale
fold_all_batch_norms_to_scale(qsim)
# end BatchNorm fold to scale

os.makedirs('./output/', exist_ok=True)
qsim.export(path='./output/', filename_prefix='mnist_after_bn_re_estimation_qat_range_learning')
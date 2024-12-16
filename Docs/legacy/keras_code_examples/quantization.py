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
import numpy as np
import tensorflow as tf

from aimet_tensorflow.keras import quantsim

def evaluate(model: tf.keras.Model, forward_pass_callback_args):
    """
    This is intended to be the user-defined model evaluation function. AIMET requires the above signature. So if the
    user's eval function does not match this signature, please create a simple wrapper.
    Use representative dataset that covers diversity in training data to compute optimal encodings.

    :param model: Model to evaluate
    :param forward_pass_callback_args: These argument(s) are passed to the forward_pass_callback as-is. Up to
           the user to determine the type of this parameter. E.g. could be simply an integer representing the number
           of data samples to use. Or could be a tuple of parameters or an object representing something more complex.
           If set to None, forward_pass_callback will be invoked with no parameters.
    """
    dummy_x, _ = forward_pass_callback_args
    model(dummy_x)

def quantize_model():
    model = tf.keras.applications.resnet50.ResNet50(weights=None, classes=10)
    sim = quantsim.QuantizationSimModel(model)

    # Generate some dummy data
    dummy_x = np.random.randn(10, 224, 224, 3)
    dummy_y = np.random.randint(0, 10, size=(10,))
    dummy_y = tf.keras.utils.to_categorical(dummy_y, num_classes=10)

    # Compute encodings
    sim.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    sim.compute_encodings(evaluate, forward_pass_callback_args=(dummy_x, dummy_y))

    # Do some fine-tuning
    sim.model.fit(x=dummy_x, y=dummy_y, epochs=10)

quantize_model()

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf
import tensorflow_datasets as tfds

import os
from typing import Callable, Any
import pytest

from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme, QuantizationDataType


# Load and preprocess Imagenette
def _load_imagenette(
    preprocess_input: Callable, batch_size: int = 32
) -> tf.data.Dataset:
    # Imagenette to ImageNet label mapping
    imagenette_to_imagenet = {
        0: 0,
        1: 217,
        2: 482,
        3: 491,
        4: 497,
        5: 566,
        6: 569,
        7: 571,
        8: 574,
        9: 701,
    }

    keys = tf.constant(list(imagenette_to_imagenet.keys()), dtype=tf.int64)
    values = tf.constant(list(imagenette_to_imagenet.values()), dtype=tf.int64)
    table_init = tf.lookup.KeyValueTensorInitializer(keys, values)
    label_lookup = tf.lookup.StaticHashTable(table_init, default_value=-1)

    def _preprocess(image, label):
        image = tf.image.resize(image, (224, 224))
        image = preprocess_input(image)
        label = label_lookup.lookup(label)
        return image, label

    ds = tfds.load("imagenette/320px", split="validation", as_supervised=True)
    return ds.map(_preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Accuracy computation
def _compute_top1_accuracy(
    model: tf.keras.Model, dataset: tf.data.Dataset, max_samples_to_consider: int = 1000
):
    correct = 0
    total = 0
    for i, (images, labels) in dataset.enumerate():
        preds = model(images, training=False)
        top1 = tf.math.argmax(preds, axis=-1)
        correct += tf.reduce_sum(tf.cast(tf.equal(top1, labels), tf.int32)).numpy()
        total += images.shape[0]

        if i > max_samples_to_consider:
            break
    return correct / total


@pytest.mark.parametrize("model", ["mobilenet", "resnet"])
@pytest.mark.parametrize("precision", ["w8a8"])
def test_e2e_accuracy(model: str, precision: str):
    if precision == "w8a8":
        activation_bw = 8
        param_bw = 8
    elif precision == "w8a16":
        activation_bw = 16
        param_bw = 8
    else:
        raise RuntimeError(
            f"Unsupported precision {precision} provided for e2e accuracy tests."
        )

    # Select model and preprocessing
    if model == "mobilenet":
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
    elif model == "resnet":
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
        model = tf.keras.applications.ResNet50(weights="imagenet")
    else:
        raise RuntimeError(f"Unsupported model {model} provided for e2e accuracy test.")

    # Load validation data
    val_ds = _load_imagenette(preprocess_input)
    model.trainable = False

    sim = QuantizationSimModel(
        model=model,
        quant_scheme=QuantScheme.post_training_tf,
        default_output_bw=activation_bw,
        default_param_bw=param_bw,
        rounding_mode="nearest",
        config_file="default",
    )

    def forward_pass(model, _):
        for images, _ in val_ds.take(100):
            model(images, training=False)

    sim.compute_encodings(
        forward_pass_callback=forward_pass, forward_pass_callback_args=None
    )

    quant_top1 = _compute_top1_accuracy(sim.model, val_ds)
    fp_top1 = _compute_top1_accuracy(model, val_ds)
    print(f"Quantized Top-1 Accuracy: {quant_top1:.4f}")
    print(f"FP32 Top-1 Accuracy: {fp_top1:.4f}")

    # Accuracy difference must be less than 2.5%
    assert fp_top1 - quant_top1 < 0.025

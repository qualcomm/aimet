# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Unit tests for Adaround Activation Sampler"""

import os
import copy

import numpy as np
import onnx
import onnxruntime as ort

from .models.models_for_tests import simple_relu_model
from .models import models_for_tests
from aimet_onnx.adaround.activation_sampler import (
    ActivationSampler as AdaroundActivationSampler,
)
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import CachedDataset, make_dummy_input, add_hook_to_get_activation
from aimet_onnx.adascale.activation_sampler import (
    ActivationSampler as AdascaleActivationSampler,
)


class TestAdaroundActivationSampler:
    """
    AdaRound Activation Sampler Unit Test Cases
    """

    def test_activation_sampler_conv(self, tmp_path):
        model = simple_relu_model()
        sim = QuantizationSimModel(model)
        activation_sampler = AdaroundActivationSampler(
            "output", "input", sim, model.model
        )
        data_loader = [np.random.rand(1, 3, 32, 32).astype(np.float32)]
        cached_dataset = CachedDataset(data_loader, 1, tmp_path)
        print(cached_dataset)
        all_inp_data, all_out_data = (
            activation_sampler.sample_and_place_all_acts_on_cpu(cached_dataset)
        )

        # NOTE: Since all inputs are positive, the input and output of relu
        #       should be exactly equal to each other
        assert np.equal(all_out_data, all_inp_data).all()
        assert all_inp_data[0].shape == (1, 3, 32, 32)


class TestAdascaleActivationSampler:
    """
    Adascale Activation Sampler Unit Test Cases
    """

    def test_activation_sampler_relu(self, tmp_path):
        model = simple_relu_model()
        model_path = os.path.join(tmp_path, "model.onnx")
        onnx.save(model.model, model_path)
        activation_sampler = AdascaleActivationSampler(
            "output", model_path, ["CPUExecutionProvider"]
        )

        data_loader = [{"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}]
        all_data = activation_sampler.sample_and_place_all_acts_on_cpu(data_loader)
        assert np.equal(all_data, [[list(x.values()) for x in data_loader]]).all()
        assert all_data[0][0].shape == (1, 3, 32, 32)

    def test_sample_model_input(self, tmp_path):
        model = models_for_tests.simple_relu_model().model

        save_path = str(tmp_path / "model.onnx")
        onnx.save(copy.deepcopy(model), save_path, save_as_external_data=True)

        sampler = AdascaleActivationSampler(
            ["input"], save_path, ["CPUExecutionProvider"]
        )

        dummy_input = make_dummy_input(model)
        input_tensor = copy.deepcopy(dummy_input["input"])
        sample = sampler.sample_acts(dummy_input)
        assert np.array_equal(sample[0], input_tensor)

    def test_sample_multiple_tensors(self, tmp_path):
        model = models_for_tests.matmul_add_model()

        save_path = str(tmp_path / "model.onnx")
        onnx.save(copy.deepcopy(model), save_path, save_as_external_data=True)

        activations = ["matmul_1.output", "matmul_2.output"]
        sampler = AdascaleActivationSampler(
            activations, save_path, ["CPUExecutionProvider"]
        )

        dummy_input = make_dummy_input(model)
        sample = sampler.sample_acts(dummy_input)

        for act in activations:
            add_hook_to_get_activation(model, act)

        sess = ort.InferenceSession(model.SerializeToString())
        act_1, act_2 = sess.run(activations, dummy_input)

        assert np.array_equal(sample[0], act_1)
        assert np.array_equal(sample[1], act_2)

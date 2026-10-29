# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import copy
import json
import os
import shutil
import tempfile

import torch
import ml_dtypes
import numpy as np
import onnxruntime as ort
import pytest
from torch.utils.data import Dataset, DataLoader

from aimet_onnx.utils import (
    make_dummy_input,
    OrtInferenceSession,
    _run_session_to_numpy,
)
from .models import models_for_tests
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.layer_output_utils import LayerOutput, LayerOutputUtil
from .models.models_for_tests import (
    build_dummy_model_with_dynamic_input,
    model_with_ignore_ops,
)
from .utils import tmp_dir

# Fetch appropriate execution providers depending on availability
providers = ["CPUExecutionProvider"]
if "CUDAExecutionProvider" in ort.get_available_providers():
    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]


def get_original_model_artifacts():
    model = build_dummy_model_with_dynamic_input()
    output_names = [node.name for node in model.graph.input]
    for node in model.graph.node:
        output_names.extend(node.output)
    input_dict = make_dummy_input(model)
    output_names = [
        name.replace("/", "_").replace(".", "_") for name in output_names
    ]  # sanitization to get valid file names
    return model, output_names, input_dict


def get_quantsim_artifacts():
    model, _, input_dict = get_original_model_artifacts()

    def callback(session, input_dict):
        session.run(None, input_dict)

    quantsim = QuantizationSimModel(
        model=model, dummy_input=input_dict, providers=["CPUExecutionProvider"]
    )
    quantsim.compute_encodings(callback, input_dict)

    output_names = [node.name for node in quantsim.model.model.graph.input]
    for node in quantsim.model.model.graph.node:
        output_names.extend(node.output)
    output_names = [
        name[: -len("_updated")] for name in output_names if name.endswith("_updated")
    ]
    output_names = [
        name.replace("/", "_").replace(".", "_") for name in output_names
    ]  # sanitization to get valid file names

    return quantsim, output_names, input_dict


class TestLayerOutput:
    def test_get_original_model_outputs(self):
        """Test whether outputs are generated for all the layers of an original model"""

        # Get original model artifacts
        model, output_names, input_dict = get_original_model_artifacts()

        temp_dir_path = os.path.dirname(os.path.abspath(__file__))
        temp_dir_path = os.path.join(temp_dir_path, "temp_dir")

        # Obtain layer-outputs of original model
        layer_output = LayerOutput(model, providers, temp_dir_path)
        output_name_to_output_val_dict = layer_output.get_outputs(input_dict)

        # Verify whether all outputs are generated and have sanitized names
        for name in output_names:
            assert name in output_name_to_output_val_dict, (
                "Output not generated: " + name
            )

        # Verify whether captured outputs are correct. This can only be checked for final output of the model
        session = OrtInferenceSession(model, providers, path=temp_dir_path)
        assert np.array_equal(
            session.run(None, input_dict)[0], output_name_to_output_val_dict["output"]
        )

        # Delete temp_dir
        shutil.rmtree(temp_dir_path, ignore_errors=False, onerror=None)

    def test_get_quantsim_model_outputs(self, tmp_dir):
        """Test whether outputs are generated for all the layers of a quantsim model"""

        # Get quantsim artifacts
        quantsim, output_names, input_dict = get_quantsim_artifacts()

        # Obtain layer-outputs of quantsim model
        layer_output = LayerOutput(quantsim.model.model, providers, tmp_dir)
        output_name_to_output_val_dict = layer_output.get_outputs(input_dict)

        # Verify whether all outputs are generated and have sanitized names
        for name in output_names:
            assert name in output_name_to_output_val_dict, (
                "Output not generated: " + name
            )

        # Verify whether captured outputs are correct. This can only be checked for final output of the model
        session = OrtInferenceSession(quantsim.model.model, providers, path=tmp_dir)
        assert np.array_equal(
            session.run(None, input_dict)[0], output_name_to_output_val_dict["output"]
        )
        pass


def get_dataset_artifacts():
    class DummyDataset(Dataset):
        def __init__(self, count):
            Dataset.__init__(self)
            self.random_img = []
            while count:
                self.random_img.append(torch.rand(3, 32, 32))
                count -= 1

        def __len__(self):
            return len(self.random_img)

        def __getitem__(self, idx):
            return self.random_img[idx]

    data_count = 4
    dummy_dataset = DummyDataset(data_count)
    dummy_dataloader = DataLoader(dataset=dummy_dataset, batch_size=2)

    return dummy_dataset, dummy_dataloader, data_count


class TestLayerOutputUtil:
    def test_generate_layer_outputs(self, tmp_dir):
        """Test whether input files and corresponding layer-output files are generated"""

        # Get quantsim artifacts
        quantsim, output_names, input_dict = get_quantsim_artifacts()

        # Get dataset artifacts
        dummy_dataset, dummy_data_loader, data_count = get_dataset_artifacts()

        # Generate layer-outputs
        layer_out_dict_list = []
        layer_output_util = LayerOutputUtil(
            model=quantsim.model.model, dir_path=tmp_dir
        )
        for input_batch in dummy_data_loader:
            for single_input in input_batch:
                layer_output_util.generate_layer_outputs(
                    single_input.unsqueeze(0).numpy()
                )
                layer_out_dict_list.append(
                    copy.deepcopy(
                        layer_output_util.layer_output.get_outputs(
                            {"input": single_input.unsqueeze(0).numpy()}
                        )
                    )
                )

        # Verify number of inputs
        assert data_count == len(os.listdir(os.path.join(tmp_dir, "inputs")))

        # Verify number of layer-output folders
        assert data_count == len(os.listdir(os.path.join(tmp_dir, "outputs")))

        # Verify number of layer-outputs
        saved_layer_outputs = os.listdir(
            os.path.join(tmp_dir, "outputs", "layer_outputs_0")
        )
        saved_layer_outputs = [i[: -len(".raw")] for i in saved_layer_outputs]
        for name in output_names:
            assert name in saved_layer_outputs

        # Ensure generated layer-outputs can be correctly loaded for layer-output comparison
        saved_last_layer_output = np.fromfile(
            os.path.join(tmp_dir, "outputs", "layer_outputs_0", "output.raw"),
            dtype=np.float32,
        ).reshape((1, 10))
        session = OrtInferenceSession(quantsim.model.model, providers, path=tmp_dir)
        input_dict = {
            "input": np.expand_dims(dummy_dataset.__getitem__(0).numpy(), axis=0)
        }
        assert np.array_equal(session.run(None, input_dict)[0], saved_last_layer_output)

        # Ensure the saved outputs are with axis-transform
        for idx, layer_out_dict in enumerate(layer_out_dict_list):
            layer_out = layer_out_dict["conv_output_3"]

            saved_conv_output = np.fromfile(
                os.path.join(
                    tmp_dir,
                    "outputs",
                    f"layer_outputs_{idx}",
                    "conv_output_3.raw",
                ),
                dtype=np.float32,
            ).reshape(layer_out.shape)
            np.testing.assert_array_equal(layer_out, saved_conv_output)

    def test_generate_layeroutput_for_all_ops(self, tmp_dir):
        """Test to check if layer outputs for all ops are being generated on QuantiSim model"""
        model = model_with_ignore_ops(tmp_dir)

        input = np.random.rand(1, 3, 32, 32).astype(np.float32)
        org_path = os.path.join(tmp_dir, "org")
        lop = LayerOutputUtil(model, org_path)
        lop.generate_layer_outputs(input)

        sim = QuantizationSimModel(model)

        sim_path = os.path.join(tmp_dir, "sim")
        lop2 = LayerOutputUtil(sim.model.model, sim_path)
        lop2.generate_layer_outputs(input)

        json_file_name = "layer_output_name_order.json"

        with open(os.path.join(org_path, json_file_name), "r") as f:
            org_op_list = json.load(f)["layer_output_names"]

        with open(os.path.join(sim_path, json_file_name), "r") as f:
            sim_op_list = json.load(f)["layer_output_names"]

        assert len(org_op_list) == len(sim_op_list)
        assert set(org_op_list) == set(sim_op_list)


@pytest.mark.cuda
@pytest.mark.parametrize("io_dtype", [torch.float32, torch.bfloat16])
def test_generate_bfloat16_layer_outputs(tmp_dir, io_dtype):
    """
    Given: Model computing in bfloat16, with float32 or bfloat16 graph inputs/outputs
    When: Generate its layer outputs
    Then: Every layer output is saved with its own dtype's size, bfloat16 ones as 2-byte
          raw values that read back as the model's output
    """
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDA EP not available")
    model = models_for_tests.simple_mlp_model(torch.bfloat16, io_dtype=io_dtype)
    np_dtype = np.float32 if io_dtype == torch.float32 else ml_dtypes.bfloat16
    x = np.random.randn(4, 16).astype(np_dtype)

    LayerOutputUtil(model, tmp_dir).generate_layer_outputs(x)

    outputs_dir = os.path.join(tmp_dir, "outputs", "layer_outputs_0")
    # fc1 output: [4, 32] in bfloat16
    fc1 = os.path.join(outputs_dir, "_fc1_Gemm_output_0.raw")
    assert os.path.getsize(fc1) == 4 * 32 * 2
    # The graph output, in io_dtype, reads back as the model's output
    (expected,) = _run_session_to_numpy(
        OrtInferenceSession(model, providers), {"input": x}
    )
    saved = np.fromfile(os.path.join(outputs_dir, "output.raw"), np_dtype)
    assert np.array_equal(saved, expected.ravel())

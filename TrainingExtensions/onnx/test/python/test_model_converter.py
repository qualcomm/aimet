# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from onnx.utils import extract_model
import onnxruntime as ort

from aimet_onnx.adascale.model_converter import (
    get_pt_block,
    copy_pt_weights_to_onnx,
)

from aimet_onnx.quantsim import QuantizationSimModel
import pytest
import os
import torch
from onnx import numpy_helper
import onnx_ir
import numpy as np
from dataclasses import dataclass
import copy
from aimet_onnx.common.utils import compute_psnr
from aimet_onnx.llm_topology.block_boundaries import (
    get_decoder_block_boundaries,
)
from aimet_onnx.common.utils import AimetLogger
import onnx
from .models.models_for_tests import weight_gemm_model
from .utils import (
    add_genai_tests_path,
    tmp_dir,
    force_random_weight_init,
    random_chunked_dataset,
)

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.AdaScale)

import torch
from torch import nn as nn


def _update_torch_weights(model, set_zeros: bool = False):
    for param in model.parameters():
        if set_zeros:
            param.data.zero_()
        else:
            param.data.fill_(1.0)
            # param.data *= 1.1
            # TODO update wts not to 1 but some random tensor or make a 10% increment.
            # need to pass the value it was set to _check_onnx_weights


def _check_torch_weights(model, are_zeros: bool = False):
    for param in model.parameters():
        if are_zeros:
            assert param.data.equal(torch.zeros_like(param.data))
        else:
            assert param.data.equal(torch.ones_like(param.data))


def _check_onnx_weights(model, layers_to_check: set = None, are_zeros: bool = False):
    for initializer in model.graph.initializers.values():
        if layers_to_check is not None and initializer.name not in layers_to_check:
            continue
        weight_array = initializer.const_value.numpy()
        if are_zeros:
            assert (weight_array == 0.0).all()
        else:
            if not (weight_array == 1.0).all():
                _logger.info("Weight mismatch: %s", initializer.name)
            else:
                _logger.info("Weight Match : %s", initializer.name)
            assert (weight_array == 1.0).all()


@pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
def test_model_round_trip_with_qwen(add_genai_tests_path, tmp_dir):
    from GenAILab.qai_hub_lm.models.generator import Generator
    from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import (
        TorchONNXInterface,
    )
    from GenAILab.bench.onnx.quant_recipes import _prefill_inputs
    from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
    from transformers import AutoConfig

    torch.manual_seed(0)
    small_model = True
    context_length = 32
    sequence_length = 16
    model_id = "Qwen/Qwen2.5-0.5B"
    vocab_size = 1024

    with force_random_weight_init(vocab_size=vocab_size):
        entry = LLM_ONNX.instantiate_float_model(
            model_id, context_length, sequence_length, small_model=small_model
        )
    collection = LLM_ONNX.instantiate_quantsim(entry)
    sim = collection.backbone
    llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if small_model:
        llm_config.num_hidden_layers = 2
    ################ Input for qwen2.5
    tokenizer = LLM_ONNX.instantiate_tokenizer(model_id)

    train_dataset = random_chunked_dataset(vocab_size, context_length)
    quantsim_with_torch_interface = TorchONNXInterface(sim, llm_config)
    generator = Generator(
        quantsim_with_torch_interface, tokenizer, sequence_length, context_length
    )

    inputs = _prefill_inputs(sim, generator, train_dataset, num_iterations=5)

    ################ fp32 onnx model
    CHECKPOINT_DIR = str(os.path.join(tmp_dir, "onnx_checkpoints_debugging"))
    CHECKPOINT_FP_DIR = str(os.path.join(CHECKPOINT_DIR, "fp_model.onnx"))
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # converter = ModelConverter(CHECKPOINT_DIR)
    fp32_model = copy.deepcopy(sim.model.model)
    fp32_model = QuantizationSimModel.remove_quantizers(fp32_model)
    onnx.save_model(
        fp32_model,
        CHECKPOINT_FP_DIR,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="fp_model.data",
    )
    common_inputs = ["attention_mask", "position_ids"]
    adascale_blocks_end_points = get_decoder_block_boundaries(sim.model.model)
    block_inputs = [adascale_blocks_end_points[0][0]]

    model_before_block = os.path.join(CHECKPOINT_DIR, "before_decoder_block.onnx")
    fp_model_path = CHECKPOINT_FP_DIR  # converter._get_onnx_fp_model(fp32_model)
    extract_model(
        fp_model_path, model_before_block, list(inputs[0].keys()), block_inputs
    )
    before_session = ort.InferenceSession(
        model_before_block, providers=["CPUExecutionProvider"]
    )
    block_input_tensor = before_session.run(block_inputs, inputs[0])
    for block_id, (block_start, block_end) in enumerate(
        get_decoder_block_boundaries(sim.model.model)
    ):
        block_inputs = [block_start]
        block_input_names = (
            block_inputs
            + common_inputs
            + [f"past_key_{block_id}_in", f"past_value_{block_id}_in"]
        )
        block_output_names = [block_end]
        block_input_output_names = (block_input_names, block_output_names)
        sim_model = onnx_ir.from_proto(sim.model.model)
        onnx_ir.passes.common.TopologicalSortPass().call(sim_model)
        pt_block, param_map = get_pt_block(sim_model, block_input_output_names)
        ################ run forward pass 1 through onnx block
        block_model_path = os.path.join(CHECKPOINT_DIR, "block_fp32.onnx")
        extract_model(
            fp_model_path, block_model_path, block_input_names, block_output_names
        )
        onnx_fp_block_sess = ort.InferenceSession(
            block_model_path, providers=["CPUExecutionProvider"]
        )
        block_test_inputs = inputs[0].copy()
        block_test_inputs[block_inputs[0]] = block_input_tensor[0]
        for name in inputs[0].keys():
            if name not in block_input_names:
                del block_test_inputs[name]
        onnx_fp_out = onnx_fp_block_sess.run(None, block_test_inputs)
        ################ run forward pass 2 through converted pytorch(assert 1==2)
        torch_out = (
            pt_block(
                torch.from_numpy(block_input_tensor[0]).float(),
                torch.from_numpy(inputs[0]["attention_mask"]).long(),
                torch.from_numpy(inputs[0]["position_ids"]).long(),
                torch.from_numpy(inputs[0][f"past_key_{block_id}_in"]).float(),
                torch.from_numpy(inputs[0][f"past_value_{block_id}_in"]).float(),
            )
            .detach()
            .numpy()
        )
        assert compute_psnr(onnx_fp_out[0], torch_out) == 100

        ################ Update torch weights to 1
        _update_torch_weights(pt_block, set_zeros=False)
        ################ copy_pt_weights_to_onnx copies updated wts to onnx from pt
        copy_pt_weights_to_onnx(pt_block, sim_model, param_map)
        layers_to_check = set(param_map.values())
        ################ check if the onnx wts are updated for `layers of interest`
        _check_onnx_weights(sim_model, layers_to_check, are_zeros=False)


@pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
@pytest.mark.parametrize(
    "dynamo, opset_version",
    [(False, 17), (False, 18), (True, 18), (False, 19), (False, 20), (False, 21)],
)
def test_model_round_trip_with_qwen_dynamo(
    add_genai_tests_path, tmp_dir, dynamo, opset_version
):
    from unittest.mock import patch
    from GenAILab.qai_hub_lm.models.generator import Generator
    from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import (
        TorchONNXInterface,
    )
    from GenAILab.bench.onnx.quant_recipes import _prefill_inputs
    from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
    from transformers import AutoConfig

    small_model = True
    context_length = 32
    sequence_length = 16
    model_id = "Qwen/Qwen2.5-0.5B"
    vocab_size = 1024

    _real_export = torch.onnx.export

    def _patched_export(*args, **kwargs):
        kwargs["dynamo"] = dynamo
        kwargs["opset_version"] = opset_version
        return _real_export(*args, **kwargs)

    with (
        patch(
            "GenAILab.qai_hub_lm.backends.onnx.export_utils.ONNX_OPSET_VERSION",
            opset_version,
        ),
        patch(
            "GenAILab.qai_hub_lm.backends.onnx.export_utils.torch.onnx.export",
            side_effect=_patched_export,
        ),
        force_random_weight_init(vocab_size=vocab_size),
    ):
        entry = LLM_ONNX.instantiate_float_model(
            model_id, context_length, sequence_length, small_model=small_model
        )

    collection = LLM_ONNX.instantiate_quantsim(entry)
    sim = collection.backbone
    llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if small_model:
        llm_config.num_hidden_layers = 2

    tokenizer = LLM_ONNX.instantiate_tokenizer(model_id)
    torch.manual_seed(0)
    train_dataset = random_chunked_dataset(vocab_size, context_length)
    quantsim_with_torch_interface = TorchONNXInterface(sim, llm_config)
    generator = Generator(
        quantsim_with_torch_interface, tokenizer, sequence_length, context_length
    )

    inputs = _prefill_inputs(sim, generator, train_dataset, num_iterations=5)

    CHECKPOINT_DIR = str(os.path.join(tmp_dir, "onnx_checkpoints_debugging"))
    CHECKPOINT_FP_DIR = str(os.path.join(CHECKPOINT_DIR, "fp_model.onnx"))
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    fp32_model = copy.deepcopy(sim.model.model)
    fp32_model = QuantizationSimModel.remove_quantizers(fp32_model)
    onnx.save_model(
        fp32_model,
        CHECKPOINT_FP_DIR,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="fp_model.data",
    )
    common_inputs = ["attention_mask", "position_ids"]
    adascale_blocks_end_points = get_decoder_block_boundaries(sim.model.model)
    block_inputs = [adascale_blocks_end_points[0][0]]

    model_before_block = os.path.join(CHECKPOINT_DIR, "before_decoder_block.onnx")
    fp_model_path = CHECKPOINT_FP_DIR
    extract_model(
        fp_model_path, model_before_block, list(inputs[0].keys()), block_inputs
    )
    before_session = ort.InferenceSession(
        model_before_block, providers=["CPUExecutionProvider"]
    )
    block_input_tensor = before_session.run(block_inputs, inputs[0])
    for block_id, (block_start, block_end) in enumerate(
        get_decoder_block_boundaries(sim.model.model)
    ):
        block_inputs = [block_start]
        block_input_names = (
            block_inputs
            + common_inputs
            + [f"past_key_{block_id}_in", f"past_value_{block_id}_in"]
        )
        block_output_names = [block_end]
        block_input_output_names = (block_input_names, block_output_names)
        sim.model.topological_sort()
        sim_model = onnx_ir.from_proto(sim.model.model)
        pt_block, param_map = get_pt_block(sim_model, block_input_output_names)

        block_model_path = os.path.join(CHECKPOINT_DIR, "block_fp32.onnx")
        extract_model(
            fp_model_path, block_model_path, block_input_names, block_output_names
        )
        onnx_fp_block_sess = ort.InferenceSession(
            block_model_path, providers=["CPUExecutionProvider"]
        )
        block_test_inputs = inputs[0].copy()
        block_test_inputs[block_inputs[0]] = block_input_tensor[0]
        for name in inputs[0].keys():
            if name not in block_input_names:
                del block_test_inputs[name]
        onnx_fp_out = onnx_fp_block_sess.run(None, block_test_inputs)

        torch_out = (
            pt_block(
                torch.from_numpy(block_input_tensor[0]).float(),
                torch.from_numpy(inputs[0]["attention_mask"]).long(),
                torch.from_numpy(inputs[0]["position_ids"]).long(),
                torch.from_numpy(inputs[0][f"past_key_{block_id}_in"]).float(),
                torch.from_numpy(inputs[0][f"past_value_{block_id}_in"]).float(),
            )
            .detach()
            .numpy()
        )
        assert compute_psnr(onnx_fp_out[0], torch_out) == 100

        _update_torch_weights(pt_block, set_zeros=False)
        copy_pt_weights_to_onnx(pt_block, sim_model, param_map)

        layers_to_check = set(param_map.values())
        _check_onnx_weights(sim_model, layers_to_check, are_zeros=False)


class SimpleConvModel(nn.Module):
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x


class ModelWithConvs(torch.nn.Module):
    def __init__(self):
        super(ModelWithConvs, self).__init__()

        self.layer1 = torch.nn.Conv2d(64, 32, (3, 3))
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.layer2 = torch.nn.Conv2d(32, 64, (3, 3))

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)


class ModelWithConsecutiveConvBlocks(torch.nn.Module):
    def __init__(self):
        super(ModelWithConsecutiveConvBlocks, self).__init__()
        self.blocks = torch.nn.ModuleList(ModelWithConvs() for _ in range(2))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for linear_block in self.blocks:
            x = linear_block(x)
        x = self.softmax(x)
        return x


def test_model_with_conv(tmp_dir):
    # Instantiate and export the model
    model = SimpleConvModel()
    dummy_input = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image

    onnx_model_path = os.path.join(tmp_dir, "simple_conv_model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    onnx_model = onnx_ir.load(onnx_model_path)
    pt_block, param_map = get_pt_block(onnx_model, (["input"], ["output"]))
    # forwardpass through onnx == forward pass through pt Block
    onnx_block_model_sess = ort.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"]
    )

    onnx_output = onnx_block_model_sess.run(None, {"input": dummy_input.numpy()})
    torch_out = pt_block(dummy_input)
    diff = onnx_output[0] - torch_out.detach().numpy()
    assert diff.max() <= 0.000001
    assert compute_psnr(onnx_output[0], torch_out.detach().numpy()) == 100
    # update pt wts call copy wts
    _update_torch_weights(pt_block, set_zeros=False)

    copy_pt_weights_to_onnx(pt_block, onnx_model, param_map)
    layers_to_check = set(param_map.values())
    _check_onnx_weights(onnx_model, layers_to_check, are_zeros=False)


@pytest.mark.parametrize("transposed_weight", [False, True])
def test_model_with_gemm(transposed_weight):
    # Gemm stores its weight as [in, out] (transB=0) or [out, in] (transB=1)
    # copy_pt_weights_to_onnx must reload it in the matching orientation.
    onnx_model = onnx_ir.from_proto(
        weight_gemm_model(6, 4, transposed_weight=transposed_weight)
    )
    dummy_input = np.random.randn(1, 6).astype(np.float32)
    pt_block, param_map = get_pt_block(onnx_model, (["input"], ["output"]))
    assert param_map

    onnx_output = ort.InferenceSession(
        onnx_ir.to_proto(onnx_model).SerializeToString()
    ).run(None, {"input": dummy_input})

    # onnx -> torch -> reload back to onnx must preserve the output
    copy_pt_weights_to_onnx(pt_block, onnx_model, param_map)
    reloaded_output = ort.InferenceSession(
        onnx_ir.to_proto(onnx_model).SerializeToString()
    ).run(None, {"input": dummy_input})
    assert compute_psnr(onnx_output[0], reloaded_output[0]) == 100


@pytest.mark.parametrize(
    "input_names, output_names, extracted_graph_inp_shape",
    [
        (
            ["/blocks.0/relu1/Relu_output_0"],
            ["/blocks.1/relu1/Relu_output_0"],
            (1, 32, 126, 126),
        ),
        (
            ["/blocks.1/relu1/Relu_output_0"],
            ["/blocks.1/layer2/Conv_output_0"],
            (1, 32, 122, 122),
        ),
    ],
)
def test_model_with_ModelWithConsecutiveConvBlocks(
    input_names, output_names, extracted_graph_inp_shape, tmp_dir
):
    # Instantiate and export the model
    model = ModelWithConsecutiveConvBlocks()
    dummy_input = torch.randn(
        1, 64, 128, 128
    )  # Batch size 1, 64 channels, 128x128 image
    onnx_model_path = os.path.join(tmp_dir, "simple_conv_model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    onnx_model = onnx.load(onnx_model_path)

    get_onnx_block_model = extract_model(
        onnx_model_path, "extracted.onnx", input_names, output_names
    )
    onnx_model = onnx_ir.from_proto(onnx_model)
    pt_block, param_map = get_pt_block(onnx_model, (input_names, output_names))

    # forwardpass through onnx == forward pass through pt Block
    onnx_block_model_sess = ort.InferenceSession(
        "extracted.onnx", providers=["CPUExecutionProvider"]
    )
    dummy_input_for_extracted_graph = torch.randn(*extracted_graph_inp_shape)
    onnx_output = onnx_block_model_sess.run(
        None, {input_names[0]: dummy_input_for_extracted_graph.numpy()}
    )
    torch_out = pt_block(dummy_input_for_extracted_graph)
    diff = onnx_output[0] - torch_out.detach().numpy()
    assert diff.max() <= 0.00001
    assert compute_psnr(onnx_output[0], torch_out.detach().numpy()) == 100
    # update pt wts call copy wts
    _update_torch_weights(pt_block, set_zeros=False)

    copy_pt_weights_to_onnx(pt_block, onnx_model, param_map)
    layers_to_check = set(param_map.values())
    _check_onnx_weights(onnx_model, layers_to_check, are_zeros=False)


@pytest.mark.skip_on_windows_amd64("Dynamo export fails on amd64")
@pytest.mark.skip_on_windows_arm64("Dynamo export fails on arm64")
@pytest.mark.parametrize(
    "dynamo, opset_version", [(False, 17), (False, 18), (True, 18)]
)
def test_model_with_conv_with_dynamo(tmp_dir, dynamo, opset_version):
    # Instantiate and export the model
    model = SimpleConvModel()
    dummy_input = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image

    onnx_model_path = os.path.join(tmp_dir, "simple_conv_model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
        opset_version=opset_version,
    )
    onnx_model = onnx_ir.load(onnx_model_path)
    pt_block, param_map = get_pt_block(onnx_model, (["input"], ["output"]))
    # forwardpass through onnx == forward pass through pt Block
    onnx_block_model_sess = ort.InferenceSession(
        onnx_model_path, providers=["CPUExecutionProvider"]
    )

    onnx_output = onnx_block_model_sess.run(None, {"input": dummy_input.numpy()})
    torch_out = pt_block(dummy_input)
    diff = onnx_output[0] - torch_out.detach().numpy()
    assert diff.max() <= 0.000001
    assert compute_psnr(onnx_output[0], torch_out.detach().numpy()) == 100
    # update pt wts call copy wts
    _update_torch_weights(pt_block, set_zeros=False)

    copy_pt_weights_to_onnx(pt_block, onnx_model, param_map)
    layers_to_check = set(param_map.values())
    _check_onnx_weights(onnx_model, layers_to_check, are_zeros=False)

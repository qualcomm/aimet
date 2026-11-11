# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# pylint: disable=all
# [setup]
import torch
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from datasets import load_dataset

# General setup that can be changed as needed
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(pretrained=True).eval().to(device)
num_batches = 32
data = load_dataset('imagenet-1k', streaming=True, split="train")
data_loader = DataLoader(data, batch_size=num_batches, num_workers = 4)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# [step_1]
from aimet_torch import QuantizationSimModel
from aimet_torch.common.quantsim_config.utils import get_path_for_per_channel_config

# Use a per-channel quantization config; update it if your model needs a different config.
sim = QuantizationSimModel(model=model, dummy_input=dummy_input, config_file=get_path_for_per_channel_config())
# [step_2]
num_epochs = 20
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        output = sim.model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# [step_3]
from aimet_torch.bn_reestimation import reestimate_bn_stats
from aimet_torch.batch_norm_fold import fold_all_batch_norms_to_scale

reestimate_bn_stats(sim.model, data_loader)
fold_all_batch_norms_to_scale(sim)

# [step_4]
import aimet_torch
aimet_torch.onnx.export(
    sim.model,
    dummy_input,
    "quantized_mobilenet.onnx",
    dynamo=False,
    opset_version=21,
)

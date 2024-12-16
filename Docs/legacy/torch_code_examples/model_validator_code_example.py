# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" These are code examples to be used when generating AIMET documentation via Sphinx """

import torch
import torch.nn.functional as F

# Model Validator related imports
from aimet_torch.model_validator.model_validator import ModelValidator


class ModelWithReusedNodes(torch.nn.Module):
    """ Model that reuses a relu module. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithReusedNodes, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.linear = torch.nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ModelWithoutReusedNodes(torch.nn.Module):
    """ Model that is fixed to not reuse modules. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithoutReusedNodes, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.linear = torch.nn.Linear(2592, 10)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ModelWithFunctionalLinear(torch.nn.Module):
    """ Model that uses a torch functional linear layer. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithFunctionalLinear, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = F.linear(x, torch.randn(10, 2592))
        return x


class ModelWithoutFunctionalLinear(torch.nn.Module):
    """ Model that is fixed to use a linear module instead of functional. Expects input of shape (1, 3, 32, 32) """

    def __init__(self):
        super(ModelWithoutFunctionalLinear, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.linear = torch.nn.Linear(2592, 10)
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(torch.randn(10, 2592))

    def forward(self, *inputs):
        x = self.conv1(inputs[0])
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def validate_example_model():

    # Load the model to validate
    model = ModelWithReusedNodes()

    # Output of ModelValidator.validate_model will be True if model is valid, False otherwise
    ModelValidator.validate_model(model, model_input=torch.rand(1, 3, 32, 32))


if __name__ == '__main__':
    validate_example_model()

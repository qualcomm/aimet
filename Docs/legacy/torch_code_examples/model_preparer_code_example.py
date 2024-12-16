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
# pylint: skip-file

# ModelPreparer imports

import torch
import torch.nn.functional as F
from aimet_torch.model_preparer import prepare_model

# End of import statements


class ModelWithFunctionalReLU(torch.nn.Module):
    """ Model that uses functional ReLU instead of nn.Modules. Expects input of shape (1, 3, 32, 32) """
    def __init__(self):
        super(ModelWithFunctionalReLU, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).relu()
        return x


class ModelWithReusedReLU(torch.nn.Module):
    """ Model that uses single ReLU instances multiple times in the forward. Expects input of shape (1, 3, 32, 32) """
    def __init__(self):
        super(ModelWithReusedReLU, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class ModelWithElementwiseAddOp(torch.nn.Module):
    def __init__(self):
        super(ModelWithElementwiseAddOp, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 6, 5)

    def forward(self, *inputs):
        x1 = self.conv1(inputs[0])
        x2 = self.conv2(inputs[1])
        x = x1 + x2
        return x


def model_preparer_functional_example():

    # Load the model and keep in eval() mode
    model = ModelWithFunctionalReLU().eval()
    input_shape = (1, 3, 32, 32)
    input_tensor = torch.randn(*input_shape)

    # Call to prepare_model API
    prepared_model = prepare_model(model)
    print(prepared_model)

    # Compare the outputs of original and transformed model
    assert torch.allclose(model(input_tensor), prepared_model(input_tensor))


def model_preparer_reused_example():

    # Load the model and keep in eval() mode
    model = ModelWithReusedReLU().eval()
    input_shape = (1, 3, 32, 32)
    input_tensor = torch.randn(*input_shape)

    # Call to prepare_model API
    prepared_model = prepare_model(model)
    print(prepared_model)

    # Compare the outputs of original and transformed model
    assert torch.allclose(model(input_tensor), prepared_model(input_tensor))


def model_preparer_elementwise_add_example():

    # Load the model and keep in eval() mode
    model = ModelWithElementwiseAddOp().eval()
    input_shape = (1, 3, 32, 32)
    input_tensor = [torch.randn(*input_shape), torch.randn(*input_shape)]

    # Call to prepare_model API
    prepared_model = prepare_model(model)
    print(prepared_model)

    # Compare the outputs of original and transformed model
    assert torch.allclose(model(*input_tensor), prepared_model(*input_tensor))


if __name__ == '__main__':
    model_preparer_functional_example()
    model_preparer_reused_example()
    model_preparer_elementwise_add_example()
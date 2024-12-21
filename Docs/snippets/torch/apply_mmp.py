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

# [setup]
import torch
from torchvision.models import mobilenet_v2
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.v2.mixed_precision import MixedPrecisionConfigurator

input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape).cuda()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(pretrained=True).eval().to(device)

# create the sim object. Feel free to change the default settings as you wish
quant_sim = QuantizationSimModel(model,
                                 dummy_input=dummy_input,
                                 default_param_bw=8,
                                 default_output_bw=8,
                                 config_file=get_path_for_per_channel_config())

# create the MMP configurator object
mp_configurator = MixedPrecisionConfigurator(quant_sim)
# [set_precision_leaf]
mp_configurator.set_precision(quant_sim.model.features[1].conv[0][0], activation='int16', param={'weight': 'int16'})
# [set_precision_non_leaf]
mp_configurator.set_precision(quant_sim.model.features[3].conv[1], activation='int16', param={'weight': 'int16'})
# [set_precision_type]
mp_configurator.set_precision(torch.nn.AvgPool2d, activation='int16')
# [set_precision_model_input]
mp_configurator.set_model_input_precision(activation='int16')
# [set_precision_model_output]
mp_configurator.set_model_output_precision(activation='int16')
# [apply]
mp_configurator.apply()

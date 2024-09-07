//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================

#include <DlQuantization/IQuantizer.hpp>
#include <DlQuantization/Quantization.hpp>
#include <DlQuantization/QuantizerFactory.hpp>
#include <DlQuantization/EncodingRescale.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <torch/extension.h>


class AimetEncodingRescaler
{
public:
    AimetEncodingRescaler()
    {

    }

    std::tuple<at::Tensor, at::Tensor> getRescaledOutputAndBias(at::Tensor input, py::list weight_scale, float input_scale,
                                                                DlQuantization::TfEncoding out_enc, bool isOffsetWrap)
    {
        at::Tensor output = input;

        at::IntArrayRef sizes = input.sizes();
        size_t inputTensorSize = 1;

        for (auto size: sizes)
            inputTensorSize *= size;
        
        //scaling_params.numel() will return the total number of elements in the input tensor, 
        //if scaling_params.numel() is zero, we need to create an empty tensor here. 
        //For the torch::TensorOptions(), it just make sure that scaling_params has same 
        //data type and compute device with input tensor.
        if(!scaling_params.numel())
        {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device().type(), 
                                                                                input.device().index());
            scaling_params = torch::empty({weight_scale.size()}, options);
        }
        else
        {
            if((input.device().type() != scaling_params.device().type()) || 
               (input.device().type() == torch::kCUDA && (input.device().index() != scaling_params.device().index())))
            {
                auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device().type(), 
                                                                                    input.device().index());
                scaling_params = torch::empty({weight_scale.size()}, options);
            }
        }    
        DlQuantization::ConvSpecArgs<float> encodingArgs = {.out_encoding_delta = static_cast<float>(out_enc.delta),
                                                              .out_encoding_offset = static_cast<float>(out_enc.offset),
                                                              .input_scale = input_scale,
                                                              .bw = out_enc.bw,
                                                              .weight_scale = weight_scale.cast<std::vector<float>>(),
        };

        DlQuantization::getRescaledOutputAndBias<float>(input.data_ptr<float>(), inputTensorSize, encodingArgs,
                                                 output.data_ptr<float>(), scaling_params.data_ptr<float>(),
                                                 input.is_cuda(), isOffsetWrap);

        return std::make_tuple(scaling_params, output);
    }

private:
    at::Tensor scaling_params;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<AimetEncodingRescaler>(m, "AimetEncodingRescaler")
        .def(pybind11::init())
        .def("getRescaledOutputAndBias", &AimetEncodingRescaler::getRescaledOutputAndBias);
}

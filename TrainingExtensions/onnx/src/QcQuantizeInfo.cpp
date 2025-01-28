//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include "QcQuantizeInfo.h"
#include "DlQuantization/Quantization.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(libquant_info, m)
{
    pybind11::class_<QcQuantizeInfo>(m, "QcQuantizeInfo")
        .def(py::init<>())
        .def_readwrite("tensorQuantizerRef", &QcQuantizeInfo::tensorQuantizer)
        .def_property("encoding", &QcQuantizeInfo::getEncodings, &QcQuantizeInfo::setEncodings)
        .def_readwrite("opMode", &QcQuantizeInfo::opMode)
        .def_readwrite("name", &QcQuantizeInfo::name)
        .def_readwrite("enabled", &QcQuantizeInfo::enabled)
        .def_readwrite("useSymmetricEncoding", &QcQuantizeInfo::useSymmetricEncoding)
        .def_readwrite("usePerChannelMode", &QcQuantizeInfo::usePerChannelMode)
        .def_readwrite("isIntDataType", &QcQuantizeInfo::isIntDataType)
        .def_readwrite("channelAxis", &QcQuantizeInfo::channelAxis)
        .def_readwrite("blockSize", &QcQuantizeInfo::blockSize)
        .def_readwrite("blockAxis", &QcQuantizeInfo::blockAxis);
}

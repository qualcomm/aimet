#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2016-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
#==============================================================================

add_library(MoDlQuantization STATIC

      src/IQuantizationAlgorithm.hpp
      src/MainQuantizationClass.cpp
      src/MainQuantizationClass.hpp
      src/spec_functions.hpp
      src/spec_functions.cpp
      src/math_functions.cpp
      src/math_functions.hpp
      src/quantization_utils.cpp
      src/quantization_utils.hpp
      src/QuantizerFactory.cpp
      src/TfEnhancedQuantizer.cpp
      src/TfEnhancedQuantizer.hpp
      src/TfQuantizer.cpp
      src/TfQuantizer.hpp
      src/trim_functions.cpp
      src/trim_functions.hpp

      src/EntropyEncodingAnalyzer.cpp
      src/EntropyEncodingAnalyzer.h
      src/MseEncodingAnalyzer.cpp
      src/MseEncodingAnalyzer.h
      src/PercentileEncodingAnalyzer.cpp
      src/PercentileEncodingAnalyzer.h
      src/TfEnhancedEncodingAnalyzer.cpp
      src/TfEnhancedEncodingAnalyzer.h
      src/TfEncodingAnalyzer.h
      src/TfEncodingAnalyzer.cpp
      src/TensorQuantizationSim.h
      src/TensorQuantizationSim.cpp
      src/EncodingAnalyzerForPython.cpp
      src/TensorQuantizationSimForPython.cpp
      src/TensorQuantizer.cpp
      src/GraphQuantizer.cpp
      src/Fp16Quantization.cpp
      src/ParserModule.cpp

      include/DlQuantization/IQuantizer.hpp
      include/DlQuantization/Quantization.hpp
      include/DlQuantization/QuantizerFactory.hpp
      include/DlQuantization/IQuantizationEncodingAnalyzer.hpp
      include/DlQuantization/ITensorQuantizationSim.h
      include/DlQuantization/EncodingAnalyzerForPython.h
      include/DlQuantization/TensorQuantizationSimForPython.h
      include/DlQuantization/TensorQuantizer.h
      include/DlQuantization/GraphQuantizer.h
      include/DlQuantization/Fp16Quantization.hpp
      include/DlQuantization/ParserModule.h
      include/DlQuantization/XmlTypes.h
      include/DlQuantization/EncodingRescale.hpp
      )


if (ENABLE_CUDA)
target_compile_options(MoDlQuantization
      PRIVATE
         -DGPU_QUANTIZATION_ENABLED=1
      )
endif (ENABLE_CUDA)

target_include_directories(MoDlQuantization
      PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
      )

target_link_libraries(MoDlQuantization PUBLIC
    pybind11
    pugixml
    )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if (ENABLE_CUDA)

    add_library(MoDlQuantizationCuda
      src/spec_functions.cuh
      src/spec_functions.cu

      src/cuda_util.cu
      src/trim_functions.cu
      src/cuda_util.hpp
      src/trim_functions.cuh
      src/math_functions.cu
      )

    target_compile_options(MoDlQuantizationCuda
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --use_fast_math>
            -Xcompiler -fPIC
            -DGPU_QUANTIZATION_ENABLED=1
            )

endif (ENABLE_CUDA)

if (ENABLE_TESTS)
    add_subdirectory(test)
endif()

if (SKBUILD)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/DlQuantization
            DESTINATION aimet_common/x86_64-linux-gnu/include)
else (SKBUILD)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/DlQuantization
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/x86_64-linux-gnu/include)

    whl_add_whl_prep_common_target(DlQuantization)
endif (SKBUILD)

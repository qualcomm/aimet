#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018-2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

include_directories(BEFORE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

add_executable(MoDlQuantizationTest
      test_quantization_lib.hpp
      test_quantization_lib_tf_enhanced_cpugpu.cpp
      test_quantization_lib_tf_cpu.cpp
      test_quantization_lib_tf_cpugpu.cpp
      TestTfEnhancedEncodingAnalyzer.cpp
      TestTfEncodingAnalyzer.cpp
      TestTensorQuantizationSim.cpp
      TestTensorQuantizer.cpp
      TestGraphQuantizer.cpp
      TestUtilities.cpp
      TestPercentileEncodingAnalyzer.cpp
      TestMseEncodingAnalyzer.cpp
      TestEntropyEncodingAnalyzer.cpp
      TestQuantizationFunctions.cpp
      TestEncodingRescale.cpp
      )

if (ENABLE_CUDA)
    target_compile_options(MoDlQuantizationTest
            PRIVATE
                -DGPU_QUANTIZATION_ENABLED=1
            )

    target_link_libraries(MoDlQuantizationTest
            MoDlQuantization
            MoDlQuantizationCuda
            CUDA::cublas
            gtest_main
            )

else (ENABLE_CUDA)
    target_link_libraries(MoDlQuantizationTest
            MoDlQuantization
            gtest_main
            )

endif (ENABLE_CUDA)


add_test(NAME MoDlQuantizationTest
      COMMAND $<TARGET_FILE:MoDlQuantizationTest> --gtest_output=xml:cpp_test_output.xml)

add_test(NAME MoDlQuantizationPythonTest
      COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output.xml
      )

set_property(TEST MoDlQuantizationPythonTest
      APPEND PROPERTY ENVIRONMENT "${AIMET_PYTHONPATH}")


#GTEST_ADD_TESTS(MoDlQuantizationTest "--gtest_output=xml:cpp_test_output.xml" AUTO)
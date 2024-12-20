#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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

if (ENABLE_CUDA)
    add_test(OnnxTrainingExtensionTest
            ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output.xml
            )

else (ENABLE_CUDA)
    add_test(OnnxTrainingExtensionTest
            ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} -m "not cuda" --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_test_output.xml
            )

endif (ENABLE_CUDA)

add_subdirectory(python/models/customops)

if(NOT DEFINED ENV{FAST_RELEASE})
    # When building PyPI wheel packages (aka FAST_RELEASE),
    # we only want to test the pure wheel package without relying on any other build artifacts
    set_property(TEST OnnxTrainingExtensionTest
          APPEND PROPERTY ENVIRONMENT "${AIMET_PYTHONPATH}")
endif()

if(DEFINED ENV{DEPENDENCY_DATA_PATH})
    set_property(TEST OnnxTrainingExtensionTest
        APPEND PROPERTY ENVIRONMENT "${DEPENDENCY_DATA_PATH}")
endif(DEFINED ENV{DEPENDENCY_DATA_PATH})

if (ENABLE_CUDA)
    add_executable(OnnxTrainingExtensionCppTest
            TestOnnxTensorOps.cpp)

    target_compile_options(OnnxTrainingExtensionCppTest
            PRIVATE
            -DONNX_CUDA=1
            )

    target_include_directories(OnnxTrainingExtensionCppTest PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../src
            ${EIGEN3_INCLUDE_DIR}
            ${ONNXRUNTIME_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
            )

    target_link_libraries(OnnxTrainingExtensionCppTest
            gtest_main
            OnnxCppOps
            MoDlQuantization
            MoDlQuantizationCuda
            CUDA::cublas
            )

    add_test(OnnxTrainingExtensionCppTest
            OnnxTrainingExtensionCppTest --gtest_output=xml:cpp_test_output.xml)
endif (ENABLE_CUDA)

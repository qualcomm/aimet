#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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

add_library(TrainingExtensionsTf SHARED
        src/QcQuantizeStaticOp.hpp
        src/QcQuantizeStaticOp.cpp
        src/QcQuantizePerChannelOp.cpp
        src/QcQuantizePerChannelOp.hpp
        src/AimetOpUtils.h
        src/AimetOpUtils.cpp
        src/QcQuantizeRecurrentParamOp.hpp
        src/QcQuantizeRecurrentParamOp.cpp
        src/QcQuantizeOp.hpp
        src/QcQuantizeOp.cpp
        src/AimetFp16OpUtils.h)

target_include_directories(TrainingExtensionsTf PRIVATE
        ${TF_LIB_DIR}/include
        ${TF_LIB_DIR}/include/external/nsync/public
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
        )

target_link_libraries(TrainingExtensionsTf PUBLIC
        MoDlCompression
        MoDlQuantization
        TensorFlow
        PyWrapTensorFlowInternal
        ${Python_LIBRARIES}
        )

if (ENABLE_CUDA)
    target_compile_options(TrainingExtensionsTf PRIVATE
            -DGOOGLE_CUDA=1
            )

    target_link_libraries(TrainingExtensionsTf PUBLIC
            MoDlQuantizationCuda
            TrainingExtensionsTfCuda
            )
endif(ENABLE_CUDA)

set_target_properties(TrainingExtensionsTf PROPERTIES
      OUTPUT_NAME "aimet_tf_ops"
      )

if(SKBUILD)
else(SKBUILD)
    set_target_properties(TrainingExtensionsTf PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/aimet_common"
          )
endif(SKBUILD)

set(INIT_PY "\
#==============================================================================\\n\
#  @@-COPYRIGHT-START-@@\\n\
#\\n\
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.\\n\
#\\n\
#  Redistribution and use in source and binary forms, with or without\\n\
#  modification, are permitted provided that the following conditions are met:\\n\
#\\n\
#  1. Redistributions of source code must retain the above copyright notice,\\n\
#     this list of conditions and the following disclaimer.\\n\
#\\n\
#  2. Redistributions in binary form must reproduce the above copyright notice,\\n\
#     this list of conditions and the following disclaimer in the documentation\\n\
#     and/or other materials provided with the distribution.\\n\
#\\n\
#  3. Neither the name of the copyright holder nor the names of its contributors\\n\
#     may be used to endorse or promote products derived from this software\\n\
#     without specific prior written permission.\\n\
#\\n\
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"\\n\
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE\\n\
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE\\n\
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE\\n\
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR\\n\
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF\\n\
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS\\n\
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN\\n\
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)\\n\
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE\\n\
#  POSSIBILITY OF SUCH DAMAGE.\\n\
#\\n\
#  SPDX-License-Identifier: BSD-3-Clause\\n\
#\\n\
#  @@-COPYRIGHT-END-@@\\n\
#==============================================================================\\n\
import pathlib\\n\
import tensorflow as tf\\n\
__module = tf.load_op_library(str(pathlib.Path(__file__).parent.parent / '$<TARGET_FILE_NAME:TrainingExtensionsTf>'))\\n\
for __name in dir(__module):\\n\
\\tif not __name.startswith('_'):\\n\
\\t\\tglobals()[__name] = getattr(__module, __name)\\n\
del __name\\n\
del __module\\n\
del tf\\n\
del pathlib\\n\
")
add_custom_command(TARGET TrainingExtensionsTf POST_BUILD
      # Create a directory with the same name as *.so with __init__.py file to force python loads
      # a trampoline module which would call tf.load_op_library.
      COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:TrainingExtensionsTf>/$<TARGET_FILE_PREFIX:TrainingExtensionsTf>$<TARGET_FILE_BASE_NAME:TrainingExtensionsTf>
      COMMAND echo ${INIT_PY} > "$<TARGET_FILE_DIR:TrainingExtensionsTf>/$<TARGET_FILE_PREFIX:TrainingExtensionsTf>$<TARGET_FILE_BASE_NAME:TrainingExtensionsTf>/__init__.py"
      VERBATIM
)

get_target_property(dirs MoDlQuantization INCLUDE_DIRECTORIES)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/include ${TF_LIB_DIR}/include/external/nsync/public ${dirs})

if (ENABLE_CUDA)
    add_library(TrainingExtensionsTfCuda
            src/AimetOpUtilsGpu.cu
            )
    if (TF_VERSION VERSION_GREATER_EQUAL "2.9")
        set(ABI_VALUE "1")
    else ()
        set(ABI_VALUE "0")
    endif ()
    target_compile_options(TrainingExtensionsTfCuda PRIVATE
            -Xcompiler
            -D_GLIBCXX_USE_CXX11_ABI=${ABI_VALUE}
            -DGOOGLE_CUDA=1
            -I ${TF_LIB_DIR}/include
            --expt-relaxed-constexpr
            )
endif(ENABLE_CUDA)

if(SKBUILD)
    install(FILES $<TARGET_FILE:TrainingExtensionsTf>
        DESTINATION aimet_common
    )

    install(DIRECTORY $<TARGET_FILE_DIR:TrainingExtensionsTf>/$<TARGET_FILE_PREFIX:TrainingExtensionsTf>$<TARGET_FILE_BASE_NAME:TrainingExtensionsTf>
        DESTINATION aimet_common
    )
else(SKBUILD)
    install(FILES $<TARGET_FILE:TrainingExtensionsTf>
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/aimet_common
    )

    install(DIRECTORY $<TARGET_FILE_DIR:TrainingExtensionsTf>/$<TARGET_FILE_PREFIX:TrainingExtensionsTf>$<TARGET_FILE_BASE_NAME:TrainingExtensionsTf>
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/aimet_common
    )
endif(SKBUILD)

add_custom_target(whl_prep_cp_tensorflow
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_TENSORFLOW_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:TrainingExtensionsTf> ${WHL_PREP_AIMET_TENSORFLOW_DIR}/$<TARGET_FILE_NAME:TrainingExtensionsTf>
    COMMAND ${PATCHELF_EXE} --set-rpath '$$ORIGIN:$ORIGIN/../../tensorflow:$$ORIGIN/../../tensorflow/python' ${WHL_PREP_AIMET_TENSORFLOW_DIR}/$<TARGET_FILE_NAME:TrainingExtensionsTf>
    DEPENDS TrainingExtensionsTf TfTrainingExtensionCppTest
)
add_custom_target(whl_prep_ln_tensorflow
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WHL_PREP_AIMET_TENSORFLOW_DIR}
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:TrainingExtensionsTf> ${WHL_PREP_AIMET_TENSORFLOW_DIR}/$<TARGET_FILE_NAME:TrainingExtensionsTf>
    DEPENDS TrainingExtensionsTf TfTrainingExtensionCppTest
)
add_dependencies(whl_prep_cp whl_prep_cp_tensorflow)
add_dependencies(whl_prep_ln whl_prep_ln_tensorflow)

whl_add_whl_action_target(tensorflow)

add_subdirectory(src/python)

if (ENABLE_TESTS)
    add_subdirectory(test)
endif()

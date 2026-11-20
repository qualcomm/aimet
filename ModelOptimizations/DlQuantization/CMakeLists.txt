# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_library(MoDlQuantization STATIC

      src/math_functions.cpp
      src/math_functions.hpp
      src/quantization_utils.cpp
      src/quantization_utils.hpp
      src/QuantizerFactory.cpp
      src/trim_functions.cpp
      src/trim_functions.hpp
      src/tensor_utils.cpp
      src/tensor_utils.hpp

      src/MinMaxEncodingAnalyzer.cpp
      src/MinMaxEncodingAnalyzer.h
      src/PercentileEncodingAnalyzer.cpp
      src/PercentileEncodingAnalyzer.h
      src/TfEnhancedEncodingAnalyzer.cpp
      src/TfEnhancedEncodingAnalyzer.h
      src/TfEncodingAnalyzer.h
      src/TfEncodingAnalyzer.cpp
      src/EncodingAnalyzerWrapper.h
      src/EncodingAnalyzerWrapper.cpp
      src/TensorQuantizer.cpp
      src/Fp16Quantization.cpp

      include/DlQuantization/Quantization.hpp
      include/DlQuantization/QuantizationType.hpp
      include/DlQuantization/QuantizerFactory.hpp
      include/DlQuantization/IQuantizationEncodingAnalyzer.hpp
      include/DlQuantization/TensorQuantizer.h
      include/DlQuantization/Fp16Quantization.hpp
      )


if (ENABLE_CUDA)
target_compile_options(MoDlQuantization
      PRIVATE
         -DGPU_QUANTIZATION_ENABLED=1
      )
endif (ENABLE_CUDA)

target_link_libraries(MoDlQuantization
      PUBLIC
        Eigen3::Eigen
      )

target_include_directories(MoDlQuantization
      PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
      )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if (ENABLE_CUDA)

    add_library(MoDlQuantizationCuda
      src/cuda_util.cu
      src/trim_functions.cu
      src/cuda_util.hpp
      src/trim_functions.cuh
      src/math_functions.cu
      src/tensor_utils.cu
      )

    target_include_directories(MoDlQuantizationCuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(MoDlQuantizationCuda
        PUBLIC
          Eigen3::Eigen
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
    if (ENABLE_TORCH)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/DlQuantization
                DESTINATION aimet_torch/common/${AIMET_PLATFORM_DIR}/include)
    endif()
    if (ENABLE_ONNX)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/DlQuantization
                DESTINATION aimet_onnx/common/${AIMET_PLATFORM_DIR}/include)
    endif()
else (SKBUILD)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/DlQuantization
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${AIMET_PLATFORM_DIR}/include)
endif (SKBUILD)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_subdirectory(src/python)

if (ENABLE_TESTS)
    add_subdirectory(test)
endif()

add_library(OnnxCppOps SHARED
        src/QcQuantizeOp.h
        src/AimetOpUtils.h
        src/AimetOpUtils.cpp
        src/RegisterCustomOps.h
        src/RegisterCustomOps.cpp
        src/QcQuantizeInfo.h
        src/QcQuantizeOp.cpp
        src/OnnxOpUtils.h
)

target_include_directories(OnnxCppOps PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ModelOptimizations/DlQuantization/src
        )

target_link_libraries(OnnxCppOps PUBLIC
        MoDlQuantization
        Eigen3::Eigen
        )

if (ENABLE_CUDA)
    target_compile_options(OnnxCppOps PRIVATE
            -DONNX_CUDA=1
            )

    target_link_libraries(OnnxCppOps PUBLIC
            MoDlQuantizationCuda
            )
endif (ENABLE_CUDA)

if (SKBUILD)
    set_target_properties(OnnxCppOps PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "aimet_onnxrt_ops"
        )

    # Set RPATH using platform-specific syntax
    if (APPLE)
        set_target_properties(OnnxCppOps PROPERTIES
            INSTALL_RPATH "@loader_path"
            )
    elseif (NOT WIN32)
        set_target_properties(OnnxCppOps PROPERTIES
            INSTALL_RPATH "\$ORIGIN"
            )
    endif()

    install(TARGETS OnnxCppOps
            LIBRARY DESTINATION aimet_onnx/common
            RUNTIME DESTINATION aimet_onnx/common)

else (SKBUILD)
    set_target_properties(OnnxCppOps PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "aimet_onnxrt_ops"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/_aimet_common"
        )

install(TARGETS OnnxCppOps
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/_aimet_common
        )

endif (SKBUILD)

# ---- Cython libquant_info module (Python Limited API / Stable ABI) ----

set(QUANT_INFO_PYX "${CMAKE_CURRENT_SOURCE_DIR}/src/python/aimet_onnx/libquant_info.pyx")
set(QUANT_INFO_PXD "${CMAKE_CURRENT_SOURCE_DIR}/src/python/aimet_onnx/libquant_info.pxd")
set(QUANT_INFO_CPP "${CMAKE_CURRENT_BINARY_DIR}/libquant_info.cpp")

# Generate C++ from Cython using Python module
add_custom_command(
    OUTPUT ${QUANT_INFO_CPP}
    COMMAND ${Python3_EXECUTABLE} -m cython --cplus -3
            -I${CMAKE_CURRENT_SOURCE_DIR}/src
            -I${CMAKE_CURRENT_SOURCE_DIR}/src/python/aimet_onnx
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../ModelOptimizations/DlQuantization/include
            -o ${QUANT_INFO_CPP}
            ${QUANT_INFO_PYX}
    DEPENDS ${QUANT_INFO_PYX} ${QUANT_INFO_PXD}
    COMMENT "Generating C++ from Cython: libquant_info.pyx"
    VERBATIM
)

# Add custom target to ensure Cython generation completes before library build
# This prevents race conditions in parallel builds (-j flag)
add_custom_target(generate_quant_info_cpp DEPENDS ${QUANT_INFO_CPP})

# Build the Cython-generated C++ as a Python extension
# USE_SABI 3.10 enables Python Limited API (Stable ABI) for Python 3.10+
# This allows the extension to work across Python versions without recompilation
# and properly links against the stable ABI library (python3.lib on Windows)
Python3_add_library(quant_info MODULE USE_SABI 3.10 ${QUANT_INFO_CPP})
add_dependencies(quant_info generate_quant_info_cpp)

target_include_directories(quant_info PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ModelOptimizations/DlQuantization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ModelOptimizations/DlQuantization/src
        )

target_link_libraries(quant_info PUBLIC
        MoDlQuantization
        )

if (ENABLE_CUDA)
    target_link_libraries(quant_info PUBLIC
            MoDlQuantizationCuda
            )
endif (ENABLE_CUDA)

if (SKBUILD)
    set_target_properties(quant_info
        PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "quant_info"
        )

    # Set RPATH using platform-specific syntax
    if (APPLE)
        set_target_properties(quant_info PROPERTIES
            INSTALL_RPATH "@loader_path"
            )
    elseif (NOT WIN32)
        set_target_properties(quant_info PROPERTIES
            INSTALL_RPATH "\$ORIGIN"
            )
    endif()

    install(TARGETS quant_info
            LIBRARY DESTINATION aimet_onnx/common
            RUNTIME DESTINATION aimet_onnx/common)
else (SKBUILD)

set_target_properties(quant_info
        PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "quant_info"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/_aimet_common/"
        )

install(TARGETS quant_info
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/_aimet_common
        )

endif (SKBUILD)

# Set Python extension suffix
# Windows: use .pyd (Windows Python doesn't recognize .abi3.pyd in EXTENSION_SUFFIXES)
# Unix: use .abi3.so for stable ABI
if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set_target_properties(quant_info
        PROPERTIES
        SUFFIX ".pyd"
        )
else()
    set_target_properties(quant_info
        PROPERTIES
        SUFFIX ".abi3.so"
        )
endif()

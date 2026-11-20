# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# ---- Cython libpymo module (Python Limited API / Stable ABI) ----

set(LIBPYMO_PYX "${CMAKE_CURRENT_SOURCE_DIR}/_libpymo.pyx")
set(LIBPYMO_PXD "${CMAKE_CURRENT_SOURCE_DIR}/_libpymo.pxd")
set(QUANT_TYPES_PXD "${CMAKE_CURRENT_SOURCE_DIR}/../DlQuantization/include/DlQuantization/_quant_types.pxd")
set(LIBPYMO_CPP "${CMAKE_CURRENT_BINARY_DIR}/_libpymo.cpp")

# Generate C++ from Cython using Python module
add_custom_command(
    OUTPUT ${LIBPYMO_CPP}
    COMMAND ${Python3_EXECUTABLE} -m cython --cplus -3
            -I${CMAKE_CURRENT_SOURCE_DIR}/../DlQuantization/include
            -o ${LIBPYMO_CPP}
            ${LIBPYMO_PYX}
    DEPENDS ${LIBPYMO_PYX} ${LIBPYMO_PXD} ${QUANT_TYPES_PXD}
    COMMENT "Generating C++ from Cython: libpymo.pyx"
    VERBATIM
)

# Add custom target to ensure Cython generation completes before library build
# This prevents race conditions in parallel builds (-j flag)
add_custom_target(generate_libpymo_cpp DEPENDS ${LIBPYMO_CPP})

# Build the Cython-generated C++ as a Python extension
# USE_SABI 3.10 enables Python Limited API (Stable ABI) for Python 3.10+
# This allows the extension to work across Python versions without recompilation
# and properly links against the stable ABI library (python3.lib on Windows)
Python3_add_library(PyModelOptimizations MODULE USE_SABI 3.10 ${LIBPYMO_CPP})
add_dependencies(PyModelOptimizations generate_libpymo_cpp)

target_include_directories(PyModelOptimizations PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../DlQuantization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../DlQuantization/src
        )

target_link_libraries(PyModelOptimizations PUBLIC
        MoDlQuantization
        )

if (ENABLE_CUDA)
    target_link_libraries(PyModelOptimizations PUBLIC
            MoDlQuantizationCuda
            )
endif (ENABLE_CUDA)

set_target_properties(PyModelOptimizations
      PROPERTIES
         OUTPUT_NAME "pymo"
         PREFIX "_lib"
      )

if (SKBUILD)
    if (APPLE)
        set_target_properties(PyModelOptimizations PROPERTIES
            INSTALL_RPATH "@loader_path:@loader_path/../../.."
            )
    elseif (NOT WIN32)
        set_target_properties(PyModelOptimizations PROPERTIES
            INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../.."
            )
    endif()
    if (ENABLE_ONNX)
        install(TARGETS PyModelOptimizations
                LIBRARY DESTINATION aimet_onnx/common
                RUNTIME DESTINATION aimet_onnx/common)
    endif()
else (SKBUILD)
    set_target_properties(PyModelOptimizations
          PROPERTIES
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/_aimet_common/"
          )

    install(TARGETS PyModelOptimizations
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/_aimet_common
    )

endif (SKBUILD)

# Set Python extension suffix
# Windows: use .pyd (Windows Python doesn't recognize .abi3.pyd in EXTENSION_SUFFIXES)
# Unix: use .abi3.so for stable ABI
if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set_target_properties(PyModelOptimizations
          PROPERTIES
          SUFFIX ".pyd"
          )
else()
    set_target_properties(PyModelOptimizations
          PROPERTIES
          SUFFIX ".abi3.so"
          )
endif()

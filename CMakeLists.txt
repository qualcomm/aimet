#=============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
#
#=============================================================================
cmake_minimum_required(VERSION 3.19)

project(aimet)

include(GNUInstallDirs)

include(cmake/PreparePyTorch.cmake)
include(cmake/PrepareONNX.cmake)
include(cmake/PrepareTensorFlow.cmake)
include(cmake/PreparePyBind11.cmake)

# Project-global settings
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Boolean describing whether the value of CXX_STANDARD is a requirement.")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Boolean specifying whether compiler specific extensions are requested.")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(NOTICE "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX} (for install target)")

if(NOT DEFINED AIMET_PYTHONPATH)
    set(AIMET_PYTHONPATH "PYTHONPATH=${CMAKE_BINARY_DIR}/artifacts" CACHE STRING "python path")
endif()
set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/common/src/python")

set(WHL_EDITABLE_MODE OFF CACHE BOOL "Enable editable mode, wheels would have symlinks to C++ part instead of copies")
set(WHL_PREP_DIR "${CMAKE_BINARY_DIR}/whlprep" CACHE STRING "A path to store extra files which should be included in the wheels")
set(WHL_PREP_AIMET_COMMON_DIR "${WHL_PREP_DIR}/aimet_common" CACHE STRING "A path to store extra files which should be included in the aimet_common wheel")
set(WHL_PREP_AIMET_TORCH_DIR "${WHL_PREP_DIR}/aimet_torch" CACHE STRING "A path to store extra files which should be included in the aimet_torch wheel")
set(WHL_PREP_AIMET_TENSORFLOW_DIR "${WHL_PREP_DIR}/aimet_tensorflow" CACHE STRING "A path to store extra files which should be included in the aimet_tensorflow wheel")
set(WHL_PREP_AIMET_ONNX_DIR "${WHL_PREP_DIR}/aimet_onnx" CACHE STRING "A path to store extra files which should be included in the aimet_onnx wheel")
mark_as_advanced(WHL_PREP_DIR WHL_PREP_AIMET_COMMON_DIR WHL_PREP_AIMET_TORCH_DIR WHL_PREP_AIMET_TENSORFLOW_DIR WHL_PREP_AIMET_ONNX_DIR)

# Set the software version from version.txt file (if not already set)
if(NOT DEFINED SW_VERSION)
    file(STRINGS "packaging/version.txt" SW_VERSION)
    message(STATUS "Set SW_VERSION = ${SW_VERSION} from ${CMAKE_CURRENT_SOURCE_DIR}/packaging/version.txt")
else()
    message(STATUS "SW_VERSION already set to ${SW_VERSION}.")
endif()

# Set default CMake options
option(ENABLE_CUDA "Enable use of CUDA" ON)
option(ENABLE_TORCH "Enable AIMET-Torch build" ON)
option(ENABLE_ONNX "Enable AIMET-ONNX build" OFF)
option(ENABLE_TENSORFLOW "Enable AIMET-TensorFlow build" OFF)
option(ENABLE_TESTS "Enable building tests" ON)

option(ENABLE_GCC_PRE_CXX11_ABI "Compile using pre-C++11 ABI" OFF)
option(AUTO_ENABLE_GCC_PRE_CXX11_ABI "Auto-enable pre-C++11 ABI when deemed necessary" ON)

set(GCC_ABI_VERSION "0" CACHE STRING "Set GCC compiler ABI version (will set -fabi-version; AUTO_SET_GCC_ABI_VERSION=ON will override)")
option(AUTO_SET_GCC_ABI_VERSION "Automatically try to set GCC compiler ABI version" ON)

message(STATUS "AIMET build configuration:")
message(STATUS "** ENABLE_CUDA = ${ENABLE_CUDA}")
message(STATUS "** ENABLE_TORCH = ${ENABLE_TORCH}")
message(STATUS "** ENABLE_ONNX = ${ENABLE_ONNX}")
message(STATUS "** ENABLE_TENSORFLOW = ${ENABLE_TENSORFLOW}")
message(STATUS "** ENABLE_TESTS = ${ENABLE_TESTS}")
message(STATUS "** ---")
message(STATUS "** ENABLE_GCC_PRE_CXX11_ABI  = ${ENABLE_GCC_PRE_CXX11_ABI}")
message(STATUS "** AUTO_ENABLE_GCC_PRE_CXX11_ABI  = ${AUTO_ENABLE_GCC_PRE_CXX11_ABI}")
message(STATUS "** ---")
message(STATUS "** GCC_ABI_VERSION = ${GCC_ABI_VERSION}")
message(STATUS "** AUTO_SET_GCC_ABI_VERSION = ${AUTO_SET_GCC_ABI_VERSION}")
message(STATUS "** ---")

if (ENABLE_TORCH AND ENABLE_TENSORFLOW)
    # Remove this once it's proven feasible.
    message(WARNING "AIMET Torch and TensorFlow support currently cannot be built at the same time.")
endif()

# Find Python libraries
set(Python3_FIND_VIRTUALENV "STANDARD")
find_package(Python3 COMPONENTS Interpreter Development)
message(STATUS "Found Python3: ${Python3_FOUND}, at ${Python3_LIBRARIES}")

# -------------------------------
# Compilation flags - not (yet) per target
# -------------------------------

if (ENABLE_TORCH)
    if (AUTO_ENABLE_GCC_PRE_CXX11_ABI)
        # michof: Current PyTorch wheels appear to have been compiled with this setting.
        # In practice, we don't see C++11 ABI issues when this is disabled.
        message(NOTICE "** Force set ENABLE_GCC_PRE_CXX11_ABI = ON (Reason: PyTorch)")
        set(ENABLE_GCC_PRE_CXX11_ABI ON CACHE BOOL "" FORCE)
    endif()

    if (AUTO_SET_GCC_ABI_VERSION)
        # See, e.g., https://github.com/pytorch/pytorch/blob/v2.3.1/CMakeLists.txt#L64
        # michof: Current PyTorch wheels appear to have been compiled with this setting, so this
        # needs to be set here to ensure pybind11 internals keep working (this will influence
        # PYBIND11_BUILD_ABI, which should not be set directly).
        # NB, this ABI version is set in PyTorch only when GLIBCXX_USE_CXX11_ABI==0.
        message(NOTICE "** Set GCC_ABI_VERSION = 11 (Reason: PyTorch)")
        set(GCC_ABI_VERSION "11" CACHE STRING "" FORCE)
    endif()
endif()

if (ENABLE_TENSORFLOW)
    set_tensorflow_version() # we already need the version right below
    if (AUTO_ENABLE_GCC_PRE_CXX11_ABI AND TF_VERSION VERSION_LESS "2.9")
        # michof: Keep behavior as in previous versions of this file.
        message(NOTICE "** Set ENABLE_GCC_PRE_CXX11_ABI = ON (Reason: TensorFlow <2.9)")
        set(ENABLE_GCC_PRE_CXX11_ABI ON CACHE BOOL "" FORCE)
    endif()
endif()

if(DEFINED CMAKE_CXX_FLAGS)
    if(NOT CMAKE_CXX_FLAGS MATCHES "-O(0|1|2|3|s|fast)")
        message(NOTICE "No optimization flag found. Setting optimization to -O3.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif()
else()
    message(NOTICE "CMAKE_CXX_FLAGS not defined. Setting optimization to -O3.")
    set(CMAKE_CXX_FLAGS "-O3")
endif()

if (ENABLE_GCC_PRE_CXX11_ABI)
    message(NOTICE "Adding to CXXFLAGS: -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${GCC_ABI_VERSION}" GREATER "0")
    message(NOTICE "Adding to CXXFLAGS: -fabi-version=${GCC_ABI_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=${GCC_ABI_VERSION}")
endif()

# -------------------------------
# Centralized dependency handling
# -------------------------------

# Find system thread library. Seems to be only needed by DlCompression tests. May be optional - not sure.
find_package(Threads)

find_package(PkgConfig)
find_package (Eigen3 REQUIRED)

#find_package(OpenCV REQUIRED)
pkg_search_module(OPENCV opencv) # in case opencv will not be found, it would be built form source

include(cmake/ThirdPartyDependencies.cmake)

if (ENABLE_TORCH)
    set_torch_version()
    set_torch_cmake_prefix_path()
    check_torch_cxx_abi_compatibility()
    update_torch_cuda_arch_list()

    find_package(Torch REQUIRED)
endif()

if (ENABLE_ONNX)
    set_onnx_version()
    set_onnxruntime_variables()
    set_torch_cmake_prefix_path()

endif()

if (ENABLE_TENSORFLOW)
    set_tensorflow_version()
    set_tensorflow_library_path()
    add_library_tensorflow(${TF_LIB_DIR})
    add_library_pywrap_tensorflow_internal(${TF_LIB_DIR})
endif()

add_library_pybind11()
# michof: We should consider replacing the above macro simply by:
#find_package(pybind11 CONFIG REQUIRED)


# -------------------------------
# Conditional build for CUDA
# -------------------------------
set(CUDA_VER_STRING "cpu")

if (ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Truncate the string for use in version string ex. 11.6.124 --> cu116
    string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.*[0-9]*" "cu\\1\\2" CUDA_VER_STRING ${CUDAToolkit_VERSION})
    message(STATUS "Found CUDA toolkit version ${CUDAToolkit_VERSION}, using ${CUDA_VER_STRING}")
endif(ENABLE_CUDA)

if (ENABLE_TORCH)
  if (NOT ENABLE_ONNX)
    set(FMWORK_VERSION ${TORCH_VERSION})
  endif()
  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/torch/src/python")
endif (ENABLE_TORCH)

if (ENABLE_ONNX)
  set(FMWORK_VERSION ${ONNX_VERSION})
  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/onnx/src/python")
endif (ENABLE_ONNX)

if (ENABLE_TENSORFLOW)
  set(FMWORK_VERSION ${TF_VERSION})
  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/tensorflow/src/python")
endif (ENABLE_TENSORFLOW)


# Export PYTHONPATH to the parent cmake scope (if present)
get_directory_property(hasParent PARENT_DIRECTORY)

if(hasParent)
  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}" PARENT_SCOPE)
else()
  message(STATUS "Set ${AIMET_PYTHONPATH} in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# -------------------------------
# Generate pip packages
# -------------------------------

# Set the packaging path (if not already set)
if(NOT DEFINED AIMET_PACKAGE_PATH)
    set(AIMET_PACKAGE_PATH ${CMAKE_INSTALL_PREFIX})
    message(STATUS "Set AIMET_PACKAGE_PATH = ${AIMET_PACKAGE_PATH}")
endif()

set(remote_url_cmake_opt "-DREMOTE_URL=\"\"")

if (PIP_INDEX EQUAL "reporelease")
  execute_process(COMMAND git config --get remote.origin.url WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE remote_url)

  if(NOT remote_url STREQUAL "")
    string(REGEX REPLACE "\n$" "" remote_url "${remote_url}")
    # Remove the ".git" suffix from the remote repo URL
    string(REGEX REPLACE "\\.[^.]*$" "" remote_url ${remote_url})
    string(REGEX REPLACE ".*@" "" remote_post ${remote_url})
    set(remote_url "https://${remote_post}")
    message(STATUS "Repo Remote URL = ${remote_url}")
    set(remote_url_cfg "\"${remote_url}\"")
    set(sw_version_cfg "\"${SW_VERSION}\"")

    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/setup_cfg.py" "version=${sw_version_cfg}\n")
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/packaging/setup_cfg.py" "remote_url=${remote_url_cfg}")

    set(remote_url_cmake_opt "-DREMOTE_URL=${remote_url}")
  else()
    message(FATAL_ERROR "Repo Remote URL is blank. Unable to create AIMET wheel package")
  endif()
endif()

add_custom_target(packageaimet
  # Run the install target first
  COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --target install

  # Now run the packaging target to generate wheel files
  COMMAND ${CMAKE_COMMAND} -DSW_VERSION=${SW_VERSION} -DCUDA_VER_STRING=${CUDA_VER_STRING} -DPIP_INDEX=${PIP_INDEX} ${remote_url_cmake_opt} -DPYTHON3_EXECUTABLE=${Python3_EXECUTABLE} -DAIMET_PACKAGE_PATH=${AIMET_PACKAGE_PATH} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DENABLE_CUDA=${ENABLE_CUDA} -DENABLE_TENSORFLOW=${ENABLE_TENSORFLOW} -DENABLE_TORCH=${ENABLE_TORCH} -DENABLE_ONNX=${ENABLE_ONNX} -DPATCHELF_EXE=${PATCHELF_EXE} -DFMWORK_VERSION=${FMWORK_VERSION} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/package_aimet.cmake
)


if (ENABLE_TESTS)
    enable_testing()
    add_subdirectory(NightlyTests)
endif()

add_subdirectory(ModelOptimizations)
add_subdirectory(TrainingExtensions)
add_subdirectory(Examples)
add_subdirectory(Docs)

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks")
  add_subdirectory(Benchmarks)
endif()

# -------------------------------
# Upload pip packages
# -------------------------------

# Check the pip config file path and set it to a default value (if not set)
if(NOT DEFINED PIP_CONFIG_FILE)
  set(PIP_CONFIG_FILE "None")
endif()

# Check the pip index name and set it to a default value (if not set)
if(NOT DEFINED PIP_INDEX)
  set(PIP_INDEX "None")
endif()

# Check the pip certificate path and set it to a default value (if not set)
if(NOT DEFINED PIP_CERT_FILE)
  set(PIP_CERT_FILE "None")
endif()

#TODO For some reason, this package upload target does NOT work as expected and needs to be debugged
add_custom_target(upload
    # Now run the packaging target to upload the pip package
    COMMAND ${CMAKE_COMMAND} -DPIP_CONFIG_FILE=${PIP_CONFIG_FILE} -DPIP_INDEX=${PIP_INDEX} -DPIP_CERT_FILE=${PIP_CERT_FILE} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/upload_aimet.cmake
    DEPENDS packageaimet
)

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.26)

project(aimet)

include(GNUInstallDirs)

include(cmake/PreparePyTorch.cmake)
include(cmake/PrepareONNX.cmake)
include(cmake/PreparePyBind11.cmake)

# Project-global settings
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Boolean describing whether the value of CXX_STANDARD is a requirement.")
set(CMAKE_CXX_EXTENSIONS ON CACHE BOOL "Boolean specifying whether compiler specific extensions are requested.")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(NOTICE "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX} (for install target)")

if(NOT DEFINED AIMET_PYTHONPATH)
    set(AIMET_PYTHONPATH "PYTHONPATH=${CMAKE_BINARY_DIR}/artifacts" CACHE STRING "python path")
endif()
set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/common/src/python")

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
option(ENABLE_TESTS "Enable building tests" ON)

option(ENABLE_GCC_PRE_CXX11_ABI "Compile using pre-C++11 ABI" OFF)
option(AUTO_ENABLE_GCC_PRE_CXX11_ABI "Auto-enable pre-C++11 ABI when deemed necessary" ON)

set(GCC_ABI_VERSION "0" CACHE STRING "Set GCC compiler ABI version (will set -fabi-version; AUTO_SET_GCC_ABI_VERSION=ON will override)")
option(AUTO_SET_GCC_ABI_VERSION "Automatically try to set GCC compiler ABI version" ON)

message(STATUS "AIMET build configuration:")
message(STATUS "** ENABLE_CUDA = ${ENABLE_CUDA}")
message(STATUS "** ENABLE_TORCH = ${ENABLE_TORCH}")
message(STATUS "** ENABLE_ONNX = ${ENABLE_ONNX}")
message(STATUS "** ENABLE_TESTS = ${ENABLE_TESTS}")
message(STATUS "** ---")
message(STATUS "** ENABLE_GCC_PRE_CXX11_ABI  = ${ENABLE_GCC_PRE_CXX11_ABI}")
message(STATUS "** AUTO_ENABLE_GCC_PRE_CXX11_ABI  = ${AUTO_ENABLE_GCC_PRE_CXX11_ABI}")
message(STATUS "** ---")
message(STATUS "** GCC_ABI_VERSION = ${GCC_ABI_VERSION}")
message(STATUS "** AUTO_SET_GCC_ABI_VERSION = ${AUTO_SET_GCC_ABI_VERSION}")
message(STATUS "** ---")

# Find Python libraries
# On Windows, the executable is typically 'python' not 'python3'
if (WIN32)
    set(_python_cmd python)
else()
    set(_python_cmd python3)
endif()

execute_process(
    COMMAND ${_python_cmd} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    OUTPUT_VARIABLE Python3_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _python_version_result
)

set(Python3_FIND_VIRTUALENV "STANDARD")
if (_python_version_result EQUAL 0 AND Python3_VERSION)
    find_package(Python3 ${Python3_VERSION} EXACT COMPONENTS Interpreter Development Development.SABIModule REQUIRED)
else()
    find_package(Python3 COMPONENTS Interpreter Development Development.SABIModule REQUIRED)
endif()
message(STATUS "Found Python3: ${Python3_FOUND}, at ${Python3_LIBRARIES}")

# On Windows, Python3_add_library() with Py_LIMITED_API expects python3.lib (stable ABI library).
# Many Python distributions (uv, nuget, embedded) only include pythonXY.lib.
# Create a copy of pythonXY.lib as python3.lib if it doesn't exist.
if (WIN32 AND Python3_LIBRARY)
    get_filename_component(_python_lib_dir "${Python3_LIBRARY}" DIRECTORY)
    set(_python3_lib "${_python_lib_dir}/python3.lib")
    if (NOT EXISTS "${_python3_lib}")
        message(STATUS "python3.lib not found, copying from ${Python3_LIBRARY}")
        # Use configure_file for CMake 3.19 compatibility (file(COPY_FILE) requires 3.21)
        configure_file("${Python3_LIBRARY}" "${_python3_lib}" COPYONLY)
        # Verify the copy succeeded
        if (NOT EXISTS "${_python3_lib}")
            message(FATAL_ERROR "Failed to create python3.lib from ${Python3_LIBRARY}. "
                    "Please manually copy ${Python3_LIBRARY} to ${_python3_lib}")
        endif()
    endif()
endif()

# Override Python3_SOABI for Windows ARM64 cross-compilation
# When cross-compiling for ARM64 from x64, the host Python's SOABI is win_amd64
# but we need win_arm64 for the target platform
if (WIN32)
    string(TOUPPER "${CMAKE_GENERATOR_PLATFORM}" _gen_platform_upper)
    string(TOUPPER "${CMAKE_SYSTEM_PROCESSOR}" _sys_proc_upper)
    if (_gen_platform_upper STREQUAL "ARM64" OR _sys_proc_upper MATCHES "ARM64|AARCH64")
        # Replace win_amd64 with win_arm64 in SOABI
        string(REPLACE "win_amd64" "win_arm64" Python3_SOABI "${Python3_SOABI}")
        message(STATUS "Cross-compiling for Windows ARM64, adjusted Python3_SOABI to: ${Python3_SOABI}")
    endif()
endif()

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

if (MSVC)
    # MSVC compiler flags
    # /permissive- enables standard conformance (required for 'not', 'and', 'or' keywords)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
    if(NOT CMAKE_CXX_FLAGS MATCHES "/O[12d]")
        message(NOTICE "No optimization flag found. Setting optimization to /O2.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    endif()
else()
    # GCC/Clang compiler flags
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
endif()

# -------------------------------
# Centralized dependency handling
# -------------------------------

# Find system thread library. Seems to be only needed by DlCompression tests. May be optional - not sure.
find_package(Threads)

if (NOT WIN32)
    find_package(PkgConfig)
endif()
find_package (Eigen3 REQUIRED)

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
    update_onnx_cuda_arch_list()
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
    # Include CUDA directories
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

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

if (ENABLE_TESTS)
    enable_testing()
    add_subdirectory(NightlyTests)
endif()

if (ENABLE_ONNX)
    add_subdirectory(ModelOptimizations)
endif()
add_subdirectory(TrainingExtensions)
add_subdirectory(Examples)
add_subdirectory(Docs)

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks")
  add_subdirectory(Benchmarks)
endif()

#=============================================================================
#
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2018-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
include(FetchContent)

##########
# Patchelf
##########

# michof: TODO: Do we need Patchelf at all? Can't we do the RPATH fiddling with CMake built-in tools,
# such as setting CMAKE_INSTALL_RPATH per-target?

find_program(PATCHELF_EXE patchelf
             PATHS ${CMAKE_BINARY_DIR}/_deps/patchelf-src/bin)

if (PATCHELF_EXE)
    # michof: TODO: Consider removing this path.
    message(STATUS "Patchelf: Found in '${PATCHELF_EXE}'")
elseif (EXISTS $ENV{DEPENDENCY_DATA_PATH}/patchelf.tar.gz)
    # michof: TODO: Needs testing. Move to using FetchContent_Declare as below.
    message(STATUS "Patchelf: Setting up from internal cache")
    file(ARCHIVE_EXTRACT INPUT $ENV{DEPENDENCY_DATA_PATH}/patchelf.tar.gz
    DESTINATION ${CMAKE_BINARY_DIR}/_deps/patchelf-src/)
    set(PATCHELF_EXE ${CMAKE_BINARY_DIR}/_deps/patchelf-src/bin/patchelf)
else()
    # FIXME Better to include patchefl into docker image, although seems it is not trivial

    if (DEFINED ENV{PATCHELF_INTERNAL_URL})
        message(STATUS "Patchelf: Using Internal URL: $ENV{PATCHELF_INTERNAL_URL}")
        FetchContent_Declare(patchelf
        URL "$ENV{PATCHELF_INTERNAL_URL}/patchelf-0.15.0-x86_64.tar.gz"
        )
    else()
        message(NOTICE "Patchelf: Fetching from external URL")
        FetchContent_Declare(patchelf
            URL "https://github.com/NixOS/patchelf/releases/download/0.15.0/patchelf-0.15.0-x86_64.tar.gz"
        )
    endif()

    FetchContent_MakeAvailable(patchelf)
    set(PATCHELF_EXE ${patchelf_SOURCE_DIR}/bin/patchelf)
endif()

message(STATUS "** PATCHELF_EXE = ${PATCHELF_EXE}")

############
# GoogleTest
############

find_program(GOOGLETEST_EXE googletest
             PATHS ${CMAKE_CURRENT_SOURCE_DIR}/google)

if (GOOGLETEST_EXE)
    # michof: TODO: Consider removing this path.
    message(STATUS "GoogleTest: Found in '${GOOGLETEST_EXE}'")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/google EXCLUDE_FROM_ALL)
elseif (EXISTS $ENV{DEPENDENCY_DATA_PATH}/googletest.zip)
    # michof: TODO: this flow needs testing - specifying the version number here as subdirectory is very brittle.
    # Move to using FetchContent_Declare as below.
    message(STATUS "GoogleTest: Setting up from internal cache")
    file(ARCHIVE_EXTRACT INPUT $ENV{DEPENDENCY_DATA_PATH}/googletest.zip
    DESTINATION ${CMAKE_BINARY_DIR}/_deps/googletest-src/)
    add_subdirectory(${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest-release-1.12.1 ${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest-release-1.12.1 EXCLUDE_FROM_ALL)
else ()
    message(NOTICE "GoogleTest: Fetching from external URL")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.1
        EXCLUDE_FROM_ALL
    )

    # michof: We need to revert to this older style of getting the dependency to include the
    # EXCLUDE_FROM_ALL keyword below. This implies that install rules will be ignored, which is
    # intended.
    # CMake 3.28 or later supports adding EXCLUDE_FROM_ALL to the FetchContent_Declare call; then,
    # we can just call FetchContent_MakeAvailable() instead of the below.
    FetchContent_GetProperties(googletest)
    if (NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
endif ()

########
# OpenCV
########

if (NOT OPENCV_FOUND AND NOT OpenCV_FOUND)
  # Aimet requires opencv which might not be installed since it is not part of neither
  # reqs_dep_*.txt nor reqs_pip_*.txt. Download and compile opencv for user only if
  # opencv is not found.
  set(BUILD_LIST        "core" CACHE INTERNAL "[OpenCV] ")
  set(WITH_LAPACK       ON  CACHE INTERNAL "[OpenCV] ")
  set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_EXAMPLES    OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_ITT         OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_JAVA        OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_opencv_apps OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_opencv_python2  OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_opencv_python3  OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_PERF_TESTS  OFF CACHE INTERNAL "[OpenCV] ")
  set(BUILD_TESTS       OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_CUDA         OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_FFMPEG       OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_GTK          OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_IPP          OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_JPEG         OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_OPENEXR      OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_OPENJPEG     OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_PNG          OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_TIFF         OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_V4L          OFF CACHE INTERNAL "[OpenCV] ")
  set(WITH_WEBP         OFF CACHE INTERNAL "[OpenCV] ")

  FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG        4.6.0
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(opencv)
  set(OPENCV_LINK_LIBRARIES opencv_core)
  get_target_property(OPENCV_INCLUDE_DIRS opencv_core INCLUDE_DIRECTORIES)
  set(OPENCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS})
endif()

######################
# ONNX Runtime Headers
######################

if (ENABLE_ONNX)
  # Aimet ONNX extension requires headers for onnxruntime which might not be installed
  # since they are not part of neither reqs_dep_*.txt nor reqs_pip_*.txt.
  # Download and install onnxruntime headers for user only if onnxruntime package is installed
  # and onnxruntime_cxx_api.h can't be found in /opt/onnxruntime
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c" "import onnxruntime; print(onnxruntime.__version__)"
      RESULT_VARIABLE ONNXRUNTIME_NOT_FOUND
      OUTPUT_VARIABLE ONNXRUNTIME_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  file(GLOB_RECURSE ONNXRUNTIME_HEADER_PATH  "/opt/onnxruntime/*onnxruntime_cxx_api.h")

  if (ONNXRUNTIME_NOT_FOUND EQUAL 0)
    if (NOT ONNXRUNTIME_HEADER_PATH)
        message(NOTICE "ONNX Runtime: Fetching from external URL")
        if (ENABLE_CUDA)
            set(ONNXRUNTIME_URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz)
        else (ENABLE_CUDA)
            set(ONNXRUNTIME_URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz)
        endif (ENABLE_CUDA)
        FetchContent_Declare(
          onnxruntime_headers
          URL ${ONNXRUNTIME_URL}
          EXCLUDE_FROM_ALL
        )
        FetchContent_MakeAvailable(onnxruntime_headers)
    else()
        set(onnxruntime_headers_SOURCE_DIR "/opt/onnxruntime")
    endif()
    set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH};${onnxruntime_headers_SOURCE_DIR}/include" PARENT_SCOPE)
  endif()
endif()


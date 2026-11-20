# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
include(FetchContent)

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
        FetchContent_MakeAvailable(googletest)
    endif ()
endif ()

######################
# ONNX Runtime Headers
######################

if (ENABLE_ONNX)
  # The onnxruntime C/C++ headers are vendored in third_party/onnxruntime/include
  # (see set_onnxruntime_variables() in cmake/PrepareONNX.cmake) since the
  # pip-installed onnxruntime package does not ship them and they are no longer
  # fetched from GitHub releases at build time.
  if (NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "onnxruntime headers not found at ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include")
  endif()
endif()


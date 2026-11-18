# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
include(FetchContent)

##########
# Patchelf
##########

# michof: TODO: Do we need Patchelf at all? Can't we do the RPATH fiddling with CMake built-in tools,
# such as setting CMAKE_INSTALL_RPATH per-target?

# Patchelf is only needed on Linux (not Windows or MacOS)
if (NOT WIN32 AND NOT APPLE)
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
endif()

######################
# install_name_tool
######################

# install_name_tool is only needed on macOS
if (APPLE)
    find_program(INSTALL_NAME_TOOL_EXE install_name_tool)

    if (INSTALL_NAME_TOOL_EXE)
        message(STATUS "install_name_tool: Found in '${INSTALL_NAME_TOOL_EXE}'")
    else()
        message(FATAL_ERROR "install_name_tool not found. Please install Xcode command line tools using: xcode-select --install")
    endif()

    message(STATUS "** INSTALL_NAME_TOOL_EXE = ${INSTALL_NAME_TOOL_EXE}")
endif()

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


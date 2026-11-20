# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

if (SKBUILD)
    add_subdirectory(src/python)
else (SKBUILD)
    add_subdirectory(src/python)

    if (ENABLE_TESTS)
        if (ENABLE_ONNX)
            # Skip PyTorch tests when building for ONNX.
            # NB, some of the PyTorch tests fail when executed in ONNX build.
            # michof: We should consider enabling these tests and making sure they succeed, as long as
            # we depend on PyTorch for ONNX build and as long as test Docker images are materially
            # different.
            message(NOTICE "**********************************************")
            message(NOTICE "ONNX build is enabled, skipping PyTorch tests.")
            message(NOTICE "**********************************************")
        else()
            add_subdirectory(test)
        endif()
    endif()
endif (SKBUILD)

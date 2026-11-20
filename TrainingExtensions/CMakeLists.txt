# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_subdirectory(common)

if (ENABLE_TORCH)
    add_subdirectory(torch)
endif(ENABLE_TORCH)

if (ENABLE_ONNX)
    add_subdirectory(onnx)
endif(ENABLE_ONNX)

#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2018, Qualcomm Innovation Center, Inc. All rights reserved.
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
#==============================================================================

add_subdirectory(src/python)

if (ENABLE_TESTS)
    add_subdirectory(test)
endif()

add_custom_target(whl_prep_cp_common DEPENDS
    whl_prep_cp_common_DlCompression
    whl_prep_cp_common_DlEqualization
    whl_prep_cp_common_DlQuantization
    whl_prep_cp_common_PyModelOptimizations
)
add_dependencies(whl_prep_cp whl_prep_cp_common)

add_custom_target(whl_prep_ln_common DEPENDS
    whl_prep_ln_common_DlCompression
    whl_prep_ln_common_DlEqualization
    whl_prep_ln_common_DlQuantization
    whl_prep_ln_common_PyModelOptimizations
)

add_custom_target(generate_deps
    COMMAND python3
            "${CMAKE_CURRENT_SOURCE_DIR}/src/python/aimet_common/_gen.py"
            --output-dir ${CMAKE_BINARY_DIR}/artifacts/aimet_common
)
if(SKBUILD)
install(DIRECTORY ${CMAKE_BINARY_DIR}/artifacts/aimet_common/
        DESTINATION aimet_common
        FILES_MATCHING
        PATTERN "_deps.py"
      )
else()
install(DIRECTORY ${CMAKE_BINARY_DIR}/artifacts/aimet_common/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/aimet_common
        FILES_MATCHING
        PATTERN "_deps.py"
      )
endif()

add_dependencies(whl_prep_ln whl_prep_ln_common)
whl_add_whl_action_target(common)

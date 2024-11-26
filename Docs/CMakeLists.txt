#=============================================================================
#
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
#
#=============================================================================

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DESTINATION .
      PATTERN "CMakeFiles*" EXCLUDE
      PATTERN "Makefile" EXCLUDE
      PATTERN "*.cmake" EXCLUDE
      )

add_custom_target(doc
      COMMAND ${CMAKE_COMMAND} -E env
      "${AIMET_PYTHONPATH}"
      "SW_VERSION=${SW_VERSION}"
      sphinx-build -v -T -b html ${CMAKE_BINARY_DIR}/Docs_SOURCE ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(doc
    copy_doc_source copy_examples)

add_custom_target(copy_doc_source
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/Docs_SOURCE)

add_custom_target(doc_beta
      COMMAND ${CMAKE_COMMAND} -E env
      "${AIMET_PYTHONPATH}"
      "SW_VERSION=${SW_VERSION}"
      sphinx-build -v -T -b html ${CMAKE_BINARY_DIR}/Docs_SOURCE/beta ${CMAKE_CURRENT_BINARY_DIR}/beta -c ${CMAKE_BINARY_DIR}/Docs_SOURCE/beta)

add_dependencies(doc_beta
    copy_doc_source copy_examples_beta_docs)

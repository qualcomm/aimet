# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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
# =============================================================================

# CMake file to generate AIMET packages

set(src_packaging_dir "${SOURCE_DIR}/packaging")
set(src_deps_dir "${src_packaging_dir}/dependencies")
set(build_packaging_dir "${CMAKE_BINARY_DIR}/packaging")
set(package_common "aimet_common")

# First delete the existing packaging directory if it exists
file(REMOVE_RECURSE ${build_packaging_dir})

if(DEFINED ENV{FAST_RELEASE})
  set(src_deps_dir "${src_packaging_dir}/dependencies/fast-release")
endif()

# set variant name
if(DEFINED ENV{AIMET_VARIANT})
  set(variant_name $ENV{AIMET_VARIANT})
else()
  if(ENABLE_TORCH AND ENABLE_TENSORFLOW)
    set(variant_name "tf-torch")
  elseif(ENABLE_ONNX)
    set(variant_name "onnx")
  elseif(ENABLE_TORCH)
    set(variant_name "torch")
  elseif(ENABLE_TENSORFLOW)
    set(variant_name "tf")
  else()
    set(variant_name "tf-torch")
  endif()

  if(ENABLE_CUDA)
    set(variant_name ${variant_name}-gpu)
  else()
    set(variant_name ${variant_name}-cpu)
  endif()
endif()

# Common dependencies
set(deps_name_list_aimet_common "reqs_deb_common.txt" "reqs_pip_common.txt")

# Initialize empty package array list
set(package_name_list "")

# Set a GPU flag for use by setup scripts
set(CUDA_OPTION "")
if(ENABLE_CUDA)
  set(CUDA_OPTION "--gpu")
endif()

# Setup Tensorflow package dependencies if required
if(ENABLE_TENSORFLOW)
  # Add AIMET Tensorflow package to package array list
  list(APPEND package_name_list tensorflow)

  # Initialize TF deps list with AIMET Common dependencies
  set(deps_name_list_tensorflow ${deps_name_list_aimet_common})

  # Tensorflow dependencies that are common to CPU and GPU
  list(APPEND deps_name_list_tensorflow "reqs_pip_tf_common.txt")

  if(ENABLE_CUDA)
    # Tensorflow GPU dependencies
    list(APPEND deps_name_list_tensorflow "reqs_deb_tf_gpu.txt" "reqs_pip_tf_gpu.txt")
  else()
    # Tensorflow CPU dependencies
    list(APPEND deps_name_list_tensorflow "reqs_pip_tf_cpu.txt")
  endif()
endif()

# Setup Torch package dependencies if required
if(ENABLE_TORCH)
  # Add AIMET Torch package to package array list
  list(APPEND package_name_list torch)

  # Initialize Torch deps list with AIMET Common dependencies
  set(deps_name_list_torch ${deps_name_list_aimet_common})

  # Torch dependencies that are common to CPU and GPU
  list(APPEND deps_name_list_torch "reqs_pip_torch_common.txt")

  if(ENABLE_CUDA)
    # Torch GPU dependencies
    list(APPEND deps_name_list_torch "reqs_deb_torch_gpu.txt" "reqs_pip_torch_gpu.txt")
  else()
    # Torch CPU dependencies
    list(APPEND deps_name_list_torch "reqs_pip_torch_cpu.txt")
  endif()
endif()

# Setup Onnx package dependencies if required
if(ENABLE_ONNX)
  # Add AIMET Onnx package to package array list
  list(APPEND package_name_list onnx)

  # Initialize ONNX deps list with AIMET Common dependencies
  set(deps_name_list_onnx ${deps_name_list_aimet_common})

  # Onnx dependencies that are common to CPU and GPU
  list(APPEND deps_name_list_onnx "reqs_pip_onnx_common.txt")

  if(ENABLE_CUDA)
    # Onnx GPU dependencies
    list(APPEND deps_name_list_onnx "reqs_deb_onnx_gpu.txt" "reqs_pip_onnx_gpu.txt")
  else()
    # Onnx CPU dependencies
    list(APPEND deps_name_list_onnx "reqs_pip_onnx_cpu.txt")
  endif()
endif()


# Pull just the version from the string (ex. 1.13+c116 --> 1.13)
execute_process(COMMAND bash -c "echo ${FMWORK_VERSION} | tr -s ' ' | cut -d\"+\" -f1"
                OUTPUT_VARIABLE fmwork_ver_bare
                OUTPUT_STRIP_TRAILING_WHITESPACE)


# Loop over the package array list to generate wheel files
foreach(package ${package_name_list})
  set(pkg_staging_path "${build_packaging_dir}/${package}")
  set(pkg_common_staging_path "${pkg_staging_path}/${package_common}")
  set(pkg_variant_staging_path "${pkg_staging_path}/aimet_${package}")
  set(pkg_deps_staging_path "${pkg_variant_staging_path}/bin")

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory "${pkg_staging_path}" "${pkg_common_staging_path}"
      "${pkg_variant_staging_path}" "${pkg_common_staging_path}/x86_64-linux-gnu"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${AIMET_PACKAGE_PATH}/lib/python/${package_common}" "${pkg_common_staging_path}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${AIMET_PACKAGE_PATH}/lib/python/aimet_${package}" "${pkg_variant_staging_path}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${AIMET_PACKAGE_PATH}/lib/x86_64-linux-gnu" "${pkg_common_staging_path}/x86_64-linux-gnu"
    COMMAND ${CMAKE_COMMAND} -E copy "${src_packaging_dir}/NOTICE.txt" "${pkg_staging_path}/"
    COMMAND ${CMAKE_COMMAND} -E copy "${src_packaging_dir}/README.txt" "${pkg_staging_path}/"
    )
  execute_process(
    # Delete binaries from aimet_common which should not be part of the package
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${pkg_deps_staging_path}"
  )
  # convert file names to the absolute paths by prepending corresponded directory
  list(TRANSFORM deps_name_list_${package} PREPEND "${src_deps_dir}/${variant_name}/")

  # Copy the dependency and other files into the staging subfolder
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${pkg_deps_staging_path}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy 
      ${deps_name_list_${package}}
      "${src_packaging_dir}/INSTALL.txt"
      "${src_packaging_dir}/envsetup.sh"
      "${pkg_deps_staging_path}/"
  )

  if(EXISTS "${src_packaging_dir}/LICENSE.pdf")
    # optional, might not be present on host
    file(COPY "${src_packaging_dir}/LICENSE.pdf" DESTINATION ${build_packaging_dir}/)
    file(COPY "${src_packaging_dir}/LICENSE.pdf" DESTINATION ${pkg_deps_staging_path}/)
  endif()

  configure_file("${CMAKE_CURRENT_LIST_DIR}/pypi_readme.md.in" "${pkg_staging_path}/pypi_readme_${package}.md" @ONLY)
  configure_file("${CMAKE_CURRENT_LIST_DIR}/setup.py.in" "${pkg_staging_path}/setup-${package}.py" @ONLY)
  configure_file(${CMAKE_CURRENT_LIST_DIR}/MANIFEST.in.in ${pkg_staging_path}/MANIFEST.in @ONLY)

  # Update RPATH to relative paths ($ORIGIN) to not bother by a path where python is installed
  # Linux loader would be able to resolve dependencies without LD_LIBRARY_PATH
  execute_process(
    COMMAND find ${pkg_staging_path} -name "Aimet*.so" -exec ${PATCHELF_EXE} --set-rpath $ORIGIN:$ORIGIN/../torch/lib {} \;
  )
  execute_process(
    COMMAND find ${pkg_staging_path} -name "libaimet_tf_ops*.so" -exec ${PATCHELF_EXE} --set-rpath $ORIGIN:$ORIGIN/../../tensorflow:$ORIGIN/../../tensorflow/python {} \;
  )

  # Invoke the setup tools script to create the wheel packages.
  message(VERBOSE "*** Running setup script for package ${package} ***")
  execute_process(
    COMMAND ${PYTHON3_EXECUTABLE} setup-${package}.py sdist bdist_wheel --dist-dir ${build_packaging_dir}/dist ${CUDA_OPTION}
    WORKING_DIRECTORY ${pkg_staging_path}
    OUTPUT_VARIABLE output_var
    RESULT_VARIABLE setup_return_string
  )
endforeach()

# Rename the package wheel files to conform to manylinux format for PyPi
message(STATUS "Package rename command: ${rename_package_cmd}")
execute_process(COMMAND bash "-c" "${src_packaging_dir}/convert_wheel_format.sh ${CMAKE_BINARY_DIR}"
                OUTPUT_VARIABLE convert_wheel_output
                RESULT_VARIABLE convert_wheel_result
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(convert_wheel_result EQUAL "1")
  message( FATAL_ERROR "Wheel package conversion failed")
endif()

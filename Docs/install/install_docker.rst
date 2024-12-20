.. # =============================================================================
   #  @@-COPYRIGHT-START-@@
   #
   #  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

.. _installation-docker:

##############################
AIMET Installation in Docker
##############################

This page provides instructions to install AIMET package inside a development docker container.

Set variant
~~~~~~~~~~~
Set the `<variant_string>` to ONE of the following depending on your desired variant
    #. For the PyTorch 2.1 GPU variant, use `torch-gpu`
    #. For the PyTorch 2.1 CPU variant, use `torch-cpu`
    #. For the PyTorch 1.13 GPU variant, use `torch-gpu-pt113`
    #. For the TensorFlow GPU variant, use `tf-gpu`
    #. For the TensorFlow CPU variant, use `tf-cpu`
    #. For the ONNX GPU variant, use `onnx-gpu`
    #. For the ONNX CPU variant, use `onnx-cpu`

.. code-block:: bash

    export AIMET_VARIANT=<variant_string>


Use prebuilt docker image
~~~~~~~~~~~~~~~~~~~~~~~~~
Follow these instructions to use one of the pre-built docker images:

.. code-block:: bash

    WORKSPACE="<absolute_path_to_workspace>"
    docker_image_name="artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:latest.${AIMET_VARIANT}"
    docker_container_name="aimet-dev-<any_name>"

**NOTE:** Feel free to modify the `docker_container_name` as needed.

Build docker image locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Follow these instructions ONLY if you want to build the docker image locally. If not, skip to the next section.

.. code-block:: bash

    WORKSPACE="<absolute_path_to_workspace>"
    docker_image_name="aimet-dev-docker:<any_tag>"
    docker_container_name="aimet-dev-<any_name>"
    docker build -t ${docker_image_name} -f $WORKSPACE/aimet/Jenkins/Dockerfile.${AIMET_VARIANT} .

**NOTE:** Feel free to modify the `docker_image_name` and `docker_container_name` as needed.

Start docker container
~~~~~~~~~~~~~~~~~~~~~~~
Ensure that a docker named `$docker_container_name` is not already running; otherwise remove the existing container and then start a new container as follows:

.. code-block:: bash

    docker ps -a | grep ${docker_container_name} && docker kill ${docker_container_name}

    docker run --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
    -v "/local/mnt/workspace":"/local/mnt/workspace" \
    --entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name}


**NOTE:**
    #. Feel free to modify the above `docker run` command based on the environment and filesystem on your host machine.
    #. If nvidia-docker 2.0 is installed, then add `--gpus all` to the `docker run` commands in order to enable GPU access inside the docker container.
    #. If nvidia-docker 1.0 is installed, then replace `docker run` with `nvidia-docker run` in order to enable GPU access inside the docker container.
    #. Port forwarding needs to be done in order to run the Visualization APIs from docker container. This can be achieved by running the docker container as follows:

.. code-block:: bash

    port_id="<any-port-number>"

    docker run -p ${port_id}:${port_id} --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
    -v "/local/mnt/workspace":"/local/mnt/workspace" \
    --entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name}

Install AIMET packages
~~~~~~~~~~~~~~~~~~~~~~~

From PyPI
=========

The default AIMET Torch GPU variant may be installed from PyPI as follows:
    - Go to https://pypi.org/project/aimet-torch
    - Browse the Requirements section of each Release to identify the version you wish to install. Following are some tips:
        - For Pytorch 2.2.2 GPU with CUDA 12.1, use aimet-torch>=1.32.2
        - For Pytorch 2.1.2 GPU with CUDA 12.1, use aimet-torch==1.32.1.post1
        - For PyTorch 1.13 GPU with CUDA 11.7, use aimet-torch==1.31.2

Run the following commands to install the package (prepend with "sudo" and/or package version as needed):

.. code-block:: bash

    apt-get install liblapacke -y
    python3 -m pip install aimet-torch

From Release Package
====================

We also host python wheel packages for different variants which may be installed as follows:
    - Go to https://github.com/quic/aimet/releases
    - Identify the release tag of the package that you wish to install
    - Identify the .whl file corresponding to the package variant that you wish to install
    - Follow the instructions below to install AIMET from the .whl file

Set the package details as follows:

.. code-block:: bash

    # Set the release tag ex. "1.34.0"
    export release_tag="<version release tag>"

    # Construct the download root URL
    export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"

    # Set the wheel file name with extension
    # ex. "aimet_torch-1.34.0.cu121-cp310-cp310-manylinux_2_34_x86_64.whl"
    export wheel_file_name="<wheel file name>"

    # NOTE: Do the following ONLY for the PyTorch and ONNX variant packages!
    export find_pkg_url_str="-f https://download.pytorch.org/whl/torch_stable.html"

Install the selected AIMET package as specified below:

**NOTE:** Python dependencies will automatically get installed.

.. code-block:: bash

    # Install the wheel package
    python3 -m pip install ${download_url}/${wheel_file_name} ${find_pkg_url_str}

Environment setup
~~~~~~~~~~~~~~~~~

Set the common environment variables as follows:

.. code-block:: bash

    source /usr/local/lib/python3.10/dist-packages/aimet_common/bin/envsetup.sh

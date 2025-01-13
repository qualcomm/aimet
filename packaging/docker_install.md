# AIMET Installation and Usage in Docker
This page provides instructions to build, install and use the AIMET software in docker. Please follow the instructions in the order provided, unless specified otherwise.

- [Requirements](#requirements)
- [Get the code](#get-the-code)
- [Setup the environment](#setup-the-environment)
  - [Build docker image](#build-docker-image)
  - [Start docker container](#start-docker-container)
- [Build code and install](#build-code-and-install)
- [Set package and library paths](#set-package-and-library-paths)
- [Usage examples and documentation](#usage-examples-and-documentation)
- [Docker information](#docker-information)
  - [Set variant](#set-variant)
  - [Use prebuilt docker image](#use-prebuilt-docker-image)
  - [Build docker image locally](#build-docker-image-locally)
  - [Start docker container manually](#start-docker-container-manually)
  - [Build and launch docker using script](#build-and-launch-docker-using-script)

## Requirements
The AIMET package requires the following host platform setup:
- 64-bit Intel x86-compatible processor
- Linux Ubuntu: 22.04 LTS
- bash command shell
- For GPU variants:
  - Nvidia GPU card (Compute capability 5.2 or later)
  - nvidia-docker - Installation instructions: https://github.com/NVIDIA/nvidia-docker

To use the GPU accelerated training modules an Nvidia CUDA enabled GPU with a minimum Nvidia driver version of 455+ is required. Using the latest driver is always recommended, especially if using a newer GPU. Both CUDA and cuDNN (the more advanced CUDA interface) enabled GPUs are supported.

Recommended host system hardware requirements:
- Intel i7 multicore CPU w/hyperthreading
- 16+ GB RAM
- 500GB+ SSD hard drive
- For GPU variants:
  - GPU: Nvidia GeForce GTX 1080 or Tesla V100

While these are not minimum requirements, they are recommended for good performance when training large networks.

## Get the code
To obtain the code, first define a workspace and follow these instructions:

```bash
WORKSPACE="<absolute_path_to_workspace>"
mkdir $WORKSPACE && cd $WORKSPACE
git clone https://github.com/quic/aimet.git
```
Clone the google test repo as follows:
```
cd aimet
mkdir -p ./ThirdParty/googletest
pushd ./ThirdParty/googletest
git clone https://github.com/google/googletest.git -b release-1.12.1  googletest-release-1.12.1
popd
```

## Setup the environment
In order to build and run AIMET code, several dependencies are required (such as python, cmake, tensorflow, pytorch, onnx, etc). [Docker files](../Jenkins) and [Docker images](https://artifacts.codelinaro.org/ui/native/codelinaro-aimet/aimet-dev) with all prerequisites and dependencies are available for each AIMET variant. Following are the available development options:
- Use the appropriate [pre-built Docker image](https://artifacts.codelinaro.org/ui/native/codelinaro-aimet/aimet-dev) using the instructions [here](#docker-information). This is the *recommended* option.
- Build the docker image locally and launch a launch container docker using the instructions [here](#docker-information).
- Install the dependencies on your machine and setup your environment using [the appropriate Dockerfile](../Jenkins) as a guide.

Set the *common* environment variables as follows:
```bash
source $WORKSPACE/aimet/packaging/envsetup.sh
```

## Build code and install
Follow these instructions to build the AIMET code:

> NOTE: **If you are inside the docker, set `WORKSPACE="<absolute_path_to_workspace>"` again.**
```bash
cd $WORKSPACE/aimet
mkdir build && cd build

# Run cmake (be sure to set the flags in the below command depending on your variant)
# To build for GPU, use -DENABLE_CUDA=ON. To build for CPU, use -DENABLE_CUDA=OFF.
# To include torch, use -DENABLE_TORCH=ON. To exclude torch, use -DENABLE_TORCH=OFF.
# To include tensorflow, use -DENABLE_TENSORFLOW=ON. To exclude tensorflow, use -DENABLE_TENSORFLOW=OFF.
# To include onnx, use -DENABLE_ONNX=ON. To exclude onnx, use -DENABLE_ONNX=OFF.
# Add a installation location -DCMAKE_INSTALL_PREFIX="<install_path>"
# Following is an example command for the Torch GPU variant with installation within the current build path. Please modify it as appropriate.
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_CUDA=ON -DENABLE_TORCH=ON -DENABLE_TENSORFLOW=OFF -DENABLE_ONNX=OFF -DCMAKE_INSTALL_PREFIX="./staging/universal"

make -j8
```

After a successful build, install the package using the following instructions:
```bash
cd $WORKSPACE/aimet/build
make install
```
Once the installation step is complete, the AIMET package is created at `$WORKSPACE/aimet/build/staging/universal/lib/`.

## Setup paths
Setup the package and library paths as follows:
```bash
export PYTHONPATH=$WORKSPACE/aimet/build/staging/universal/lib/python:$PYTHONPATH
```
At this point, we are all set to use AIMET!

## Usage examples and documentation
The following steps would generate AIMET documentation including the user guide, examples and API documentation at `$WORKSPACE/aimet/build/staging/universal/Docs`:

```bash
cd $WORKSPACE/aimet/build
make doc
```

To begin navigating the documentation, open the page `$WORKSPACE/aimet/build/staging/universal/Docs/user_guide/index.html` on any browser.

## Generate wheel packages
Run the following commands to generate installable python wheel packages:
```bash
cd $WORKSPACE/aimet/build
make packageaimet
```

## Docker information
Code may *optionally* be developed inside a development docker container. This section describes how to build a docker image and launch a container using the provided [Dockerfiles](../Jenkins).

### Set variant
Set the `<variant_string>` to ONE of the following depending on your desired variant
- For the PyTorch 2.1 GPU variant, use `"torch-gpu"`
- For the PyTorch 2.1 CPU variant, use `"torch-cpu"`
- For the TensorFlow GPU variant, use `"tf-gpu"`
- For the TensorFlow CPU variant, use `"tf-cpu"`
- For the ONNX GPU variant, use `"onnx-gpu"`
- For the ONNX CPU variant, use `"onnx-cpu"`
```bash
export AIMET_VARIANT=<variant_string>
```

### Use prebuilt docker image
Follow these instructions to use one of the pre-built docker images:
```bash
WORKSPACE="<absolute_path_to_workspace>"
docker_image_name="artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:latest.${AIMET_VARIANT}"
docker_container_name="aimet-dev-<any_name>"
```

> NOTE: Feel free to modify the `docker_container_name` as needed.

### Build docker image locally
Follow these instructions ONLY if you want to build the docker image locally. If not, skip to the next section.
```bash
WORKSPACE="<absolute_path_to_workspace>"
docker_image_name="aimet-dev-docker:<any_tag>"
docker_container_name="aimet-dev-<any_name>"
docker build -t ${docker_image_name} -f $WORKSPACE/aimet/Jenkins/Dockerfile.${AIMET_VARIANT} .
```

> NOTE: Feel free to modify the `docker_image_name` and `docker_container_name` as needed.

### Start docker container manually
Ensure that a docker named `$docker_container_name` is not already running; otherwise remove the existing container and then start a new container as follows:
```bash
docker ps -a | grep ${docker_container_name} && docker kill ${docker_container_name}

docker run --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
  -v "/local/mnt/workspace":"/local/mnt/workspace" \
  --entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name}
```

> **NOTE**
* Feel free to modify the above `docker run` command based on the environment and filesystem on your host machine.
* If nvidia-docker 2.0 is installed, then add `--gpus all` to the `docker run` commands in order to enable GPU access inside the docker container.
* If nvidia-docker 1.0 is installed, then replace `docker run` with `nvidia-docker run` in order to enable GPU access inside the docker container.
* Port forwarding needs to be done in order to run the Visualization APIs from docker container. This can be achieved by running the docker container as follows:

```bash
port_id="<any-port-number>"

docker run -p ${port_id}:${port_id} --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
  -v "/local/mnt/workspace":"/local/mnt/workspace" \
  --entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name}
```

### Build and launch docker using script
The development docker may also be built and launched in interactive mode using the provided script as follows:

> **_NOTE:_** Add the '-l' option to use the pre-built docker image.

```
cd aimet
./buildntest.sh -e AIMET_VARIANT -i [-l]
```
If additional directories need to be mounted, use `-m` option with list of targeted directories separated by space **surrounded by double quotes `""`**
```
cd aimet
./buildntest.sh -e AIMET_VARIANT -i -m "sample_dir_1 sample_dir2" [-l]
```

To help construct user-specific docker commands, the dry-run option (`-n`) can be used with the above script which prints out the equivalent docker command(s):
```
cd aimet
./buildntest.sh -e AIMET_VARIANT -i -n [-l]
# OR
./buildntest.sh -e AIMET_VARIANT -i -n -m "sample_dir_1 sample_dir2" [-l]
```

# AIMET 2.0
## Get the code
To obtain the code, first define a workspace and follow these instructions:

```bash
git clone https://github.com/quic/aimet.git
cd aimet
```
## Setup the environment
In order to build and run AIMET code, several dependencies are required. The [docker file](../Jenkins/fast-release/Dockerfile.ci) provides a docker image with all installed dependencies. To buil a docker image the regular `docker build` commnad should be used. Depends on the desired variant extda arguments should be provided to `docker build` command to specify version of python, cuda, torch, onnx, tensorflow. For example:
```bash
docker build --tag aimet --build-arg VER_PYTHON=3.10 --build-arg VER_CUDA=12.1.1 --build-arg VER_TORCH=2.1.2 --file  Jenkins/fast-release/Dockerfile.ci .
```
To build AIMET:
```bash
docker run --rm --gpus all -v $PWD:$PWD -w $PWD aimet bash -c '. /etc/profile.d/conda.sh ; CMAKE_ARGS='-DENABLE_CUDA=ON -DENABLE_TORCH=ON -DENABLE_ONNX=OFF -DENABLE_TENSORFLOW=OFF' python3 -m build --no-isolation .'
```
To install AIMET:
```bash
docker run --rm --gpus all -v $PWD:$PWD -w $PWD aimet bash -c '. /etc/profile.d/conda.sh ; CMAKE_ARGS='-DENABLE_CUDA=ON -DENABLE_TORCH=ON -DENABLE_ONNX=OFF -DENABLE_TENSORFLOW=OFF' python3 -m pip install --no-build-isolation -e .'
```
You can also build and/or install AIMET in your own environment. You should have installed cuda and C++ compiler, libeigen3, ninja and pip-tool (to install all dependencies required to build selected AIMET variant):
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 bash -c '\
    export CMAKE_ARGS="-DENABLE_CUDA=ON -DENABLE_TORCH=ON -DENABLE_ONNX=OFF -DENABLE_TENSORFLOW=OFF" ; \
    apt update -qq && apt install -y git g++ libeigen3-dev ninja-build python3-pip &>/dev/null && \
    python3 -m pip install pip-tools && \
    git clone https://github.com/quic/aimet.git && \
    python3 -m piptools compile --resolver=backtracking --extra=.,dev,test  --output-file=- aimet/pyproject.toml | python3 -m pip install -r /dev/stdin && \
    python3 -m pip install --no-build-isolation -e ./aimet
    python3 -m pytest ./aimet/TrainingExtensions/{common,torch}/test/python
'
```

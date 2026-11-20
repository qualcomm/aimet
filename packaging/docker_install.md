# AIMET Development in Docker
This page describes how to build AIMET from source inside a Docker container. To build directly on
your machine instead, see [Building from source](../Docs/overview/install/build_from_source.rst).

- [Requirements](#requirements)
- [Setup the environment](#setup-the-environment)
  - [Get the code](#get-the-code)
  - [Build the docker image](#build-the-docker-image)
  - [Start a docker container](#start-a-docker-container)
- [Build and test AIMET](#build-and-test-aimet)

## Requirements
- 64-bit x86 Linux host (Ubuntu 22.04 LTS recommended) with bash
- [Docker](https://docs.docker.com/engine/install/)
- For GPU builds:
  - NVIDIA GPU and a recent NVIDIA driver
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Setup the environment

### Get the code
```bash
WORKSPACE="<absolute_path_to_workspace>"
mkdir -p $WORKSPACE && cd $WORKSPACE
git clone https://github.com/qualcomm/aimet.git
cd aimet
```

### Build the docker image
[`Jenkins/fast-release/Dockerfile.ci`](../Jenkins/fast-release/Dockerfile.ci) builds an image with
all build and test dependencies pre-installed in a virtual environment at `/opt/venv`. Select the
variant with build arguments:

| Build arg         | Meaning                                                           |
|-------------------|-------------------------------------------------------------------|
| `VER_PYTHON`      | Python version (default `3.10`)                                   |
| `VER_CUDA`        | CUDA toolkit version, e.g. `13.2.0`. Leave unset for a CPU image  |
| `VER_TORCH`       | Set (e.g. `2.*`) to include PyTorch                               |
| `VER_ONNXRUNTIME` | Set (e.g. `1.19.2`) to include ONNX Runtime                       |

Run the command from the repository root. For example, for the torch GPU variant:
```bash
docker_image_name="aimet-dev:torch-gpu"
docker build --tag ${docker_image_name} \
  --build-arg VER_PYTHON=3.12 \
  --build-arg VER_CUDA=13.2.0 \
  --build-arg VER_TORCH="2.*" \
  --file Jenkins/fast-release/Dockerfile.ci .
```

### Start a docker container
`--pull=never` makes sure Docker only uses the image you built locally and never pulls an image
with the same name from a public registry.
```bash
docker run --pull=never --rm -it --gpus all \
  -v ${WORKSPACE}:${WORKSPACE} -w ${WORKSPACE}/aimet \
  ${docker_image_name}
```

> **NOTE**
> * Drop `--gpus all` for CPU images.
> * The container runs as root, because the image's tools (e.g. `uv`) are installed for root. Files
>   it creates in your workspace will be owned by root.
> * Add more `-v` mounts (e.g. `-v ${HOME}:${HOME}`) or port forwarding (`-p <port>:<port>`) as needed.

## Build and test AIMET
Inside the container, use [`scripts/all/build_and_test.py`](../scripts/all/build_and_test.py) with
the pre-installed virtual environment:
```bash
# Build the aimet-torch GPU wheel (output: wheelhouse/)
python3 scripts/all/build_and_test.py --task build --package aimet_torch --gpu --venv /opt/venv

# Build (if needed) and run the aimet-torch unit tests
python3 scripts/all/build_and_test.py --task test --package aimet_torch --gpu --venv /opt/venv

# List all available tasks
python3 scripts/all/build_and_test.py --task list
```

Use `--package aimet_onnx` for ONNX images, and drop `--gpu` for CPU images. See
[Building from source](../Docs/overview/install/build_from_source.rst) for all options.

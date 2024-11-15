# =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
# =============================================================================
"""
This script is an example of how PyTorch lightning can be used for DDP(distributed data parallel) with AIMET for training
For this example we use a MV2 model and perform QAT with range learning.
The steps are as follows:
    1) Create a Quantization Sim model with the required arguments. A user can refer to the
      example code in qat_range_learning.ipynb for explanation on how to create & compute encodings for QAT range learning.
      Note: This step is done on a single GPU
    2) Use DDP for evaluating the model
    3) Save checkpoint to use with DDP training
"""

# pylint: skip-file

import os
import socket
import argparse

from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchmetrics

from aimet_common.defs import QuantScheme
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch import quantsim
from aimet_torch import batch_norm_fold
from aimet_torch.model_preparer import prepare_model


def find_free_network_port() -> int:
    """Finds a free port on localhost.
    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def dist_eval_func(model, imagenet_dir, batch_size, world_size):
    """
    Spawns multiple processes in order to evaluate the model passed to it
    :param model: Quantization Sim model
    :param imagenet_dir: Directory path for images
    :param batch_size: Batch size used for eval
    :param world_size: Number of processes spawned based on number of GPUs and nodes available
    """
    manager = mp.Manager()
    res = manager.dict()
    port_id = str(find_free_network_port())
    # The evaluate_ddp function gets copied to each of the process and runs, torchmetrics helps combine the
    # results on each process
    mp.spawn(evaluate_ddp, args=(world_size, port_id, model, imagenet_dir, batch_size, res), nprocs=world_size, join=True)
    return res


def get_quant_eval(imagenet_dir, batch_size, device):
    """
    Forward pass callback
    :param imagenet_dir: Directory path for images
    :param batch_size: Batch size used for eval
    :param device: Which device (Cuda/CPU) model is on
    """
    def evaluate_quant(model, _):
        val_dir = os.path.join(imagenet_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_set = datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=1,
                                                 pin_memory=True)

        model.to(device)
        model.eval()

        with torch.no_grad():
            for i, (images, _) in tqdm(enumerate(val_loader), total=100):
                images = images.to(device)
                # compute output
                _ = model(images)
                if i == 100:
                    break

    return evaluate_quant


def evaluate_ddp(rank, world_size, port_id, model, imagenet_dir, batch_size, results):
    """
    Evaluates model
    :param rank: Current process number
    :param world_size: Number of processes
    :param port_id: The port on which this process runs
    :param model: QuantizationSim model
    :param imagenet_dir: Directory path for images
    :param batch_size: Batch size used for eval
    :param results: Dict that collects results from all processes
    """
    # pylint: disable=too-many-locals

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_id

    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Initializations
    metric = torchmetrics.Accuracy()
    num_workers = 1

    # Get validation data
    val_dir = os.path.join(imagenet_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_set = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # Used so that each process sees a different data
    val_sampler = DistributedSampler(dataset=val_set)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             sampler=val_sampler,
                                             pin_memory=True)

    model.metric = metric

    # Move the model to corresponding process
    model = model.to(rank)

    # Create an instance of DDP in order to perform distributed eval
    model = DDP(model, device_ids=[rank])
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(rank, non_blocking=True)
            target = target.to(rank, non_blocking=True)

            # compute output
            output = model(images)
            acc = metric(output, target)

            print_freq = 10
            if rank in [0, 1] and i % print_freq == 0:  # print only for rank 0
                print(f"Accuracy on batch {i}: {acc} - rank {rank}")

        # metric on all batches and all accelerators using custom accumulation
        # accuracy is same across both accelerators
        acc = metric.compute()
        print(f"Accuracy on all data: {acc}, accelerator rank: {rank}")

        # Reseting internal state such that metric ready for new data
        metric.reset()

    if rank == 0:
        results['top-1 acc'] = float(acc)
    # cleanup
    dist.destroy_process_group()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='PyTorch DDP Eval')
    parser.add_argument('--world_size', default=2, type=int, help="number of total nodes")
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N')
    parser.add_argument(
        '--model_path',
        help="path to the quantized model's saved checkpoint for QAT",
        default='na'
    )
    parser.add_argument('--imagenet_dir', help="path to imagenet_dir", required=True)
    parser.add_argument('--output_file', help="path to quantsim output file", required=True)
    args = parser.parse_args()

    # STEP 1
    # We instantiate a MV2 model
    model = torchvision.models.mobilenet_v2(pretrained=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    forward_pass_callback = get_quant_eval(args.imagenet_dir, args.batch_size, device)

    # Perform AIMET Quantization
    prepared_model = prepare_model(model)
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape)
    _ = batch_norm_fold.fold_all_batch_norms(prepared_model, input_shape)

    # Note: As of now only range learning quantization schemes as supported
    quant_sim = QuantizationSimModel(
        prepared_model,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.training_range_learning_with_tf_init,
        default_param_bw=8,
        default_output_bw=8
    )

    # Compute Encodings
    quant_sim.compute_encodings(forward_pass_callback, forward_pass_callback_args=None)
    print("Finished Compute Encodings")

    # STEP 2
    dist_eval_func(quant_sim.model.cpu(), args.imagenet_dir, args.batch_size, args.world_size)

    # STEP 3
    quant_sim.model.to('cpu')
    quantsim.save_checkpoint(quant_sim, args.output_file)


if __name__ == "__main__":
    main()

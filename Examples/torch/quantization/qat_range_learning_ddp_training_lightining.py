# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
This script uses PyTorch lightning is an example of how DDP(distributed data parallel) can be used with AIMET for evaluation.
Please run the qat_range_learning_ddp_eval script first to get a saved model whose checkpoint is passed to args in this module
For this example we use a MV2 model and perform QAT with range learning.
The steps are as follows:
    1) Pass a quantized model for which compute encodings has been already performed as part of args
    2) Use PyTorch lightning DDP for evaluating the model
    3) Use PyTorch lightning DDP for Training the model
"""

# pylint: skip-file

import os
import argparse
import pickle
import torch

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy


# =======================define module==========================#
class LitImageNet(LightningModule):
    """
    Creates a Lightning Module for ImageNet dataset
    """

    def __init__(
        self,
        imagenet_dir,
        batch_size=1024,
        model_path="na",
        learning_rate=0.000001,
        num_classes=1000,
    ):
        super().__init__()

        # Setup hyper-parameters. setup weight-decay and other optimizer parameters here, and pass it down in the configure optimizers function.
        self.imagenet_dir = imagenet_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # ImageNet specific attributes
        self.num_classes = num_classes
        file_path = model_path
        with open(file_path, "rb") as file:
            quant_sim = pickle.load(file)
        self.model = quant_sim.model
        self.accuracy = Accuracy()

    def forward(self, x):
        """
        Model forward pass
        """
        x = self.model(x)
        return x

    def training_step(self, batch, _):
        """Training  step"""
        # Notice that no optimizer.step, torch.no_grad. model.eval is required
        images, labels = batch
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        return loss

    def validation_step(self, batch, _):
        """Validation step used b lightning"""
        images, labels = batch
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        pred = torch.argmax(logits, dim=1)
        self.accuracy(pred, labels)

        # self.log is pytorch's inbuilt definition.
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_acc", self.accuracy, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_epoch_end(self, _):
        """Runs at the end of validation step to print accuracy"""
        val_accuracy = self.accuracy.compute()
        if self.trainer.is_global_zero:
            print("\n------------------------------------------------------------")
            print("\nVALIDATION ACCURACY ===> ", val_accuracy.cpu().detach().numpy())
            print("\n------------------------------------------------------------")

        # Resetting accuracy is not mandatory in more latest releases.
        self.accuracy.reset()

    def test_step(self, batch, batch_idx):
        """Runs validation"""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configures optimizer for training"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        """Loads training data"""
        train_dataset = datasets.ImageFolder(
            os.path.join(self.imagenet_dir, "train"),
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        )
        return DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def test_dataloader(self):
        """Loads test data"""
        # test_dataset = datasets.ImageFolder('/nvme/dataset/imagenet/val',
        test_dataset = datasets.ImageFolder(
            os.path.join(self.imagenet_dir, "val"),
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        )
        test_sampler = DistributedSampler(dataset=test_dataset)
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            sampler=test_sampler,
        )

    def val_dataloader(self):
        """Loads validation data"""
        # test_dataloader can be reusued here since test and validation dataloader is the same for Imagenet.
        test_dataset = datasets.ImageFolder(
            os.path.join(self.imagenet_dir, "val"),
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        )
        return DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


# =======================setting up arguments===================
def main():
    """Main function"""
    # STEP 1
    parser = argparse.ArgumentParser(description="PyTorch Lightning DDP")
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("-b", "--batch_size", default=128, type=int, metavar="N")
    parser.add_argument(
        "--learning_rate", default=0.000001, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--num_classes",
        default=1000,
        type=int,
        help="Number of classes for the network.",
    )
    parser.add_argument(
        "--model_path",
        help="path to the quantized model's saved checkpoint for QAT",
        default="na",
    )
    parser.add_argument("--imagenet_dir", help="path to imagenet_dir", required=True)
    args = parser.parse_args()

    print("TRAINING QUANTIZED MODEL ON DDP ...")
    print("================================================" + str(args.batch_size))
    print(
        "================================================"
        + str(torch.cuda.device_count())
    )

    # For full deterministic reproducible behavior set seed_everything and have deterministic flag to True in trainer function
    pl.seed_everything(0, workers=True)

    # This is the model class initiation. Importing LitImageNet from lightning_torchvision_ddp.py
    model = LitImageNet(
        imagenet_dir=args.imagenet_dir,
        batch_size=args.batch_size,
        model_path=args.model_path,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes,
    )

    # Define the trainer here.
    trainer = pl.Trainer(
        deterministic=True,  # For full reproducibility. Does not work with DP. Only for DDP.
        strategy="DDP",  # Sets the DDP strategy for lightning
        accelerator="gpu",
        devices=-1,  # -1 means all available GPUs
        max_epochs=args.epochs,
        limit_train_batches=10,
    )

    # STEP 2
    trainer.test(model)

    # STEP 3
    trainer.fit(model)


if __name__ == "__main__":
    main()

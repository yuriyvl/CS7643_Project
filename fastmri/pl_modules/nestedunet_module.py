"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
from fastmri.models import NestedUnet
from torch.nn import functional as F

from .mri_module import MriModule
from .unet_module import UnetModule
from matplotlib import pyplot as plt
import os


class NestedUnetModule(MriModule):
    def __init__(
        self,
        in_chans=3,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.01,
        lr_step_size=20,
        lr_gamma=0.1,
        weight_decay=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.nestedunet = NestedUnet(
            in_chans=self.in_chans,
            out_chans=self.out_chans
        )

    def forward(self, image):
        return self.nestedunet(image.unsqueeze(1)).squeeze(1)

    def training_step(self, batch, batch_idx):
        image, target, _, _, _, _, _ = batch
        output = self(image)
        loss = F.l1_loss(output, target)

        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num, max_value = batch
        output = self(image)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output * std + mean,
            "target": target * std + mean,
            "val_loss": F.l1_loss(output, target),
        }

    def test_step(self, batch, batch_idx):
        image, _, mean, std, fname, slice_num, _ = batch
        output = self.forward(image)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        # Slice 22 resembles a complete knee, hence save this image slice for input, output and target
        if slice_num.item() == 22:
            if not os.path.exists(''.join(fname)):
            os.makedirs(''.join(fname))
            
            title = ''.join(fname) + "_" + str(slice_num.item())
            
            title1 = title + "_image"
            plt.imshow(image.permute(1, 2, 0).detach().cpu(), cmap='gray')
            plt.savefig(''.join(fname) + "./{}.png".format(title1))
            
            title1 = title + "_output"
            plt.imshow(output.permute(1, 2, 0).detach().cpu(), cmap='gray')
            plt.savefig(''.join(fname) + "./{}.png".format(title1))

            title1 = title + "_target"
            plt.imshow(target.permute(1, 2, 0).detach().cpu(), cmap='gray')
            plt.savefig(''.join(fname) + "./{}.png".format(title1))

        return {
            "fname": fname,
            "slice": slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return UnetModule.add_model_specific_args(parent_parser)

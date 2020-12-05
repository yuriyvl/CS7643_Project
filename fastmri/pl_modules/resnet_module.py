"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
from fastmri.models import ResNet, ResNet5Block
from torch.nn import functional as F

from .mri_module import MriModule
from torchviz import make_dot
import sys
from matplotlib import pyplot as plt
import os


class ResNetModule(MriModule):
    """
    ResNet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
        self,
        num_filters=32,
        filter_size=3,
        T=4,
        num_filters_start=2,
        num_filters_end=2,
        batch_norm=True,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            in_channels (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.T = T
        self.num_filters_start = num_filters_start
        self.num_filters_end = num_filters_end
        self.batch_norm=batch_norm
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        # self.resnet = ResNet(
        #     in_channels=self.in_channels,
        #     latent_channels=self.latent_channels,
        #     num_blocks=self.num_blocks,
        #     kernel_size=self.kernel_size,
        #     bias=self.bias,
        #     batch_norm=self.batch_norm
        # )
        self.resnet = ResNet5Block(
            num_filters=self.num_filters,
            filter_size=self.filter_size,
            T=self.T,
            num_filters_start=self.num_filters_start,
            num_filters_end=self.num_filters_end,
            batch_norm=self.batch_norm
        )

    def forward(self, image):
        return self.resnet(image.unsqueeze(1)).squeeze(1)

    def training_step(self, batch, batch_idx):
        image, target, _, _, _, _, _ = batch
        output = self(image)
        # make_dot(output, params=dict(list(self.resnet.named_parameters()))).render("dunet_torchviz.png", format="png")
        # sys.exit()
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
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_channels", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--num_depth_blocks", default=1, type=int, help="num_depth_blocks"
        )
        parser.add_argument(
            "--growth_rate", default=32, type=int, help="growth_rate"
        )
        parser.add_argument(
            "--num_layers", default=1, type=int, help="num_layers"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser

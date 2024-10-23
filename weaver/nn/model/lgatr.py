# new file, needed for new architecture

# actual model def (adopted from Dolores Gatr.py)

from os import path
import sys
from lgatr import GATr, SelfAttentionConfig, MLPConfig # gatr is folder in Dolores case? -> _init_.py in lgatr defines all of these
from lgatr.interface import embed_point, extract_scalar, extract_point, embed_scalar
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union, List
import dgl
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from xformers.ops.fmha import BlockDiagonalMask
from torch_scatter import scatter
from weaver.utils.logger_wandb import (
    log_confussion_matrix_wandb,
    log_roc_curves,
    log_histograms,
)
from weaver.nn.model.layers.mlp_readout_layer import MLPReadout


class LGATr(L.LightningModule):
    """Example wrapper around a GATr model.

    Expects input data that consists of a point cloud: one 3D point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.

    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(
        self,
        args,
        dev,
        input_dim: int = 35,
        output_dim: int = 4,
        n_postgn_dense_blocks: int = 3,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
        k_gravnet: int = 7,
        activation: str = "elu",
        weird_batchnom=False,
        blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=128,
    ):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.input_dim = 3
        self.output_dim = 4
        self.args = args
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=33, #adjust this
            out_s_channels=14, #adjust this?
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        # self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.01)
        # self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(1, momentum=0.01)
        self.MLP_layer = MLPReadout(4 + 14, 7) #adjust this

    def forward(self, g): # where do I need to define g?
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 3)
            Point cloud input data

        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """
        inputs = g.pos
        # TODO move this to the other type of scalar with more channels
        scalar_inputs = g.x
        # inputs = self.ScaledGooeyBatchNorm2_1(inputs)
        multivector, scalars = self.embed_into_ga(inputs, scalar_inputs)
        mask = self.build_attention_mask(g)
        # Pass data through GATr
        embedded_outputs, scalar_outputs = self.gatr(
            multivector, scalars=scalars, attention_mask=mask
        )  # (..., num_points, 1, 16)
        # assert embedded_outputs.shape[2:] == (1, 16)

        # Extract position
        out = self.extract_from_ga(embedded_outputs, scalar_outputs, g)
        out = self.MLP_layer(out)
        return out

    def extract_from_ga(self, multivector_outputs, scalar_outputs, g):
        # assert multivector_outputs.shape[2:] == (1, 16)
        # assert scalar_outputs.shape[2:] == (1,)
        #nodewise_outputs = extract_scalar(
        #    multivector_outputs
        #)  # (..., num_points, 1, 1)
        scalars_from_geometry = extract_scalar(multivector)[0, :, :, 0]
        output = torch.cat((scalars_from_geometry, scalar_outputs), dim=1)
        # sum per batch and calculate output
        mean_per_graph = scatter(output, g.batch, dim=0, reduce="mean")
        return mean_per_graph

    def embed_into_ga(self, inputs, scalar_inputs):

        # inputs = inputs.unsqueeze(0)

        # Embed point cloud in PGA
        multivector = embed_point(inputs)
        embedded_inputs = multivector.unsqueeze(-2)  # (B*num_points, 1, 16)
        scalars = scalar_inputs  # [B*num_points,channels]

        return embedded_inputs, scalars

    def build_attention_mask(self, inputs):
        """Construct attention mask from pytorch geometric batch.

        Parameters
        ----------
        inputs : torch_geometric.data.Batch
            Data batch.

        Returns
        -------
        attention_mask : xformers.ops.fmha.BlockDiagonalMask
            Block-diagonal attention mask: within each sample, each token can attend to each other
            token.
        """
        return BlockDiagonalMask.from_seqlens(torch.bincount(inputs.batch).tolist())

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        label = batch[1].long()

        model_output = self(inputs)
        logits = model_output

        loss_normal = self.criterion(logits, label.view(-1))
        loss_ce = torch.mean(loss_normal)
        loss = loss_ce
        self.log("loss_step", loss)
        if self.trainer.is_global_zero:
            wandb.log({"loss classification": loss})
            # self.log("loss classification", loss)
        print("T step", batch_idx, self.trainer.global_rank, loss)
        self.loss_final = self.loss_final + loss
        self.num_batches = self.num_batches + 1
        self.correct = (
            self.correct + (torch.argmax(logits, dim=1) == label).sum().item()
        )
        self.num_examples = self.num_examples + label.shape[0]
        return loss

    def on_train_epoch_end(self):
        # if self.trainer.is_global_zero:
        self.log("loss_epoch_end", self.loss_final / self.num_batches)
        self.log("acc_epoch_end", self.correct / self.num_examples)

    def on_train_epoch_start(self):
        # self.make_mom_zero()
        self.loss_final = 0
        self.num_batches = 0
        self.correct = 0
        self.num_examples = 0

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        label = batch[1].long()

        model_output = self(inputs)
        logits = model_output
        loss = self.criterion(logits, label.view(-1))
        print("V step", batch_idx, self.trainer.global_rank, loss)
        # print("loss validation", loss)
        if self.trainer.is_global_zero:
            # self.scores.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            # self.labels_all.append(label.detach().cpu().numpy())
            wandb.log({"loss val classification": loss})
            if batch_idx < 50:
                self.loss_final_val = self.loss_final_val + loss
                self.num_batches_val = self.num_batches_val + 1
                self.correct_val = (
                    self.correct_val
                    + (torch.argmax(logits, dim=1) == label).sum().item()
                )
                self.num_examples_val = self.num_examples_val + label.shape[0]

    def on_validation_epoch_start(self):
        # self.scores = []
        # self.labels_all = []
        self.loss_final_val = 0
        self.num_batches_val = 0
        self.correct_val = 0
        self.num_examples_val = 0

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero and self.current_epoch > 1:
            wandb.log(
                {"loss_epoch_end_val": self.loss_final_val / self.num_batches_val}
            )
            wandb.log({"acc_epoch_end_val": self.correct_val / self.num_examples_val})
            # self.scores = np.concatenate(self.scores)
            # self.labels_all = np.concatenate(self.labels_all)
            # scores_wandb = self.scores[0:10000]
            # y_true_wandb = self.labels_all[0:10000]
            # log_confussion_matrix_wandb(y_true_wandb, scores_wandb, self.current_epoch)
            # log_roc_curves(y_true_wandb, scores_wandb, self.current_epoch)

    def make_mom_zero(self):
        if self.current_epoch > 2 or self.args.predict:
            self.ScaledGooeyBatchNorm2_1.momentum = 0
            # self.ScaledGooeyBatchNorm2_2.momentum = 0
            # for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            #     gravnet_block.batchnorm_gravnet1.momentum = 0
            #     gravnet_block.batchnorm_gravnet2.momentum = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                "interval": "epoch",
                "monitor": "loss_epoch_end",
                "frequency": 3
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
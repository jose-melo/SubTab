""""
    Taken from: https://github.com/bioinf-jku/SNNs

    [1] Klambauer, Günter, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. “Self-Normalizing Neural Networks.” arXiv, September 7, 2017. https://doi.org/10.48550/arXiv.1706.02515.

"""

import math
import typing as ty
import pytorch_lightning as pl
from pathlib import Path

import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE, BaseModel


class SNNBase(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_layers: ty.List[int],
        dropout: float,
        d_out: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
    ) -> None:
        super().__init__()
        assert d_layers

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.layers = (
            nn.ModuleList(
                [
                    nn.Linear(d_layers[i - 1] if i else d_in, x)
                    for i, x in enumerate(d_layers)
                ]
            )
            if d_layers
            else None
        )

        self.normalizations = None
        self.activation = nn.SELU()
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

        # Ensure correct initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight.data, mode="fan_in", nonlinearity="linear"
                )
                nn.init.zeros_(m.bias.data)

        self.apply(init_weights)

    @property
    def d_embedding(self) -> int:
        return self.head.id_in  # type: ignore[code]

    def encode(self, x_num, x_cat):
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        layers = self.layers or []
        for i, m in enumerate(layers):
            x = m(x)
            if self.normalizations:
                x = self.normalizations[i](x)
            x = self.activation(x)
            if self.dropout:
                x = F.alpha_dropout(x, self.dropout, self.training)
        return x

    def calculate_output(self, x: Tensor) -> Tensor:
        x = self.head(x)
        x = x.squeeze(-1)
        return x

    def forward(self, x_num: Tensor, x_cat) -> Tensor:
        return self.calculate_output(self.encode(x_num, x_cat))


class SNN(BaseModel):

    def __init__(
        self,
        out_dim: int,
        input_dim: int,
        d_layers: ty.List[int],
        dropout: float,
        d_out: int,
        d_embedding: int,
        cardinalities: ty.Optional[ty.List[int]],
        loss: nn.Module = ...,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        dataset_name: str = None,
        **kwargs,
    ) -> None:

        self.input_dim = input_dim
        self.d_layers = d_layers
        self.dropout = dropout
        self.d_out = d_out
        self.categories = [x[1] for x in cardinalities]
        self.d_embedding = d_embedding
        self.cardinalities = cardinalities

        super().__init__(
            d_out,
            out_dim,
            loss,
            lr,
            weight_decay,
            dataset_name=dataset_name,
        )

    def build_encoder(self):

        class EncoderSNN(nn.Module):
            def __init__(
                self,
                n_features: int,
                cardinalities: list[tuple[int, int]],
                snn: nn.Module,
                *args,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)

                self.n_features = n_features
                self.cardinalities = cardinalities
                self.snn = snn

            def forward(self, x: torch.Tensor):
                categorical_idx_colums = [x[0] for x in self.cardinalities]
                numerical_idx_colums = [
                    idx
                    for idx in range(self.n_features)
                    if idx not in categorical_idx_colums
                ]

                cat_x = x[:, categorical_idx_colums]
                cat_x = cat_x.long()
                num_x = x[:, numerical_idx_colums]

                return self.snn(num_x, cat_x)

        snn = SNNBase(
            d_in=self.input_dim,
            d_layers=self.d_layers,
            dropout=self.dropout,
            d_out=self.d_out,
            categories=self.categories,
            d_embedding=self.d_embedding,
        )

        encoder = EncoderSNN(
            self.input_dim + len(self.cardinalities),
            self.cardinalities,
            snn,
        )

        return encoder

    @staticmethod
    def get_model_args(
        datamodule: pl.LightningDataModule,
        args: ty.OrderedDict,
        model_args: ty.OrderedDict,
        **kwargs,
    ):
        model_args.n_cont_features = datamodule.dataset.D - len(
            datamodule.dataset.cardinalities
        )

        model_args.cardinalities = datamodule.dataset.cardinalities

        if not args.using_embedding:
            model_args.input_dim = len(datamodule.dataset.num_features)

            model_args.out_dim = (
                len(set(datamodule.dataset.y))
                if args.task_type == TASK_TYPE.MULTI_CLASS
                else 1
            )

        model_args.summary_input = datamodule.dataset.D
        model_args.dataset_name = datamodule.dataset.name
        return model_args

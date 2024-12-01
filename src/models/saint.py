"""
    SAINT

    [1] Somepalli, Gowthami, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, and Tom Goldstein. SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training.â€ arXiv, June 2, 2021. https://doi.org/10.48550/arXiv.2106.01342.
    
    [2] https://github.com/somepago/saint/blob
    /main/models/model.py 
"""

import numpy as np
from sklearn.calibration import LabelEncoder
from torch.nn import Parameter
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE, BaseModel, EncodeEmbeddingFeatures
import pytorch_lightning as pl
import typing as ty


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        """
        Args:
            n_features: the number of continous features.
            d_embedding: the embedding size.
        """
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, however: {n_features=}")
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, however: {d_embedding=}")

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f"The input must have at least two dimensions, however: {x.ndim=}"
            )

        x = x[..., None] * self.weight
        x = x + self.bias[None]
        return x


class CategoricalEmbeddings(nn.Module):
    """Embeddings for categorical features.

    **Examples**

    >>> cardinalities = [3, 10]
    >>> x = torch.tensor([
    ...     [0, 5],
    ...     [1, 7],
    ...     [0, 2],
    ...     [2, 4]
    ... ])
    >>> x.shape  # (batch_size, n_cat_features)
    torch.Size([4, 2])
    >>> m = CategoricalEmbeddings(cardinalities, d_embedding=5)
    >>> m(x).shape  # (batch_size, n_cat_features, d_embedding)
    torch.Size([4, 2, 5])
    """

    def __init__(
        self, cardinalities: list[int], d_embedding: int, bias: bool = True
    ) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature.
            d_embedding: the embedding size.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of a feature value. For each feature, a separate
                non-shared bias vector is allocated.
                In the paper, FT-Transformer uses `bias=True`.
        """
        super().__init__()
        if not cardinalities:
            raise ValueError("cardinalities must not be empty")
        if any(x <= 0 for x in cardinalities):
            i, value = next((i, x) for i, x in enumerate(cardinalities) if x <= 0)
            raise ValueError(
                "cardinalities must contain only positive values,"
                f" however: cardinalities[{i}]={value}"
            )
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, however: {d_embedding=}")

        self.embeddings = nn.ModuleList(
            [nn.Embedding(x, d_embedding) for x in cardinalities]
        )
        self.bias = (
            Parameter(torch.empty(len(cardinalities), d_embedding)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
        for m in self.embeddings:
            nn.init.uniform_(m.weight, -d_rsqrt, d_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f"The input must have at least two dimensions, however: {x.ndim=}"
            )
        n_features = len(self.embeddings)
        if x.shape[-1] != n_features:
            raise ValueError(
                "The last input dimension (the number of categorical features) must be"
                " equal to the number of cardinalities passed to the constructor."
                f" However: {x.shape[-1]=}, len(cardinalities)={n_features}"
            )

        x = torch.stack(
            [self.embeddings[i](x[..., i]) for i in range(n_features)], dim=-2
        )
        if self.bias is not None:
            x = x + self.bias
        return x


class _CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def forward(self, batch_dims: tuple[int]) -> Tensor:
        if not batch_dims:
            raise ValueError("The input must be non-empty")

        return self.weight.expand(*batch_dims, 1, -1)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ff_encodings(x, B):
    x_proj = (2.0 * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        nfeats,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        style="col",
    ):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == "colrow":
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim,
                                Residual(
                                    Attention(
                                        dim,
                                        heads=heads,
                                        dim_head=dim_head,
                                        dropout=attn_dropout,
                                    )
                                ),
                            ),
                            PreNorm(
                                dim, Residual(FeedForward(dim, dropout=ff_dropout))
                            ),
                            PreNorm(
                                dim * nfeats,
                                Residual(
                                    Attention(
                                        dim * nfeats,
                                        heads=heads,
                                        dim_head=64,
                                        dropout=attn_dropout,
                                    )
                                ),
                            ),
                            PreNorm(
                                dim * nfeats,
                                Residual(FeedForward(dim * nfeats, dropout=ff_dropout)),
                            ),
                        ]
                    )
                )
            else:
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim * nfeats,
                                Residual(
                                    Attention(
                                        dim * nfeats,
                                        heads=heads,
                                        dim_head=64,
                                        dropout=attn_dropout,
                                    )
                                ),
                            ),
                            PreNorm(
                                dim * nfeats,
                                Residual(FeedForward(dim * nfeats, dropout=ff_dropout)),
                            ),
                        ]
                    )
                )

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        if self.style == "colrow":
            for attn1, ff1, attn2, ff2 in self.layers:
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, "b n d -> 1 b (n d)")
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, "1 b (n d) -> b n d", n=n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, "b n d -> 1 b (n d)")
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, "1 b (n d) -> b n d", n=n)
        return x


# transformer
class Transformer(nn.Module):
    def __init__(
        self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Residual(
                                Attention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=attn_dropout,
                                )
                            ),
                        ),
                        PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


# mlp
class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred


class SAINTBase(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        num_special_tokens=0,
        attn_dropout=0.0,
        ff_dropout=0.0,
        cont_embeddings="MLP",
        scalingfactor=10,
        attentiontype="col",
        final_mlp_style="common",
        y_dim=2,
    ):
        super().__init__()
        assert all(
            map(lambda n: n > 0, categories)
        ), "number of each category must be positive"

        # categories related calculations

        self.categories = categories
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(
            torch.tensor(list(categories)), (1, 0), value=num_special_tokens
        )
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer("categories_offset", categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == "MLP":
            self.simple_MLP = nn.ModuleList(
                [simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)]
            )
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == "pos_singleMLP":
            self.simple_MLP = nn.ModuleList(
                [simple_MLP([1, 100, self.dim]) for _ in range(1)]
            )
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print("Continous features are not passed through attention")
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        # transformer
        if attentiontype == "col":
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
        elif attentiontype in ["row", "colrow"]:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype,
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

        self.mlp_output = simple_MLP(
            [
                dim * (self.num_categories + self.num_continuous),
                dim * (self.num_categories + self.num_continuous),
                dim_out,
            ],
        )
        if self.final_mlp_style == "common":
            self.mlp1 = simple_MLP([dim, (self.total_tokens) * 2, self.total_tokens])
            self.mlp2 = simple_MLP([dim, (self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            self.mlp2 = sep_MLP(
                dim, self.num_continuous, np.ones(self.num_continuous).astype(int)
            )

        self.mlpfory = simple_MLP([dim, 1000, y_dim])
        self.pt_mlp = simple_MLP(
            [
                dim * (self.num_continuous + self.num_categories),
                6 * dim * (self.num_continuous + self.num_categories) // 5,
                dim * (self.num_continuous + self.num_categories) // 2,
            ]
        )
        self.pt_mlp2 = simple_MLP(
            [
                dim * (self.num_continuous + self.num_categories),
                6 * dim * (self.num_continuous + self.num_categories) // 5,
                dim * (self.num_continuous + self.num_categories) // 2,
            ]
        )
        self.continuous_embeddings = LinearEmbeddings(num_continuous, self.dim)

        if len(self.categories) != 0:
            self.categorical_embeddings = CategoricalEmbeddings(
                self.categories, self.dim, True
            )

    def forward(self, x_categ, x_cont):
        """Do the forward pass."""
        x_embeddings: list[Tensor] = []

        x_embeddings.append(self.continuous_embeddings(x_cont))

        if x_categ is not None:
            x_embeddings.append(self.categorical_embeddings(x_categ))

        x = torch.cat(x_embeddings, dim=1)

        x = self.transformer(x)
        x = self.mlp_output(x.reshape(x.shape[0], -1))

        return x


class SAINT(BaseModel):

    def __init__(
        self,
        out_dim: int,
        dim: int,
        input_dim: int,
        cardinalities: list[tuple[int, int]],
        depth: int,
        heads: int,
        dim_head: int = 16,
        mlp_hidden_mults: tuple = (4, 2),
        mlp_act: str = None,
        num_special_tokens: int = 0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        cont_embeddings: str = "MLP",
        scalingfactor: int = 10,
        attentiontype: str = "col",
        final_mlp_style: str = "common",
        y_dim: int = 2,
        loss: nn.Module = ...,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        dataset_name: str = None,
        num_epochs: int = None,
        encoder_type: str = "conv",
        input_embed_dim: int = 64,
        iterations_per_epoch: int = None,
        using_embedding: bool = False,
        **kwargs,
    ) -> None:

        self.input_dim = input_dim
        self.dim = dim
        self.cardinalities = cardinalities
        self.categories = [x[1] for x in cardinalities]
        self.num_continuous = self.input_dim - len(self.categories)
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.dim_out = dim
        self.mlp_hidden_units = mlp_hidden_mults
        self.mlp_act = mlp_act
        self.num_special_tokens = num_special_tokens
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.cont_embeddings = cont_embeddings
        self.scalingfactor = scalingfactor
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style
        self.y_dim = y_dim
        self.encoder_type = encoder_type
        self.input_embed_dim = input_embed_dim
        self.using_embedding = using_embedding

        T_max = num_epochs * iterations_per_epoch

        super().__init__(
            head_dimension=dim,
            out_dim=out_dim,
            loss=loss,
            lr=lr,
            weight_decay=weight_decay,
            T_max=T_max,
            dataset_name=dataset_name,
        )
        self.print_params()

    def build_encoder(self):

        class EncoderSaint(nn.Module):

            def __init__(
                self,
                n_features: int,
                cardinalities: list[tuple[int, int]],
                saint: nn.Module,
                *args,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)
                self.n_features = n_features
                self.cardinalities = cardinalities
                self.saint = saint

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

                if cat_x.shape[1] == 0:
                    cat_x = None

                return self.saint(cat_x, num_x)

        saint = SAINTBase(
            categories=self.categories,
            num_continuous=self.num_continuous,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            dim_out=self.dim_out,
            mlp_hidden_mults=self.mlp_hidden_units,
            mlp_act=self.mlp_act,
            num_special_tokens=self.num_special_tokens,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            cont_embeddings=self.cont_embeddings,
            scalingfactor=self.scalingfactor,
            attentiontype=self.attentiontype,
            final_mlp_style=self.final_mlp_style,
            y_dim=self.y_dim,
        )

        encoder = EncoderSaint(
            n_features=self.input_dim,
            cardinalities=self.cardinalities,
            saint=saint,
        )

        return nn.Sequential(
            (
                EncodeEmbeddingFeatures(
                    input_dim=self.input_dim,
                    emb_dim=self.dim,
                    encoder_type=self.encoder_type,
                    input_embed_dim=self.input_embed_dim,
                )
                if self.using_embedding
                else nn.Identity()
            ),
            encoder,
        )

    @staticmethod
    def get_model_args(
        datamodule: pl.LightningDataModule,
        args: ty.OrderedDict,
        model_args: ty.OrderedDict,
        dataset: BaseDataset = None,
        **kwargs,
    ):
        if dataset is None:
            dataset = datamodule.dataset

        if hasattr(dataset, "H"):
            model_args.input_embed_dim = dataset.H
        else:
            model_args.input_embed_dim = None

        extra_cls = 1 if args.using_embedding else 0
        model_args.input_dim = (
            extra_cls
            + datamodule.dataset.D
            + sum([x[1] - 1 for x in datamodule.dataset.cardinalities])
        )
        if args.task_type == TASK_TYPE.MULTI_CLASS:
            model_args.out_dim = len(set(datamodule.dataset.y))
        else:
            model_args.out_dim = 1

        if args.using_embedding:
            model_args.summary_input = (
                args.batch_size,
                model_args.input_dim,
                model_args.input_embed_dim,
            )
        else:
            model_args.summary_input = (args.batch_size, model_args.input_dim)

        if not hasattr(model_args, "dataset_name"):
            model_args.dataset_name = datamodule.dataset.name
        datamodule.setup("train")
        model_args.iterations_per_epoch = len(datamodule.train_dataloader())
        model_args.num_epochs = args.exp_train_total_epochs

        model_args.cardinalities = dataset.cardinalities
        model_args.n_cont_features = (
            dataset.D - sum([x[1] - 1 for x in dataset.cardinalities]) + extra_cls
        )

        return model_args

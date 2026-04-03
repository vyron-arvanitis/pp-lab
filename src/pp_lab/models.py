from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import load_pdg_mapping
from .data import masked_average, normalize_adjacency, normalize_inputs

PDG_MAPPING = load_pdg_mapping()
DEFAULT_NUM_PDG_IDS = len(PDG_MAPPING)


class GCN(nn.Module):
    def __init__(self, num_features: int, units: int):
        super().__init__()
        self.linear = nn.Linear(num_features, units)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return adjacency @ self.linear(inputs)


class OutputLayer(nn.Module):
    def __init__(self, num_inputs: int):
        super().__init__()
        self.output_layer = nn.Sequential(nn.Linear(num_inputs, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(x)


class DeepSetBase(nn.Module):
    def __init__(self, num_features: int = 8, units: int = 32):
        super().__init__()
        self.per_item_mlp = nn.Sequential(
            nn.Linear(num_features, units),
            nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.per_item_mlp(x)
        x = masked_average(x, mask) if mask is not None else x.mean(axis=-2)
        return self.global_mlp(x)


class FlatMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        max_len: int,
        hidden_dims: tuple[int, ...],
        concat_mask: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_len = max_len
        self.concat_mask = concat_mask

        input_dim = num_features * max_len + (max_len if concat_mask else 0)
        layers = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs, mask=None):
        x = inputs["feat"]
        batch_size, padded_len, num_features = x.shape

        if padded_len < self.max_len:
            pad_len = self.max_len - padded_len
            pad = x.new_zeros((batch_size, pad_len, num_features))
            x = torch.cat([x, pad], dim=1)
            if mask is not None:
                pad_mask = torch.ones((batch_size, pad_len), device=mask.device)
                mask = torch.cat([mask, pad_mask], dim=1)
        elif padded_len > self.max_len:
            x = x[:, : self.max_len]
            if mask is not None:
                mask = mask[:, : self.max_len]

        x = x.reshape(batch_size, -1)

        if self.concat_mask:
            if mask is None:
                mask = x.new_zeros((batch_size, self.max_len))
            else:
                mask = mask.float()
            x = torch.cat([x, mask], dim=1)

        return self.net(x)


class DeepSet(nn.Module):
    def __init__(self, num_features: int = 8, units: int = 32):
        super().__init__()
        self.deep_set_layer = DeepSetBase(num_features, units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs: dict, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.deep_set_layer(inputs["feat"], mask)
        return self.output_layer(x)


class CombinedModel(nn.Module):
    def __init__(
        self,
        num_features: int = 8,
        embed_dim: int = 8,
        num_pdg_ids: int = DEFAULT_NUM_PDG_IDS,
        units: int = 32,
        dropout_rate: float = 0.3,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        del dropout_rate, num_heads, num_layers
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.deep_set_layer = DeepSetBase(num_features=num_features + embed_dim, units=units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs: dict, mask: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.embedding_layer(inputs["pdg"])
        x = torch.cat([inputs["feat"], embeddings], -1)
        x = self.deep_set_layer(x, mask)
        return self.output_layer(x)


class DeepSetWithGCN(nn.Module):
    def __init__(
        self,
        num_features: int = 8,
        units: int = 32,
        dropout_rate: float = 0.3,
        num_heads: int = 4,
        num_layers: int = 2,
        embed_dim: int = 8,
    ):
        super().__init__()
        del dropout_rate, num_heads, num_layers, embed_dim
        self.gcn_layer = GCN(num_features, units)
        self.deep_set_layer = DeepSetBase(units, units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs: dict, mask: torch.Tensor | None = None) -> torch.Tensor:
        adjacency = normalize_adjacency(inputs["adj"])
        x = F.relu(self.gcn_layer(inputs["feat"], adjacency))
        x = self.deep_set_layer(x, mask)
        return self.output_layer(x)


class CombinedModelWithGCN(nn.Module):
    def __init__(
        self,
        num_features: int = 8,
        embed_dim: int = 8,
        num_pdg_ids: int = DEFAULT_NUM_PDG_IDS,
        units: int = 32,
        dropout_rate: float = 0.3,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        del num_heads, num_layers
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.gcn_layer = GCN(num_features + embed_dim, units)
        self.batch_norm = nn.BatchNorm1d(units)
        self.dropout = nn.Dropout(dropout_rate)
        self.deep_set_layer = DeepSetBase(units, units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs: dict, mask: torch.Tensor | None = None) -> torch.Tensor:
        adjacency = normalize_adjacency(inputs["adj"])
        embeddings = self.embedding_layer(inputs["pdg"])
        x = torch.cat([inputs["feat"], embeddings], -1)
        x = F.relu(self.gcn_layer(x, adjacency))
        x = self.deep_set_layer(x, mask)
        return self.output_layer(x)


class OptimalModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        units: int = 32,
        dropout_rate: float = 0.17,
        negative_slope: float = 0.01,
        embed_dim: int = 8,
        num_pdg_ids: int = DEFAULT_NUM_PDG_IDS,
    ):
        super().__init__()

        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.input_layer = GCN(num_features + embed_dim, units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList(
            [
                self.input_layer,
                nn.BatchNorm1d(num_features + embed_dim),
                nn.LeakyReLU(negative_slope),
                nn.Dropout(dropout_rate),
            ]
        )

        for index in range(3):
            if index in (0, 1):
                self.layers.append(GCN(units, units))
            else:
                self.layers.append(nn.Linear(units, units))
            self.layers.append(nn.BatchNorm1d(units))
            self.layers.append(nn.LeakyReLU(negative_slope))
            self.layers.append(nn.Dropout(dropout_rate))

        self.global_mlp = nn.Sequential(
            nn.Linear(units, units),
            nn.BatchNorm1d(units),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout_rate),
            nn.Linear(units, 1),
        )

    def forward(self, inputs: dict, mask: torch.Tensor | None = None) -> torch.Tensor:
        adjacency = normalize_adjacency(inputs["adj"])
        embeddings = self.dropout(self.embedding_layer(inputs["pdg"]))
        x = torch.cat([inputs["feat"], embeddings], dim=-1)

        for offset in range(0, len(self.layers), 4):
            layer = self.layers[offset]
            batch_norm = self.layers[offset + 1]
            activation = self.layers[offset + 2]
            dropout = self.layers[offset + 3]

            x = batch_norm(x.transpose(1, 2)).transpose(1, 2)
            x = activation(layer(x, adjacency) if isinstance(layer, GCN) else layer(x))
            x = dropout(x)

        pooled = masked_average(x, mask) if mask is not None else x.mean(axis=-2)
        return self.global_mlp(pooled)


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_features: int = 8,
        embed_dim: int = 8,
        num_pdg_ids: int = DEFAULT_NUM_PDG_IDS,
        units: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.17,
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.input_proj = nn.Linear(num_features + embed_dim, units)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=units,
            nhead=num_heads,
            dim_feedforward=units * 2,
            dropout=dropout_rate,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs: dict, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        embeddings = self.embedding_layer(inputs["pdg"])
        x = torch.cat([inputs["feat"], embeddings], dim=-1)
        x = self.input_proj(x).transpose(0, 1)
        x = self.transformer_encoder(x).transpose(0, 1)
        return self.output_layer(x.mean(dim=1))


class NormalizedCombinedModelWithGCN(nn.Module):
    def __init__(
        self,
        num_features: int = 8,
        embed_dim: int = 8,
        num_pdg_ids: int = DEFAULT_NUM_PDG_IDS,
        units: int = 32,
        dropout_rate: float = 0.3,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.model = CombinedModelWithGCN(
            num_features=num_features,
            embed_dim=embed_dim,
            num_pdg_ids=num_pdg_ids,
            units=units,
            dropout_rate=dropout_rate,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(self, inputs: dict, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(normalize_inputs(inputs), mask)


class VariableDeepSetWithGCN(nn.Module):
    def __init__(self, hidden_layers: int, gcn_layers: list[int], num_features: int, units: int = 32):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.gcn_layers = gcn_layers

        self.input_layer = GCN(num_features, units) if gcn_layers and gcn_layers[0] == 1 else nn.Linear(num_features, units)
        self.layers = nn.ModuleList()
        for index in range(hidden_layers):
            if (index + 2) in gcn_layers:
                self.layers.append(GCN(units, units))
            else:
                self.layers.append(nn.Linear(units, units))

        self.global_mlp = nn.Sequential(nn.Linear(units, 1))

    def forward(self, inputs: dict, mask: torch.Tensor | None = None) -> torch.Tensor:
        adjacency = normalize_adjacency(inputs["adj"])
        x = inputs["feat"]

        if self.gcn_layers and self.gcn_layers[0] == 1:
            x = F.relu(self.input_layer(x, adjacency))
        else:
            x = F.relu(self.input_layer(x))

        for layer in self.layers:
            x = F.relu(layer(x, adjacency) if isinstance(layer, GCN) else layer(x))

        pooled = masked_average(x, mask) if mask is not None else x.mean(axis=-2)
        return self.global_mlp(pooled)


MODEL_REGISTRY = {
    "flat_mlp": FlatMLP,
    "deepset": DeepSet,
    "deepset_combined": CombinedModel,
    "deepset_gcn": DeepSetWithGCN,
    "deepset_combined_wgcn": CombinedModelWithGCN,
    "optimal_model": OptimalModel,
    "deepset_wgcn_variable": VariableDeepSetWithGCN,
    "transformer": TransformerModel,
    "deepset_combined_wgcn_normalized": NormalizedCombinedModelWithGCN,
}

DeepSet_Base_Arc = DeepSetBase
DeepSet_wGCN = DeepSetWithGCN
CombinedModel_wGCN = CombinedModelWithGCN
CombinedModel_wGCN_Normalized = NormalizedCombinedModelWithGCN
DeepSet_wGCN_variable = VariableDeepSetWithGCN


def from_config(config: dict):
    config = config.copy()
    return MODEL_REGISTRY[config.pop("model_name")](**config)

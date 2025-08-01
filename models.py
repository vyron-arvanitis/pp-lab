import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import (
    normalize_inputs, 
    normalize_adjacency, 
    masked_average
)

class GCN(nn.Module):
    """
    Simple graph convolution. Equivalent to GCN from Kipf & Welling (https://arxiv.org/abs/1609.02907)
    when fed a normalized adjacency matrix.

    Attributes
    ----------
    linear : nn.Linear
        Linear transformation applied to node features.
    """

    def __init__(self, num_features: int, units: int):
        """
        Initialize the GCN layer.

        Parameters
        ----------
        num_features : int
            Number of input features per node.
        units : int
            Number of output units (hidden features) per node.
        """

        super().__init__()
        self.linear = nn.Linear(num_features, units)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Apply the GCN layer to a batch of node features and adjacency matrices.

        Parameters
        ----------
        inputs : torch.Tensor
            Node feature matrix of shape (batch_size, num_particles, num_features).
        adjacency : torch.Tensor
            Normalized adjacency matrix of shape (batch_size, num_particles, num_nodes).

        Returns
        -------
        torch.Tensor
            Output node features of shape (batch_size, num_particles, units).
        """
        return adjacency @ self.linear(inputs)

class OutputLayer(nn.Module):
    """
    Output layer.

    Attributes
    ----------
    output_layer : nn.Sequential
        A sequential container with a single `nn.Linear` layer that maps input
        features to a single output value.
    """
    def __init__(self, num_inputs: int):
        """
        Initialize the output layer.

        Parameters
        ----------
        num_inputs : int
            Number of input features per example.
        """
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(num_inputs, 1)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the output layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_inputs).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1), representing scalar predictions.
        """
        x = self.output_layer(x)
        return x

class DeepSetLayer(nn.Module):
    """
    Deep Set layer.

    Attributes
    ----------
    per_item_mlp : nn.Sequential
        MLP applied independently to each item in the input set.
    global_mlp : nn.Sequential
        MLP applied to the aggregated representation of the input set.
    """

    def __init__(self, num_features: int=8, units: int=32):
        """
        Initialize the DeepSetLayer.

        Parameters
        ----------
        num_features : int
            Number of input features per element in the set (default is 8).
        units : int
            Number of hidden units in both the per-item and global MLPs (default is 32).
        """
        super().__init__()
        self.per_item_mlp = nn.Sequential(
            nn.Linear(num_features, units),
            nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU()
        )

    def forward(self, x:torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the DeepSetLayer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_particles, num_features).
        mask : torch.Tensor
            Boolean mask tensor of shape (batch_size, num_particles) where `True`
            indicates particles to ignore in the aggregation.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, units), representing the
            permutation-invariant representation of each input set.
        """
        x = self.per_item_mlp(x)
        if mask is not None:
            x = masked_average(x, mask)
        else:
            x = x.mean(axis=-2)
        x = self.global_mlp(x)
        return x


# --------------------------------------------------
# Below are the model definitions
# adjust these to your needs and/or add new definitions
# --------------------------------------------------

with open("pdg_mapping.json") as f:
    PDG_MAPPING = json.load(f)


class DeepSet(nn.Module):
    """
    Deep Set model 

    Attributes
    ----------
    deep_set_layer : DeepSetLayer
        Implementation of the DeepSetLayer
    output_layer : OutputLayer
        Implementation of the OutputLayer
    """

    def __init__(self, num_features: int=8, units: int=32):
        """
        Initialize the DeepSet model.

        Parameters
        ----------
        num_features : int
            Number of input features per element in the set (default is 8).
        units : int
            Number of hidden units used in both the DeepSetLayer and OutputLayer (default is 32).
        """

        super().__init__()
        self.deep_set_layer = DeepSetLayer(num_features, units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs: dict, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the DeepSetLayer.

        Parameters
        ----------
        inputs : dict
            Input tensor of shape (batch_size, num_particles, num_features).
        mask : torch.Tensor
            Boolean mask tensor of shape (batch_size, num_particles) where `True`
            indicates particles to ignore in the aggregation.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, units), representing the
            permutation-invariant representation of each input set.
        """

        x = inputs["feat"]
        x = self.deep_set_layer(x, mask)
        x = self.output_layer(x)
        return x

class CombinedModel(nn.Module):
    """
    CombinedModel model 

    Attributes
    ----------
    embedding_layer : nn.Embedding
        Implementation of an embedding layer for particle type identifiers (PDG codes).
    deep_set_layer : DeepSetLayer
        Implementation of the DeepSetLayer
    output_layer : OutputLayer
        Implementation of the OutputLayer
    """

    def __init__(self, 
        num_features=8,
        embed_dim=8, 
        num_pdg_ids=len(PDG_MAPPING), 
        units=32,  
        dropout_rate=0.3, 
        num_heads=4, 
        num_layers=2):

        """
        Initialize the CombinedModel.

        Parameters
        ----------
        num_features : int
            Number of continuous input features per particle (default is 8).
        embed_dim : int
            Dimensionality of the learned embeddings for PDG IDs (default is 8).
        num_pdg_ids : int
            Number of distinct PDG IDs (excluding padding or unknown class).
        units : int
            Number of hidden units in the DeepSetLayer and OutputLayer (default is 32).
        dropout_rate : float
            Dropout rate (not currently used but reserved for future use).
        num_heads : int
            Number of attention heads (not used in this model; reserved for extensions).
        num_layers : int
            Number of layers (not used in this model; reserved for extensions).
        """

        super().__init__()
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.deep_set_layer = DeepSetLayer(num_features=num_features+ embed_dim, units=units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs: dict, mask: torch.Tensor=None):
        """
        Forward pass of the CombinedModel.

        Parameters
        ----------
        inputs : dict
            A dictionary with keys:
                - "feat": tensor of shape (batch_size, num_particles, num_features),
                  containing continuous per-particle features.
                - "pdg": tensor of shape (batch_size, num_particles), containing integer
                  PDG codes used for categorical embedding.
        mask : torch.Tensor
            Boolean tensor of shape (batch_size, num_particles), indicating which particles
            to ignore in the aggregation step.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, 1), containing scalar predictions per input set.
        """
        pdg = inputs["pdg"]
        feat = inputs["feat"]
        emb = self.embedding_layer(pdg)
        x = torch.cat([feat, emb], -1)

        x = self.deep_set_layer(x, mask)
        x = self.output_layer(x)
        return x


class GraphNetwork(nn.Module):
    """
    GCN based model with adjacency matrices and features as input
    """
    def __init__(self, num_features=8, units=32,  dropout_rate=0.3, num_heads=4, num_layers=2, embed_dim=8):
        super().__init__()
        self.gcn_layer = GCN(num_features, units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs, mask=None):
        adj = inputs["adj"]
        feat = inputs["feat"]
        adj = normalize_adjacency(adj)
        x = self.gcn_layer(feat, adj)
        x = self.output_layer(x)
        return x


class DeepSet_wGCN(nn.Module):
    def __init__(self, num_features=8, units=32,  dropout_rate=0.3, num_heads=4, num_layers=2, embed_dim=8):
        super().__init__()
        self.gcn_layer = GCN(num_features, units)
        self.deep_set_layer = DeepSetLayer(units, units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs, mask=None):
        x = inputs["feat"]
        adj = inputs["adj"]
        adj = normalize_adjacency(adj)
        x = self.gcn_layer(x, adj)
        x = self.deep_set_layer(x, mask)

        x = self.output_layer(x)
        return x


class CombinedModel_wGCN(nn.Module):
    def __init__(self, num_features=8, embed_dim=8, num_pdg_ids=len(PDG_MAPPING), units=32, dropout_rate=0.3, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.gcn_layer = GCN(num_features + embed_dim, units)

        #  Add BatchNorm
        self.batch_norm = nn.BatchNorm1d(units)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer defined here
        # Note: Dropout is applied after the embedding layer and GCN layer
        # This is to prevent overfitting by randomly dropping units during training
        self.deep_set_layer = DeepSetLayer(units, units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs, mask=None):
        pdg = inputs["pdg"]
        feat = inputs["feat"]
        adj = inputs["adj"]
        adj = normalize_adjacency(adj)

        emb = self.embedding_layer(pdg)
        emb = self.dropout(emb)  # Apply dropout after the embeddings
        x = torch.cat([feat, emb], -1)
        x = self.gcn_layer(x, adj)

        # Apply Norm
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)  # Dropout after GCN

        x = self.deep_set_layer(x, mask)
        x = self.output_layer(x)
        return x
    


class OptimalModel(nn.Module):
    def __init__(self, num_features, units=32, dropout_rate=0.17, negative_slope=0.01, embed_dim=8, num_pdg_ids=len(PDG_MAPPING),):
        super().__init__()

        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)

        self.input_layer = GCN(num_features + embed_dim, units)
        self.batch_norm = nn.BatchNorm1d(units)
        self.activation = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout_rate)
        
        
        self.layers = nn.ModuleList()
        self.layers.append(self.input_layer)
        self.layers.append(nn.BatchNorm1d(num_features + embed_dim))
        self.layers.append(nn.LeakyReLU(negative_slope))
        self.layers.append(nn.Dropout(dropout_rate))

        for i in range(3):
            if i == 0 or i == 1:
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
            nn.Linear(units, 1)
        )

    def forward(self, inputs, mask=None):
        inputs = self.normalize_inputs(inputs)
        pdg = inputs["pdg"]
        adj = normalize_adjacency(inputs["adj"])
        feat = inputs["feat"]

        emb = self.embedding_layer(pdg)
        emb = self.dropout(emb)
        x = torch.cat([feat, emb], dim=-1)

        for i in range(0, len(self.layers), 4):
            layer = self.layers[i]
            bn = self.layers[i + 1]
            act = self.layers[i + 2]
            do = self.layers[i + 3]

            x = x.transpose(1, 2)
            x = bn(x)
            x = x.transpose(1, 2)

            if isinstance(layer, GCN):
                x = act(layer(x, adj))
            else:
                x = act(layer(x))
        
            x = do(x)
        
        if mask is not None:
            x = masked_average(x, mask)
        else:
            x = x.mean(axis=-2)
            
        return self.global_mlp(x)


class TransformerModel(nn.Module):
    def __init__(self, num_features=8, embed_dim=8, num_pdg_ids=len(PDG_MAPPING), units=32, num_heads=4, num_layers=2, dropout_rate=0.17):
        super().__init__()

        # Particle type embedding
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)

        # Linear projection of features + embeddings
        self.input_proj = nn.Linear(num_features + embed_dim, units)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=units,
            nhead=num_heads,
            dim_feedforward=units * 2,
            dropout=dropout_rate,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global pooling: mean over particle dimension
        # self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output layer
        self.output_layer = OutputLayer(units)

    def forward(self, inputs, mask=None):
        pdg = inputs["pdg"]
        feat = inputs["feat"]

        # Embed particle types
        emb = self.embedding_layer(pdg)

        # Concatenate features and embeddings
        x = torch.cat([feat, emb], dim=-1)

        # Project to Transformer input size
        x = self.input_proj(x)

        # Apply Transformer encoder, batch first argument should be True
        # Transformer expects [seq_len, batch, features], so we transpose
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)

        # Global pooling (mean over particles)
        x = x.mean(dim=1)

        # Final classification
        x = self.output_layer(x)
        return x

class CombinedModel_wGCN_Normalized(nn.Module):
    def __init__(self, num_features=8, embed_dim=8, num_pdg_ids=len(PDG_MAPPING), units=32, dropout_rate=0.3, num_heads=4, num_layers=2):
        super().__init__()
        self.model = CombinedModel_wGCN(
            num_features=num_features,
            embed_dim=embed_dim,
            num_pdg_ids=num_pdg_ids,
            units=units
        )

    def forward(self, inputs, mask=None):
        inputs = normalize_inputs(inputs)
        x = self.model(inputs, mask)
        return x


class DeepSet_wGCN_variable(nn.Module):
    def __init__(self, hidden_layers, gcn_layers, num_features, units=32):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.gcn_layers = gcn_layers

        # Input layer
        if gcn_layers and gcn_layers[0] == 1:
            self.input_layer = GCN(num_features, units)
        else:
            self.input_layer = nn.Linear(num_features, units)

        # Hidden layers
        self.layers = nn.ModuleList()
        for i in range(hidden_layers):
            if (i+2) in gcn_layers:
                self.layers.append(GCN(units, units))
            else:
                self.layers.append(nn.Linear(units, units))

        # Global MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 1)
        )

    def forward(self, inputs, mask=None):
        adj = inputs["adj"]
        feat = inputs["feat"]

        adj = normalize_adjacency(adj)

        if self.gcn_layers and self.gcn_layers[0] == 1:
            x = F.relu(self.input_layer(feat,adj))
        else:
            x = F.relu(self.input_layer(feat))

        for layer in self.layers:
            if isinstance(layer, GCN):
                x = F.relu(layer(x, adj))
            else:
                x = F.relu(layer(x))

        if mask is not None:
            x = masked_average(x, mask)
        else:
            x = x.mean(axis=-2)

        return self.global_mlp(x)

class CombinedModel_wGCN_variable(nn.Module):
    def __init__(self, num_features=8, embed_dim=8, num_pdg_ids=len(PDG_MAPPING), units=32):
        super().__init__()
        self.embedding = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.deep_set = DeepSet_wGCN_variable(num_features=num_features + embed_dim, units=units, layer_in="linear", hidden_layers=6, gcn_layers=[0,1,2,3,4,5])

    def forward(self, inputs, mask=None):
        pdg = inputs["pdg"]
        feat = inputs["feat"]
        emb = self.embedding(pdg)
        x = torch.cat([feat, emb], -1)
        return self.deep_set(dict(feat=x, adj=inputs["adj"]), mask=mask)

def from_config(config):
    """
    Mapping of model_name to model (useful for streamlining studies)
    """
    models = {
        "deepset": DeepSet,
        "deepset_combined": CombinedModel,
        "deepset_gcn": DeepSet_wGCN,
        "deepset_combined_wgcn": CombinedModel_wGCN,
        "optimal_model": OptimalModel,
        "deepset_wgcn_variable": DeepSet_wGCN_variable,
        "transformer": TransformerModel,
        "deepset_combined_wgcn_normalized": CombinedModel_wGCN_Normalized
    }
    config = config.copy()
    return models[config.pop("model_name")](**config)

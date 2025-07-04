import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def normalize_adjacency(adj):
    """
    Normalizes an adjacency matrix as proposed in Kipf & Welling (https://arxiv.org/abs/1609.02907)

    The scalefactor for each entry is given by 1 / c_ij
    where c_ij = sqrt(N_i) * sqrt(N_j)
    where N_i and N_j are the number of neighbors (Node degrees) of Node i and j.
    """
    deg_diag = adj.sum(axis=2)
    deg12_diag = torch.where(deg_diag != 0, deg_diag**-0.5, 0)
    # normalization coefficients are outer product of inverse square root of degree vector
    # gives coeffs_ij = 1 / sqrt(N_i) / sqrt(N_j)
    coeffs = deg12_diag[:, :, np.newaxis] @ deg12_diag[:, np.newaxis, :]
    return adj.float() * coeffs


class GCN(nn.Module):
    """
    Simple graph convolution. Equivalent to GCN from Kipf & Welling (https://arxiv.org/abs/1609.02907)
    when fed a normalized adjacency matrix.
    """

    def __init__(self, num_features, units):
        super().__init__()
        self.linear = nn.Linear(num_features, units)

    def forward(self, inputs, adjacency):
        return adjacency @ self.linear(inputs)

def masked_average(batch, mask):
    batch = batch.masked_fill(mask[..., np.newaxis], 0)
    sizes = (~mask).sum(axis=1, keepdim=True)
    return batch.sum(axis=1) / sizes

class OutputLayer(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(num_inputs, 1)
        )
    
    def forward(self, x):
        x = self.output_layer(x)
        return x
    
class DeepSetLayer(nn.Module):
    def __init__(self, num_features=8, units=32):
        super().__init__()
        self.per_item_mlp = nn.Sequential(
            nn.Linear(num_features, units),
            nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU()
        )

    def forward(self, x, mask=None):
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

# This works!
class DeepSet(nn.Module):
    def __init__(self, num_features=8, units=32):
        super().__init__()
        self.deep_set_layer = DeepSetLayer(num_features, units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs, mask=None):
        x = inputs["feat"]
        x = self.deep_set_layer(x, mask)
        x = self.output_layer(x)
        return x

# This works!
class CombinedModel(nn.Module):
    def __init__(self, num_features=8, embed_dim=8, num_pdg_ids=len(PDG_MAPPING), units=32):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.deep_set_layer = DeepSetLayer(num_features=num_features+ embed_dim, units=units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs, mask=None):
        pdg = inputs["pdg"]
        feat = inputs["feat"]
        emb = self.embedding_layer(pdg)
        x = torch.cat([feat, emb], -1)

        x = self.deep_set_layer(x, mask)
        x = self.output_layer(x)
        return x

# This does not work!
class GraphNetwork(nn.Module):
    """
    GCN based model with adjacency matrices and features as input
    """
    def __init__(self, num_features=7, units=32):
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
        
# This works!
class DeepSet_GCN(nn.Module):
    def __init__(self, num_features=8, units=32):
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
    
# This works!
class CombinedModel_wGCN(nn.Module):
    def __init__(self, num_features=8, embed_dim=8, num_pdg_ids=len(PDG_MAPPING), units=32, dropout_rate=0.3):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.gcn_layer = GCN(num_features + embed_dim, units)
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


class TransformerModel(nn.Module):
    def __init__(self, num_feat=8, embed_dim=8, num_pdg_ids=len(PDG_MAPPING), units=32, num_heads=4, num_layers=2, dropout_rate=0.17):
        super().__init__()

        # Particle type embedding
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)

        # Linear projection of features + embeddings
        self.input_proj = nn.Linear(num_feat + embed_dim, units)

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
        #self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output layer
        self.output_layer = nn.Linear(units, 1)

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


# Todo 09
class CombinedModel_wGCN_Normalized(nn.Module):
    def __init__(self, num_features=8, embed_dim=8, num_pdg_ids=len(PDG_MAPPING), units=32):
        super().__init__()
        self.model = CombinedModel_wGCN(
            num_features=num_features,
            embed_dim=embed_dim,
            num_pdg_ids=num_pdg_ids,
            units=units
        )

    def normalize_inputs(self, inputs):
        x = inputs["feat"]
        # x.shape = (batch_size, num_particles, num_features)
        mean = x.mean(dim=(0,1), keepdim=True) # Collapse batch and particle dimensions end up with (num_featurs) -> one mean per feature!
        std = x.std(dim=(0,1), keepdim=True) + 1e-8  # avoid divide-by-zero
        x_norm = (x - mean) / std
        return {**inputs, "feat": x_norm}

    def forward(self, inputs, mask=None):
        inputs = self.normalize_inputs(inputs)
        x = self.model(inputs, mask)
        return x
    
# Todo 04
class DeepSet_wGCN_variable(nn.Module):
    def __init__(self, hidden_layers, gcn_layers, layer_in, num_features, units=32):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.gcn_layers = gcn_layers
        self.layer_in = layer_in

        # Input layer
        if layer_in == "linear":
            self.input_layer = nn.Linear(num_features, units)
        elif layer_in == "gcn":
            self.input_layer = GCN(num_features, units)
        else:
            print("Error input layer")

        # Hidden layers
        self.layers = nn.ModuleList()
        for i in range(hidden_layers):
            if i in gcn_layers:
                self.layers.append(GCN(units, units))
            else:
                self.layers.append(nn.Linear(units, units))

        # Global MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(units, 1)
        )
        
    def forward(self, inputs, mask=None):
        adj = inputs["adj"]
        feat = inputs["feat"]

        adj = normalize_adjacency(adj)

        if self.layer_in == "linear":
            x = F.relu(self.input_layer(feat))
        elif self.layer_in == "gcn":
            x = F.relu(self.input_layer(feat, adj))

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
        self.deep_set = DeepSet_GCN_variable(num_features=num_features + embed_dim, units=units, layer_in="linear", hidden_layers=6, gcn_layers=[0,1,2,3,4,5])

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
        "gcn": GraphNetwork,
        "deepset_gcn": DeepSet_GCN,
        "deepset_combined_wgcn": CombinedModel_wGCN,
        "transformer": TransformerModel,
        "deepset_combined_wgcn_normalized" :  CombinedModel_wGCN_Normalized,
    }
    config = config.copy()
    return models[config.pop("model_name")](**config)

import json
import numpy as np
import torch
from torch import nn

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
    def __init__(self, num_feat=8, embed_dim=8, num_pdg_ids=len(PDG_MAPPING), units=32):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.deep_set_layer = DeepSetLayer(num_features=num_feat + embed_dim, units=units)
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
    def __init__(self, num_features=8, units=32):
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
    def __init__(self, num_feat=8, embed_dim=8, num_pdg_ids=len(PDG_MAPPING), units=32):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_pdg_ids + 1, embed_dim)
        self.gcn_layer = GCN(num_feat + embed_dim, units)
        self.deep_set_layer = DeepSetLayer(units, units)
        self.output_layer = OutputLayer(units)

    def forward(self, inputs, mask=None):
        pdg = inputs["pdg"]
        feat = inputs["feat"]
        adj = inputs["adj"]
        adj = normalize_adjacency(adj)

        emb = self.embedding_layer(pdg)
        x = torch.cat([feat, emb], -1)
        x = self.gcn_layer(x, adj)
        x = self.deep_set_layer(x, mask)
        x = self.output_layer(x)
        return x


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
    }
    config = config.copy()
    return models[config.pop("model_name")](**config)

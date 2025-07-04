import json
from pathlib import Path

import awkward as ak
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from utils import (
    load_data,
    get_adj,
    preprocess,
    GraphDataset,
    collate_fn,
    fit,
)

from models import from_config

# Set up the data
df, labels = load_data("smartbkg_dataset_4k.parquet", row_groups=[0])
with open("pdg_mapping.json") as f:
    pdg_mapping = dict(json.load(f))

#give variables
hidden_layers = 6
gcn_layers = [0,1,2,3,4,5]
input_layer = "linear"
coordinates = "cylindrical"

feature_columns_map = {
    "cartesian": ["prodTime", "x", "y", "z", "energy", "px", "py", "pz"],
    "cylindrical": ["r", "z", "p_xy", "pz", "prodTime", "energy"]
}

feature_columns = feature_columns_map.get(coordinates)

data = preprocess(df, pdg_mapping=pdg_mapping, feature_columns=feature_columns, coordinates = coordinates)
data["adj"] = [get_adj(index, mother) for index, mother in zip(data["index"], data["mother"])]


config = {
    "model_name": "deepset_wgcn_variable",
    "num_features": len(feature_columns),
    "units": 32,
    "hidden_layers": hidden_layers,
    "gcn_layers": gcn_layers,
    "layer_in": input_layer,
}

total_layers = hidden_layers + 2
gcn_indices = [i + 2 for i in gcn_layers]
if input_layer == "gcn":
    gcn_indices.insert(0, 1)  

gcn_str = ''.join(str(i) for i in gcn_indices) if gcn_indices else "none"
tag = f"DS_{total_layers}_GCN_{gcn_str}_{coordinates}"

# Create save path
save_path = Path("saved_models")
save_path.mkdir(exist_ok=True)
model_path = save_path / tag
model_path.mkdir(exist_ok=True)

with open(model_path / f"config.json", "w") as f:
    json.dump(config, f)

# Build model
model = from_config(config) 

# Train model
data_train = {}
data_val = {}
(
    data_train["features"], data_val["features"],
    data_train["pdg_mapped"], data_val["pdg_mapped"],
    data_train["adj"], data_val["adj"],
    y_train, y_val
) = train_test_split(data["features"], data["pdg_mapped"], data["adj"], labels)

dl_train, dl_val = [
    torch.utils.data.DataLoader(
        GraphDataset(feat=x["features"], pdg=x["pdg_mapped"], adj=x["adj"], y=y),
        batch_size=256,
        collate_fn=collate_fn
    )
    for x, y in [(data_train, y_train), (data_val, y_val)]
]

history = []
history = fit(model, dl_train, dl_val, epochs=10, history=history)

# save
df_history = pd.DataFrame(history)
df_history.to_csv(model_path / "history.csv")
torch.save(model.state_dict(), model_path / "state.pt")
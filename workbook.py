import json
from pathlib import Path

import awkward as ak
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import (
    load_data,
    get_adj,
    preprocess,
    GraphDataset,
    collate_fn,
    fit,
)
from models import from_config

def main():
    coordinates = "cartesian"

    feature_columns_map = {
        "cartesian": ["prodTime", "x", "y", "z", "energy", "px", "py", "pz"],
        "cylindrical": ["r", "z", "p_xy", "pz", "prodTime", "energy"]
    }

    # --- Device to use ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"
    print("Using device:", device)

    feature_columns = feature_columns_map.get(coordinates)

    config = {
    "model_name": "deepset_combined_wgcn_normalized",
    "embed_dim": 8,          # add embedding size
    #"dropout_rate": 0.179,     # add dropout     30 was better
    # "num_heads": 4,          # add number of heads
    # "num_layers": 2,         # add number of transformer layers
    "units": 32,
    "num_features": len(feature_columns)
    }   

    # --- Select Model ---
    tag = f'{config["model_name"]}_{coordinates}'

    print("Selected model: ", tag)

    # --- Configuration ---
    dataset_path = "smartbkg_dataset_4k_training.parquet"
    pdg_map_path = "pdg_mapping.json"
    save_path = Path("models_fulltrain")
    save_path.mkdir(exist_ok=True)
    model_path = save_path / tag
    model_path.mkdir(exist_ok=True)

    # --- Load and preprocess data ---
    print("Loading data...")
    df, labels = load_data(dataset_path, row_groups=[0,1,2,3])

    with open(pdg_map_path) as f:
        pdg_mapping = dict(json.load(f))

    print("Preprocessing...")
    data = preprocess(df, pdg_mapping=pdg_mapping, feature_columns=feature_columns, coordinates=coordinates)
    data["adj"] = [get_adj(index, mother) for index, mother in zip(data["index"], data["mother"])]

    # --- Split data ---
    print("Splitting data...")
    (
        features_train, features_val,
        pdg_train, pdg_val,
        adj_train, adj_val,
        y_train, y_val
    ) = train_test_split(data["features"], data["pdg_mapped"], data["adj"], labels)

    data_train = {"features": features_train, "pdg_mapped": pdg_train, "adj": adj_train}
    data_val = {"features": features_val, "pdg_mapped": pdg_val, "adj": adj_val}

    dl_train, dl_val = [
        torch.utils.data.DataLoader(
            GraphDataset(feat=x["features"], pdg=x["pdg_mapped"], adj=x["adj"], y=y),
            batch_size=256,
            collate_fn=collate_fn
        )
        for x, y in [(data_train, y_train), (data_val, y_val)]
    ]

    # --- Model config ---

    with open(model_path / "config.json", "w") as f:
        json.dump(config, f)

    model = from_config(config)

    # --- Train model ---
    print("Training...")
    history = []
    history = fit(model, dl_train, dl_val, epochs=10, history=history, weight_decay=1e-4, device=device)   # Adjusted weight decay so its a hyperparameters

    # --- Save history ---
    df_history = pd.DataFrame(history)
    df_history.to_csv(model_path / "history.csv", index=False)

    # --- Save model ---
    torch.save(model.state_dict(), model_path / "state.pt")
    print("Model and training history saved to:", model_path)

    df_history.plot()
    plt.title("Training History")
    plt.savefig(model_path / "history.png")
    print("History plot saved.")

if __name__ == "__main__":
    main()
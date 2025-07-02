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
    # --- Select Model ---
    tag = "deepset_combined_wgcn"
    print("Selected model: ", tag)

    # --- Configuration ---
    feature_columns = ["prodTime", "x", "y", "z", "energy", "px", "py", "pz"]
    dataset_path = "smartbkg_dataset_4k.parquet"
    pdg_map_path = "pdg_mapping.json"
    save_path = Path("saved_models")
    save_path.mkdir(exist_ok=True)
    model_path = save_path / tag
    model_path.mkdir(exist_ok=True)

    # --- Load and preprocess data ---
    print("Loading data...")
    df, labels = load_data(dataset_path, row_groups=[0])

    with open(pdg_map_path) as f:
        pdg_mapping = dict(json.load(f))

    print("Preprocessing...")
    data = preprocess(df, pdg_mapping=pdg_mapping, feature_columns=feature_columns)
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
    config = {
        "model_name": "deepset_combined_wgcn",
        "units": 32,
    }
    with open(model_path / "config.json", "w") as f:
        json.dump(config, f)

    model = from_config(config)

    # --- Train model ---
    print("Training...")
    history = []
    history = fit(model, dl_train, dl_val, epochs=10, history=history)

    # --- Save history ---
    df_history = pd.DataFrame(history)
    df_history.to_csv(model_path / "history.csv", index=False)

    # --- Save model ---
    torch.save(model.state_dict(), model_path / "state.pt")
    print("Model and training history saved to:", model_path)

    df_history.plot()
    plt.title("Deepset Combined GCN no 'p'")
    plt.savefig(model_path / "history.png")
    print("History plot saved.")

if __name__ == "__main__":
    main()
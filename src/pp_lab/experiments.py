from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .config import get_feature_columns, load_pdg_mapping, resolve_path
from .data import GraphDataset, collate_fn, get_adj, load_data, preprocess


def load_graph_data(
    dataset_path: str | Path,
    row_groups: list[int] | int | None,
    coordinates: str = "cartesian",
    feature_columns: list[str] | None = None,
    pdg_mapping_path: str | Path | None = None,
):
    features = feature_columns or get_feature_columns(coordinates)
    df, labels = load_data(str(resolve_path(dataset_path)), row_groups=row_groups)
    mapping = load_pdg_mapping() if pdg_mapping_path is None else load_pdg_mapping(pdg_mapping_path)
    data = preprocess(df, pdg_mapping=mapping, feature_columns=features, coordinates=coordinates)
    data["adj"] = [get_adj(index, mother) for index, mother in zip(data["index"], data["mother"])]
    return data, labels, features


def split_graph_data(data, labels, test_size: float = 0.25, random_state: int | None = None):
    (
        features_train,
        features_val,
        pdg_train,
        pdg_val,
        adj_train,
        adj_val,
        y_train,
        y_val,
    ) = train_test_split(
        data["features"],
        data["pdg_mapped"],
        data["adj"],
        labels,
        test_size=test_size,
        random_state=random_state,
    )
    train = {"features": features_train, "pdg_mapped": pdg_train, "adj": adj_train}
    val = {"features": features_val, "pdg_mapped": pdg_val, "adj": adj_val}
    return train, val, y_train, y_val


def make_dataloaders(train_data, val_data, y_train, y_val, batch_size: int):
    return [
        torch.utils.data.DataLoader(
            GraphDataset(feat=data["features"], pdg=data["pdg_mapped"], adj=data["adj"], y=labels),
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        for data, labels in ((train_data, y_train), (val_data, y_val))
    ]


def save_training_outputs(model_path: Path, config: dict, history: list[dict[str, float]], model) -> None:
    model_path.mkdir(parents=True, exist_ok=True)

    with (model_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(model_path / "history.csv", index=False)
    torch.save(model.state_dict(), model_path / "state.pt")

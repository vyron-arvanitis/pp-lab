from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pandas as pd
import torch


def masked_average(batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch = batch.masked_fill(mask[..., np.newaxis], 0)
    sizes = (~mask).sum(axis=1, keepdim=True)
    return batch.sum(axis=1) / sizes


def normalize_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    features = inputs["feat"]
    mean = features.mean(dim=(0, 1), keepdim=True)
    std = features.std(dim=(0, 1), keepdim=True) + 1e-8
    return {**inputs, "feat": (features - mean) / std}


def load_data(filename: str, row_groups: list[int] | int | None):
    data = ak.from_parquet(filename, row_groups=row_groups)
    labels = data.label.to_numpy()
    particles = ak.to_dataframe(data.particles, levelname=lambda level: {0: "event", 1: "particle"}[level])
    label_frame = pd.DataFrame(labels, columns=["label"])
    label_frame.index = label_frame.index.rename("event")
    return particles.join(label_frame), labels


def map_np(array: np.ndarray, mapping: dict[int, int], fallback: Any) -> np.ndarray:
    unique, inv = np.unique(array, return_inverse=True)
    mapped = np.array([mapping.get(value, fallback) for value in unique])
    return mapped[inv]


def preprocess(
    df: pd.DataFrame,
    pdg_mapping: dict[int, int],
    feature_columns: list[str],
    coordinates: str = "cartesian",
) -> dict[str, list[np.ndarray]]:
    frame = df.assign(pdg_mapped=map_np(df.pdg, pdg_mapping, fallback=len(pdg_mapping) + 1))
    if coordinates == "cylindrical":
        frame = transform_to_cylindrical(frame)

    flat = {
        "features": frame[feature_columns].to_numpy(),
        "pdg_mapped": frame["pdg_mapped"].to_numpy(),
        "index": frame["index"].to_numpy(),
        "mother": frame["mother_index"].to_numpy(),
    }

    data: dict[str, list[np.ndarray]] = {}
    for indices in frame.groupby("event").indices.values():
        for key, array in flat.items():
            data.setdefault(key, [])
            data[key].append(array[indices])
    return data


def pad_sequences(sequences: list[np.ndarray], maxlen: int | None = None) -> np.ndarray:
    if maxlen is None:
        maxlen = max(len(array) for array in sequences)
    if sequences[0].ndim == 2:
        shape = (len(sequences), maxlen, sequences[0].shape[-1])
    else:
        shape = (len(sequences), maxlen)
    batch = np.zeros(shape, dtype=sequences[0].dtype)
    for index, array in enumerate(sequences):
        batch[index, : len(array)] = array[:maxlen]
    return batch


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    degrees = adj.sum(axis=2)
    inv_sqrt = torch.where(degrees != 0, degrees**-0.5, 0)
    coeffs = inv_sqrt[:, :, np.newaxis] @ inv_sqrt[:, np.newaxis, :]
    return adj.float() * coeffs


def pad_adjacencies(adj_list: list[np.ndarray]) -> np.ndarray:
    maxlen = max(len(adj) for adj in adj_list)
    batch = np.zeros((len(adj_list), maxlen, maxlen), dtype=bool)
    for index, adj in enumerate(adj_list):
        batch[index, : len(adj), : len(adj)] = adj
    return batch


def get_adj(index: np.ndarray, mother: np.ndarray) -> np.ndarray:
    return (
        (mother[np.newaxis, :] == index[:, np.newaxis])
        | (index[np.newaxis, :] == mother[:, np.newaxis])
        | (index[np.newaxis, :] == index[:, np.newaxis])
    )


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, feat, pdg, adj, y):
        self.feat = feat
        self.pdg = pdg
        self.adj = adj
        self.y = y

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, index):
        features = {
            "feat": self.feat[index],
            "pdg": self.pdg[index],
            "adj": self.adj[index],
        }
        return features, self.y[index]


def collate_fn(inputs):
    feat, pdg, adj = [[x[key] for x, _ in inputs] for key in ["feat", "pdg", "adj"]]
    labels = [y for _, y in inputs]
    batch = {
        "feat": torch.tensor(pad_sequences(feat)),
        "pdg": torch.tensor(pad_sequences(pdg)),
        "adj": torch.tensor(pad_adjacencies(adj)),
    }
    mask = (batch["feat"] == 0).all(axis=-1)
    return batch, torch.tensor(labels), mask


def transform_to_cylindrical(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    required = {"x", "y", "px", "py"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"Cannot transform to cylindrical coordinates. Missing columns: {missing}.")

    frame["r"] = np.sqrt(frame["x"] ** 2 + frame["y"] ** 2)
    frame["p_xy"] = np.sqrt(frame["px"] ** 2 + frame["py"] ** 2)
    frame = frame.drop(columns=["x", "y", "px", "py"])
    return frame

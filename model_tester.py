import json
from pathlib import Path

import awkward as ak
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import accuracy_score

from models import from_config
from utils import (
    load_data,
    get_adj,
    preprocess,
    GraphDataset,
    collate_fn,
)

tag = "deepset_combined"
model_path = Path("saved_models") / tag

with open(model_path / "config.json") as f:
    config = json.load(f)


model = from_config(config)
model.load_state_dict(torch.load(model_path / "state.pt", map_location="cpu"))
model.eval()

df, labels = load_data("smartbkg_dataset_4k_testing.parquet", row_groups=[0,1,2,3])
with open("pdg_mapping.json") as f:
    pdg_mapping = dict(json.load(f))

coordinates = "cartesian"

feature_columns_map = {
    "cartesian": ["prodTime", "x", "y", "z", "energy", "px", "py", "pz"],
    "cylindrical": ["r", "z", "p_xy", "pz", "prodTime", "energy"]
}

feature_columns = feature_columns_map.get(coordinates)

data = preprocess(df, pdg_mapping=pdg_mapping, feature_columns=feature_columns, coordinates = coordinates)
data["adj"] = [get_adj(index, mother) for index, mother in zip(data["index"], data["mother"])]

dataset_test = GraphDataset(
    feat=data["features"],
    pdg=data["pdg_mapped"],
    adj=data["adj"],
    y=labels
)

dl_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=False
)
all_preds = []
all_labels = []


with torch.no_grad():
    for batch in dl_test:
        inputs, targets, mask = batch
        logits = model(inputs, mask=mask).squeeze(-1)
        preds = torch.sigmoid(logits)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(targets.cpu().numpy())

# Final test metrics
preds_np = np.concatenate(all_preds)
labels_np = np.concatenate(all_labels)

pred_classes = (preds_np > 0.5).astype(int)
true_classes = labels_np.astype(int)

acc = accuracy_score(true_classes, pred_classes)

df_test_metrics = pd.DataFrame([{
    "test_accuracy": acc
}])

df_test_metrics.to_csv(model_path / "test_metrics.csv", index=False)
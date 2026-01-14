import optuna
import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from models import from_config
from utils import (
    load_data,
    get_adj,
    preprocess,
    GraphDataset,
    collate_fn,
    fit,
)

# --- Data setup ---
dataset_path = "smartbkg_dataset_4k.parquet"
pdg_map_path = "pdg_mapping.json"

print("Loading data...")
df, labels = load_data(dataset_path, row_groups=[0])

with open(pdg_map_path) as f:
    pdg_mapping = dict(json.load(f))

print("Preprocessing...")
feature_columns = ["prodTime", "x", "y", "z", "energy", "px", "py", "pz"]
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
        batch_size=128,  # Fixed for now; you can tune this too
        collate_fn=collate_fn
    )
    for x, y in [(data_train, y_train), (data_val, y_val)]
]

# --- Objective function for Optuna ---
def objective(trial):
    # Suggest hyperparameters
    embed_dim = trial.suggest_int("embed_dim", 4, 32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Model config
    config = {
        "model_name": "deepset_combined_wgcn",
        "embed_dim": embed_dim,
        "dropout_rate": dropout_rate,
        "units": 32,  # Fixed for now, but could tune
    }
    model = from_config(config)

    # Train and evaluate
    history = fit(
        model,
        dl_train,
        dl_val,
        epochs=30,  # Adjust epochs for more/less tuning
        device="cuda" if torch.cuda.is_available() else "cpu",
        patience=3,  # Early stopping patience
        weight_decay=weight_decay
    )

    # Return validation loss to minimize
    best_val_loss = min(h["val_loss"] for h in history)
    return best_val_loss

# --- Run Optuna study ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)  # Adjust n_trials for more/less tuning

# --- Print results ---
print("Best trial:")
print("  Value: {:.5f}".format(study.best_value))
print("  Params: ")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# --- Save study results ---
save_path = Path("optuna_results")
save_path.mkdir(exist_ok=True)

# Save best parameters
with open(save_path / "best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)

# Save all trials to CSV
df_trials = study.trials_dataframe()
df_trials.to_csv(save_path / "trials.csv", index=False)

import matplotlib.pyplot as plt

# Plot optimization history manually
values = [trial.value for trial in study.trials]
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(values) + 1), values, marker='o')
plt.title("Optuna Optimization History")
plt.xlabel("Trial")
plt.ylabel("Validation Loss")
plt.grid(True)

# Save the figure
plt.savefig(save_path / "optimization_history.png")
plt.close()
print("Optimization history saved.")

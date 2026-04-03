from __future__ import annotations

import argparse
import json

from pp_lab.config import DEFAULT_DATASET_PATH, OPTUNA_RESULTS_DIR, ensure_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run an Optuna study over selected model hyperparameters.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--pdg-mapping-path", default="pdg_mapping.json")
    parser.add_argument("--coordinates", choices=["cartesian", "cylindrical"], default="cartesian")
    parser.add_argument("--row-groups", nargs="*", type=int, default=[0])
    parser.add_argument("--output-dir", default=str(OPTUNA_RESULTS_DIR))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import matplotlib.pyplot as plt
    import optuna
    import pandas as pd
    import torch

    from pp_lab.experiments import load_graph_data, make_dataloaders, split_graph_data
    from pp_lab.models import from_config
    from pp_lab.training import fit
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    data, labels, feature_columns = load_graph_data(
        dataset_path=args.dataset_path,
        row_groups=args.row_groups,
        coordinates=args.coordinates,
        pdg_mapping_path=args.pdg_mapping_path,
    )
    train_data, val_data, y_train, y_val = split_graph_data(data, labels)
    dl_train, dl_val = make_dataloaders(train_data, val_data, y_train, y_val, batch_size=args.batch_size)

    def objective(trial):
        config = {
            "model_name": "deepset_combined_wgcn",
            "num_features": len(feature_columns),
            "units": args.units,
            "embed_dim": trial.suggest_int("embed_dim", 4, 32),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        }
        model = from_config(config)
        history = fit(
            model,
            dl_train,
            dl_val,
            epochs=args.epochs,
            device=device,
            patience=args.patience,
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        )
        return min(record["val_loss"] for record in history)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    output_dir = ensure_dir(args.output_dir)
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(study.best_params, handle, indent=2)

    trials_frame = study.trials_dataframe()
    trials_frame.to_csv(output_dir / "trials.csv", index=False)

    values = [trial.value for trial in study.trials]
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.title("Optuna Optimization History")
    plt.xlabel("Trial")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_history.png")
    plt.close()

    print(f"Best trial value: {study.best_value:.5f}")
    print(pd.Series(study.best_params).to_string())
    print(f"Saved Optuna outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

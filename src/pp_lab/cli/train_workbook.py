from __future__ import annotations

import argparse
from pathlib import Path

from pp_lab.config import DEFAULT_DATASET_PATH, FULL_TRAIN_MODELS_DIR, ensure_dir


def build_model_config(args, num_features: int) -> dict:
    config = {
        "model_name": args.model_name,
        "num_features": num_features,
        "units": args.units,
    }

    if args.model_name in {
        "deepset_combined",
        "deepset_combined_wgcn",
        "deepset_combined_wgcn_normalized",
        "optimal_model",
        "transformer",
    }:
        config["embed_dim"] = args.embed_dim

    if args.model_name in {
        "deepset_combined",
        "deepset_combined_wgcn",
        "deepset_combined_wgcn_normalized",
        "transformer",
    }:
        config["dropout_rate"] = args.dropout_rate

    if args.model_name == "transformer":
        config["num_heads"] = args.num_heads
        config["num_layers"] = args.num_layers

    if args.model_name == "optimal_model":
        config["dropout_rate"] = args.dropout_rate
        config["negative_slope"] = args.negative_slope

    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a model on the SmartBKG dataset.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--pdg-mapping-path", default="pdg_mapping.json")
    parser.add_argument("--coordinates", choices=["cartesian", "cylindrical"], default="cartesian")
    parser.add_argument("--row-groups", nargs="*", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--model-name",
        choices=[
            "deepset",
            "deepset_combined",
            "deepset_gcn",
            "deepset_combined_wgcn",
            "deepset_combined_wgcn_normalized",
            "optimal_model",
            "transformer",
        ],
        default="deepset_combined_wgcn_normalized",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output-dir", default=str(FULL_TRAIN_MODELS_DIR))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--dropout-rate", type=float, default=0.17)
    parser.add_argument("--negative-slope", type=float, default=0.01)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    from pp_lab.experiments import load_graph_data, make_dataloaders, save_training_outputs, split_graph_data
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

    config = build_model_config(args, num_features=len(feature_columns))
    model = from_config(config)

    tag = args.tag or f"{args.model_name}_{args.coordinates}"
    model_path = ensure_dir(Path(args.output_dir) / tag)

    history = fit(
        model,
        dl_train,
        dl_val,
        epochs=args.epochs,
        device=device,
        patience=args.patience,
        weight_decay=args.weight_decay,
    )
    save_training_outputs(model_path, config, history, model)

    history_frame = pd.DataFrame(history)
    history_frame.plot()
    plt.title("Training History")
    plt.tight_layout()
    plt.savefig(model_path / "history.png")
    plt.close()
    print(f"Saved model, config, and history to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

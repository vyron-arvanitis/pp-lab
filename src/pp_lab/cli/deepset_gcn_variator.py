from __future__ import annotations

import argparse
from pathlib import Path

from pp_lab.config import DEFAULT_TRAINING_DATASET_PATH, SAVED_MODELS_DIR, ensure_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a variable DeepSet/GCN architecture.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_TRAINING_DATASET_PATH))
    parser.add_argument("--pdg-mapping-path", default="pdg_mapping.json")
    parser.add_argument("--coordinates", choices=["cartesian", "cylindrical"], default="cartesian")
    parser.add_argument("--row-groups", nargs="*", type=int, default=[0])
    parser.add_argument("--hidden-layers", type=int, default=6)
    parser.add_argument("--gcn-layers", nargs="*", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", default=str(SAVED_MODELS_DIR))
    parser.add_argument("--tag", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from pp_lab.experiments import load_graph_data, make_dataloaders, save_training_outputs, split_graph_data
    from pp_lab.models import from_config
    from pp_lab.training import fit
    data, labels, feature_columns = load_graph_data(
        dataset_path=args.dataset_path,
        row_groups=args.row_groups,
        coordinates=args.coordinates,
        pdg_mapping_path=args.pdg_mapping_path,
    )
    train_data, val_data, y_train, y_val = split_graph_data(data, labels)
    dl_train, dl_val = make_dataloaders(train_data, val_data, y_train, y_val, batch_size=args.batch_size)

    total_layers = args.hidden_layers + 2
    gcn_suffix = "".join(str(index) for index in args.gcn_layers) if args.gcn_layers else "none"
    tag = args.tag or f"DS_{total_layers}_GCN_{gcn_suffix}"
    model_path = ensure_dir(Path(args.output_dir) / tag)

    config = {
        "model_name": "deepset_wgcn_variable",
        "num_features": len(feature_columns),
        "units": args.units,
        "hidden_layers": args.hidden_layers,
        "gcn_layers": args.gcn_layers,
    }
    model = from_config(config)
    history = fit(model, dl_train, dl_val, epochs=args.epochs)
    save_training_outputs(model_path, config, history, model)
    print(f"Saved variable-architecture run to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

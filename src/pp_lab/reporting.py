from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import MODEL_COMPARISON_DIR, SAVED_MODELS_DIR, ensure_dir, resolve_path

X_AXIS_LABELS = {
    "deepset_combined_wgcn": "all features",
    "deepset_combined_wgcn_no_E": "no E",
    "deepset_combined_wgcn_no_p": r"no $\vec{p}$",
    "deepset_combined_wgcn_no_prodTime": "no prodTime",
    "deepset_combined_wgcn_no_x": "no x",
    "deepset_combined_wgcn_no_x_y": "no x/y",
    "deepset_combined_wgcn_no_x_y_z": "no x/y/z",
    "deepset_combined_wgcn_normalized": "normalized",
    "deepset_combined_wgcn_reversed": "reversed",
    "deepset": "DeepSet",
    "deepset_combined": "CombinedModel",
    "deepset_gcn": "DeepSet_wGCN",
    "transformer": "TransformerModel",
    "DS_3_GCN_1": "3 layers (GCN: 1)",
    "DS_4_GCN_1": "4 layers (GCN: 1)",
    "DS_4_GCN_12": "4 layers (GCN: 1,2)",
    "DS_4_GCN_123": "4 layers (GCN: 1,2,3)",
    "DS_4_GCN_2": "4 layers (GCN: 2)",
    "DS_4_GCN_23": "4 layers (GCN: 2,3)",
    "DS_4_GCN_3": "4 layers (GCN: 3)",
    "DS_5_GCN_1": "5 layers (GCN: 1)",
    "DS_5_GCN_14": "5 layers (GCN: 1,4)",
    "DS_5_GCN_1234": "5 layers (GCN: 1-4)",
    "DS_6_GCN_1": "6 layers (GCN: 1)",
    "DS_6_GCN_135": "6 layers (GCN: 1,3,5)",
    "DS_7_GCN_1": "7 layers (GCN: 1)",
    "DS_7_GCN_1357": "7 layers (GCN: 1,3,5,7)",
    "DS_8_GCN_234567": "8 layers (GCN: 2-7)",
    "DS_8_GCN_234567_cylindrical": "8 layers (GCN: 2-7),\ncylindrical",
}

CATEGORIES = {
    "feature_importance": [
        "deepset_combined_wgcn",
        "deepset_combined_wgcn_no_E",
        "deepset_combined_wgcn_no_p",
        "deepset_combined_wgcn_no_prodTime",
        "deepset_combined_wgcn_no_x",
        "deepset_combined_wgcn_no_x_y",
        "deepset_combined_wgcn_no_x_y_z",
        "deepset_combined_wgcn_normalized",
        "deepset_combined_wgcn_reversed",
    ],
    "best_models": [
        "deepset",
        "deepset_combined",
        "deepset_gcn",
        "deepset_combined_wgcn",
        "transformer",
        "deepset_combined_wgcn_normalized",
    ],
    "graph_variants": [name for name in X_AXIS_LABELS if name.startswith("DS_")],
}


def compare_models(
    category: str,
    base_path: str | Path = SAVED_MODELS_DIR,
    output_dir: str | Path = MODEL_COMPARISON_DIR,
):
    if category not in CATEGORIES:
        valid = ", ".join(sorted(CATEGORIES))
        raise ValueError(f"Unknown category '{category}'. Expected one of: {valid}.")

    model_root = resolve_path(base_path)
    summary = {}
    for model_name in CATEGORIES[category]:
        history_file = model_root / model_name / "history.csv"
        if history_file.exists():
            frame = pd.read_csv(history_file)
            summary[model_name] = {
                "best_val_loss": frame["val_loss"].min(),
                "best_val_acc": frame["val_acc"].max(),
            }
        else:
            print(f"Missing: {history_file}")

    summary_frame = pd.DataFrame.from_dict(summary, orient="index").sort_values("best_val_loss")
    summary_frame["x_axis_label"] = summary_frame.index.map(X_AXIS_LABELS)

    plt.figure(figsize=(10, 5))
    x = range(len(summary_frame))
    plt.scatter(x, summary_frame["best_val_loss"], marker="o", label="Best Val Loss")
    plt.scatter(x, summary_frame["best_val_acc"], marker="s", label="Best Val Acc")
    plt.xticks(x, summary_frame["x_axis_label"], rotation=45, ha="center")
    plt.axhline(summary_frame["best_val_loss"].min(), color="red", linestyle="--", label="Min Val Loss")
    plt.axhline(summary_frame["best_val_acc"].max(), color="blue", linestyle="--", label="Max Val Acc")
    plt.ylabel("Metric Value")
    plt.title(f"Validation Metrics - {category.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = ensure_dir(output_dir) / f"{category}_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")
    return output_path, summary_frame

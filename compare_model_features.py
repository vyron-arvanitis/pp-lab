import matplotlib.pyplot as plt
import os
import pandas as pd

base_path = "saved_models"

# Mapping: model_name -> x-axis label
x_axis_map = {
    # Feature importance
    "deepset_combined_wgcn": "all features",
    "deepset_combined_wgcn_no_E": "no E",
    "deepset_combined_wgcn_no_p": r"no $\vec{p}$",
    "deepset_combined_wgcn_no_prodTime": "no prodTime",
    "deepset_combined_wgcn_no_x": "no x",
    "deepset_combined_wgcn_no_x_y": "no x/y",
    "deepset_combined_wgcn_no_x_y_z": "no x/y/z",
    "deepset_combined_wgcn_normalize": "normalized",
    "deepset_combined_wgcn_reversed": "reversed",

    # Best models
    "deepset": "DeepSet",
    "deepset_combined": "CombinedModel",
    "deepset_gcn": "DeepSet_wGCN",
    "deepset_combined_wgcn": "CombinedModel_wGCN",
    "transformer": "TransformerModel",
    "deepset_combined_wgcn_normalized": "CombinedModel_wGCN_Normalized",

    # Graph layer variants
    "DS_3_GCN_1": "3 layers (GCN: 1)",
    "DS_4_GCN_1": "4 layers (GCN: 1)",
    "DS_4_GCN_12": "4 layers (GCN: 1,2)",
    "DS_4_GCN_123": "4 layers (GCN: 1,2,3)",
    "DS_4_GCN_2": "4 layers (GCN: 2)",
    "DS_4_GCN_23": "4 layers (GCN: 2,3)",
    "DS_4_GCN_3": "4 layers (GCN: 3)",
    "DS_5_GCN_1": "5 layers (GCN: 1)",
    "DS_5_GCN_14": "5 layers (GCN: 1,4)",
    "DS_5_GCN_1234": "5 layers (GCN: 1–4)",
    "DS_6_GCN_1": "6 layers (GCN: 1)",
    "DS_6_GCN_135": "6 layers (GCN: 1,3,5)",
    "DS_7_GCN_1": "7 layers (GCN: 1)",
    "DS_7_GCN_1357": "7 layers (GCN: 1,3,5,7)",
    "DS_8_GCN_234567": "8 layers (GCN: 2–7)",
    "DS_8_GCN_234567_cylindrical": f"8 layers (GCN: 2–7), \ncylindrical"
}

# Categories
categories = {
    "feature_importance": [
        "deepset_combined_wgcn",
        "deepset_combined_wgcn_no_E",
        "deepset_combined_wgcn_no_p",
        "deepset_combined_wgcn_no_prodTime",
        "deepset_combined_wgcn_no_x",
        "deepset_combined_wgcn_no_x_y",
        "deepset_combined_wgcn_no_x_y_z",
        "deepset_combined_wgcn_normalize",
        "deepset_combined_wgcn_reversed"
    ],
    "best_models": [
        "deepset",
        "deepset_combined",
        "deepset_gcn",
        "deepset_combined_wgcn",
        "transformer",
        "deepset_combined_wgcn_normalized"
    ],
    "graph_variants": [
        key for key in x_axis_map.keys() if key.startswith("DS_")
    ]
}


def compare_models(category="feature_importance"):
    assert category in categories, f"Unknown category '{category}'"

    model_list = categories[category]
    summary = {}

    for model_name in model_list:
        model_path = os.path.join(base_path, model_name)
        history_file = os.path.join(model_path, "history.csv")

        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            best_val_loss = df["val_loss"].min()
            best_val_acc = df["val_acc"].max()
            summary[model_name] = {
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc
            }
        else:
            print(f"Missing: {model_name}/history.csv")

    # Prepare DataFrame
    summary_df = pd.DataFrame.from_dict(summary, orient="index").sort_values("best_val_loss")
    summary_df["x_axis_mapping"] = summary_df.index.map(x_axis_map)

    # Plotting
    plt.figure(figsize=(10, 5))
    x = range(len(summary_df))
    plt.scatter(x, summary_df["best_val_loss"], marker="o", label="Best Val Loss")
    plt.scatter(x, summary_df["best_val_acc"], marker="s", label="Best Val Acc")
    plt.xticks(x, summary_df["x_axis_mapping"], rotation=45, ha="center")

    plt.axhline(y=summary_df["best_val_loss"].min(), color='red', linestyle='--', label='Min Val Loss')
    plt.axhline(y=summary_df["best_val_acc"].max(), color='blue', linestyle='--', label='Max Val Acc')

    plt.ylabel("Metric Value")
    plt.title(f"Validation Metrics - {category.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_dir = "compare_model_features"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{category}_comparison.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


# Example usage:
compare_models("feature_importance")
compare_models("best_models")
compare_models("graph_variants")

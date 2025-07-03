import matplotlib.pyplot as plt
import os 
import pandas as pd

base_path = "saved_models"
summary = {}



x_axis_map = {
    "deepset_combined_wgcn": "all features",
    "deepset_combined_wgcn_no_E": "no E",
    "deepset_combined_wgcn_no_p": r"no $\vec{p}$",
    "deepset_combined_wgcn_no_prodTime": "no prodTime",
    "deepset_combined_wgcn_no_x_y": "no x/y",
    "deepset_combined_wgcn_no_x_y_z": "no x/y/z",
    "deepset_combined_wgcn_no_x"     : "no x",
    "deepset_combined_wgcn_normalize" : "norm",
    "deepset_combined_wgcn_reversed" : "reversed"
}


for model_name in os.listdir(base_path):
    if "_wgcn" not in model_name:
        continue

    model_path = os.path.join(base_path, model_name)
    history_file = os.path.join(model_path, 'history.csv')

    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        best_val_loss = df["val_loss"].min()
        best_val_acc = df["val_acc"].max()
        summary[model_name] = {
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc
        }
    else:
        print(f"Attention! history.csv not found in {model_name}")

# Convert summary into a DataFrame
summary_df = pd.DataFrame.from_dict(summary, orient="index").sort_values("best_val_loss")
summary_df["x_axis_mapping"] = summary_df.index.map(x_axis_map)
print(summary_df)


# Get the maximum validation loss and validation accuracy 

max_val_loss= summary_df["best_val_loss"].min()
max_val_acc= summary_df["best_val_acc"].max()

# Draw vertical line between them on y-axis at a fixed x position (e.g., to the right of the plot)


# Plot the values
plt.figure(figsize=(10, 5))
x = range(len(summary_df))
plt.scatter(x, summary_df["best_val_loss"], marker="o", label="Best Val Loss")
plt.scatter(x, summary_df["best_val_acc"], marker="s", label="Best Val Acc")
plt.xticks(x, summary_df["x_axis_mapping"], rotation=0, ha="center")

plt.axhline(y=max_val_loss, color='red', linestyle='--', label='Min Val Loss')
plt.axhline(y=max_val_acc, color='blue', linestyle='--', label='Max Val Acc')

plt.ylabel("Metric Value")
plt.title("Best Val Loss and Accuracy per _wgcn Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compare_model_features/comparison.png")

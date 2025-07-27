import numpy as np
import pandas as pd
import awkward as ak
import torch
import torch.nn.functional as F


def load_data(filename, row_groups):
    "Load course data into a pandas dataframe. Also returns the labels for each event as a numpy array."
    data = ak.from_parquet(filename, row_groups=row_groups)
    labels = data.label.to_numpy()
    df_particles = ak.to_dataframe(data.particles, levelname=lambda i: {0: "event", 1: "particle"}[i])
    df_label = pd.DataFrame(labels, columns=["label"])
    df_label.index = df_label.index.rename("event")
    df = df_particles.join(df_label)
    return df, labels


def map_np(array, mapping, fallback):
    """
    Apply a mapping over a numpy array - along the lines of
    https://stackoverflow.com/a/16993364
    """
    # inv is the original array with the values replaced by their indices in the unique array
    unique, inv = np.unique(array, return_inverse=True)
    np_mapping = np.array([mapping.get(x, fallback) for x in unique])
    return np_mapping[inv]


def preprocess(df, pdg_mapping, feature_columns, coordinates="cartesian"):
    """
    Preprocess data from pandas DataFrame and return dictionary of flat numpy arrays per event.

    Args:
        df (pd.DataFrame): Input data containing particle features and metadata
        pdg_maping (dict): Mapping from pdg ids to token ids
        feature_columns (list): Columns to use as input features
        coordinates (str): Coordinate system to use for features 
                Must be either "cartesian" (default), or "cylindrical".
                "cylindrical" assumes cylindrical symmetry, excluding angular coordinates
    """
    df = df.assign(pdg_mapped=map_np(df.pdg, pdg_mapping, fallback=len(pdg_mapping) + 1))
    if coordinates == "cartesian":
        flat = {
        "features": df[feature_columns].to_numpy(),
        "pdg_mapped": df["pdg_mapped"].to_numpy(),
        "index": df["index"].to_numpy(),
        "mother": df["mother_index"].to_numpy(),
        }
    elif coordinates == "cylindrical":
        flat = {
        "features": transform_to_cylindrical(df)[feature_columns].to_numpy(),
        "pdg_mapped": df["pdg_mapped"].to_numpy(),
        "index": df["index"].to_numpy(),
        "mother": df["mother_index"].to_numpy(),
        }
    
    data = {}
    for idx in df.groupby("event").indices.values():
        for k, array in flat.items():
            data.setdefault(k, [])
            data[k].append(array[idx])
    return data


def pad_sequences(sequences, maxlen=None):
    """
    Converts a list of sequences to a numpy array with fixed length on the
    sequence dimension (second to last) where entries for sequences shorter than
    `maxlen` are padded with zeros. If maxlen is not given, it is determined
    from the list of sequences.

    Similar to https://keras.io/api/preprocessing/timeseries/#padsequences-function
    but works for 2D arrays as well
    """
    if maxlen is None:
        maxlen = max(len(array) for array in sequences)
    if sequences[0].ndim == 2:
        shape = (len(sequences), maxlen, sequences[0].shape[-1])
    else:
        shape = (len(sequences), maxlen)
    batch = np.zeros(shape, dtype=sequences[0].dtype)
    for i, array in enumerate(sequences):
        batch[i, : len(array)] = array[:maxlen]
    return batch


def pad_adjacencies(adj_list):
    """
    Converts a sequence of adjacency matrices to a 3D numpy array of fixed
    matrix size with zero padded entries
    """
    maxlen = max(len(adj) for adj in adj_list)
    batch = np.zeros((len(adj_list), maxlen, maxlen), dtype=bool)
    for i, adj in enumerate(adj_list):
        batch[i, : len(adj), : len(adj)] = adj
    return batch


def get_adj(index, mother):
    """
    Construct adjacency matrix from arrays of indices and mother indices

    Returns the adjacency matrix with mother-daughter and
    daughter-mother relations as well as self-loops (diagonal is 1) as an array
    """
    return (
        (mother[np.newaxis, :] == index[:, np.newaxis]) # mother-daughter relations
        | (index[np.newaxis, :] == mother[:, np.newaxis]) # daughter-mother relations
        | (index[np.newaxis, :] == index[:, np.newaxis]) # self loops
    )


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, feat, pdg, adj, y):
        self.feat = feat
        self.pdg = pdg
        self.adj = adj
        self.y = y

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, i):
        x = {
            "feat": self.feat[i],
            "pdg": self.pdg[i],
            "adj": self.adj[i]
        }
        y = self.y[i]
        return x, y


def collate_fn(inputs):
    feat, pdg, adj = [
        [x[key] for x, y in inputs] for key in ["feat", "pdg", "adj"]
    ]
    y = [y for x, y in inputs]
    x = {
        "feat": torch.tensor(pad_sequences(feat)),
        "pdg": torch.tensor(pad_sequences(pdg)),
        "adj": torch.tensor(pad_adjacencies(adj)),
    }
    y = torch.tensor(y)
    mask = (x["feat"] == 0).all(axis=-1)
    return x, y, mask


def loss_fn(logits, y):
    return F.binary_cross_entropy_with_logits(logits.squeeze(), y.float())


def accuracy_fn(logits, y):
    return ((logits.squeeze().sigmoid() >= 0.5) == y).float().mean()


def fit(model, dl_train, dl_val, epochs=50, device="cpu", history=None, patience=5, weight_decay=1e-5):
    """
    Train a model with optional early stopping.

    Args:
        model: PyTorch model to train
        dl_train: DataLoader for training set
        dl_val: DataLoader for validation set
        epochs: Total number of epochs to train
        device: Device to use ("cpu" or "cuda")
        history: Existing history to continue training
        patience: Number of epochs to wait for improvement before stopping
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay) # Adam optimizer with weight decay as ahyperparameter

    best_val_loss = float("inf")
    patience_counter = 0

    def train_step(x, y, mask):
        model.train()
        optimizer.zero_grad()
        logits = model(x, mask=mask)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        return logits.detach().cpu(), loss.detach().cpu()

    def test_step(x, y, mask):
        model.eval()
        with torch.no_grad():
            logits = model(x, mask=mask)
            return logits.cpu(), loss_fn(logits, y.to(device)).cpu()

    def to_device(x, y, mask):
        if isinstance(x, dict):
            x = {k: v.to(device) for k, v in x.items()}
        else:
            x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        return x, y, mask

    if history is None:
        history = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # --- Training ---
        train_loss = []
        train_acc = []
        for i, (x, y, mask) in enumerate(dl_train):
            x, y, mask = to_device(x, y, mask)
            logits, loss = train_step(x, y, mask)
            train_loss.append(float(loss))
            train_acc.append(float(accuracy_fn(logits, y)))

        # --- Validation ---
        val_loss = []
        val_acc = []
        for x, y, mask in dl_val:
            x, y, mask = to_device(x, y, mask)
            logits, loss = test_step(x, y, mask)
            val_loss.append(float(loss))
            val_acc.append(float(accuracy_fn(logits, y)))

        # --- Epoch summary ---
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        avg_train_acc = np.mean(train_acc)
        avg_val_acc = np.mean(val_acc)
        print(
            f"Train loss: {avg_train_loss:.4f}, Train acc: {avg_train_acc:.4f} | "
            f"Val loss: {avg_val_loss:.4f}, Val acc: {avg_val_acc:.4f}"
        )

        history.append({
            "loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "acc": avg_train_acc,
            "val_acc": avg_val_acc,
        })

        # --- Early stopping logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return history

def transform_to_cylindrical(df):
        '''
        Transform Cartesian coordinates and momenta in a DataFrame to cylindrical form.

        Args: 
            df (pd.DataFrame): A pandas Dataframe containing at least the columns 
                'x', 'y', 'z', 'px', 'py', 'pz', 'prodTime', and 'energy'

        Returns:
            pd.Dataframe: A DataFrame with cylindrical features:
                 'r', 'z', 'p_xy', 'pz', 'prodTime', and 'energy'

        Note: 
            The angular coordinate is not included, assuming cylindrical symmetry of the system
        '''
        df = df.copy()

        # Compute radial position and transverse momentum
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        px, py = df["px"].to_numpy(), df["py"].to_numpy()

        df["r"] = np.sqrt(x**2 + y**2)
        df["p_xy"] = np.sqrt(px**2 + py**2)

        # Select and return processed features as a NumPy array or tensor
        feats = df[["r", "z", "p_xy", "pz", "prodTime", "energy"]]
        return feats
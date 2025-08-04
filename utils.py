import numpy as np
import pandas as pd
import awkward as ak
import torch
import torch.nn.functional as F

def masked_average(batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the average over a masked batch of sequences.

    Parameters
    ----------
    batch : torch.Tensor
        A tensor of shape (batch_size, num_particles, num_features) containing the data.
    mask : torch.Tensor
        A boolean tensor of shape (batch_size, num_particles) where `True` indicates
        elements to ignore (mask out) and `False` indicates valid entries.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_features) containing the mean of the unmasked
        entries for each sample in the batch.
    """

    batch = batch.masked_fill(mask[..., np.newaxis], 0)
    sizes = (~mask).sum(axis=1, keepdim=True)
    return batch.sum(axis=1) / sizes

def normalize_inputs(inputs: dict) -> dict:
    """
    Normalize per-feature values across all particles and batches

    Parameters
    ----------
    inputs : dict
        A dictionary containing the inputs to be normalized

    Returns
    -------
    dict
        A copy of the input dictionary with the inputs tensor normalized.
    """

    x = inputs["feat"]
    # x.shape = (batch_size, num_particles, num_features)
    mean = x.mean(dim=(0,1), keepdim=True) # Collapse batch and particle dimensions end up with (num_featurs) -> one mean per feature!
    std = x.std(dim=(0,1), keepdim=True) + 1e-8  # avoid divide-by-zero
    x_norm = (x - mean) / std
    return {**inputs, "feat": x_norm}

def load_data(filename: str, row_groups: list):
    """Load course data into a pandas dataframe. 
    Also returns the labels for each event as a numpy array.

    Parameters
    ----------
    filename : str
        Path to the input parquet file.
    row_groups : list or int or None
        Row groups to load from the parquet file (passed to awkward).

    Returns
    -------
    df : pd.DataFrame
        A pandas DataFrame containing all particles, indexed by (event, particle).
    labels : np.ndarray
        Array of event-level labels (shape: [num_events]).
    """

    data = ak.from_parquet(filename, row_groups=row_groups)
    labels = data.label.to_numpy()
    df_particles = ak.to_dataframe(data.particles, levelname=lambda i: {0: "event", 1: "particle"}[i])
    df_label = pd.DataFrame(labels, columns=["label"])
    df_label.index = df_label.index.rename("event")
    df = df_particles.join(df_label)
    return df, labels


def map_np(array: np.ndarray, mapping: dict, fallback: Any):
    """
    Apply a mapping over a numpy array - along the lines of
    https://stackoverflow.com/a/16993364

    Parameters
    ----------
    array : np.ndarray
        Input array of values to be mapped.
    mapping : dict
        Mapping from unique values in array to new values.
    fallback : Any
        Value to assign when an input is not present in the mapping.

    Returns
    -------
    np.ndarray
        Array of mapped values, same shape as input.
    """

    # inv is the original array with the values replaced by their indices in the unique array
    unique, inv = np.unique(array, return_inverse=True)
    np_mapping = np.array([mapping.get(x, fallback) for x in unique])
    return np_mapping[inv]

def preprocess(df, pdg_mapping, feature_columns, coordinates="cartesian"):
    """
    Preprocess data from pandas DataFrame and return dictionary of flat numpy arrays per event.

    Parameters
    ----------
        df (pd.DataFrame): Input data containing particle features and metadata
        pdg_maping (dict): Mapping from pdg ids to token ids
        feature_columns (list): Columns to use as input features
        coordinates (str): Coordinate system to use for features 
                Must be either "cartesian" (default), or "cylindrical".
                "cylindrical" assumes cylindrical symmetry, excluding angular coordinates
    Returns
    -------
    dict
        Dictionary with keys "features", "pdg_mapped", "index", and "mother",
        each mapped to a list of arrays (one per event).
    """

    df = df.assign(pdg_mapped=map_np(df.pdg, pdg_mapping, fallback=len(pdg_mapping) + 1))
    
    if coordinates == "cylindrical":
        df = transform_to_cylindrical(df)

    flat = {
        "features": df[feature_columns].to_numpy(),
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


def pad_sequences(sequences: list[np.ndarray], maxlen=None):
    """
    Converts a list of sequences to a numpy array with fixed length on the
    sequence dimension (second to last) where entries for sequences shorter than
    `maxlen` are padded with zeros. If maxlen is not given, it is determined
    from the list of sequences.

    Similar to https://keras.io/api/preprocessing/timeseries/#padsequences-function
    but works for 2D arrays as well


    Parameters
    ----------
    sequences : list of np.ndarray
        List of 1D or 2D arrays (variable length).
    maxlen : int, optional
        Length to pad all sequences to. If None, uses the length of the longest sequence.

    Returns
    -------
    np.ndarray
        Array of shape (batch, maxlen, ...) with shorter sequences padded with zeros.
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

def normalize_adjacency(adj: torch.Tensor):
    """
    Normalizes an adjacency matrix as proposed in Kipf & Welling (https://arxiv.org/abs/1609.02907)

    The scalefactor for each entry is given by 1 / c_ij
    where c_ij = sqrt(N_i) * sqrt(N_j)
    where N_i and N_j are the number of neighbors (Node degrees) of Node i and j.

    Parameters
    ----------
    adj : torch.Tensor
        Batch of adjacency matrices of shape (batch_size, num_nodes, num_nodes).

    Returns
    -------
    torch.Tensor
        Normalized adjacency matrices of the same shape as input.
    """

    deg_diag = adj.sum(axis=2)
    deg12_diag = torch.where(deg_diag != 0, deg_diag**-0.5, 0)
    # normalization coefficients are outer product of inverse square root of degree vector
    # gives coeffs_ij = 1 / sqrt(N_i) / sqrt(N_j)
    coeffs = deg12_diag[:, :, np.newaxis] @ deg12_diag[:, np.newaxis, :]
    return adj.float() * coeffs

def pad_adjacencies(adj_list: list[np.ndarray]):
    """
    Converts a sequence of adjacency matrices to a 3D numpy array of fixed
    matrix size with zero padded entries

    Parameters
    ----------
    adj_list : list of np.ndarray
        List of (num_nodes, num_nodes) boolean adjacency matrices, one per event.

    Returns
    -------
    np.ndarray
        3D array of shape (batch_size, max_nodes, max_nodes), zero-padded.
    """

    maxlen = max(len(adj) for adj in adj_list)
    batch = np.zeros((len(adj_list), maxlen, maxlen), dtype=bool)
    for i, adj in enumerate(adj_list):
        batch[i, : len(adj), : len(adj)] = adj
    return batch


def get_adj(index: np.ndarray, mother: np.ndarray) -> np.ndarray:
    """
    Construct adjacency matrix from arrays of indices and mother indices

    Returns the adjacency matrix with mother-daughter and
    daughter-mother relations as well as self-loops (diagonal is 1) as an array

    Parameters
    ----------
    index : np.ndarray
        Array of particle indices.
    mother : np.ndarray
        Array of mother indices, same length as index.

    Returns
    -------
    np.ndarray
        Boolean adjacency matrix of shape (num_particles, num_particles).
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
    """
    Collate function for DataLoader: pads all sequences and adjacencies in a batch.

    Parameters
    ----------
    inputs : list of tuple
        Each tuple is (x, y), where x is a dict with keys "feat", "pdg", "adj"
        and y is a label.

    Returns
    -------
    x : dict
        Dictionary of batch tensors ("feat", "pdg", "adj").
    y : torch.Tensor
        Tensor of batch labels.
    mask : torch.Tensor
        Boolean mask indicating which particles are padding.
    """
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


def loss_fn(logits: torch.Tensor, y: torch.Tensor):
    """
    Compute the binary cross-entropy loss between logits and targets.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs (logits), shape (batch_size,) or (batch_size, 1).
    y : torch.Tensor
        Ground truth binary labels, shape (batch_size,) or (batch_size, 1).

    Returns
    -------
    torch.Tensor
        Scalar loss (average over batch).
    """
    return F.binary_cross_entropy_with_logits(logits.squeeze(), y.float())


def accuracy_fn(logits: torch.Tensor, y: torch.Tensor):
        """
    Compute accuracy given logits and targets.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs (logits).
    y : torch.Tensor
        Ground truth binary labels.

    Returns
    -------
    float
        Fraction of correct predictions (between 0 and 1).
    """

    return ((logits.squeeze().sigmoid() >= 0.5) == y).float().mean()


def fit(model, dl_train, dl_val, epochs=50, device="cpu", history=None, patience=5, weight_decay=1e-5):
    """
    Train a model with optional early stopping.

    Parameters
    ----------
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
            train_acc.append(float(accuracy_fn(logits, y.to(device))))

        # --- Validation ---
        val_loss = []
        val_acc = []
        for x, y, mask in dl_val:
            x, y, mask = to_device(x, y, mask)
            logits, loss = test_step(x, y, mask)
            val_loss.append(float(loss))
            val_acc.append(float(accuracy_fn(logits, y.to(device))))

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

        Parameters
        ----------
            df (pd.DataFrame): A pandas Dataframe containing at least the columns 
                'x', 'y', and 'px', 'py'.

        Returns
        -------
            pd.Dataframe: A DataFrame with cylindrical features:
                 'r', 'p_xy', replacing 'x', 'y', and 'px', 'py'

        Note: 
            The angular coordinate is not included, assuming rotational symmetry of the system
        '''
        df = df.copy()
        
        if 'x' in df.columns and 'y' in df.columns and 'px' in df.columns and 'py' in df.columns:
            df['r'] = np.sqrt(df['x']**2 + df['y']**2)
            df['p_xy'] = np.sqrt(df['px']**2 + df['py']**2)
            df.drop(['px','py'], axis = 1, inplace=True)
            df.drop(['x','y'], axis = 1, inplace=True)

        else:
            print("Missing 'x', 'y', 'px', or 'py' column")
    
        return df
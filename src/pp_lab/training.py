from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.squeeze(), y.float())


def accuracy_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((logits.squeeze().sigmoid() >= 0.5) == y).float().mean()


def fit(
    model,
    dl_train,
    dl_val,
    epochs: int = 50,
    device: str = "cpu",
    history: list[dict[str, float]] | None = None,
    patience: int = 5,
    weight_decay: float = 1e-5,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0

    def to_device(x, y, mask):
        if isinstance(x, dict):
            x = {key: value.to(device) for key, value in x.items()}
        else:
            x = x.to(device)
        return x, y.to(device), mask.to(device)

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
            return logits.cpu(), loss_fn(logits, y).cpu()

    if history is None:
        history = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = []
        train_acc = []
        for x, y, mask in dl_train:
            x, y, mask = to_device(x, y, mask)
            logits, loss = train_step(x, y, mask)
            train_loss.append(float(loss))
            train_acc.append(float(accuracy_fn(logits, y.cpu())))

        val_loss = []
        val_acc = []
        for x, y, mask in dl_val:
            x, y, mask = to_device(x, y, mask)
            logits, loss = test_step(x, y, mask)
            val_loss.append(float(loss))
            val_acc.append(float(accuracy_fn(logits, y.cpu())))

        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        avg_train_acc = np.mean(train_acc)
        avg_val_acc = np.mean(val_acc)
        print(
            f"Train loss: {avg_train_loss:.4f}, Train acc: {avg_train_acc:.4f} | "
            f"Val loss: {avg_val_loss:.4f}, Val acc: {avg_val_acc:.4f}"
        )

        history.append(
            {
                "loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "acc": avg_train_acc,
                "val_acc": avg_val_acc,
            }
        )

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

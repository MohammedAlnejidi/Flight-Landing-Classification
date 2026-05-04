from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_classifier import LSTMClassifier
from utils.preprocessing_v2 import load_preprocessed_landing_splits


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "landing_lstm_classifier_checkpoint.pt"

EPOCHS = 50
BATCH_SIZE = 256
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8


def predict_probabilities(
    model: LSTMClassifier,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    model.eval()
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    probabilities = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            logits = model(batch)
            probabilities.append(torch.softmax(logits, dim=1).cpu())

    return torch.cat(probabilities)


def predict_classes(
    model: LSTMClassifier,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    probabilities = predict_probabilities(model, X, device, batch_size=batch_size)
    return probabilities.argmax(dim=1).numpy()


def compute_multiclass_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def train_classifier(
    model: LSTMClassifier,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[LSTMClassifier, dict[str, float]]:
    model = model.to(device)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    class_counts = torch.bincount(y_train, minlength=4).float()
    class_weights = class_counts.sum() / torch.clamp(class_counts, min=1.0)
    class_weights = (class_weights / class_weights.mean()).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mcc = float("-inf")
    best_metrics = {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "mcc": float("-inf"),
    }
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=device.type == "cuda")
            batch_y = batch_y.to(device, non_blocking=device.type == "cuda")

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_pred = predict_classes(model, X_val, device, batch_size=batch_size)
        val_scores = compute_multiclass_scores(y_val.numpy(), val_pred)

        if val_scores["mcc"] > best_val_mcc:
            best_val_mcc = val_scores["mcc"]
            best_metrics = val_scores.copy()
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{epochs} - train loss: {train_loss:.6f} - "
            f"val accuracy: {val_scores['accuracy']:.4f} - "
            f"val macro_f1: {val_scores['macro_f1']:.4f} - "
            f"val mcc: {val_scores['mcc']:.4f}"
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(best validation MCC: {best_val_mcc:.4f})."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, X_train, y_train, X_val, y_val, _, _ = load_preprocessed_landing_splits()

    model = LSTMClassifier(
        input_dim=X_train.shape[2],
        hidden_dim=HIDDEN_DIM,
        num_classes=len(splits.class_names),
    )

    print("Landing Anomaly Detection - 4-Class Classification")
    print("Using device:", device)
    print("Training shape:", X_train.shape, y_train.shape)
    print("Validation shape:", X_val.shape, y_val.shape)
    print("Classes:", splits.class_names)

    model, best_val_metrics = train_classifier(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    val_pred = predict_classes(model, X_val, device, batch_size=BATCH_SIZE)
    val_scores = compute_multiclass_scores(y_val.numpy(), val_pred)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(X_train.shape[2]),
            "hidden_dim": int(HIDDEN_DIM),
            "num_classes": len(splits.class_names),
            "feature_names": splits.feature_names,
            "class_names": splits.class_names,
            "best_val_accuracy": float(best_val_metrics["accuracy"]),
            "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
            "best_val_weighted_f1": float(best_val_metrics["weighted_f1"]),
            "best_val_mcc": float(best_val_metrics["mcc"]),
            "val_accuracy": float(val_scores["accuracy"]),
            "val_macro_f1": float(val_scores["macro_f1"]),
            "val_weighted_f1": float(val_scores["weighted_f1"]),
            "val_mcc": float(val_scores["mcc"]),
        },
        CHECKPOINT_PATH,
    )

    print("Saved checkpoint:", CHECKPOINT_PATH)
    print(f"Best validation MCC: {best_val_metrics['mcc']:.4f}")
    print(f"Best validation macro F1: {best_val_metrics['macro_f1']:.4f}")
    print(f"Best validation accuracy: {best_val_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

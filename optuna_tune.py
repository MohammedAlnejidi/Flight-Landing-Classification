from __future__ import annotations

from pathlib import Path

import torch

try:
    import optuna
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Optuna is not available in this Python environment. "
        "Install it in the same environment that runs this project with: "
        "'python -m pip install optuna'"
    ) from exc

from models.lstm_classifier import LSTMClassifier
from train_classifier import train_classifier
from utils.preprocessing_v2 import load_preprocessed_landing_splits


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation"

N_TRIALS = 15
TUNING_EPOCHS = 12


def objective(trial: optuna.Trial) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, X_train, y_train, X_val, y_val, _, _ = load_preprocessed_landing_splits()

    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    num_layers = trial.suggest_int("num_layers", 1, 2)

    model = LSTMClassifier(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=len(splits.class_names),
    )

    _, best_metrics = train_classifier(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        device,
        epochs=TUNING_EPOCHS,
        batch_size=batch_size,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return float(best_metrics["mcc"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction="maximize", study_name="landing_lstm_mcc_tuning")
    study.optimize(objective, n_trials=N_TRIALS)

    print("Optuna tuning complete")
    print(f"Best MCC: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    with (OUTPUT_DIR / "optuna_best_params.txt").open("w", encoding="utf-8") as file:
        file.write(f"Best MCC: {study.best_value:.6f}\n")
        for key, value in study.best_params.items():
            file.write(f"{key}: {value}\n")

    study.trials_dataframe().to_csv(OUTPUT_DIR / "optuna_trials.csv", index=False)


if __name__ == "__main__":
    main()

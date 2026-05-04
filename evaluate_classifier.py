from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef

from models.lstm_classifier import LSTMClassifier
from train_classifier import CHECKPOINT_PATH, predict_probabilities
from utils.preprocessing_v2 import load_metadata, load_preprocessed_landing_splits


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation"


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def format_confusion_matrix(confusion: np.ndarray, class_names: list[str]) -> str:
    row_labels = [f"T:{name}" for name in class_names]
    col_labels = [f"P:{name}" for name in class_names]
    first_col_width = max(len(""), max(len(label) for label in row_labels))
    col_width = max(10, max(len(label) for label in col_labels))

    lines = []
    header = " " * (first_col_width + 2) + "".join(label.rjust(col_width) for label in col_labels)
    lines.append(header)
    for row_label, row in zip(row_labels, confusion):
        lines.append(row_label.ljust(first_col_width + 2) + "".join(str(int(value)).rjust(col_width) for value in row))
    return "\n".join(lines)


def per_class_prediction_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> list[dict[str, float | int | str]]:
    summary = []
    for class_index, class_name in enumerate(class_names):
        mask = y_true == class_index
        total = int(mask.sum())
        correct = int(np.logical_and(mask, y_pred == class_index).sum())
        predicted = int((y_pred == class_index).sum())
        accuracy = 0.0 if total == 0 else correct / total
        summary.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "predicted": predicted,
                "class_accuracy": accuracy,
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the LSTM classifier for 4-class landing anomaly detection.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Path to trained checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, _, _, _, _, X_test, y_test = load_preprocessed_landing_splits()
    metadata = load_metadata()

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)

    model = LSTMClassifier(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_classes=int(checkpoint.get("num_classes", 4)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    probabilities = predict_probabilities(model, X_test, device).numpy()
    y_true = y_test.numpy().astype(int)
    y_pred = probabilities.argmax(axis=1)

    metrics = compute_multiclass_metrics(y_true, y_pred)
    class_names = checkpoint.get("class_names", ["Nominal", "Speed High", "Path High", "Flaps Late Setting"])
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred)
    per_class_summary = per_class_prediction_summary(y_true, y_pred, class_names)
    correct_predictions = int((y_true == y_pred).sum())
    total_predictions = int(len(y_true))
    wrong_predictions = total_predictions - correct_predictions

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "project": "Landing Anomaly Detection - 4-Class Classification",
        "device": str(device),
        "metrics": metrics,
        "test_sequences": int(len(X_test)),
        "correct_predictions": correct_predictions,
        "wrong_predictions": wrong_predictions,
        "class_names": class_names,
        "per_class_summary": per_class_summary,
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
        "feature_names": checkpoint.get("feature_names", []),
    }
    with (OUTPUT_DIR / "landing_classifier_summary.json").open("w", encoding="utf-8") as file:
        json.dump(result_payload, file, indent=2)

    test_metadata = metadata.iloc[splits.test_indices].reset_index(drop=True)
    predictions_df = test_metadata.copy()
    predictions_df["true_label"] = y_true
    predictions_df["predicted_label"] = y_pred
    predictions_df["true_class_name"] = [class_names[idx] for idx in y_true]
    predictions_df["predicted_class_name"] = [class_names[idx] for idx in y_pred]
    for class_index, class_name in enumerate(class_names):
        safe_name = class_name.lower().replace(" ", "_")
        predictions_df[f"prob_{safe_name}"] = probabilities[:, class_index]
    predictions_df.to_csv(OUTPUT_DIR / "landing_classifier_test_predictions.csv", index=False)

    confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
    confusion_df.to_csv(OUTPUT_DIR / "landing_classifier_confusion_matrix.csv")
    pd.DataFrame(per_class_summary).to_csv(OUTPUT_DIR / "landing_classifier_per_class_summary.csv", index=False)

    print("Landing Anomaly Detection - 4-Class Classification")
    print("Device:", device)
    print(
        "Metrics:",
        f"accuracy={metrics['accuracy']:.4f}",
        f"macro_f1={metrics['macro_f1']:.4f}",
        f"weighted_f1={metrics['weighted_f1']:.4f}",
        f"mcc={metrics['mcc']:.4f}",
    )
    print(f"Overall correct: {correct_predictions}/{total_predictions} ({metrics['accuracy']:.4f})")
    print(f"Overall wrong:   {wrong_predictions}/{total_predictions} ({wrong_predictions / total_predictions:.4f})")
    print("")
    print("Per-class results:")
    for row in per_class_summary:
        class_name = row["class_name"]
        class_report = report[class_name]
        print(
            f"  [{row['class_index']}] {class_name}: "
            f"correct={row['correct']}/{row['total']} "
            f"predicted={row['predicted']} "
            f"class_accuracy={row['class_accuracy']:.4f} "
            f"precision={class_report['precision']:.4f} "
            f"recall={class_report['recall']:.4f} "
            f"f1={class_report['f1-score']:.4f}"
        )
    print("")
    print("Confusion matrix (rows=true, cols=predicted):")
    print(format_confusion_matrix(confusion, class_names))


if __name__ == "__main__":
    main()

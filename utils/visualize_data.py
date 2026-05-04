from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing_v2 import CLASS_NAMES, FEATURE_NAMES, load_metadata, _load_raw_npz


PLOTS_DIR = PROJECT_ROOT / "plots"


def _ensure_plots_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def plot_class_balance(labels: np.ndarray, output_dir: Path) -> None:
    counts = np.bincount(labels.astype(int), minlength=len(CLASS_NAMES))
    plt.figure(figsize=(8, 5))
    bars = plt.bar(CLASS_NAMES, counts, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    plt.title("Class Balance")
    plt.ylabel("Sequence Count")
    plt.xticks(rotation=10)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(count)), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "01_class_balance.png", dpi=150)
    plt.close()


def plot_feature_distributions(data: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    selected = [3, 4, 8, 9, 11, 18]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, feature_idx in zip(axes, selected):
        feature_values = data[:, :, feature_idx].reshape(-1)
        ax.hist(feature_values, bins=60, color="#4c78a8", alpha=0.85)
        ax.set_title(FEATURE_NAMES[feature_idx], fontsize=10)
    fig.suptitle("Selected Feature Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "02_feature_distributions.png", dpi=150)
    plt.close()


def plot_sample_sequences(data: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    selected_features = [4, 8, 9, 11]
    feature_titles = [FEATURE_NAMES[idx] for idx in selected_features]
    fig, axes = plt.subplots(len(CLASS_NAMES), len(selected_features), figsize=(16, 10), sharex=True)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    for class_idx, class_name in enumerate(CLASS_NAMES):
        sample_indices = np.where(labels == class_idx)[0]
        if len(sample_indices) == 0:
            continue
        sample = data[sample_indices[0]]
        for col_idx, feature_idx in enumerate(selected_features):
            ax = axes[class_idx, col_idx]
            ax.plot(sample[:, feature_idx], color=colors[class_idx], linewidth=1.5)
            if class_idx == 0:
                ax.set_title(feature_titles[col_idx], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=10)
    fig.suptitle("One Example Sequence per Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "03_sample_sequences_by_class.png", dpi=150)
    plt.close()


def plot_feature_correlation(data: np.ndarray, output_dir: Path) -> None:
    flattened = data.reshape(-1, data.shape[-1])
    corr = np.corrcoef(flattened, rowvar=False)

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(FEATURE_NAMES)), range(len(FEATURE_NAMES)), rotation=90)
    plt.yticks(range(len(FEATURE_NAMES)), range(len(FEATURE_NAMES)))
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "04_feature_correlation_heatmap.png", dpi=150)
    plt.close()


def plot_route_distribution(metadata: pd.DataFrame, output_dir: Path) -> None:
    top_routes = (
        metadata.assign(route=metadata["departure_airport"].astype(str) + " -> " + metadata["arrival_airport"].astype(str))
        .groupby(["route", "label"])
        .size()
        .reset_index(name="count")
    )
    total_by_route = top_routes.groupby("route")["count"].sum().sort_values(ascending=False).head(10).index
    subset = top_routes[top_routes["route"].isin(total_by_route)].copy()
    pivot = subset.pivot(index="route", columns="label", values="count").fillna(0)
    pivot.columns = [CLASS_NAMES[int(col)] for col in pivot.columns]
    pivot = pivot.reindex(columns=CLASS_NAMES, fill_value=0)

    pivot.plot(kind="barh", stacked=True, figsize=(12, 7), color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    plt.title("Top 10 Routes by Sequence Count")
    plt.xlabel("Sequence Count")
    plt.ylabel("Route")
    plt.tight_layout()
    plt.savefig(output_dir / "05_top_routes_by_class.png", dpi=150)
    plt.close()


def main() -> None:
    output_dir = _ensure_plots_dir()
    data, labels = _load_raw_npz()
    metadata = load_metadata()

    plot_class_balance(labels, output_dir)
    plot_feature_distributions(data, labels, output_dir)
    plot_sample_sequences(data, labels, output_dir)
    plot_feature_correlation(data, output_dir)
    plot_route_distribution(metadata, output_dir)

    print("Landing Anomaly Detection")
    print("Saved plots to:", output_dir)
    print("Created:")
    print("  01_class_balance.png")
    print("  02_feature_distributions.png")
    print("  03_sample_sequences_by_class.png")
    print("  04_feature_correlation_heatmap.png")
    print("  05_top_routes_by_class.png")


if __name__ == "__main__":
    main()

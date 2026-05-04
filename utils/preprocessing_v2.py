from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "AILERON POSITION LH",
    "AILERON POSITION RH",
    "CORRECTED ANGLE OF ATTACK",
    "BARO CORRECT ALTITUDE LSP",
    "COMPUTED AIRSPEED LSP",
    "SELECTED COURSE",
    "DRIFT ANGLE",
    "ELEVATOR POSITION LEFT",
    "T.E. FLAP POSITION",
    "GLIDESLOPE DEVIATION",
    "SELECTED HEADING",
    "LOCALIZER DEVIATION",
    "CORE SPEED AVG",
    "TOTAL PRESSURE LSP",
    "PITCH ANGLE LSP",
    "ROLL ANGLE LSP",
    "RUDDER POSITION",
    "TRUE HEADING LSP",
    "VERTICAL ACCELERATION",
    "WIND SPEED",
]

CLASS_NAMES = [
    "Nominal",
    "Speed High",
    "Path High",
    "Flaps Late Setting",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_dir() -> Path:
    return project_root() / "dataset"


def default_raw_npz_path() -> Path:
    candidates = [
        dataset_dir() / "DASHlink_full_fourclass_raw_comp.npz",
        dataset_dir() / "DASHlink_full_fourclass_raw_comp (1).npz",
        Path(r"c:\Users\njidi\Downloads\DASHlink_full_fourclass_raw_comp.npz"),
        Path(r"c:\Users\njidi\Downloads\DASHlink_full_fourclass_raw_comp (1).npz"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_raw_meta_path() -> Path:
    candidates = [
        dataset_dir() / "DASHlink_full_fourclass_raw_meta.csv",
        dataset_dir() / "DASHlink_full_fourclass_raw_meta (1).csv",
        Path(r"c:\Users\njidi\Downloads\DASHlink_full_fourclass_raw_meta.csv"),
        Path(r"c:\Users\njidi\Downloads\DASHlink_full_fourclass_raw_meta (1).csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_preprocessed_splits_path() -> Path:
    return dataset_dir() / "landing_anomaly_fourclass_splits.npz"


@dataclass
class LandingAnomalySplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    scaler: StandardScaler

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.long),
            torch.tensor(self.X_val, dtype=torch.float32),
            torch.tensor(self.y_val, dtype=torch.long),
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.long),
        )


def _load_raw_npz(npz_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    npz_path = Path(npz_path or default_raw_npz_path())
    if not npz_path.exists():
        raise FileNotFoundError(
            "Could not find the DASHlink .npz dataset. Put "
            "'DASHlink_full_fourclass_raw_comp.npz' in either the project 'dataset' folder "
            f"or Downloads. Last checked path: {npz_path}"
        )

    data = np.load(npz_path, allow_pickle=True)
    if "data" not in data or "label" not in data:
        raise ValueError(f"{npz_path} must contain 'data' and 'label' arrays.")

    X = np.asarray(data["data"], dtype=np.float32)
    y_raw = np.asarray(data["label"], dtype=np.int64)
    return X, y_raw


def _transform_split(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    original_shape = X.shape
    X_scaled = scaler.transform(X.reshape(-1, original_shape[-1])).reshape(original_shape)
    return X_scaled.astype(np.float32)


def build_preprocessed_landing_splits(
    raw_npz_path: str | Path | None = None,
    output_path: str | Path | None = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
    random_state: int = 42,
) -> LandingAnomalySplits:
    X, y_raw = _load_raw_npz(raw_npz_path)
    indices = np.arange(len(y_raw))

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X,
        y_raw,
        indices,
        train_size=train_size,
        stratify=y_raw,
        random_state=random_state,
    )

    relative_val_size = val_size / (1.0 - train_size)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp,
        y_temp,
        idx_temp,
        train_size=relative_val_size,
        stratify=y_temp,
        random_state=random_state,
    )

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    splits = LandingAnomalySplits(
        X_train=_transform_split(scaler, X_train),
        X_val=_transform_split(scaler, X_val),
        X_test=_transform_split(scaler, X_test),
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        train_indices=idx_train.astype(np.int64),
        val_indices=idx_val.astype(np.int64),
        test_indices=idx_test.astype(np.int64),
        feature_names=FEATURE_NAMES.copy(),
        class_names=CLASS_NAMES.copy(),
        scaler=scaler,
    )

    if output_path is not None:
        save_landing_splits(splits, output_path)

    return splits


def save_landing_splits(splits: LandingAnomalySplits, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=splits.X_train,
        X_val=splits.X_val,
        X_test=splits.X_test,
        y_train=splits.y_train,
        y_val=splits.y_val,
        y_test=splits.y_test,
        train_indices=splits.train_indices,
        val_indices=splits.val_indices,
        test_indices=splits.test_indices,
        feature_names=np.array(splits.feature_names),
        class_names=np.array(splits.class_names),
        scaler_mean=splits.scaler.mean_,
        scaler_scale=splits.scaler.scale_,
    )


def load_landing_splits_npz(npz_path: str | Path) -> LandingAnomalySplits:
    data = np.load(npz_path, allow_pickle=True)

    scaler = StandardScaler()
    scaler.mean_ = data["scaler_mean"]
    scaler.scale_ = data["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    return LandingAnomalySplits(
        X_train=data["X_train"],
        X_val=data["X_val"],
        X_test=data["X_test"],
        y_train=data["y_train"],
        y_val=data["y_val"],
        y_test=data["y_test"],
        train_indices=data["train_indices"],
        val_indices=data["val_indices"],
        test_indices=data["test_indices"],
        feature_names=data["feature_names"].tolist(),
        class_names=data["class_names"].tolist(),
        scaler=scaler,
    )


def load_preprocessed_landing_splits(
    npz_path: str | Path | None = None,
) -> tuple[LandingAnomalySplits, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    npz_path = Path(npz_path or default_preprocessed_splits_path())
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find {npz_path}. Run preprocessing_v2.py first.")

    splits = load_landing_splits_npz(npz_path)
    return (splits, *splits.to_tensors())


def load_metadata(meta_path: str | Path | None = None) -> pd.DataFrame:
    meta_path = Path(meta_path or default_raw_meta_path())
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find DASHlink metadata: {meta_path}")
    return pd.read_csv(meta_path)


if __name__ == "__main__":
    output_file = default_preprocessed_splits_path()
    splits = build_preprocessed_landing_splits(output_path=output_file)
    train_counts = np.bincount(splits.y_train, minlength=len(CLASS_NAMES))
    val_counts = np.bincount(splits.y_val, minlength=len(CLASS_NAMES))
    test_counts = np.bincount(splits.y_test, minlength=len(CLASS_NAMES))

    print("Landing Anomaly Detection")
    print("Saved:", output_file)
    print("Train shape:", splits.X_train.shape, "Labels:", splits.y_train.shape)
    print("Validation shape:", splits.X_val.shape, "Labels:", splits.y_val.shape)
    print("Test shape:", splits.X_test.shape, "Labels:", splits.y_test.shape)
    print("Feature count:", len(splits.feature_names))
    print("Classes:", splits.class_names)
    print("Class distribution:")
    for idx, class_name in enumerate(splits.class_names):
        print(
            f"  [{idx}] {class_name}: "
            f"train={int(train_counts[idx])} "
            f"val={int(val_counts[idx])} "
            f"test={int(test_counts[idx])}"
        )

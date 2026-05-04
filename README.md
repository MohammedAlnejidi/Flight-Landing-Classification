# Flight Landing Classification

This project trains an LSTM classifier for four-class flight landing anomaly classification. It uses DASHlink-derived flight sequence data and predicts one of four landing behavior classes: `Nominal`, `Speed High`, `Path High`, and `Flaps Late Setting`.

## Project Structure

```text
Flight-Landing-Classification/
  main.py                    # Command overview
  train_classifier.py         # LSTM training workflow
  evaluate_classifier.py      # Test evaluation and exported reports
  optuna_tune.py              # Optuna hyperparameter tuning
  models/
    lstm_classifier.py        # LSTM classifier model
  utils/
    preprocessing_v2.py       # Dataset splitting and scaling
    visualize_data.py         # Dataset visualization utilities
```

## Setup

Create and activate a virtual environment, then install the project dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the raw DASHlink dataset files in:

```text
Flight-Landing-Classification/dataset/
```

Expected files include:

```text
DASHlink_full_fourclass_raw_comp.npz
DASHlink_full_fourclass_raw_meta.csv
```

The dataset folder is ignored by Git so large or licensed data is not uploaded accidentally.

## Preprocess Data

```powershell
python Flight-Landing-Classification/utils/preprocessing_v2.py
```

This creates the preprocessed train, validation, and test splits locally in the ignored `dataset` folder.

## Train Classifier

```powershell
python Flight-Landing-Classification/train_classifier.py
```

Training saves a local checkpoint named `landing_lstm_classifier_checkpoint.pt`, which is ignored by Git.

## Evaluate Classifier

```powershell
python Flight-Landing-Classification/evaluate_classifier.py
```

Evaluation writes local reports into the ignored `evaluation` folder.

## Tune Hyperparameters

```powershell
python Flight-Landing-Classification/optuna_tune.py
```

## Results

The latest recorded test evaluation produced the following results:

| Metric | Value |
|---|---:|
| Accuracy | 95.47% |
| Macro F1 | 86.31% |
| Weighted F1 | 95.75% |
| MCC | 79.38% |
| Correct Predictions | 14,299 / 14,977 |
| Wrong Predictions | 678 / 14,977 |

Per-class accuracy:

| Class | Accuracy |
|---|---:|
| Nominal | 95.97% |
| Speed High | 92.97% |
| Path High | 83.99% |
| Flaps Late Setting | 93.71% |

## Conclusion

The LSTM classifier performed well on the flight landing classification task, with strong overall accuracy and weighted F1 score. The model was especially effective on the majority `Nominal` class and also performed well on `Speed High` and `Flaps Late Setting`.

The lower performance on `Path High` suggests that this class is harder to distinguish from other landing patterns, likely because it has fewer examples and may share temporal characteristics with nearby classes. Future improvements could include stronger class balancing, additional sequence features, more hyperparameter tuning, and comparison with transformer-based or CNN-LSTM sequence models.

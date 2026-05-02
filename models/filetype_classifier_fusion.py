from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import random
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "byte_chunks_govdocs1_512_noRandNoise.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "artifacts_fusion_512_noRandNoise"


@dataclass
class TrainConfig:
    dataset: str
    output_dir: str
    batch_size: int = 256
    epochs: int = 40
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42
    rf_trees: int = 300
    rf_max_depth: int | None = None


class ByteStatsFusionCNN(nn.Module):
    """CNN(raw bytes) + MLP(statistical features) -> classifier."""

    def __init__(self, num_classes: int, num_stat_features: int):
        super().__init__()
        self.byte_features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.stat_features = nn.Sequential(
            nn.Linear(num_stat_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_bytes: torch.Tensor, x_stats: torch.Tensor) -> torch.Tensor:
        byte_features = self.byte_features(x_bytes)
        byte_pooled = torch.cat([self.avg_pool(byte_features), self.max_pool(byte_features)], dim=1)
        byte_pooled = byte_pooled.flatten(start_dim=1)
        stat_embedding = self.stat_features(x_stats)
        return self.classifier(torch.cat([byte_pooled, stat_embedding], dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path):
    data = np.load(path)
    if "features" not in data.files:
        raise ValueError("Dataset has no 'features' array. Rebuild it with construct_dataset_for_classification.py.")

    x = data["X"].astype(np.float32) / 255.0
    features = data["features"].astype(np.float32)
    y = data["y"].astype(np.int64)
    label_names = data["label_names"].astype(str).tolist()
    feature_names = data["feature_names"].astype(str).tolist() if "feature_names" in data.files else None
    split = data["split"].astype(np.int64) if "split" in data.files else None
    split_names = data["split_names"].astype(str).tolist() if "split_names" in data.files else None
    return x, features, y, label_names, feature_names, split, split_names


def split_dataset(
    x: np.ndarray,
    features: np.ndarray,
    y: np.ndarray,
    split: np.ndarray | None,
    config: TrainConfig,
):
    if split is not None:
        train_mask = split == 0
        val_mask = split == 1
        test_mask = split == 2

        if not train_mask.any() or not val_mask.any() or not test_mask.any():
            raise ValueError("Dataset split must contain train=0, val=1, and test=2 samples.")

        return (
            x[train_mask],
            x[val_mask],
            x[test_mask],
            features[train_mask],
            features[val_mask],
            features[test_mask],
            y[train_mask],
            y[val_mask],
            y[test_mask],
        )

    x_train_val, x_test, f_train_val, f_test, y_train_val, y_test = train_test_split(
        x,
        features,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )
    relative_val_size = config.val_size / (1.0 - config.test_size)
    x_train, x_val, f_train, f_val, y_train, y_val = train_test_split(
        x_train_val,
        f_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=config.seed,
        stratify=y_train_val,
    )

    return x_train, x_val, x_test, f_train, f_val, f_test, y_train, y_val, y_test


def scale_features(f_train, f_val, f_test):
    scaler = StandardScaler()
    f_train = scaler.fit_transform(f_train).astype(np.float32)
    f_val = scaler.transform(f_val).astype(np.float32)
    f_test = scaler.transform(f_test).astype(np.float32)
    return f_train, f_val, f_test, scaler


def compute_class_weights(y_train: np.ndarray, num_classes: int, device: torch.device):
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Training split has no samples for classes: {missing}")

    weights = len(y_train) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(
    x: np.ndarray,
    features: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.from_numpy(x).unsqueeze(1)
    features_tensor = torch.from_numpy(features)
    y_tensor = torch.from_numpy(y)
    return DataLoader(
        TensorDataset(x_tensor, features_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for x_batch, features_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        features_batch = features_batch.to(device)
        y_batch = y_batch.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x_batch, features_batch)
            loss = criterion(logits, y_batch)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_items += batch_size

    return total_loss / total_items, total_correct / total_items


def predict_fusion_model(model, loader, device):
    model.eval()
    predictions = []
    probabilities = []
    targets = []

    with torch.no_grad():
        for x_batch, features_batch, y_batch in loader:
            logits = model(x_batch.to(device), features_batch.to(device))
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.argmax(dim=1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            targets.append(y_batch.numpy())

    return np.concatenate(targets), np.concatenate(predictions), np.concatenate(probabilities)


def compute_top_k_metrics(y_true: np.ndarray, probabilities: np.ndarray, max_k: int = 5):
    metrics = {}
    labels = np.arange(probabilities.shape[1])

    for k in range(1, min(max_k, probabilities.shape[1]) + 1):
        metrics[f"top_{k}_accuracy"] = top_k_accuracy_score(
            y_true,
            probabilities,
            k=k,
            labels=labels,
        )

    return metrics


def train_fusion_model(
    x_train,
    x_val,
    x_test,
    f_train,
    f_val,
    f_test,
    y_train,
    y_val,
    y_test,
    label_names,
    feature_names,
    scaler,
    config: TrainConfig,
    output_dir: Path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training fusion CNN on {device}")

    train_loader = make_loader(x_train, f_train, y_train, config.batch_size, shuffle=True)
    val_loader = make_loader(x_val, f_val, y_val, config.batch_size, shuffle=False)
    test_loader = make_loader(x_test, f_test, y_test, config.batch_size, shuffle=False)

    model = ByteStatsFusionCNN(num_classes=len(label_names), num_stat_features=f_train.shape[1]).to(device)
    class_weights = compute_class_weights(y_train, len(label_names), device)
    print("Fusion CNN class weights:", [round(float(value), 4) for value in class_weights.cpu()])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    best_path = output_dir / "byte_chunk_fusion_cnn.pt"

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "seconds": elapsed,
        }
        history.append(row)

        print(
            f"epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": label_names,
                    "feature_names": feature_names,
                    "config": asdict(config),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "class_weights": class_weights.detach().cpu().numpy().tolist(),
                    "feature_scaler": scaler,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred, y_prob = predict_fusion_model(model, test_loader, device)
    test_accuracy = accuracy_score(y_true, y_pred)
    top_k_metrics = compute_top_k_metrics(y_true, y_prob)

    metrics = {
        "test_accuracy": test_accuracy,
        "top_k_accuracy": top_k_metrics,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "weights_path": str(best_path),
    }

    return model, metrics


def train_random_forest_on_features(f_train, f_test, y_train, y_test, label_names, config: TrainConfig, output_dir: Path):
    print("Training Random Forest on statistical features")
    model = RandomForestClassifier(
        n_estimators=config.rf_trees,
        max_depth=config.rf_max_depth,
        n_jobs=-1,
        random_state=config.seed,
        class_weight="balanced",
    )
    model.fit(f_train, y_train)

    y_pred = model.predict(f_test)
    y_prob = model.predict_proba(f_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    top_k_metrics = compute_top_k_metrics(y_test, y_prob)
    model_path = output_dir / "stat_features_random_forest.joblib"
    joblib.dump(
        {
            "model": model,
            "label_names": label_names,
            "config": asdict(config),
        },
        model_path,
    )

    metrics = {
        "test_accuracy": test_accuracy,
        "top_k_accuracy": top_k_metrics,
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "model_path": str(model_path),
    }

    return model, metrics


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_classification_report_csv(path: Path, report: dict, label_names: list[str]) -> None:
    rows = []
    for label_name in label_names:
        metrics = report[label_name]
        rows.append(
            {
                "class": label_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": metrics["support"],
            }
        )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_csv(path: Path, matrix: list[list[int]], label_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *label_names])
        for label_name, row in zip(label_names, matrix):
            writer.writerow([label_name, *row])


def save_normalized_confusion_matrix_csv(path: Path, matrix: list[list[int]], label_names: list[str]) -> None:
    matrix_array = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix_array.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix_array,
        row_sums,
        out=np.zeros_like(matrix_array),
        where=row_sums != 0,
    )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *label_names])
        for label_name, row in zip(label_names, normalized):
            writer.writerow([label_name, *[f"{value:.6f}" for value in row]])


def print_per_class_metrics(model_name: str, report: dict, label_names: list[str]) -> None:
    print(f"\n{model_name} per-class metrics")
    print(f"{'class':<16} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    print("-" * 62)

    for label_name in label_names:
        metrics = report[label_name]
        print(
            f"{label_name:<16} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1-score']:>10.4f} "
            f"{int(metrics['support']):>10,}"
        )

    macro = report["macro avg"]
    weighted = report["weighted avg"]
    print("-" * 62)
    print(
        f"{'macro avg':<16} "
        f"{macro['precision']:>10.4f} "
        f"{macro['recall']:>10.4f} "
        f"{macro['f1-score']:>10.4f} "
        f"{int(macro['support']):>10,}"
    )
    print(
        f"{'weighted avg':<16} "
        f"{weighted['precision']:>10.4f} "
        f"{weighted['recall']:>10.4f} "
        f"{weighted['f1-score']:>10.4f} "
        f"{int(weighted['support']):>10,}"
    )


def print_top_k_metrics(model_name: str, metrics: dict) -> None:
    print(f"\n{model_name} top-k accuracy")
    for key, value in sorted(metrics.items(), key=lambda item: int(item[0].split("_")[1])):
        print(f"  {key}: {value:.4f}")


def print_confusion_matrix(model_name: str, matrix: list[list[int]], label_names: list[str]) -> None:
    matrix_array = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix_array.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix_array,
        row_sums,
        out=np.zeros_like(matrix_array),
        where=row_sums != 0,
    )

    print(f"\n{model_name} normalized confusion matrix")
    print("Rows are actual classes, columns are predicted classes.")
    header = "actual\\pred".ljust(14) + " ".join(name[:7].rjust(7) for name in label_names)
    print(header)
    print("-" * len(header))
    for label_name, row in zip(label_names, normalized):
        values = " ".join(f"{value:7.2f}" for value in row)
        print(f"{label_name[:13].ljust(14)}{values}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train byte+feature fusion file type classifiers.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to .npz dataset.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for models and metrics.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    dataset_path = Path(config.dataset)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, features, y, label_names, feature_names, split, split_names = load_dataset(dataset_path)
    (
        x_train,
        x_val,
        x_test,
        f_train,
        f_val,
        f_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(x, features, y, split, config)
    f_train, f_val, f_test, scaler = scale_features(f_train, f_val, f_test)

    print(f"Dataset: {dataset_path}")
    print(f"Labels: {label_names}")
    print(f"Statistical features: {f_train.shape[1]}")
    if split_names is not None:
        print(f"Using predefined file-level split: {split_names}")
    else:
        print("Using generated chunk-level split because dataset has no split array.")
    print(f"Train: {x_train.shape}, validation: {x_val.shape}, test: {x_test.shape}")

    _, fusion_metrics = train_fusion_model(
        x_train,
        x_val,
        x_test,
        f_train,
        f_val,
        f_test,
        y_train,
        y_val,
        y_test,
        label_names,
        feature_names,
        scaler,
        config,
        output_dir,
    )
    _, rf_metrics = train_random_forest_on_features(
        f_train,
        f_test,
        y_train,
        y_test,
        label_names,
        config,
        output_dir,
    )

    metrics = {
        "config": asdict(config),
        "label_names": label_names,
        "feature_names": feature_names,
        "splits": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "fusion_cnn": fusion_metrics,
        "random_forest_features": rf_metrics,
        "accuracy_comparison": {
            "fusion_cnn": fusion_metrics["test_accuracy"],
            "random_forest_features": rf_metrics["test_accuracy"],
            "winner": "fusion_cnn"
            if fusion_metrics["test_accuracy"] >= rf_metrics["test_accuracy"]
            else "random_forest_features",
        },
    }

    metrics_path = output_dir / "training_metrics.json"
    fusion_report_csv = output_dir / "fusion_cnn_per_class_metrics.csv"
    rf_report_csv = output_dir / "random_forest_features_per_class_metrics.csv"
    fusion_confusion_csv = output_dir / "fusion_cnn_confusion_matrix.csv"
    rf_confusion_csv = output_dir / "random_forest_features_confusion_matrix.csv"
    fusion_confusion_norm_csv = output_dir / "fusion_cnn_confusion_matrix_normalized.csv"
    rf_confusion_norm_csv = output_dir / "random_forest_features_confusion_matrix_normalized.csv"

    save_json(metrics_path, metrics)
    save_classification_report_csv(fusion_report_csv, fusion_metrics["classification_report"], label_names)
    save_classification_report_csv(rf_report_csv, rf_metrics["classification_report"], label_names)
    save_confusion_matrix_csv(fusion_confusion_csv, fusion_metrics["confusion_matrix"], label_names)
    save_confusion_matrix_csv(rf_confusion_csv, rf_metrics["confusion_matrix"], label_names)
    save_normalized_confusion_matrix_csv(
        fusion_confusion_norm_csv,
        fusion_metrics["confusion_matrix"],
        label_names,
    )
    save_normalized_confusion_matrix_csv(
        rf_confusion_norm_csv,
        rf_metrics["confusion_matrix"],
        label_names,
    )

    print("\nFinal comparison")
    print(f"  Fusion CNN accuracy:             {fusion_metrics['test_accuracy']:.4f}")
    print(f"  Random Forest features accuracy: {rf_metrics['test_accuracy']:.4f}")
    print(f"  Winner: {metrics['accuracy_comparison']['winner']}")
    print_top_k_metrics("Fusion CNN", fusion_metrics["top_k_accuracy"])
    print_top_k_metrics("Random Forest features", rf_metrics["top_k_accuracy"])
    print_per_class_metrics("Fusion CNN", fusion_metrics["classification_report"], label_names)
    print_per_class_metrics("Random Forest features", rf_metrics["classification_report"], label_names)
    print_confusion_matrix("Fusion CNN", fusion_metrics["confusion_matrix"], label_names)
    print_confusion_matrix("Random Forest features", rf_metrics["confusion_matrix"], label_names)
    print(f"Saved metrics to: {metrics_path.resolve()}")
    print(f"Saved Fusion CNN per-class CSV to: {fusion_report_csv.resolve()}")
    print(f"Saved Random Forest features per-class CSV to: {rf_report_csv.resolve()}")
    print(f"Saved Fusion CNN confusion matrix CSV to: {fusion_confusion_csv.resolve()}")
    print(f"Saved Random Forest features confusion matrix CSV to: {rf_confusion_csv.resolve()}")
    print(f"Saved Fusion CNN normalized confusion matrix CSV to: {fusion_confusion_norm_csv.resolve()}")
    print(f"Saved Random Forest features normalized confusion matrix CSV to: {rf_confusion_norm_csv.resolve()}")


if __name__ == "__main__":
    main()

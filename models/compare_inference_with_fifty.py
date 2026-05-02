from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from internship_repo.models.filetype_classifier import ByteChunkCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "byte_chunks_govdocs1_512.npz"
DEFAULT_OUR_CHECKPOINT = PROJECT_ROOT / "models" / "artifacts" / "byte_chunk_cnn.pt"
FIFTY_ROOT = PROJECT_ROOT / "models" / "fifty"
FIFTY_LABELS_PATH = FIFTY_ROOT / "fifty" / "utilities" / "labels.json"
FIFTY_MODELS_DIR = FIFTY_ROOT / "fifty" / "utilities" / "models"


def load_tensorflow_model_loader():
    try:
        from tensorflow.keras.models import load_model

        return load_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is not installed, so FiFTy .h5 models cannot be loaded. "
            "Install a TensorFlow/Keras environment compatible with FiFTy, then rerun this script."
        ) from exc


def load_dataset(path: Path, split_name: str):
    data = np.load(path)
    x = data["X"]
    y = data["y"].astype(np.int64)
    label_names = data["label_names"].astype(str).tolist()

    if "split" in data.files:
        split = data["split"].astype(np.int64)
        split_names = data["split_names"].astype(str).tolist()
        split_id = split_names.index(split_name)
        mask = split == split_id
        x = x[mask]
        y = y[mask]

    return x, y, label_names


def load_our_model(checkpoint_path: Path, fallback_label_count: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_names = checkpoint.get("label_names")
    num_classes = len(label_names) if label_names is not None else fallback_label_count

    model = ByteChunkCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, label_names


def predict_our_model(model, x: np.ndarray, batch_size: int, device: torch.device):
    probabilities = []

    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].astype(np.float32) / 255.0
        batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

    probabilities = np.concatenate(probabilities)
    predictions = probabilities.argmax(axis=1)
    return predictions, probabilities


def load_fifty_labels(scenario: int):
    labels = json.loads(FIFTY_LABELS_PATH.read_text(encoding="utf-8"))
    return labels[str(scenario)]


def load_fifty_model(block_size: int, scenario: int, model_path: Path | None):
    load_model = load_tensorflow_model_loader()

    if model_path is None:
        model_path = FIFTY_MODELS_DIR / f"{block_size}_{scenario}.h5"

    if not model_path.exists():
        raise FileNotFoundError(f"FiFTy model not found: {model_path}")

    return load_model(str(model_path), compile=False), model_path


def predict_fifty_model(model, x: np.ndarray, batch_size: int):
    probabilities = model.predict(x.astype(np.uint8), batch_size=batch_size, verbose=0)
    predictions = probabilities.argmax(axis=1)
    return predictions, probabilities


def top_k_accuracy_from_probs(y_true_names, probabilities, label_names, k: int):
    correct = 0
    for true_name, probs in zip(y_true_names, probabilities):
        top_indices = np.argsort(probs)[-k:]
        top_names = {label_names[index] for index in top_indices}
        correct += true_name in top_names
    return correct / len(y_true_names) if len(y_true_names) else 0.0


def write_prediction_sample(
    path: Path,
    y_true_names,
    our_pred_names,
    fifty_pred_names,
    our_probabilities,
    fifty_probabilities,
    our_labels,
    fifty_labels,
    limit: int = 5000,
):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "true",
                "our_pred",
                "our_confidence",
                "our_top3",
                "fifty_pred",
                "fifty_confidence",
                "fifty_top3",
            ],
        )
        writer.writeheader()

        for index in range(min(limit, len(y_true_names))):
            our_top3 = np.argsort(our_probabilities[index])[-3:][::-1]
            fifty_top3 = np.argsort(fifty_probabilities[index])[-3:][::-1]
            writer.writerow(
                {
                    "index": index,
                    "true": y_true_names[index],
                    "our_pred": our_pred_names[index],
                    "our_confidence": float(our_probabilities[index].max()),
                    "our_top3": " ".join(our_labels[i] for i in our_top3),
                    "fifty_pred": fifty_pred_names[index],
                    "fifty_confidence": float(fifty_probabilities[index].max()),
                    "fifty_top3": " ".join(fifty_labels[i] for i in fifty_top3),
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare our byte CNN inference with FiFTy inference.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--our-checkpoint", default=str(DEFAULT_OUR_CHECKPOINT))
    parser.add_argument("--fifty-model", default=None, help="Explicit path to a FiFTy .h5 model.")
    parser.add_argument("--fifty-scenario", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all samples.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models" / "inference_comparison"))
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    our_checkpoint_path = Path(args.our_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, dataset_labels = load_dataset(dataset_path, args.split)
    if args.max_samples > 0:
        x = x[: args.max_samples]
        y = y[: args.max_samples]

    if x.shape[1] != args.block_size:
        raise ValueError(
            f"Dataset chunks are {x.shape[1]} bytes, but FiFTy block size is {args.block_size}. "
            "Use matching --block-size or rebuild the dataset."
        )

    fifty_labels = load_fifty_labels(args.fifty_scenario)
    common_labels = sorted(set(dataset_labels) & set(fifty_labels))
    common_label_set = set(common_labels)
    y_true_names_all = np.asarray([dataset_labels[index] for index in y])
    common_mask = np.asarray([name in common_label_set for name in y_true_names_all])

    if not common_mask.any():
        raise ValueError("No overlapping labels between our dataset and FiFTy labels.")

    x_common = x[common_mask]
    y_true_names = y_true_names_all[common_mask]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    our_model, checkpoint_labels = load_our_model(our_checkpoint_path, len(dataset_labels), device)
    our_labels = checkpoint_labels if checkpoint_labels is not None else dataset_labels

    fifty_model, fifty_model_path = load_fifty_model(
        args.block_size,
        args.fifty_scenario,
        Path(args.fifty_model) if args.fifty_model else None,
    )

    our_pred, our_prob = predict_our_model(our_model, x_common, args.batch_size, device)
    fifty_pred, fifty_prob = predict_fifty_model(fifty_model, x_common, args.batch_size)

    our_pred_names = np.asarray([our_labels[index] for index in our_pred])
    fifty_pred_names = np.asarray([fifty_labels[index] for index in fifty_pred])

    our_accuracy = accuracy_score(y_true_names, our_pred_names)
    fifty_accuracy = accuracy_score(y_true_names, fifty_pred_names)

    results = {
        "dataset": str(dataset_path),
        "split": args.split,
        "samples_total": int(len(x)),
        "samples_common_labels": int(len(x_common)),
        "common_labels": common_labels,
        "our_checkpoint": str(our_checkpoint_path),
        "fifty_model": str(fifty_model_path),
        "our_accuracy": our_accuracy,
        "fifty_accuracy": fifty_accuracy,
        "our_top_k": {
            f"top_{k}": top_k_accuracy_from_probs(y_true_names, our_prob, our_labels, k)
            for k in range(1, min(5, len(our_labels)) + 1)
        },
        "fifty_top_k": {
            f"top_{k}": top_k_accuracy_from_probs(y_true_names, fifty_prob, fifty_labels, k)
            for k in range(1, min(5, len(fifty_labels)) + 1)
        },
        "our_classification_report": classification_report(
            y_true_names,
            our_pred_names,
            labels=common_labels,
            output_dict=True,
            zero_division=0,
        ),
        "fifty_classification_report": classification_report(
            y_true_names,
            fifty_pred_names,
            labels=common_labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    metrics_path = output_dir / "comparison_metrics.json"
    sample_path = output_dir / "prediction_sample.csv"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_prediction_sample(
        sample_path,
        y_true_names,
        our_pred_names,
        fifty_pred_names,
        our_prob,
        fifty_prob,
        our_labels,
        fifty_labels,
    )

    print(f"Compared on {len(x_common):,} samples with common labels: {', '.join(common_labels)}")
    print(f"Our model accuracy: {our_accuracy:.4f}")
    print(f"FiFTy accuracy:     {fifty_accuracy:.4f}")
    print("Our top-k:", {key: round(value, 4) for key, value in results["our_top_k"].items()})
    print("FiFTy top-k:", {key: round(value, 4) for key, value in results["fifty_top_k"].items()})
    print(f"Saved metrics to: {metrics_path.resolve()}")
    print(f"Saved prediction sample to: {sample_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

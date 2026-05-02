from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BGPT_DIR = PROJECT_ROOT / "models" / "bgpt"
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "nps-2009-canon2" / "nps-2009-canon2-gen6_candidate_clusters.npz"
DEFAULT_WEIGHTS = BGPT_DIR / "weights-image.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "nps-2009-canon2" / "bgpt_scores"

sys.path.insert(0, str(BGPT_DIR))
from config import (  # noqa: E402
    BYTE_NUM_LAYERS,
    HIDDEN_SIZE,
    PATCH_LENGTH,
    PATCH_NUM_LAYERS,
    PATCH_SIZE,
)
from utils import bGPTLMHeadModel  # noqa: E402


def load_bgpt(weights_path: Path, device: torch.device):
    if not weights_path.exists():
        raise FileNotFoundError(
            f"bGPT weights not found: {weights_path}\n"
            "Download a checkpoint first, for example:\n"
            "  python models\\bgpt_carving_scorer.py download-weights --filename weights-image.pth"
        )

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    inferred_patch_length = int(state["patch_level_decoder.base.wpe.weight"].shape[0])
    inferred_hidden_size = int(state["patch_level_decoder.base.wpe.weight"].shape[1])
    inferred_byte_length = int(state["byte_level_decoder.base.transformer.wpe.weight"].shape[0])

    patch_config = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS,
        max_length=inferred_patch_length,
        max_position_embeddings=inferred_patch_length,
        hidden_size=inferred_hidden_size,
        n_head=inferred_hidden_size // 64,
        vocab_size=1,
    )
    byte_config = GPT2Config(
        num_hidden_layers=BYTE_NUM_LAYERS,
        max_length=inferred_byte_length,
        max_position_embeddings=inferred_byte_length,
        hidden_size=inferred_hidden_size,
        n_head=inferred_hidden_size // 64,
        vocab_size=256 + 1,
    )
    model = bGPTLMHeadModel(patch_config, byte_config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if not key.endswith(".attn.bias") and not key.endswith(".attn.masked_bias")]
    if missing or unexpected:
        raise RuntimeError(f"Could not load bGPT checkpoint cleanly. Missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


def download_weights(filename: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub first: pip install huggingface_hub") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id="sander-wood/bgpt",
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    )


def pad_to_patch(values: np.ndarray, pad_value: int = 256) -> np.ndarray:
    remainder = len(values) % PATCH_SIZE
    if remainder == 0:
        return values
    pad = np.full(PATCH_SIZE - remainder, pad_value, dtype=np.int64)
    return np.concatenate([values, pad])


def extension_patch(extension: str) -> np.ndarray:
    encoded = list(extension.encode("utf-8"))[:PATCH_SIZE]
    encoded += [256] * (PATCH_SIZE - len(encoded))
    return np.array(encoded, dtype=np.int64)


def sequence_for_score(context: np.ndarray, candidate: np.ndarray, extension: str):
    context = pad_to_patch(context.astype(np.int64))
    candidate = pad_to_patch(candidate.astype(np.int64))
    prefix = np.concatenate([extension_patch(extension), context])
    sequence = np.concatenate([prefix, candidate])
    candidate_start_patch = len(prefix) // PATCH_SIZE
    return sequence, candidate_start_patch


@torch.no_grad()
def candidate_nll_batch(model, contexts, candidates, extension: str, device: torch.device) -> np.ndarray:
    sequences = []
    candidate_start_patches = []
    for context, candidate in zip(contexts, candidates):
        sequence, candidate_start_patch = sequence_for_score(context, candidate, extension)
        sequences.append(sequence)
        candidate_start_patches.append(candidate_start_patch)

    max_len = max(len(seq) for seq in sequences)
    padded = np.full((len(sequences), max_len), 256, dtype=np.int64)
    masks = np.zeros((len(sequences), max_len // PATCH_SIZE), dtype=np.int64)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
        masks[index, : len(sequence) // PATCH_SIZE] = 1

    patches = torch.from_numpy(padded).to(device).reshape(len(sequences), -1, PATCH_SIZE)
    mask_tensor = torch.from_numpy(masks).to(device)
    encoded = model.patch_level_decoder(patches, mask_tensor)["last_hidden_state"]

    losses = []
    for row in range(len(sequences)):
        valid_patches = int(mask_tensor[row].sum().item())
        patch_losses = []
        for target_patch_index in range(max(1, candidate_start_patches[row]), valid_patches):
            encoded_patch = encoded[row, target_patch_index - 1].reshape(1, 1, -1)
            target_patch = patches[row, target_patch_index].reshape(1, PATCH_SIZE)
            labels = torch.cat(
                [
                    torch.full((1, 1), model.special_token_id, device=device, dtype=torch.long),
                    target_patch,
                ],
                dim=1,
            )
            token_embeds = F.embedding(labels, model.byte_level_decoder.base.transformer.wte.weight)
            inputs_embeds = torch.cat([encoded_patch, token_embeds[:, 1:, :]], dim=1)
            logits = model.byte_level_decoder.base(inputs_embeds=inputs_embeds).logits
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            patch_losses.append(F.cross_entropy(shift_logits, shift_labels, reduction="mean"))
        losses.append(torch.stack(patch_losses).mean().item())

    return np.array(losses, dtype=np.float32)


@torch.no_grad()
def generate_bytes(model, seed: np.ndarray, extension: str, byte_count: int, device: torch.device, temperature: float):
    byte_list = np.concatenate([extension_patch(extension), pad_to_patch(seed.astype(np.int64))]).tolist()
    while len(byte_list) < PATCH_LENGTH * PATCH_SIZE and len(byte_list) < len(seed) + PATCH_SIZE + byte_count:
        input_patches = torch.tensor([byte_list], dtype=torch.long, device=device).unsqueeze(0)
        if input_patches.shape[-1] % PATCH_SIZE != 0:
            input_patches = torch.cat(
                [
                    input_patches.squeeze(0),
                    torch.full(
                        (1, PATCH_SIZE - input_patches.shape[-1] % PATCH_SIZE),
                        256,
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            ).unsqueeze(0)
        generated_patch = model.generate(input_patches, top_k=0, top_p=1.0, temperature=temperature)
        for token in generated_patch:
            if token == 256 or len(byte_list) >= len(seed) + PATCH_SIZE + byte_count:
                break
            byte_list.append(int(token))
        if generated_patch and generated_patch[-1] == 256:
            break
    generated = np.array(byte_list[PATCH_SIZE + len(pad_to_patch(seed.astype(np.int64))) :], dtype=np.uint8)
    return generated[:byte_count]


def byte_distribution(values: np.ndarray) -> np.ndarray:
    counts = np.bincount(values.astype(np.uint8), minlength=256).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return np.full(256, 1.0 / 256.0, dtype=np.float64)
    return counts / total


def paper_similarity_metrics(predicted: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float, float]:
    pred_dist = byte_distribution(predicted)
    cand_dist = byte_distribution(candidate)
    eps = 1e-12

    chi = float(np.sum(((pred_dist - cand_dist) ** 2) / (pred_dist + cand_dist + eps)))

    midpoint = 0.5 * (pred_dist + cand_dist)
    kl_pred = np.sum(pred_dist * np.log2((pred_dist + eps) / (midpoint + eps)))
    kl_cand = np.sum(cand_dist * np.log2((cand_dist + eps) / (midpoint + eps)))
    jsd = float(0.5 * (kl_pred + kl_cand))

    denom = np.linalg.norm(pred_dist) * np.linalg.norm(cand_dist)
    cosine = float(np.dot(pred_dist, cand_dist) / denom) if denom > 0 else 0.0

    paper_score = float((0.01 * chi) + (10.0 * jsd) - (10.0 * cosine))
    return chi, jsd, cosine, paper_score


def classifier_scores(data, label: str) -> tuple[np.ndarray, np.ndarray]:
    labels = data["our_label_names"].astype(str)
    if label not in labels:
        return np.zeros(len(data["cluster_number"]), dtype=np.float32), np.zeros(len(data["cluster_number"]), dtype=np.float32)
    index = int(np.where(labels == label)[0][0])
    return data["our_cluster_mean_probs"][:, index], data["our_cluster_max_probs"][:, index]


def score_candidates(args) -> None:
    """
    Candidate pruning and scoring used in the report:
    1. Start from pre-extracted unallocated clusters in candidate_clusters.npz.
    2. Pick seed clusters by JPEG SOI signatures, or by the file-type classifier if
       signatures are unavailable.
    3. Optionally remove candidate clusters that also contain JPEG SOI, because they
       are likely starts of other files, not continuations of the current seed.
    4. Optionally keep only the strongest candidates according to the classifier.
    5. Rank remaining candidates with bGPT continuation likelihood and optional
       paper-style distribution similarity against bGPT-generated bytes.
    """
    data = np.load(args.candidates, allow_pickle=True)
    cluster_bytes = data["cluster_bytes"]
    cluster_numbers = data["cluster_number"].astype(int)
    has_seed_signature = data["has_jpg_soi"].astype(bool)
    jpg_mean, jpg_max = classifier_scores(data, args.extension)

    if args.seed_clusters:
        requested = {int(value) for value in args.seed_clusters.split(",")}
        seed_indices = np.array([idx for idx, cluster in enumerate(cluster_numbers) if int(cluster) in requested], dtype=int)
    else:
        seed_indices = np.where(has_seed_signature)[0]
        if len(seed_indices) == 0:
            seed_indices = np.argsort(jpg_mean)[::-1][: args.max_seeds]
        else:
            seed_indices = seed_indices[: args.max_seeds]

    # First pruning stage: define the pool of possible next fragments.
    candidate_indices = np.arange(len(cluster_numbers))
    if args.exclude_soi_candidates:
        # A candidate with its own JPEG SOI marker is treated as a new file start.
        candidate_indices = candidate_indices[~data["has_jpg_soi"].astype(bool)]
    if args.max_candidates:
        # Optional speed/quality tradeoff: keep only classifier-promising fragments.
        ranked = np.argsort(jpg_mean)[::-1]
        if args.exclude_soi_candidates:
            ranked = ranked[~data["has_jpg_soi"][ranked].astype(bool)]
        keep = ranked[: args.max_candidates]
        candidate_indices = np.unique(keep)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Loading bGPT on {device}: {args.weights}")
    model = load_bgpt(args.weights, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    nll_matrix = np.full((len(seed_indices), len(candidate_indices)), np.nan, dtype=np.float32)

    for seed_row, seed_idx in enumerate(seed_indices):
        seed_cluster = int(cluster_numbers[seed_idx])
        context = cluster_bytes[seed_idx][-args.context_bytes :]
        candidate_prefixes = [cluster_bytes[idx][: args.score_bytes] for idx in candidate_indices]
        contexts = [context] * len(candidate_prefixes)
        nll_values = []
        for start in range(0, len(candidate_prefixes), args.batch_size):
            nll_values.append(
                candidate_nll_batch(
                    model,
                    contexts[start : start + args.batch_size],
                    candidate_prefixes[start : start + args.batch_size],
                    args.extension,
                    device,
                )
            )
        nll_values = np.concatenate(nll_values)
        nll_matrix[seed_row] = nll_values
        # Lower NLL means the candidate bytes are more likely after this seed.
        relative = np.exp(-(nll_values - np.nanmin(nll_values)))

        paper_values = None
        paper_relative = np.ones_like(relative, dtype=np.float32)
        if args.paper_term:
            predicted = generate_bytes(
                model,
                context,
                args.extension,
                args.paper_bytes,
                device,
                temperature=args.temperature,
            )
            paper_rows = [
                paper_similarity_metrics(predicted, cluster_bytes[idx][: args.paper_bytes])
                for idx in candidate_indices
            ]
            paper_values = np.array(paper_rows, dtype=np.float32)
            paper_scores = paper_values[:, 3]
            # The paper-style score is a distance, so lower is better.
            paper_relative = np.exp(-(paper_scores - np.nanmin(paper_scores))).astype(np.float32)
            paper_path = args.output_dir / f"seed_{seed_cluster}_paper_prediction.bin"
            paper_path.write_bytes(predicted.tobytes())

        for local_idx, candidate_idx in enumerate(candidate_indices):
            if candidate_idx == seed_idx:
                continue
            paper_chi = paper_jsd = paper_cosine = paper_score = ""
            if paper_values is not None:
                paper_chi = float(paper_values[local_idx, 0])
                paper_jsd = float(paper_values[local_idx, 1])
                paper_cosine = float(paper_values[local_idx, 2])
                paper_score = float(paper_values[local_idx, 3])
            rows.append(
                {
                    "seed_cluster": seed_cluster,
                    "candidate_cluster": int(cluster_numbers[candidate_idx]),
                    "bgpt_nll": float(nll_values[local_idx]),
                    "bgpt_relative_score": float(relative[local_idx]),
                    "paper_chi": paper_chi,
                    "paper_jsd": paper_jsd,
                    "paper_cosine": paper_cosine,
                    "paper_score": paper_score,
                    "paper_relative_score": float(paper_relative[local_idx]),
                    "classifier_jpg_mean": float(jpg_mean[candidate_idx]),
                    "classifier_jpg_max": float(jpg_max[candidate_idx]),
                    "has_jpg_soi": int(data["has_jpg_soi"][candidate_idx]),
                    "has_jpg_eoi": int(data["has_jpg_eoi"][candidate_idx]),
                    "signature_count": int(data["signature_count"][candidate_idx]),
                    # Larger combined_score means a more plausible next fragment.
                    "combined_score": float(
                        relative[local_idx]
                        * paper_relative[local_idx]
                        * max(float(jpg_mean[candidate_idx]), 1e-6)
                    ),
                }
            )

        for sample in range(args.samples):
            generated = generate_bytes(
                model,
                context,
                args.extension,
                args.generated_bytes,
                device,
                temperature=args.temperature,
            )
            sample_path = args.output_dir / f"seed_{seed_cluster}_sample_{sample + 1}.bin"
            sample_path.write_bytes(generated.tobytes())

        print(f"Scored seed cluster {seed_cluster}: {len(candidate_indices):,} candidates")

    rows.sort(key=lambda row: (row["seed_cluster"], -row["combined_score"]))
    csv_path = args.output_dir / "bgpt_candidate_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.output_dir / "bgpt_candidate_scores.npz"
    np.savez_compressed(
        npz_path,
        seed_cluster=cluster_numbers[seed_indices].astype(np.int32),
        candidate_cluster=cluster_numbers[candidate_indices].astype(np.int32),
        bgpt_nll=nll_matrix,
        classifier_jpg_mean=jpg_mean[candidate_indices].astype(np.float32),
        classifier_jpg_max=jpg_max[candidate_indices].astype(np.float32),
        context_bytes=np.array(args.context_bytes),
        score_bytes=np.array(args.score_bytes),
        paper_term=np.array(args.paper_term),
        paper_bytes=np.array(args.paper_bytes),
    )
    print(f"Saved scores CSV: {csv_path}")
    print(f"Saved scores NPZ: {npz_path}")

    print("Top links by combined score:")
    for row in rows[: args.print_top]:
        paper_part = ""
        if row["paper_score"] != "":
            paper_part = f", paper={row['paper_score']:.4f}, cos={row['paper_cosine']:.4f}"
        print(
            f"  {row['seed_cluster']} -> {row['candidate_cluster']}: "
            f"combined={row['combined_score']:.6f}, "
            f"nll={row['bgpt_nll']:.4f}, jpg_mean={row['classifier_jpg_mean']:.4f}"
            f"{paper_part}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Score carving candidate clusters with bGPT likelihood.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-weights", help="Download official bGPT weights from Hugging Face.")
    download_parser.add_argument("--filename", default="weights-image.pth")
    download_parser.add_argument("--output-dir", type=Path, default=BGPT_DIR)

    score_parser = subparsers.add_parser("score", help="Score candidate clusters as possible continuations.")
    score_parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    score_parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    score_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    score_parser.add_argument("--extension", default="jpg")
    score_parser.add_argument("--context-bytes", type=int, default=4096)
    score_parser.add_argument("--score-bytes", type=int, default=512)
    score_parser.add_argument("--batch-size", type=int, default=8)
    score_parser.add_argument("--max-seeds", type=int, default=5)
    score_parser.add_argument("--max-candidates", type=int, default=0)
    score_parser.add_argument("--seed-clusters", default="")
    score_parser.add_argument("--exclude-soi-candidates", action="store_true")
    score_parser.add_argument("--samples", type=int, default=3)
    score_parser.add_argument("--generated-bytes", type=int, default=512)
    score_parser.add_argument("--paper-term", action="store_true")
    score_parser.add_argument("--paper-bytes", type=int, default=512)
    score_parser.add_argument("--temperature", type=float, default=1.0)
    score_parser.add_argument("--print-top", type=int, default=20)
    score_parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download-weights":
        path = download_weights(args.filename, args.output_dir)
        print(f"Downloaded weights: {path}")
    elif args.command == "score":
        if args.context_bytes % PATCH_SIZE != 0 or args.score_bytes % PATCH_SIZE != 0:
            raise ValueError(f"context-bytes and score-bytes must be divisible by {PATCH_SIZE}")
        if args.paper_bytes % PATCH_SIZE != 0:
            raise ValueError(f"paper-bytes must be divisible by {PATCH_SIZE}")
        score_candidates(args)


if __name__ == "__main__":
    main()

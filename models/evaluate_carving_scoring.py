from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANON2_DIR = PROJECT_ROOT / "data" / "nps-2009-canon2"
GROUNDTRUTH_DIR = CANON2_DIR / "groundtruth6"
DEFAULT_CANDIDATES = CANON2_DIR / "nps-2009-canon2-gen6_candidate_clusters.npz"
DEFAULT_SCORING = CANON2_DIR / "bgpt_scores" / "bgpt_candidate_scores.csv"
DEFAULT_OUTPUT = CANON2_DIR / "scoring_evaluation.csv"


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def load_candidate_clusters(path: Path):
    data = np.load(path)
    cluster_bytes = data["cluster_bytes"]
    cluster_numbers = data["cluster_number"].astype(int)
    cluster_size = int(data["cluster_size"])
    return cluster_bytes, cluster_numbers, cluster_size


def index_candidate_prefixes(cluster_bytes: np.ndarray, prefix_size: int = 4096):
    index = {}
    for row, values in enumerate(cluster_bytes):
        key = values[:prefix_size].tobytes()
        index.setdefault(key, []).append(row)
    return index


def map_groundtruth_file_to_candidate_clusters(
    file_path: Path,
    cluster_bytes: np.ndarray,
    cluster_numbers: np.ndarray,
    cluster_size: int,
    prefix_index,
    min_last_chunk: int,
):
    content = file_path.read_bytes()
    chunks = [content[start : start + cluster_size] for start in range(0, len(content), cluster_size)]
    mapped = []

    for chunk_index, chunk in enumerate(chunks):
        is_last = chunk_index == len(chunks) - 1
        if is_last and len(chunk) < min_last_chunk:
            mapped.append(None)
            continue

        if len(chunk) == cluster_size:
            key = chunk[:4096]
            candidate_rows = prefix_index.get(key, [])
            matches = [
                row
                for row in candidate_rows
                if cluster_bytes[row].tobytes() == chunk
            ]
        else:
            key = chunk[:4096]
            candidate_rows = prefix_index.get(key, [])
            matches = [
                row
                for row in candidate_rows
                if cluster_bytes[row][: len(chunk)].tobytes() == chunk
            ]

        if len(matches) == 1:
            mapped.append(int(cluster_numbers[matches[0]]))
        elif len(matches) > 1:
            mapped.append(int(cluster_numbers[matches[0]]))
        else:
            mapped.append(None)

    return {
        "filename": file_path.name,
        "size": len(content),
        "sha1": sha1_bytes(content),
        "chunk_count": len(chunks),
        "mapped_clusters": mapped,
    }


def load_scoring(path: Path):
    by_seed = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            seed = int(row["seed_cluster"])
            candidate = int(row["candidate_cluster"])
            score = float(row["combined_score"])
            by_seed.setdefault(seed, []).append((candidate, score, row))

    for seed, rows in by_seed.items():
        rows.sort(key=lambda item: item[1], reverse=True)
        by_seed[seed] = rows
    return by_seed


def rank_candidate(scoring_by_seed, seed: int, target: int):
    rows = scoring_by_seed.get(seed)
    if rows is None:
        return None, None
    for index, (candidate, score, row) in enumerate(rows, start=1):
        if candidate == target:
            return index, row
    return None, None


def evaluate(args) -> None:
    cluster_bytes, cluster_numbers, cluster_size = load_candidate_clusters(args.candidates)
    prefix_index = index_candidate_prefixes(cluster_bytes)
    scoring_by_seed = load_scoring(args.scoring)

    files = sorted({path.resolve() for path in args.groundtruth.glob("*") if path.suffix.lower() == ".jpg"})
    mappings = [
        map_groundtruth_file_to_candidate_clusters(
            file_path,
            cluster_bytes,
            cluster_numbers,
            cluster_size,
            prefix_index,
            args.min_last_chunk,
        )
        for file_path in files
    ]

    rows = []
    for mapping in mappings:
        clusters = mapping["mapped_clusters"]
        for chunk_index in range(len(clusters) - 1):
            seed = clusters[chunk_index]
            target = clusters[chunk_index + 1]
            if seed is None or target is None:
                continue
            rank, score_row = rank_candidate(scoring_by_seed, seed, target)
            rows.append(
                {
                    "filename": mapping["filename"],
                    "sha1": mapping["sha1"],
                    "chunk_index": chunk_index,
                    "seed_cluster": seed,
                    "target_cluster": target,
                    "rank": "" if rank is None else rank,
                    "top1": int(rank == 1) if rank is not None else 0,
                    "top3": int(rank is not None and rank <= 3),
                    "top5": int(rank is not None and rank <= 5),
                    "top10": int(rank is not None and rank <= 10),
                    "combined_score": "" if score_row is None else score_row["combined_score"],
                    "bgpt_nll": "" if score_row is None else score_row["bgpt_nll"],
                    "classifier_jpg_mean": "" if score_row is None else score_row["classifier_jpg_mean"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "sha1",
        "chunk_index",
        "seed_cluster",
        "target_cluster",
        "rank",
        "top1",
        "top3",
        "top5",
        "top10",
        "combined_score",
        "bgpt_nll",
        "classifier_jpg_mean",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mapped_files = [m for m in mappings if any(cluster is not None for cluster in m["mapped_clusters"])]
    fully_mapped_files = [m for m in mappings if all(cluster is not None for cluster in m["mapped_clusters"])]
    evaluated = [row for row in rows if row["rank"] != ""]

    print(f"Groundtruth files: {len(files):,}")
    print(f"Files with at least one chunk in candidate pool: {len(mapped_files):,}")
    print(f"Files fully mapped to candidate pool: {len(fully_mapped_files):,}")
    print(f"Evaluated true transitions: {len(rows):,}")
    print(f"Transitions found in scoring CSV: {len(evaluated):,}")
    if evaluated:
        for key in ["top1", "top3", "top5", "top10"]:
            value = sum(int(row[key]) for row in evaluated) / len(evaluated)
            print(f"{key}: {value:.2%}")
        ranks = [int(row["rank"]) for row in evaluated]
        mrr = sum(1.0 / rank for rank in ranks) / len(ranks)
        print(f"MRR: {mrr:.4f}")
    print(f"Saved evaluation: {args.output}")

    if args.print_mappings:
        for mapping in mappings:
            if any(cluster is not None for cluster in mapping["mapped_clusters"]):
                print(mapping["filename"], mapping["mapped_clusters"])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate carving scoring against groundtruth JPEG files.")
    parser.add_argument("--groundtruth", type=Path, default=GROUNDTRUTH_DIR)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--scoring", type=Path, default=DEFAULT_SCORING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-last-chunk", type=int, default=4096)
    parser.add_argument("--print-mappings", action="store_true")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()

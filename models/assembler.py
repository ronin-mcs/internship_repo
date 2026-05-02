from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


CANON2_DIR = Path("data") / "nps-2009-canon2"
GENERATION = 6
SECTOR_SIZE = 512
RAW_SIZE = 31_129_600
REPORT_XML = CANON2_DIR / "report.xml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSION_CHECKPOINT = PROJECT_ROOT / "models" / "artifacts" / "artifacts_fusion_512_noRandNoise" / "byte_chunk_fusion_cnn.pt"
FIFTY_MODEL = PROJECT_ROOT / "models" / "fifty" / "fifty" / "utilities" / "models" / "512_1.h5"
FIFTY_LABELS = PROJECT_ROOT / "models" / "fifty" / "fifty" / "utilities" / "labels.json"
SIGNATURES = {
    "jpg_soi": bytes.fromhex("ff d8 ff"),
    "jpg_eoi": bytes.fromhex("ff d9"),
    "png": bytes.fromhex("89 50 4e 47 0d 0a 1a 0a"),
    "pdf": b"%PDF",
    "gif87a": b"GIF87a",
    "gif89a": b"GIF89a",
    "zip": bytes.fromhex("50 4b 03 04"),
}
EXPECTED_SHA1 = {
    1: "67364b0894a0465d6ada8c4966b6bbcaf7039082",
    2: "0e3cdef3b1a7d3762f9704bfd4349033fe808eda",
    3: "7dc8be7f3993c37f101c0ed0fec4274abccacf3c",
    4: "ed1c7dea94096ad309b32037cb6d43a291952d8d",
    5: "63e7f9daf8dbcd1744579e579f3f0fddebe2ee90",
    6: "4742c325f10583dab1eb4c55d0d45ab3beb99eb3",
}


def e01_path(generation: int = GENERATION) -> Path:
    return CANON2_DIR / f"nps-2009-canon2-gen{generation}.E01"


def raw_path(generation: int = GENERATION) -> Path:
    return CANON2_DIR / f"nps-2009-canon2-gen{generation}.raw"


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_project_root() -> None:
    if not CANON2_DIR.exists():
        raise FileNotFoundError(
            f"Expected {CANON2_DIR} to exist. Run this script from the project root."
        )


def convert_e01_to_raw(generation: int = GENERATION, overwrite: bool = False) -> Path:
    require_project_root()

    source = e01_path(generation)
    target = raw_path(generation)

    if not source.exists():
        raise FileNotFoundError(f"E01 image not found: {source}")

    if target.exists() and not overwrite:
        print(f"Raw image already exists: {target}")
        return target

    ewfexport = shutil.which("ewfexport")
    if ewfexport is None:
        raise RuntimeError(
            "ewfexport is not installed or not in PATH, so I cannot convert E01 to raw here.\n"
            "Install libewf tools, then run:\n"
            f"  ewfexport -t \"{target.with_suffix('')}\" -f raw \"{source}\"\n\n"
            "Alternative: use FTK Imager / Arsenal Image Mounter to export the E01 as raw/dd, "
            f"then save it as:\n  {target}"
        )

    target_prefix = str(target.with_suffix(""))
    command = [
        ewfexport,
        "-f",
        "raw",
        "-t",
        target_prefix,
        str(source),
    ]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)

    exported = Path(target_prefix)
    if exported.exists() and exported != target:
        exported.replace(target)

    if not target.exists():
        raise FileNotFoundError(
            f"ewfexport finished, but expected raw file was not created: {target}"
        )

    return target


def load_raw_image(generation: int = GENERATION) -> Path:
    require_project_root()
    path = raw_path(generation)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw image not found: {path}\n"
            f"Run: python models\\assembler.py convert --generation {generation}"
        )
    return path


def print_raw_info(path: Path, generation: int = GENERATION) -> None:
    size = path.stat().st_size
    digest = sha1_file(path)
    expected = EXPECTED_SHA1.get(generation)

    print(f"Raw image: {path}")
    print(f"Size: {size:,} bytes")
    print(f"Sectors ({SECTOR_SIZE} bytes): {size // SECTOR_SIZE:,}")
    print(f"SHA1: {digest}")
    if expected is not None:
        print(f"Expected SHA1: {expected}")
        print(f"SHA1 match: {digest == expected}")
    if size != RAW_SIZE:
        print(f"Warning: expected raw size {RAW_SIZE:,} bytes for canon2 images.")


def read_sector(path: Path, sector: int, count: int = 1) -> bytes:
    with path.open("rb") as file:
        file.seek(sector * SECTOR_SIZE)
        return file.read(count * SECTOR_SIZE)


def le16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "little")


def le32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "little")


def parse_fat_boot_sector(raw_image: Path):
    mbr = read_sector(raw_image, 0)
    partition_lba = None

    for index in range(4):
        entry_offset = 446 + index * 16
        partition_type = mbr[entry_offset + 4]
        start_lba = le32(mbr, entry_offset + 8)
        sector_count = le32(mbr, entry_offset + 12)
        if partition_type != 0 and sector_count > 0:
            partition_lba = start_lba
            break

    if partition_lba is None:
        partition_lba = 0

    boot = read_sector(raw_image, partition_lba)
    bytes_per_sector = le16(boot, 11)
    sectors_per_cluster = boot[13]
    reserved_sector_count = le16(boot, 14)
    number_of_fats = boot[16]
    root_entry_count = le16(boot, 17)
    total_sectors_16 = le16(boot, 19)
    total_sectors_32 = le32(boot, 32)
    fat_size_16 = le16(boot, 22)
    fat_size_32 = le32(boot, 36)
    root_cluster = le32(boot, 44) if fat_size_16 == 0 else 0
    fs_info_sector = le16(boot, 48) if fat_size_16 == 0 else 0
    backup_boot_sector = le16(boot, 50) if fat_size_16 == 0 else 0
    volume_label_offset = 71 if fat_size_16 == 0 else 43
    filesystem_type_offset = 82 if fat_size_16 == 0 else 54
    volume_label = boot[volume_label_offset : volume_label_offset + 11].decode("ascii", errors="ignore").strip()
    filesystem_type = boot[filesystem_type_offset : filesystem_type_offset + 8].decode("ascii", errors="ignore").strip()

    total_sectors = total_sectors_16 if total_sectors_16 else total_sectors_32
    fat_size = fat_size_16 if fat_size_16 else fat_size_32
    root_dir_sectors = math.ceil((root_entry_count * 32) / bytes_per_sector)
    first_fat_sector = partition_lba + reserved_sector_count
    first_root_dir_sector = partition_lba + reserved_sector_count + number_of_fats * fat_size
    first_data_sector = first_root_dir_sector + root_dir_sectors
    relative_first_data_sector = reserved_sector_count + number_of_fats * fat_size + root_dir_sectors
    data_sectors = total_sectors - relative_first_data_sector
    cluster_count = data_sectors // sectors_per_cluster
    cluster_size = bytes_per_sector * sectors_per_cluster

    return {
        "bytes_per_sector": bytes_per_sector,
        "partition_lba": partition_lba,
        "sectors_per_cluster": sectors_per_cluster,
        "cluster_size": cluster_size,
        "reserved_sector_count": reserved_sector_count,
        "number_of_fats": number_of_fats,
        "root_entry_count": root_entry_count,
        "root_dir_sectors": root_dir_sectors,
        "fat_size_sectors": fat_size,
        "first_fat_sector": first_fat_sector,
        "first_root_dir_sector": first_root_dir_sector,
        "first_data_sector": first_data_sector,
        "relative_first_data_sector": relative_first_data_sector,
        "data_sectors": data_sectors,
        "cluster_count": cluster_count,
        "root_cluster": root_cluster,
        "fs_info_sector": fs_info_sector,
        "backup_boot_sector": backup_boot_sector,
        "total_sectors": total_sectors,
        "volume_label": volume_label,
        "filesystem_type": filesystem_type,
    }


def cluster_to_sector(cluster_number: int, fat32):
    return fat32["first_data_sector"] + (cluster_number - 2) * fat32["sectors_per_cluster"]


def sector_to_data_cluster(sector: int, fat32):
    if sector < fat32["first_data_sector"]:
        return None
    relative_sector = sector - fat32["first_data_sector"]
    cluster_index = relative_sector // fat32["sectors_per_cluster"]
    if cluster_index >= fat32["cluster_count"]:
        return None
    return cluster_index + 2


def iter_report_runs(report_xml: Path = REPORT_XML):
    root = ET.parse(report_xml).getroot()
    for file_object in root.findall("fileobject"):
        filename = file_object.findtext("filename", default="")
        alloc = file_object.findtext("ALLOC", default="")
        used = file_object.findtext("USED", default="")
        reference = file_object.find("reference")
        reason = reference.findtext("reason", default="") if reference is not None else ""
        byte_runs = file_object.find("byte_runs")
        if byte_runs is None:
            continue

        for run in byte_runs.findall("run"):
            yield {
                "filename": filename,
                "alloc": alloc,
                "used": used,
                "reason": reason,
                "img_offset": int(run.attrib["img_offset"]),
                "length": int(run.attrib["len"]),
                "file_offset": int(run.attrib.get("file_offset", 0)),
            }


def sectors_touched(offset: int, length: int):
    start_sector = offset // SECTOR_SIZE
    end_sector = (offset + length - 1) // SECTOR_SIZE
    return range(start_sector, end_sector + 1)


def build_sector_masks(raw_image: Path, report_xml: Path = REPORT_XML):
    sector_count = raw_image.stat().st_size // SECTOR_SIZE
    is_reported_file = np.zeros(sector_count, dtype=bool)
    is_resident_file = np.zeros(sector_count, dtype=bool)
    is_residual_file = np.zeros(sector_count, dtype=bool)
    run_touch_count = np.zeros(sector_count, dtype=np.uint16)

    file_rows = []
    for item in iter_report_runs(report_xml):
        touched = list(sectors_touched(item["img_offset"], item["length"]))
        reason = item["reason"]

        for sector in touched:
            if 0 <= sector < sector_count:
                is_reported_file[sector] = True
                run_touch_count[sector] += 1
                if reason == "resident file":
                    is_resident_file[sector] = True
                elif reason == "residual file":
                    is_residual_file[sector] = True

        file_rows.append(
            {
                "filename": item["filename"],
                "reason": reason,
                "alloc": item["alloc"],
                "used": item["used"],
                "img_offset": item["img_offset"],
                "length": item["length"],
                "start_sector": touched[0],
                "end_sector": touched[-1],
                "sector_count": len(touched),
            }
        )

    return {
        "sector_count": sector_count,
        "is_reported_file": is_reported_file,
        "is_resident_file": is_resident_file,
        "is_residual_file": is_residual_file,
        "run_touch_count": run_touch_count,
        "file_rows": file_rows,
    }


def save_preprocessing_outputs(raw_image: Path, masks, output_prefix: Path) -> None:
    sector_index = np.arange(masks["sector_count"], dtype=np.int32)
    offset = sector_index.astype(np.int64) * SECTOR_SIZE
    candidate_not_resident = ~masks["is_resident_file"]
    candidate_not_reported = ~masks["is_reported_file"]

    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")
    runs_csv_path = output_prefix.with_name(output_prefix.name + "_runs.csv")

    np.savez_compressed(
        npz_path,
        sector_index=sector_index,
        offset=offset,
        is_reported_file=masks["is_reported_file"],
        is_resident_file=masks["is_resident_file"],
        is_residual_file=masks["is_residual_file"],
        candidate_not_resident=candidate_not_resident,
        candidate_not_reported=candidate_not_reported,
        run_touch_count=masks["run_touch_count"],
        sector_size=np.array(SECTOR_SIZE),
        raw_image=np.array(str(raw_image)),
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sector",
                "offset",
                "is_reported_file",
                "is_resident_file",
                "is_residual_file",
                "candidate_not_resident",
                "candidate_not_reported",
                "run_touch_count",
            ]
        )
        for idx in range(masks["sector_count"]):
            writer.writerow(
                [
                    int(sector_index[idx]),
                    int(offset[idx]),
                    int(masks["is_reported_file"][idx]),
                    int(masks["is_resident_file"][idx]),
                    int(masks["is_residual_file"][idx]),
                    int(candidate_not_resident[idx]),
                    int(candidate_not_reported[idx]),
                    int(masks["run_touch_count"][idx]),
                ]
            )

    with runs_csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "filename",
            "reason",
            "alloc",
            "used",
            "img_offset",
            "length",
            "start_sector",
            "end_sector",
            "sector_count",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(masks["file_rows"])

    print(f"Saved sector masks: {npz_path}")
    print(f"Saved sector CSV: {csv_path}")
    print(f"Saved report runs CSV: {runs_csv_path}")


def preprocess_known_sectors(generation: int = GENERATION) -> None:
    raw_image = load_raw_image(generation)
    masks = build_sector_masks(raw_image)
    output_prefix = CANON2_DIR / f"nps-2009-canon2-gen{generation}_sector_map"
    save_preprocessing_outputs(raw_image, masks, output_prefix)

    total = masks["sector_count"]
    resident = int(masks["is_resident_file"].sum())
    residual = int(masks["is_residual_file"].sum())
    reported = int(masks["is_reported_file"].sum())

    print("Preprocessing summary")
    print(f"  total sectors: {total:,}")
    print(f"  resident file sectors: {resident:,} ({resident / total:.2%})")
    print(f"  residual file sectors: {residual:,} ({residual / total:.2%})")
    print(f"  all report.xml file sectors: {reported:,} ({reported / total:.2%})")
    print(f"  candidates if excluding resident sectors: {total - resident:,}")
    print(f"  candidates if excluding all reported sectors: {total - reported:,}")


def print_fat_info(raw_image: Path) -> None:
    fat32 = parse_fat_boot_sector(raw_image)
    print("FAT layout")
    for key in [
        "filesystem_type",
        "volume_label",
        "partition_lba",
        "bytes_per_sector",
        "sectors_per_cluster",
        "cluster_size",
        "reserved_sector_count",
        "number_of_fats",
        "root_entry_count",
        "root_dir_sectors",
        "fat_size_sectors",
        "first_fat_sector",
        "first_root_dir_sector",
        "first_data_sector",
        "data_sectors",
        "cluster_count",
        "root_cluster",
        "total_sectors",
    ]:
        print(f"  {key}: {fat32[key]}")


def build_cluster_map(raw_image: Path):
    fat32 = parse_fat_boot_sector(raw_image)
    sector_masks = build_sector_masks(raw_image)
    cluster_numbers = np.arange(2, fat32["cluster_count"] + 2, dtype=np.int32)
    cluster_count = len(cluster_numbers)
    sectors_per_cluster = fat32["sectors_per_cluster"]
    first_sector = np.array(
        [cluster_to_sector(int(cluster), fat32) for cluster in cluster_numbers],
        dtype=np.int32,
    )
    offset = first_sector.astype(np.int64) * fat32["bytes_per_sector"]

    resident_sector_count = np.zeros(cluster_count, dtype=np.uint16)
    residual_sector_count = np.zeros(cluster_count, dtype=np.uint16)
    reported_sector_count = np.zeros(cluster_count, dtype=np.uint16)

    for idx, start_sector in enumerate(first_sector):
        end_sector = min(start_sector + sectors_per_cluster, sector_masks["sector_count"])
        sector_slice = slice(start_sector, end_sector)
        resident_sector_count[idx] = sector_masks["is_resident_file"][sector_slice].sum()
        residual_sector_count[idx] = sector_masks["is_residual_file"][sector_slice].sum()
        reported_sector_count[idx] = sector_masks["is_reported_file"][sector_slice].sum()

    is_resident_cluster = resident_sector_count > 0
    is_residual_cluster = residual_sector_count > 0
    is_reported_cluster = reported_sector_count > 0
    is_full_resident_cluster = resident_sector_count == sectors_per_cluster
    is_full_reported_cluster = reported_sector_count == sectors_per_cluster

    return {
        "fat32": fat32,
        "cluster_numbers": cluster_numbers,
        "first_sector": first_sector,
        "offset": offset,
        "resident_sector_count": resident_sector_count,
        "residual_sector_count": residual_sector_count,
        "reported_sector_count": reported_sector_count,
        "is_resident_cluster": is_resident_cluster,
        "is_residual_cluster": is_residual_cluster,
        "is_reported_cluster": is_reported_cluster,
        "is_full_resident_cluster": is_full_resident_cluster,
        "is_full_reported_cluster": is_full_reported_cluster,
    }


def save_cluster_map(raw_image: Path, cluster_map, output_prefix: Path) -> None:
    fat32 = cluster_map["fat32"]
    candidate_not_resident = ~cluster_map["is_resident_cluster"]
    candidate_not_reported = ~cluster_map["is_reported_cluster"]
    candidate_not_full_resident = ~cluster_map["is_full_resident_cluster"]

    npz_path = output_prefix.with_suffix(".npz")
    csv_path = output_prefix.with_suffix(".csv")

    np.savez_compressed(
        npz_path,
        cluster_number=cluster_map["cluster_numbers"],
        first_sector=cluster_map["first_sector"],
        offset=cluster_map["offset"],
        resident_sector_count=cluster_map["resident_sector_count"],
        residual_sector_count=cluster_map["residual_sector_count"],
        reported_sector_count=cluster_map["reported_sector_count"],
        is_resident_cluster=cluster_map["is_resident_cluster"],
        is_residual_cluster=cluster_map["is_residual_cluster"],
        is_reported_cluster=cluster_map["is_reported_cluster"],
        is_full_resident_cluster=cluster_map["is_full_resident_cluster"],
        is_full_reported_cluster=cluster_map["is_full_reported_cluster"],
        candidate_not_resident=candidate_not_resident,
        candidate_not_reported=candidate_not_reported,
        candidate_not_full_resident=candidate_not_full_resident,
        bytes_per_sector=np.array(fat32["bytes_per_sector"]),
        sectors_per_cluster=np.array(fat32["sectors_per_cluster"]),
        cluster_size=np.array(fat32["cluster_size"]),
        first_data_sector=np.array(fat32["first_data_sector"]),
        raw_image=np.array(str(raw_image)),
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "cluster_number",
                "first_sector",
                "offset",
                "resident_sector_count",
                "residual_sector_count",
                "reported_sector_count",
                "is_resident_cluster",
                "is_residual_cluster",
                "is_reported_cluster",
                "is_full_resident_cluster",
                "is_full_reported_cluster",
                "candidate_not_resident",
                "candidate_not_reported",
                "candidate_not_full_resident",
            ]
        )
        for idx, cluster_number in enumerate(cluster_map["cluster_numbers"]):
            writer.writerow(
                [
                    int(cluster_number),
                    int(cluster_map["first_sector"][idx]),
                    int(cluster_map["offset"][idx]),
                    int(cluster_map["resident_sector_count"][idx]),
                    int(cluster_map["residual_sector_count"][idx]),
                    int(cluster_map["reported_sector_count"][idx]),
                    int(cluster_map["is_resident_cluster"][idx]),
                    int(cluster_map["is_residual_cluster"][idx]),
                    int(cluster_map["is_reported_cluster"][idx]),
                    int(cluster_map["is_full_resident_cluster"][idx]),
                    int(cluster_map["is_full_reported_cluster"][idx]),
                    int(candidate_not_resident[idx]),
                    int(candidate_not_reported[idx]),
                    int(candidate_not_full_resident[idx]),
                ]
            )

    print(f"Saved cluster map: {npz_path}")
    print(f"Saved cluster CSV: {csv_path}")


def preprocess_clusters(generation: int = GENERATION) -> None:
    raw_image = load_raw_image(generation)
    print_fat_info(raw_image)
    cluster_map = build_cluster_map(raw_image)
    output_prefix = CANON2_DIR / f"nps-2009-canon2-gen{generation}_cluster_map"
    save_cluster_map(raw_image, cluster_map, output_prefix)

    total = len(cluster_map["cluster_numbers"])
    resident = int(cluster_map["is_resident_cluster"].sum())
    residual = int(cluster_map["is_residual_cluster"].sum())
    reported = int(cluster_map["is_reported_cluster"].sum())
    full_resident = int(cluster_map["is_full_resident_cluster"].sum())
    full_reported = int(cluster_map["is_full_reported_cluster"].sum())

    print("Cluster preprocessing summary")
    print(f"  total data clusters: {total:,}")
    print(f"  clusters touching resident files: {resident:,} ({resident / total:.2%})")
    print(f"  clusters touching residual files: {residual:,} ({residual / total:.2%})")
    print(f"  clusters touching report.xml files: {reported:,} ({reported / total:.2%})")
    print(f"  fully resident clusters: {full_resident:,} ({full_resident / total:.2%})")
    print(f"  fully reported clusters: {full_reported:,} ({full_reported / total:.2%})")
    print(f"  candidates excluding clusters that touch resident files: {total - resident:,}")
    print(f"  candidates excluding only fully resident clusters: {total - full_resident:,}")
    print(f"  candidates excluding clusters that touch all reported files: {total - reported:,}")


def load_cluster_candidate_lookup(generation: int = GENERATION):
    path = CANON2_DIR / f"nps-2009-canon2-gen{generation}_cluster_map.npz"
    if not path.exists():
        preprocess_clusters(generation)

    data = np.load(path)
    return {
        "cluster_number": data["cluster_number"],
        "candidate_not_resident": data["candidate_not_resident"],
        "candidate_not_reported": data["candidate_not_reported"],
        "candidate_not_full_resident": data["candidate_not_full_resident"],
        "first_data_sector": int(data["first_data_sector"]),
        "sectors_per_cluster": int(data["sectors_per_cluster"]),
    }


def find_all(data: bytes, needle: bytes):
    start = 0
    while True:
        index = data.find(needle, start)
        if index == -1:
            break
        yield index
        start = index + 1


def scan_signatures(generation: int = GENERATION) -> None:
    raw_image = load_raw_image(generation)
    fat = parse_fat_boot_sector(raw_image)
    cluster_lookup = load_cluster_candidate_lookup(generation)
    raw_bytes = raw_image.read_bytes()

    rows = []
    for signature_name, pattern in SIGNATURES.items():
        for offset in find_all(raw_bytes, pattern):
            sector = offset // fat["bytes_per_sector"]
            in_sector_offset = offset % fat["bytes_per_sector"]
            cluster_number = sector_to_data_cluster(sector, fat)

            candidate_not_resident = None
            candidate_not_reported = None
            candidate_not_full_resident = None

            if cluster_number is not None:
                cluster_index = int(cluster_number - 2)
                if 0 <= cluster_index < len(cluster_lookup["cluster_number"]):
                    candidate_not_resident = bool(cluster_lookup["candidate_not_resident"][cluster_index])
                    candidate_not_reported = bool(cluster_lookup["candidate_not_reported"][cluster_index])
                    candidate_not_full_resident = bool(cluster_lookup["candidate_not_full_resident"][cluster_index])

            rows.append(
                {
                    "signature": signature_name,
                    "offset": offset,
                    "sector": sector,
                    "in_sector_offset": in_sector_offset,
                    "cluster_number": cluster_number if cluster_number is not None else "",
                    "candidate_not_resident": candidate_not_resident,
                    "candidate_not_reported": candidate_not_reported,
                    "candidate_not_full_resident": candidate_not_full_resident,
                }
            )

    rows.sort(key=lambda row: (row["offset"], row["signature"]))
    output_csv = CANON2_DIR / f"nps-2009-canon2-gen{generation}_signature_scan.csv"
    output_npz = CANON2_DIR / f"nps-2009-canon2-gen{generation}_signature_scan.npz"

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "signature",
            "offset",
            "sector",
            "in_sector_offset",
            "cluster_number",
            "candidate_not_resident",
            "candidate_not_reported",
            "candidate_not_full_resident",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        output_npz,
        signature=np.array([row["signature"] for row in rows]),
        offset=np.array([row["offset"] for row in rows], dtype=np.int64),
        sector=np.array([row["sector"] for row in rows], dtype=np.int32),
        in_sector_offset=np.array([row["in_sector_offset"] for row in rows], dtype=np.int16),
        cluster_number=np.array(
            [row["cluster_number"] if row["cluster_number"] != "" else -1 for row in rows],
            dtype=np.int32,
        ),
        candidate_not_resident=np.array(
            [False if row["candidate_not_resident"] is None else row["candidate_not_resident"] for row in rows],
            dtype=bool,
        ),
        candidate_not_reported=np.array(
            [False if row["candidate_not_reported"] is None else row["candidate_not_reported"] for row in rows],
            dtype=bool,
        ),
        candidate_not_full_resident=np.array(
            [
                False if row["candidate_not_full_resident"] is None else row["candidate_not_full_resident"]
                for row in rows
            ],
            dtype=bool,
        ),
    )

    print(f"Saved signature scan CSV: {output_csv}")
    print(f"Saved signature scan NPZ: {output_npz}")
    print("Signature counts:")
    counts = {}
    for row in rows:
        counts[row["signature"]] = counts.get(row["signature"], 0) + 1
    for signature_name, count in sorted(counts.items()):
        print(f"  {signature_name}: {count:,}")

    jpg_starts = [row for row in rows if row["signature"] == "jpg_soi"]
    candidate_jpg_starts = [row for row in jpg_starts if row["candidate_not_resident"]]
    print(f"JPEG SOI total: {len(jpg_starts):,}")
    print(f"JPEG SOI in candidate_not_resident clusters: {len(candidate_jpg_starts):,}")


def load_signature_flags(generation: int = GENERATION):
    path = CANON2_DIR / f"nps-2009-canon2-gen{generation}_signature_scan.npz"
    if not path.exists():
        scan_signatures(generation)
    data = np.load(path)
    return {
        "signature": data["signature"].astype(str),
        "offset": data["offset"],
        "sector": data["sector"],
        "cluster_number": data["cluster_number"],
    }


def import_feature_extractor():
    module_path = PROJECT_ROOT / "data" / "construct_dataset_for_classification.py"
    spec = importlib.util.spec_from_file_location("construct_dataset_for_classification", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import feature extractor from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_features


def load_our_fusion_model(checkpoint_path: Path = FUSION_CHECKPOINT):
    import torch

    sys.path.insert(0, str(PROJECT_ROOT / "models"))
    from internship_repo.models.filetype_classifier_fusion import ByteStatsFusionCNN

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    label_names = checkpoint["label_names"]
    feature_names = checkpoint["feature_names"]
    scaler = checkpoint["feature_scaler"]
    model = ByteStatsFusionCNN(num_classes=len(label_names), num_stat_features=len(feature_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, scaler, label_names, feature_names, device


def predict_our_fusion(model, scaler, sectors: np.ndarray, batch_size: int, device):
    import torch

    extract_features = import_feature_extractor()
    x = sectors.astype(np.float32) / 255.0
    features = extract_features(sectors).astype(np.float32)
    features = scaler.transform(features).astype(np.float32)
    probabilities = []

    for start in range(0, len(x), batch_size):
        x_batch = torch.from_numpy(x[start : start + batch_size]).unsqueeze(1).to(device)
        f_batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        with torch.no_grad():
            logits = model(x_batch, f_batch)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(probabilities, axis=0).astype(np.float32)


def try_load_fifty_model(model_path: Path = FIFTY_MODEL):
    try:
        from tensorflow.keras.models import load_model
    except ModuleNotFoundError:
        return None, None, "TensorFlow/Keras is not installed"

    if not model_path.exists():
        return None, None, f"FiFTy model not found: {model_path}"

    try:
        import json

        labels = json.loads(FIFTY_LABELS.read_text(encoding="utf-8"))["1"]
        model = load_model(str(model_path), compile=False)
        return model, labels, None
    except Exception as exc:
        return None, None, f"Could not load FiFTy model: {type(exc).__name__}: {exc}"


def predict_fifty(model, sectors: np.ndarray, batch_size: int):
    probabilities = model.predict(sectors.astype(np.uint8), batch_size=batch_size, verbose=0)
    return probabilities.astype(np.float32)


def build_candidate_clusters(generation: int = GENERATION, batch_size: int = 512) -> None:
    raw_image = load_raw_image(generation)
    fat = parse_fat_boot_sector(raw_image)
    cluster_map_path = CANON2_DIR / f"nps-2009-canon2-gen{generation}_cluster_map.npz"
    if not cluster_map_path.exists():
        preprocess_clusters(generation)

    cluster_map = np.load(cluster_map_path)
    candidate_mask = cluster_map["candidate_not_resident"].astype(bool)
    cluster_numbers = cluster_map["cluster_number"][candidate_mask]
    first_sectors = cluster_map["first_sector"][candidate_mask]
    offsets = cluster_map["offset"][candidate_mask]
    sectors_per_cluster = int(cluster_map["sectors_per_cluster"])
    cluster_size = int(cluster_map["cluster_size"])

    cluster_bytes = np.empty((len(cluster_numbers), cluster_size), dtype=np.uint8)
    with raw_image.open("rb") as file:
        for index, offset in enumerate(offsets):
            file.seek(int(offset))
            data = file.read(cluster_size)
            if len(data) != cluster_size:
                data = data + bytes(cluster_size - len(data))
            cluster_bytes[index] = np.frombuffer(data, dtype=np.uint8)

    sector_bytes = cluster_bytes.reshape(len(cluster_numbers) * sectors_per_cluster, SECTOR_SIZE)

    signature_data = load_signature_flags(generation)
    cluster_to_index = {int(cluster): idx for idx, cluster in enumerate(cluster_numbers)}
    has_jpg_soi = np.zeros(len(cluster_numbers), dtype=bool)
    has_jpg_eoi = np.zeros(len(cluster_numbers), dtype=bool)
    signature_count = np.zeros(len(cluster_numbers), dtype=np.uint16)
    jpg_soi_offsets = []

    for signature, offset, cluster_number in zip(
        signature_data["signature"],
        signature_data["offset"],
        signature_data["cluster_number"],
    ):
        idx = cluster_to_index.get(int(cluster_number))
        if idx is None:
            continue
        signature_count[idx] += 1
        if signature == "jpg_soi":
            has_jpg_soi[idx] = True
            jpg_soi_offsets.append(int(offset))
        elif signature == "jpg_eoi":
            has_jpg_eoi[idx] = True

    print(f"Candidate clusters: {len(cluster_numbers):,}")
    print(f"Candidate sectors for inference: {len(sector_bytes):,}")
    print(f"Candidate clusters with JPEG SOI: {int(has_jpg_soi.sum()):,}")
    print(f"Candidate clusters with JPEG EOI: {int(has_jpg_eoi.sum()):,}")

    print("Loading our fusion model...")
    our_model, our_scaler, our_labels, our_feature_names, device = load_our_fusion_model()
    our_sector_probs = predict_our_fusion(our_model, our_scaler, sector_bytes, batch_size, device)
    our_sector_probs = our_sector_probs.reshape(len(cluster_numbers), sectors_per_cluster, -1)
    our_cluster_mean_probs = our_sector_probs.mean(axis=1)
    our_cluster_max_probs = our_sector_probs.max(axis=1)

    print("Loading FiFTy model...")
    fifty_model, fifty_labels, fifty_error = try_load_fifty_model()
    fifty_available = fifty_model is not None
    if fifty_model is not None:
        fifty_sector_probs = predict_fifty(fifty_model, sector_bytes, batch_size)
        fifty_sector_probs = fifty_sector_probs.reshape(len(cluster_numbers), sectors_per_cluster, -1)
        fifty_cluster_mean_probs = fifty_sector_probs.mean(axis=1)
        fifty_cluster_max_probs = fifty_sector_probs.max(axis=1)
    else:
        print(f"FiFTy unavailable: {fifty_error}")
        fifty_labels = []
        fifty_sector_probs = np.empty((len(cluster_numbers), sectors_per_cluster, 0), dtype=np.float32)
        fifty_cluster_mean_probs = np.empty((len(cluster_numbers), 0), dtype=np.float32)
        fifty_cluster_max_probs = np.empty((len(cluster_numbers), 0), dtype=np.float32)

    output_path = CANON2_DIR / f"nps-2009-canon2-gen{generation}_candidate_clusters.npz"
    np.savez_compressed(
        output_path,
        cluster_bytes=cluster_bytes,
        cluster_number=cluster_numbers.astype(np.int32),
        first_sector=first_sectors.astype(np.int32),
        offset=offsets.astype(np.int64),
        has_jpg_soi=has_jpg_soi,
        has_jpg_eoi=has_jpg_eoi,
        signature_count=signature_count,
        sectors_per_cluster=np.array(sectors_per_cluster),
        sector_size=np.array(SECTOR_SIZE),
        cluster_size=np.array(cluster_size),
        our_label_names=np.array(our_labels),
        our_feature_names=np.array(our_feature_names),
        our_sector_probs=our_sector_probs.astype(np.float32),
        our_cluster_mean_probs=our_cluster_mean_probs.astype(np.float32),
        our_cluster_max_probs=our_cluster_max_probs.astype(np.float32),
        fifty_available=np.array(fifty_available),
        fifty_error=np.array("" if fifty_error is None else fifty_error),
        fifty_label_names=np.array(fifty_labels),
        fifty_sector_probs=fifty_sector_probs.astype(np.float32),
        fifty_cluster_mean_probs=fifty_cluster_mean_probs.astype(np.float32),
        fifty_cluster_max_probs=fifty_cluster_max_probs.astype(np.float32),
        raw_image=np.array(str(raw_image)),
        first_data_sector=np.array(fat["first_data_sector"]),
    )
    print(f"Saved candidate clusters: {output_path}")

    our_top = np.argmax(our_cluster_mean_probs, axis=1)
    print("Our model cluster mean top-label counts:")
    for label_id, count in sorted(zip(*np.unique(our_top, return_counts=True))):
        print(f"  {our_labels[int(label_id)]}: {int(count):,}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and load nps-2009-canon2 raw disk image.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert E01 to raw using ewfexport.")
    convert_parser.add_argument("--generation", type=int, default=GENERATION)
    convert_parser.add_argument("--overwrite", action="store_true")

    info_parser = subparsers.add_parser("info", help="Print raw image information.")
    info_parser.add_argument("--generation", type=int, default=GENERATION)

    sector_parser = subparsers.add_parser("sector", help="Print first bytes of a sector from raw image.")
    sector_parser.add_argument("sector", type=int)
    sector_parser.add_argument("--count", type=int, default=1)
    sector_parser.add_argument("--generation", type=int, default=GENERATION)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Create sector masks for known resident/residual file data from report.xml.",
    )
    preprocess_parser.add_argument("--generation", type=int, default=GENERATION)

    fat_parser = subparsers.add_parser("fat-info", help="Parse and print FAT boot-sector layout.")
    fat_parser.add_argument("--generation", type=int, default=GENERATION)

    cluster_parser = subparsers.add_parser(
        "cluster-map",
        help="Create cluster-level candidate masks using FAT32 cluster size and report.xml.",
    )
    cluster_parser.add_argument("--generation", type=int, default=GENERATION)

    signature_parser = subparsers.add_parser(
        "signature-scan",
        help="Scan raw image for common file signatures and map hits to sectors/clusters.",
    )
    signature_parser.add_argument("--generation", type=int, default=GENERATION)

    candidates_parser = subparsers.add_parser(
        "candidate-clusters",
        help="Extract candidate clusters and run our fusion classifier plus FiFTy when available.",
    )
    candidates_parser.add_argument("--generation", type=int, default=GENERATION)
    candidates_parser.add_argument("--batch-size", type=int, default=512)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "convert":
        path = convert_e01_to_raw(args.generation, overwrite=args.overwrite)
        print_raw_info(path, args.generation)
    elif args.command == "info":
        path = load_raw_image(args.generation)
        print_raw_info(path, args.generation)
    elif args.command == "sector":
        path = load_raw_image(args.generation)
        data = read_sector(path, args.sector, args.count)
        print(f"Read {len(data)} bytes from sector {args.sector}")
        print(data[:64].hex(" "))
    elif args.command == "preprocess":
        preprocess_known_sectors(args.generation)
    elif args.command == "fat-info":
        path = load_raw_image(args.generation)
        print_fat_info(path)
    elif args.command == "cluster-map":
        preprocess_clusters(args.generation)
    elif args.command == "signature-scan":
        scan_signatures(args.generation)
    elif args.command == "candidate-clusters":
        build_candidate_clusters(args.generation, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

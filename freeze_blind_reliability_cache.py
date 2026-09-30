#!/usr/bin/env python3
"""
Freeze blind, non-circular reliability descriptors for the RG-POSR review study.

The script reads the original 10-s window index and sequence caches, computes
eight descriptors per modality directly from the observed standardised
sequence, and fits robust descriptor normalisation using only enrolled
Session-1 training-reference windows.

It never reads the original 25 reliability descriptors or natural reliability
targets. It does not modify the historical experiment.

Expected workspace
------------------
Onyeka_Review_Experiment/
├── 00_original_reference/
│   ├── reliability_window_index_10s_original.csv
│   └── normalization_stats_original.npz
└── 02_reliability_models/
    ├── blind_reliability.py
    └── freeze_blind_reliability_cache.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from blind_reliability import (
    DESCRIPTOR_DIM,
    DESCRIPTOR_NAMES,
    MODALITIES,
    blind_descriptors,
)


VERSION = "1.0.0-review"
EXPECTED_DIMS = {"head": 13, "left": 22, "right": 22, "gaze": 19}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def choose_device(name: str) -> torch.device:
    normalized = str(name).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def load_normalization(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalization archive:\n{path}")

    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}

    with np.load(path, allow_pickle=False) as archive:
        for modality in MODALITIES:
            mean_key = f"{modality}_mean"
            std_key = f"{modality}_std"

            if mean_key not in archive.files or std_key not in archive.files:
                raise KeyError(
                    f"{path} must contain {mean_key!r} and {std_key!r}; "
                    f"available keys are {archive.files}."
                )

            mean = archive[mean_key].astype(np.float32)
            std = archive[std_key].astype(np.float32)
            std = np.where(np.abs(std) > 1e-8, std, 1.0).astype(np.float32)

            expected = EXPECTED_DIMS[modality]
            if mean.shape != (expected,) or std.shape != (expected,):
                raise RuntimeError(
                    f"{modality} normalization shape mismatch: "
                    f"mean={mean.shape}, std={std.shape}, expected={(expected,)}."
                )

            means[modality] = mean
            stds[modality] = std

    return means, stds


def load_index(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing window index:\n{path}")

    frame = pd.read_csv(path)
    frame.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in frame.columns
    ]

    required = {
        "window_id",
        "player_id",
        "role",
        "session_order",
        "split",
        "start_index",
        "stop_index",
        "resolved_cache_path",
        "eligible_primary",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Window index is missing columns: {missing}")

    frame["window_id"] = frame["window_id"].astype(str)
    frame["player_id"] = frame["player_id"].astype(str).str.zfill(2)
    frame["role"] = frame["role"].astype(str).str.strip().str.lower()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()

    for column in ("session_order", "start_index", "stop_index"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)

    frame["eligible_primary"] = parse_bool_series(frame["eligible_primary"])
    frame["resolved_cache_path"] = frame["resolved_cache_path"].astype(str)
    frame["output_row"] = np.arange(len(frame), dtype=np.int64)

    if frame["window_id"].duplicated().any():
        duplicate = frame.loc[frame["window_id"].duplicated(), "window_id"].iloc[0]
        raise RuntimeError(f"Duplicate window_id detected: {duplicate}")

    invalid_bounds = frame["stop_index"] <= frame["start_index"]
    if invalid_bounds.any():
        raise RuntimeError(
            "At least one window has stop_index <= start_index:\n"
            f"{frame.loc[invalid_bounds].head()}"
        )

    missing_caches = sorted(
        {
            path_value
            for path_value in frame["resolved_cache_path"].unique()
            if not Path(path_value).is_file()
        }
    )
    if missing_caches:
        preview = "\n".join(missing_caches[:10])
        raise FileNotFoundError(
            f"{len(missing_caches)} referenced cache files are missing. "
            f"First entries:\n{preview}"
        )

    training_mask = (
        frame["role"].eq("enrolled")
        & frame["session_order"].eq(1)
        & frame["split"].eq("train")
        & frame["eligible_primary"]
    ).to_numpy(dtype=bool)

    if not training_mask.any():
        raise RuntimeError("No enrolled Session-1 eligible training windows found.")

    return frame, training_mask


def robust_normalization(
    raw_features: np.ndarray,
    training_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    training = raw_features[training_mask].astype(np.float64)

    median = np.median(training, axis=0)
    mad = 1.4826 * np.median(np.abs(training - median), axis=0)
    std = np.std(training, axis=0)

    scale = np.where(
        mad > 1e-6,
        mad,
        np.where(std > 1e-6, std, 1.0),
    )

    standardized = (raw_features - median[None, :, :]) / scale[None, :, :]
    standardized = np.nan_to_num(
        standardized,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    )
    standardized = np.clip(standardized, -10.0, 10.0).astype(np.float32)

    return median.astype(np.float32), scale.astype(np.float32), standardized


def compute_all_descriptors(
    frame: pd.DataFrame,
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    output = np.empty(
        (len(frame), len(MODALITIES), DESCRIPTOR_DIM),
        dtype=np.float32,
    )

    grouped = list(frame.groupby("resolved_cache_path", sort=False))
    total_caches = len(grouped)

    for cache_number, (cache_path, rows) in enumerate(grouped, start=1):
        with np.load(cache_path, allow_pickle=False) as archive:
            missing = [modality for modality in MODALITIES if modality not in archive.files]
            if missing:
                raise KeyError(f"{cache_path} is missing modality arrays: {missing}")

            arrays = {
                modality: archive[modality].astype(np.float32)
                for modality in MODALITIES
            }

        # Windows may theoretically have different lengths. Group by length so
        # that each processing batch can be stacked safely.
        rows = rows.copy()
        rows["_length"] = rows["stop_index"] - rows["start_index"]

        for _, same_length in rows.groupby("_length", sort=False):
            records = list(same_length.itertuples(index=False))

            for offset in range(0, len(records), batch_size):
                chunk = records[offset : offset + batch_size]
                output_rows = np.asarray(
                    [int(record.output_row) for record in chunk],
                    dtype=np.int64,
                )

                for modality_index, modality in enumerate(MODALITIES):
                    sequence_batch = np.stack(
                        [
                            (
                                arrays[modality][
                                    int(record.start_index) : int(record.stop_index)
                                ]
                                - means[modality]
                            )
                            / stds[modality]
                            for record in chunk
                        ],
                        axis=0,
                    ).astype(np.float32)

                    sequence_batch = np.nan_to_num(
                        sequence_batch,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                    tensor = torch.from_numpy(sequence_batch).to(device=device)
                    descriptors = blind_descriptors(tensor).cpu().numpy().astype(np.float32)
                    output[output_rows, modality_index, :] = descriptors

        if cache_number == 1 or cache_number % 10 == 0 or cache_number == total_caches:
            print(
                f"Processed cache {cache_number}/{total_caches}: {cache_path}",
                flush=True,
            )

    if not np.isfinite(output).all():
        raise RuntimeError("Non-finite values remain in the blind descriptor cache.")

    return output


def distribution_rows(
    raw_features: np.ndarray,
    training_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for modality_index, modality in enumerate(MODALITIES):
        for descriptor_index, descriptor_name in enumerate(DESCRIPTOR_NAMES):
            for partition, mask in (
                ("all_windows", np.ones(len(raw_features), dtype=bool)),
                ("training_reference", training_mask),
                ("non_training", ~training_mask),
            ):
                values = raw_features[mask, modality_index, descriptor_index].astype(
                    np.float64
                )
                rows.append(
                    {
                        "modality": modality,
                        "descriptor": descriptor_name,
                        "partition": partition,
                        "count": int(len(values)),
                        "minimum": float(np.min(values)),
                        "q01": float(np.quantile(values, 0.01)),
                        "q05": float(np.quantile(values, 0.05)),
                        "q25": float(np.quantile(values, 0.25)),
                        "median": float(np.median(values)),
                        "mean": float(np.mean(values)),
                        "q75": float(np.quantile(values, 0.75)),
                        "q95": float(np.quantile(values, 0.95)),
                        "q99": float(np.quantile(values, 0.99)),
                        "maximum": float(np.max(values)),
                        "standard_deviation": float(np.std(values)),
                    }
                )

    return rows


def build(
    index_path: Path,
    normalization_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "blind_reliability_window_features_10s.npz"
    normalization_output = output_dir / "blind_reliability_normalization.json"
    distribution_output = output_dir / "blind_reliability_descriptor_summary.csv"
    report_path = output_dir / "blind_reliability_freeze_report.json"

    outputs = (
        cache_path,
        normalization_output,
        distribution_output,
        report_path,
    )

    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing review outputs. "
            "Use --overwrite only after archiving the previous run:\n"
            f"{joined}"
        )

    frame, training_mask = load_index(index_path)
    means, stds = load_normalization(normalization_path)

    raw_features = compute_all_descriptors(
        frame=frame,
        means=means,
        stds=stds,
        device=device,
        batch_size=batch_size,
    )

    median, scale, standardized = robust_normalization(
        raw_features,
        training_mask,
    )

    temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary_cache,
        raw_features=raw_features,
        standardized_features=standardized,
        training_reference_mask=training_mask.astype(np.uint8),
        descriptor_names=np.asarray(DESCRIPTOR_NAMES, dtype="U64"),
        modality_names=np.asarray(MODALITIES, dtype="U16"),
        window_ids=frame["window_id"].astype(str).to_numpy(dtype="U64"),
    )
    os.replace(temporary_cache, cache_path)

    normalization_payload = {
        "version": VERSION,
        "status": "FROZEN",
        "method": "median_and_scaled_MAD_with_standard_deviation_fallback",
        "fit_partition": (
            "role=enrolled AND session_order=1 AND split=train "
            "AND eligible_primary=true"
        ),
        "modalities": list(MODALITIES),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "median": median.tolist(),
        "scale": scale.tolist(),
        "clip_after_standardization": [-10.0, 10.0],
        "target_included": False,
        "corruption_type_or_severity_included": False,
    }
    save_json(normalization_output, normalization_payload)

    summary = pd.DataFrame(distribution_rows(raw_features, training_mask))
    summary.to_csv(distribution_output, index=False)

    report = {
        "version": VERSION,
        "status": "COMPLETED",
        "purpose": "Freeze blind, non-circular clean-sequence reliability descriptors",
        "index_path": str(index_path),
        "index_sha256": sha256(index_path),
        "normalization_input_path": str(normalization_path),
        "normalization_input_sha256": sha256(normalization_path),
        "output_dir": str(output_dir),
        "windows": int(len(frame)),
        "participants": int(frame["player_id"].nunique()),
        "unique_cache_files": int(frame["resolved_cache_path"].nunique()),
        "training_reference_windows": int(training_mask.sum()),
        "non_training_windows": int((~training_mask).sum()),
        "modalities": list(MODALITIES),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "descriptor_shape": list(raw_features.shape),
        "normalization_method": normalization_payload["method"],
        "old_25_descriptor_cache_read": False,
        "old_natural_targets_read": False,
        "target_stored_in_cache": False,
        "corruption_type_or_severity_stored_in_cache": False,
        "existing_processed_dataset_modified": False,
        "elapsed_minutes": float((time.time() - started) / 60.0),
        "generated_files": [
            str(cache_path),
            str(normalization_output),
            str(distribution_output),
        ],
    }
    save_json(report_path, report)

    print()
    print("=" * 76)
    print("BLIND RELIABILITY DESCRIPTOR FREEZE COMPLETED")
    print("=" * 76)
    print(f"Windows: {len(frame)}")
    print(f"Participants: {frame['player_id'].nunique()}")
    print(f"Training-reference windows: {training_mask.sum()}")
    print(f"Descriptor shape: {raw_features.shape}")
    print(f"Device: {device}")
    print(f"Report: {report_path}")

    return report


def synthetic_workspace(root: Path) -> tuple[Path, Path]:
    reference = root / "00_original_reference"
    reference.mkdir(parents=True)

    cache_path = root / "synthetic_cache.npz"
    rng = np.random.default_rng(42)
    arrays = {
        "head": rng.normal(size=(600, 13)).astype(np.float32),
        "left": rng.normal(size=(600, 22)).astype(np.float32),
        "right": rng.normal(size=(600, 22)).astype(np.float32),
        "gaze": rng.normal(size=(600, 19)).astype(np.float32),
    }
    arrays["head"][300:450] = 0.0
    np.savez_compressed(cache_path, **arrays)

    rows = []
    for index, start in enumerate((0, 150, 300, 450)):
        rows.append(
            {
                "window_id": f"w{index}",
                "player_id": "01" if index < 2 else "02",
                "role": "enrolled",
                "session_order": 1 if index < 3 else 2,
                "split": "train" if index < 3 else "test",
                "start_index": start,
                "stop_index": start + 150,
                "resolved_cache_path": str(cache_path),
                "eligible_primary": True,
            }
        )

    index_path = reference / "reliability_window_index_10s_original.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)

    stats = {}
    for modality, dimension in EXPECTED_DIMS.items():
        stats[f"{modality}_mean"] = np.zeros(dimension, dtype=np.float32)
        stats[f"{modality}_std"] = np.ones(dimension, dtype=np.float32)
    normalization_path = reference / "normalization_stats_original.npz"
    np.savez_compressed(normalization_path, **stats)

    return index_path, normalization_path


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="blind_reliability_freeze_test_") as folder:
        root = Path(folder)
        index_path, normalization_path = synthetic_workspace(root)
        output_dir = root / "02_reliability_models" / "review_blind_cache"

        report = build(
            index_path=index_path,
            normalization_path=normalization_path,
            output_dir=output_dir,
            device=torch.device("cpu"),
            batch_size=2,
            overwrite=False,
        )

        cache_path = output_dir / "blind_reliability_window_features_10s.npz"
        with np.load(cache_path, allow_pickle=False) as archive:
            raw = archive["raw_features"]
            standardized = archive["standardized_features"]
            training_mask = archive["training_reference_mask"].astype(bool)
            assert raw.shape == (4, 4, DESCRIPTOR_DIM)
            assert standardized.shape == raw.shape
            assert training_mask.tolist() == [True, True, True, False]
            assert np.isfinite(raw).all()
            assert np.isfinite(standardized).all()
            # Synthetic window 2 has complete head dropout.
            assert raw[2, 0, 0] > 0.99

        assert report["old_25_descriptor_cache_read"] is False
        assert report["old_natural_targets_read"] is False
        assert report["target_stored_in_cache"] is False

    print("=" * 76)
    print("FREEZE BLIND RELIABILITY CACHE SELF-TEST PASSED")
    print("=" * 76)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    experiment_root = script_dir.parent
    reference_dir = experiment_root / "00_original_reference"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=reference_dir / "reliability_window_index_10s_original.csv",
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=reference_dir / "normalization_stats_original.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "review_blind_cache",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    build(
        index_path=args.index.resolve(),
        normalization_path=args.normalization.resolve(),
        output_dir=args.output.resolve(),
        device=choose_device(args.device),
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

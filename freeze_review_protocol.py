#!/usr/bin/env python3
"""
Freeze the RG-POSR protocol and build explicit window-level reliability vectors.

This program never modifies the existing Who Is Alyx processed dataset.

It:
1. validates the exact 13/22/22/19 identity schema and normalization;
2. verifies the frozen 49-enrolled/20-unknown cross-session protocol;
3. resolves stale F: cache paths after a dataset copy to G:;
4. fits kinematic/disagreement thresholds using enrolled Session-1 training
   windows only;
5. creates explicit 6/6/6/7 reliability vectors for every 10-second window;
6. standardizes reliability descriptors using training-only robust statistics;
7. writes checksums, frozen equations, corruption/evaluation plans, and
   machine-readable caches for the prototype baseline and RG-POSR.

No model is trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

VERSION = "1.0.0"
MODALITIES = ("head", "left", "right", "gaze")
EXPECTED_DIMS = {"head": 13, "left": 22, "right": 22, "gaze": 19}

MOTION_RELIABILITY_NAMES = (
    "availability_fraction",
    "mean_source_quality",
    "low_quality_fraction",
    "constant_pose_fraction",
    "kinematic_jitter_median",
    "kinematic_outlier_fraction",
)
GAZE_RELIABILITY_NAMES = (
    "availability_fraction",
    "mean_source_quality",
    "low_quality_fraction",
    "low_eye_openness_fraction",
    "invalid_pupil_fraction",
    "gaze_jitter_median",
    "binocular_disagreement_outlier_fraction",
)

TRAINING_QUANTILE = 0.99
LOW_QUALITY_THRESHOLD = 0.50
LOW_OPENNESS_THRESHOLD = 0.10
CONSTANT_POSE_DELTA = 1e-3
ROBUST_SCALE_FLOOR = 1e-6


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_simple_yaml(path: Path, value: Any) -> None:
    def scalar(item: Any) -> str:
        if item is None:
            return "null"
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, (int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                return json.dumps(str(item))
            return repr(item)
        text = str(item)
        if (
            text == ""
            or text.lower() in {"true", "false", "null"}
            or any(character in text for character in ":#{}[],'\"\n\t")
        ):
            return json.dumps(text)
        return text

    def emit(item: Any, indentation: int = 0) -> list[str]:
        prefix = " " * indentation
        lines: list[str] = []
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(child, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(emit(child, indentation + 2))
                else:
                    lines.append(f"{prefix}{key}: {scalar(child)}")
        elif isinstance(item, list):
            for child in item:
                if isinstance(child, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.extend(emit(child, indentation + 2))
                else:
                    lines.append(f"{prefix}- {scalar(child)}")
        else:
            lines.append(f"{prefix}{scalar(item)}")
        return lines

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(emit(value)) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_name(value: str) -> str:
    return (
        str(value).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def find_file(root: Path, preferred: Sequence[str], pattern: str) -> Path:
    for name in preferred:
        candidate = root / name
        if candidate.is_file():
            return candidate
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} below {root}")
    return matches[0]


def load_schema_and_stats(processed_root: Path) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    schema_path = find_file(processed_root, ("feature_schema.json",), "*feature*schema*.json")
    stats_path = find_file(
        processed_root,
        ("normalization_stats.json",),
        "*normalization*stats*.json",
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return schema_path, schema, stats_path, stats


def validate_identity_schema(
    schema: dict[str, Any],
    stats: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []

    for modality in MODALITIES:
        names = list(schema.get(modality, []))
        stat_block = stats.get(modality, {})
        stat_names = list(stat_block.get("feature_names", []))
        means = list(stat_block.get("mean", []))
        stds = list(stat_block.get("std", []))
        counts = list(stat_block.get("count", []))

        expected = EXPECTED_DIMS[modality]
        lengths = {
            "schema": len(names),
            "stats_names": len(stat_names),
            "mean": len(means),
            "std": len(stds),
            "count": len(counts),
        }
        if any(length != expected for length in lengths.values()):
            errors.append(f"{modality}: expected {expected}, observed {lengths}")
        if names != stat_names:
            errors.append(f"{modality}: schema names do not match normalization names")

        for index, name in enumerate(names):
            mean = float(means[index]) if index < len(means) else float("nan")
            std = float(stds[index]) if index < len(stds) else float("nan")
            count = int(counts[index]) if index < len(counts) else 0
            rows.append(
                {
                    "modality": modality,
                    "feature_index": index,
                    "feature_name": name,
                    "dimension": expected,
                    "mean": mean,
                    "std": std,
                    "count": count,
                    "near_constant": bool(math.isfinite(std) and std <= 1e-5),
                    "units": infer_units(name),
                    "feature_group": infer_group(name),
                    "normalization": "z_score_using_enrolled_session1_training_statistics",
                }
            )
    return pd.DataFrame(rows), errors


def infer_units(name: str) -> str:
    if name.endswith("_m"):
        return "metres"
    if name.endswith("_mps"):
        return "metres_per_second"
    if name.endswith("_rps"):
        return "radians_per_second"
    if name.endswith("_rad"):
        return "radians"
    if name.endswith("_mm"):
        return "millimetres"
    if "button" in name or "pressed" in name or "touched" in name:
        return "binary"
    if "trackpad_" in name or "pupil_sensor_" in name:
        return "normalized_device_coordinate"
    if "rot_" in name or "gaze_" in name:
        return "unitless"
    if "openness" in name or "trigger" in name:
        return "bounded_unitless"
    return "unitless"


def infer_group(name: str) -> str:
    if "rel_pos" in name or "local_pos" in name:
        return "position"
    if "rel_rot" in name:
        return "quaternion"
    if "ang_vel" in name:
        return "angular_velocity"
    if "_vel_" in name:
        return "linear_velocity"
    if "gaze_combined" in name:
        return "combined_gaze_direction"
    if "gaze_left" in name or "gaze_right" in name:
        return "monocular_gaze_direction"
    if "openness" in name:
        return "eye_openness"
    if "pupil_diameter" in name:
        return "pupil_diameter"
    if "pupil_sensor" in name:
        return "pupil_sensor_position"
    if "binocular_disagreement" in name:
        return "gaze_disagreement"
    if "angular_speed" in name:
        return "gaze_angular_speed"
    return "controller_interaction"


def resolve_cache_path(original: str, processed_root: Path) -> Path:
    candidate = Path(str(original))
    if candidate.is_file():
        return candidate
    remapped = processed_root / "resampled_sessions" / candidate.name
    if remapped.is_file():
        return remapped
    matches = list(processed_root.rglob(candidate.name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Cache not found. Original={original}; remapped={remapped}"
        )
    raise RuntimeError(f"Ambiguous cache filename {candidate.name}: {matches[:3]}")


def load_manifest(processed_root: Path, duration: int) -> tuple[Path, pd.DataFrame]:
    path = find_file(
        processed_root,
        ("window_manifest.csv",),
        "*window*manifest*.csv",
    )
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [normalized_name(column) for column in frame.columns]
    required = {
        "window_id",
        "player_id",
        "role",
        "session_id",
        "session_order",
        "split",
        "duration_seconds",
        "start_index",
        "stop_index",
        "cache_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Window manifest missing columns: {missing}")

    frame["player_id"] = frame["player_id"].astype(str).str.zfill(2)
    frame["duration_seconds"] = pd.to_numeric(
        frame["duration_seconds"], errors="coerce"
    )
    frame["session_order"] = pd.to_numeric(frame["session_order"], errors="coerce")
    frame["start_index"] = pd.to_numeric(frame["start_index"], errors="raise").astype(int)
    frame["stop_index"] = pd.to_numeric(frame["stop_index"], errors="raise").astype(int)

    frame = frame[frame["duration_seconds"] == float(duration)].copy()
    if frame.empty:
        raise RuntimeError(f"No {duration}-second windows in {path}")

    for column in ("eligible_primary", "eligible_motion", "eligible_gaze"):
        if column in frame.columns:
            frame[column] = parse_bool_series(frame[column])
        else:
            frame[column] = True

    frame["role"] = frame["role"].astype(str).str.strip().str.lower()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame["resolved_cache_path"] = [
        str(resolve_cache_path(value, processed_root))
        for value in frame["cache_path"]
    ]
    frame = frame.sort_values(
        ["player_id", "session_order", "session_id", "start_index", "window_id"]
    ).reset_index(drop=True)
    return path, frame


def load_split_manifest(processed_root: Path) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    path = find_file(
        processed_root,
        ("participant_split_manifest.csv",),
        "*participant*split*.csv",
    )
    frame = pd.read_csv(path, dtype=str)
    frame.columns = [normalized_name(column) for column in frame.columns]
    required = {"player_id", "role", "development_session", "test_session"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Participant split missing columns: {missing}")

    frame["player_id"] = frame["player_id"].astype(str).str.zfill(2)
    frame["role"] = frame["role"].astype(str).str.lower()
    role_counts = frame["role"].value_counts().to_dict()
    enrolled = int(role_counts.get("enrolled", 0))
    unknown = int(role_counts.get("unknown", role_counts.get("final_unknown", 0)))
    summary = {
        "participants": int(frame["player_id"].nunique()),
        "enrolled": enrolled,
        "final_unknown": unknown,
        "role_counts": role_counts,
    }
    if summary["participants"] != 69 or enrolled != 49 or unknown != 20:
        raise RuntimeError(
            "Frozen participant split mismatch; expected 69 total, "
            f"49 enrolled, 20 unknown; observed {summary}"
        )
    return path, frame, summary


def read_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"head", "left", "right", "gaze", "quality", "time_seconds"}
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"{path} missing NPZ arrays: {sorted(missing)}")
        return {
            "head": archive["head"].astype(np.float32),
            "left": archive["left"].astype(np.float32),
            "right": archive["right"].astype(np.float32),
            "gaze": archive["gaze"].astype(np.float32),
            "quality": archive["quality"].astype(np.float32),
            "time_seconds": archive["time_seconds"].astype(np.float64),
        }


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25.0, axis=0)
    q75 = np.nanpercentile(values, 75.0, axis=0)
    scale = (q75 - q25) / 1.349
    standard = np.nanstd(values, axis=0)
    scale = np.where(scale > ROBUST_SCALE_FLOOR, scale, standard)
    scale = np.where(scale > ROBUST_SCALE_FLOOR, scale, 1.0)
    return median.astype(np.float32), scale.astype(np.float32)


def modality_stats_arrays(
    stats: dict[str, Any],
    modality: str,
) -> tuple[np.ndarray, np.ndarray]:
    block = stats[modality]
    mean = np.asarray(block["mean"], dtype=np.float32)
    std = np.asarray(block["std"], dtype=np.float32)
    std = np.where(np.abs(std) > 1e-8, std, 1.0).astype(np.float32)
    return mean, std


def frame_kinematic_score(values: np.ndarray, std: np.ndarray) -> np.ndarray:
    kinematics = values[:, 7:13]
    scales = std[7:13]
    standardized = np.divide(
        kinematics,
        scales,
        out=np.zeros_like(kinematics, dtype=np.float32),
        where=np.abs(scales) > 1e-8,
    )
    return np.linalg.norm(standardized, axis=1)


def motion_descriptors(
    values: np.ndarray,
    quality: np.ndarray,
    std: np.ndarray,
    outlier_threshold: float,
) -> np.ndarray:
    finite_quality = np.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
    available = finite_quality > 0.0

    pose = values[:, :7]
    pose_scale = std[:7]
    pose_standardized = np.divide(
        pose,
        pose_scale,
        out=np.zeros_like(pose, dtype=np.float32),
        where=np.abs(pose_scale) > 1e-8,
    )
    if len(pose_standardized) > 1:
        delta = np.max(np.abs(np.diff(pose_standardized, axis=0)), axis=1)
        constant_fraction = float(np.mean(delta < CONSTANT_POSE_DELTA))
    else:
        constant_fraction = 1.0

    kinematics = values[:, 7:13]
    kin_scale = std[7:13]
    kin_standardized = np.divide(
        kinematics,
        kin_scale,
        out=np.zeros_like(kinematics, dtype=np.float32),
        where=np.abs(kin_scale) > 1e-8,
    )
    if len(kin_standardized) > 1:
        jitter = float(np.median(np.linalg.norm(np.diff(kin_standardized, axis=0), axis=1)))
    else:
        jitter = 0.0

    scores = np.linalg.norm(kin_standardized, axis=1)
    valid_scores = scores[available & np.isfinite(scores)]
    outlier_fraction = (
        float(np.mean(valid_scores > outlier_threshold))
        if valid_scores.size
        else 1.0
    )
    return np.asarray(
        [
            float(np.mean(available)),
            float(np.mean(finite_quality)),
            float(np.mean(finite_quality < LOW_QUALITY_THRESHOLD)),
            constant_fraction,
            jitter,
            outlier_fraction,
        ],
        dtype=np.float32,
    )


def gaze_descriptors(
    values: np.ndarray,
    quality: np.ndarray,
    std: np.ndarray,
    disagreement_threshold: float,
) -> np.ndarray:
    finite_quality = np.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
    available = finite_quality > 0.0

    openness = values[:, 9:11]
    pupil = values[:, 11:13]
    low_openness = (
        ~np.isfinite(openness).all(axis=1)
        | (openness <= LOW_OPENNESS_THRESHOLD).any(axis=1)
    )
    invalid_pupil = (
        ~np.isfinite(pupil).all(axis=1)
        | (pupil <= 0.0).any(axis=1)
    )

    angular_speed = values[:, 18]
    angular_scale = float(std[18]) if abs(float(std[18])) > 1e-8 else 1.0
    angular_standardized = angular_speed / angular_scale
    if len(angular_standardized) > 1:
        jitter = float(np.median(np.abs(np.diff(angular_standardized))))
    else:
        jitter = 0.0

    disagreement = values[:, 17]
    valid_disagreement = disagreement[available & np.isfinite(disagreement)]
    disagreement_outlier_fraction = (
        float(np.mean(valid_disagreement > disagreement_threshold))
        if valid_disagreement.size
        else 1.0
    )

    return np.asarray(
        [
            float(np.mean(available)),
            float(np.mean(finite_quality)),
            float(np.mean(finite_quality < LOW_QUALITY_THRESHOLD)),
            float(np.mean(low_openness)),
            float(np.mean(invalid_pupil)),
            jitter,
            disagreement_outlier_fraction,
        ],
        dtype=np.float32,
    )


def is_training_reference(frame: pd.DataFrame) -> pd.Series:
    mask = (
        frame["role"].eq("enrolled")
        & frame["session_order"].eq(1)
        & frame["split"].eq("train")
        & frame["eligible_primary"].astype(bool)
    )
    return mask


def fit_frame_thresholds(
    frame: pd.DataFrame,
    stats: dict[str, Any],
) -> dict[str, float]:
    training = frame[is_training_reference(frame)]
    if training.empty:
        raise RuntimeError("No enrolled Session-1 eligible training windows found.")

    motion_scores: dict[str, list[np.ndarray]] = {
        "head": [],
        "left": [],
        "right": [],
    }
    gaze_disagreement: list[np.ndarray] = []

    grouped = training.groupby("resolved_cache_path", sort=False)
    for number, (cache_path, rows) in enumerate(grouped, start=1):
        arrays = read_cache(cache_path)
        for row in rows.itertuples(index=False):
            start = int(row.start_index)
            stop = int(row.stop_index)
            quality = arrays["quality"][start:stop]
            for modality, quality_index in (("head", 0), ("left", 1), ("right", 2)):
                _, std = modality_stats_arrays(stats, modality)
                scores = frame_kinematic_score(arrays[modality][start:stop], std)
                valid = quality[:, quality_index] > 0.0
                selected = scores[valid & np.isfinite(scores)]
                if selected.size:
                    motion_scores[modality].append(selected.astype(np.float32))
            disagreement = arrays["gaze"][start:stop, 17]
            valid_gaze = quality[:, 3] > 0.0
            selected = disagreement[valid_gaze & np.isfinite(disagreement)]
            if selected.size:
                gaze_disagreement.append(selected.astype(np.float32))
        if number % 10 == 0:
            print(f"Threshold fitting: processed {number}/{len(grouped)} caches", flush=True)

    thresholds: dict[str, float] = {}
    for modality, pieces in motion_scores.items():
        if not pieces:
            raise RuntimeError(f"No training kinematic scores for {modality}")
        values = np.concatenate(pieces)
        thresholds[f"{modality}_kinematic_outlier_threshold"] = float(
            np.quantile(values, TRAINING_QUANTILE)
        )
    if not gaze_disagreement:
        raise RuntimeError("No training gaze-disagreement values.")
    thresholds["gaze_disagreement_outlier_threshold"] = float(
        np.quantile(np.concatenate(gaze_disagreement), TRAINING_QUANTILE)
    )
    thresholds["training_quantile"] = TRAINING_QUANTILE
    thresholds["low_quality_threshold"] = LOW_QUALITY_THRESHOLD
    thresholds["low_openness_threshold"] = LOW_OPENNESS_THRESHOLD
    thresholds["constant_pose_delta_standardized"] = CONSTANT_POSE_DELTA
    thresholds["fit_window_count"] = int(len(training))
    return thresholds


def reliability_feature_names() -> list[str]:
    names: list[str] = []
    for modality in ("head", "left", "right"):
        names.extend(f"{modality}__{name}" for name in MOTION_RELIABILITY_NAMES)
    names.extend(f"gaze__{name}" for name in GAZE_RELIABILITY_NAMES)
    return names


def compute_all_window_features(
    frame: pd.DataFrame,
    stats: dict[str, Any],
    thresholds: dict[str, float],
) -> tuple[np.ndarray, pd.DataFrame]:
    feature_names = reliability_feature_names()
    matrix = np.empty((len(frame), len(feature_names)), dtype=np.float32)
    index_rows: list[dict[str, Any]] = []

    row_position = {index: position for position, index in enumerate(frame.index)}
    grouped = frame.groupby("resolved_cache_path", sort=False)

    for cache_number, (cache_path, rows) in enumerate(grouped, start=1):
        arrays = read_cache(cache_path)
        for row in rows.itertuples():
            start = int(row.start_index)
            stop = int(row.stop_index)
            quality = arrays["quality"][start:stop]
            pieces: list[np.ndarray] = []
            for modality, quality_index in (("head", 0), ("left", 1), ("right", 2)):
                _, std = modality_stats_arrays(stats, modality)
                pieces.append(
                    motion_descriptors(
                        arrays[modality][start:stop],
                        quality[:, quality_index],
                        std,
                        thresholds[f"{modality}_kinematic_outlier_threshold"],
                    )
                )
            _, gaze_std = modality_stats_arrays(stats, "gaze")
            pieces.append(
                gaze_descriptors(
                    arrays["gaze"][start:stop],
                    quality[:, 3],
                    gaze_std,
                    thresholds["gaze_disagreement_outlier_threshold"],
                )
            )
            vector = np.concatenate(pieces).astype(np.float32)
            position = row_position[row.Index]
            matrix[position] = vector
            index_rows.append(
                {
                    "row_position": position,
                    "window_id": str(row.window_id),
                    "player_id": str(row.player_id),
                    "role": str(row.role),
                    "session_id": str(row.session_id),
                    "session_order": int(row.session_order),
                    "split": str(row.split),
                    "calibration_fold": getattr(row, "calibration_fold", -1),
                    "duration_seconds": float(row.duration_seconds),
                    "start_index": int(row.start_index),
                    "stop_index": int(row.stop_index),
                    "eligible_primary": bool(row.eligible_primary),
                    "eligible_motion": bool(row.eligible_motion),
                    "eligible_gaze": bool(row.eligible_gaze),
                    "resolved_cache_path": str(row.resolved_cache_path),
                }
            )
        if cache_number % 10 == 0 or cache_number == len(grouped):
            print(
                f"Reliability features: processed {cache_number}/{len(grouped)} caches",
                flush=True,
            )

    index_frame = pd.DataFrame(index_rows).sort_values("row_position").reset_index(drop=True)
    if index_frame["row_position"].tolist() != list(range(len(frame))):
        raise RuntimeError("Reliability index alignment failed.")
    return matrix, index_frame


def fit_descriptor_normalization(
    frame: pd.DataFrame,
    raw_matrix: np.ndarray,
) -> dict[str, Any]:
    training_mask = is_training_reference(frame).to_numpy()
    training_values = raw_matrix[training_mask]
    if len(training_values) == 0:
        raise RuntimeError("No training values for descriptor normalization.")
    median, scale = robust_center_scale(training_values)
    return {
        "method": "training_only_median_and_iqr_over_1.349",
        "feature_names": reliability_feature_names(),
        "median": median.tolist(),
        "scale": scale.tolist(),
        "training_window_count": int(len(training_values)),
    }


def standardize_descriptors(
    raw_matrix: np.ndarray,
    normalization: dict[str, Any],
) -> np.ndarray:
    median = np.asarray(normalization["median"], dtype=np.float32)
    scale = np.asarray(normalization["scale"], dtype=np.float32)
    standardized = (raw_matrix - median) / scale
    return np.nan_to_num(
        standardized,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    ).astype(np.float32)


def quality_feature_spec() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = {
        "availability_fraction": (
            "fraction of frames with source quality greater than zero",
            "higher_is_better",
            "[0,1]",
        ),
        "mean_source_quality": (
            "mean frozen pipeline quality over the window",
            "higher_is_better",
            "[0,1]",
        ),
        "low_quality_fraction": (
            f"fraction of frames with quality below {LOW_QUALITY_THRESHOLD}",
            "lower_is_better",
            "[0,1]",
        ),
        "constant_pose_fraction": (
            "fraction of adjacent standardized pose frames whose maximum absolute change "
            f"is below {CONSTANT_POSE_DELTA}",
            "lower_is_better",
            "[0,1]",
        ),
        "kinematic_jitter_median": (
            "median L2 change of training-standardized linear/angular velocity channels",
            "lower_is_better",
            "robust_continuous",
        ),
        "kinematic_outlier_fraction": (
            f"fraction of valid frames above the enrolled-training {TRAINING_QUANTILE:.2f} "
            "kinematic-score quantile",
            "lower_is_better",
            "[0,1]",
        ),
        "low_eye_openness_fraction": (
            f"fraction of frames where either eye openness is <= {LOW_OPENNESS_THRESHOLD}",
            "lower_is_better",
            "[0,1]",
        ),
        "invalid_pupil_fraction": (
            "fraction of frames with non-finite or non-positive pupil diameter",
            "lower_is_better",
            "[0,1]",
        ),
        "gaze_jitter_median": (
            "median absolute change in training-standardized gaze angular speed",
            "lower_is_better",
            "robust_continuous",
        ),
        "binocular_disagreement_outlier_fraction": (
            f"fraction of valid frames above the enrolled-training {TRAINING_QUANTILE:.2f} "
            "binocular-disagreement quantile",
            "lower_is_better",
            "[0,1]",
        ),
    }
    for modality in ("head", "left", "right"):
        for name in MOTION_RELIABILITY_NAMES:
            definition, direction, units = definitions[name]
            rows.append(
                {
                    "modality": modality,
                    "feature_name": name,
                    "definition": definition,
                    "direction": direction,
                    "units": units,
                    "computed_from": "processed_session_cache_before_model_normalization",
                    "test_time_available": True,
                }
            )
    for name in GAZE_RELIABILITY_NAMES:
        definition, direction, units = definitions[name]
        rows.append(
            {
                "modality": "gaze",
                "feature_name": name,
                "definition": definition,
                "direction": direction,
                "units": units,
                "computed_from": "processed_session_cache_before_model_normalization",
                "test_time_available": True,
            }
        )
    return pd.DataFrame(rows)


def protocol_payload(
    split_summary: dict[str, Any],
    manifest: pd.DataFrame,
    source_hashes: dict[str, str],
    thresholds: dict[str, Any],
    descriptor_normalization: dict[str, Any],
) -> dict[str, Any]:
    split_counts = manifest["split"].value_counts().to_dict()
    return {
        "protocol_version": "RG_POSR_FROZEN_V1",
        "status": "FROZEN",
        "method": "Reliability-Gated Prototypical Open-Set Recognition",
        "acronym": "RG-POSR",
        "participant_protocol": {
            **split_summary,
            "development_session": "earliest usable session",
            "test_session": "latest usable session",
            "final_unknown_used_for_training": False,
            "final_unknown_used_for_preprocessing_fit": False,
            "final_unknown_used_for_threshold_selection": False,
            "pseudo_unknown_calibration_folds": 5,
        },
        "sampling_and_windows": {
            "sample_rate_hz": 15,
            "identity_window_seconds": 10,
            "identity_window_frames": 150,
            "aggregation_seconds": [30, 60, 120],
            "manifest_split_counts_for_10s": split_counts,
        },
        "identity_schema": {
            "dimensions": EXPECTED_DIMS,
            "normalization": "frozen enrolled Session-1 training mean/std",
            "quaternion_treatment": (
                "raw quaternion unit normalization, relative quaternion construction, "
                "unit normalization, then component-wise frozen z-score"
            ),
            "constant_channels_retained_for_baseline_comparability": [
                "left_ul_button_pressed",
                "left_ul_button_touched",
                "right_ul_button_pressed",
                "right_ul_button_touched",
            ],
        },
        "reliability_schema": {
            "dimensions": {"head": 6, "left": 6, "right": 6, "gaze": 7},
            "total_dimension": 25,
            "synthetic_corruption_severity_is_model_input": False,
            "quality_features_available_at_test_time": True,
            "descriptor_normalization": descriptor_normalization["method"],
            "thresholds_fit_using": (
                "enrolled Session-1 eligible 10-second training windows only"
            ),
        },
        "prototype_decision": {
            "modality_embedding": "L2-normalized",
            "fusion": "reliability-weighted sum followed by L2 normalization",
            "prototype": "one trainable L2-normalized prototype per enrolled identity",
            "distance": "cosine distance 1-z_dot_p",
            "effective_reliability": "sum(alpha_m * r_m)",
            "scaled_distance": "d / max(rho_eff, epsilon)^gamma",
            "gamma_candidates": [0.0, 0.5, 1.0, 2.0],
            "gamma_selection": "pseudo-unknown calibration only",
            "thresholds": "one frozen threshold per aggregation duration",
        },
        "training_objective": {
            "classification": "angular-margin classification over prototypes",
            "reliability": "Huber reliability-consistency loss",
            "monotonic": "paired ranking across corruption severities",
            "clean_severity": 0.0,
            "severity_target": "1-severity",
        },
        "source_sha256": source_hashes,
        "fitted_frame_thresholds": thresholds,
    }


def method_markdown() -> str:
    return r"""# Frozen RG-POSR method equations

For modality \(m\):

\[
z_m = \frac{f_m(x_m)}{\|f_m(x_m)\|_2},
\qquad
r_m = \sigma(g_m(q_m)).
\]

The explicit quality vector \(q_m\) is drawn only from the frozen reliability
cache or recomputed from synthetically corrupted inputs. Corruption severity is
a supervision label and is never supplied to the reliability head.

\[
\alpha_m = \frac{r_m}{\sum_j r_j+\epsilon},
\qquad
z = \operatorname{norm}\left(\sum_m \alpha_m z_m\right).
\]

Each enrolled identity has one unit-normalized trainable prototype \(p_u\):

\[
d_u = 1-z^\top p_u.
\]

Effective reliability and reliability-scaled distance are:

\[
\rho_{\mathrm{eff}} = \sum_m \alpha_m r_m,
\qquad
d'_u = \frac{d_u}{\max(\rho_{\mathrm{eff}},\epsilon)^\gamma}.
\]

The nearest identity is accepted only when the minimum scaled distance is below
the aggregation-duration-specific threshold selected using Session-1
validation and pseudo-unknown calibration folds. Session-2 known users and the
20 final unknown users are never used for model, gamma, or threshold selection.
"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    processed_root = Path(args.processed_root).expanduser().resolve()
    audit_root = Path(args.audit_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    schema_path, schema, stats_path, stats = load_schema_and_stats(processed_root)
    identity_table, schema_errors = validate_identity_schema(schema, stats)
    if schema_errors:
        raise RuntimeError("Identity schema validation failed: " + "; ".join(schema_errors))

    split_path, split_frame, split_summary = load_split_manifest(processed_root)
    manifest_path, manifest = load_manifest(processed_root, args.duration)

    unknown_bad = manifest[
        manifest["role"].isin({"unknown", "final_unknown"})
        & manifest["split"].isin({"train", "validation"})
    ]
    if not unknown_bad.empty:
        raise RuntimeError("Final unknown users appear in train/validation manifest rows.")

    audit_report_path = audit_root / "audit_report.json"
    if not audit_report_path.is_file():
        raise FileNotFoundError(f"Missing prior audit report: {audit_report_path}")
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    if audit_report.get("unresolved_items"):
        raise RuntimeError(
            f"Prior audit contains unresolved items: {audit_report['unresolved_items']}"
        )

    source_hashes = {
        "feature_schema.json": sha256(schema_path),
        "normalization_stats.json": sha256(stats_path),
        "participant_split_manifest.csv": sha256(split_path),
        "window_manifest.csv": sha256(manifest_path),
        "prior_audit_report.json": sha256(audit_report_path),
    }

    thresholds = fit_frame_thresholds(manifest, stats)
    raw_matrix, index_frame = compute_all_window_features(manifest, stats, thresholds)
    descriptor_normalization = fit_descriptor_normalization(manifest, raw_matrix)
    standardized_matrix = standardize_descriptors(raw_matrix, descriptor_normalization)

    feature_names = reliability_feature_names()
    training_mask = is_training_reference(manifest).to_numpy()
    natural_quality_targets = np.column_stack(
        [
            raw_matrix[:, feature_names.index(f"{modality}__mean_source_quality")]
            for modality in MODALITIES
        ]
    ).astype(np.float32)

    protocol_dir = output / "00_protocol"
    feature_dir = output / "01_features"
    method_dir = output / "02_method"
    experiment_dir = output / "03_experiments"
    cache_dir = output / "04_cache"

    save_csv(feature_dir / "identity_feature_table_frozen_v1.csv", identity_table)
    reliability_spec = quality_feature_spec()
    save_csv(feature_dir / "reliability_feature_table_frozen_v1.csv", reliability_spec)
    save_json(feature_dir / "reliability_frame_thresholds_v1.json", thresholds)
    save_json(
        feature_dir / "reliability_descriptor_normalization_v1.json",
        descriptor_normalization,
    )
    save_json(feature_dir / "source_checksums_v1.json", source_hashes)

    save_csv(cache_dir / "reliability_window_index_10s.csv", index_frame)
    np.savez_compressed(
        cache_dir / "reliability_window_features_10s.npz",
        raw_features=raw_matrix,
        standardized_features=standardized_matrix,
        natural_quality_targets=natural_quality_targets,
        training_reference_mask=training_mask.astype(np.uint8),
        feature_names=np.asarray(feature_names, dtype="U96"),
        modality_names=np.asarray(MODALITIES, dtype="U16"),
        window_ids=index_frame["window_id"].astype(str).to_numpy(dtype="U64"),
    )

    protocol = protocol_payload(
        split_summary,
        manifest,
        source_hashes,
        thresholds,
        descriptor_normalization,
    )
    save_json(protocol_dir / "frozen_protocol_v1.json", protocol)
    write_simple_yaml(protocol_dir / "frozen_protocol_v1.yaml", protocol)
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "rg_posr_frozen_equations_v1.md").write_text(
        method_markdown(), encoding="utf-8"
    )

    corruption = {
        "status": "FROZEN",
        "version": "RG_POSR_CORRUPTION_V1",
        "training_seen": {
            "modality_dropout": [0.0, 1.0],
            "gaussian_noise_std_fraction": [0.1, 0.25, 0.5, 0.75],
            "temporal_burst_loss_fraction": [0.1, 0.25, 0.5],
        },
        "heldout_unseen": {
            "stuck_sensor_fraction": [0.1, 0.25, 0.5],
            "motion_bias_or_drift_fraction": [0.1, 0.25, 0.5],
            "timestamp_jitter_fraction": [0.1, 0.25, 0.5],
        },
        "severity_target": "1-severity",
        "same_corruption_realization_for_paired_models": True,
        "test_unknown_never_used_for_selection": True,
    }
    evaluation = {
        "status": "FROZEN",
        "version": "RG_POSR_EVALUATION_V1",
        "seeds": [42, 52, 62, 72, 82],
        "aggregation_seconds": [30, 60, 120],
        "threshold_calibration": (
            "Session-1 validation plus five pseudo-unknown folds only"
        ),
        "metrics": [
            "closed_set_accuracy",
            "balanced_accuracy",
            "macro_f1",
            "known_acceptance_rate",
            "known_false_rejection_rate",
            "unknown_false_acceptance_rate",
            "unknown_true_rejection_rate",
            "AUROC",
            "AUPR",
            "EER",
            "OSCR",
        ],
        "statistics": [
            "paired_seed_comparison",
            "participant_level_bootstrap_95_percent_CI",
            "paired_participant_permutation_test",
        ],
        "ablation_order": [
            "fixed_fusion_softmax",
            "plain_prototype",
            "reliability_fusion_only",
            "reliability_rejection_only",
            "full_without_consistency",
            "full_rg_posr",
        ],
    }
    write_simple_yaml(experiment_dir / "corruption_protocol_frozen_v1.yaml", corruption)
    write_simple_yaml(experiment_dir / "evaluation_plan_frozen_v1.yaml", evaluation)

    summary_rows = []
    for index, name in enumerate(feature_names):
        train_values = raw_matrix[training_mask, index]
        all_values = raw_matrix[:, index]
        summary_rows.append(
            {
                "feature_name": name,
                "training_mean": float(np.mean(train_values)),
                "training_std": float(np.std(train_values)),
                "training_median": float(np.median(train_values)),
                "training_q05": float(np.quantile(train_values, 0.05)),
                "training_q95": float(np.quantile(train_values, 0.95)),
                "all_mean": float(np.mean(all_values)),
                "all_std": float(np.std(all_values)),
            }
        )
    save_csv(feature_dir / "reliability_feature_distribution_summary_v1.csv",
             pd.DataFrame(summary_rows))

    report = {
        "version": VERSION,
        "status": "FROZEN",
        "processed_root": str(processed_root),
        "audit_root": str(audit_root),
        "output_root": str(output),
        "duration_seconds": args.duration,
        "identity_dimensions": EXPECTED_DIMS,
        "reliability_dimensions": {"head": 6, "left": 6, "right": 6, "gaze": 7},
        "reliability_total_dimension": len(feature_names),
        "participant_split": split_summary,
        "window_count": int(len(manifest)),
        "training_reference_window_count": int(training_mask.sum()),
        "source_checksums": source_hashes,
        "frame_thresholds": thresholds,
        "cache_paths_remapped_to_processed_root": int(
            sum(
                Path(str(original)) != Path(str(resolved))
                for original, resolved in zip(
                    manifest["cache_path"], manifest["resolved_cache_path"]
                )
            )
        ),
        "existing_processed_dataset_modified": False,
        "model_training_started": False,
        "elapsed_minutes": float((time.time() - started) / 60.0),
        "next_experiment": "plain_normalized_prototype_open_set_baseline",
    }
    save_json(output / "freeze_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("PROTOCOL FREEZE AND RELIABILITY CACHE COMPLETED")
    return report


def synthetic_environment(root: Path) -> tuple[Path, Path]:
    processed = root / "processed"
    audit = root / "audit"
    cache_dir = processed / "resampled_sessions"
    cache_dir.mkdir(parents=True)
    audit.mkdir(parents=True)

    schema = {
        "head": [f"head_{i}" for i in range(13)],
        "left": [f"left_{i}" for i in range(22)],
        "right": [f"right_{i}" for i in range(22)],
        "gaze": [f"gaze_{i}" for i in range(19)],
        "quality": ["head_quality", "left_quality", "right_quality", "gaze_quality"],
    }
    stats = {}
    for modality, dimension in EXPECTED_DIMS.items():
        stats[modality] = {
            "feature_names": schema[modality],
            "mean": [0.0] * dimension,
            "std": [1.0] * dimension,
            "count": [1000] * dimension,
        }
    (processed / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (processed / "normalization_stats.json").write_text(json.dumps(stats), encoding="utf-8")

    participants = []
    rows = []
    rng = np.random.default_rng(42)
    window_counter = 0
    for player_index in range(69):
        player = f"{player_index + 1:02d}"
        role = "enrolled" if player_index < 49 else "unknown"
        participants.append(
            {
                "player_id": player,
                "role": role,
                "development_session": "s1",
                "test_session": "s2",
                "calibration_fold": player_index % 5 if role == "enrolled" else -1,
            }
        )
        for session_order, session in ((1, "s1"), (2, "s2")):
            length = 450
            head = rng.normal(size=(length, 13)).astype(np.float32)
            left = rng.normal(size=(length, 22)).astype(np.float32)
            right = rng.normal(size=(length, 22)).astype(np.float32)
            gaze = rng.normal(size=(length, 19)).astype(np.float32)
            gaze[:, 9:11] = np.clip(rng.normal(0.9, 0.05, size=(length, 2)), 0, 1)
            gaze[:, 11:13] = np.clip(rng.normal(4.5, 0.3, size=(length, 2)), 1, 8)
            gaze[:, 17] = np.abs(rng.normal(0.15, 0.05, size=length))
            gaze[:, 18] = np.abs(rng.normal(1.0, 0.2, size=length))
            quality = np.clip(rng.normal(0.95, 0.05, size=(length, 4)), 0, 1).astype(np.float32)
            cache = cache_dir / f"player_{player}__session_{session}.npz"
            np.savez_compressed(
                cache,
                time_seconds=np.arange(length) / 15.0,
                head=head,
                left=left,
                right=right,
                gaze=gaze,
                quality=quality,
            )
            if role == "unknown" and session_order == 1:
                continue
            split = "train" if session_order == 1 else (
                "test_known" if role == "enrolled" else "test_unknown"
            )
            for start in range(0, length, 150):
                rows.append(
                    {
                        "window_id": f"w{window_counter}",
                        "player_id": player,
                        "role": role,
                        "session_id": session,
                        "session_order": session_order,
                        "split": split,
                        "calibration_fold": player_index % 5 if role == "enrolled" else -1,
                        "duration_seconds": 10,
                        "sample_rate_hz": 15,
                        "start_index": start,
                        "stop_index": start + 150,
                        "cache_path": str(Path("F:/old/resampled_sessions") / cache.name),
                        "eligible_primary": True,
                        "eligible_motion": True,
                        "eligible_gaze": True,
                    }
                )
                window_counter += 1

    pd.DataFrame(participants).to_csv(
        processed / "participant_split_manifest.csv", index=False
    )
    pd.DataFrame(rows).to_csv(processed / "window_manifest.csv", index=False)
    (audit / "audit_report.json").write_text(
        json.dumps({"unresolved_items": [], "feature_table_status": "IDENTITY_FEATURES_RESOLVED"}),
        encoding="utf-8",
    )
    return processed, audit


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="rg_posr_freeze_test_") as folder:
        root = Path(folder)
        processed, audit = synthetic_environment(root)
        args = argparse.Namespace(
            processed_root=str(processed),
            audit_root=str(audit),
            output=str(root / "output"),
            duration=10,
            overwrite=True,
            self_test=False,
        )
        report = build(args)
        output = Path(report["output_root"])
        required = [
            "freeze_report.json",
            "00_protocol/frozen_protocol_v1.json",
            "00_protocol/frozen_protocol_v1.yaml",
            "01_features/identity_feature_table_frozen_v1.csv",
            "01_features/reliability_feature_table_frozen_v1.csv",
            "01_features/reliability_frame_thresholds_v1.json",
            "01_features/reliability_descriptor_normalization_v1.json",
            "04_cache/reliability_window_index_10s.csv",
            "04_cache/reliability_window_features_10s.npz",
        ]
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise AssertionError(f"Missing self-test outputs: {missing}")
        with np.load(
            output / "04_cache/reliability_window_features_10s.npz",
            allow_pickle=False,
        ) as archive:
            if archive["raw_features"].shape[1] != 25:
                raise AssertionError("Reliability dimension is not 25.")
            if archive["raw_features"].shape != archive["standardized_features"].shape:
                raise AssertionError("Raw/standardized shapes differ.")
        if report["participant_split"]["enrolled"] != 49:
            raise AssertionError("Synthetic frozen split was not preserved.")
    print("SELF-TEST PASSED")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Freeze RG-POSR protocol and build reliability cache."
    )
    value.add_argument("--processed-root")
    value.add_argument("--audit-root")
    value.add_argument("--output")
    value.add_argument("--duration", type=int, default=10)
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--self-test", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return
    missing = [
        name
        for name, value in (
            ("--processed-root", args.processed_root),
            ("--audit-root", args.audit_root),
            ("--output", args.output),
        )
        if not value
    ]
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join(missing))
    build(args)


if __name__ == "__main__":
    main()

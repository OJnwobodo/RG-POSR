#!/usr/bin/env python3
"""
Blind missing-/corrupted-modality robustness evaluation for the revised
RG-POSR reviewer experiment.

Principal comparison
--------------------
uniform vs nonlinear reliability-gated RG-POSR, using the already-trained
protocol-complete models for seeds 42, 52, 62, 72 and 82.

Key safeguard
-------------
Corruption metadata is used ONLY to generate the physical perturbation.
The reliability predictor receives no corruption type or severity. Its
eight descriptors are recomputed from the actually observed corrupted
standardised sequence via blind_reliability.descriptor_tensor().

No training, prototype update, gamma selection, threshold selection, or
Session-2 recalibration is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Compatibility with older NumPy used on the experiment laptop.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

import evaluate_blind_robustness as legacy
from blind_reliability import (
    DESCRIPTOR_NAMES,
    MODALITIES as BLIND_MODALITIES,
    ReliabilityEstimator,
    descriptor_tensor,
    reliability_gates,
    standardize_descriptors,
)

VERSION = "3.0.0-review-blind"
MODALITIES = ("head", "left", "right", "gaze")
EXPECTED_DIMS = {"head": 13, "left": 22, "right": 22, "gaze": 19}
DESCRIPTOR_DIM = 8
VARIANTS = ("uniform", "nonlinear")

if tuple(BLIND_MODALITIES) != MODALITIES:
    raise RuntimeError(
        f"blind_reliability modality order {tuple(BLIND_MODALITIES)} "
        f"does not match evaluator order {MODALITIES}"
    )

DEFAULT_SCENARIOS = (
    "clean",
    "head_missing",
    "left_controller_missing",
    "right_controller_missing",
    "gaze_missing",
    "both_controllers_missing",
    "controllers_noise_moderate",
    "gaze_noise_moderate",
    "severe_multi_sensor",
)

PAIRED_METRICS = (
    "closed_set_accuracy",
    "closed_set_macro_f1",
    "known_acceptance_rate_macro_user",
    "unknown_true_rejection_rate_macro_user",
    "balanced_detection_rate_macro_user",
    "open_set_overall_accuracy",
    "unknown_detection_auroc",
    "unknown_detection_aupr",
    "equal_error_rate_approx",
    "oscr_auc",
)


@dataclass
class RunConfig:
    processed_root: str
    frozen_root: str
    review_models_root: str
    blind_cache_root: str
    output_root: str
    seeds: tuple[int, ...]
    aggregation_seconds: tuple[int, ...]
    scenarios: tuple[str, ...]
    severity_sweep: tuple[float, ...]
    batch_size: int
    num_workers: int
    torch_threads: int
    device: str
    quick: bool
    strict_protocol: bool


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def parse_int_list(values: Sequence[str] | None, default: Sequence[int]) -> tuple[int, ...]:
    if values is None:
        return tuple(default)
    result: list[int] = []
    for value in values:
        for token in str(value).replace(",", " ").split():
            result.append(int(token))
    return tuple(result)


def parse_float_list(values: Sequence[str] | None, default: Sequence[float]) -> tuple[float, ...]:
    if values is None:
        return tuple(default)
    result: list[float] = []
    for value in values:
        for token in str(value).replace(",", " ").split():
            result.append(float(token))
    return tuple(result)


def stable_seed(*parts: str) -> int:
    value = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(value[:8], 16)


class NPZCache:
    def __init__(self, capacity: int = 4):
        self.capacity = int(capacity)
        self.values: dict[str, dict[str, np.ndarray]] = {}
        self.order: list[str] = []

    def get(self, path: str) -> dict[str, np.ndarray]:
        if path in self.values:
            if path in self.order:
                self.order.remove(path)
            self.order.append(path)
            return self.values[path]

        with np.load(path, allow_pickle=False) as archive:
            value = {
                modality: archive[modality].astype(np.float32)
                for modality in MODALITIES
            }

        self.values[path] = value
        self.order.append(path)
        while len(self.order) > self.capacity:
            oldest = self.order.pop(0)
            self.values.pop(oldest, None)
        return value


def apply_missing(values: np.ndarray) -> np.ndarray:
    return np.zeros_like(values, dtype=np.float32)


def apply_noise(
    values: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Preserve the original robustness protocol exactly.
    noise_scale = 0.10 + 0.40 * float(severity)
    noise = rng.normal(0.0, noise_scale, size=values.shape).astype(np.float32)
    return (values + noise).astype(np.float32)


def apply_burst(
    values: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    output = values.copy()
    length = int(output.shape[0])
    burst = max(1, int(round(length * float(severity))))
    start = 0 if length <= burst else int(rng.integers(0, length - burst + 1))
    output[start : start + burst] = 0.0
    return output.astype(np.float32)


def corrupt_one_modality(
    sequences: dict[str, np.ndarray],
    modality: str,
    mode: str,
    severity: float,
    rng: np.random.Generator,
) -> None:
    """
    Modify ONLY the observed modality sequence.

    No quality vector is accepted here, so corruption type/severity cannot
    be injected into reliability descriptors by construction.
    """
    severity = float(severity)
    if severity <= 0.0 or mode == "clean":
        return
    if mode == "missing":
        sequences[modality] = apply_missing(sequences[modality])
    elif mode == "noise":
        sequences[modality] = apply_noise(sequences[modality], severity, rng)
    elif mode == "burst":
        sequences[modality] = apply_burst(sequences[modality], severity, rng)
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")


def corrupt_window_np(
    sequences: dict[str, np.ndarray],
    scenario: str,
    severity: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Apply the original physical sequence perturbations without modifying
    or constructing any reliability/quality descriptor.
    """
    rng = np.random.default_rng(int(seed))
    severity = float(severity)

    if scenario == "clean":
        return sequences

    if scenario == "head_missing":
        corrupt_one_modality(sequences, "head", "missing", 1.0, rng)
    elif scenario == "left_controller_missing":
        corrupt_one_modality(sequences, "left", "missing", 1.0, rng)
    elif scenario == "right_controller_missing":
        corrupt_one_modality(sequences, "right", "missing", 1.0, rng)
    elif scenario == "gaze_missing":
        corrupt_one_modality(sequences, "gaze", "missing", 1.0, rng)
    elif scenario == "both_controllers_missing":
        corrupt_one_modality(sequences, "left", "missing", 1.0, rng)
        corrupt_one_modality(sequences, "right", "missing", 1.0, rng)
    elif scenario == "controllers_noise_moderate":
        corrupt_one_modality(sequences, "left", "noise", max(severity, 0.5), rng)
        corrupt_one_modality(sequences, "right", "noise", max(severity, 0.5), rng)
    elif scenario == "gaze_noise_moderate":
        corrupt_one_modality(sequences, "gaze", "noise", max(severity, 0.5), rng)
    elif scenario == "severe_multi_sensor":
        corrupt_one_modality(sequences, "left", "missing", 1.0, rng)
        corrupt_one_modality(sequences, "right", "noise", 1.0, rng)
        corrupt_one_modality(sequences, "gaze", "burst", 0.75, rng)
        corrupt_one_modality(sequences, "head", "noise", 0.5, rng)
    elif scenario.startswith("sweep_"):
        _, modality, mode = scenario.split("_", 2)
        if modality == "leftcontroller":
            modality = "left"
        elif modality == "rightcontroller":
            modality = "right"
        if modality not in MODALITIES:
            raise ValueError(f"Unknown sweep modality: {modality}")
        corrupt_one_modality(sequences, modality, mode, severity, rng)
    else:
        raise ValueError(f"Unknown corruption scenario: {scenario}")

    return sequences


class BlindRobustnessDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        means: dict[str, np.ndarray],
        stds: dict[str, np.ndarray],
        scenario: str,
        severity: float,
    ):
        self.frame = frame.reset_index(drop=True)
        self.means = means
        self.stds = stds
        self.scenario = str(scenario)
        self.severity = float(severity)
        self.cache = NPZCache(capacity=4)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        arrays = self.cache.get(str(row["resolved_cache_path"]))
        start = int(row["start_index"])
        stop = int(row["stop_index"])

        sequences: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            values = arrays[modality][start:stop].copy()
            values = (values - self.means[modality]) / self.stds[modality]
            values = np.nan_to_num(
                values, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            sequences[modality] = values

        # Deterministic matched perturbation. The same window/scenario receives
        # the same corruption regardless of evaluated model variant.
        sequences = corrupt_window_np(
            sequences,
            self.scenario,
            self.severity,
            seed=stable_seed(str(row.get("window_id", index)), self.scenario),
        )

        tensors = [
            torch.from_numpy(sequences[modality].astype(np.float32))
            for modality in MODALITIES
        ]
        return (*tensors, torch.tensor(index, dtype=torch.long))


class ReviewRGPOSR(nn.Module):
    """
    Checkpoint-compatible architecture for the reviewer-trained RG-POSR models.

    IMPORTANT:
    The reviewer checkpoints store reliability parameters under
    ``reliability_estimator.heads.<modality>...``.  We therefore reuse the
    exact ReliabilityEstimator implementation from blind_reliability.py
    instead of reconstructing a look-alike head.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        embedding_size: int,
        num_layers: int,
        dropout: float,
        reliability_hidden_size: int,
        reliability_variant: str,
    ):
        super().__init__()
        self.reliability_variant = str(reliability_variant)

        self.branches = nn.ModuleDict(
            {
                modality: legacy.GRUBranch(
                    EXPECTED_DIMS[modality],
                    hidden_size,
                    embedding_size,
                    num_layers,
                    dropout,
                )
                for modality in MODALITIES
            }
        )

        # Match training order/naming: prototypes + reliability_estimator.
        self.prototypes = nn.Parameter(
            torch.empty(num_classes, embedding_size)
        )
        nn.init.xavier_uniform_(self.prototypes)

        self.reliability_estimator = ReliabilityEstimator(
            variant=self.reliability_variant,
            hidden_size=reliability_hidden_size,
            dropout=dropout,
        )

    def encode(
        self,
        head: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        gaze: torch.Tensor,
        raw_descriptors: torch.Tensor,
        standardized_descriptors: torch.Tensor,
    ):
        values = {
            "head": head,
            "left": left,
            "right": right,
            "gaze": gaze,
        }

        modality_embeddings = torch.stack(
            [self.branches[m](values[m]) for m in MODALITIES],
            dim=1,
        )

        reliability = self.reliability_estimator(
            raw_descriptors,
            standardized_descriptors,
        )
        gates, effective = reliability_gates(reliability)

        fused = torch.sum(
            gates.unsqueeze(2) * modality_embeddings,
            dim=1,
        )
        fused = nn.functional.normalize(fused, dim=1)
        return fused, reliability, gates, effective

def _normalization_candidate(value: Any):
    if not isinstance(value, dict):
        return None
    if "median" not in value or "scale" not in value:
        return None
    try:
        median = np.asarray(value["median"], dtype=np.float32)
        scale = np.asarray(value["scale"], dtype=np.float32)
    except Exception:
        return None
    if median.size != 32 or scale.size != 32:
        return None
    return median.reshape(4, 8), scale.reshape(4, 8)


def load_blind_normalization(root: Path):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Blind cache root not found: {root}")

    for path in sorted(root.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidate = _normalization_candidate(raw)
        if candidate is not None:
            median, scale = candidate
            scale = np.where(np.abs(scale) > 1e-8, scale, 1.0).astype(
                np.float32
            )
            return median.astype(np.float32), scale, path

    for path in sorted(root.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "median" in archive and "scale" in archive:
                    candidate = _normalization_candidate(
                        {
                            "median": archive["median"],
                            "scale": archive["scale"],
                        }
                    )
                    if candidate is not None:
                        median, scale = candidate
                        scale = np.where(
                            np.abs(scale) > 1e-8, scale, 1.0
                        ).astype(np.float32)
                        return median.astype(np.float32), scale, path
        except Exception:
            continue

    raise RuntimeError(
        "Could not locate a 4x8/32-element blind descriptor median/scale "
        f"under {root}"
    )


def load_protocol(
    processed_root: Path,
    frozen_root: Path,
    strict: bool,
):
    freeze_report = json.loads(
        (frozen_root / "freeze_report.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (frozen_root / "validation_report.json").read_text(encoding="utf-8")
    )
    if freeze_report.get("status") != "FROZEN" or not validation.get(
        "passed"
    ):
        raise RuntimeError("Frozen protocol is invalid or missing.")

    if strict:
        split = freeze_report.get("participant_split", {})
        observed = (
            int(split.get("participants", -1)),
            int(split.get("enrolled", -1)),
            int(split.get("final_unknown", -1)),
        )
        if observed != (69, 49, 20):
            raise RuntimeError(
                f"Expected frozen 69/49/20 split, observed {observed}"
            )

    source_hashes = json.loads(
        (
            frozen_root / "01_features/source_checksums_v1.json"
        ).read_text(encoding="utf-8")
    )
    current_sources = {
        "feature_schema.json": processed_root / "feature_schema.json",
        "normalization_stats.json": processed_root / "normalization_stats.json",
        "participant_split_manifest.csv": (
            processed_root / "participant_split_manifest.csv"
        ),
        "window_manifest.csv": processed_root / "window_manifest.csv",
    }
    for name, path in current_sources.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if strict and legacy.sha256(path) != source_hashes.get(name):
            raise RuntimeError(f"Frozen source checksum changed: {name}")

    split_frame = pd.read_csv(
        processed_root / "participant_split_manifest.csv",
        dtype=str,
    )
    split_frame.columns = [
        legacy.normalized_name(column) for column in split_frame.columns
    ]
    split_frame["player_id"] = (
        split_frame["player_id"].astype(str).str.zfill(2)
    )
    split_frame["role"] = split_frame["role"].astype(str).str.lower()

    index = pd.read_csv(
        frozen_root / "04_cache/reliability_window_index_10s.csv",
        dtype={"player_id": str, "window_id": str},
        low_memory=False,
    )
    index.columns = [
        legacy.normalized_name(column) for column in index.columns
    ]
    index["player_id"] = index["player_id"].astype(str).str.zfill(2)
    index["role"] = index["role"].astype(str).str.lower()
    index["split"] = index["split"].astype(str).str.lower()

    for column in ("session_order", "start_index", "stop_index"):
        index[column] = pd.to_numeric(
            index[column], errors="raise"
        ).astype(int)

    for column in ("eligible_primary", "eligible_motion", "eligible_gaze"):
        index[column] = legacy.parse_bool_series(index[column])

    stats = json.loads(
        (processed_root / "normalization_stats.json").read_text(
            encoding="utf-8"
        )
    )
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        means[modality] = np.asarray(
            stats[modality]["mean"], dtype=np.float32
        )
        std = np.asarray(stats[modality]["std"], dtype=np.float32)
        stds[modality] = np.where(
            np.abs(std) > 1e-8, std, 1.0
        ).astype(np.float32)

    return split_frame, index, means, stds


def session2_frames(
    split: pd.DataFrame,
    index: pd.DataFrame,
    quick: bool,
):
    enrolled = sorted(
        split.loc[split["role"] == "enrolled", "player_id"].astype(str)
    )
    unknown = sorted(
        split.loc[
            split["role"].isin(["unknown", "final_unknown"]),
            "player_id",
        ].astype(str)
    )
    if quick:
        enrolled = enrolled[:8]
        unknown = unknown[:3]

    known = index[
        index["player_id"].isin(enrolled)
        & index["session_order"].eq(2)
        & index["eligible_primary"]
    ].copy()

    unknown_frame = index[
        index["player_id"].isin(unknown)
        & index["session_order"].eq(2)
        & index["eligible_primary"]
    ].copy()

    return known, unknown_frame, enrolled, unknown


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def variant_output_root(
    review_models_root: Path,
    seed: int,
    variant: str,
) -> Path:
    root = (
        review_models_root
        / f"protocol_seed{int(seed)}_{variant}"
    )
    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing completed {variant} output for seed {seed}: {root}"
        )
    return root


def load_review_model(
    output_root: Path,
    seed: int,
    variant: str,
    device: torch.device,
):
    checkpoint_path = (
        output_root
        / "models"
        / f"seed_{seed}"
        / "final"
        / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = legacy.safe_torch_load(checkpoint_path, device)
    config = checkpoint["model_config"]
    participants = list(config["participants"])

    model = ReviewRGPOSR(
        num_classes=len(participants),
        hidden_size=int(config["hidden_size"]),
        embedding_size=int(config["embedding_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        reliability_hidden_size=int(config["reliability_hidden_size"]),
        reliability_variant=variant,
    ).to(device)

    state = checkpoint["model_state_dict"]

    # Reviewer nonlinear checkpoints should contain the exact 8->32->1 heads:
    # reliability_estimator.heads.<modality>.network.0.weight = [32, 8]
    # reliability_estimator.heads.<modality>.network.3.weight = [1, 32]
    if variant == "nonlinear":
        expected_prefixes = [
            f"reliability_estimator.heads.{m}."
            for m in MODALITIES
        ]
        absent = [
            prefix
            for prefix in expected_prefixes
            if not any(k.startswith(prefix) for k in state)
        ]
        if absent:
            raise RuntimeError(
                "Nonlinear checkpoint does not contain the expected "
                f"reviewer reliability estimator modules: {absent}"
            )

        first_layer_shapes = {
            tuple(v.shape)
            for k, v in state.items()
            if k.startswith("reliability_estimator.heads.")
            and k.endswith("network.0.weight")
        }
        final_layer_shapes = {
            tuple(v.shape)
            for k, v in state.items()
            if k.startswith("reliability_estimator.heads.")
            and k.endswith("network.3.weight")
        }
        if first_layer_shapes != {(32, 8)}:
            raise RuntimeError(
                "Expected nonlinear first-layer shape (32, 8), observed "
                f"{sorted(first_layer_shapes)}"
            )
        if final_layer_shapes != {(1, 32)}:
            raise RuntimeError(
                "Expected nonlinear final-layer shape (1, 32), observed "
                f"{sorted(final_layer_shapes)}"
            )

    # Exact loading is deliberate: we should evaluate precisely the saved model,
    # not a partially compatible reconstruction.
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, checkpoint["participant_to_class"], checkpoint_path

def load_calibration(output_root: Path, seed: int):
    path = output_root / f"seed_{seed}_selected_calibration.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(float(key)): {
            "threshold": float(value["threshold"]),
            "gamma": float(value["gamma"]),
        }
        for key, value in raw.items()
    }


@torch.no_grad()
def infer_variant(
    model: ReviewRGPOSR,
    variant: str,
    frame: pd.DataFrame,
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    descriptor_median: np.ndarray,
    descriptor_scale: np.ndarray,
    scenario: str,
    severity: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    dataset = BlindRobustnessDataset(
        frame,
        means,
        stds,
        scenario,
        severity,
    )
    loader = make_loader(
        dataset,
        batch_size,
        num_workers,
        device,
    )

    embeddings = np.empty(
        (len(frame), model.prototypes.shape[1]),
        dtype=np.float32,
    )
    reliability = np.empty((len(frame), 4), dtype=np.float32)
    gates = np.empty((len(frame), 4), dtype=np.float32)
    effective = np.empty(len(frame), dtype=np.float32)

    median_t = torch.from_numpy(descriptor_median).to(device)
    scale_t = torch.from_numpy(descriptor_scale).to(device)

    for head, left, right, gaze, indices in loader:
        head = head.to(device, non_blocking=True)
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        gaze = gaze.to(device, non_blocking=True)

        sequences = {
            "head": head,
            "left": left,
            "right": right,
            "gaze": gaze,
        }

        # This is the critical reviewer correction: descriptors are computed
        # from the corrupted observations themselves. No scenario or severity
        # argument exists in descriptor_tensor().
        raw_descriptors = descriptor_tensor(sequences)
        standardized = standardize_descriptors(
            raw_descriptors,
            median_t,
            scale_t,
        )

        emb, rel, alpha, rho = model.encode(
            head,
            left,
            right,
            gaze,
            raw_descriptors,
            standardized,
        )

        rows = indices.numpy()
        embeddings[rows] = emb.detach().cpu().numpy().astype(np.float32)
        reliability[rows] = rel.detach().cpu().numpy().astype(np.float32)
        gates[rows] = alpha.detach().cpu().numpy().astype(np.float32)
        effective[rows] = rho.detach().cpu().numpy().astype(np.float32)

    return embeddings, reliability, gates, effective


def evaluate_variant_seed(
    *,
    variant: str,
    seed: int,
    scenario: str,
    severity: float,
    known_frame: pd.DataFrame,
    unknown_frame: pd.DataFrame,
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    descriptor_median: np.ndarray,
    descriptor_scale: np.ndarray,
    config: RunConfig,
    device: torch.device,
):
    review_root = Path(config.review_models_root).expanduser().resolve()
    output_root = variant_output_root(review_root, seed, variant)

    model, participant_to_class, checkpoint_path = load_review_model(
        output_root,
        seed,
        variant,
        device,
    )
    calibration = load_calibration(output_root, seed)

    known_outputs = infer_variant(
        model,
        variant,
        known_frame,
        means,
        stds,
        descriptor_median,
        descriptor_scale,
        scenario,
        severity,
        config.batch_size,
        config.num_workers,
        device,
    )
    unknown_outputs = infer_variant(
        model,
        variant,
        unknown_frame,
        means,
        stds,
        descriptor_median,
        descriptor_scale,
        scenario,
        severity,
        config.batch_size,
        config.num_workers,
        device,
    )

    prototypes = nn.functional.normalize(
        model.prototypes, dim=1
    ).detach().cpu().numpy().astype(np.float32)

    rows: list[dict[str, Any]] = []
    for duration in config.aggregation_seconds:
        (
            k_emb,
            k_rel,
            k_gate,
            k_eff,
            k_meta,
        ) = legacy.aggregate_arrays(
            known_frame,
            known_outputs[0],
            duration,
            known_outputs[1],
            known_outputs[2],
            known_outputs[3],
        )
        (
            u_emb,
            u_rel,
            u_gate,
            u_eff,
            u_meta,
        ) = legacy.aggregate_arrays(
            unknown_frame,
            unknown_outputs[0],
            duration,
            unknown_outputs[1],
            unknown_outputs[2],
            unknown_outputs[3],
        )

        if duration not in calibration:
            raise KeyError(
                f"Duration {duration} missing from {output_root.name} "
                f"seed-{seed} calibration."
            )

        gamma = float(calibration[duration]["gamma"])
        threshold = float(calibration[duration]["threshold"])

        metrics = legacy.evaluate_open_set(
            k_emb,
            u_emb,
            k_meta,
            u_meta,
            prototypes,
            participant_to_class,
            threshold,
            gamma=gamma,
            known_effective=k_eff,
            unknown_effective=u_eff,
            known_rel=k_rel,
            unknown_rel=u_rel,
            known_gates=k_gate,
            unknown_gates=u_gate,
        )

        rows.append(
            {
                "method": variant,
                "seed": int(seed),
                "scenario": scenario,
                "severity": float(severity),
                "aggregation_seconds": int(duration),
                "gamma": gamma,
                "threshold": threshold,
                "checkpoint": str(checkpoint_path),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def degradation_from_clean(metrics: pd.DataFrame) -> pd.DataFrame:
    clean = metrics[
        (metrics["scenario"] == "clean")
        & np.isclose(metrics["severity"], 0.0)
    ].copy()

    key = ["method", "seed", "aggregation_seconds"]
    metric_columns = [
        column
        for column in PAIRED_METRICS
        if column in metrics.columns
    ]

    clean = clean[key + metric_columns].copy()
    clean = clean.rename(
        columns={column: f"{column}_clean" for column in metric_columns}
    )

    merged = metrics.merge(clean, on=key, how="left")
    rows = []
    for _, row in merged.iterrows():
        item = {
            "method": row["method"],
            "seed": int(row["seed"]),
            "scenario": row["scenario"],
            "severity": float(row["severity"]),
            "aggregation_seconds": int(row["aggregation_seconds"]),
        }
        for metric in metric_columns:
            item[f"{metric}_degradation_from_clean"] = float(
                row[metric] - row[f"{metric}_clean"]
            )
        rows.append(item)
    return pd.DataFrame(rows)


def paired_nonlinear_vs_uniform(metrics: pd.DataFrame):
    uniform = metrics[metrics["method"] == "uniform"].copy()
    nonlinear = metrics[metrics["method"] == "nonlinear"].copy()

    key = [
        "seed",
        "scenario",
        "severity",
        "aggregation_seconds",
    ]
    merged = nonlinear.merge(
        uniform,
        on=key,
        suffixes=("_nonlinear", "_uniform"),
        how="inner",
    )

    rows = []
    for _, row in merged.iterrows():
        item = {
            key_name: row[key_name]
            for key_name in key
        }
        for metric in PAIRED_METRICS:
            a = f"{metric}_nonlinear"
            b = f"{metric}_uniform"
            if a in merged.columns and b in merged.columns:
                item[f"{metric}_nonlinear"] = float(row[a])
                item[f"{metric}_uniform"] = float(row[b])
                item[f"{metric}_delta_nonlinear_minus_uniform"] = (
                    float(row[a]) - float(row[b])
                )
        rows.append(item)

    paired = pd.DataFrame(rows)

    summary_rows = []
    if not paired.empty:
        group_keys = ["scenario", "severity", "aggregation_seconds"]
        for keys, group in paired.groupby(group_keys, dropna=False):
            for metric in PAIRED_METRICS:
                column = f"{metric}_delta_nonlinear_minus_uniform"
                if column not in group.columns:
                    continue
                values = group[column].dropna().astype(float)
                if values.empty:
                    continue
                improvement = (
                    -values
                    if metric
                    in {
                        "equal_error_rate_approx",
                    }
                    else values
                )
                summary_rows.append(
                    {
                        "scenario": keys[0],
                        "severity": float(keys[1]),
                        "aggregation_seconds": int(keys[2]),
                        "metric": metric,
                        "mean_delta_nonlinear_minus_uniform": float(
                            values.mean()
                        ),
                        "std_delta": float(values.std(ddof=1))
                        if len(values) > 1
                        else 0.0,
                        "nonlinear_improved_seed_count": int(
                            np.sum(improvement > 0)
                        ),
                        "seed_count": int(len(values)),
                    }
                )

    return paired, pd.DataFrame(summary_rows)


def reliability_response(metrics: pd.DataFrame):
    data = metrics[
        (metrics["method"] == "nonlinear")
        & metrics["scenario"].str.startswith("sweep_")
    ].copy()

    rows = []
    for _, row in data.iterrows():
        _, modality_token, corruption_type = str(
            row["scenario"]
        ).split("_", 2)
        modality = {
            "leftcontroller": "left",
            "rightcontroller": "right",
        }.get(modality_token, modality_token)

        rows.append(
            {
                "seed": int(row["seed"]),
                "aggregation_seconds": int(
                    row["aggregation_seconds"]
                ),
                "modality": modality,
                "corruption_type": corruption_type,
                "severity": float(row["severity"]),
                "target_modality_reliability": float(
                    row[f"known_mean_{modality}_reliability"]
                ),
                "target_modality_gate": float(
                    row[f"known_mean_{modality}_gate"]
                ),
                "closed_set_accuracy": float(
                    row["closed_set_accuracy"]
                ),
                "unknown_detection_auroc": float(
                    row["unknown_detection_auroc"]
                ),
                "oscr_auc": float(row["oscr_auc"]),
            }
        )

    response = pd.DataFrame(rows)
    if response.empty:
        return response, pd.DataFrame()

    summary = (
        response.groupby(
            [
                "aggregation_seconds",
                "modality",
                "corruption_type",
                "severity",
            ],
            as_index=False,
        )
        .agg(
            target_modality_reliability_mean=(
                "target_modality_reliability",
                "mean",
            ),
            target_modality_reliability_std=(
                "target_modality_reliability",
                "std",
            ),
            target_modality_gate_mean=(
                "target_modality_gate",
                "mean",
            ),
            target_modality_gate_std=(
                "target_modality_gate",
                "std",
            ),
            closed_set_accuracy_mean=("closed_set_accuracy", "mean"),
            auroc_mean=("unknown_detection_auroc", "mean"),
            oscr_mean=("oscr_auc", "mean"),
            count=("seed", "count"),
        )
    )

    monotonicity_rows = []
    for keys, group in summary.groupby(
        ["aggregation_seconds", "modality", "corruption_type"]
    ):
        group = group.sort_values("severity")
        rel = group["target_modality_reliability_mean"].to_numpy()
        gate = group["target_modality_gate_mean"].to_numpy()
        rel_diff = np.diff(rel)
        gate_diff = np.diff(gate)
        monotonicity_rows.append(
            {
                "aggregation_seconds": int(keys[0]),
                "modality": keys[1],
                "corruption_type": keys[2],
                "reliability_nonincreasing_steps": int(
                    np.sum(rel_diff <= 1e-6)
                ),
                "reliability_total_steps": int(len(rel_diff)),
                "gate_nonincreasing_steps": int(
                    np.sum(gate_diff <= 1e-6)
                ),
                "gate_total_steps": int(len(gate_diff)),
                "reliability_drop_0_to_max": float(rel[0] - rel[-1])
                if len(rel)
                else float("nan"),
                "gate_drop_0_to_max": float(gate[0] - gate[-1])
                if len(gate)
                else float("nan"),
            }
        )

    return response, pd.DataFrame(monotonicity_rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.columns
        if column
        not in {
            "method",
            "scenario",
            "checkpoint",
            "seed",
            "aggregation_seconds",
            "severity",
        }
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]

    rows = []
    for keys, group in metrics.groupby(
        ["method", "scenario", "severity", "aggregation_seconds"],
        dropna=False,
    ):
        for metric in numeric:
            values = group[metric].dropna().astype(float)
            if values.empty:
                continue
            rows.append(
                {
                    "method": keys[0],
                    "scenario": keys[1],
                    "severity": float(keys[2]),
                    "aggregation_seconds": int(keys[3]),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1))
                    if len(values) > 1
                    else 0.0,
                    "count": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def build_scenario_pairs(config: RunConfig):
    pairs: list[tuple[str, float]] = [("clean", 0.0)]

    for scenario in config.scenarios:
        if scenario != "clean":
            pairs.append((scenario, 1.0))

    # Quick run verifies the corrected path without launching the expensive
    # full severity-response sweep.
    if not config.quick:
        for modality in (
            "head",
            "leftcontroller",
            "rightcontroller",
            "gaze",
        ):
            for mode in ("noise", "burst", "missing"):
                for severity in config.severity_sweep:
                    pairs.append(
                        (
                            f"sweep_{modality}_{mode}",
                            float(severity),
                        )
                    )

    seen = set()
    unique = []
    for pair in pairs:
        if pair not in seen:
            unique.append(pair)
            seen.add(pair)
    return unique


def run(config: RunConfig, overwrite: bool = False):
    started = time.time()

    output = Path(config.output_root).expanduser().resolve()
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    device = legacy.choose_device(config.device)
    if config.torch_threads > 0:
        torch.set_num_threads(config.torch_threads)

    processed_root = Path(config.processed_root).expanduser().resolve()
    frozen_root = Path(config.frozen_root).expanduser().resolve()
    blind_cache_root = Path(
        config.blind_cache_root
    ).expanduser().resolve()

    split, index, means, stds = load_protocol(
        processed_root,
        frozen_root,
        config.strict_protocol,
    )
    known_frame, unknown_frame, enrolled, unknown = session2_frames(
        split,
        index,
        config.quick,
    )

    descriptor_median, descriptor_scale, norm_source = (
        load_blind_normalization(blind_cache_root)
    )

    pairs = build_scenario_pairs(config)
    save_json(output / "run_config.json", asdict(config))

    frames = []
    for seed in config.seeds:
        for scenario, severity in pairs:
            result_file = (
                output
                / "per_seed"
                / (
                    f"seed_{seed}_{scenario}_severity_"
                    f"{str(severity).replace('.', 'p')}.csv"
                )
            )

            if result_file.is_file():
                result = pd.read_csv(result_file)
            else:
                variant_frames = []
                for variant in VARIANTS:
                    variant_frames.append(
                        evaluate_variant_seed(
                            variant=variant,
                            seed=seed,
                            scenario=scenario,
                            severity=severity,
                            known_frame=known_frame,
                            unknown_frame=unknown_frame,
                            means=means,
                            stds=stds,
                            descriptor_median=descriptor_median,
                            descriptor_scale=descriptor_scale,
                            config=config,
                            device=device,
                        )
                    )
                result = pd.concat(
                    variant_frames,
                    ignore_index=True,
                )
                save_csv(result_file, result)

            frames.append(result)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "severity": severity,
                        "rows": len(result),
                    }
                ),
                flush=True,
            )

    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize_metrics(metrics)
    paired, paired_summary = paired_nonlinear_vs_uniform(metrics)
    degradation = degradation_from_clean(metrics)
    response, monotonicity = reliability_response(metrics)

    save_csv(output / "blind_robustness_seed_metrics.csv", metrics)
    save_csv(output / "blind_robustness_metrics_summary.csv", summary)
    save_csv(
        output / "paired_nonlinear_vs_uniform.csv",
        paired,
    )
    save_csv(
        output / "paired_nonlinear_vs_uniform_summary.csv",
        paired_summary,
    )
    save_csv(
        output / "blind_robustness_degradation_from_clean.csv",
        degradation,
    )
    save_csv(
        output / "blind_reliability_response_by_seed.csv",
        response,
    )
    save_csv(
        output / "blind_reliability_response_monotonicity.csv",
        monotonicity,
    )

    report = {
        "script_version": VERSION,
        "experiment": "review_blind_corruption_robustness",
        "methods_compared": list(VARIANTS),
        "processed_root": str(processed_root),
        "frozen_root": str(frozen_root),
        "review_models_root": str(
            Path(config.review_models_root).expanduser().resolve()
        ),
        "blind_cache_root": str(blind_cache_root),
        "blind_normalization_source": str(norm_source),
        "blind_descriptor_names": list(DESCRIPTOR_NAMES),
        "descriptor_shape": [4, 8],
        "corruption_type_or_severity_given_to_predictor": False,
        "descriptor_recomputation_from_observed_corrupted_sequence": True,
        "original_25_descriptor_cache_read_for_inference": False,
        "clean_reference_used_at_inference": False,
        "models_retrained_for_robustness": False,
        "prototypes_updated_for_robustness": False,
        "thresholds_or_gamma_recalibrated_for_robustness": False,
        "final_unknown_used_for_calibration": False,
        "deterministic_corruption_key": "window_id + scenario",
        "seeds": list(config.seeds),
        "aggregation_seconds": list(config.aggregation_seconds),
        "scenario_count": len(pairs),
        "scenarios": [
            {"name": name, "severity": severity}
            for name, severity in pairs
        ],
        "session2_known_windows": int(len(known_frame)),
        "session2_unknown_windows": int(len(unknown_frame)),
        "enrolled_users": int(len(enrolled)),
        "final_unknown_users": int(len(unknown)),
        "metric_rows": int(len(metrics)),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }

    save_json(output / "experiment_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("BLIND ROBUSTNESS EXPERIMENT COMPLETED", flush=True)
    return report


def self_test():
    if "corruption" in inspect.signature(descriptor_tensor).parameters:
        raise AssertionError(
            "descriptor_tensor unexpectedly accepts corruption metadata."
        )
    if "severity" in inspect.signature(descriptor_tensor).parameters:
        raise AssertionError(
            "descriptor_tensor unexpectedly accepts severity metadata."
        )

    rng = np.random.default_rng(123)
    base = {
        modality: rng.normal(
            0.0,
            1.0,
            size=(20, EXPECTED_DIMS[modality]),
        ).astype(np.float32)
        for modality in MODALITIES
    }

    a = {
        key: value.copy()
        for key, value in base.items()
    }
    b = {
        key: value.copy()
        for key, value in base.items()
    }

    a = corrupt_window_np(
        a, "gaze_noise_moderate", 1.0, seed=12345
    )
    b = corrupt_window_np(
        b, "gaze_noise_moderate", 1.0, seed=12345
    )

    for modality in MODALITIES:
        if not np.array_equal(a[modality], b[modality]):
            raise AssertionError(
                "Deterministic corruption check failed."
            )

    missing = {
        key: value.copy()
        for key, value in base.items()
    }
    missing = corrupt_window_np(
        missing, "gaze_missing", 1.0, seed=12345
    )
    if not np.allclose(missing["gaze"], 0.0):
        raise AssertionError("Missing-gaze corruption failed.")

    batch = {
        modality: torch.from_numpy(
            missing[modality][None, :, :]
        )
        for modality in MODALITIES
    }
    descriptors = descriptor_tensor(batch)
    if tuple(descriptors.shape) != (1, 4, 8):
        raise AssertionError(
            f"Expected descriptor shape (1,4,8), got "
            f"{tuple(descriptors.shape)}"
        )

    gaze_inactive = float(descriptors[0, 3, 0].item())
    if gaze_inactive < 0.999:
        raise AssertionError(
            "Blind descriptors did not detect complete gaze dropout."
        )

    print("=" * 72)
    print("BLIND ROBUSTNESS EVALUATOR SELF-TEST PASSED")
    print("=" * 72)
    print(f"Descriptor names: {tuple(DESCRIPTOR_NAMES)}")
    print("Corruption metadata is not accepted by descriptor_tensor().")
    print("Deterministic physical corruption check passed.")
    print("Complete missingness is detected from the observed sequence.")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Blind robustness evaluation of revised uniform vs nonlinear "
            "RG-POSR models."
        )
    )
    parser.add_argument("--processed-root")
    parser.add_argument("--frozen-root")
    parser.add_argument("--review-models-root")
    parser.add_argument("--blind-cache-root")
    parser.add_argument("--output")
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--aggregation-seconds", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--severity-sweep", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    if args.self_test:
        self_test()
        return

    required = {
        "--processed-root": args.processed_root,
        "--frozen-root": args.frozen_root,
        "--review-models-root": args.review_models_root,
        "--blind-cache-root": args.blind_cache_root,
        "--output": args.output,
    }
    missing = [
        name for name, value in required.items()
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required arguments: " + ", ".join(missing)
        )

    if args.quick:
        seeds = (42,)
        aggregation = (120,)
        scenarios = (
            "clean",
            "gaze_missing",
            "both_controllers_missing",
        )
        severity = (0.0, 1.0)
        strict = False
        batch_size = min(args.batch_size, 16)
    else:
        seeds = parse_int_list(
            args.seeds,
            (42, 52, 62, 72, 82),
        )
        aggregation = parse_int_list(
            args.aggregation_seconds,
            (30, 60, 120),
        )
        scenarios = (
            tuple(args.scenarios)
            if args.scenarios
            else DEFAULT_SCENARIOS
        )
        severity = parse_float_list(
            args.severity_sweep,
            (0.0, 0.25, 0.5, 0.75, 1.0),
        )
        strict = True
        batch_size = args.batch_size

    config = RunConfig(
        processed_root=args.processed_root,
        frozen_root=args.frozen_root,
        review_models_root=args.review_models_root,
        blind_cache_root=args.blind_cache_root,
        output_root=args.output,
        seeds=seeds,
        aggregation_seconds=aggregation,
        scenarios=scenarios,
        severity_sweep=severity,
        batch_size=batch_size,
        num_workers=args.num_workers,
        torch_threads=args.torch_threads,
        device=args.device,
        quick=args.quick,
        strict_protocol=strict,
    )
    run(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

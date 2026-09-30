#!/usr/bin/env python3
"""
Full Reliability-Gated Prototypical Open-Set Recognition (RG-POSR).

Frozen protocol
---------------
- Who Is Alyx XR behavioural biometrics
- head, left controller, right controller, gaze
- 49 enrolled users, 20 final unknown users
- Session 1 training/validation and pseudo-unknown calibration
- Session 2 final known/unknown evaluation
- 10-second windows at 15 Hz
- 30/60/120-second decision aggregation
- five seeds and five calibration folds
- final unknown users are never used for model, gamma, or threshold selection

Method
------
Each modality has:
1. a temporal GRU identity encoder;
2. an explicit reliability vector (6/6/6/7 descriptors);
3. a reliability MLP producing r_m in [0,1].

Fusion:
    alpha_m = r_m / sum_j r_j
    z = normalize(sum_m alpha_m z_m)

Open-set decision:
    d_u = 1 - z^T p_u
    rho_eff = sum_m alpha_m r_m
    d'_u = d_u / max(rho_eff, eps)^gamma

Training:
- angular-margin prototype classification;
- reliability-consistency Huber loss;
- reliability/gate alignment loss;
- monotonic ranking under synthetic corruption;
- training-only modality dropout, Gaussian noise and temporal burst loss.

The implementation is epoch-resumable and produces a paired five-seed comparison
against the completed plain normalized prototype baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

VERSION = "1.0.0"
METHOD = "full_rg_posr"
MODALITIES = ("head", "left", "right", "gaze")
EXPECTED_DIMS = {"head": 13, "left": 22, "right": 22, "gaze": 19}
RELIABILITY_DIMS = {"head": 6, "left": 6, "right": 6, "gaze": 7}
RELIABILITY_SLICES = {
    "head": slice(0, 6),
    "left": slice(6, 12),
    "right": slice(12, 18),
    "gaze": slice(18, 25),
}
GAMMA_DEFAULTS = (0.0, 0.5, 1.0, 2.0)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    last_error = None

    for attempt in range(5):
        temporary = path.with_name(
            f"{path.name}.{os.getpid()}.{attempt}.tmp"
        )

        try:
            if temporary.exists():
                temporary.unlink()

            frame.to_csv(
                temporary,
                index=False,
            )

            os.replace(
                temporary,
                path,
            )

            return

        except OSError as exc:
            last_error = exc

            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass

            if attempt < 4:
                time.sleep(
                    1.0 * (attempt + 1)
                )

    raise last_error


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def safe_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )


def parse_int_list(values: Sequence[str] | None, default: Sequence[int]) -> tuple[int, ...]:
    if values is None:
        return tuple(default)
    result: list[int] = []
    for value in values:
        for token in str(value).replace(",", " ").split():
            result.append(int(token))
    return tuple(result)


def parse_float_list(
    values: Sequence[str] | None, default: Sequence[float]
) -> tuple[float, ...]:
    if values is None:
        return tuple(default)
    result: list[float] = []
    for value in values:
        for token in str(value).replace(",", " ").split():
            result.append(float(token))
    return tuple(result)


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RunConfig:
    processed_root: str
    frozen_root: str
    prototype_output: str | None
    output_root: str
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    aggregation_seconds: tuple[int, ...]
    gamma_candidates: tuple[float, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    embedding_size: int
    num_layers: int
    dropout: float
    reliability_hidden_size: int
    arcface_scale: float
    arcface_margin: float
    lambda_reliability: float
    lambda_gate_alignment: float
    lambda_monotonic: float
    monotonic_margin: float
    corruption_probability: float
    patience: int
    num_workers: int
    torch_threads: int
    device: str
    amp: bool
    quick: bool
    strict_protocol: bool


class NPZCache:
    def __init__(self, capacity: int = 4):
        self.capacity = int(capacity)
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str) -> dict[str, np.ndarray]:
        if path in self.values:
            value = self.values.pop(path)
            self.values[path] = value
            return value
        with np.load(path, allow_pickle=False) as archive:
            value = {
                modality: archive[modality].astype(np.float32)
                for modality in MODALITIES
            }
        self.values[path] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


class TrainingDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        means: dict[str, np.ndarray],
        stds: dict[str, np.ndarray],
        participant_to_class: dict[str, int],
        raw_reliability: np.ndarray,
        natural_targets: np.ndarray,
    ):
        self.frame = frame.reset_index(drop=True)
        self.means = means
        self.stds = stds
        self.participant_to_class = participant_to_class
        self.raw_reliability = raw_reliability
        self.natural_targets = natural_targets
        self.cache = NPZCache(capacity=4)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        arrays = self.cache.get(str(row["resolved_cache_path"]))
        start = int(row["start_index"])
        stop = int(row["stop_index"])
        sequences = []
        for modality in MODALITIES:
            values = arrays[modality][start:stop]
            values = (values - self.means[modality]) / self.stds[modality]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            sequences.append(torch.from_numpy(values.astype(np.float32)))
        reliability_row = int(row["_reliability_row"])
        raw_quality = torch.from_numpy(
            self.raw_reliability[reliability_row].astype(np.float32)
        )
        natural_target = torch.from_numpy(
            self.natural_targets[reliability_row].astype(np.float32)
        )
        label = self.participant_to_class[str(row["player_id"])]
        return (
            *sequences,
            raw_quality,
            natural_target,
            torch.tensor(label, dtype=torch.long),
        )


class InferenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        means: dict[str, np.ndarray],
        stds: dict[str, np.ndarray],
        raw_reliability: np.ndarray,
        natural_targets: np.ndarray,
    ):
        self.frame = frame.reset_index(drop=True)
        self.means = means
        self.stds = stds
        self.raw_reliability = raw_reliability
        self.natural_targets = natural_targets
        self.cache = NPZCache(capacity=4)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        arrays = self.cache.get(str(row["resolved_cache_path"]))
        start = int(row["start_index"])
        stop = int(row["stop_index"])
        sequences = []
        for modality in MODALITIES:
            values = arrays[modality][start:stop]
            values = (values - self.means[modality]) / self.stds[modality]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            sequences.append(torch.from_numpy(values.astype(np.float32)))
        reliability_row = int(row["_reliability_row"])
        raw_quality = torch.from_numpy(
            self.raw_reliability[reliability_row].astype(np.float32)
        )
        natural_target = torch.from_numpy(
            self.natural_targets[reliability_row].astype(np.float32)
        )
        return (
            *sequences,
            raw_quality,
            natural_target,
            torch.tensor(index, dtype=torch.long),
        )


class GRUBranch(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        embedding_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(values)
        embedding = self.projection(hidden[-1])
        return nn.functional.normalize(embedding, dim=1)


class ReliabilityHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, quality: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(quality)).squeeze(1)


class FullRGPOSR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        embedding_size: int,
        num_layers: int,
        dropout: float,
        reliability_hidden_size: int,
    ):
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                modality: GRUBranch(
                    EXPECTED_DIMS[modality],
                    hidden_size,
                    embedding_size,
                    num_layers,
                    dropout,
                )
                for modality in MODALITIES
            }
        )
        self.reliability_heads = nn.ModuleDict(
            {
                modality: ReliabilityHead(
                    RELIABILITY_DIMS[modality],
                    reliability_hidden_size,
                    dropout,
                )
                for modality in MODALITIES
            }
        )
        self.prototypes = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.prototypes)

    def predict_reliability(self, standardized_quality: torch.Tensor) -> torch.Tensor:
        outputs = []
        for modality in MODALITIES:
            quality = standardized_quality[:, RELIABILITY_SLICES[modality]]
            outputs.append(self.reliability_heads[modality](quality))
        return torch.stack(outputs, dim=1)

    def encode(
        self,
        head: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        gaze: torch.Tensor,
        standardized_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_map = {
            "head": head,
            "left": left,
            "right": right,
            "gaze": gaze,
        }
        modality_embeddings = torch.stack(
            [
                self.branches[modality](sequence_map[modality])
                for modality in MODALITIES
            ],
            dim=1,
        )
        reliability = self.predict_reliability(standardized_quality)
        alpha = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused = torch.sum(alpha.unsqueeze(2) * modality_embeddings, dim=1)
        fused = nn.functional.normalize(fused, dim=1)
        effective_reliability = torch.sum(alpha * reliability, dim=1)
        return fused, reliability, alpha, effective_reliability

    def cosine_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        prototypes = nn.functional.normalize(self.prototypes, dim=1)
        return embedding @ prototypes.t()

    def forward(
        self,
        head: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        gaze: torch.Tensor,
        standardized_quality: torch.Tensor,
    ):
        embedding, reliability, alpha, effective = self.encode(
            head, left, right, gaze, standardized_quality
        )
        cosine = self.cosine_logits(embedding)
        return embedding, cosine, reliability, alpha, effective


def arcface_logits(
    cosine: torch.Tensor,
    labels: torch.Tensor,
    scale: float,
    margin: float,
) -> torch.Tensor:
    cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-7))
    phi = cosine * math.cos(margin) - sine * math.sin(margin)
    threshold = math.cos(math.pi - margin)
    correction = math.sin(math.pi - margin) * margin
    phi = torch.where(cosine > threshold, phi, cosine - correction)
    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    return scale * (one_hot * phi + (1.0 - one_hot) * cosine)


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
    if freeze_report.get("status") != "FROZEN" or not validation.get("passed"):
        raise RuntimeError("Frozen RG-POSR protocol is missing or invalid.")

    if strict:
        split = freeze_report.get("participant_split", {})
        observed = (
            int(split.get("participants", -1)),
            int(split.get("enrolled", -1)),
            int(split.get("final_unknown", -1)),
        )
        if observed != (69, 49, 20):
            raise RuntimeError(f"Expected frozen 69/49/20 protocol, observed {observed}")

    source_hashes = json.loads(
        (frozen_root / "01_features/source_checksums_v1.json").read_text(
            encoding="utf-8"
        )
    )
    current_sources = {
        "feature_schema.json": processed_root / "feature_schema.json",
        "normalization_stats.json": processed_root / "normalization_stats.json",
        "participant_split_manifest.csv": processed_root / "participant_split_manifest.csv",
        "window_manifest.csv": processed_root / "window_manifest.csv",
    }
    for name, path in current_sources.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if strict and sha256(path) != source_hashes.get(name):
            raise RuntimeError(f"Frozen source checksum changed: {name}")

    split_frame = pd.read_csv(
        processed_root / "participant_split_manifest.csv", dtype=str
    )
    split_frame.columns = [normalized_name(column) for column in split_frame.columns]
    split_frame["player_id"] = split_frame["player_id"].astype(str).str.zfill(2)
    split_frame["role"] = split_frame["role"].astype(str).str.lower()
    split_frame["calibration_fold"] = pd.to_numeric(
        split_frame["calibration_fold"], errors="coerce"
    ).fillna(-1).astype(int)

    index = pd.read_csv(
        frozen_root / "04_cache/reliability_window_index_10s.csv",
        dtype={"player_id": str, "window_id": str},
        low_memory=False,
    )
    index.columns = [normalized_name(column) for column in index.columns]
    index["player_id"] = index["player_id"].astype(str).str.zfill(2)
    index["role"] = index["role"].astype(str).str.lower()
    index["split"] = index["split"].astype(str).str.lower()
    for column in ("session_order", "start_index", "stop_index", "calibration_fold"):
        index[column] = pd.to_numeric(index[column], errors="raise").astype(int)
    for column in ("eligible_primary", "eligible_motion", "eligible_gaze"):
        index[column] = parse_bool_series(index[column])
    index["_reliability_row"] = np.arange(len(index), dtype=int)

    cache_path = frozen_root / "04_cache/reliability_window_features_10s.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        raw_reliability = archive["raw_features"].astype(np.float32)
        standardized_reliability = archive["standardized_features"].astype(np.float32)
        natural_targets = archive["natural_quality_targets"].astype(np.float32)
        feature_names = archive["feature_names"].astype(str).tolist()
        window_ids = archive["window_ids"].astype(str)

    if raw_reliability.shape != (len(index), 25):
        raise RuntimeError(
            f"Expected reliability cache {(len(index), 25)}, "
            f"observed {raw_reliability.shape}"
        )
    if not np.array_equal(
        index["window_id"].astype(str).to_numpy(), window_ids.astype(str)
    ):
        raise RuntimeError("Reliability cache/window index misalignment.")

    descriptor_normalization = json.loads(
        (
            frozen_root
            / "01_features/reliability_descriptor_normalization_v1.json"
        ).read_text(encoding="utf-8")
    )
    rel_median = np.asarray(descriptor_normalization["median"], dtype=np.float32)
    rel_scale = np.asarray(descriptor_normalization["scale"], dtype=np.float32)
    rel_scale = np.where(np.abs(rel_scale) > 1e-8, rel_scale, 1.0).astype(np.float32)

    stats = json.loads(
        (processed_root / "normalization_stats.json").read_text(encoding="utf-8")
    )
    means = {}
    stds = {}
    for modality in MODALITIES:
        means[modality] = np.asarray(stats[modality]["mean"], dtype=np.float32)
        std = np.asarray(stats[modality]["std"], dtype=np.float32)
        stds[modality] = np.where(np.abs(std) > 1e-8, std, 1.0).astype(np.float32)
        if len(means[modality]) != EXPECTED_DIMS[modality]:
            raise RuntimeError(f"{modality} normalization dimension mismatch")

    return (
        split_frame,
        index,
        means,
        stds,
        raw_reliability,
        standardized_reliability,
        natural_targets,
        feature_names,
        rel_median,
        rel_scale,
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch * 100003)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def standardize_quality(
    raw_quality: torch.Tensor,
    median: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    standardized = (raw_quality - median) / scale
    return torch.nan_to_num(standardized, nan=0.0, posinf=10.0, neginf=-10.0)


def apply_synthetic_corruption(
    sequences: dict[str, torch.Tensor],
    raw_quality: torch.Tensor,
    natural_targets: torch.Tensor,
    median: torch.Tensor,
    scale: torch.Tensor,
    probability: float,
    seed: int,
):
    """
    Synthetic corruption is never passed as a model input.

    Corruption choices per sample/modality:
    0 clean
    1 complete modality dropout
    2 additive Gaussian sensor noise
    3 contiguous temporal burst loss
    """
    device = raw_quality.device
    batch_size = raw_quality.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    corrupted = {key: value.clone() for key, value in sequences.items()}
    quality = raw_quality.clone()
    targets = natural_targets.clone().clamp(0.0, 1.0)
    severity_matrix = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
    type_matrix = torch.zeros((batch_size, 4), dtype=torch.long, device=device)

    severity_choices = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    random_prob = torch.rand((batch_size, 4), generator=generator)
    active = random_prob < probability
    types = torch.randint(1, 4, (batch_size, 4), generator=generator)
    severity_indices = torch.randint(
        0, len(severity_choices), (batch_size, 4), generator=generator
    )
    severities = severity_choices[severity_indices] * active.float()

    for modality_index, modality in enumerate(MODALITIES):
        seq = corrupted[modality]
        local_slice = RELIABILITY_SLICES[modality]
        local_quality = quality[:, local_slice]
        for sample_index in range(batch_size):
            severity = float(severities[sample_index, modality_index].item())
            if severity <= 0:
                continue
            corruption_type = int(types[sample_index, modality_index].item())
            severity_matrix[sample_index, modality_index] = severity
            type_matrix[sample_index, modality_index] = corruption_type

            if corruption_type == 1:
                seq[sample_index].zero_()
                local_quality[sample_index, 0] = 0.0
                local_quality[sample_index, 1] = 0.0
                local_quality[sample_index, 2] = 1.0
                if modality == "gaze":
                    local_quality[sample_index, 3] = 1.0
                    local_quality[sample_index, 4] = 1.0
                    local_quality[sample_index, 5] = 0.0
                    local_quality[sample_index, 6] = 0.0
                else:
                    local_quality[sample_index, 3] = 1.0
                    local_quality[sample_index, 4] = 0.0
                    local_quality[sample_index, 5] = 0.0
                targets[sample_index, modality_index] = 0.0

            elif corruption_type == 2:
                noise_scale = 0.10 + 0.40 * severity
                noise_generator = torch.Generator(device="cpu")
                noise_generator.manual_seed(
                    int(seed + 1000003 * (modality_index + 1) + sample_index)
                )
                noise = torch.randn(
                    seq[sample_index].shape,
                    generator=noise_generator,
                    dtype=seq.dtype,
                ).to(device) * noise_scale
                seq[sample_index] = seq[sample_index] + noise
                local_quality[sample_index, 2] = torch.maximum(
                    local_quality[sample_index, 2],
                    torch.tensor(0.5 * severity, device=device),
                )
                if modality == "gaze":
                    local_quality[sample_index, 5] += severity
                    local_quality[sample_index, 6] = torch.maximum(
                        local_quality[sample_index, 6],
                        torch.tensor(0.5 * severity, device=device),
                    )
                else:
                    local_quality[sample_index, 4] += severity
                    local_quality[sample_index, 5] = torch.maximum(
                        local_quality[sample_index, 5],
                        torch.tensor(0.5 * severity, device=device),
                    )
                targets[sample_index, modality_index] *= 1.0 - 0.70 * severity

            elif corruption_type == 3:
                length = seq.shape[1]
                burst = max(1, int(round(length * severity)))
                max_start = max(0, length - burst)
                start_generator = torch.Generator(device="cpu")
                start_generator.manual_seed(
                    int(seed + 2000003 * (modality_index + 1) + sample_index)
                )
                start = (
                    int(torch.randint(0, max_start + 1, (1,), generator=start_generator).item())
                    if max_start > 0
                    else 0
                )
                seq[sample_index, start : start + burst].zero_()
                local_quality[sample_index, 0] *= 1.0 - severity
                local_quality[sample_index, 1] *= 1.0 - severity
                local_quality[sample_index, 2] = torch.maximum(
                    local_quality[sample_index, 2],
                    torch.tensor(severity, device=device),
                )
                targets[sample_index, modality_index] *= 1.0 - severity

        quality[:, local_slice] = local_quality
        corrupted[modality] = seq

    standardized_quality = standardize_quality(quality, median, scale)
    return corrupted, standardized_quality, targets.clamp(0.0, 1.0), severity_matrix, type_matrix


def gate_entropy(alpha: torch.Tensor) -> torch.Tensor:
    return -torch.sum(alpha * torch.log(alpha.clamp_min(1e-8)), dim=1)


def train_epoch(
    model: FullRGPOSR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp: bool,
    config: RunConfig,
    rel_median: torch.Tensor,
    rel_scale: torch.Tensor,
    seed: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "reliability_loss": 0.0,
        "gate_alignment_loss": 0.0,
        "monotonic_loss": 0.0,
        "mean_effective_reliability": 0.0,
        "mean_gate_entropy": 0.0,
    }
    count = 0
    criterion = nn.CrossEntropyLoss()

    for batch_index, batch in enumerate(loader):
        head, left, right, gaze, raw_quality, targets, labels = batch
        sequence_map = {
            "head": head.to(device, non_blocking=True),
            "left": left.to(device, non_blocking=True),
            "right": right.to(device, non_blocking=True),
            "gaze": gaze.to(device, non_blocking=True),
        }
        raw_quality = raw_quality.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_standardized = standardize_quality(raw_quality, rel_median, rel_scale)

        corruption_seed = seed * 100000000 + epoch * 100000 + batch_index
        corrupted, corrupted_quality, corrupted_targets, severities, _ = (
            apply_synthetic_corruption(
                sequence_map,
                raw_quality,
                targets,
                rel_median,
                rel_scale,
                config.corruption_probability,
                corruption_seed,
            )
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=(amp and device.type == "cuda"),
        ):
            embedding, cosine, reliability, alpha, effective = model(
                corrupted["head"],
                corrupted["left"],
                corrupted["right"],
                corrupted["gaze"],
                corrupted_quality,
            )
            logits = arcface_logits(
                cosine, labels, config.arcface_scale, config.arcface_margin
            )
            classification_loss = criterion(logits, labels)
            reliability_loss = nn.functional.smooth_l1_loss(
                reliability, corrupted_targets
            )

            target_alpha = corrupted_targets.clamp_min(0.0)
            target_alpha = target_alpha / target_alpha.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            gate_alignment_loss = nn.functional.mse_loss(alpha, target_alpha)

            clean_reliability = model.predict_reliability(clean_standardized)
            active = severities > 0
            monotonic_margin = config.monotonic_margin * severities
            ranking = torch.relu(
                monotonic_margin - (clean_reliability - reliability)
            )
            monotonic_loss = (
                ranking[active].mean()
                if torch.any(active)
                else torch.zeros((), device=device)
            )

            loss = (
                classification_loss
                + config.lambda_reliability * reliability_loss
                + config.lambda_gate_alignment * gate_alignment_loss
                + config.lambda_monotonic * monotonic_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_count = len(labels)
        count += batch_count
        totals["loss"] += float(loss.item()) * batch_count
        totals["classification_loss"] += float(classification_loss.item()) * batch_count
        totals["reliability_loss"] += float(reliability_loss.item()) * batch_count
        totals["gate_alignment_loss"] += float(gate_alignment_loss.item()) * batch_count
        totals["monotonic_loss"] += float(monotonic_loss.item()) * batch_count
        totals["mean_effective_reliability"] += float(effective.mean().item()) * batch_count
        totals["mean_gate_entropy"] += float(gate_entropy(alpha).mean().item()) * batch_count

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def infer_outputs(
    model: FullRGPOSR,
    frame: pd.DataFrame,
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
    raw_reliability: np.ndarray,
    natural_targets: np.ndarray,
    rel_median: np.ndarray,
    rel_scale: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
):
    dataset = InferenceDataset(
        frame, means, stds, raw_reliability, natural_targets
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    embedding_size = model.prototypes.shape[1]
    embeddings = np.empty((len(frame), embedding_size), dtype=np.float32)
    reliabilities = np.empty((len(frame), 4), dtype=np.float32)
    gates = np.empty((len(frame), 4), dtype=np.float32)
    effective = np.empty(len(frame), dtype=np.float32)
    targets_out = np.empty((len(frame), 4), dtype=np.float32)

    median_t = torch.from_numpy(rel_median).to(device)
    scale_t = torch.from_numpy(rel_scale).to(device)
    model.eval()
    for batch in loader:
        head, left, right, gaze, raw_quality, targets, indices = batch
        head = head.to(device, non_blocking=True)
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        gaze = gaze.to(device, non_blocking=True)
        raw_quality = raw_quality.to(device, non_blocking=True)
        standardized = standardize_quality(raw_quality, median_t, scale_t)
        with torch.autocast(
            device_type=device.type,
            enabled=(amp and device.type == "cuda"),
        ):
            embedding, reliability, alpha, rho = model.encode(
                head, left, right, gaze, standardized
            )
        rows = indices.numpy()
        embeddings[rows] = embedding.detach().cpu().numpy().astype(np.float32)
        reliabilities[rows] = reliability.detach().cpu().numpy().astype(np.float32)
        gates[rows] = alpha.detach().cpu().numpy().astype(np.float32)
        effective[rows] = rho.detach().cpu().numpy().astype(np.float32)
        targets_out[rows] = targets.numpy().astype(np.float32)

    return embeddings, reliabilities, gates, effective, targets_out


def class_metrics(
    true_indices: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    if len(true_indices) == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "minimum_class_accuracy": float("nan"),
            "median_class_accuracy": float("nan"),
        }
    per_class = []
    for class_index in range(num_classes):
        mask = true_indices == class_index
        if np.any(mask):
            per_class.append(float(np.mean(predictions[mask] == true_indices[mask])))
    return {
        "accuracy": float(accuracy_score(true_indices, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_indices, predictions)
        ),
        "macro_f1": float(
            f1_score(true_indices, predictions, average="macro", zero_division=0)
        ),
        "minimum_class_accuracy": float(np.min(per_class)),
        "median_class_accuracy": float(np.median(per_class)),
    }


def validation_selection_metrics(
    model: FullRGPOSR,
    frame: pd.DataFrame,
    means,
    stds,
    raw_reliability,
    natural_targets,
    rel_median,
    rel_scale,
    participant_to_class,
    config,
    device,
):
    embeddings, reliabilities, gates, effective, targets = infer_outputs(
        model,
        frame,
        means,
        stds,
        raw_reliability,
        natural_targets,
        rel_median,
        rel_scale,
        config.batch_size,
        config.num_workers,
        device,
        config.amp,
    )
    prototypes = (
        nn.functional.normalize(model.prototypes, dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    similarities = embeddings @ prototypes.T
    predictions = np.argmax(similarities, axis=1)
    true_indices = np.asarray(
        [participant_to_class[str(value)] for value in frame["player_id"]],
        dtype=int,
    )
    metrics = class_metrics(true_indices, predictions, len(participant_to_class))
    metrics.update(
        {
            "mean_effective_reliability": float(np.mean(effective)),
            "mean_gate_entropy": float(
                np.mean(-np.sum(gates * np.log(np.clip(gates, 1e-8, 1.0)), axis=1))
            ),
            "reliability_target_mae": float(np.mean(np.abs(reliabilities - targets))),
        }
    )
    return metrics


def model_paths(output: Path, seed: int, model_name: str):
    directory = output / "models" / f"seed_{seed}" / model_name
    return (
        directory / "last.pt",
        directory / "best.pt",
        directory / "history.csv",
    )


def train_or_resume_model(
    *,
    model_name: str,
    seed: int,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    participant_ids: list[str],
    means,
    stds,
    raw_reliability,
    natural_targets,
    rel_median,
    rel_scale,
    config: RunConfig,
    output: Path,
    device: torch.device,
) -> FullRGPOSR:
    participant_to_class = {
        participant: index for index, participant in enumerate(participant_ids)
    }
    training_dataset = TrainingDataset(
        train_frame,
        means,
        stds,
        participant_to_class,
        raw_reliability,
        natural_targets,
    )

    initialization_seed = seed + sum(ord(character) for character in model_name)
    set_seed(initialization_seed)
    model = FullRGPOSR(
        len(participant_ids),
        config.hidden_size,
        config.embedding_size,
        config.num_layers,
        config.dropout,
        config.reliability_hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = bool(config.amp and device.type == "cuda")

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=use_amp)
    elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        raise RuntimeError(
            "This PyTorch installation provides no compatible GradScaler."
        )

    median_t = torch.from_numpy(rel_median).to(device)
    scale_t = torch.from_numpy(rel_scale).to(device)
    last_path, best_path, history_path = model_paths(output, seed, model_name)

    model_config = {
        "version": VERSION,
        "method": METHOD,
        "model_name": model_name,
        "seed": seed,
        "participants": participant_ids,
        "hidden_size": config.hidden_size,
        "embedding_size": config.embedding_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "reliability_hidden_size": config.reliability_hidden_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "arcface_scale": config.arcface_scale,
        "arcface_margin": config.arcface_margin,
        "lambda_reliability": config.lambda_reliability,
        "lambda_gate_alignment": config.lambda_gate_alignment,
        "lambda_monotonic": config.lambda_monotonic,
        "monotonic_margin": config.monotonic_margin,
        "corruption_probability": config.corruption_probability,
        "batch_size": config.batch_size,
    }
    config_hash = hashlib.sha256(
        json.dumps(model_config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    history = pd.DataFrame()
    start_epoch = 1
    best_score = (-1.0, -1.0)
    best_epoch = 0
    epochs_without_improvement = 0

    if last_path.is_file():
        saved = safe_torch_load(last_path, device)
        if saved.get("config_hash") != config_hash:
            raise RuntimeError(
                f"Resume configuration mismatch for seed={seed}, model={model_name}"
            )
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if saved.get("scaler_state_dict"):
            scaler.load_state_dict(saved["scaler_state_dict"])
        start_epoch = int(saved["epoch_completed"]) + 1
        best_score = tuple(float(value) for value in saved["best_score"])
        best_epoch = int(saved["best_epoch"])
        epochs_without_improvement = int(
            saved.get("epochs_without_improvement", 0)
        )
        if history_path.is_file():
            history = pd.read_csv(history_path)
            history = history[
                pd.to_numeric(history["epoch"], errors="coerce")
                <= int(saved["epoch_completed"])
            ].copy()
        print(
            f"RESUMING {model_name}, seed={seed}, next_epoch={start_epoch}",
            flush=True,
        )

    if start_epoch > config.epochs and best_path.is_file():
        best = safe_torch_load(best_path, device)
        model.load_state_dict(best["model_state_dict"], strict=True)
        return model

    for epoch in range(start_epoch, config.epochs + 1):
        loader = make_loader(
            training_dataset,
            config.batch_size,
            True,
            seed,
            epoch,
            config.num_workers,
            device,
        )
        started = time.time()
        training = train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            device,
            config.amp,
            config,
            median_t,
            scale_t,
            seed,
            epoch,
        )
        validation = validation_selection_metrics(
            model,
            validation_frame,
            means,
            stds,
            raw_reliability,
            natural_targets,
            rel_median,
            rel_scale,
            participant_to_class,
            config,
            device,
        )
        score = (
            validation["minimum_class_accuracy"],
            validation["balanced_accuracy"],
        )
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "epoch_seconds": time.time() - started,
            "improved_best": improved,
            **{f"training_{key}": value for key, value in training.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        history = (
            history.sort_values("epoch")
            .drop_duplicates("epoch", keep="last")
            .reset_index(drop=True)
        )
        save_csv(history_path, history)

        payload = {
            "script_version": VERSION,
            "config_hash": config_hash,
            "model_config": model_config,
            "epoch_completed": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_score": list(best_score),
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "participant_to_class": participant_to_class,
        }
        atomic_torch_save(payload, last_path)
        if improved:
            best_payload = dict(payload)
            best_payload["selected_epoch"] = epoch
            atomic_torch_save(best_payload, best_path)

        print(
            json.dumps(
                {
                    "model": model_name,
                    "seed": seed,
                    **row,
                    "best_epoch": best_epoch,
                }
            ),
            flush=True,
        )
        print(
            f"CHECKPOINT SAVED: {model_name}, seed={seed}, epoch={epoch}",
            flush=True,
        )

        if (
            config.patience > 0
            and epoch >= 5
            and epochs_without_improvement >= config.patience
        ):
            print(
                f"EARLY STOP: {model_name}, seed={seed}, best_epoch={best_epoch}",
                flush=True,
            )
            break

    if not best_path.is_file():
        raise RuntimeError(f"No best checkpoint created for {model_name}, seed={seed}")
    best = safe_torch_load(best_path, device)
    model.load_state_dict(best["model_state_dict"], strict=True)
    return model


def aggregate_outputs(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    reliabilities: np.ndarray,
    gates: np.ndarray,
    effective: np.ndarray,
    aggregation_seconds: int,
):
    if "duration_seconds" in frame.columns:
        durations = pd.to_numeric(
            frame["duration_seconds"], errors="coerce"
        ).dropna().unique()
        base_seconds = int(round(float(durations[0]))) if len(durations) == 1 else 10
    else:
        base_seconds = 10
    if aggregation_seconds % base_seconds != 0:
        raise ValueError(
            f"Aggregation duration must be a multiple of {base_seconds} seconds."
        )
    count = aggregation_seconds // base_seconds

    working = frame.reset_index(drop=True).copy()
    working["_output_row"] = np.arange(len(working))
    records = []
    output_embeddings = []
    output_reliabilities = []
    output_gates = []
    output_effective = []

    group_columns = ["player_id", "session_id", "session_order", "role", "split"]
    for keys, group in working.groupby(group_columns, sort=False, dropna=False):
        group = group.sort_values("start_index").reset_index(drop=True)
        usable = (len(group) // count) * count
        for start in range(0, usable, count):
            block = group.iloc[start : start + count]
            rows = block["_output_row"].to_numpy(dtype=int)
            vector = embeddings[rows].mean(axis=0)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-12:
                continue
            output_embeddings.append((vector / norm).astype(np.float32))
            output_reliabilities.append(reliabilities[rows].mean(axis=0))
            output_gates.append(gates[rows].mean(axis=0))
            output_effective.append(float(np.mean(effective[rows])))
            records.append(
                {
                    "player_id": str(keys[0]),
                    "session_id": str(keys[1]),
                    "session_order": int(keys[2]),
                    "role": str(keys[3]),
                    "split": str(keys[4]),
                    "aggregation_seconds": int(aggregation_seconds),
                    "start_index": int(block["start_index"].iloc[0]),
                    "stop_index": int(block["stop_index"].iloc[-1]),
                    "window_count": int(len(block)),
                }
            )

    if not output_embeddings:
        empty = np.empty((0, embeddings.shape[1]), dtype=np.float32)
        return (
            empty,
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            pd.DataFrame(records),
        )
    return (
        np.stack(output_embeddings),
        np.stack(output_reliabilities),
        np.stack(output_gates),
        np.asarray(output_effective, dtype=np.float32),
        pd.DataFrame(records),
    )


def scaled_distances(
    embeddings: np.ndarray,
    effective: np.ndarray,
    prototypes: np.ndarray,
    gamma: float,
):
    similarities = embeddings @ prototypes.T
    raw_distances = 1.0 - similarities
    scale = np.maximum(effective, 1e-4) ** float(gamma)
    distances = raw_distances / scale[:, None]
    predictions = np.argmin(distances, axis=1)
    minimum = distances[np.arange(len(distances)), predictions]
    raw_minimum = raw_distances[np.arange(len(raw_distances)), predictions]
    return (
        minimum.astype(np.float32),
        predictions.astype(int),
        distances.astype(np.float32),
        raw_minimum.astype(np.float32),
    )


def macro_rate(values: np.ndarray, users: np.ndarray) -> float:
    rates = [
        float(np.mean(values[users == user])) for user in np.unique(users)
    ]
    return float(np.mean(rates)) if rates else float("nan")


def choose_threshold(
    known_distances: np.ndarray,
    known_users: np.ndarray,
    unknown_distances: np.ndarray,
    unknown_users: np.ndarray,
):
    values = np.unique(
        np.concatenate([known_distances, unknown_distances]).astype(np.float64)
    )
    candidates = np.concatenate(
        [
            [values[0] - 1e-8],
            (values[:-1] + values[1:]) / 2.0,
            [values[-1] + 1e-8],
        ]
    )
    best_key = None
    best = None
    for threshold in candidates:
        known_accept = known_distances <= threshold
        unknown_reject = unknown_distances > threshold
        known_macro = macro_rate(known_accept, known_users)
        unknown_macro = macro_rate(unknown_reject, unknown_users)
        balanced = 0.5 * (known_macro + unknown_macro)
        false_accept = 1.0 - unknown_macro
        key = (balanced, -false_accept, -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = (
                float(threshold),
                {
                    "known_acceptance_macro_user": known_macro,
                    "pseudo_unknown_rejection_macro_user": unknown_macro,
                    "balanced_detection": balanced,
                },
            )
    return best


def eer_value(labels_unknown: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels_unknown, scores)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def oscr_auc(
    known_correct: np.ndarray,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> float:
    thresholds = np.unique(np.concatenate([known_scores, unknown_scores]))
    fpr = [0.0]
    ccr = [0.0]
    for threshold in thresholds:
        fpr.append(float(np.mean(unknown_scores <= threshold)))
        ccr.append(float(np.mean(known_correct & (known_scores <= threshold))))
    order = np.argsort(fpr)
    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(np.asarray(ccr)[order], np.asarray(fpr)[order]))


def evaluate_open_set(
    known_embeddings,
    known_reliabilities,
    known_gates,
    known_effective,
    known_meta,
    unknown_embeddings,
    unknown_reliabilities,
    unknown_gates,
    unknown_effective,
    unknown_meta,
    prototypes,
    participant_to_class,
    threshold,
    gamma,
):
    known_distance, known_prediction, all_distances, known_raw_distance = (
        scaled_distances(
            known_embeddings, known_effective, prototypes, gamma
        )
    )
    unknown_distance, unknown_prediction, _, unknown_raw_distance = (
        scaled_distances(
            unknown_embeddings, unknown_effective, prototypes, gamma
        )
    )
    true_known = np.asarray(
        [participant_to_class[str(value)] for value in known_meta["player_id"]],
        dtype=int,
    )
    known_correct = known_prediction == true_known
    known_accept = known_distance <= threshold
    unknown_accept = unknown_distance <= threshold
    closed = class_metrics(true_known, known_prediction, len(participant_to_class))
    top5 = np.argsort(all_distances, axis=1)[:, : min(5, all_distances.shape[1])]
    top5_accuracy = float(
        np.mean([truth in choices for truth, choices in zip(true_known, top5)])
    )

    labels_unknown = np.concatenate(
        [
            np.zeros(len(known_distance), dtype=int),
            np.ones(len(unknown_distance), dtype=int),
        ]
    )
    scores = np.concatenate([known_distance, unknown_distance])

    metrics = {
        "closed_set_accuracy": closed["accuracy"],
        "closed_set_balanced_accuracy": closed["balanced_accuracy"],
        "closed_set_macro_f1": closed["macro_f1"],
        "closed_set_top5_accuracy": top5_accuracy,
        "gamma": float(gamma),
        "threshold": float(threshold),
        "known_acceptance_rate": float(np.mean(known_accept)),
        "known_false_rejection_rate": float(1.0 - np.mean(known_accept)),
        "unknown_false_acceptance_rate": float(np.mean(unknown_accept)),
        "unknown_true_rejection_rate": float(1.0 - np.mean(unknown_accept)),
        "known_correct_and_accepted_rate": float(
            np.mean(known_correct & known_accept)
        ),
        "known_accuracy_among_accepted": float(
            np.mean(known_correct[known_accept]) if np.any(known_accept) else 0.0
        ),
        "open_set_overall_accuracy": float(
            (
                np.sum(known_correct & known_accept)
                + np.sum(~unknown_accept)
            )
            / (len(known_accept) + len(unknown_accept))
        ),
        "unknown_detection_auroc": float(roc_auc_score(labels_unknown, scores)),
        "unknown_detection_aupr": float(
            average_precision_score(labels_unknown, scores)
        ),
        "equal_error_rate_approx": eer_value(labels_unknown, scores),
        "oscr_auc": oscr_auc(known_correct, known_distance, unknown_distance),
        "known_mean_scaled_distance": float(np.mean(known_distance)),
        "unknown_mean_scaled_distance": float(np.mean(unknown_distance)),
        "known_unknown_scaled_distance_gap": float(
            np.mean(unknown_distance) - np.mean(known_distance)
        ),
        "known_mean_raw_distance": float(np.mean(known_raw_distance)),
        "unknown_mean_raw_distance": float(np.mean(unknown_raw_distance)),
        "known_mean_effective_reliability": float(np.mean(known_effective)),
        "unknown_mean_effective_reliability": float(np.mean(unknown_effective)),
        "known_gate_entropy": float(
            np.mean(
                -np.sum(
                    known_gates * np.log(np.clip(known_gates, 1e-8, 1.0)),
                    axis=1,
                )
            )
        ),
        "unknown_gate_entropy": float(
            np.mean(
                -np.sum(
                    unknown_gates * np.log(np.clip(unknown_gates, 1e-8, 1.0)),
                    axis=1,
                )
            )
        ),
        "known_acceptance_rate_macro_user": macro_rate(
            known_accept, known_meta["player_id"].astype(str).to_numpy()
        ),
        "unknown_false_acceptance_rate_macro_user": macro_rate(
            unknown_accept, unknown_meta["player_id"].astype(str).to_numpy()
        ),
    }
    for modality_index, modality in enumerate(MODALITIES):
        metrics[f"known_mean_{modality}_reliability"] = float(
            np.mean(known_reliabilities[:, modality_index])
        )
        metrics[f"unknown_mean_{modality}_reliability"] = float(
            np.mean(unknown_reliabilities[:, modality_index])
        )
        metrics[f"known_mean_{modality}_gate"] = float(
            np.mean(known_gates[:, modality_index])
        )
        metrics[f"unknown_mean_{modality}_gate"] = float(
            np.mean(unknown_gates[:, modality_index])
        )

    rows = []
    for index, row in known_meta.reset_index(drop=True).iterrows():
        item = {
            **row.to_dict(),
            "known_or_unknown": "known",
            "true_class": int(true_known[index]),
            "predicted_class": int(known_prediction[index]),
            "scaled_distance": float(known_distance[index]),
            "raw_distance": float(known_raw_distance[index]),
            "effective_reliability": float(known_effective[index]),
            "accepted": bool(known_accept[index]),
            "correct": bool(known_correct[index]),
        }
        for modality_index, modality in enumerate(MODALITIES):
            item[f"{modality}_reliability"] = float(
                known_reliabilities[index, modality_index]
            )
            item[f"{modality}_gate"] = float(known_gates[index, modality_index])
        rows.append(item)
    for index, row in unknown_meta.reset_index(drop=True).iterrows():
        item = {
            **row.to_dict(),
            "known_or_unknown": "unknown",
            "true_class": -1,
            "predicted_class": int(unknown_prediction[index]),
            "scaled_distance": float(unknown_distance[index]),
            "raw_distance": float(unknown_raw_distance[index]),
            "effective_reliability": float(unknown_effective[index]),
            "accepted": bool(unknown_accept[index]),
            "correct": bool(not unknown_accept[index]),
        }
        for modality_index, modality in enumerate(MODALITIES):
            item[f"{modality}_reliability"] = float(
                unknown_reliabilities[index, modality_index]
            )
            item[f"{modality}_gate"] = float(unknown_gates[index, modality_index])
        rows.append(item)
    return metrics, pd.DataFrame(rows)


def participant_ids_by_fold(split: pd.DataFrame, fold: int):
    enrolled = split[split["role"] == "enrolled"].copy()
    included = sorted(
        enrolled.loc[enrolled["calibration_fold"] != fold, "player_id"].astype(str)
    )
    heldout = sorted(
        enrolled.loc[enrolled["calibration_fold"] == fold, "player_id"].astype(str)
    )
    if not included or not heldout:
        raise RuntimeError(f"Calibration fold {fold} has empty included/heldout users.")
    return included, heldout


def rows_for_players(
    index: pd.DataFrame,
    players: Sequence[str],
    *,
    session_order: int,
    split_names: Sequence[str],
):
    return index[
        index["player_id"].isin(players)
        & index["session_order"].eq(session_order)
        & index["split"].isin([name.lower() for name in split_names])
        & index["eligible_primary"]
    ].copy()


def calibrate_seed(
    seed,
    split,
    index,
    means,
    stds,
    raw_reliability,
    natural_targets,
    rel_median,
    rel_scale,
    config,
    output,
    device,
):
    fold_rows = []
    for fold in config.folds:
        included, heldout = participant_ids_by_fold(split, fold)
        train_frame = rows_for_players(
            index, included, session_order=1, split_names=("train",)
        )
        known_validation = rows_for_players(
            index, included, session_order=1, split_names=("validation",)
        )
        pseudo_unknown = rows_for_players(
            index, heldout, session_order=1, split_names=("validation",)
        )
        model = train_or_resume_model(
            model_name=f"calibration_fold_{fold}",
            seed=seed,
            train_frame=train_frame,
            validation_frame=known_validation,
            participant_ids=included,
            means=means,
            stds=stds,
            raw_reliability=raw_reliability,
            natural_targets=natural_targets,
            rel_median=rel_median,
            rel_scale=rel_scale,
            config=config,
            output=output,
            device=device,
        )
        participant_to_class = {
            participant: class_index
            for class_index, participant in enumerate(included)
        }
        prototypes = (
            nn.functional.normalize(model.prototypes, dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        known_outputs = infer_outputs(
            model,
            known_validation,
            means,
            stds,
            raw_reliability,
            natural_targets,
            rel_median,
            rel_scale,
            config.batch_size,
            config.num_workers,
            device,
            config.amp,
        )
        unknown_outputs = infer_outputs(
            model,
            pseudo_unknown,
            means,
            stds,
            raw_reliability,
            natural_targets,
            rel_median,
            rel_scale,
            config.batch_size,
            config.num_workers,
            device,
            config.amp,
        )

        for duration in config.aggregation_seconds:
            known_agg = aggregate_outputs(
                known_validation, *known_outputs[:4], duration
            )
            unknown_agg = aggregate_outputs(
                pseudo_unknown, *unknown_outputs[:4], duration
            )
            (
                known_embeddings,
                known_reliabilities,
                known_gates,
                known_effective,
                known_meta,
            ) = known_agg
            (
                unknown_embeddings,
                unknown_reliabilities,
                unknown_gates,
                unknown_effective,
                unknown_meta,
            ) = unknown_agg

            for gamma in config.gamma_candidates:
                known_distance, known_prediction, _, _ = scaled_distances(
                    known_embeddings, known_effective, prototypes, gamma
                )
                unknown_distance, _, _, _ = scaled_distances(
                    unknown_embeddings, unknown_effective, prototypes, gamma
                )
                known_true = np.asarray(
                    [
                        participant_to_class[str(value)]
                        for value in known_meta["player_id"]
                    ],
                    dtype=int,
                )
                threshold, calibration = choose_threshold(
                    known_distance,
                    known_meta["player_id"].astype(str).to_numpy(),
                    unknown_distance,
                    unknown_meta["player_id"].astype(str).to_numpy(),
                )
                fold_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "aggregation_seconds": duration,
                        "gamma": gamma,
                        "included_users": len(included),
                        "pseudo_unknown_users": len(heldout),
                        "known_blocks": len(known_meta),
                        "pseudo_unknown_blocks": len(unknown_meta),
                        "threshold": threshold,
                        "known_closed_set_accuracy": float(
                            np.mean(known_prediction == known_true)
                        ),
                        "known_mean_effective_reliability": float(
                            np.mean(known_effective)
                        ),
                        "pseudo_unknown_mean_effective_reliability": float(
                            np.mean(unknown_effective)
                        ),
                        **calibration,
                    }
                )

    fold_frame = pd.DataFrame(fold_rows)
    selected = {}
    for duration, duration_frame in fold_frame.groupby("aggregation_seconds"):
        gamma_summary = (
            duration_frame.groupby("gamma", as_index=False)
            .agg(
                balanced_detection=("balanced_detection", "mean"),
                known_acceptance=("known_acceptance_macro_user", "mean"),
                pseudo_unknown_rejection=(
                    "pseudo_unknown_rejection_macro_user",
                    "mean",
                ),
            )
            .sort_values(
                [
                    "balanced_detection",
                    "pseudo_unknown_rejection",
                    "known_acceptance",
                    "gamma",
                ],
                ascending=[False, False, False, True],
            )
        )
        gamma = float(gamma_summary.iloc[0]["gamma"])
        rows = duration_frame[
            np.isclose(duration_frame["gamma"], gamma)
        ]
        selected[int(duration)] = {
            "gamma": gamma,
            "threshold": float(np.median(rows["threshold"])),
            "mean_calibration_balanced_detection": float(
                rows["balanced_detection"].mean()
            ),
            "mean_calibration_known_acceptance": float(
                rows["known_acceptance_macro_user"].mean()
            ),
            "mean_calibration_pseudo_unknown_rejection": float(
                rows["pseudo_unknown_rejection_macro_user"].mean()
            ),
        }
    return selected, fold_frame


def final_seed_evaluation(
    seed,
    selected_calibration,
    split,
    index,
    means,
    stds,
    raw_reliability,
    natural_targets,
    rel_median,
    rel_scale,
    config,
    output,
    device,
):
    enrolled = sorted(
        split.loc[split["role"] == "enrolled", "player_id"].astype(str)
    )
    unknown = sorted(
        split.loc[
            split["role"].isin(["unknown", "final_unknown"]), "player_id"
        ].astype(str)
    )
    train_frame = rows_for_players(
        index, enrolled, session_order=1, split_names=("train",)
    )
    validation_frame = rows_for_players(
        index, enrolled, session_order=1, split_names=("validation",)
    )
    known_test = index[
        index["player_id"].isin(enrolled)
        & index["session_order"].eq(2)
        & index["eligible_primary"]
    ].copy()
    unknown_test = index[
        index["player_id"].isin(unknown)
        & index["session_order"].eq(2)
        & index["eligible_primary"]
    ].copy()

    model = train_or_resume_model(
        model_name="final",
        seed=seed,
        train_frame=train_frame,
        validation_frame=validation_frame,
        participant_ids=enrolled,
        means=means,
        stds=stds,
        raw_reliability=raw_reliability,
        natural_targets=natural_targets,
        rel_median=rel_median,
        rel_scale=rel_scale,
        config=config,
        output=output,
        device=device,
    )
    participant_to_class = {
        participant: class_index
        for class_index, participant in enumerate(enrolled)
    }
    prototypes = (
        nn.functional.normalize(model.prototypes, dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    known_outputs = infer_outputs(
        model,
        known_test,
        means,
        stds,
        raw_reliability,
        natural_targets,
        rel_median,
        rel_scale,
        config.batch_size,
        config.num_workers,
        device,
        config.amp,
    )
    unknown_outputs = infer_outputs(
        model,
        unknown_test,
        means,
        stds,
        raw_reliability,
        natural_targets,
        rel_median,
        rel_scale,
        config.batch_size,
        config.num_workers,
        device,
        config.amp,
    )

    metric_rows = []
    prediction_rows = []
    for duration in config.aggregation_seconds:
        known_agg = aggregate_outputs(
            known_test, *known_outputs[:4], duration
        )
        unknown_agg = aggregate_outputs(
            unknown_test, *unknown_outputs[:4], duration
        )
        calibration = selected_calibration[int(duration)]
        metrics, predictions = evaluate_open_set(
            *known_agg[:4],
            known_agg[4],
            *unknown_agg[:4],
            unknown_agg[4],
            prototypes,
            participant_to_class,
            calibration["threshold"],
            calibration["gamma"],
        )
        metric_rows.append(
            {
                "method": METHOD,
                "seed": seed,
                "aggregation_seconds": duration,
                **calibration,
                **metrics,
            }
        )
        predictions.insert(0, "seed", seed)
        predictions.insert(1, "method", METHOD)
        prediction_rows.append(predictions)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def summarize_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    identifiers = {
        "method",
        "seed",
        "aggregation_seconds",
    }
    numeric = [
        column
        for column in seed_metrics.columns
        if column not in identifiers
        and pd.api.types.is_numeric_dtype(seed_metrics[column])
    ]
    rows = []
    for (method, duration), group in seed_metrics.groupby(
        ["method", "aggregation_seconds"]
    ):
        for metric in numeric:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            sem = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            rows.append(
                {
                    "method": method,
                    "aggregation_seconds": int(duration),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": std,
                    "ci95_half_width": 1.96 * sem,
                    "count": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def compare_to_prototype(
    full_metrics: pd.DataFrame,
    prototype_output: Path,
):
    baseline_path = prototype_output / "seed_metrics.csv"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline = pd.read_csv(baseline_path)
    key_columns = ["seed", "aggregation_seconds"]
    merged = full_metrics.merge(
        baseline,
        on=key_columns,
        suffixes=("_rg_posr", "_prototype"),
        how="inner",
    )
    metrics = [
        "closed_set_accuracy",
        "closed_set_macro_f1",
        "known_acceptance_rate",
        "known_false_rejection_rate",
        "unknown_false_acceptance_rate",
        "unknown_detection_auroc",
        "unknown_detection_aupr",
        "equal_error_rate_approx",
        "oscr_auc",
        "open_set_overall_accuracy",
    ]
    rows = []
    for _, row in merged.iterrows():
        item = {
            "seed": int(row["seed"]),
            "aggregation_seconds": int(row["aggregation_seconds"]),
        }
        for metric in metrics:
            full_column = f"{metric}_rg_posr"
            base_column = f"{metric}_prototype"
            if full_column in merged.columns and base_column in merged.columns:
                item[f"{metric}_rg_posr"] = float(row[full_column])
                item[f"{metric}_prototype"] = float(row[base_column])
                item[f"{metric}_delta"] = float(
                    row[full_column] - row[base_column]
                )
        rows.append(item)
    paired = pd.DataFrame(rows)

    summary_rows = []
    for duration, group in paired.groupby("aggregation_seconds"):
        for metric in metrics:
            delta_column = f"{metric}_delta"
            if delta_column not in group.columns:
                continue
            values = group[delta_column].dropna()
            summary_rows.append(
                {
                    "aggregation_seconds": int(duration),
                    "metric": metric,
                    "mean_delta_rg_posr_minus_prototype": float(values.mean()),
                    "std_delta": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "improved_seed_count": int(
                        np.sum(
                            values > 0
                            if metric
                            not in {
                                "known_false_rejection_rate",
                                "unknown_false_acceptance_rate",
                                "equal_error_rate_approx",
                            }
                            else values < 0
                        )
                    ),
                    "seed_count": int(len(values)),
                }
            )
    return paired, pd.DataFrame(summary_rows)


def run(config: RunConfig, overwrite: bool = False):
    started = time.time()
    processed_root = Path(config.processed_root).expanduser().resolve()
    frozen_root = Path(config.frozen_root).expanduser().resolve()
    output = Path(config.output_root).expanduser().resolve()
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    device = choose_device(config.device)
    if config.torch_threads > 0:
        torch.set_num_threads(config.torch_threads)

    (
        split,
        index,
        means,
        stds,
        raw_reliability,
        standardized_reliability,
        natural_targets,
        feature_names,
        rel_median,
        rel_scale,
    ) = load_protocol(processed_root, frozen_root, config.strict_protocol)

    if config.quick:
        enrolled = sorted(
            split.loc[split["role"] == "enrolled", "player_id"].astype(str)
        )[:8]
        unknown = sorted(
            split.loc[
                split["role"].isin(["unknown", "final_unknown"]), "player_id"
            ].astype(str)
        )[:3]
        selected_players = set(enrolled + unknown)
        split = split[split["player_id"].isin(selected_players)].copy()
        index = index[index["player_id"].isin(selected_players)].copy()
        split.loc[split["role"] == "enrolled", "calibration_fold"] = [
            value % 2 for value in range(int((split["role"] == "enrolled").sum()))
        ]
        fold_map = split.set_index("player_id")["calibration_fold"].to_dict()
        index["calibration_fold"] = [
            int(fold_map.get(player, -1)) for player in index["player_id"]
        ]

    save_json(output / "run_config.json", asdict(config))
    all_metrics = []
    all_calibration = []
    all_predictions = []

    for seed in config.seeds:
        metrics_path = output / f"seed_{seed}_metrics.csv"
        predictions_path = output / f"seed_{seed}_predictions.csv"
        calibration_path = output / f"seed_{seed}_calibration.csv"
        selected_path = output / f"seed_{seed}_selected_calibration.json"

        if (
            metrics_path.is_file()
            and predictions_path.is_file()
            and calibration_path.is_file()
            and selected_path.is_file()
        ):
            print(f"SKIPPING COMPLETED SEED {seed}", flush=True)
            seed_metrics = pd.read_csv(metrics_path)
            predictions = pd.read_csv(predictions_path)
            calibration = pd.read_csv(calibration_path)
        else:
            selected_calibration, calibration = calibrate_seed(
                seed,
                split,
                index,
                means,
                stds,
                raw_reliability,
                natural_targets,
                rel_median,
                rel_scale,
                config,
                output,
                device,
            )
            save_json(selected_path, selected_calibration)
            seed_metrics, predictions = final_seed_evaluation(
                seed,
                selected_calibration,
                split,
                index,
                means,
                stds,
                raw_reliability,
                natural_targets,
                rel_median,
                rel_scale,
                config,
                output,
                device,
            )
            save_csv(metrics_path, seed_metrics)
            save_csv(predictions_path, predictions)
            save_csv(calibration_path, calibration)

        all_metrics.append(seed_metrics)
        all_predictions.append(predictions)
        all_calibration.append(calibration)

    seed_metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    calibration = pd.concat(all_calibration, ignore_index=True)
    summary = summarize_metrics(seed_metrics)

    save_csv(output / "seed_metrics.csv", seed_metrics)
    save_csv(output / "metrics_summary.csv", summary)
    save_csv(output / "calibration_gamma_threshold_summary.csv", calibration)
    save_csv(output / "test_predictions.csv", predictions)

    paired_rows = 0
    if config.prototype_output:
        prototype_output = Path(config.prototype_output).expanduser().resolve()
        paired, paired_summary = compare_to_prototype(
            seed_metrics, prototype_output
        )
        save_csv(output / "paired_prototype_comparison.csv", paired)
        save_csv(output / "paired_prototype_comparison_summary.csv", paired_summary)
        paired_rows = len(paired)

    report = {
        "script_version": VERSION,
        "method": METHOD,
        "title": (
            "Reliability-Gated Prototypical Open-Set Recognition for "
            "Cross-Session Behavioural Biometrics in Extended Reality"
        ),
        "processed_root": str(processed_root),
        "frozen_root": str(frozen_root),
        "prototype_output": config.prototype_output,
        "output_root": str(output),
        "device": str(device),
        "seeds": list(config.seeds),
        "calibration_folds": list(config.folds),
        "aggregation_seconds": list(config.aggregation_seconds),
        "gamma_candidates": list(config.gamma_candidates),
        "models_per_seed": len(config.folds) + 1,
        "identity_dimensions": EXPECTED_DIMS,
        "reliability_dimensions": RELIABILITY_DIMS,
        "reliability_total_dimension": 25,
        "reliability_features_used": True,
        "reliability_gated_fusion_used": True,
        "reliability_scaled_rejection_used": True,
        "synthetic_corruption_supervision_used": True,
        "final_unknown_used_for_calibration": False,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "metric_rows": len(seed_metrics),
        "paired_comparison_rows": paired_rows,
        "metrics": seed_metrics.to_dict(orient="records"),
    }
    save_json(output / "experiment_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("FULL RG-POSR EXPERIMENT COMPLETED")
    return report


def synthetic_environment(root: Path):
    processed = root / "processed"
    frozen = root / "frozen"
    cache_dir = processed / "resampled_sessions"
    cache_dir.mkdir(parents=True)
    (frozen / "01_features").mkdir(parents=True)
    (frozen / "04_cache").mkdir(parents=True)

    enrolled_count = 4
    unknown_count = 2
    participants = []
    index_rows = []
    raw_reliability_rows = []
    target_rows = []
    rng = np.random.default_rng(7)
    window_id = 0

    schema = {
        modality: [f"{modality}_{i}" for i in range(dimension)]
        for modality, dimension in EXPECTED_DIMS.items()
    }
    stats = {
        modality: {
            "feature_names": schema[modality],
            "mean": [0.0] * dimension,
            "std": [1.0] * dimension,
            "count": [1000] * dimension,
        }
        for modality, dimension in EXPECTED_DIMS.items()
    }
    (processed / "feature_schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )
    (processed / "normalization_stats.json").write_text(
        json.dumps(stats), encoding="utf-8"
    )

    for player_index in range(enrolled_count + unknown_count):
        player = f"{player_index + 1:02d}"
        role = "enrolled" if player_index < enrolled_count else "unknown"
        fold = player_index % 2 if role == "enrolled" else -1
        participants.append(
            {
                "player_id": player,
                "role": role,
                "development_session": "s1",
                "test_session": "s2",
                "calibration_fold": fold,
            }
        )
        for session_order, session in ((1, "s1"), (2, "s2")):
            if role == "unknown" and session_order == 1:
                continue
            frame_count = 120
            arrays = {}
            phase = player_index * 0.8 + session_order * 0.08
            for modality, dimension in EXPECTED_DIMS.items():
                time_axis = np.arange(frame_count, dtype=np.float32)[:, None]
                values = np.repeat(
                    np.sin(time_axis * 0.08 + phase), dimension, axis=1
                )
                values += rng.normal(0, 0.05, size=values.shape)
                arrays[modality] = values.astype(np.float32)
            cache = cache_dir / f"player_{player}_{session}.npz"
            np.savez_compressed(
                cache,
                time_seconds=np.arange(frame_count) / 15.0,
                quality=np.ones((frame_count, 4), dtype=np.float32),
                **arrays,
            )
            splits = (
                [("train", 0, 60), ("validation", 60, 120)]
                if session_order == 1
                else [
                    (
                        "test_known" if role == "enrolled" else "test_unknown",
                        0,
                        120,
                    )
                ]
            )
            for split_name, start_all, stop_all in splits:
                for start in range(start_all, stop_all, 15):
                    stop = start + 15
                    reliability = np.concatenate(
                        [
                            np.asarray(
                                [1.0, 0.95, 0.05, 0.02, 0.05, 0.01],
                                dtype=np.float32,
                            )
                            for _ in range(3)
                        ]
                        + [
                            np.asarray(
                                [1.0, 0.95, 0.05, 0.02, 0.02, 0.05, 0.01],
                                dtype=np.float32,
                            )
                        ]
                    )
                    reliability += rng.normal(
                        0, 0.01, size=reliability.shape
                    ).astype(np.float32)
                    raw_reliability_rows.append(reliability)
                    target_rows.append(
                        np.clip(
                            rng.normal(0.95, 0.02, size=4),
                            0,
                            1,
                        ).astype(np.float32)
                    )
                    index_rows.append(
                        {
                            "window_id": f"w{window_id}",
                            "player_id": player,
                            "role": role,
                            "session_id": session,
                            "session_order": session_order,
                            "split": split_name,
                            "calibration_fold": fold,
                            "duration_seconds": 1,
                            "start_index": start,
                            "stop_index": stop,
                            "eligible_primary": True,
                            "eligible_motion": True,
                            "eligible_gaze": True,
                            "resolved_cache_path": str(cache),
                        }
                    )
                    window_id += 1

    split_frame = pd.DataFrame(participants)
    split_frame.to_csv(
        processed / "participant_split_manifest.csv", index=False
    )
    index_frame = pd.DataFrame(index_rows)
    index_frame.to_csv(processed / "window_manifest.csv", index=False)
    index_frame.to_csv(
        frozen / "04_cache/reliability_window_index_10s.csv", index=False
    )

    raw_reliability = np.stack(raw_reliability_rows).astype(np.float32)
    targets = np.stack(target_rows).astype(np.float32)
    median = np.median(raw_reliability, axis=0).astype(np.float32)
    scale = np.std(raw_reliability, axis=0).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    standardized = (raw_reliability - median) / scale
    feature_names = []
    for modality in ("head", "left", "right"):
        feature_names.extend(
            f"{modality}__feature_{index}" for index in range(6)
        )
    feature_names.extend(f"gaze__feature_{index}" for index in range(7))
    np.savez_compressed(
        frozen / "04_cache/reliability_window_features_10s.npz",
        raw_features=raw_reliability,
        standardized_features=standardized,
        natural_quality_targets=targets,
        training_reference_mask=np.ones(len(index_frame), dtype=np.uint8),
        feature_names=np.asarray(feature_names, dtype="U64"),
        modality_names=np.asarray(MODALITIES, dtype="U16"),
        window_ids=index_frame["window_id"].astype(str).to_numpy(dtype="U64"),
    )
    (
        frozen / "01_features/reliability_descriptor_normalization_v1.json"
    ).write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "median": median.tolist(),
                "scale": scale.tolist(),
            }
        ),
        encoding="utf-8",
    )
    source_hashes = {
        "feature_schema.json": sha256(processed / "feature_schema.json"),
        "normalization_stats.json": sha256(
            processed / "normalization_stats.json"
        ),
        "participant_split_manifest.csv": sha256(
            processed / "participant_split_manifest.csv"
        ),
        "window_manifest.csv": sha256(processed / "window_manifest.csv"),
    }
    (
        frozen / "01_features/source_checksums_v1.json"
    ).write_text(json.dumps(source_hashes), encoding="utf-8")
    (frozen / "freeze_report.json").write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "participant_split": {
                    "participants": enrolled_count + unknown_count,
                    "enrolled": enrolled_count,
                    "final_unknown": unknown_count,
                },
            }
        ),
        encoding="utf-8",
    )
    (frozen / "validation_report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    return processed, frozen


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="rg_posr_full_test_") as folder:
        root = Path(folder)
        processed, frozen = synthetic_environment(root)
        config = RunConfig(
            processed_root=str(processed),
            frozen_root=str(frozen),
            prototype_output=None,
            output_root=str(root / "output"),
            seeds=(42,),
            folds=(0, 1),
            aggregation_seconds=(2, 4),
            gamma_candidates=(0.0, 1.0),
            epochs=2,
            batch_size=8,
            learning_rate=0.003,
            weight_decay=0.0,
            hidden_size=8,
            embedding_size=8,
            num_layers=1,
            dropout=0.0,
            reliability_hidden_size=8,
            arcface_scale=8.0,
            arcface_margin=0.2,
            lambda_reliability=1.0,
            lambda_gate_alignment=0.5,
            lambda_monotonic=0.5,
            monotonic_margin=0.1,
            corruption_probability=0.5,
            patience=0,
            num_workers=0,
            torch_threads=1,
            device="cpu",
            amp=False,
            quick=False,
            strict_protocol=False,
        )
        report = run(config, overwrite=True)
        output = Path(report["output_root"])
        required = [
            "seed_metrics.csv",
            "metrics_summary.csv",
            "calibration_gamma_threshold_summary.csv",
            "test_predictions.csv",
            "experiment_report.json",
        ]
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise AssertionError(f"Missing self-test outputs: {missing}")
        metrics = pd.read_csv(output / "seed_metrics.csv")
        if len(metrics) != 2:
            raise AssertionError("Expected two duration rows.")
        if not set(metrics["aggregation_seconds"]) == {2, 4}:
            raise AssertionError("Aggregation durations are incorrect.")
        for column in (
            "unknown_detection_auroc",
            "known_mean_effective_reliability",
            "known_mean_head_gate",
            "gamma",
        ):
            if column not in metrics.columns:
                raise AssertionError(f"Missing metric {column}")
            if not np.isfinite(metrics[column]).all():
                raise AssertionError(f"Non-finite metric {column}")
    print("SELF-TEST PASSED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train full reliability-gated prototypical open-set recognition."
    )
    parser.add_argument("--processed-root")
    parser.add_argument("--frozen-root")
    parser.add_argument("--prototype-output")
    parser.add_argument("--output")
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--aggregation-seconds", nargs="*", default=None)
    parser.add_argument("--gamma-candidates", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--reliability-hidden-size", type=int, default=32)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    parser.add_argument("--lambda-reliability", type=float, default=1.0)
    parser.add_argument("--lambda-gate-alignment", type=float, default=0.5)
    parser.add_argument("--lambda-monotonic", type=float, default=0.5)
    parser.add_argument("--monotonic-margin", type=float, default=0.15)
    parser.add_argument("--corruption-probability", type=float, default=0.6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    missing = [
        name
        for name, value in (
            ("--processed-root", args.processed_root),
            ("--frozen-root", args.frozen_root),
            ("--output", args.output),
        )
        if not value
    ]
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join(missing))

    if args.quick:
        seeds = (42,)
        folds = (0, 1)
        aggregation_seconds = (30,)
        gamma_candidates = (0.0, 1.0)
        epochs = min(args.epochs, 2)
        batch_size = min(args.batch_size, 16)
        hidden_size = min(args.hidden_size, 16)
        embedding_size = min(args.embedding_size, 16)
        reliability_hidden_size = min(args.reliability_hidden_size, 8)
        patience = 0
        strict_protocol = False
    else:
        seeds = parse_int_list(args.seeds, (42, 52, 62, 72, 82))
        folds = parse_int_list(args.folds, (0, 1, 2, 3, 4))
        aggregation_seconds = parse_int_list(
            args.aggregation_seconds, (30, 60, 120)
        )
        gamma_candidates = parse_float_list(
            args.gamma_candidates, GAMMA_DEFAULTS
        )
        epochs = args.epochs
        batch_size = args.batch_size
        hidden_size = args.hidden_size
        embedding_size = args.embedding_size
        reliability_hidden_size = args.reliability_hidden_size
        patience = args.patience
        strict_protocol = True

    config = RunConfig(
        processed_root=args.processed_root,
        frozen_root=args.frozen_root,
        prototype_output=args.prototype_output,
        output_root=args.output,
        seeds=seeds,
        folds=folds,
        aggregation_seconds=aggregation_seconds,
        gamma_candidates=gamma_candidates,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        reliability_hidden_size=reliability_hidden_size,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        lambda_reliability=args.lambda_reliability,
        lambda_gate_alignment=args.lambda_gate_alignment,
        lambda_monotonic=args.lambda_monotonic,
        monotonic_margin=args.monotonic_margin,
        corruption_probability=args.corruption_probability,
        patience=patience,
        num_workers=args.num_workers,
        torch_threads=args.torch_threads,
        device=args.device,
        amp=not args.no_amp,
        quick=args.quick,
        strict_protocol=strict_protocol,
    )
    run(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

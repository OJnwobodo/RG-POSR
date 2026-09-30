#!/usr/bin/env python3
"""
Reviewer-revision RG-POSR training driver.

This driver reuses the frozen identity-learning, calibration, aggregation, and
open-set evaluation implementation from train_review_rg_posr_base.py while
replacing the historical circular reliability pathway with blind,
non-circular reliability descriptors and independent clean-to-corrupted
fidelity supervision.

Reviewer-requested variants
---------------------------
- uniform: uniform fusion; no reliability prediction
- direct_active_fraction: direct availability mapping
- linear: learned linear reliability heads
- nonlinear: learned nonlinear reliability heads

Corruption type and severity are never supplied to a predictor. The clean
reference is used only during training to construct the independent fidelity
target and is not required at inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from blind_reliability import (
    DESCRIPTOR_DIM,
    DESCRIPTOR_NAMES,
    MODALITIES,
    RELIABILITY_VARIANTS,
    ReliabilityEstimator,
    apply_blind_synthetic_corruption,
    reliability_gates,
    standardize_descriptors,
)


VERSION = "2.0.0-review"
LEARNED_VARIANTS = {"linear", "nonlinear"}
REVIEW_VARIANT = "nonlinear"
REVIEW_BLIND_CACHE_ROOT: Path | None = None


def _load_base_module():
    path = Path(__file__).resolve().with_name("train_review_rg_posr_base.py")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing base implementation:\n{path}\n"
            "Rename the historical working copy to train_review_rg_posr_base.py."
        )
    name = "rg_posr_review_base"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


def _require_blind_root() -> Path:
    if REVIEW_BLIND_CACHE_ROOT is None:
        raise RuntimeError("Blind reliability cache root has not been configured.")
    return REVIEW_BLIND_CACHE_ROOT


class ReviewTrainingDataset(Dataset):
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
        self.cache = base.NPZCache(capacity=4)

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
        raw_descriptors = torch.from_numpy(
            self.raw_reliability[reliability_row].astype(np.float32)
        )
        label = self.participant_to_class[str(row["player_id"])]
        # A placeholder preserves the historical tuple shape. It is never used
        # as a target in the revised training pathway.
        placeholder = torch.full((len(MODALITIES),), float("nan"), dtype=torch.float32)
        return (
            *sequences,
            raw_descriptors,
            placeholder,
            torch.tensor(label, dtype=torch.long),
        )


class ReviewInferenceDataset(Dataset):
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
        self.cache = base.NPZCache(capacity=4)

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
        raw_descriptors = torch.from_numpy(
            self.raw_reliability[reliability_row].astype(np.float32)
        )
        placeholder = torch.full((len(MODALITIES),), float("nan"), dtype=torch.float32)
        return (
            *sequences,
            raw_descriptors,
            placeholder,
            torch.tensor(index, dtype=torch.long),
        )


class ReviewRGPOSR(nn.Module):
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
        self.reliability_variant = REVIEW_VARIANT
        self.branches = nn.ModuleDict(
            {
                modality: base.GRUBranch(
                    base.EXPECTED_DIMS[modality],
                    hidden_size,
                    embedding_size,
                    num_layers,
                    dropout,
                )
                for modality in MODALITIES
            }
        )
        # Initialize prototypes before variant-specific heads so identity branch
        # and prototype initializations are matched across variants.
        self.prototypes = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.prototypes)
        self.reliability_estimator = ReliabilityEstimator(
            variant=self.reliability_variant,
            hidden_size=reliability_hidden_size,
            dropout=dropout,
        )

    def predict_reliability(
        self,
        raw_descriptors: torch.Tensor,
        standardized_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        return self.reliability_estimator(
            raw_descriptors,
            standardized_descriptors,
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
        sequence_map = {
            "head": head,
            "left": left,
            "right": right,
            "gaze": gaze,
        }
        modality_embeddings = torch.stack(
            [self.branches[m](sequence_map[m]) for m in MODALITIES],
            dim=1,
        )
        reliability = self.predict_reliability(
            raw_descriptors,
            standardized_descriptors,
        )
        alpha, effective = reliability_gates(reliability)
        fused = torch.sum(alpha.unsqueeze(2) * modality_embeddings, dim=1)
        fused = nn.functional.normalize(fused, dim=1)
        return fused, reliability, alpha, effective

    def cosine_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        prototypes = nn.functional.normalize(self.prototypes, dim=1)
        return embedding @ prototypes.t()

    def forward(
        self,
        head: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        gaze: torch.Tensor,
        raw_descriptors: torch.Tensor,
        standardized_descriptors: torch.Tensor,
    ):
        embedding, reliability, alpha, effective = self.encode(
            head,
            left,
            right,
            gaze,
            raw_descriptors,
            standardized_descriptors,
        )
        cosine = self.cosine_logits(embedding)
        return embedding, cosine, reliability, alpha, effective


def load_protocol_review(
    processed_root: Path,
    frozen_root: Path,
    strict: bool,
):
    blind_root = _require_blind_root()

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
        if strict and base.sha256(path) != source_hashes.get(name):
            raise RuntimeError(f"Frozen source checksum changed: {name}")

    split_frame = pd.read_csv(
        processed_root / "participant_split_manifest.csv", dtype=str
    )
    split_frame.columns = [base.normalized_name(c) for c in split_frame.columns]
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
    index.columns = [base.normalized_name(c) for c in index.columns]
    index["player_id"] = index["player_id"].astype(str).str.zfill(2)
    index["role"] = index["role"].astype(str).str.lower()
    index["split"] = index["split"].astype(str).str.lower()
    for column in ("session_order", "start_index", "stop_index", "calibration_fold"):
        index[column] = pd.to_numeric(index[column], errors="raise").astype(int)
    for column in ("eligible_primary", "eligible_motion", "eligible_gaze"):
        index[column] = base.parse_bool_series(index[column])
    index["_reliability_row"] = np.arange(len(index), dtype=int)

    cache_path = blind_root / "blind_reliability_window_features_10s.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as archive:
        raw_reliability = archive["raw_features"].astype(np.float32)
        standardized_reliability = archive["standardized_features"].astype(np.float32)
        training_reference_mask = archive["training_reference_mask"].astype(bool)
        descriptor_names = archive["descriptor_names"].astype(str).tolist()
        modality_names = archive["modality_names"].astype(str).tolist()
        window_ids = archive["window_ids"].astype(str)

    expected_shape = (len(index), len(MODALITIES), DESCRIPTOR_DIM)
    if raw_reliability.shape != expected_shape:
        raise RuntimeError(
            f"Expected blind reliability cache {expected_shape}, "
            f"observed {raw_reliability.shape}."
        )
    if standardized_reliability.shape != expected_shape:
        raise RuntimeError("Blind raw/standardized descriptor shapes differ.")
    if descriptor_names != list(DESCRIPTOR_NAMES):
        raise RuntimeError(
            f"Blind descriptor order mismatch: {descriptor_names}"
        )
    if modality_names != list(MODALITIES):
        raise RuntimeError(f"Blind modality order mismatch: {modality_names}")
    if not np.array_equal(
        index["window_id"].astype(str).to_numpy(), window_ids.astype(str)
    ):
        raise RuntimeError("Blind descriptor cache/window index misalignment.")

    expected_training_mask = (
        index["role"].eq("enrolled")
        & index["session_order"].eq(1)
        & index["split"].eq("train")
        & index["eligible_primary"]
    ).to_numpy(dtype=bool)
    if not np.array_equal(training_reference_mask, expected_training_mask):
        raise RuntimeError("Blind descriptor training-reference mask mismatch.")

    normalization = json.loads(
        (blind_root / "blind_reliability_normalization.json").read_text(
            encoding="utf-8"
        )
    )
    if normalization.get("target_included") is not False:
        raise RuntimeError("Blind normalization unexpectedly includes a target.")
    if normalization.get("corruption_type_or_severity_included") is not False:
        raise RuntimeError("Blind normalization includes corruption metadata.")
    rel_median = np.asarray(normalization["median"], dtype=np.float32)
    rel_scale = np.asarray(normalization["scale"], dtype=np.float32)
    if rel_median.shape != (len(MODALITIES), DESCRIPTOR_DIM):
        raise RuntimeError(f"Blind median shape mismatch: {rel_median.shape}")
    if rel_scale.shape != rel_median.shape:
        raise RuntimeError(f"Blind scale shape mismatch: {rel_scale.shape}")
    rel_scale = np.where(np.abs(rel_scale) > 1e-8, rel_scale, 1.0).astype(np.float32)

    stats = json.loads(
        (processed_root / "normalization_stats.json").read_text(encoding="utf-8")
    )
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        means[modality] = np.asarray(stats[modality]["mean"], dtype=np.float32)
        std = np.asarray(stats[modality]["std"], dtype=np.float32)
        stds[modality] = np.where(np.abs(std) > 1e-8, std, 1.0).astype(np.float32)
        if len(means[modality]) != base.EXPECTED_DIMS[modality]:
            raise RuntimeError(f"{modality} normalization dimension mismatch")

    # Placeholder only: no original natural target is read or used.
    target_placeholder = np.full(
        (len(index), len(MODALITIES)),
        np.nan,
        dtype=np.float32,
    )

    return (
        split_frame,
        index,
        means,
        stds,
        raw_reliability,
        standardized_reliability,
        target_placeholder,
        descriptor_names,
        rel_median,
        rel_scale,
    )


def train_epoch_review(
    model: ReviewRGPOSR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    config,
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
        "mean_fidelity_target": 0.0,
    }
    count = 0
    criterion = nn.CrossEntropyLoss()
    learned_variant = model.reliability_variant in LEARNED_VARIANTS

    for batch_index, batch in enumerate(loader):
        head, left, right, gaze, raw_clean, _, labels = batch
        sequence_map = {
            "head": head.to(device, non_blocking=True),
            "left": left.to(device, non_blocking=True),
            "right": right.to(device, non_blocking=True),
            "gaze": gaze.to(device, non_blocking=True),
        }
        raw_clean = raw_clean.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_standardized = standardize_descriptors(
            raw_clean,
            rel_median,
            rel_scale,
        )

        corruption_seed = seed * 100000000 + epoch * 100000 + batch_index
        corrupted, corrupted_raw, fidelity_targets, _, _ = (
            apply_blind_synthetic_corruption(
                sequence_map,
                probability=config.corruption_probability,
                seed=corruption_seed,
            )
        )
        corrupted_standardized = standardize_descriptors(
            corrupted_raw,
            rel_median,
            rel_scale,
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
                corrupted_raw,
                corrupted_standardized,
            )
            logits = base.arcface_logits(
                cosine,
                labels,
                config.arcface_scale,
                config.arcface_margin,
            )
            classification_loss = criterion(logits, labels)

            zero = classification_loss.new_zeros(())
            if learned_variant:
                reliability_loss = nn.functional.smooth_l1_loss(
                    reliability,
                    fidelity_targets,
                )
                target_alpha = fidelity_targets.clamp_min(0.0)
                target_alpha = target_alpha / target_alpha.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-6)
                gate_alignment_loss = nn.functional.mse_loss(alpha, target_alpha)

                clean_reliability = model.predict_reliability(
                    raw_clean,
                    clean_standardized,
                )
                target_drop = (1.0 - fidelity_targets).clamp(0.0, 1.0)
                active = target_drop > 1e-6
                ranking = torch.relu(
                    config.monotonic_margin * target_drop
                    - (clean_reliability - reliability)
                )
                monotonic_loss = (
                    ranking[active].mean() if torch.any(active) else zero
                )
            else:
                reliability_loss = zero
                gate_alignment_loss = zero
                monotonic_loss = zero

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
        totals["mean_gate_entropy"] += float(base.gate_entropy(alpha).mean().item()) * batch_count
        totals["mean_fidelity_target"] += float(fidelity_targets.mean().item()) * batch_count

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def infer_outputs_review(
    model: ReviewRGPOSR,
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
    dataset = ReviewInferenceDataset(
        frame,
        means,
        stds,
        raw_reliability,
        natural_targets,
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
    reliabilities = np.empty((len(frame), len(MODALITIES)), dtype=np.float32)
    gates = np.empty((len(frame), len(MODALITIES)), dtype=np.float32)
    effective = np.empty(len(frame), dtype=np.float32)
    target_placeholder = np.full(
        (len(frame), len(MODALITIES)),
        np.nan,
        dtype=np.float32,
    )

    median_t = torch.from_numpy(rel_median).to(device)
    scale_t = torch.from_numpy(rel_scale).to(device)
    model.eval()

    for batch in loader:
        head, left, right, gaze, raw_descriptors, _, indices = batch
        head = head.to(device, non_blocking=True)
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        gaze = gaze.to(device, non_blocking=True)
        raw_descriptors = raw_descriptors.to(device, non_blocking=True)
        standardized = standardize_descriptors(
            raw_descriptors,
            median_t,
            scale_t,
        )
        with torch.autocast(
            device_type=device.type,
            enabled=(amp and device.type == "cuda"),
        ):
            embedding, reliability, alpha, rho = model.encode(
                head,
                left,
                right,
                gaze,
                raw_descriptors,
                standardized,
            )
        rows = indices.numpy()
        embeddings[rows] = embedding.detach().cpu().numpy().astype(np.float32)
        reliabilities[rows] = reliability.detach().cpu().numpy().astype(np.float32)
        gates[rows] = alpha.detach().cpu().numpy().astype(np.float32)
        effective[rows] = rho.detach().cpu().numpy().astype(np.float32)

    return embeddings, reliabilities, gates, effective, target_placeholder


def validation_selection_metrics_review(
    model,
    frame,
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
    embeddings, reliabilities, gates, effective, _ = infer_outputs_review(
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
    metrics = base.class_metrics(
        true_indices,
        predictions,
        len(participant_to_class),
    )
    metrics.update(
        {
            "mean_effective_reliability": float(np.mean(effective)),
            "mean_predicted_reliability": float(np.mean(reliabilities)),
            "mean_gate_entropy": float(
                np.mean(
                    -np.sum(
                        gates * np.log(np.clip(gates, 1e-8, 1.0)),
                        axis=1,
                    )
                )
            ),
        }
    )
    return metrics


def install_review_pathway(variant: str, blind_cache_root: Path) -> None:
    global REVIEW_VARIANT, REVIEW_BLIND_CACHE_ROOT

    if variant not in RELIABILITY_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}; choose from {RELIABILITY_VARIANTS}."
        )
    REVIEW_VARIANT = variant
    REVIEW_BLIND_CACHE_ROOT = blind_cache_root.expanduser().resolve()

    base.VERSION = VERSION
    base.METHOD = f"review_{variant}_rg_posr"
    base.MODALITIES = MODALITIES
    base.RELIABILITY_DIMS = {modality: DESCRIPTOR_DIM for modality in MODALITIES}
    base.TrainingDataset = ReviewTrainingDataset
    base.InferenceDataset = ReviewInferenceDataset
    base.FullRGPOSR = ReviewRGPOSR
    base.load_protocol = load_protocol_review
    base.train_epoch = train_epoch_review
    base.infer_outputs = infer_outputs_review
    base.validation_selection_metrics = validation_selection_metrics_review


def corrected_report(output_root: Path, variant: str, blind_root: Path) -> dict[str, Any]:
    report_path = output_root / "experiment_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "script_version": VERSION,
            "method": f"review_{variant}_rg_posr",
            "reliability_variant": variant,
            "blind_cache_root": str(blind_root),
            "reliability_dimensions": {
                modality: DESCRIPTOR_DIM for modality in MODALITIES
            },
            "reliability_total_dimension": len(MODALITIES) * DESCRIPTOR_DIM,
            "blind_descriptor_names": list(DESCRIPTOR_NAMES),
            "original_25_descriptor_cache_read": False,
            "original_natural_quality_targets_read": False,
            "independent_fidelity_target_constructed_online": True,
            "clean_reference_used_at_inference": False,
            "corruption_type_or_severity_given_to_predictor": False,
            "descriptor_recomputation_from_observed_corrupted_sequence": True,

            # Precise reviewer-facing metadata. Corruption augmentation is
            # shared across variants, but only learned heads use the
            # independent fidelity target as supervision.
            "synthetic_corruption_augmentation_used": True,
            "synthetic_corruption_supervision_used": (
                variant in LEARNED_VARIANTS
            ),
            "independent_fidelity_supervision_used": (
                variant in LEARNED_VARIANTS
            ),
            "reliability_features_used": variant != "uniform",
            "blind_descriptors_used_for_fusion": variant != "uniform",
            "direct_reliability_mapping_used": (
                variant == "direct_active_fraction"
            ),
            "learned_reliability_head_used": (
                variant in LEARNED_VARIANTS
            ),
            "reliability_losses_active": variant in LEARNED_VARIANTS,
            "reliability_gated_fusion_used": variant != "uniform",
            "reliability_scaled_rejection_used": variant != "uniform",
        }
    )
    base.save_json(report_path, report)
    return report


def run_review(config, overwrite: bool, variant: str, blind_root: Path):
    install_review_pathway(variant, blind_root)
    report = base.run(config, overwrite=overwrite)
    corrected = corrected_report(Path(config.output_root).resolve(), variant, blind_root)
    print(json.dumps(corrected, indent=2), flush=True)
    print("REVIEW RG-POSR EXPERIMENT COMPLETED", flush=True)
    return corrected


def self_test() -> None:
    from freeze_blind_reliability_cache import build as freeze_blind_cache

    with tempfile.TemporaryDirectory(prefix="review_rg_posr_test_") as folder:
        root = Path(folder)
        processed, frozen = base.synthetic_environment(root)

        stats_json = json.loads(
            (processed / "normalization_stats.json").read_text(encoding="utf-8")
        )
        normalization_npz = root / "normalization_stats.npz"
        values = {}
        for modality in MODALITIES:
            values[f"{modality}_mean"] = np.asarray(
                stats_json[modality]["mean"], dtype=np.float32
            )
            values[f"{modality}_std"] = np.asarray(
                stats_json[modality]["std"], dtype=np.float32
            )
        np.savez_compressed(normalization_npz, **values)

        blind_root = root / "blind_cache"
        freeze_blind_cache(
            index_path=frozen / "04_cache/reliability_window_index_10s.csv",
            normalization_path=normalization_npz,
            output_dir=blind_root,
            device=torch.device("cpu"),
            batch_size=16,
            overwrite=False,
        )

        for variant in RELIABILITY_VARIANTS:
            output = root / f"output_{variant}"
            config = base.RunConfig(
                processed_root=str(processed),
                frozen_root=str(frozen),
                prototype_output=None,
                output_root=str(output),
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
                quick=True,
                strict_protocol=False,
            )
            report = run_review(
                config,
                overwrite=True,
                variant=variant,
                blind_root=blind_root,
            )
            if report["reliability_variant"] != variant:
                raise AssertionError("Variant not recorded in report.")
            metrics = pd.read_csv(output / "seed_metrics.csv")
            if len(metrics) != 2:
                raise AssertionError("Expected two duration rows.")
            if not np.isfinite(
                metrics[[
                    "closed_set_accuracy",
                    "unknown_detection_auroc",
                    "open_set_overall_accuracy",
                ]].to_numpy(dtype=float)
            ).all():
                raise AssertionError("Non-finite self-test metrics.")

    print("=" * 76)
    print("REVIEW RG-POSR TRAINING SELF-TEST PASSED FOR ALL FOUR VARIANTS")
    print("=" * 76)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train reviewer-revision RG-POSR with blind reliability."
    )
    parser.add_argument("--processed-root")
    parser.add_argument("--frozen-root")
    parser.add_argument("--blind-cache-root")
    parser.add_argument("--prototype-output")
    parser.add_argument("--output")
    parser.add_argument(
        "--reliability-variant",
        choices=RELIABILITY_VARIANTS,
        default="nonlinear",
    )
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
            ("--blind-cache-root", args.blind_cache_root),
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
        seeds = base.parse_int_list(args.seeds, (42, 52, 62, 72, 82))
        folds = base.parse_int_list(args.folds, (0, 1, 2, 3, 4))
        aggregation_seconds = base.parse_int_list(
            args.aggregation_seconds,
            (30, 60, 120),
        )
        gamma_candidates = base.parse_float_list(
            args.gamma_candidates,
            base.GAMMA_DEFAULTS,
        )
        epochs = args.epochs
        batch_size = args.batch_size
        hidden_size = args.hidden_size
        embedding_size = args.embedding_size
        reliability_hidden_size = args.reliability_hidden_size
        patience = args.patience
        strict_protocol = True

    config = base.RunConfig(
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

    run_review(
        config,
        overwrite=args.overwrite,
        variant=args.reliability_variant,
        blind_root=Path(args.blind_cache_root).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()

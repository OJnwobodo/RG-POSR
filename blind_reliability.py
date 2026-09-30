#!/usr/bin/env python3
"""
Blind, non-circular reliability utilities for the RG-POSR review experiment.

Design rules
------------
1. Reliability descriptors are computed directly from the observed
   standardised sequence.
2. Corruption type and severity are never included in the predictor input.
3. The clean sequence is used only during training to construct the
   independent clean-to-corrupted fidelity target.
4. At inference, only the observed sequence is required.
5. The module supports uniform, direct, linear, and nonlinear reliability
   variants under a common interface.

This file is intended to be imported by train_review_rg_posr.py.
"""

from __future__ import annotations

import argparse
import math
from typing import Mapping

import torch
from torch import nn


MODALITIES = ("head", "left", "right", "gaze")
DESCRIPTOR_NAMES = (
    "inactive_frame_fraction",
    "flatline_transition_fraction",
    "median_frame_norm",
    "q95_frame_norm",
    "median_delta_norm",
    "q95_delta_norm",
    "temporal_energy_ratio",
    "mean_feature_standard_deviation",
)
DESCRIPTOR_DIM = len(DESCRIPTOR_NAMES)
INACTIVE_DESCRIPTOR_INDEX = DESCRIPTOR_NAMES.index("inactive_frame_fraction")
EPSILON = 1e-8

RELIABILITY_VARIANTS = (
    "uniform",
    "direct_active_fraction",
    "linear",
    "nonlinear",
)


def _validate_sequence(sequence: torch.Tensor) -> None:
    if sequence.ndim != 3:
        raise ValueError(
            "Expected sequence shape [batch, time, features], "
            f"observed {tuple(sequence.shape)}."
        )
    if sequence.shape[0] < 1 or sequence.shape[1] < 1 or sequence.shape[2] < 1:
        raise ValueError(f"Sequence must be non-empty, observed {tuple(sequence.shape)}.")


@torch.no_grad()
def blind_descriptors(sequence: torch.Tensor) -> torch.Tensor:
    """Compute eight severity-blind descriptors from an observed sequence.

    Parameters
    ----------
    sequence:
        Standardised values with shape [batch, time, features].

    Returns
    -------
    torch.Tensor
        Descriptor matrix with shape [batch, 8].

    Notes
    -----
    The function deliberately does not accept corruption metadata. Descriptor
    computation is detached from autograd because the descriptors are observed
    quality covariates rather than a differentiable transformation trained
    through the identity encoder.
    """

    _validate_sequence(sequence)

    values = torch.nan_to_num(
        sequence.detach(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).float()

    feature_scale = math.sqrt(float(values.shape[2]))
    frame_norm = torch.linalg.vector_norm(values, dim=2) / feature_scale
    inactive = frame_norm <= 1e-8

    if values.shape[1] > 1:
        deltas = values[:, 1:, :] - values[:, :-1, :]
        delta_norm = torch.linalg.vector_norm(deltas, dim=2) / feature_scale
        flatline = delta_norm <= 1e-3
    else:
        delta_norm = torch.zeros(
            (values.shape[0], 1),
            dtype=values.dtype,
            device=values.device,
        )
        flatline = torch.ones_like(delta_norm, dtype=torch.bool)

    frame_energy = torch.mean(frame_norm.square(), dim=1)
    delta_energy = torch.mean(delta_norm.square(), dim=1)

    descriptors = torch.stack(
        [
            inactive.float().mean(dim=1),
            flatline.float().mean(dim=1),
            torch.quantile(frame_norm, 0.50, dim=1),
            torch.quantile(frame_norm, 0.95, dim=1),
            torch.quantile(delta_norm, 0.50, dim=1),
            torch.quantile(delta_norm, 0.95, dim=1),
            delta_energy / (frame_energy + EPSILON),
            torch.std(values, dim=1, unbiased=False).mean(dim=1),
        ],
        dim=1,
    )

    return torch.nan_to_num(
        descriptors,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    )


@torch.no_grad()
def descriptor_tensor(
    sequences: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return blind descriptors with shape [batch, modality, descriptor]."""

    missing = [modality for modality in MODALITIES if modality not in sequences]
    if missing:
        raise KeyError(f"Missing modality sequences: {missing}")

    batch_sizes = {int(sequences[m].shape[0]) for m in MODALITIES}
    if len(batch_sizes) != 1:
        raise ValueError(f"Modality batch sizes differ: {batch_sizes}")

    return torch.stack(
        [blind_descriptors(sequences[modality]) for modality in MODALITIES],
        dim=1,
    )


@torch.no_grad()
def fidelity_target(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
) -> torch.Tensor:
    """Independent clean-to-corrupted signal fidelity in [0, 1].

    The target is the non-negative cosine similarity multiplied by a symmetric
    sequence-energy retention ratio. It equals 1 for identical non-zero
    sequences and 0 for complete dropout.

    Both tensors must have shape [batch, time, features]. The clean sequence is
    required only during training and is never needed at inference.
    """

    _validate_sequence(clean)
    _validate_sequence(corrupted)

    if clean.shape != corrupted.shape:
        raise ValueError(
            "Clean/corrupted shapes differ: "
            f"{tuple(clean.shape)} versus {tuple(corrupted.shape)}."
        )

    clean_values = torch.nan_to_num(
        clean.detach(), nan=0.0, posinf=0.0, neginf=0.0
    ).float()
    corrupted_values = torch.nan_to_num(
        corrupted.detach(), nan=0.0, posinf=0.0, neginf=0.0
    ).float()

    batch_size = clean_values.shape[0]
    clean_flat = clean_values.reshape(batch_size, -1)
    corrupted_flat = corrupted_values.reshape(batch_size, -1)

    clean_norm = torch.linalg.vector_norm(clean_flat, dim=1)
    corrupted_norm = torch.linalg.vector_norm(corrupted_flat, dim=1)

    both_zero = (clean_norm <= EPSILON) & (corrupted_norm <= EPSILON)
    corrupted_zero = (clean_norm > EPSILON) & (corrupted_norm <= EPSILON)

    cosine = torch.sum(clean_flat * corrupted_flat, dim=1) / (
        clean_norm * corrupted_norm
    ).clamp_min(EPSILON)

    energy_ratio = torch.minimum(clean_norm, corrupted_norm) / torch.maximum(
        clean_norm, corrupted_norm
    ).clamp_min(EPSILON)

    target = torch.clamp(torch.clamp_min(cosine, 0.0) * energy_ratio, 0.0, 1.0)
    target = torch.where(both_zero, torch.ones_like(target), target)
    target = torch.where(corrupted_zero, torch.zeros_like(target), target)

    return target


def standardize_descriptors(
    raw_descriptors: torch.Tensor,
    median: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Robustly standardise descriptors.

    raw_descriptors must have shape [batch, 4, 8]. median and scale may have
    shape [4, 8] or [1, 4, 8].
    """

    if raw_descriptors.ndim != 3:
        raise ValueError(
            "Expected descriptor shape [batch, modality, descriptor], "
            f"observed {tuple(raw_descriptors.shape)}."
        )
    if raw_descriptors.shape[1:] != (len(MODALITIES), DESCRIPTOR_DIM):
        raise ValueError(
            f"Expected descriptor tail {(len(MODALITIES), DESCRIPTOR_DIM)}, "
            f"observed {tuple(raw_descriptors.shape[1:])}."
        )

    safe_scale = torch.where(scale.abs() > EPSILON, scale, torch.ones_like(scale))
    standardized = (raw_descriptors - median) / safe_scale
    return torch.nan_to_num(
        standardized,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    )


def _nested_burst_bounds(
    length: int,
    severity: float,
    centre: int,
) -> tuple[int, int]:
    burst_length = max(1, int(round(float(severity) * length)))
    if burst_length >= length:
        return 0, length

    start = centre - burst_length // 2
    start = max(0, min(start, length - burst_length))
    return start, start + burst_length


@torch.no_grad()
def apply_blind_synthetic_corruption(
    sequences: Mapping[str, torch.Tensor],
    probability: float,
    seed: int,
    severity_choices: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply training-only corruption and recompute supervision blindly.

    Returns
    -------
    corrupted_sequences:
        Dict of corrupted standardised modality sequences.
    raw_descriptors:
        Blind descriptors recomputed from corrupted sequences, shape [B,4,8].
    targets:
        Independent fidelity targets, shape [B,4].
    severities:
        Applied severities for loss diagnostics only, shape [B,4].
    corruption_types:
        0 clean, 1 dropout, 2 Gaussian noise, 3 temporal burst.

    Important
    ---------
    Corruption type and severity are returned for diagnostics and monotonic
    training only. They must never be concatenated with the reliability-head
    inputs. The predictors receive only raw_descriptors.
    """

    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Probability must be in [0,1], observed {probability}.")
    if not severity_choices:
        raise ValueError("severity_choices cannot be empty.")

    missing = [modality for modality in MODALITIES if modality not in sequences]
    if missing:
        raise KeyError(f"Missing modality sequences: {missing}")

    device = sequences[MODALITIES[0]].device
    batch_size = int(sequences[MODALITIES[0]].shape[0])

    for modality in MODALITIES:
        _validate_sequence(sequences[modality])
        if sequences[modality].shape[0] != batch_size:
            raise ValueError("All modalities must share the same batch size.")
        if sequences[modality].device != device:
            raise ValueError("All modalities must be on the same device.")

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))

    severity_values = torch.tensor(severity_choices, dtype=torch.float32)
    active = torch.rand((batch_size, len(MODALITIES)), generator=cpu_generator)
    active = active < probability
    types_cpu = torch.randint(
        1,
        4,
        (batch_size, len(MODALITIES)),
        generator=cpu_generator,
    )
    severity_indices = torch.randint(
        0,
        len(severity_values),
        (batch_size, len(MODALITIES)),
        generator=cpu_generator,
    )
    severities_cpu = severity_values[severity_indices] * active.float()
    types_cpu = types_cpu * active.long()

    corrupted = {
        modality: torch.nan_to_num(
            sequences[modality].detach().clone(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for modality in MODALITIES
    }

    for modality_index, modality in enumerate(MODALITIES):
        values = corrupted[modality]
        time_length = int(values.shape[1])

        for sample_index in range(batch_size):
            severity = float(severities_cpu[sample_index, modality_index].item())
            corruption_type = int(types_cpu[sample_index, modality_index].item())

            if corruption_type == 0 or severity <= 0.0:
                continue

            if corruption_type == 1:
                values[sample_index].zero_()

            elif corruption_type == 2:
                noise_generator = torch.Generator(device="cpu")
                noise_generator.manual_seed(
                    int(seed + 1_000_003 * (modality_index + 1) + sample_index)
                )
                sigma = 0.10 + 0.40 * severity
                noise = torch.randn(
                    values[sample_index].shape,
                    generator=noise_generator,
                    dtype=values.dtype,
                ).to(device=device)
                values[sample_index].add_(sigma * noise)

            elif corruption_type == 3:
                centre_generator = torch.Generator(device="cpu")
                centre_generator.manual_seed(
                    int(seed + 2_000_003 * (modality_index + 1) + sample_index)
                )
                centre = int(
                    torch.randint(
                        0,
                        max(1, time_length),
                        (1,),
                        generator=centre_generator,
                    ).item()
                )
                start, stop = _nested_burst_bounds(
                    time_length,
                    severity,
                    centre,
                )
                values[sample_index, start:stop].zero_()

            else:
                raise RuntimeError(f"Unexpected corruption type {corruption_type}.")

        corrupted[modality] = values

    raw_descriptors = descriptor_tensor(corrupted)
    targets = torch.stack(
        [
            fidelity_target(sequences[modality], corrupted[modality])
            for modality in MODALITIES
        ],
        dim=1,
    )

    return (
        corrupted,
        raw_descriptors,
        targets,
        severities_cpu.to(device=device),
        types_cpu.to(device=device),
    )


class _NonlinearHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(values)).squeeze(1)


class ReliabilityEstimator(nn.Module):
    """Common reliability estimator for all reviewer-requested variants."""

    def __init__(
        self,
        variant: str,
        hidden_size: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        if variant not in RELIABILITY_VARIANTS:
            raise ValueError(
                f"Unknown reliability variant {variant!r}; "
                f"choose from {RELIABILITY_VARIANTS}."
            )

        self.variant = variant

        if variant == "linear":
            self.heads = nn.ModuleDict(
                {
                    modality: nn.Linear(DESCRIPTOR_DIM, 1)
                    for modality in MODALITIES
                }
            )
        elif variant == "nonlinear":
            self.heads = nn.ModuleDict(
                {
                    modality: _NonlinearHead(
                        DESCRIPTOR_DIM,
                        hidden_size,
                        dropout,
                    )
                    for modality in MODALITIES
                }
            )
        else:
            self.heads = nn.ModuleDict()

    def forward(
        self,
        raw_descriptors: torch.Tensor,
        standardized_descriptors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if raw_descriptors.ndim != 3:
            raise ValueError(
                "Expected raw descriptors [batch,4,8], "
                f"observed {tuple(raw_descriptors.shape)}."
            )

        batch_size = int(raw_descriptors.shape[0])

        if self.variant == "uniform":
            return torch.ones(
                (batch_size, len(MODALITIES)),
                dtype=raw_descriptors.dtype,
                device=raw_descriptors.device,
            )

        if self.variant == "direct_active_fraction":
            reliability = 1.0 - raw_descriptors[:, :, INACTIVE_DESCRIPTOR_INDEX]
            return torch.clamp(reliability, 0.0, 1.0)

        if standardized_descriptors is None:
            raise ValueError(
                f"Variant {self.variant!r} requires standardized descriptors."
            )

        outputs = []

        for modality_index, modality in enumerate(MODALITIES):
            values = standardized_descriptors[:, modality_index, :]

            if self.variant == "linear":
                outputs.append(
                    torch.sigmoid(self.heads[modality](values)).squeeze(1)
                )
            elif self.variant == "nonlinear":
                outputs.append(self.heads[modality](values))
            else:
                raise RuntimeError(f"Unhandled reliability variant {self.variant!r}.")

        return torch.stack(outputs, dim=1)


def reliability_gates(
    reliability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized gates and aggregate effective reliability."""

    if reliability.ndim != 2 or reliability.shape[1] != len(MODALITIES):
        raise ValueError(
            f"Expected reliability shape [batch,{len(MODALITIES)}], "
            f"observed {tuple(reliability.shape)}."
        )

    positive = torch.clamp(reliability, 0.0, 1.0)
    gates = positive / positive.sum(dim=1, keepdim=True).clamp_min(EPSILON)
    effective = torch.sum(gates * positive, dim=1)
    return gates, effective


def self_test(device_name: str = "cpu") -> None:
    device = torch.device(device_name)
    torch.manual_seed(7)

    batch_size = 6
    time_steps = 150
    dimensions = {
        "head": 13,
        "left": 22,
        "right": 22,
        "gaze": 19,
    }

    clean = {
        modality: torch.randn(
            batch_size,
            time_steps,
            dimensions[modality],
            device=device,
        )
        for modality in MODALITIES
    }

    raw_clean = descriptor_tensor(clean)
    assert raw_clean.shape == (batch_size, 4, DESCRIPTOR_DIM)
    assert torch.isfinite(raw_clean).all()

    for modality in MODALITIES:
        identical = fidelity_target(clean[modality], clean[modality])
        assert torch.allclose(
            identical,
            torch.ones_like(identical),
            atol=1e-6,
        )

        dropout = fidelity_target(
            clean[modality],
            torch.zeros_like(clean[modality]),
        )
        assert torch.allclose(
            dropout,
            torch.zeros_like(dropout),
            atol=1e-6,
        )

    corrupted, descriptors, targets, severities, types = (
        apply_blind_synthetic_corruption(
            clean,
            probability=1.0,
            seed=12345,
        )
    )

    assert descriptors.shape == (batch_size, 4, DESCRIPTOR_DIM)
    assert targets.shape == (batch_size, 4)
    assert severities.shape == (batch_size, 4)
    assert types.shape == (batch_size, 4)
    assert torch.isfinite(descriptors).all()
    assert torch.isfinite(targets).all()
    assert torch.all((targets >= 0.0) & (targets <= 1.0))

    median = torch.zeros((4, DESCRIPTOR_DIM), device=device)
    scale = torch.ones((4, DESCRIPTOR_DIM), device=device)
    standardized = standardize_descriptors(descriptors, median, scale)

    for variant in RELIABILITY_VARIANTS:
        estimator = ReliabilityEstimator(
            variant=variant,
            hidden_size=16,
            dropout=0.0,
        ).to(device)

        reliability = estimator(descriptors, standardized)
        gates, effective = reliability_gates(reliability)

        assert reliability.shape == (batch_size, 4)
        assert gates.shape == (batch_size, 4)
        assert effective.shape == (batch_size,)
        assert torch.isfinite(reliability).all()
        assert torch.isfinite(gates).all()
        assert torch.isfinite(effective).all()
        assert torch.allclose(
            gates.sum(dim=1),
            torch.ones(batch_size, device=device),
            atol=1e-6,
        )

    # Verify nested burst target is non-increasing for a fixed clean sequence.
    one = {"head": clean["head"][:1]}
    burst_targets = []
    centre = time_steps // 2

    for severity in (0.25, 0.50, 0.75, 1.00):
        burst = one["head"].clone()
        start, stop = _nested_burst_bounds(time_steps, severity, centre)
        burst[:, start:stop].zero_()
        burst_targets.append(float(fidelity_target(one["head"], burst).item()))

    differences = [
        later - earlier
        for earlier, later in zip(burst_targets[:-1], burst_targets[1:])
    ]
    assert all(value <= 1e-6 for value in differences), burst_targets

    print("=" * 72)
    print("BLIND RELIABILITY MODULE SELF-TEST PASSED")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Descriptor names: {DESCRIPTOR_NAMES}")
    print(f"Corrupted target range: {targets.min().item():.6f} to {targets.max().item():.6f}")
    print(f"Nested burst targets: {burst_targets}")
    print("No corruption type or severity is included in predictor descriptors.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic module checks.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used by the self-test, e.g. cpu or cuda.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test(arguments.device)
    else:
        print(
            "This module is designed to be imported. "
            "Run with --self-test to validate it."
        )

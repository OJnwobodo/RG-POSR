from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


# ================================================================
# FIXED PATHS / PROTOCOL
# ================================================================

def first_existing_path(*candidates: Path) -> Path:
    """Return the first existing path, preserving the first candidate for clear errors."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


SOURCE_SCRIPT = first_existing_path(
    Path(
        r"G:\ONYEKA_NEW_EXPERIMENT\Onyeka_Review_Experiment"
        r"\02_reliability_models\train_review_rg_posr.py"
    ),
    Path(
        r"D:\ONYEKA_NEW_EXPERIMENT\Onyeka_Review_Experiment"
        r"\02_reliability_models\train_review_rg_posr.py"
    ),
)

PROCESSED_ROOT = first_existing_path(
    Path(r"G:\ONYEKA\who-is-alyx-processed"),
    Path(r"D:\ONYEKA\who-is-alyx-processed"),
)

FROZEN_ROOT = first_existing_path(
    Path(r"G:\ONYEKA_NEW_EXPERIMENT\RG_POSR_FROZEN_v1"),
    Path(r"D:\ONYEKA_NEW_EXPERIMENT\RG_POSR_FROZEN_v1"),
)

REVIEW_MODELS_ROOT = first_existing_path(
    Path(
        r"G:\ONYEKA_NEW_EXPERIMENT\Onyeka_Review_Experiment"
        r"\02_reliability_models"
    ),
    Path(
        r"D:\ONYEKA_NEW_EXPERIMENT\Onyeka_Review_Experiment"
        r"\02_reliability_models"
    ),
)

BLIND_CACHE_ROOT = (
    REVIEW_MODELS_ROOT
    / "review_blind_cache"
)

# Keep the tuned run separate from the original fixed-T=1 results.
OUTPUT = Path(
    r"D:\ONYEKA_NEW_EXPERIMENT"
    r"\rg-posr-energy-review-nonlinear-tuned"
)


SEEDS = (42, 52, 62, 72, 82)
FOLDS = (0, 1, 2, 3, 4)
DURATIONS = (30, 60, 120)

# Fixed a priori log-spaced search grid. Temperature is selected using
# Session-1 pseudo-unknown calibration only, separately for each seed
# and aggregation duration, then frozen before Session-2 evaluation.
TEMPERATURE_CANDIDATES = (
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)
METHOD = "energy_score"

# Keep this on CPU for exact compatibility with the earlier evaluator.
DEVICE = torch.device("cpu")


# ================================================================
# NUMPY COMPATIBILITY
# ================================================================

# Your installed NumPy version does not provide np.trapezoid,
# while the RG-POSR OSCR implementation calls it.
if not hasattr(np, "trapezoid") and hasattr(np, "trapz"):
    np.trapezoid = np.trapz


# ================================================================
# IMPORT THE ACTUAL REVIEW/NONLINEAR IMPLEMENTATION
# ================================================================

def import_rg_posr():

    if not SOURCE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Review source script not found:\n{SOURCE_SCRIPT}"
        )

    source_dir = str(SOURCE_SCRIPT.parent)

    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    spec = importlib.util.spec_from_file_location(
        "rg_posr_energy_review_source",
        SOURCE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import source script:\n{SOURCE_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    # Force the architecture corresponding to the checkpoints
    # protocol_seedXX_nonlinear.
    if hasattr(module, "REVIEW_VARIANT"):
        module.REVIEW_VARIANT = "nonlinear"

    # load_protocol_review() obtains the blind reliability
    # descriptor data from this global variable.
    module.REVIEW_BLIND_CACHE_ROOT = BLIND_CACHE_ROOT

    return module


RG = import_rg_posr()


# ================================================================
# RESOLVE SHARED FUNCTIONS
# ================================================================

def rg_function(name: str):

    if hasattr(RG, name):
        return getattr(RG, name)

    base = getattr(RG, "base", None)

    if (
        base is not None
        and hasattr(base, name)
    ):
        return getattr(base, name)

    raise AttributeError(
        f"Function {name!r} was not found in "
        "train_review_rg_posr.py or its base module."
    )


# ================================================================
# CHECKPOINT PATHS
# ================================================================

def calibration_checkpoint(
    seed: int,
    fold: int,
) -> Path:

    return (
        REVIEW_MODELS_ROOT
        / f"protocol_seed{seed}_nonlinear"
        / "models"
        / f"seed_{seed}"
        / f"calibration_fold_{fold}"
        / "best.pt"
    )


def final_checkpoint(
    seed: int,
) -> Path:

    return (
        REVIEW_MODELS_ROOT
        / f"protocol_seed{seed}_nonlinear"
        / "models"
        / f"seed_{seed}"
        / "final"
        / "best.pt"
    )


# ================================================================
# CHECKPOINT LOADING
# ================================================================

def load_checkpoint(
    path: Path,
):

    if not path.is_file():
        raise FileNotFoundError(path)

    try:

        return torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location=DEVICE,
        )


# ================================================================
# RECONSTRUCT THE EXACT REVIEW MODEL
# ================================================================

def build_model(
    checkpoint_path: Path,
):

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    if "model_config" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint has no model_config:\n"
            f"{checkpoint_path}"
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint has no model_state_dict:\n"
            f"{checkpoint_path}"
        )

    config = checkpoint["model_config"]

    # ReviewRGPOSR reads REVIEW_VARIANT when it is constructed.
    if hasattr(RG, "REVIEW_VARIANT"):
        RG.REVIEW_VARIANT = "nonlinear"

    model = RG.ReviewRGPOSR(
        len(config["participants"]),
        int(config["hidden_size"]),
        int(config["embedding_size"]),
        int(config["num_layers"]),
        float(config["dropout"]),
        int(config["reliability_hidden_size"]),
    ).to(DEVICE)

    # strict=True is deliberate.
    # We want an error if this is not the exact model architecture.
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    participant_to_class = {
        str(participant): index
        for index, participant
        in enumerate(config["participants"])
    }

    arcface_scale = float(
        config["arcface_scale"]
    )

    return (
        model,
        participant_to_class,
        arcface_scale,
    )


# ================================================================
# PROTOTYPES
# ================================================================

def normalized_prototypes(
    model,
) -> np.ndarray:

    return (
        torch.nn.functional.normalize(
            model.prototypes,
            dim=1,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ================================================================
# ENERGY SCORE
# ================================================================

def stable_logsumexp(
    values: np.ndarray,
) -> np.ndarray:

    maximum = np.max(
        values,
        axis=1,
        keepdims=True,
    )

    return (
        np.squeeze(
            maximum,
            axis=1,
        )
        +
        np.log(
            np.sum(
                np.exp(
                    values
                    - maximum
                ),
                axis=1,
            )
        )
    )


def energy_scores(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    scale: float,
    temperature: float,
):

    temperature = float(temperature)

    if (
        not np.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise ValueError(
            f"Temperature must be positive and finite; got {temperature}."
        )

    cosine = np.clip(
        embeddings
        @ prototypes.T,
        -1.0,
        1.0,
    )

    logits = (
        scale
        * cosine
    )

    energy = (
        -temperature
        * stable_logsumexp(
            logits
            / temperature
        )
    )

    prediction = np.argmax(
        logits,
        axis=1,
    ).astype(np.int64)

    return (
        energy.astype(np.float64),
        prediction,
        logits.astype(np.float32),
    )


# ================================================================
# USER-MACRO RATE
# ================================================================

def macro_rate(
    mask: np.ndarray,
    users: np.ndarray,
) -> float:

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    users = np.asarray(
        users
    ).astype(str)

    rates = []

    for user in np.unique(users):

        user_mask = (
            users == user
        )

        rates.append(
            float(
                np.mean(
                    mask[user_mask]
                )
            )
        )

    if not rates:
        return float("nan")

    return float(
        np.mean(rates)
    )


# ================================================================
# REVIEW MODEL INFERENCE
# ================================================================

def infer(
    model,
    frame,
    means,
    stds,
    raw_reliability,
    target_placeholder,
    rel_median,
    rel_scale,
):

    return RG.infer_outputs_review(
        model,
        frame,
        means,
        stds,
        raw_reliability,
        target_placeholder,
        rel_median,
        rel_scale,
        64,
        0,
        DEVICE,
        False,
    )


# ================================================================
# AGGREGATION
# ================================================================

def aggregate(
    frame,
    outputs,
    duration: int,
):

    aggregate_outputs = rg_function(
        "aggregate_outputs"
    )

    # outputs are:
    # embedding,
    # reliability,
    # gates,
    # effective reliability,
    # target placeholder
    #
    # Only the first four are used by aggregate_outputs().
    return aggregate_outputs(
        frame,
        *outputs[:4],
        duration,
    )


# ================================================================
# SESSION-1 PSEUDO-UNKNOWN CALIBRATION
# ================================================================

def calibrate_seed(
    seed,
    split,
    index,
    means,
    stds,
    raw_reliability,
    target_placeholder,
    rel_median,
    rel_scale,
):

    participant_ids_by_fold = rg_function(
        "participant_ids_by_fold"
    )

    rows_for_players = rg_function(
        "rows_for_players"
    )

    choose_threshold = rg_function(
        "choose_threshold"
    )

    rows = []

    for fold in FOLDS:

        included, heldout = (
            participant_ids_by_fold(
                split,
                fold,
            )
        )

        known_frame = rows_for_players(
            index,
            included,
            session_order=1,
            split_names=("validation",),
        )

        pseudo_frame = rows_for_players(
            index,
            heldout,
            session_order=1,
            split_names=("validation",),
        )

        checkpoint = calibration_checkpoint(
            seed,
            fold,
        )

        model, _, scale = build_model(
            checkpoint
        )

        prototypes = normalized_prototypes(
            model
        )

        known_outputs = infer(
            model,
            known_frame,
            means,
            stds,
            raw_reliability,
            target_placeholder,
            rel_median,
            rel_scale,
        )

        pseudo_outputs = infer(
            model,
            pseudo_frame,
            means,
            stds,
            raw_reliability,
            target_placeholder,
            rel_median,
            rel_scale,
        )

        for duration in DURATIONS:

            (
                known_embeddings,
                _,
                _,
                _,
                known_meta,
            ) = aggregate(
                known_frame,
                known_outputs,
                duration,
            )

            (
                pseudo_embeddings,
                _,
                _,
                _,
                pseudo_meta,
            ) = aggregate(
                pseudo_frame,
                pseudo_outputs,
                duration,
            )

            for temperature in TEMPERATURE_CANDIDATES:

                known_energy, _, _ = energy_scores(
                    known_embeddings,
                    prototypes,
                    scale,
                    temperature,
                )

                pseudo_energy, _, _ = energy_scores(
                    pseudo_embeddings,
                    prototypes,
                    scale,
                    temperature,
                )

                (
                    threshold,
                    calibration,
                ) = choose_threshold(
                    known_energy,
                    known_meta[
                        "player_id"
                    ]
                    .astype(str)
                    .to_numpy(),

                    pseudo_energy,
                    pseudo_meta[
                        "player_id"
                    ]
                    .astype(str)
                    .to_numpy(),
                )

                rows.append(
                    {
                        "method":
                            METHOD,

                        "seed":
                            seed,

                        "fold":
                            fold,

                        "aggregation_seconds":
                            duration,

                        "temperature":
                            float(temperature),

                        "arcface_scale":
                            scale,

                        "included_users":
                            len(included),

                        "pseudo_unknown_users":
                            len(heldout),

                        "known_blocks":
                            len(known_meta),

                        "pseudo_unknown_blocks":
                            len(pseudo_meta),

                        "threshold":
                            float(threshold),

                        "known_energy_mean":
                            float(
                                np.mean(
                                    known_energy
                                )
                            ),

                        "pseudo_unknown_energy_mean":
                            float(
                                np.mean(
                                    pseudo_energy
                                )
                            ),

                        "known_unknown_energy_gap":
                            float(
                                np.mean(
                                    pseudo_energy
                                )
                                -
                                np.mean(
                                    known_energy
                                )
                            ),

                        **calibration,
                    }
                )

        del model

    fold_frame = pd.DataFrame(
        rows
    )

    selected = []

    for duration, duration_group in fold_frame.groupby(
        "aggregation_seconds",
        sort=True,
    ):

        candidate_rows = []

        for temperature, temperature_group in duration_group.groupby(
            "temperature",
            sort=True,
        ):

            candidate_rows.append(
                {
                    "temperature":
                        float(temperature),

                    "mean_calibration_balanced_detection":
                        float(
                            temperature_group[
                                "balanced_detection"
                            ].mean()
                        ),

                    "mean_calibration_known_acceptance":
                        float(
                            temperature_group[
                                "known_acceptance_macro_user"
                            ].mean()
                        ),

                    "mean_calibration_pseudo_unknown_rejection":
                        float(
                            temperature_group[
                                "pseudo_unknown_rejection_macro_user"
                            ].mean()
                        ),
                }
            )

        candidate_frame = pd.DataFrame(
            candidate_rows
        )

        # Mirror the RG-POSR gamma-selection protocol:
        # 1) higher mean B_cal,
        # 2) higher mean pseudo-unknown rejection,
        # 3) higher mean known-user acceptance,
        # 4) smaller temperature as deterministic final tie-break.
        candidate_frame = candidate_frame.sort_values(
            [
                "mean_calibration_balanced_detection",
                "mean_calibration_pseudo_unknown_rejection",
                "mean_calibration_known_acceptance",
                "temperature",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            kind="mergesort",
        ).reset_index(
            drop=True
        )

        winner = candidate_frame.iloc[0]

        selected_temperature = float(
            winner[
                "temperature"
            ]
        )

        winner_folds = duration_group.loc[
            np.isclose(
                pd.to_numeric(
                    duration_group[
                        "temperature"
                    ],
                    errors="raise",
                ).to_numpy(
                    dtype=np.float64
                ),
                selected_temperature,
            )
        ].copy()

        if len(winner_folds) != len(FOLDS):
            raise RuntimeError(
                f"Expected {len(FOLDS)} folds for selected "
                f"temperature={selected_temperature}, seed={seed}, "
                f"duration={duration}; found {len(winner_folds)}."
            )

        selected.append(
            {
                "method":
                    METHOD,

                "seed":
                    seed,

                "aggregation_seconds":
                    int(duration),

                "temperature":
                    selected_temperature,

                # After temperature selection, freeze the median
                # of the five fold-specific thresholds for that T.
                "threshold":
                    float(
                        np.median(
                            winner_folds[
                                "threshold"
                            ]
                        )
                    ),

                "mean_calibration_balanced_detection":
                    float(
                        winner[
                            "mean_calibration_balanced_detection"
                        ]
                    ),

                "mean_calibration_known_acceptance":
                    float(
                        winner[
                            "mean_calibration_known_acceptance"
                        ]
                    ),

                "mean_calibration_pseudo_unknown_rejection":
                    float(
                        winner[
                            "mean_calibration_pseudo_unknown_rejection"
                        ]
                    ),
            }
        )

    return (
        pd.DataFrame(selected),
        fold_frame,
    )


# ================================================================
# FINAL SESSION-2 KNOWN / UNKNOWN SETS
# ================================================================

def test_frames(
    split: pd.DataFrame,
    index: pd.DataFrame,
):

    roles = (
        split["role"]
        .astype(str)
        .str.lower()
    )

    enrolled = sorted(
        split.loc[
            roles.eq("enrolled"),
            "player_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    unknown = sorted(
        split.loc[
            roles.isin(
                [
                    "final_unknown",
                    "unknown",
                ]
            ),
            "player_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(enrolled) != 49:

        raise RuntimeError(
            "Frozen protocol mismatch: "
            f"expected 49 enrolled users, "
            f"found {len(enrolled)}."
        )

    if len(unknown) != 20:

        raise RuntimeError(
            "Frozen protocol mismatch: "
            f"expected 20 final unknown users, "
            f"found {len(unknown)}."
        )

    eligible = (
        index[
            "eligible_primary"
        ]
        .astype(bool)
    )

    known_test = index[
        index[
            "player_id"
        ]
        .astype(str)
        .isin(enrolled)

        & index[
            "session_order"
        ].eq(2)

        & eligible
    ].copy()

    unknown_test = index[
        index[
            "player_id"
        ]
        .astype(str)
        .isin(unknown)

        & index[
            "session_order"
        ].eq(2)

        & eligible
    ].copy()

    if known_test.empty:
        raise RuntimeError(
            "Session-2 known test set is empty."
        )

    if unknown_test.empty:
        raise RuntimeError(
            "Session-2 unknown test set is empty."
        )

    return (
        enrolled,
        unknown,
        known_test,
        unknown_test,
    )


# ================================================================
# FINAL FROZEN SESSION-2 EVALUATION
# ================================================================

def evaluate_seed(
    seed,
    selected,
    split,
    index,
    means,
    stds,
    raw_reliability,
    target_placeholder,
    rel_median,
    rel_scale,
):

    eer_value = rg_function(
        "eer_value"
    )

    oscr_auc_function = rg_function(
        "oscr_auc"
    )

    (
        enrolled,
        _unknown_users,
        known_test,
        unknown_test,
    ) = test_frames(
        split,
        index,
    )

    checkpoint = final_checkpoint(
        seed
    )

    (
        model,
        participant_to_class,
        scale,
    ) = build_model(
        checkpoint
    )

    prototypes = normalized_prototypes(
        model
    )

    # Class order itself does not have to equal lexical order.
    # The checkpoint mapping is authoritative.
    if (
        set(participant_to_class)
        != set(enrolled)
    ):

        missing = sorted(
            set(enrolled)
            - set(participant_to_class)
        )

        extra = sorted(
            set(participant_to_class)
            - set(enrolled)
        )

        raise RuntimeError(
            f"Participant-set mismatch for seed {seed}. "
            f"Missing={missing}; Extra={extra}"
        )

    known_outputs = infer(
        model,
        known_test,
        means,
        stds,
        raw_reliability,
        target_placeholder,
        rel_median,
        rel_scale,
    )

    unknown_outputs = infer(
        model,
        unknown_test,
        means,
        stds,
        raw_reliability,
        target_placeholder,
        rel_median,
        rel_scale,
    )

    metric_rows = []
    prediction_frames = []

    inverse = {
        class_index: participant
        for (
            participant,
            class_index,
        )
        in participant_to_class.items()
    }

    for duration in DURATIONS:

        (
            known_embeddings,
            _,
            _,
            _,
            known_meta,
        ) = aggregate(
            known_test,
            known_outputs,
            duration,
        )

        (
            unknown_embeddings,
            _,
            _,
            _,
            unknown_meta,
        ) = aggregate(
            unknown_test,
            unknown_outputs,
            duration,
        )

        calibration_rows = (
            selected.loc[
                selected[
                    "aggregation_seconds"
                ].eq(duration)
            ]
        )

        if len(calibration_rows) != 1:

            raise RuntimeError(
                f"Expected exactly one selected calibration row "
                f"for seed={seed}, duration={duration}; "
                f"found {len(calibration_rows)}."
            )

        calibration_row = calibration_rows.iloc[0]

        temperature = float(
            calibration_row[
                "temperature"
            ]
        )

        threshold = float(
            calibration_row[
                "threshold"
            ]
        )

        (
            known_energy,
            known_prediction,
            known_logits,
        ) = energy_scores(
            known_embeddings,
            prototypes,
            scale,
            temperature,
        )

        (
            unknown_energy,
            unknown_prediction,
            _,
        ) = energy_scores(
            unknown_embeddings,
            prototypes,
            scale,
            temperature,
        )

        known_true = np.asarray(
            [
                participant_to_class[
                    str(player)
                ]
                for player
                in known_meta[
                    "player_id"
                ]
            ],
            dtype=np.int64,
        )

        # Lower/more-negative energy is treated as known.
        known_accept = (
            known_energy
            <= threshold
        )

        unknown_accept = (
            unknown_energy
            <= threshold
        )

        known_correct = (
            known_prediction
            == known_true
        )

        known_correct_accepted = (
            known_correct
            & known_accept
        )

        top_k = min(
            5,
            known_logits.shape[1],
        )

        top5 = np.argsort(
            known_logits,
            axis=1,
        )[:, -top_k:]

        top5_correct = np.asarray(
            [
                truth in row
                for (
                    truth,
                    row,
                )
                in zip(
                    known_true,
                    top5,
                )
            ],
            dtype=bool,
        )

        labels_unknown = np.concatenate(
            [
                np.zeros(
                    len(known_energy),
                    dtype=np.int64,
                ),
                np.ones(
                    len(unknown_energy),
                    dtype=np.int64,
                ),
            ]
        )

        scores = np.concatenate(
            [
                known_energy,
                unknown_energy,
            ]
        )

        calibration = (
            selected.loc[
                selected[
                    "aggregation_seconds"
                ].eq(duration)
            ]
            .iloc[0]
        )

        accepted_count = int(
            np.sum(
                known_accept
            )
        )

        open_correct = (
            int(
                np.sum(
                    known_correct_accepted
                )
            )
            +
            int(
                np.sum(
                    ~unknown_accept
                )
            )
        )

        open_total = (
            len(known_energy)
            +
            len(unknown_energy)
        )

        known_accept_macro = (
            macro_rate(
                known_accept,
                known_meta[
                    "player_id"
                ]
                .astype(str)
                .to_numpy(),
            )
        )

        unknown_false_accept_macro = (
            macro_rate(
                unknown_accept,
                unknown_meta[
                    "player_id"
                ]
                .astype(str)
                .to_numpy(),
            )
        )

        unknown_true_reject_macro = (
            1.0
            -
            unknown_false_accept_macro
        )

        balanced_detection_macro = (
            0.5
            *
            (
                known_accept_macro
                +
                unknown_true_reject_macro
            )
        )

        metric_rows.append(
            {
                "method":
                    METHOD,

                "seed":
                    seed,

                "aggregation_seconds":
                    duration,

                "temperature":
                    temperature,

                "arcface_scale":
                    scale,

                "threshold":
                    threshold,

                "mean_calibration_balanced_detection":
                    float(
                        calibration[
                            "mean_calibration_balanced_detection"
                        ]
                    ),

                "mean_calibration_known_acceptance":
                    float(
                        calibration[
                            "mean_calibration_known_acceptance"
                        ]
                    ),

                "mean_calibration_pseudo_unknown_rejection":
                    float(
                        calibration[
                            "mean_calibration_pseudo_unknown_rejection"
                        ]
                    ),

                "closed_set_accuracy":
                    float(
                        np.mean(
                            known_correct
                        )
                    ),

                "closed_set_balanced_accuracy":
                    float(
                        balanced_accuracy_score(
                            known_true,
                            known_prediction,
                        )
                    ),

                "closed_set_macro_f1":
                    float(
                        f1_score(
                            known_true,
                            known_prediction,
                            average="macro",
                            zero_division=0,
                        )
                    ),

                "closed_set_top5_accuracy":
                    float(
                        np.mean(
                            top5_correct
                        )
                    ),

                "known_acceptance_rate":
                    float(
                        np.mean(
                            known_accept
                        )
                    ),

                "known_false_rejection_rate":
                    float(
                        np.mean(
                            ~known_accept
                        )
                    ),

                "unknown_false_acceptance_rate":
                    float(
                        np.mean(
                            unknown_accept
                        )
                    ),

                "unknown_true_rejection_rate":
                    float(
                        np.mean(
                            ~unknown_accept
                        )
                    ),

                "known_correct_and_accepted_rate":
                    float(
                        np.mean(
                            known_correct_accepted
                        )
                    ),

                "known_accuracy_among_accepted":
                    (
                        float(
                            np.mean(
                                known_correct[
                                    known_accept
                                ]
                            )
                        )
                        if accepted_count
                        else float("nan")
                    ),

                "open_set_overall_accuracy":
                    float(
                        open_correct
                        / open_total
                    ),

                "unknown_detection_auroc":
                    float(
                        roc_auc_score(
                            labels_unknown,
                            scores,
                        )
                    ),

                "unknown_detection_aupr":
                    float(
                        average_precision_score(
                            labels_unknown,
                            scores,
                        )
                    ),

                "equal_error_rate_approx":
                    float(
                        eer_value(
                            labels_unknown,
                            scores,
                        )
                    ),

                "oscr_auc":
                    float(
                        oscr_auc_function(
                            known_correct,
                            known_energy,
                            unknown_energy,
                        )
                    ),

                "known_mean_energy":
                    float(
                        np.mean(
                            known_energy
                        )
                    ),

                "unknown_mean_energy":
                    float(
                        np.mean(
                            unknown_energy
                        )
                    ),

                "known_unknown_energy_gap":
                    float(
                        np.mean(
                            unknown_energy
                        )
                        -
                        np.mean(
                            known_energy
                        )
                    ),

                "known_acceptance_rate_macro_user":
                    float(
                        known_accept_macro
                    ),

                "unknown_false_acceptance_rate_macro_user":
                    float(
                        unknown_false_accept_macro
                    ),

                "unknown_true_rejection_rate_macro_user":
                    float(
                        unknown_true_reject_macro
                    ),

                "balanced_detection_rate_macro_user":
                    float(
                        balanced_detection_macro
                    ),

                "known_blocks":
                    int(
                        len(
                            known_meta
                        )
                    ),

                "unknown_blocks":
                    int(
                        len(
                            unknown_meta
                        )
                    ),
            }
        )

        # --------------------------------------------------------
        # Per-block prediction output
        # --------------------------------------------------------

        prediction_sets = [
            (
                known_meta,
                known_energy,
                known_prediction,
                known_accept,
                known_correct,
                "known",
            ),
            (
                unknown_meta,
                unknown_energy,
                unknown_prediction,
                unknown_accept,
                ~unknown_accept,
                "unknown",
            ),
        ]

        for (
            meta,
            energy,
            pred,
            accepted,
            correct,
            label,
        ) in prediction_sets:

            frame = meta.copy()

            frame.insert(
                0,
                "seed",
                seed,
            )

            frame.insert(
                1,
                "method",
                METHOD,
            )

            frame[
                "aggregation_seconds"
            ] = duration

            frame[
                "predicted_class"
            ] = pred

            frame[
                "predicted_player_id"
            ] = [
                inverse[
                    int(value)
                ]
                for value in pred
            ]

            frame[
                "energy_score"
            ] = energy

            frame[
                "threshold"
            ] = threshold

            frame[
                "accepted"
            ] = accepted

            frame[
                "correct"
            ] = correct

            frame[
                "known_or_unknown"
            ] = label

            if label == "known":

                frame[
                    "true_class"
                ] = known_true

            else:

                frame[
                    "true_class"
                ] = -1

            prediction_frames.append(
                frame
            )

    del model

    return (
        pd.DataFrame(
            metric_rows
        ),
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
    )


# ================================================================
# MEAN / STANDARD DEVIATION ACROSS SEEDS
# ================================================================

def summarize(
    metrics: pd.DataFrame,
) -> pd.DataFrame:

    identifiers = {
        "method",
        "seed",
        "aggregation_seconds",
    }

    numeric = [
        column
        for column
        in metrics.columns
        if (
            column
            not in identifiers
            and
            pd.api.types.is_numeric_dtype(
                metrics[column]
            )
        )
    ]

    rows = []

    for (
        method,
        duration,
    ), group in metrics.groupby(
        [
            "method",
            "aggregation_seconds",
        ],
        sort=True,
    ):

        row = {
            "method":
                method,

            "aggregation_seconds":
                int(duration),

            "seed_count":
                int(
                    group[
                        "seed"
                    ].nunique()
                ),
        }

        for column in numeric:

            values = pd.to_numeric(
                group[column],
                errors="coerce",
            )

            row[
                f"{column}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{column}_std"
            ] = float(
                values.std(
                    ddof=1
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ================================================================
# LOAD THE MATCHED NONLINEAR FULL RG-POSR RESULTS
# ================================================================

def load_full_rg_posr_metrics() -> pd.DataFrame:

    frames = []

    for seed in SEEDS:

        seed_root = (
            REVIEW_MODELS_ROOT
            / f"protocol_seed{seed}_nonlinear"
        )

        candidates = [
            seed_root
            / f"seed_{seed}_metrics.csv",

            seed_root
            / "seed_metrics.csv",
        ]

        path = None

        for candidate in candidates:

            if candidate.is_file():

                path = candidate
                break

        if path is None:

            raise FileNotFoundError(
                "Could not find the nonlinear RG-POSR "
                f"metrics for seed {seed}.\n"
                f"Checked:\n"
                + "\n".join(
                    str(x)
                    for x in candidates
                )
            )

        frame = pd.read_csv(
            path
        )

        # Keep only this seed when a combined file is used.
        if "seed" in frame.columns:

            numeric_seed = pd.to_numeric(
                frame["seed"],
                errors="coerce",
            )

            frame = frame.loc[
                numeric_seed.eq(seed)
            ].copy()

        else:

            frame = frame.copy()

            frame[
                "seed"
            ] = seed

        if (
            "aggregation_seconds"
            not in frame.columns
        ):

            raise RuntimeError(
                f"No aggregation_seconds column in:\n{path}"
            )

        frame = frame.loc[
            pd.to_numeric(
                frame[
                    "aggregation_seconds"
                ],
                errors="coerce",
            ).isin(
                DURATIONS
            )
        ].copy()

        # In case the result file contains more than one
        # fusion method, keep the nonlinear rows.
        if (
            "method"
            in frame.columns
            and
            frame[
                "aggregation_seconds"
            ].duplicated().any()
        ):

            method_text = (
                frame[
                    "method"
                ]
                .astype(str)
                .str.lower()
            )

            nonlinear_mask = (
                method_text
                .str.contains(
                    "nonlinear",
                    regex=False,
                )
            )

            if nonlinear_mask.any():

                frame = frame.loc[
                    nonlinear_mask
                ].copy()

        duration_counts = (
            frame
            .groupby(
                "aggregation_seconds"
            )
            .size()
        )

        for duration in DURATIONS:

            count = int(
                duration_counts.get(
                    duration,
                    0,
                )
            )

            if count != 1:

                methods = (
                    sorted(
                        frame[
                            "method"
                        ]
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    if (
                        "method"
                        in frame.columns
                    )
                    else []
                )

                raise RuntimeError(
                    "Could not uniquely identify "
                    "the nonlinear full RG-POSR row.\n"
                    f"Seed={seed}, "
                    f"duration={duration}, "
                    f"count={count}, "
                    f"methods={methods}, "
                    f"file={path}"
                )

        frames.append(
            frame
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    duplicates = result.duplicated(
        [
            "seed",
            "aggregation_seconds",
        ]
    )

    if duplicates.any():

        raise RuntimeError(
            "Duplicate full RG-POSR "
            "(seed, aggregation_seconds) rows."
        )

    return result


# ================================================================
# ENERGY VS FULL NONLINEAR RG-POSR
# ================================================================

def paired_comparison(
    energy_metrics: pd.DataFrame,
    full_metrics: pd.DataFrame,
) -> pd.DataFrame:

    keys = [
        "seed",
        "aggregation_seconds",
    ]

    requested_metrics = [
        "closed_set_accuracy",
        "known_acceptance_rate",
        "unknown_true_rejection_rate",
        "open_set_overall_accuracy",
        "unknown_detection_auroc",
        "unknown_detection_aupr",
        "equal_error_rate_approx",
        "oscr_auc",
        "known_acceptance_rate_macro_user",
        "unknown_false_acceptance_rate_macro_user",
        "unknown_true_rejection_rate_macro_user",
        "balanced_detection_rate_macro_user",
    ]

    common = [
        metric
        for metric
        in requested_metrics
        if (
            metric
            in energy_metrics.columns
            and
            metric
            in full_metrics.columns
        )
    ]

    if not common:

        raise RuntimeError(
            "No common metrics were found "
            "between energy and full RG-POSR."
        )

    paired = (
        energy_metrics[
            keys + common
        ]
        .merge(
            full_metrics[
                keys + common
            ],
            on=keys,
            how="inner",
            suffixes=(
                "_energy",
                "_full_rg_posr",
            ),
            validate="one_to_one",
        )
    )

    expected_rows = (
        len(SEEDS)
        * len(DURATIONS)
    )

    if len(paired) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows} paired rows, "
            f"found {len(paired)}."
        )

    for metric in common:

        paired[
            f"{metric}_energy_minus_full"
        ] = (
            paired[
                f"{metric}_energy"
            ]
            -
            paired[
                f"{metric}_full_rg_posr"
            ]
        )

    return paired


# ================================================================
# EXACT FIVE-SEED SIGN-FLIP TEST
# ================================================================

def exact_signflip_p(
    differences: np.ndarray,
) -> float:

    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    differences = differences[
        np.isfinite(
            differences
        )
    ]

    n = len(
        differences
    )

    if n == 0:

        return float(
            "nan"
        )

    observed = abs(
        float(
            np.mean(
                differences
            )
        )
    )

    null_statistics = []

    for signs in itertools.product(
        (-1.0, 1.0),
        repeat=n,
    ):

        signs_array = np.asarray(
            signs,
            dtype=np.float64,
        )

        statistic = abs(
            float(
                np.mean(
                    differences
                    * signs_array
                )
            )
        )

        null_statistics.append(
            statistic
        )

    null_statistics = np.asarray(
        null_statistics,
        dtype=np.float64,
    )

    return float(
        np.mean(
            null_statistics
            >=
            (
                observed
                - 1e-15
            )
        )
    )


def paired_signflip_statistics(
    paired: pd.DataFrame,
) -> pd.DataFrame:

    suffix = (
        "_energy_minus_full"
    )

    delta_columns = [
        column
        for column
        in paired.columns
        if column.endswith(
            suffix
        )
    ]

    rows = []

    for duration in DURATIONS:

        subset = paired.loc[
            paired[
                "aggregation_seconds"
            ].eq(duration)
        ].copy()

        for delta_column in delta_columns:

            metric = (
                delta_column[
                    :-len(suffix)
                ]
            )

            differences = (
                pd.to_numeric(
                    subset[
                        delta_column
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=np.float64
                )
            )

            differences = differences[
                np.isfinite(
                    differences
                )
            ]

            mean_diff = float(
                np.mean(
                    differences
                )
            )

            if len(differences) > 1:

                sd_diff = float(
                    np.std(
                        differences,
                        ddof=1,
                    )
                )

            else:

                sd_diff = float(
                    "nan"
                )

            if (
                np.isfinite(
                    sd_diff
                )
                and
                sd_diff > 0
            ):

                cohens_dz = (
                    mean_diff
                    / sd_diff
                )

            elif mean_diff == 0:

                cohens_dz = 0.0

            else:

                cohens_dz = float(
                    "inf"
                )

            rows.append(
                {
                    "aggregation_seconds":
                        duration,

                    "metric":
                        metric,

                    "mean_diff_energy_minus_full":
                        mean_diff,

                    "sd_diff":
                        sd_diff,

                    "cohens_dz":
                        cohens_dz,

                    "exact_signflip_p":
                        exact_signflip_p(
                            differences
                        ),

                    "seed_count":
                        int(
                            len(
                                differences
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ================================================================
# PATH / CHECKPOINT VALIDATION
# ================================================================

def validate_paths():

    missing = []

    required_paths = [
        SOURCE_SCRIPT,
        PROCESSED_ROOT,
        FROZEN_ROOT,
        BLIND_CACHE_ROOT,
    ]

    for path in required_paths:

        if not path.exists():

            missing.append(
                str(path)
            )

    for seed in SEEDS:

        for fold in FOLDS:

            path = calibration_checkpoint(
                seed,
                fold,
            )

            if not path.is_file():

                missing.append(
                    str(path)
                )

        path = final_checkpoint(
            seed
        )

        if not path.is_file():

            missing.append(
                str(path)
            )

    if missing:

        raise FileNotFoundError(
            "Required files/directories are missing:\n\n"
            +
            "\n".join(
                missing
            )
        )


# ================================================================
# OUTPUT VALIDATION
# ================================================================

def validation_report(
    metrics,
    selected,
    folds,
    predictions,
    paired,
):

    errors = []

    expected_metrics = (
        len(SEEDS)
        *
        len(DURATIONS)
    )

    expected_folds = (
        len(SEEDS)
        *
        len(FOLDS)
        *
        len(DURATIONS)
        *
        len(TEMPERATURE_CANDIDATES)
    )

    if (
        len(metrics)
        != expected_metrics
    ):

        errors.append(
            f"Expected {expected_metrics} metric rows; "
            f"found {len(metrics)}."
        )

    if (
        len(selected)
        != expected_metrics
    ):

        errors.append(
            f"Expected {expected_metrics} "
            f"selected-threshold rows; "
            f"found {len(selected)}."
        )

    if (
        len(folds)
        != expected_folds
    ):

        errors.append(
            f"Expected {expected_folds} "
            f"calibration-fold rows; "
            f"found {len(folds)}."
        )

    if predictions.empty:

        errors.append(
            "Prediction output is empty."
        )

    if (
        len(paired)
        != expected_metrics
    ):

        errors.append(
            f"Expected {expected_metrics} paired rows; "
            f"found {len(paired)}."
        )

    required_metric_columns = [
        "threshold",
        "unknown_detection_auroc",
        "unknown_detection_aupr",
        "oscr_auc",
    ]

    for column in required_metric_columns:

        if column not in metrics.columns:

            errors.append(
                f"Missing metric column: {column}"
            )

        elif (
            metrics[
                column
            ]
            .isna()
            .any()
        ):

            errors.append(
                f"Invalid NaN values in metric: {column}"
            )

    return {
        "method":
            METHOD,

        "evaluation_only":
            True,

        "training_performed":
            False,

        "source_checkpoints_modified":
            False,

        "metric_rows":
            int(
                len(metrics)
            ),

        "selected_threshold_rows":
            int(
                len(selected)
            ),

        "calibration_fold_rows":
            int(
                len(folds)
            ),

        "prediction_rows":
            int(
                len(predictions)
            ),

        "paired_rows":
            int(
                len(paired)
            ),

        "errors":
            errors,

        "passed":
            not errors,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    started = time.time()

    print(
        "=" * 72
    )

    print(
        "NONLINEAR REVIEW RG-POSR ENERGY BASELINE — TUNED TEMPERATURE"
    )

    print(
        "EVALUATION ONLY — NO TRAINING"
    )

    print(
        "=" * 72
    )

    validate_paths()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Output: {OUTPUT}"
    )

    # ------------------------------------------------------------
    # PRECHECK 1: exact nonlinear checkpoint/model compatibility
    # ------------------------------------------------------------

    print(
        "\nPRECHECK: loading nonlinear calibration checkpoint..."
    )

    (
        test_model,
        test_mapping,
        test_scale,
    ) = build_model(
        calibration_checkpoint(
            42,
            0,
        )
    )

    print(
        "CHECKPOINT LOAD OK"
    )

    print(
        f"Model class: {type(test_model).__name__}"
    )

    print(
        f"Classes: {len(test_mapping)}"
    )

    print(
        f"ArcFace scale: {test_scale}"
    )

    del test_model

    # ------------------------------------------------------------
    # REVIEW BLIND PROTOCOL
    # ------------------------------------------------------------

    RG.REVIEW_BLIND_CACHE_ROOT = (
        BLIND_CACHE_ROOT
    )

    print(
        "\nLoading frozen review protocol..."
    )

    (
        split,
        index,
        means,
        stds,
        raw_reliability,
        standardized_reliability,
        target_placeholder,
        descriptor_names,
        rel_median,
        rel_scale,
    ) = RG.load_protocol_review(
        PROCESSED_ROOT,
        FROZEN_ROOT,
        True,
    )

    # ------------------------------------------------------------
    # REMAP STALE ABSOLUTE CACHE PATHS
    # ------------------------------------------------------------
    # The frozen window index was created on G:\ and therefore may
    # contain absolute resolved_cache_path values pointing to the old
    # processed-data root.  The protocol metadata remain unchanged; only
    # the local filesystem prefix is remapped to the currently resolved
    # PROCESSED_ROOT before any DataLoader attempts to open the NPZ files.
    if "resolved_cache_path" not in index.columns:
        raise RuntimeError(
            "Frozen review index has no resolved_cache_path column."
        )

    def remap_resolved_cache_path(value):
        raw = str(value)
        current = Path(raw)
        if current.is_file():
            return str(current)

        normalized = raw.replace("/", "\\")
        marker = "\\resampled_sessions\\"
        lower = normalized.lower()
        marker_index = lower.find(marker.lower())
        if marker_index >= 0:
            suffix = normalized[marker_index + len(marker):]
            candidate = PROCESSED_ROOT / "resampled_sessions" / Path(suffix)
            return str(candidate)

        # Fallback for any path stored directly relative to the processed root.
        candidate = PROCESSED_ROOT / Path(normalized).name
        return str(candidate)

    index = index.copy()
    index["resolved_cache_path"] = index["resolved_cache_path"].map(
        remap_resolved_cache_path
    )

    missing_cache_paths = [
        value
        for value in index["resolved_cache_path"].astype(str).unique()
        if not Path(value).is_file()
    ]
    if missing_cache_paths:
        preview = "\n".join(missing_cache_paths[:10])
        raise FileNotFoundError(
            f"After remapping to {PROCESSED_ROOT}, "
            f"{len(missing_cache_paths)} cache files are still missing.\n"
            f"First missing paths:\n{preview}"
        )

    print(
        f"CACHE PATH REMAP OK: "
        f"{index['resolved_cache_path'].nunique()} NPZ files resolved under "
        f"{PROCESSED_ROOT}"
    )

    print(
        "REVIEW PROTOCOL LOAD OK"
    )

    print(
        f"Protocol return items: 10"
    )

    print(
        f"Window-index rows: {len(index)}"
    )

    print(
        f"Raw reliability shape: "
        f"{raw_reliability.shape}"
    )

    print(
        f"Standardized reliability shape: "
        f"{standardized_reliability.shape}"
    )

    print(
        f"Descriptor names: {descriptor_names}"
    )

    # ------------------------------------------------------------
    # Verify that the previous nonlinear RG-POSR result files
    # needed for the final paired comparison can be resolved BEFORE
    # spending time on inference.
    # ------------------------------------------------------------

    print(
        "\nChecking nonlinear RG-POSR comparison metrics..."
    )

    full_rg_posr_metrics = (
        load_full_rg_posr_metrics()
    )

    print(
        "NONLINEAR RG-POSR COMPARISON FILES OK"
    )

    print(
        f"Full RG-POSR rows: "
        f"{len(full_rg_posr_metrics)}"
    )

    # ------------------------------------------------------------
    # ENERGY EVALUATION
    # ------------------------------------------------------------

    selected_all = []
    folds_all = []
    metrics_all = []
    predictions_all = []

    for seed in SEEDS:

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"SEED {seed}"
        )

        print(
            "=" * 72
        )

        print(
            f"CALIBRATING ENERGY SCORE + TEMPERATURE: seed={seed}",
            flush=True,
        )

        (
            selected,
            folds,
        ) = calibrate_seed(
            seed,
            split,
            index,
            means,
            stds,
            raw_reliability,
            target_placeholder,
            rel_median,
            rel_scale,
        )

        selected.to_csv(
            OUTPUT
            / f"seed_{seed}_selected_thresholds.csv",
            index=False,
        )

        folds.to_csv(
            OUTPUT
            / f"seed_{seed}_calibration_folds.csv",
            index=False,
        )

        print(
            f"FINAL ENERGY EVALUATION: seed={seed}",
            flush=True,
        )

        (
            metrics,
            predictions,
        ) = evaluate_seed(
            seed,
            selected,
            split,
            index,
            means,
            stds,
            raw_reliability,
            target_placeholder,
            rel_median,
            rel_scale,
        )

        metrics.to_csv(
            OUTPUT
            / f"seed_{seed}_metrics.csv",
            index=False,
        )

        predictions.to_csv(
            OUTPUT
            / f"seed_{seed}_predictions.csv",
            index=False,
        )

        selected_all.append(
            selected
        )

        folds_all.append(
            folds
        )

        metrics_all.append(
            metrics
        )

        predictions_all.append(
            predictions
        )

        print(
            f"COMPLETED ENERGY BASELINE SEED {seed}",
            flush=True,
        )

    # ------------------------------------------------------------
    # MERGE FIVE SEEDS
    # ------------------------------------------------------------

    selected_frame = pd.concat(
        selected_all,
        ignore_index=True,
    )

    fold_frame = pd.concat(
        folds_all,
        ignore_index=True,
    )

    metric_frame = pd.concat(
        metrics_all,
        ignore_index=True,
    )

    prediction_frame = pd.concat(
        predictions_all,
        ignore_index=True,
    )

    summary_frame = summarize(
        metric_frame
    )

    # ------------------------------------------------------------
    # PAIRED ENERGY VS NONLINEAR RG-POSR
    # ------------------------------------------------------------

    paired_frame = paired_comparison(
        metric_frame,
        full_rg_posr_metrics,
    )

    signflip_frame = (
        paired_signflip_statistics(
            paired_frame
        )
    )

    # ------------------------------------------------------------
    # SAVE COMBINED RESULTS
    # ------------------------------------------------------------

    selected_frame.to_csv(
        OUTPUT
        / "selected_thresholds.csv",
        index=False,
    )

    fold_frame.to_csv(
        OUTPUT
        / "calibration_fold_metrics.csv",
        index=False,
    )

    metric_frame.to_csv(
        OUTPUT
        / "seed_metrics.csv",
        index=False,
    )

    prediction_frame.to_csv(
        OUTPUT
        / "test_predictions.csv",
        index=False,
    )

    summary_frame.to_csv(
        OUTPUT
        / "metrics_summary.csv",
        index=False,
    )

    paired_frame.to_csv(
        OUTPUT
        / "paired_full_rg_posr_comparison.csv",
        index=False,
    )

    signflip_frame.to_csv(
        OUTPUT
        / "paired_signflip_statistics.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    validation = validation_report(
        metric_frame,
        selected_frame,
        fold_frame,
        prediction_frame,
        paired_frame,
    )

    (
        OUTPUT
        / "validation_report.json"
    ).write_text(
        json.dumps(
            validation,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {

        "method":
            METHOD,

        "description":
            (
                "Energy-score rejection on the frozen "
                "nonlinear review RG-POSR encoder and "
                "ArcFace prototype logits"
            ),

        "evaluation_only":
            True,

        "training_performed":
            False,

        "source_checkpoints_modified":
            False,

        "temperature_candidates":
            list(
                TEMPERATURE_CANDIDATES
            ),

        "temperature_selection":
            (
                "Session-1 pseudo-unknown calibration; "
                "selected separately per seed and aggregation duration "
                "by mean balanced detection across five folds, "
                "with deterministic tie-breaks and median selected-T threshold"
            ),

        "seeds":
            list(SEEDS),

        "calibration_folds":
            list(FOLDS),

        "aggregation_seconds":
            list(DURATIONS),

        "processed_root":
            str(
                PROCESSED_ROOT
            ),

        "frozen_root":
            str(
                FROZEN_ROOT
            ),

        "review_models_root":
            str(
                REVIEW_MODELS_ROOT
            ),

        "blind_cache_root":
            str(
                BLIND_CACHE_ROOT
            ),

        "output_root":
            str(
                OUTPUT
            ),

        "elapsed_minutes":
            (
                time.time()
                - started
            )
            / 60.0,

        "validation_passed":
            validation[
                "passed"
            ],
    }

    (
        OUTPUT
        / "experiment_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # FINAL TERMINAL OUTPUT
    # ------------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TUNED ENERGY-SCORE BASELINE COMPLETED"
    )

    print(
        "=" * 72
    )

    print(
        "\nVALIDATION"
    )

    print(
        json.dumps(
            validation,
            indent=2,
        )
    )

    print(
        "\nMETRICS SUMMARY"
    )

    print(
        summary_frame
        .round(6)
        .to_string(
            index=False
        )
    )

    print(
        "\nPAIRED SIGN-FLIP STATISTICS"
    )

    print(
        signflip_frame
        .round(6)
        .to_string(
            index=False
        )
    )

    return {

        "selected_thresholds":
            selected_frame,

        "calibration_folds":
            fold_frame,

        "seed_metrics":
            metric_frame,

        "metrics_summary":
            summary_frame,

        "paired_comparison":
            paired_frame,

        "paired_signflip_statistics":
            signflip_frame,

        "validation":
            validation,
    }


# ================================================================
# RUN IN NOTEBOOK
# ================================================================

results = main()
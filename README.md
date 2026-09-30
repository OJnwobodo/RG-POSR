# RG-POSR: Reliability-Gated Prototypical Open-Set Recognition

This repository contains the code, frozen experimental protocol, evaluation scripts, launch scripts, and revision-analysis outputs for the manuscript:

**Reliability-Gated Prototypical Open-Set Recognition for Cross-Session Behavioural Biometrics in Extended Reality**

The work investigates cross-session open-set behavioural biometrics from multimodal XR signals using the **Who Is Alyx?** dataset.

---

## Overview

RG-POSR combines:

- modality-specific temporal identity encoders;
- trainable identity prototypes;
- signal-derived modality reliability estimation;
- observation-dependent multimodal fusion;
- reliability-conditioned prototype-distance rejection.

The four XR modalities are:

- head/HMD motion;
- left-controller motion;
- right-controller motion;
- gaze.

Reliability is estimated from descriptors computed directly from the observed standardised signal. Corruption type, corruption severity, and other degradation metadata are not supplied to the reliability estimator at inference time.

---

## Repository Contents

### Core RG-POSR implementation

| File | Description |
|---|---|
| `train_review_rg_posr.py` | Main revised RG-POSR training and evaluation runner |
| `train_review_rg_posr_base.py` | Base implementation used by the revised runner |
| `blind_reliability.py` | Blind reliability-descriptor and reliability-estimation functions |
| `freeze_blind_reliability_cache.py` | Utility for generating/freezing blind reliability descriptors |
| `freeze_review_protocol.py` | Utility for constructing/freezing the review protocol |

---

### Frozen protocol files

| File | Description |
|---|---|
| `frozen_protocol_v1.json` | Frozen experimental-protocol definition |
| `frozen_protocol_v1.yaml` | YAML representation of the frozen protocol |
| `corruption_protocol_frozen_v1.yaml` | Frozen missing/corrupted-modality evaluation protocol |

These files document the protocol structure used for the revised experiments.

---

### Main five-seed experiment launchers

The primary experiments use five fixed random seeds:

```text
42
52
62
72
82
```

The corresponding Windows batch files are:

```text
run_protocol_seed42_all_variants.bat
run_protocol_seed52_all_variants.bat
run_protocol_seed62_all_variants.bat
run_protocol_seed72_resume_safe.bat
run_protocol_seed82_resume_safe.bat
```

The four reliability formulations evaluated under the same frozen protocol are:

```text
uniform
direct
linear
nonlinear
```

---

## Experimental Protocol

The frozen cross-session protocol contains:

- **69 participants in total**
- **49 enrolled participants**
- **20 participant-disjoint final unknown participants**
- **5 random seeds**
- **5 participant-disjoint pseudo-unknown calibration folds**
- **10-s base identity windows**
- aggregation durations of **30 s, 60 s, and 120 s**

### Session usage

**Session 1**

Used for:

- model training;
- validation;
- participant-disjoint pseudo-unknown calibration;
- selection of open-set parameters.

**Session 2**

Reserved for:

- frozen cross-session evaluation of enrolled users;
- evaluation of participant-disjoint final unknown users.

Final unknown users are not used for training, preprocessing fitting, checkpoint selection, reliability-parameter selection, or rejection-threshold calibration.

---

## Model Configuration

The principal revised configuration uses:

```text
GRU layers per modality:        1
GRU hidden dimension:          64
Identity embedding dimension:  64
Reliability hidden dimension:  32
Reliability activation:        GELU
Reliability-head dropout:      0.2
Batch size:                    64
Maximum epochs:                30
Early-stopping patience:       8
Learning rate:                 1e-3
Weight decay:                  1e-4
ArcFace-style scale:           30
Angular margin:                0.3
Monotonic margin:              0.15
Corruption probability:        0.6
```

Optimisation uses **AdamW**.

---

## Reliability Descriptors

The learned reliability formulations use eight descriptors computed from each observed standardised modality sequence:

1. inactive-frame fraction;
2. flatline-transition fraction;
3. median frame norm;
4. 95th-percentile frame norm;
5. median temporal-difference norm;
6. 95th-percentile temporal-difference norm;
7. temporal energy ratio;
8. mean feature standard deviation.

No corruption label or severity value is passed to the reliability estimator.

---

## Reliability-Learning Objective

The full nonlinear formulation uses:

```text
lambda_reg  = 1.0
lambda_gate = 0.5
lambda_mono = 0.5
```

where the three reliability-specific objectives correspond to:

- fidelity regression;
- gate alignment;
- monotonic consistency.

The identity objective is trained jointly with these terms.

---

## Reviewer #8 Loss-Term Ablation

A focused loss-term ablation was performed to compare the complete nonlinear reliability objective against direct fidelity regression alone.

### Full nonlinear objective

```text
lambda_reg  = 1.0
lambda_gate = 0.5
lambda_mono = 0.5
```

### Regression-only nonlinear objective

```text
lambda_reg  = 1.0
lambda_gate = 0.0
lambda_mono = 0.0
```

All other experimental settings were retained, including:

- nonlinear reliability architecture;
- participant protocol;
- five random seeds;
- five calibration folds;
- aggregation durations;
- synthetic corruption augmentation;
- optimisation settings;
- checkpoint-selection procedure;
- Session 1 pseudo-unknown calibration;
- frozen Session 2 evaluation.

The corresponding analysis outputs are:

```text
reviewer8_full_vs_reg_only_summary.csv
reviewer8_full_vs_reg_only_paired_stats.csv
reviewer8_full_vs_reg_only_seedwise.csv
```

This ablation removes the gate-alignment and monotonic-consistency objectives jointly. It therefore does **not** separately identify the individual contribution of those two auxiliary objectives.

---

## Open-Set Calibration

The reliability-conditioning exponent is selected from:

```text
gamma ∈ {0, 0.5, 1, 2}
```

Selection is performed using Session 1 participant-disjoint pseudo-unknown calibration only.

The final selected operating point is frozen before Session 2 evaluation.

The reported open-set measures include:

- known-user acceptance;
- unknown-user rejection;
- balanced detection;
- open-set overall accuracy;
- AUROC;
- AUPR;
- interpolated EER;
- OSCR.

Paired comparisons across the five seeds use:

- two-sided exact sign-flip tests;
- paired Cohen's \(d_z\).

---

## Energy-Score Comparator

The final tuned energy-score evaluator is:

```text
evaluate_energy_review_nonlinear_tuned_v2.py
```

It retains the trained nonlinear RG-POSR representation and prototypes but replaces prototype-distance rejection with an energy score.

The candidate temperature set is:

```text
T ∈ {0.25, 0.5, 1, 2, 4, 8, 16}
```

Temperature selection and threshold calibration use **Session 1 only**.

The selected temperature and threshold are then frozen for Session 2 evaluation.

---

## Blind Robustness Evaluation

The final robustness evaluator is:

```text
evaluate_review_blind_robustness.py
```

The robustness experiment evaluates already trained nonlinear RG-POSR and uniform-fusion models.

No:

- model retraining;
- prototype update;
- reliability-head adaptation;
- gamma reselection;
- threshold adjustment;
- Session 2 recalibration

is performed during the robustness experiment.

### Corruption families

The controlled degradation families are:

```text
complete modality dropout
additive Gaussian noise
contiguous temporal-burst loss
```

### Severities

```text
0.25
0.50
0.75
1.00
```

Corruptions are evaluated separately for:

```text
head/HMD
left controller
right controller
gaze
```

---

## Calibration Audit Files

The repository contains:

```text
seed_42_calibration_folds.csv
seed_52_calibration_folds.csv
seed_62_calibration_folds.csv
seed_72_calibration_folds.csv
seed_82_calibration_folds.csv
```

These files contain fold-level calibration information including:

- random seed;
- calibration fold;
- aggregation duration;
- number of included users;
- number of pseudo-unknown users;
- threshold;
- known/pseudo-unknown calibration summaries;
- participant-macro acceptance/rejection quantities;
- balanced calibration performance.

> **Note:** these CSV files are calibration audit outputs and do not contain the participant identifiers assigned to each fold.

---

## Analysis Notebook

The repository also contains:

```text
RG_POSR.ipynb
```

This notebook contains revision-stage analysis and result-verification work associated with the manuscript experiments.

---

## Running the Main Experiments

The provided batch files reproduce the main five-seed reliability-model training workflow.

Before running the scripts, update local paths so that they point to:

1. the processed *Who Is Alyx?* dataset;
2. the frozen protocol directory;
3. the desired experiment-output directory.

Example:

```powershell
.\run_protocol_seed42_all_variants.bat
.\run_protocol_seed52_all_variants.bat
.\run_protocol_seed62_all_variants.bat
.\run_protocol_seed72_resume_safe.bat
.\run_protocol_seed82_resume_safe.bat
```

The batch files retain the common protocol while evaluating the reliability variants using matched random seeds and participant partitions.

---

## Running the Energy-Score Evaluation

After the nonlinear RG-POSR checkpoints have been generated, update the local path constants in:

```text
evaluate_energy_review_nonlinear_tuned_v2.py
```

and run:

```powershell
python evaluate_energy_review_nonlinear_tuned_v2.py
```

The script performs Session 1-only temperature and threshold calibration before frozen Session 2 evaluation.

---

## Running the Robustness Evaluation

Inspect the available arguments using:

```powershell
python evaluate_review_blind_robustness.py --help
```

Then provide the required local paths to:

- processed data;
- frozen protocol;
- trained model outputs;
- robustness result directory.

---

## Data Availability

The raw *Who Is Alyx?* dataset is **not redistributed in this repository**.

Users should obtain the dataset from its original source and prepare the processed data required by the supplied experimental code.

The experiment code expects the processed data structure used by the frozen protocol.

---

## Path Portability

The experiments were developed and executed on Windows.

Some scripts may contain absolute paths from the original experiment workstation. These must be changed to corresponding local paths before reproduction.

Changing local filesystem paths does not require changing:

- random seeds;
- participant/session roles;
- calibration-fold logic;
- aggregation durations;
- frozen Session 1/Session 2 protocol.

---


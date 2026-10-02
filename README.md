# RG-POSR

## Reliability-Gated Prototypical Open-Set Recognition for Cross-Session Behavioural Biometrics in Virtual Reality

This repository contains the implementation and reproducibility materials accompanying the paper:

**Reliability-Gated Prototypical Open-Set Recognition for Cross-Session Behavioural Biometrics in Virtual Reality**

by Onyeka J. Nwobodo, Jeremiah O. Abimbola, and Godlove Suila Kuaban.

RG-POSR is a multimodal open-set behavioural-biometric framework for cross-session virtual-reality data. The implementation supports head/HMD, left-controller, right-controller, and gaze modalities, together with reliability-aware multimodal fusion and prototype-based open-set recognition.

For the complete method description, experimental protocol, evaluation methodology, statistical analysis, and results, please refer to the paper.

---

## Repository contents

The repository contains materials used to implement and evaluate the experiments reported in the paper, including:

- RG-POSR model implementation;
- reliability-estimation and multimodal-fusion components;
- prototype-based open-set recognition;
- frozen participant and session protocol definitions;
- training and experiment-launch scripts;
- Session-1 calibration utilities;
- plain-prototype comparator experiments;
- energy-score evaluation utilities;
- robustness evaluation scripts;
- result aggregation and statistical-analysis utilities;
- supporting revision-analysis materials.

The manuscript should be treated as the authoritative description of the final experimental protocol and reported analyses.

---

## Dataset

The experiments use the publicly available **Who Is Alyx?** virtual-reality behavioural-biometric dataset.

Repository:

https://github.com/cschell/who-is-alyx

Zenodo archive:

https://doi.org/10.5281/zenodo.8379914

Please refer to the original dataset documentation for acquisition details and licensing information.

The participant selection, cross-session partitioning, preprocessing, and open-set evaluation protocol used in RG-POSR are described in the accompanying paper.

---

## Experimental protocol

The repository supports the participant-disjoint two-session evaluation described in the paper.

At a high level:

- Session 1 is used for model development and pseudo-unknown calibration;
- Session 2 is reserved for frozen cross-session evaluation;
- final unknown users are excluded from model development and calibration;
- calibration parameters are selected without using Session-2 observations.

The exact participant split, aggregation procedure, calibration strategy, hyperparameters, random seeds, and evaluation measures are documented in the paper.

---

## RG-POSR variants

The repository includes the reliability formulations evaluated in the paper:

- **Uniform reliability**
- **Direct reliability**
- **Linear reliability**
- **Nonlinear reliability**

It also includes the comparator and ablation configurations used in the manuscript.

For mathematical definitions and interpretation of the different formulations, please refer to the Methods section of the paper.

---

## Reliability estimation

The reported review models use modality-specific signal-derived reliability estimation.

Reliability is inferred from the observed behavioural signal rather than from corruption labels, corruption type, or corruption severity supplied at inference time.

The precise descriptor definitions, normalisation procedure, estimator architecture, supervision targets, and reliability-conditioned fusion equations are given in the paper.

---

## Open-set recognition

RG-POSR combines multimodal identity embeddings with prototype-based identity matching and open-set rejection.

The repository also includes the alternative open-set scoring evaluations described in the manuscript.

Threshold and parameter calibration follow the Session-1-only procedure specified in the paper.

---

## Robustness evaluation

The repository contains the robustness-evaluation utilities used for the controlled sensing-degradation experiments reported in the manuscript.

These evaluators operate on frozen trained models without Session-2 retraining or recalibration.

The corruption definitions, severity settings, evaluation conditions, and interpretation of the robustness experiments are documented in the paper.

---

## Reproducing the study

The general workflow is:

1. Obtain the *Who Is Alyx?* dataset.
2. Prepare the dataset according to the preprocessing protocol described in the paper.
3. Construct the frozen participant and session partitions.
4. Train the required RG-POSR formulations.
5. Perform Session-1 pseudo-unknown calibration.
6. Evaluate the frozen models on Session 2.
7. Run the comparator, ablation, and robustness experiments.
8. Aggregate the outputs and perform the statistical analyses described in the manuscript.

Because the repository contains materials produced at different stages of development and revision, users seeking to reproduce the **reported paper results should follow the final review experiment configuration and the protocol documented in the manuscript**.

---

## Code availability

The RG-POSR implementation and reproducibility materials are available at:

https://github.com/OJnwobodo/RG-POSR

---

## Data availability

The *Who Is Alyx?* dataset is available at:

https://github.com/cschell/who-is-alyx

and is archived on Zenodo at:

https://doi.org/10.5281/zenodo.8379914

---

## Citation

If you use this repository, please cite the associated paper:

> Onyeka J. Nwobodo, Jeremiah O. Abimbola, and Godlove Suila Kuaban.  
> **Reliability-Gated Prototypical Open-Set Recognition for Cross-Session Behavioural Biometrics in Virtual Reality.**

Publication details and the paper DOI can be added here once available.

---

## Contact

For questions about the implementation or reproducibility materials, please use the GitHub issue tracker or contact the corresponding author through the details provided in the paper.

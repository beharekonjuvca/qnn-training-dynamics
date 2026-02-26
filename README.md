# QNN Training

A study of how **depth**, **data re-uploading**, and **entanglement topology** affect training behavior in variational quantum neural networks (QNNs), with reproducible runs, aggregated statistics, and plots.

This repository implements a small experimental framework using **PennyLane** variational circuits and classical toy datasets to measure:

- Final accuracy / loss
- Gradient norm behavior (as a proxy for trainability issues like vanishing gradients / barren-plateau tendencies)
- Runtime
- Full learning curves (loss / grad norm / accuracy)

## Why this matters

Training variational quantum circuits is known to suffer from issues such as vanishing gradients (barren plateaus) and optimization instability. Understanding how architectural choices (depth, re-uploading, entanglement) influence gradient norm is essential for designing practical QNNs on NISQ devices.

This project provides controlled empirical evidence of how these design choices impact trainability.

## What this project does

It runs controlled experiments on a **variational quantum classifier** that outputs an expectation value:

- Model output: ⟨Z⟩ on a chosen measurement wire (default wire 0).
- Labels are mapped to {-1, +1}.
- Training objective is **MSE** between predicted ⟨Z⟩ and labels.
- We also optionally include a **trainable bias** term.  
  (Implementation in `train.py`.)

## Core ideas tested

### Depth sweep (Group A)

We vary the number of trainable layers (e.g., 1, 2, 4, 8) while keeping other settings fixed, then evaluate:

- test accuracy
- test loss
- gradient norm
- total training time

### Data re-uploading sweep (Group B)

To increase expressivity without simply increasing depth, we re-encode the same input multiple times inside the circuit (“re-uploading”).  
This implements the common QNN trick of intertwining data-dependent unitaries with trainable unitaries.

### Entanglement topology comparison (Group C)

We compare different entanglement patterns:

- `chain`: CNOTs in a linear chain
- `all`: fully-connected CNOT entanglement

This is implemented directly in `circuits.py` via `entangle(..., pattern="chain"|"all")`.

## Repository structure

```text
qnn-training-dynamics/
├── circuits.py        # angle encoding, trainable layers, entanglement patterns, QNode factory
├── datasets.py        # moons/circles generators + scaling to [0, π]
├── train.py           # training loop, logging, aggregation to CSV + JSON curves
├── experiments.py     # experiment groups A/B/C definitions + runner
├── res_plot.py        # reads results/agg.csv + curves JSONs and generates figures
├── results/           # runs.csv, agg.csv, curves/*.json (generated)
├── figures/           # exported plots (generated)
└── plots/             # optional local plot outputs
```

# Setup & Reproducibility

This project is fully reproducible using a Python virtual environment.

## 1 Clone the Repository

```bash
git clone https://github.com/beharekonjuvca/qnn-training-dynamics.git
cd qnn-training-dynamics
```

## 2 Create a Virtual Environment

It is strongly recommended to isolate dependencies using a virtual environment.

```bash
python -m venv venv
```

## 3 Activate the Environment

### Windows (CMD / PowerShell)

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 4 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- PennyLane
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- opt_einsum (recommended performance optimization)

## 5 Run All Experiments

```bash
python experiments.py
```

This will automatically generate:

- `results/runs.csv` → Per-seed experiment results
- `results/agg.csv` → Aggregated mean ± std results
- `results/curves/*.json` → Full per-epoch training curves
- `figures/` → Generated plots

## 6 Regenerate Plots (Optional)

If you want to generate plots do:

```bash
python res_plot.py
```

# Reproducibility Notes

- All experiments use fixed random seeds (default: `0, 1, 2`).
- Dataset generation is also seeded.
- Running `experiments.py` fully regenerates all results.

For exact environment reproduction:

```bash
pip freeze > requirements-lock.txt
```

# Recommended Python Version

Tested with:

```
Python 3.10+
```

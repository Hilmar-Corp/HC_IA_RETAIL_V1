# HC_IA_RETAIL_V1

Institutional-grade crypto allocation research stack based on recurrent reinforcement learning, causal data pipelines, and strict out-of-sample evaluation.

## Overview

`HC_IA_RETAIL_V1` is a 4h crypto allocation research repo focused on building and evaluating a recurrent SAC-style agent under a strict institutional research standard.

The objective is not to build a retail signal bot or a one-shot backtest.  
The objective is to build a reproducible, causal, auditable allocation engine that can be evaluated under realistic out-of-sample conditions and later integrated into a broader HilmarCorp research and allocation stack.

The current scope includes:
- 4h market data pipeline
- causal feature engineering
- observation contract enforcement
- recurrent RL training
- strict OOS / walk-forward evaluation
- robustness and leakage checks
- run registry and audit artifacts

## Core research principles

This repo follows a few non-negotiable principles:

- strict causality in data, features, and observations
- no forward leakage across train / validation / test
- explicit observation contract for the agent
- reproducible runs and auditable artifacts
- walk-forward and OOS-first evaluation
- robustness checks before downstream use
- no “lucky run” storytelling

In practical terms:
- all features must be available at decision time
- all evaluation must respect temporal ordering
- all downstream claims must survive OOS scrutiny

## Research target

The research target is a recurrent RL allocation engine operating on 4h data and producing an exposure decision over time.

The design philosophy is:
- continuous allocation logic rather than discrete prediction
- regime-aware and context-aware observations
- robust train / eval separation
- compatibility with future HilmarCorp overlay and regime-engine modules

The repo is designed to support a recurrent SAC-style setup on top of a causal 4h observation space.

## Current project scope

This repository contains the retail/RL side of the stack:
- data preparation for training
- feature engineering
- environment definition
- recurrent agent training
- OOS evaluation
- robustness diagnostics
- artifact and registry utilities

It is meant to work as a downstream consumer of disciplined market-state representations, while preserving its own strict anti-leakage guarantees.

## Repository structure

```text
HC_IA_RETAIL_V1/
├── hc_ia_retail/
│   ├── __init__.py
│   ├── audit.py
│   ├── callbacks.py
│   ├── config.py
│   ├── data.py
│   ├── env.py
│   ├── features.py
│   ├── models.py
│   ├── observation_contract.py
│   ├── regime_data.py
│   ├── robustness.py
│   ├── run_registry.py
│   ├── train.py
│   └── utils.py
├── scripts/
│   ├── data/
│   │   └── build_train_v1_dataset.py
│   ├── debug/
│   │   ├── diagnose_agent_dead.py
│   │   └── inspect_rl_pack.py
│   ├── eval/
│   │   ├── eval_oos.py
│   │   └── sanity_run.py
│   ├── prepare/
│   │   └── fetch_binance_data.py
│   ├── robust/
│   │   ├── run_robustness.py
│   │   └── run_walkforward.py
│   ├── train/
│   │   └── train.py
│   └── hc.py
├── tests/
├── utils/
├── data/
├── requirements.txt
└── README.md
```
## Data frequency and temporal contract

This repo is built around a 4h decision frequency.

Frozen assumptions:
- timezone: UTC
- regular time grid
- no future information in features or observations
- train / validation / test separation must be respected at all times
- no normalization leakage across splits

The expected downstream timing convention is:
- observation at time `t`
- action decided from information available at time `t`
- realized reward from subsequent market movement according to the environment contract

This repo treats temporal discipline as a first-class constraint.

## Model philosophy

The training stack is built around a recurrent SAC-style allocation framework.

The core idea is:
- use sequential observations rather than iid snapshots
- let the model capture temporal market context
- keep the action space compatible with allocation logic
- evaluate under strict OOS conditions rather than in-sample reward narratives

This is not framed as:
- price direction prediction
- candle classification
- retail signal generation

It is framed as:
- sequential allocation under uncertainty
- causal state representation
- robust policy evaluation

## Environment and observation contract

A central part of the repo is the observation contract.

The observation fed to the agent must be:
- causal
- aligned with the execution convention
- free of forbidden columns
- stable across splits
- testable through unit tests

The repo contains dedicated checks around:
- forbidden columns
- forward leakage
- split contamination
- VecNormalize freezing
- window-stack integrity
- funding leakage and observation consistency

This is essential because a recurrent RL agent can silently exploit tiny leakage if the observation pipeline is not aggressively constrained.

## Training

Main training entrypoints include:

```bash
python scripts/train/train.py
```
The exact configuration depends on the training setup defined in the repo.

The expected workflow is:
1. fetch / prepare data
2. build the training dataset
3. train the recurrent policy
4. run strict OOS evaluation
5. inspect artifacts and robustness outputs

## Out-of-sample evaluation

The main OOS evaluation script is:

```bash
python scripts/eval/eval_oos.py
```
This script is intended to produce disciplined evaluation artifacts, including:
- OOS performance summary
- trace files
- comparison vs baseline behaviors
- audit-ready reporting

The evaluation philosophy is:
- OOS first
- walk-forward aware
- no post-hoc storytelling
- compare policies under the same temporal contract

## Robustness and diagnostics

This repo includes dedicated robustness and debug utilities.

Examples:

```bash
python scripts/robust/run_robustness.py
python scripts/robust/run_walkforward.py
python scripts/debug/diagnose_agent_dead.py
python scripts/debug/inspect_rl_pack.py
```

These tools are here because a model that looks good nominally but fails silently under robustness checks is not acceptable.

Typical failure modes of interest include:
- hidden leakage
- dead agent behavior
- unstable observation scaling
- split contamination
- invalid feature availability
- improper normalization across train/test boundaries

## Tests

Run the test suite with:

```bash
pytest -q
```

The test suite is not optional.  
It is part of the research contract.

The repository is only considered usable if:
- observation constraints pass
- causality checks pass
- split integrity checks pass
- no forbidden columns enter the agent observation
- no forward leakage survives the pipeline

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If you want editable local development:

```bash
pip install -e .
```

## Typical workflow

### 1. Fetch or prepare data

```bash
python scripts/prepare/fetch_binance_data.py
python scripts/data/build_train_v1_dataset.py
```

### 2. Train

```bash
python scripts/train/train.py
```

### 3. Evaluate OOS

```bash
python scripts/eval/eval_oos.py
```

### 4. Run robustness checks

```bash
python scripts/robust/run_robustness.py
```

### 5. Inspect diagnostics if needed

```bash
python scripts/debug/diagnose_agent_dead.py
python scripts/debug/inspect_rl_pack.py
```

## What is tracked vs what is not

This repo is intended to track:
- code
- configs
- small, necessary reference data if explicitly required
- tests
- manifests / lightweight artifacts if relevant

This repo should not track:
- large replay buffers
- large checkpoints
- large run folders
- temporary debug dumps
- heavyweight local artifacts

Typical examples that should stay out of Git:
- `runs/`
- replay buffers
- large model checkpoints
- local experiment outputs

If a dataset or artifact is large but necessary, it should be reconstructed through scripts or stored outside the code repository.

## Institutional status of the repo

This repository is part of the HilmarCorp research stack.

Its role is not to be a finished production trading system by itself.  
Its role is to provide a disciplined RL allocation research layer that can later connect to:
- regime-engine outputs
- overlay logic
- broader OOS evaluation frameworks
- future institutional allocation modules

## What this repo is not

This repo is not:
- a retail signal bot
- a chart-pattern notebook
- a strategy zoo
- a leaderboard of lucky seeds
- a direct proof of investable production alpha
- a substitute for full institutional validation

It is a disciplined research repo for causal RL allocation on crypto 4h data.

## Philosophy

The philosophy of the project is simple:

A model is only useful if it survives causality, robustness, and OOS discipline.

That means:
- no leakage
- no lucky backtests
- no hidden train/test contamination
- no silent observation bugs
- no “works on this chart” science

The long-term goal is to build a crypto allocation component that is:
- reproducible
- auditable
- robust
- and usable inside a broader institutional-grade stack

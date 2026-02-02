# ENote: Emotion Detection in Music (Valence & Arousal)

This repository presents an end-to-end workflow for **music emotion recognition** using the **DEAM (Database for Emotion Analysis using Music)** dataset.  
We model emotion along two continuous dimensions — **valence** (positivity/negativity) and **arousal** (energy/intensity) — and demonstrate why **dynamic, time-aware models** outperform static song-level approaches.

---

## Project Overview

Music carries rich emotional information, but much of this variation occurs **over time**.  
This project begins with static regression models and shows why they fail, then transitions to a **dynamic sequence-modelling approach** that leverages DEAM’s per-second emotion annotations.

Key contributions:

- Demonstration that **static averaging collapses emotional signal**
- Reformulation of the problem into a **time-series prediction task**
- Use of **CNN-RNN / BiLSTM architectures** for temporal emotion prediction
- Visual diagnostics (PCA, timelines) used for interpretability in an academic poster

---

## Dataset

We use the **DEAM dataset**, which contains:

- 1,800+ songs and excerpts
- Continuous **per-second valence and arousal annotations**
- Corresponding audio files and extracted acoustic features

**Official dataset link:**  
https://cvml.unige.ch/databases/DEAM/

Place the `DEAM/` folder alongside the notebooks before running the pipeline.

---

## Methodology

### 1. Static Modelling

- Song-level feature aggregation
- CNN / CRNN regression
- Evaluation against a global mean baseline
- PCA diagnostics showing dominance of energy and spectral magnitude over emotion

### 2. Dynamic Modelling

- Per-second audio feature extraction
- Alignment with per-second emotion labels
- Sequence modelling using **Bidirectional LSTMs**
- Hundreds of supervised samples per track instead of one

This shift enables the model to learn **emotional transitions, trends, and short-term dynamics**.

---

## Repository Contents

- `ENote_DEAM_Analysis.ipynb`  
  Exploratory analysis, static modelling, and visual diagnostics.

- `ENote_DEAM_Analysis_clean.ipynb`  
  Cleaned notebook used for final results and poster figures.

- `Final_ENote_Project.ipynb`  
  End-to-end dynamic modelling pipeline and qualitative prediction demos.

- `dynamic_sequence_model.keras`  
  Trained BiLSTM model for per-second valence and arousal prediction.

---

## Results Summary

- Static models converge to the **global mean** and fail to capture emotional variability.
- PCA confirms that static features are dominated by non-emotional variance.
- Dynamic models successfully track **emotional trajectories over time**.
- Significantly lower error compared to static baselines.

This project was developed for an academic poster presentation.  
All experiments focus on **audio-only emotion inference**, without using lyrics during training.

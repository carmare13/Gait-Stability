# Gait Stability Assessment Using Nonlinear Analysis and Artificial Intelligence

This repository contains the code, data structure, and documentation for a research project aimed at quantifying gait stability using nonlinear dynamics, dimensionality reduction with deep learning, and unsupervised analysis of full-body IMU gait data collected on a 200-meter curved indoor track.

The project extends the ideas presented in recent nonlinear gait analysis literature—particularly the NONAN GaitPrint dataset —by integrating advanced machine learning models to extract stable, subject-invariant and pathology-sensitive gait representations.

## Project Overview

Gait stability is essential for predicting fall risk and identifying early motor decline. Traditional gait metrics (step length, cadence, stride time, etc.) capture only linear aspects of movement. However, human gait is inherently nonlinear, exhibiting:

Long-range correlations (fractal dynamics)

Dynamical stability (Lyapunov exponents)

Recurrence structures

Attractor-like behavior in joint/segment trajectories

This project provides a modern pipeline combining:

🔹 Nonlinear Analysis

Largest Lyapunov Exponent (λ₁)

Hurst exponent (HfGn)

Recurrence Quantification Analysis (RQA)

Sample entropy & multiscale entropy

Variability structure of spatiotemporal parameters

🔹 Deep Learning & Representation Learning

LSTM-Autoencoder

BiLSTM-Autoencoder

ConvLSTM-Autoencoder

Semi-supervised AE with multiple losses:

reconstruction

supervised contrastive loss

group classification loss (optional)

consistency regularization

EMA-teacher

🔹 Unsupervised Learning

HDBSCAN clustering in latent space

UMAP for manifold visualization

Cluster purity, ARI, NMI

🔹 Biomechanical Data

Full-body IMU kinematics

30–60 subjects, three age groups:

G01 – Young adults

G02 – Middle-aged adults

G03 – Older adults

18 four-minute continuous trials per subject

Curved track walking with real-world variability

321 variables per timestamp (acc, vel, pos, orientation, joint angles)

## Objectives

Dataset Transformation
Segment gait cycles, normalize temporally, and extract spatiotemporal and nonlinear stability indicators.

Dimensionality Reduction
Train deep autoencoders to capture latent structure of gait stability across subjects and groups.

Unsupervised Group Discovery
Use clustering algorithms to determine whether latent gait structure separates:

age groups

individual gaitprints

stability/impaired patterns

Explainability (XAI)
Identify which kinematic variables out of the 321 input channels contribute most to:

cluster membership

stability indicators

group separation

Evaluation and Generalization
Measure reconstruction errors, latent cluster quality, and test–retest reliability.

## Project Structure
├── data/
│ ├── S###/
│ ├── Spatiotemporal/
│ └── Zarr/
│
├── notebooks/
│ ├── preprocessing.ipynb
│ ├── nonlinear_analysis.ipynb
│ ├── AE_LSTM.ipynb
│ ├── AE_BiLSTM.ipynb
│ ├── AE_ConvLSTM.ipynb
│ └── clustering_umap_hdbscan.ipynb
│
├── src/
│ ├── preprocessing/
│ ├── nonlinear/
│ ├── models/
│ ├── evaluation/
│ └── xai/
│
├── docs/
└── README.md

Key Features
✔ Curved Track Walking

Unlike most public datasets, this project includes walking on a 200-meter indoor curved track, capturing the real-life variability missing from straight-line treadmill data—consistent with concerns raised in NONAN GaitPrint .

✔ Long Continuous Trials

Each trial contains ~48,000 samples, enabling robust nonlinear analysis such as Lyapunov exponents and fractal dynamics.

✔ Test–Retest Reliability

Repeated trials across two days allow measurement of:

intra-individual stability

inter-individual distinctiveness (gaitprint)

✔ Latent Stability Biomarkers

Autoencoders uncover multidimensional gait stability signatures beyond classical linear metrics.

✔ Explainable AI for Biomechanics

SHAP, permutation tests, and gradients identify the most influential kinematic variables for cluster separation.

Requirements

Python 3.10+

## Recommended libraries:

NumPy, Pandas, SciPy

scikit-learn

PyTorch

UMAP, HDBSCAN

Matplotlib, Seaborn

PyWavelets (entropy & fractal metrics)

Optional tools:

JupyterLab

CUDA GPU for AE training

Gazebo/SolidWorks (for simulation modules)



## Contact

For questions or collaboration opportunities, please reach out via dianacmartinez13@gmail.com


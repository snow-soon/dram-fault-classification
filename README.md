# DRAM Fault Pattern Classification & Anomaly Detection

Multi-class fault type identification and unsupervised anomaly detection for DRAM quality control using machine learning.

---

## Overview

This project addresses the challenge of identifying **specific fault types** in DRAM chips — going beyond simple pass/fail detection to classify the root cause of each defect. Two complementary approaches are implemented and compared.

| Track | Approach | Key Question |
|---|---|---|
| **A** | Multiclass Classification | *Which fault type is this?* |
| **B** | Anomaly Detection | *Is this chip abnormal?* |

---

## Fault Types

| Class | Fault Type | Primary Signal | Ratio |
|---|---|---|---|
| 0 | Normal | All features nominal | 93.0% |
| 1 | Cell Failure | `error_rate` spike | 3.0% |
| 2 | Retention Failure | `temperature` ↑, `refresh_rate` ↓ | 1.5% |
| 3 | Bridge Defect | `voltage` bimodal, `bandwidth` ↓ | 1.5% |
| 4 | Open Circuit | `latency` ↑↑, `power` ↑, `bandwidth` ↓↓ | 1.0% |

---

## Dataset Design

- **100,000 samples** with realistic 93:7 class imbalance
- **Borderline cases**: fault distributions intentionally overlap with Normal range (~30% overlap)
- **Measurement noise**: Gaussian noise added to all features to simulate sensor error
- **Bimodal voltage** in Bridge Defect: mimics intermittent early-stage defect manifestation

These design choices prevent trivially separable classes (which caused 100% accuracy in the v1 binary classifier) and create a more realistic classification challenge.

---

## Results

### Track A — Multiclass Classification

| Model | Accuracy | ROC-AUC | Bridge Defect Recall |
|---|---|---|---|
| **Random Forest** | **98.7%** | **0.998** | **74.3%** |
| Gradient Boosting | 98.4% | 0.989 | 67.0% |

### Track B — Anomaly Detection

| Model | ROC-AUC | Fault Recall | False Positive Rate |
|---|---|---|---|
| **Isolation Forest** | **0.9995** | **99.9%** | 7.7% |
| MLP Autoencoder | 0.9962 | 98.4% | 5.3% |

---

## Key Insights

**1. Bridge Defect is structurally the hardest class (74.3% recall)**
The bimodal voltage distribution means early-stage bridge defects produce near-normal voltage readings. This is not a model limitation — it reflects the genuine difficulty of detecting intermittent defects from a single test snapshot. Temporal features or repeated measurements would be needed to improve detection.

**2. Top predictive features**
`error_rate_pct` is the #1 predictor overall. `voltage` and `refresh_rate` are critical for distinguishing Bridge Defect and Retention Failure specifically.

**3. Recommended two-stage pipeline**
Use anomaly detection (Isolation Forest) as a fast first-pass gate, then apply the Random Forest classifier only on flagged chips for fault-type identification. This avoids running the heavier classification model on all chips.

---

## Project Structure

```
├── 01_data_generation.py    # Dataset generation with realistic fault distributions
├── 02_classification.py     # Track A: Random Forest vs Gradient Boosting
├── 03_anomaly_detection.py  # Track B: Isolation Forest vs MLP Autoencoder
├── data/
│   └── dram_fault_v2.csv    # Generated dataset (100K samples, 5 classes)
├── reports/
│   ├── figures/             # All output plots
│   ├── track_a_results.json # Classification metrics
│   └── track_b_results.json # Anomaly detection metrics
└── requirements.txt
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Generate dataset
python 01_data_generation.py

# Step 2: Train and evaluate classifiers
python 02_classification.py

# Step 3: Train and evaluate anomaly detection models
python 03_anomaly_detection.py
```

---

## Tech Stack

- **Python 3.10+**
- scikit-learn (RandomForest, GradientBoosting, IsolationForest, MLPRegressor)
- pandas, numpy, matplotlib, seaborn

---

## Author

Soonho Chung · soonho.chung8120@gmail.com · [github.com/snow-soon](https://github.com/snow-soon)

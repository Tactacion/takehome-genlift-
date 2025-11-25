# Customer Churn Prediction

## Overview

This project transforms raw customer event streams (logins, page views, support tickets, etc.) into fixed-dimensional features and trains two models to predict 30-day churn:

1. **Baseline**: Random Forest with balanced class weights
2. **PyTorch**: MLP with custom training loop, BatchNorm, Dropout, and weighted BCE loss

## Project Structure

```
takehome/
├── events.csv                  # Raw event stream data
├── data/
│   └── customers.parquet       # Generated customer features
├── src/
│   ├── config.py              # Centralized configuration
│   ├── preprocessing.py       # Feature engineering with temporal windowing
│   ├── models.py              # Random Forest and PyTorch MLP adapters
│   ├── prepare_data.py        # Data preparation CLI
│   └── train.py               # Model training CLI
├── requirements.txt
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone or extract the repository
cd takehome

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare Customer Features

Transform raw event streams into customer-level feature matrix:

```bash
python src/prepare_data.py --events-path events.csv
```

**Output:** `data/customers.parquet` with customer_id, label_churn_30d, and feature columns (f_*)

**What it does:**
- Applies strict temporal windowing to prevent data leakage
- Cutoff date = (max_date - 30 days)
- Features computed from last 28 days before cutoff
- Labels derived from 30 days after cutoff

**Features generated:**
- **Intensity**: Total events, unique event types, event value
- **Velocity**: (Activity L7 × 4) / Activity L28 (behavioral decay detection)
- **Recency**: Days since last event
- **Friction**: Support tickets / logins ratio, error rate

### Step 2: Train Baseline Model

Train Random Forest with balanced class weights:

```bash
python src/train.py --data-path data/customers.parquet --model-type baseline
```

**Output:**
- Performance metrics (PR-AUC, ROC-AUC, Recall, F1)
- Top 10 feature importance rankings

### Step 3: Train PyTorch Model

Train MLP with custom training loop:

```bash
python src/train.py --data-path data/customers.parquet --model-type torch
```

**What it does:**
- 2-hidden-layer MLP (64 → 32 → 1)
- BatchNorm + Dropout (0.3)
- Weighted BCE loss with dynamic pos_weight
- 80/20 train/val split with early stopping
- Adam optimizer (lr=0.001)

## Key Design Decisions

### 1. Temporal Windowing
**Decision:** Strict cutoff at (max_date - 30 days)
**Rationale:** Prevents data leakage by ensuring no future information used for features. Simulates real deployment where we predict 30 days ahead.

### 2. Velocity-Based Features
**Decision:** (L7 × 4) / L28 ratio instead of absolute counts
**Rationale:** Churn is a behavioral decay process. A power user dropping to 20% activity is higher risk than a light user staying consistent.

### 3. Class Imbalance Handling
**Decision:** RF uses `class_weight='balanced'`, MLP uses `pos_weight=num_neg/num_pos`
**Rationale:** With ~22% churn rate, naive classifiers predict "no churn" for everyone. Weighted loss forces models to learn minority class.

### 4. Evaluation Metrics
**Decision:** PR-AUC and Recall as primary metrics
**Rationale:** Accuracy is misleading with imbalanced classes. Missing a churner costs $10k-$500k revenue, false alarms cost a check-in call. Optimize for catching churners.

### 5. Random Forest over Logistic Regression
**Decision:** RF baseline instead of logistic regression
**Rationale:** Tree ensembles handle feature interactions and non-linearity naturally without manual feature engineering. Typically dominate on tabular data.

## Model Comparison

Expected results (approximate, varies by random seed):

| Metric | Random Forest | PyTorch MLP |
|--------|--------------|-------------|
| PR-AUC | 0.63-0.65 | 0.66-0.68 |
| Recall@0.5 | 0.70-0.72 | 0.75-0.81 |
| F1@0.5 | 0.60-0.63 | 0.58-0.62 |

**Interpretation:**
- MLP catches more churners (higher recall) but with more false alarms (lower F1)
- Both models perform significantly better than random (PR-AUC > 0.5)
- Top predictive features: days_since_last_event, velocity_ratio, friction_index

## Next Steps for Production

If deploying to production, prioritize:

1. **Streaming feature pipeline**: Real-time feature computation instead of batch
2. **LSTM for raw sequences**: Replace aggregated features with LSTM to process event sequences directly
3. **Model monitoring**: Track feature drift and prediction distribution shift
4. **A/B test retention impact**: Validate that interventions based on predictions actually reduce churn
5. **SHAP for interpretability**: Generate customer-specific explanations for high-risk predictions

## Methodology Reference

See `methodology.pdf` for detailed mathematical formulations and architectural decisions.

## License

Proprietary - Interview Take-Home Assignment

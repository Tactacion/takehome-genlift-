# churn prediction with time-aware transformers

predicting customer churn from event streams using a transformer encoder with sinusoidal time embeddings.

## results

```
test auroc: 0.9960
test auprc: 0.9953
training time: 4 minutes (15 epochs, early stopped at 12)
parameters: 429k
```

## why this works

most churn prediction: aggregate features (logins/week, days since last event) then xgboost.

this approach: feed raw event sequences into a transformer that actually understands time.

key ideas:
1. **sinusoidal time encoding**: log(time) + fourier features so the model can tell 5 minutes from 5 days
2. **focal loss**: handles 30% churn rate without manual reweighting
3. **mean pooling**: robust aggregation of variable-length sequences

see [APPROACH.md](APPROACH.md) for the full story of what worked and what didn't.

## quick start

```bash
# install
pip install torch pandas numpy scikit-learn tqdm

# train
PYTHONPATH=/Users/revanshphull/tkhm python src/bert_train_simple.py \
  --events-path events.csv \
  --output-dir bert_models \
  --max-epochs 15 \
  --patience 5 \
  --device cpu

# for gpu: --device cuda
```

outputs:
- `bert_models/best_model.pt` - trained model
- `bert_models/training_history.json` - metrics per epoch

## architecture

```
event sequence (variable length)
  ↓
[event_embedding + value_projection + fourier_time_encoding]
  ↓
2-layer transformer encoder (pre-norm, 4 heads)
  ↓
mean pool across time
  ↓
linear classifier
  ↓
focal loss (gamma=2.0)
```

the time encoding is the key part:

```python
log_time = log(time_delta + 1)
frequencies = [1/10000^(2k/d) for k in range(16)]
encoding = [sin(f * log_time), cos(f * log_time) for f in frequencies]
```

this lets the model detect patterns at different time scales (hours, days, weeks) simultaneously.

## data format

expects `events.csv` with columns:
- `customer_id`: unique identifier
- `event_timestamp`: iso format timestamp
- `event_type`: categorical (login, page_view, support_ticket, etc)
- `event_value`: continuous value
- `label_churn_30d`: binary (1 = churned)

## inference

```python
import torch
from src.bert_train_simple import SimpleTimeAwareEncoder

# load model
checkpoint = torch.load("bert_models/best_model.pt", weights_only=False)
model = SimpleTimeAwareEncoder(checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# predict
with torch.no_grad():
    logits = model(event_types, event_values, time_deltas, padding_mask)
    churn_prob = torch.softmax(logits, dim=-1)[0, 1].item()
```

## files

**model code:**
- `src/bert_config.py` - configuration dataclass
- `src/bert_layers.py` - sinusoidal time encoding, focal loss, transformer blocks
- `src/bert_model.py` - full model (has pytorch lightning version, not required)
- `src/bert_train_simple.py` - standalone training script (no lightning deps)
- `src/simple_dataset.py` - data loading and preprocessing

**old/experimental (not needed):**
- `src/model.py`, `src/simulation.py`, `src/train.py` - generative world model (failed, kept for reference)
- `src/prepare_data.py` - preprocessing for generative model

**docs:**
- `APPROACH.md` - detailed explanation of design choices
- `README.md` - this file

## what i tried

### attempt 1: generative world model

idea: model p(next_event, value, time | history) and do monte carlo rollouts to simulate futures.

result: mode collapse, auroc=0.50 (random). needed way more data.

### attempt 2: discriminative encoder (current)

idea: learn representations of customer state, classify those.

result: auroc=0.996. much more stable on small data.

lesson: generative approaches are cool but need 10x more data. discriminative learning works.

## key components

### time encoding

why: transformers don't understand time natively. they just see sequences.

problem: "2 logins 5 minutes apart" vs "2 logins 5 days apart" look the same without time info.

solution: log(time) + sinusoidal encoding (borrowed from positional encoding in "attention is all you need")

### focal loss

why: 30% churn rate means model can get 70% accuracy by always predicting "no churn".

standard solution: weight the minority class higher.

better solution: focal loss - automatically down-weights easy examples.

```python
focal_loss = -(1 - p)^gamma * cross_entropy
```

with gamma=2, if the model is confident and correct, loss goes to ~0. if it's uncertain or wrong, full loss applies. forces learning on hard cases.

### pre-norm transformer

post-norm: `x = layernorm(x + attention(x))`
pre-norm: `x = x + attention(layernorm(x))`

pre-norm is more stable when you can't tune learning rates perfectly. gradients flow better.

### mean pooling

alternatives:
- cls token: needs to learn how to aggregate (requires more data)
- max pooling: sensitive to outliers
- mean pooling: simple average, very robust

with 2100 training examples, simpler wins.

## hyperparameters

```python
embedding_dim = 128
num_layers = 2
num_heads = 4
feedforward_dim = 512
max_seq_length = 64
batch_size = 32
learning_rate = 2e-4
focal_gamma = 2.0
patience = 5
```

tried deeper (4-6 layers) - overfit.
tried larger embedding (256) - no improvement, slower.
tried more heads (8) - no improvement.

kept it simple, worked better.

## performance

| metric | value |
|--------|-------|
| test auroc | 0.9960 |
| test auprc | 0.9953 |
| train time | 4 min |
| params | 429k |

for comparison, a logistic regression on aggregated features gets ~0.75 auroc.
xgboost might hit 0.85-0.90.

## requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

optional for lightning version:
```
pytorch-lightning>=2.0.0
torchmetrics>=1.0.0
```

## notes

- sequences truncated to 64 events (most recent kept)
- padding mask ensures padded positions don't affect attention
- early stopping on validation auprc (patience=5)
- cosine annealing lr schedule
- gradient clipping at 1.0

## what could be better

- ensemble 5 models (usually +0.01 auroc)
- use full event history (some users have 100+ events)
- k-fold cross validation for more reliable metrics
- tune classification threshold based on business costs
- attention visualization for interpretability



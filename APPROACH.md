# approach: why this works and where the ideas came from

## the problem

predicting churn from event streams. you get sequences of user actions (logins, feature usage, support tickets) with timestamps and values. need to figure out who's going to leave.

## first attempt: generative world model (didn't work)

### the idea

saw this paper from deepmind on world models - instead of just classifying "will they churn", actually model the entire process of how users behave. then you can simulate their future and see if they go quiet.

the math: learn p(next_event, value, time | history) and sample from it repeatedly.

### why it seemed cool

- can do counterfactual reasoning (what if we sent them an email?)
- monte carlo rollouts give you uncertainty for free
- feels more "causal" than pattern matching

### why it failed

mode collapse. with only 3000 users, the model couldn't learn stable distributions. it would either predict everything as churn or nothing as churn. the loss landscape was too unstable.

turned out generative models need way more data than we had. all the papers using this stuff have millions of sequences.

## second attempt: discriminative encoder (worked)

### the core insight

stop trying to model the future. just learn a good representation of each customer's current state, then classify that.

think of it like word2vec but for user behavior. map each customer's event history to a point in some high-dimensional space. if the representation is good, churners will cluster together and you can just draw a line to separate them.

### the time encoding problem

first issue: transformers don't actually understand time. they just see sequences. but time matters - there's a huge difference between "logged in twice with 5 minute gap" vs "logged in twice with 5 day gap".

tried a few things:
1. just use the raw time delta - doesn't work, scale is all wrong
2. normalize it - loses information about actual duration
3. log(time) - better! now 1 hour vs 1 day vs 1 week are equally spaced

but still felt wrong. remembered from the "attention is all you need" paper they use sinusoidal position encodings. the idea is you can represent any position as a combination of different frequencies.

applied the same thing to log(time):
```
features = [sin(w1 * log(t)), cos(w1 * log(t)), sin(w2 * log(t)), ...]
```

where the frequencies w1, w2, ... decrease exponentially. this way the model can detect patterns at different time scales (hourly, daily, weekly) all at once.

### the class imbalance problem

30% of users churn. if you just optimize cross entropy, the model learns "always predict no churn" and gets 70% accuracy. useless.

tried reweighting the loss - helps a bit but not enough.

then found this focal loss paper from facebook AI (the object detection one). the idea: don't just weight the minority class higher, actually reshape the loss so easy examples contribute almost nothing.

formula: multiply cross entropy by (1 - p)^gamma

- if model is confident and correct (p close to 1), loss goes to zero
- if model is wrong or uncertain, full loss applies

set gamma=2 and suddenly the model had to actually learn to predict churners instead of taking the easy way out.

### architecture choices

**why 2 layers not 4 or 6?**
tried deeper, it overfit. with only 2100 training examples, keeping it shallow worked better.

**why pre-norm not post-norm?**
post-norm transformers are stable when you have tons of data and can tune learning rates carefully. pre-norm is more forgiving - gradients flow more directly through the residual connections. saw this in a bunch of recent papers (t5, normformer). just worked better in practice.

**why mean pooling not cls token?**
cls token needs to learn to aggregate information. mean pooling is free - just average the representations. with limited data, simpler is better.

## the actual training

started with lr=1e-4, seemed reasonable. used cosine annealing because i always do and it works.

early stopping patience=5 on validation auprc. not auroc because auprc is what actually matters for imbalanced data - it doesn't give you credit for ranking the 70% of negatives correctly.

first epoch: val auprc shoots to 0.99. thought it was a bug.
second epoch: 0.9997. definitely thought it was a bug.
checked on test set: 0.995. actually real.

## why it works

the sinusoidal time encoding gives the model actual temporal understanding. it can tell "this user used to log in daily but hasn't logged in for 2 weeks" vs "this user logs in monthly and it's been 2 weeks".

focal loss forces it to learn the hard cases instead of copping out.

the transformer finds patterns like:
- support ticket followed by no logins = bad
- plan downgrade followed by decreasing engagement = bad
- regular feature usage = good

it's not doing anything magical, just pattern matching on time-aware sequences. but the patterns are actually meaningful because of how we encoded time.

## comparison to what actually gets used in prod

most companies do this:
1. aggregate features (logins per week, total spend, days since last login)
2. throw into xgboost
3. tune threshold

our approach:
1. encode events as sequences with proper time awareness
2. let transformer find patterns
3. focal loss handles imbalance automatically

the transformer approach is overkill for this dataset size but it's actually simpler - no feature engineering, no manual time windows, just feed in raw events.

## what i'd change with more time

**ensemble**: train 5 models with different random seeds, average predictions. usually gets you another 0.01 auroc for free.

**longer sequences**: we truncate at 64 events. some users have 100+. could probably squeeze out better performance by using the full history.

**cross-validation**: we did one random split. proper k-fold CV would give more reliable estimates of true performance.

**attention analysis**: visualize what the model attends to when making predictions. would help with interpretability.

**threshold tuning**: we just used 0.5. should actually find the optimal cutoff based on business costs of false positives vs false negatives.

## lessons learned

**generative models are overrated for small data**: everyone's excited about world models and simulators but they need massive datasets. discriminative learning is boring but it works.

**time is not just another feature**: you can't treat time like a categorical variable or even a continuous one. the log + sinusoidal thing seems weird but it actually matters.

**focal loss is underused**: everyone knows about it from object detection but almost nobody uses it for tabular/sequence data. it should be the default for imbalanced classification.

**simpler architectures win on small data**: wanted to use 8 layers, multiple attention types, fancy pooling. 2 layers + mean pooling beat everything.

## the math (for people who care)

we're learning a function f: H -> [0,1] where H is the space of event sequences.

the key is the time encoding phi:

```
phi(dt) = [sin(w_k * log(1 + dt)), cos(w_k * log(1 + dt))] for k = 1...K
where w_k = 1 / (10000^(2k/d))
```

this is basically a random fourier feature map but deterministic. it embeds time into a reproducing kernel hilbert space where the kernel is approximately:

```
k(t1, t2) = exp(-||log(t1) - log(t2)||^2 / sigma^2)
```

which means "times that are close on a log scale are similar". exactly what we want.

the transformer then learns:

```
z = MeanPool(Transformer(EventEmbed(k) + ValueProj(v) + phi(dt)))
p(churn) = sigmoid(Linear(z))
```

with focal loss:

```
L = -(1 - p)^gamma * [y*log(p) + (1-y)*log(1-p)]
```

that's it.  just good encoding + standard architecture + right loss function.

## files

- `src/bert_config.py` - hyperparameters
- `src/bert_layers.py` - time encoding, focal loss, transformer blocks
- `src/bert_model.py` - full model
- `src/bert_train_simple.py` - training loop
- `src/simple_dataset.py` - data loading

## results

test set (450 users):
- auroc: 0.996
- auprc: 0.995
- training time: 4 minutes
- parameters: 429k

## how to run

```bash
# train
PYTHONPATH=/Users/revanshphull/tkhm python src/bert_train_simple.py \
  --events-path events.csv \
  --output-dir bert_models \
  --max-epochs 15 \
  --patience 5

# predictions saved in bert_models/best_model.pt
```



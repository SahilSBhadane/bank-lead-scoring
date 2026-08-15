# Bank Term Deposit Lead Scoring

Predicts whether a bank customer will subscribe to a term deposit as a result
of a telemarketing call — **before** the call happens, using only information
a bank would actually have going into the campaign (demographics, prior
campaign history, and macroeconomic context).

Dataset: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(Moro, Cortez & Rita, 2014) — 41,188 real telemarketing contacts from a
Portuguese bank, 11.3% conversion rate.

## Why this exists

Direct marketing / lead scoring is a standard BFSI data science problem:
which leads should a call center prioritize? This project builds a model
that's honest about what it can and can't see at prediction time — a
distinction that matters a lot in production but gets skipped in a lot of
portfolio projects.

## The leakage issue (and why it's handled explicitly)

The dataset includes `duration` — length of the last call, in seconds.
`duration = 0` is an almost perfect predictor of "no" (the call never really
connected), and successful calls run ~2.5x longer on average than
unsuccessful ones. **`duration` is only known after the call ends** — a
real lead-scoring model has to work with information available *before*
contact, or it isn't actually scoring leads, it's just describing calls that
already happened. UCI's own dataset documentation flags this explicitly.

This project trains and reports **both**, to make the effect visible instead
of hiding it:

| Model | AUC | PR-AUC | Precision | Recall | Notes |
|---|---|---|---|---|---|
| Dummy baseline (stratified random) | 0.505 | 0.114 | 0.12 | 0.12 | Sanity floor — matches base rate |
| **Real model (no `duration`)** | **0.814** | **0.486** | 0.68 | 0.25 | **The actual deliverable** |
| Benchmark only (with `duration`) | 0.955 | 0.699 | 0.70 | 0.55 | Leaky — shown for comparison only, never shipped |

The ~0.14 AUC gap between the real model and the leaky benchmark is the
leakage effect, quantified. Full breakdown (confusion matrices, full
classification reports, tuned hyperparameters, feature importances) is
written to `outputs/metrics_report.txt` on every run.

## What's actually driving predictions

Top features by importance (XGBoost, real/deliverable model):

1. `nr.employed` — macroeconomic employment index at time of contact
2. `poutcome_success` — whether this customer converted on a prior campaign
3. `emp.var.rate` — employment variation rate (macro trend)
4. `month_oct` — seasonal effect
5. `pdays` — days since last contact

Repeat-success customers and macro-economic conditions dominate — which
lines up with how term-deposit demand actually works (rate-sensitive
product, existing-relationship customers convert more easily).

## Pipeline

```
data_loader.py    → loads raw UCI CSV
preprocessing.py  → leak-safe feature split, one-hot + scaling via ColumnTransformer
modeling.py       → StratifiedKFold CV, RandomizedSearchCV hyperparameter tuning,
                     class-imbalance handling (scale_pos_weight), dummy baseline
main.py           → orchestrates: baseline → real model → leaky benchmark →
                     feature importance → metrics_report.txt → model.joblib
```

## Setup & run

```bash
pip install -r requirements.txt
python main.py
```

Outputs land in `outputs/`:
- `metrics_report.txt` — full metrics for all three models + feature importance
- `model.joblib` — the trained, deliverable (leak-free) model

## Honest limitations

- **Recall is the weak point** (0.25 at default threshold) — the model is
  conservative, missing real converters in favor of precision. For a
  call-center use case where contacting a lead is low-cost, the
  probability threshold should likely be lowered to trade precision for
  recall; this is a business decision, not a modeling one, so it's left
  tunable rather than baked in.
- Class imbalance (11.3% positive) limits how much signal any model can
  extract without more features (e.g. digital engagement data, which this
  dataset doesn't include).
- This is a real, published academic dataset, not live production data —
  the pipeline architecture is portfolio-representative of production lead
  scoring, not a claim that this exact model is deployment-ready.

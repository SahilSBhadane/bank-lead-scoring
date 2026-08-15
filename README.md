# 🏦 Bank Term Deposit Lead Scoring

### Predicting which leads will convert — *before* the call happens, using only pre-contact information

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=flat)]()
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)]()

---

## 🎯 The problem

Bank call centers can't call every lead — agent time is limited and every
call has a cost. Lead scoring answers: *which customers on the list are
actually worth calling?* This is a standard BFSI data science problem, and
this project builds a version of it that's honest about a constraint real
production models have to respect: **you can only use information available
before you pick up the phone.**

**Dataset:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(Moro, Cortez & Rita, 2014) — 41,188 real telemarketing contacts from a
Portuguese bank, 11.3% baseline conversion rate.

---

## 🔍 The leakage problem (and how it's handled)

The raw dataset includes `duration` — the length of the sales call itself,
in seconds. It's an almost perfect predictor: `duration = 0` means the call
never really connected, and successful calls run **~2.5x longer** on average
than unsuccessful ones. The catch — **you only know call duration *after*
the call has already happened.** A model trained on it isn't scoring leads,
it's describing calls that are already over. UCI's own documentation flags
this explicitly.

Instead of quietly dropping it, this project trains **both** versions and
reports the gap directly:

| Model | AUC | PR-AUC | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|
| Dummy baseline *(sanity floor)* | 0.505 | 0.114 | 0.12 | 0.12 |
| 🏆 **Real model** — no `duration` | **0.814** | **0.486** | 0.68 | 0.25 |
| ⚠️ Benchmark only — with `duration` *(leaky, not shipped)* | 0.955 | 0.699 | 0.70 | 0.55 |

**The ~0.14 AUC gap is the leakage effect, quantified.** Full metrics
(confusion matrices, classification reports, tuned hyperparameters, feature
importances) are written to `outputs/metrics_report.txt` on every run.

---

## 🧠 What's actually driving predictions

Top 5 features by importance (XGBoost, real/deliverable model):

| Rank | Feature | What it captures |
|:---:|:---|:---|
| 1 | `nr.employed` | Macroeconomic employment index at time of contact |
| 2 | `poutcome_success` | Customer converted on a *prior* campaign |
| 3 | `emp.var.rate` | Employment variation rate (macro trend) |
| 4 | `month_oct` | Seasonal effect |
| 5 | `pdays` | Days since last contact |

Macro-economic conditions and repeat-success customers dominate — which
tracks with how a rate-sensitive product like a term deposit actually
behaves: existing-relationship customers convert more easily, and appetite
shifts with the broader economy.

---

## 🏗️ Pipeline architecture

```
┌────────────────┐
│ data_loader.py │  →  loads raw UCI CSV
└───────┬────────┘
        ↓
┌────────────────────┐
│ preprocessing.py    │  →  leak-safe feature split
│                      │     one-hot + scaling via ColumnTransformer
└───────┬──────────────┘
        ↓
┌────────────────────┐
│ modeling.py          │  →  StratifiedKFold CV
│                       │     RandomizedSearchCV hyperparameter tuning
│                       │     class-imbalance handling (scale_pos_weight)
│                       │     dummy baseline for comparison
└───────┬───────────────┘
        ↓
┌────────────────────┐
│ main.py               │  →  orchestrates: baseline → real model →
│                        │     leaky benchmark → feature importance →
│                        │     metrics_report.txt → model.joblib
└────────────────────────┘
```

---

## 🚀 Setup & run

```bash
git clone https://github.com/SahilSBhadane/bank-lead-scoring.git
cd bank-lead-scoring
pip install -r requirements.txt
python main.py
```

**Note:** the raw CSV isn't bundled in this repo (keeps it lightweight) —
grab `bank-additional-full.csv` from the
[UCI dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing)
and place it in a `data/` folder before running.

Outputs land in `outputs/`:
- `metrics_report.txt` — full metrics for all three models + feature importance
- `model.joblib` — the trained, deliverable (leak-free) model

---

## ⚖️ Honest limitations

- **Recall is the real weak point** (0.25 at default threshold) — the model
  trades recall for precision by default. For a call center where contacting
  a lead is cheap, lowering the probability threshold would catch more real
  converters at the cost of more wasted calls — that's a business call, not
  a modeling one, so it's left tunable rather than baked in.
- **Class imbalance** (11.3% positive) caps how much signal any model can
  extract without richer features (e.g. digital engagement data, which this
  dataset doesn't include).
- This is a published academic dataset, not live production data — the
  pipeline is portfolio-representative of production lead scoring, not a
  claim that this exact model is deployment-ready as-is.

---

## 👨‍💻 Author

**Sahil Bhadane**
- GitHub: [@SahilSBhadane](https://github.com/SahilSBhadane)
- LinkedIn: [linkedin.com/in/04sahil](https://linkedin.com/in/04sahil)

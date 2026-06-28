# I-80 Donner Pass Closure Risk

This portfolio project analyzes official CHP road-closure posts and historical weather observations to estimate the risk that a new I-80 Donner Pass closure will begin within the next 24 hours.

The final product is a four-level risk-ranking model designed for clear communication and future dashboard use. It does not attempt to predict closure duration, because too few events have complete reopening times for a reliable duration model.

## Project overview

The modeling dataset combines:

- Manually reviewed CHP closure and reopening posts
- Hourly historical weather near the Donner Summit corridor
- Current weather conditions and backward-looking 6- and 24-hour storm summaries
- Local seasonal and time-of-day features

The analysis is limited to the snowy season, defined as October through May. Model development uses chronological winter splits rather than random row splitting so that evaluation better represents forecasting a future winter.

## Road closure risk levels

### Plain-English explanation

The model does not simply answer “closure” or “no closure.” Closures are uncommon, and a binary warning system would either miss many closures or issue too many false alarms.

Instead, the model examines current and recent weather conditions and produces a risk score: an estimate of how strongly those conditions resemble historical periods preceding an I-80 closure. That score is translated into one of four levels:

| Risk level | Model score | Interpretation |
|---|---:|---|
| **Low** | Below 5.7% | Conditions historically showed relatively little evidence of a closure beginning within 24 hours. |
| **Medium** | 5.7% to below 9.3% | Closure risk is elevated above typical Low-risk conditions. |
| **High** | 9.3% to below 40.2% | Conditions resemble weather patterns that appeared more often before closures. |
| **Extreme** | 40.2% or higher | Conditions rank among the most closure-prone historical weather periods. |

These categories are intended to communicate relative operational risk. “Extreme” does not guarantee a closure, and “Low” does not mean a closure is impossible.

In simpler terms, **Low is supposed to be common and Extreme is supposed to be rare**. Most winter hours should remain Low because most hours do not immediately precede a closure. Extreme should be reserved for a small number of unusually concerning hours. The boundary search therefore looked for a useful balance: concentrate as many historical closure-warning hours as possible in High and Extreme without assigning those labels to too many ordinary hours. A broader Extreme category would capture more closures, but it would also create many more false alarms and make “Extreme” less meaningful.

On the untouched 2024–2025 test winter, the observed rate of hours falling within 24 hours of a closure increased consistently across the categories:

| Risk level | Observed test rate | Lift versus the test baseline |
|---|---:|---:|
| Low | 1.9% | 0.34× |
| Medium | 7.3% | 1.30× |
| High | 16.4% | 2.92× |
| Extreme | 55.9% | 9.92× |

This monotonic increase is the central evidence that the categories provide a useful ranking of road-closure risk.

### How many test closures were captured?

The 2024–2025 test winter contained **17 distinct closure starts**. Fourteen had at least one model score during the preceding 24 hours; three occurred after data gaps where the model could not construct its required continuous weather history.

Two related metrics are useful:

- **Alert precision:** of all hours assigned a risk label, how many were actually within 24 hours of a closure?
- **Event coverage:** of all distinct closures with usable prediction history, how many received at least one warning at that level?

| Warning level | Positive hours / labeled hours | Alert precision | Distinct closures warned / scorable closures | Event coverage |
|---|---:|---:|---:|---:|
| High | 136 / 828 | 16.4% | 12 / 14 | 85.7% |
| Extreme | 38 / 68 | 55.9% | 3 / 14 | 21.4% |
| High or Extreme | 174 / 896 | 19.4% | 12 / 14 | 85.7% |

The High event-coverage count includes closures that first received a High warning and later escalated to Extreme. Risk levels are mutually exclusive for any single hour, but one closure can receive different levels as conditions evolve across its 24-hour warning window.

- **Extreme appeared before 3 of the 14 scorable closures** (21.4%), or 3 of all 17 test closures (17.6%).
- **High or Extreme appeared before 12 of the 14 scorable closures** (85.7%).
- Extreme was intentionally rare: only 68 test hours received that label.
- Of those 68 Extreme hours, 38 were genuinely within 24 hours of a closure, producing the 55.9% observed Extreme rate.
- Those 38 hours represented 13.1% of all 291 positive hourly warning rows in the scored test data.

These figures answer different questions. The event-level result measures whether a closure received at least one advance warning. The hourly result measures how accurately every scored hour was categorized. Extreme is not expected to capture every closure by itself; it is reserved for the smallest and most concentrated set of warnings, while High provides broader coverage.

### Technical explanation

The selected model is a constrained `RandomForestClassifier`. It outputs a continuous estimated probability that a new closure will begin within the following 24 hours. A deterministic thresholding function maps that probability to an ordered category:

```text
probability < 0.0570                → Low
0.0570 ≤ probability < 0.0931       → Medium
0.0931 ≤ probability < 0.4024       → High
probability ≥ 0.4024                → Extreme
```

The probability boundaries were not selected manually after viewing the test set. They were treated as hyperparameters and tuned using the 2023–2024 validation winter.

Candidate boundaries were defined as percentiles of the training risk-score distribution:

- Low/Medium: 50th, 60th, or 70th percentile
- Medium/High: 80th, 85th, or 90th percentile
- High/Extreme: 95th, 97th, or 98th percentile

For every candidate combination, the training percentiles were converted into probability cutoffs and applied to validation predictions. The selection objective used average precision on the resulting four-level ordinal ranking. A penalty was applied when:

- observed validation closure rates did not increase monotonically from Low to Extreme; or
- a category contained less than 0.5% of validation hours.

The winning boundaries were the 70th, 85th, and 98th percentiles of training scores. For the fitted model, these corresponded to probabilities of approximately 5.7%, 9.3%, and 40.2%. The thresholds were frozen before the final 2024–2025 test evaluation.

Conceptually, this is a concentration trade-off. The validation search favored boundaries that moved closure-warning hours toward the upper categories while keeping those categories selective. Average precision measured how effectively the four ordered levels concentrated positive outcomes near the top. The monotonicity and minimum-size rules prevented misleading category structures. Thus, the selected Extreme boundary was not simply the cutoff containing the largest number of closures; it was the cutoff that helped produce the strongest overall four-level ranking while leaving Extreme rare enough to represent exceptional conditions.

## Why risk ranking instead of binary prediction?

Road closures are rare. A model that predicts “open” almost all the time can achieve high conventional accuracy while providing little practical value. Conversely, lowering a binary alert threshold can identify more dangerous periods but produce many false alarms.

Risk ranking preserves the model’s continuous information and makes that trade-off visible:

- Higher categories concentrate historically closure-prone conditions.
- Users can decide how much warning sensitivity they need.
- A dashboard can show changing risk without claiming that every elevated period will cause a closure.
- Evaluation can focus on ranking metrics, lift, and concentration of closure-warning hours rather than misleading raw accuracy.

The trade-off remains important. Narrow Extreme bands provide stronger lift and fewer alerts but miss positive hours that fall into lower categories. Broader High or Extreme bands capture more upcoming closures but also generate more false alarms. The saved thresholds represent the best validation-tested compromise for this dataset, not a universal safety standard.

## Model development and evaluation

Three scikit-learn model families were compared:

1. Logistic regression as an interpretable baseline
2. Random forest for nonlinear thresholds and feature interactions
3. Histogram gradient boosting for boosted nonlinear relationships

The chronological split was:

- Training: winters 2016–2017 through 2022–2023
- Validation: winter 2023–2024
- Final test: winter 2024–2025
- Excluded from model selection: incomplete winter 2025–2026

The random forest was selected using validation average precision. On the final test winter it achieved:

- Average precision: **0.253**
- ROC-AUC: **0.807**
- Brier score: **0.0478**
- Target prevalence: **0.056**

The model uses weather and time information available at scoring time. Outcome fields, current closure labels, future targets, raw winter identifiers, and year are excluded from predictors. Rolling weather features require complete consecutive hourly histories to prevent stale observations from bridging ambiguous data gaps.

## Reusable model artifact

The trained risk model is saved as a reusable bundle containing:

- The fitted preprocessing pipeline
- The selected random forest
- Frozen category thresholds
- Required feature names and order
- Training, validation, and test metadata
- Held-out evaluation metrics

Important files:

- `main/notebooks/01_dataset_construction.ipynb` — closure/weather dataset construction
- `main/notebooks/02_eda.ipynb` — exploratory analysis
- `main/notebooks/03_modeling.ipynb` — model comparison, threshold tuning, and final evaluation
- `main/risk_model.py` — reusable loading and scoring helpers
- `main/outputs/modeling/i80_24h_risk_model.joblib` — trained artifact
- `main/outputs/modeling/i80_24h_risk_model_metadata.json` — human-readable model metadata

The current scope covers offline training and batch scoring. A live dashboard, weather-ingestion service, and JavaScript input layer are intentionally left for a later stage.

## Limitations

- The model identifies statistical association, not causation.
- It relies primarily on weather and time features; crashes, traffic volume, road treatment, and operational decisions are not fully represented.
- The score is trained on historical observed weather, so live use must reproduce the same feature definitions from data available at prediction time.
- Risk categories communicate relative model risk and should not replace official CHP or Caltrans guidance.

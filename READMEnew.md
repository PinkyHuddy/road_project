# I-80 Donner Pass: New Closure Risk

This is the companion analysis in `new_closures/`. It answers a narrower question than the main project:

> Given that I-80 is confirmed open now, will a new closure begin during the next 24 hours?

The original `main/` project remains unchanged in purpose: it estimates whether the road will be closed at any point in the next 24 hours, including a closure already in progress. This companion model is an advance-warning model for new closure onsets.

## Plain-English summary

Each prediction starts during an hour when the road is known to be open. The model compares current and recent weather with historical conditions and returns a new-closure risk score. That score is translated into **Low**, **Medium**, **High**, or **Extreme** risk.

Closed hours are not treated as negative examples. Hours with uncertain road status or an incomplete/uncertain future window are also excluded. This matters because calling those hours “no new closure” would give the model false answers.

On the untouched 2024–2025 test winter, Extreme covered only 4.7% of eligible hours. A new closure followed within 24 hours for 30.5% of those Extreme hours, compared with 5.7% of all eligible test hours. Extreme conditions therefore had **5.38 times the baseline risk**.

## Data and target construction

The project uses the same manually labeled CHP-derived closure/reopening records and Donner Summit hourly weather as the main analysis. Only October through May is retained.

For a prediction at hour `t`:

- the road must be confirmed open at `t`;
- the positive target is a labeled closure start at any exact hourly timestamp from `t + 1` through `t + 24`;
- the negative target requires all 24 future timestamps and known road status throughout the window;
- a confirmed future closure start remains positive even if a later hour in the window is uncertain; and
- currently closed, currently ambiguous, incomplete, and otherwise uncertain windows receive no target and are excluded.

After these rules, the modeling dataset contains 47,906 eligible open-hour rows. There are 3,531 positive 24-hour rows, a 7.37% overall positive rate. These are overlapping warning hours, not 3,531 distinct closure events.

## Features

The models use weather that is available at scoring time: snowfall, precipitation, pressure, cloud cover, wind, temperature, snow depth, humidity, and weather code. They also use cyclic month/hour terms and trailing 6- and 24-hour weather summaries. No closure status, closure-start label, future weather, or future target is used as a predictor.

Trailing features require exact hourly continuity. Missing history is not silently bridged.

## Modeling design

The notebook compares logistic regression, random forest, and scikit-learn histogram gradient boosting. Model selection maximizes validation average precision because new closures are rare and ranking the positive cases is more informative than raw accuracy.

The temporal split is:

- Training: winters 2016–2017 through 2022–2023
- Validation: winter 2023–2024
- Final test: winter 2024–2025, evaluated after model and thresholds were frozen
- Excluded: incomplete winter 2025–2026

The selected model is a regularized random forest with constrained depth and leaf size. Its held-out test metrics are:

| Metric | Test result |
|---|---:|
| Baseline new-closure rate | 5.67% |
| Average precision | 0.298 |
| ROC AUC | 0.784 |
| Brier score | 0.048 |

## Risk categories

The selected model produces a score interpreted as estimated 24-hour new-closure probability. Validation-score percentiles were treated as hyperparameters. Candidate boundaries were compared using separation in observed closure rates, concentration of positive hours in higher categories, and penalties for non-monotonic or impractically small categories.

The selected validation percentiles were the 60th, 85th, and 95th percentiles. Freezing those validation-derived score values produced these operational thresholds:

| Category | Model score |
|---|---:|
| Low | below 0.0502 |
| Medium | 0.0502 to below 0.0859 |
| High | 0.0859 to below 0.2618 |
| Extreme | 0.2618 or higher |

Percentiles describe where a score sits relative to validation conditions; they are not closure probabilities. For example, the 95th-percentile boundary was the score exceeded by roughly the riskiest 5% of validation hours. The frozen numeric thresholds—not newly calculated daily percentiles—are used at test and scoring time.

Held-out category performance was:

| Risk | Test hours | New closures within 24h | Observed rate | Share of test hours | Lift |
|---|---:|---:|---:|---:|---:|
| Low | 3,088 | 64 | 2.07% | 60.8% | 0.37× |
| Medium | 1,060 | 48 | 4.53% | 20.9% | 0.80× |
| High | 690 | 103 | 14.93% | 13.6% | 2.63× |
| Extreme | 239 | 73 | 30.54% | 4.7% | 5.38× |

“Closures” in this table means positive hourly forecast rows. One physical closure can create multiple positive lead-time rows, so these counts must not be presented as distinct events captured.

## Reproducible project structure

Run the notebooks in order:

1. `new_closures/notebooks/01_dataset_construction.ipynb`
2. `new_closures/notebooks/02_eda.ipynb`
3. `new_closures/notebooks/03_modeling.ipynb`

The reusable artifact is saved at `new_closures/outputs/modeling/i80_new_closure_24h_risk_model.joblib`. It contains the preprocessing pipeline, fitted model, frozen category thresholds, feature order, split metadata, and test metrics. `new_closures/risk_model.py` provides the reusable scoring interface.

## Limitations

This is a historical risk-ranking model, not a causal weather model or an operational guarantee. CHP post timestamps are used as closure-onset timestamps, so posting delay is a possible source of label timing error. Results also depend on the project’s hand-labeled definition of a closure event. Performance comes from one held-out winter and should be monitored across future winters before operational use.

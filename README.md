# I-80 Donner Pass Closure Risk

This portfolio project analyzes official CHP road-closure posts and historical weather observations to estimate the risk that I-80 Donner Pass will be closed at any point within the next 24 hours.

The final product is a four-level risk-ranking model designed for clear communication and future dashboard use. It does not attempt to predict closure duration, because too few events have complete reopening times for a reliable duration model.

## Installation

Create and activate a Python virtual environment, then install every notebook and project dependency from the root requirements file:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The requirements cover the main construction, EDA, and modeling notebooks as well as the archived weather, Twitter, Supabase, Facebook, and PeMS exploration code. Standard-library and local project imports do not require separate packages.

## Project overview

The modeling dataset combines:

- Manually reviewed CHP closure and reopening posts
- Hourly historical weather near the Donner Summit corridor
- Current weather conditions and backward-looking 6- and 24-hour storm summaries
- Local seasonal and time-of-day features

The analysis is limited to the snowy season, defined as October through May. Model development uses chronological winter splits rather than random row splitting so that evaluation better represents forecasting a future winter.

The target is road status over a future window—not merely a new closure start. For each scored hour, the label is positive when I-80 is confirmed closed during at least one of the following 24 hours. This includes a road that is already closed and remains closed. A row is negative only when every future hour is observed and open; uncertain future windows are excluded unless they contain a confirmed closure.

## Road closure risk levels

### Plain-English explanation

The model does not simply answer “closure” or “no closure.” Closures are uncommon, and a binary warning system would either miss many closures or issue too many false alarms.

Instead, the model examines current and recent weather conditions and produces a risk score: an estimate of how strongly those conditions resemble historical periods preceding an I-80 closure. That score is translated into one of four levels:

| Risk level | Model score | Interpretation |
|---|---:|---|
| **Low** | Below 6.5% | Conditions historically showed relatively little evidence of the road being closed within 24 hours. |
| **Medium** | 6.5% to below 15.1% | Closure risk is elevated above typical Low-risk conditions. |
| **High** | 15.1% to below 50.5% | Conditions resemble weather patterns that appeared more often before or during closures. |
| **Extreme** | 50.5% or higher | Conditions rank among the most closure-prone historical weather periods. |

These categories are intended to communicate relative operational risk. “Extreme” does not guarantee a closure, and “Low” does not mean a closure is impossible.

In simpler terms, **Low is supposed to be common and Extreme is supposed to be rare**. Most winter hours should remain Low because most hours do not immediately precede a closure. Extreme should be reserved for a small number of unusually concerning hours. The boundary search therefore looked for a useful balance: concentrate as many historical closure-warning hours as possible in High and Extreme without assigning those labels to too many ordinary hours. A broader Extreme category would capture more closures, but it would also create many more false alarms and make “Extreme” less meaningful.

On the untouched 2024–2025 test winter, the observed rate of hours falling within 24 hours of a closure increased consistently across the categories:

| Risk level | Observed test rate | Lift versus the test baseline |
|---|---:|---:|
| Low | 2.8% | 0.49× |
| Medium | 9.0% | 1.55× |
| High | 13.9% | 2.39× |
| Extreme | 43.6% | 7.50× |

This monotonic increase is the central evidence that the categories provide a useful ranking of road-closure risk.

> **Key 24-hour result:** An average scored hour in the test winter had a 5.8% chance of the road being closed at some point within the next 24 hours. For hours ranked Extreme, that observed rate increased to 43.6%. In other words, an Extreme hour was approximately **7.5 times more likely** than the average test hour to precede or overlap a closure within that horizon.

This is the model's Extreme **lift**: `43.6% ÷ 5.8% ≈ 7.5×`. Lift compares a category's observed positive rate with the overall positive rate. It does not mean every Extreme period coincided with a future closure; 43.6% of Extreme hours were positive and 56.4% were false alarms.

### How many test closures were captured?

The 2024–2025 test winter contained **17 distinct closure starts**. Fourteen had at least one model score during the preceding 24 hours; three occurred after data gaps where the model could not construct its required continuous weather history.

Two related metrics are useful:

- **Alert precision:** of all hours assigned a risk label, how many were actually within 24 hours of a closure?
- **Event coverage:** of all distinct closures with usable prediction history, how many received at least one warning at that level?

| Warning level | Positive hours / labeled hours | Alert precision | Distinct closures warned / scorable closures | Event coverage |
|---|---:|---:|---:|---:|
| High | 63 / 454 | 13.9% | 9 / 14 | 64.3% |
| Extreme | 44 / 101 | 43.6% | 5 / 14 | 35.7% |
| High or Extreme | 107 / 555 | 19.3% | 11 / 14 | 78.6% |

The High event-coverage count includes closures that first received a High warning and later escalated to Extreme. Risk levels are mutually exclusive for any single hour, but one closure can receive different levels as conditions evolve across its 24-hour warning window.

- **Extreme appeared before 5 of the 14 scorable closures** (35.7%), or 5 of all 17 test closures (29.4%).
- **High or Extreme appeared before 11 of the 14 scorable closures** (78.6%).
- Extreme was intentionally rare: only 101 test hours received that label.
- Of those 101 Extreme hours, 44 were positive for road closure within 24 hours, producing the 43.6% observed Extreme rate.
- Those 44 hours represented 14.8% of all 297 positive hourly rows in the scored test data.

These figures answer different questions. The event-level result measures whether a closure received at least one advance warning. The hourly result measures how accurately every scored hour was categorized. Extreme is not expected to capture every closure by itself; it is reserved for the smallest and most concentrated set of warnings, while High provides broader coverage.

### Technical explanation

The selected model is a constrained `RandomForestClassifier`. It outputs a continuous estimated probability that the road will be closed at any point within the following 24 hours. A deterministic thresholding function maps that probability to an ordered category:

```text
probability < 0.0645                → Low
0.0645 ≤ probability < 0.1511       → Medium
0.1511 ≤ probability < 0.5051       → High
probability ≥ 0.5051                → Extreme
```

The probability boundaries were not selected manually after viewing the test set. They were treated as hyperparameters and tuned using the 2023–2024 validation winter.

Candidate boundaries were defined as percentiles of the training risk-score distribution:

- Low/Medium: 50th, 60th, or 70th percentile
- Medium/High: 80th, 85th, or 90th percentile
- High/Extreme: 95th, 97th, or 98th percentile

For every candidate combination, the training percentiles were converted into probability cutoffs and applied to validation predictions. The selection objective used average precision on the resulting four-level ordinal ranking. A penalty was applied when:

- observed validation closure rates did not increase monotonically from Low to Extreme; or
- a category contained less than 0.5% of validation hours.

The winning boundaries were the 70th, 90th, and 98th percentiles of training scores. For the fitted model, these corresponded to probabilities of approximately 6.5%, 15.1%, and 50.5%. The thresholds were frozen before the final 2024–2025 test evaluation.

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

- Average precision: **0.259**
- ROC-AUC: **0.769**
- Brier score: **0.0530**
- Target prevalence: **0.058**

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
- `main/score_current_risk.py` — validated command-line scorer for live weather inputs
- `main/outputs/modeling/i80_24h_risk_model.joblib` — trained artifact
- `main/outputs/modeling/i80_24h_risk_model_metadata.json` — human-readable model metadata

### Score current weather

The command-line scorer accepts a CSV containing at least 24 consecutive hourly observations in the same units as the training weather data. It reconstructs the trailing features, loads the saved preprocessing/model pipeline, and returns a website-ready JSON result.

```bash
python main/score_current_risk.py hourly_weather.csv --road-status open
```

To write the result for a website or scheduled job:

```bash
python main/score_current_risk.py hourly_weather.csv \
  --road-status open \
  --output main/outputs/current_risk.json
```

The required raw columns are `date`, `snowfall`, `precipitation`, `surface_pressure`, `cloud_cover_low`, `cloud_cover_mid`, `wind_speed_100m`, `temperature_2m`, `snow_depth`, `relative_humidity_2m`, and `weather_code`. Timestamps must be unique and exactly one hour apart. The script refuses to score incomplete histories rather than silently filling or bridging missing hours.

The road-status argument is an operational display override, not a model feature. A confirmed closure displays `Closed Now`; an unknown official status displays `Road Status Unavailable`; otherwise the output displays the model's Low–Extreme risk level. Official-road-status extraction, scheduling, and website delivery remain later pipeline stages.

### Fetch live weather inputs

`main/get_inputs.py` uses the same Open-Meteo provider, Donner Summit coordinates, UTC timestamps, imperial units, retry behavior, and hourly weather fields as the historical weather extractor. It requests recent hourly conditions and the next 24 forecast hours from Open-Meteo's live Forecast API:

```bash
python main/get_inputs.py
```

It writes three auditable files under `main/outputs/live_inputs/`:

- `model_weather_history.csv` — the 24 consecutive recent/current rows passed to the scorer
- `weather_forecast_24h.csv` — the current and following 23 forecast hours for website display
- `weather_request_metadata.json` — source, coordinates, elevation, units, variables, and fetch time

The forecast file is deliberately separate from model history. The saved model was trained on current and trailing weather, so future forecast values must not be substituted into its rolling features. Also, live Open-Meteo inputs may differ from the historical archive/reanalysis used for training; this is a deployment distribution shift that should be monitored and eventually validated with saved live forecasts and realized outcomes.

## AWS Lambda pipeline

The cloud scoring path is separated by responsibility:

| File | Responsibility |
|---|---|
| `main/lambda/lambda_handler.py` | Orchestrates one pipeline run |
| `main/lambda/weather_api.py` | Retrieves Open-Meteo history and forecast weather |
| `main/lambda/closure_api.py` | Retrieves current road status from a configured JSON endpoint |
| `main/lambda/features.py` | Validates hourly continuity and reproduces training features |
| `main/lambda/scoring.py` | Loads the artifact, predicts probability, and assigns risk |
| `main/lambda/storage.py` | Writes output to S3 or `/tmp` |

Both common Lambda layouts are supported:

- Package layout: include the repository's `main/` directory and set the handler to `main.lambda.lambda_handler.lambda_handler`.
- Flat layout: place the contents of `main/lambda/` at the zip root and set the handler to `lambda_handler.lambda_handler`.

The deployment must also include `i80_24h_risk_model.joblib`. Set `MODEL_PATH` to its deployed location, or place it beside `scoring.py`. Dependencies are listed in `main/lambda/requirements-lambda.txt` and must be built for the Lambda runtime's Linux architecture—not copied from a macOS virtual environment. Because pandas and scikit-learn are large compiled dependencies, a Lambda container image or dependency layer is generally safer than manually uploading source files alone.

Supported environment variables:

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Absolute deployed path to the joblib artifact |
| `OUTPUT_BUCKET` | S3 bucket for current and historical JSON outputs |
| `ROAD_STATUS_URL` | JSON endpoint for official road status |
| `ROAD_STATUS_JSON_PATH` | Dot-separated status field; defaults to `status` |
| `ROAD_STATUS_API_KEY` | Optional road-status API credential |
| `ROAD_STATUS_API_KEY_HEADER` | Credential header; defaults to `X-API-Key` |

In production, `OUTPUT_BUCKET=i80-road-closure-hsmith`. The dashboard reads `i80/latest/current_risk.json`; this stable object is overwritten after every successful run. The remaining prefixes are append-only historical records partitioned by UTC date:

- `i80/raw_weather/date=YYYY-MM-DD/` stores the parsed API response used by each run.
- `i80/features/date=YYYY-MM-DD/` stores the cleaned model feature row.
- `i80/predictions/date=YYYY-MM-DD/` stores each prediction result.
- `i80/closure_status/date=YYYY-MM-DD/` stores the road-status response, including unavailable status.

Each run creates a new timestamped JSON object instead of appending rows to a shared CSV. That design makes individual executions easier to audit, replay, debug, and use for future retraining. It also keeps the dashboard's latest-result lookup simple while preserving complete prediction history.

If `OUTPUT_BUCKET` is configured, the Lambda execution role needs `s3:PutObject` permission under `i80/*`. Without it, the same key hierarchy is written beneath `/tmp/i80/` for local testing. If the road-status endpoint is absent or fails, the weather model still runs but the public display becomes `Road Status Unavailable`; the pipeline never assumes that missing status data means the road is open.

## Limitations

- The model identifies statistical association, not causation.
- It relies primarily on weather and time features; crashes, traffic volume, road treatment, and operational decisions are not fully represented.
- The score is trained on historical observed weather, so live use must reproduce the same feature definitions from data available at prediction time.
- Risk categories communicate relative model risk and should not replace official CHP or Caltrans guidance.

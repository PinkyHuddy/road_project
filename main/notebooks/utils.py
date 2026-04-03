import pandas as pd
import numpy as np


def build_closure_intervals(
    closures_df,
    closure_col="closure_time",
    reopen_col="reopening_time",
    missing_reopen_hours=24
):
    """
    Build a clean interval table from closure/reopening timestamps.

    Parameters
    ----------
    closures_df : pd.DataFrame
        DataFrame containing closure and reopening timestamps.
    closure_col : str
        Column name for closure timestamp.
    reopen_col : str
        Column name for reopening timestamp.
    missing_reopen_hours : int
        If reopening time is missing, create a placeholder end time
        = closure_time + missing_reopen_hours. This is only used to define
        the blackout period where weather rows should remain unlabeled/NA.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - closure_start
        - closure_end
        - has_reopening_time
    """
    intervals = closures_df.copy()

    intervals[closure_col] = pd.to_datetime(intervals[closure_col], errors="coerce")
    intervals[reopen_col] = pd.to_datetime(intervals[reopen_col], errors="coerce")

    intervals = intervals.dropna(subset=[closure_col]).copy()

    intervals["closure_start"] = intervals[closure_col]
    intervals["has_reopening_time"] = intervals[reopen_col].notna()

    intervals["closure_end"] = intervals[reopen_col]
    missing_mask = intervals["closure_end"].isna()
    intervals.loc[missing_mask, "closure_end"] = (
        intervals.loc[missing_mask, "closure_start"] +
        pd.Timedelta(hours=missing_reopen_hours)
    )

    intervals = intervals[["closure_start", "closure_end", "has_reopening_time"]]
    intervals = intervals.sort_values("closure_start").reset_index(drop=True)

    return intervals


def apply_closure_to_weather(
    weather_df,
    intervals_df,
    weather_time_col="datetime",
    closure_label_col="closure"
):
    """
    Annotate weather rows using closure intervals.

    Label precedence
    ----------------
    1 (confirmed closure) > NA (unknown after closure with no reopen) > 0 (open)

    Rules
    -----
    - For intervals with a real reopening time:
        mark closure = 1 from the closure hour through the reopening hour.
    - For intervals with NO reopening time:
        mark closure = 1 for the closure hour only,
        then mark the next 23 hourly rows as NA.
    - All other rows get closure = 0.
    """
    weather = weather_df.copy()
    weather[weather_time_col] = pd.to_datetime(weather[weather_time_col], errors="coerce")
    weather[closure_label_col] = 0.0

    # Split intervals into missing-reopen and known-reopen
    missing_reopen = intervals_df.loc[~intervals_df["has_reopening_time"]].copy()
    known_reopen = intervals_df.loc[intervals_df["has_reopening_time"]].copy()

    # Pass 1: assign NA windows for missing reopen cases
    for _, row in missing_reopen.iterrows():
        start = pd.to_datetime(row["closure_start"]).floor("h")

        # closure hour should be 1 later, so NA starts after that
        na_start = start + pd.Timedelta(hours=1)
        na_end = start + pd.Timedelta(hours=23)

        na_mask = (weather[weather_time_col] >= na_start) & (weather[weather_time_col] <= na_end)

        # only write NA where label is still 0
        weather.loc[na_mask & (weather[closure_label_col] == 0), closure_label_col] = np.nan

    # Pass 2: assign all confirmed/known 1s
    for _, row in known_reopen.iterrows():
        start = pd.to_datetime(row["closure_start"]).floor("h")
        end = pd.to_datetime(row["closure_end"]).floor("h")

        one_mask = (weather[weather_time_col] >= start) & (weather[weather_time_col] <= end)
        weather.loc[one_mask, closure_label_col] = 1.0

    # Pass 3: assign closure-hour 1s for missing reopen cases
    for _, row in missing_reopen.iterrows():
        start = pd.to_datetime(row["closure_start"]).floor("h")
        first_hour_mask = weather[weather_time_col] == start
        weather.loc[first_hour_mask, closure_label_col] = 1.0

    return weather

def make_future_closure_target(df, time_col="date", start_col="closure_start", horizon_hours=24):
    df = df.copy().sort_values(time_col).reset_index(drop=True)

    closure_start_times = df.loc[df[start_col] == 1, time_col]
    target = np.zeros(len(df), dtype=int)

    for i, current_time in enumerate(df[time_col]):
        window_end = current_time + pd.Timedelta(hours=horizon_hours)
        target[i] = int(((closure_start_times > current_time) & (closure_start_times <= window_end)).any())

    df[f"will_close_in_{horizon_hours}h"] = target
    return df
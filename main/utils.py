import pandas as pd
import numpy as np


def build_closure_intervals(
    closures_df,
    closure_col="closure_time",
    reopen_col="reopening_time",
    missing_reopen_hours=24
):
    """
    Build raw closure intervals from closure/reopening timestamps.

    If reopening time is missing, create a placeholder end time
    = closure_time + missing_reopen_hours. This is used only to define
    an ambiguous/blackout window for weather labeling.
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


def build_event_intervals(
    intervals_df,
    start_col="closure_start",
    end_col="closure_end",
    has_reopen_col="has_reopening_time",
    max_gap_hours=6
):
    """
    Build event-level closure intervals using two steps:

    1. Collapse rows with the same closure_end into one event
       by taking the earliest closure_start.
    2. Merge overlapping or near-adjacent resulting intervals.

    Rows without reopening times are kept as separate rows in step 1,
    then may still be merged in step 2 if they overlap or are very close.
    """
    df = intervals_df.copy()

    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")

    known = df.loc[df[has_reopen_col]].copy()
    missing = df.loc[~df[has_reopen_col]].copy()

    # Step 1: collapse known rows by shared end time
    if not known.empty:
        known_collapsed = (
            known.groupby(end_col, as_index=False)
            .agg({start_col: "min"})
        )
        known_collapsed = known_collapsed[[start_col, end_col]]
    else:
        known_collapsed = pd.DataFrame(columns=[start_col, end_col])

    # Missing-reopen rows stay separate for now
    missing_kept = missing[[start_col, end_col]].copy()

    combined = pd.concat(
        [known_collapsed, missing_kept],
        ignore_index=True
    ).sort_values(start_col).reset_index(drop=True)

    # Step 2: merge overlap / near-adjacent intervals
    max_gap = pd.Timedelta(hours=max_gap_hours)

    if combined.empty:
        return combined

    merged = []
    current_start = combined.loc[0, start_col]
    current_end = combined.loc[0, end_col]

    for i in range(1, len(combined)):
        next_start = combined.loc[i, start_col]
        next_end = combined.loc[i, end_col]

        if next_start <= current_end + max_gap:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start = next_start
            current_end = next_end

    merged.append((current_start, current_end))

    event_intervals = pd.DataFrame(merged, columns=[start_col, end_col])
    return event_intervals


def apply_closure_to_weather(
    weather_df,
    intervals_df,
    weather_time_col="datetime",
    closure_label_col="closure"
):
    """
    Annotate weather rows using raw closure intervals.

    Label precedence:
    1 (confirmed closure) > NA (unknown after missing reopen) > 0 (open)

    Rules
    -----
    - If reopening time exists:
        mark closure = 1 from closure hour through reopening hour.
    - If reopening time is missing:
        mark closure = 1 for closure hour only,
        then mark next 23 hours as NA.
    - All other rows get 0.
    """
    weather = weather_df.copy()
    weather[weather_time_col] = pd.to_datetime(weather[weather_time_col], errors="coerce")
    weather[closure_label_col] = 0.0

    missing_reopen = intervals_df.loc[~intervals_df["has_reopening_time"]].copy()
    known_reopen = intervals_df.loc[intervals_df["has_reopening_time"]].copy()

    # Pass 1: NA windows for missing reopen cases
    for _, row in missing_reopen.iterrows():
        start = pd.to_datetime(row["closure_start"]).floor("h")
        na_start = start + pd.Timedelta(hours=1)
        na_end = start + pd.Timedelta(hours=23)

        na_mask = (weather[weather_time_col] >= na_start) & (weather[weather_time_col] <= na_end)
        weather.loc[na_mask & (weather[closure_label_col] == 0), closure_label_col] = np.nan

    # Pass 2: confirmed closure windows
    for _, row in known_reopen.iterrows():
        start = pd.to_datetime(row["closure_start"]).floor("h")
        end = pd.to_datetime(row["closure_end"]).floor("h")

        one_mask = (weather[weather_time_col] >= start) & (weather[weather_time_col] <= end)
        weather.loc[one_mask, closure_label_col] = 1.0

    # Pass 3: missing-reopen closure hour only
    for _, row in missing_reopen.iterrows():
        start = pd.to_datetime(row["closure_start"]).floor("h")
        first_hour_mask = weather[weather_time_col] == start
        weather.loc[first_hour_mask, closure_label_col] = 1.0

    return weather


def add_closure_start_column(
    weather_df,
    event_intervals_df,
    weather_time_col="date",
    start_col="closure_start",
    output_col="closure_start"
):
    """
    Mark event-level closure starts in an hourly weather dataframe.

    Uses merged event intervals, not raw post-level intervals.
    """
    df = weather_df.copy()
    df[weather_time_col] = pd.to_datetime(df[weather_time_col], errors="coerce")

    event_start_hours = pd.to_datetime(event_intervals_df[start_col], errors="coerce").dt.floor("h")
    df[output_col] = df[weather_time_col].isin(event_start_hours).astype(int)

    return df


def make_future_closure_target(
    df,
    time_col="date",
    start_col="closure_start",
    closure_col="closure",
    horizon_hours=24,
    open_only=True
):
    """
    Create a target for whether a closure start will occur within the next horizon.

    For each closure start at time T:
    - rows from T - horizon_hours through T - 1 hour get target = 1
    - the closure-start row itself does NOT get target = 1

    If open_only=True, only rows with closure == 0 are labeled positive.
    """
    df = df.copy().sort_values(time_col).reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    target_col = f"will_close_in_{horizon_hours}h"
    df[target_col] = 0

    closure_starts = df.loc[df[start_col] == 1, time_col]

    for start_time in closure_starts:
        window_start = start_time - pd.Timedelta(hours=horizon_hours)
        window_end = start_time

        mask = (df[time_col] >= window_start) & (df[time_col] < window_end)

        if open_only:
            mask = mask & (df[closure_col] == 0)

        df.loc[mask, target_col] = 1

    return df
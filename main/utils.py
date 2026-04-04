import numpy as np
import pandas as pd


def _require_columns(df, required_cols, df_name="DataFrame"):
    """Raise a helpful error if required columns are missing."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def _coerce_datetime(series):
    """Convert a Series to datetime, coercing invalid values to NaT."""
    return pd.to_datetime(series, errors="coerce")


def _align_datetime_series(reference, other, reference_name="reference", other_name="other"):
    """
    Align `other` to the timezone style of `reference`.

    Rules:
    - aware + aware  -> convert `other` to reference timezone
    - naive + naive  -> leave as-is
    - aware + naive or naive + aware -> raise, to avoid silent mistakes
    """
    ref_tz = reference.dt.tz
    other_tz = other.dt.tz

    if ref_tz is None and other_tz is None:
        return other
    if ref_tz is not None and other_tz is not None:
        return other.dt.tz_convert(ref_tz)

    raise ValueError(
        f"Timezone mismatch: {reference_name} has tz={ref_tz}, "
        f"but {other_name} has tz={other_tz}. "
        "Make both datetime columns either timezone-aware in the same timezone "
        "or both timezone-naive before calling this function."
    )


def build_closure_intervals(
    closures_df,
    closure_col="closure_time",
    reopen_col="reopening_time",
    missing_reopen_hours=24,
):
    """
    Build raw closure intervals from closure / reopening timestamps.

    If reopening time is missing, create a placeholder end time:
        closure_start + missing_reopen_hours

    This placeholder end time is only used to define the ambiguous
    post-closure window during weather labeling.
    """
    _require_columns(closures_df, [closure_col, reopen_col], df_name="closures_df")

    if missing_reopen_hours <= 0:
        raise ValueError("missing_reopen_hours must be positive.")

    intervals = closures_df.copy()
    intervals[closure_col] = _coerce_datetime(intervals[closure_col])
    intervals[reopen_col] = _coerce_datetime(intervals[reopen_col])

    intervals = intervals.dropna(subset=[closure_col]).copy()

    intervals["closure_start"] = intervals[closure_col]
    intervals["has_reopening_time"] = intervals[reopen_col].notna()
    intervals["closure_end"] = intervals[reopen_col]

    missing_mask = intervals["closure_end"].isna()
    intervals.loc[missing_mask, "closure_end"] = (
        intervals.loc[missing_mask, "closure_start"]
        + pd.Timedelta(hours=missing_reopen_hours)
    )

    bad_known = (
        intervals["has_reopening_time"]
        & (intervals["closure_end"] < intervals["closure_start"])
    )
    if bad_known.any():
        raise ValueError(
            "Found reopening times earlier than closure times in known intervals."
        )

    intervals = (
        intervals[["closure_start", "closure_end", "has_reopening_time"]]
        .sort_values("closure_start")
        .reset_index(drop=True)
    )

    return intervals


def build_event_intervals(
    intervals_df,
    start_col="closure_start",
    end_col="closure_end",
    has_reopen_col="has_reopening_time",
    max_gap_hours=6,
):
    """
    Build event-level closure intervals in two steps:

    1. Collapse rows with the same closure_end into one event
       by taking the earliest closure_start.
    2. Merge overlapping or near-adjacent resulting intervals
       using max_gap_hours.
    """
    _require_columns(intervals_df, [start_col, end_col], df_name="intervals_df")

    if max_gap_hours < 0:
        raise ValueError("max_gap_hours must be nonnegative.")

    df = intervals_df.copy()
    df[start_col] = _coerce_datetime(df[start_col])
    df[end_col] = _coerce_datetime(df[end_col])

    df = df.dropna(subset=[start_col, end_col]).copy()

    if has_reopen_col not in df.columns:
        df[has_reopen_col] = True

    known = df.loc[df[has_reopen_col]].copy()
    missing = df.loc[~df[has_reopen_col]].copy()

    # Step 1: collapse known rows by shared end time
    if not known.empty:
        known_collapsed = (
            known.groupby(end_col, as_index=False)
            .agg({start_col: "min"})
            [[start_col, end_col]]
        )
    else:
        known_collapsed = pd.DataFrame(columns=[start_col, end_col])

    # Missing-reopen rows stay separate for now
    missing_kept = missing[[start_col, end_col]].copy()

    combined = (
        pd.concat([known_collapsed, missing_kept], ignore_index=True)
        .sort_values(start_col)
        .reset_index(drop=True)
    )

    if combined.empty:
        return combined

    max_gap = pd.Timedelta(hours=max_gap_hours)

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
    closure_label_col="closure",
):
    """
    Annotate hourly weather rows using raw closure intervals.

    Label precedence:
        1 (confirmed closure) > NA (ambiguous after missing reopen) > 0 (open)

    Rules
    -----
    - If reopening time exists:
        mark closure = 1 from closure hour through reopening hour.
    - If reopening time is missing:
        mark closure = 1 for closure hour only,
        then mark the next 23 hours as NA.
    - All other rows get 0.
    """
    _require_columns(weather_df, [weather_time_col], df_name="weather_df")
    _require_columns(
        intervals_df,
        ["closure_start", "closure_end", "has_reopening_time"],
        df_name="intervals_df",
    )

    weather = weather_df.copy()
    weather[weather_time_col] = _coerce_datetime(weather[weather_time_col])
    weather[closure_label_col] = 0.0

    intervals = intervals_df.copy()
    intervals["closure_start"] = _coerce_datetime(intervals["closure_start"])
    intervals["closure_end"] = _coerce_datetime(intervals["closure_end"])

    intervals["closure_start"] = _align_datetime_series(
        weather[weather_time_col],
        intervals["closure_start"],
        reference_name=f"weather_df[{weather_time_col!r}]",
        other_name="intervals_df['closure_start']",
    )
    intervals["closure_end"] = _align_datetime_series(
        weather[weather_time_col],
        intervals["closure_end"],
        reference_name=f"weather_df[{weather_time_col!r}]",
        other_name="intervals_df['closure_end']",
    )

    missing_reopen = intervals.loc[~intervals["has_reopening_time"]].copy()
    known_reopen = intervals.loc[intervals["has_reopening_time"]].copy()

    # Pass 1: mark ambiguous windows for missing-reopen cases
    for _, row in missing_reopen.iterrows():
        start = row["closure_start"].floor("h")
        na_start = start + pd.Timedelta(hours=1)
        na_end = start + pd.Timedelta(hours=23)

        na_mask = (
            (weather[weather_time_col] >= na_start)
            & (weather[weather_time_col] <= na_end)
        )
        weather.loc[
            na_mask & (weather[closure_label_col] == 0),
            closure_label_col
        ] = np.nan

    # Pass 2: mark confirmed closure windows
    for _, row in known_reopen.iterrows():
        start = row["closure_start"].floor("h")
        end = row["closure_end"].floor("h")

        closed_mask = (
            (weather[weather_time_col] >= start)
            & (weather[weather_time_col] <= end)
        )
        weather.loc[closed_mask, closure_label_col] = 1.0

    # Pass 3: for missing-reopen rows, mark only the closure hour as confirmed closed
    for _, row in missing_reopen.iterrows():
        start = row["closure_start"].floor("h")
        first_hour_mask = weather[weather_time_col] == start
        weather.loc[first_hour_mask, closure_label_col] = 1.0

    return weather


def add_closure_start_column(
    weather_df,
    event_intervals_df,
    weather_time_col="date",
    start_col="closure_start",
    output_col="closure_start",
):
    """
    Mark event-level closure starts in an hourly weather dataframe.

    Event starts are floored to the hourly grid before matching.
    """
    _require_columns(weather_df, [weather_time_col], df_name="weather_df")
    _require_columns(event_intervals_df, [start_col], df_name="event_intervals_df")

    df = weather_df.copy()
    df[weather_time_col] = _coerce_datetime(df[weather_time_col])

    event_start_hours = _coerce_datetime(event_intervals_df[start_col]).dt.floor("h")
    event_start_hours = _align_datetime_series(
        df[weather_time_col],
        event_start_hours,
        reference_name=f"weather_df[{weather_time_col!r}]",
        other_name=f"event_intervals_df[{start_col!r}]",
    )

    event_start_hours = pd.Series(event_start_hours).drop_duplicates()
    df[output_col] = df[weather_time_col].isin(event_start_hours).astype(int)

    return df


def make_future_closure_target(
    df,
    time_col="date",
    start_col="closure_start",
    closure_col="closure",
    horizon_hours=24,
    open_only=True,
):
    """
    Create a target for whether a closure start will occur within the next horizon.

    For each closure start at time T:
    - rows from T - horizon_hours through T - 1 hour get target = 1
    - the closure-start row itself does NOT get target = 1

    If open_only=True, only rows with closure == 0 are labeled positive.
    """
    required_cols = [time_col, start_col]
    if open_only:
        required_cols.append(closure_col)
    _require_columns(df, required_cols, df_name="df")

    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive.")

    out = df.copy().sort_values(time_col).reset_index(drop=True)
    out[time_col] = _coerce_datetime(out[time_col])

    target_col = f"will_close_in_{horizon_hours}h"
    out[target_col] = 0

    closure_starts = (
        out.loc[out[start_col] == 1, time_col]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    for start_time in closure_starts:
        window_start = start_time - pd.Timedelta(hours=horizon_hours)
        window_end = start_time

        mask = (out[time_col] >= window_start) & (out[time_col] < window_end)

        if open_only:
            mask = mask & (out[closure_col] == 0)

        out.loc[mask, target_col] = 1

    return out
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
    # An all-missing datetime series has no meaningful timezone. Returning it
    # unchanged lets callers handle censored/unknown timestamps without a
    # false timezone-mismatch error.
    if not other.notna().any():
        return other

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
    closure_flag_col=None,
    metadata_cols=None,
):
    """
    Build raw closure intervals from closure / reopening timestamps.

    Only rows explicitly marked as closures are retained when
    ``closure_flag_col`` is supplied. Missing reopening times remain missing;
    they are right-censored observations, not invented 24-hour durations.
    """
    _require_columns(closures_df, [closure_col, reopen_col], df_name="closures_df")

    intervals = closures_df.copy()
    if closure_flag_col is not None:
        _require_columns(intervals, [closure_flag_col], df_name="closures_df")
        intervals = intervals.loc[intervals[closure_flag_col].eq(1)].copy()
    intervals[closure_col] = _coerce_datetime(intervals[closure_col])
    intervals[reopen_col] = _coerce_datetime(intervals[reopen_col])

    intervals = intervals.dropna(subset=[closure_col]).copy()

    intervals["closure_start"] = intervals[closure_col]
    intervals["has_reopening_time"] = intervals[reopen_col].notna()
    intervals["closure_end"] = intervals[reopen_col]

    known = intervals["has_reopening_time"]
    if known.any():
        known_ends = _align_datetime_series(
            intervals.loc[known, "closure_start"],
            intervals.loc[known, "closure_end"],
            reference_name="known closure times",
            other_name="known reopening times",
        )
        if (known_ends < intervals.loc[known, "closure_start"]).any():
            raise ValueError(
                "Found reopening times earlier than closure times in known intervals."
            )

    metadata_cols = metadata_cols or []
    _require_columns(intervals, metadata_cols, df_name="closures_df")
    output_cols = [
        "closure_start", "closure_end", "has_reopening_time", *metadata_cols
    ]
    intervals = (
        intervals[output_cols]
        .sort_values("closure_start")
        .reset_index(drop=True)
    )

    return intervals


def build_event_intervals(
    intervals_df,
    start_col="closure_start",
    end_col="closure_end",
    has_reopen_col="has_reopening_time",
    max_gap_hours=12,
):
    """
    Build event-level closure intervals in two steps:

    1. Collapse rows with the same closure_end into one event
       by taking the earliest closure_start.
    2. Attach closure updates to an already-known interval when their start
       falls inside it.
    3. Cluster remaining unknown-end closure reports by ``max_gap_hours``.

    The output keeps ``has_reopening_time`` and ``source_record_count`` so
    observed-duration events can be distinguished from censored events.
    Unknown ends are never replaced with a fabricated duration.
    """
    _require_columns(intervals_df, [start_col, end_col], df_name="intervals_df")

    if max_gap_hours < 0:
        raise ValueError("max_gap_hours must be nonnegative.")

    df = intervals_df.copy()
    df[start_col] = _coerce_datetime(df[start_col])
    df[end_col] = _coerce_datetime(df[end_col])

    df = df.dropna(subset=[start_col]).copy()

    if has_reopen_col not in df.columns:
        df[has_reopen_col] = True

    df[has_reopen_col] = df[has_reopen_col].fillna(False).astype(bool)
    known = df.loc[df[has_reopen_col] & df[end_col].notna()].copy()
    missing = df.loc[~(df[has_reopen_col] & df[end_col].notna())].copy()

    # Step 1: collapse known rows by shared end time
    if not known.empty:
        known_collapsed = known.groupby(end_col, as_index=False).agg(
            **{start_col: (start_col, "min")},
            source_record_count=(start_col, "size"),
        )
        known_collapsed[has_reopen_col] = True
    else:
        known_collapsed = pd.DataFrame(
            columns=[start_col, end_col, "source_record_count", has_reopen_col]
        )

    # A missing-end report inside a known interval is an update, not a new event.
    unattached = []
    for _, row in missing.sort_values(start_col).iterrows():
        containing = known_collapsed.loc[
            (known_collapsed[start_col] <= row[start_col])
            & (known_collapsed[end_col] >= row[start_col])
        ]
        if containing.empty:
            unattached.append(row[start_col])
        else:
            idx = containing[end_col].idxmin()
            known_collapsed.loc[idx, "source_record_count"] += 1

    # Consecutive unmatched reports close together are treated as updates to
    # one censored event. This assumption remains explicit in the output.
    censored = []
    max_gap = pd.Timedelta(hours=max_gap_hours)
    for start in sorted(unattached):
        if not censored or start - censored[-1][1] > max_gap:
            censored.append([start, start, 1])
        else:
            censored[-1][1] = start
            censored[-1][2] += 1

    censored_df = pd.DataFrame(
        [
            {
                start_col: first,
                end_col: pd.NaT,
                "last_closure_report": last,
                "source_record_count": count,
                has_reopen_col: False,
            }
            for first, last, count in censored
        ]
    )
    known_collapsed["last_closure_report"] = pd.NaT
    out = pd.concat([known_collapsed, censored_df], ignore_index=True)
    out = out.sort_values(start_col).reset_index(drop=True)
    out[start_col] = _coerce_datetime(out[start_col])
    out[end_col] = _coerce_datetime(out[end_col])
    out.insert(0, "event_id", [f"I80-{i:04d}" for i in range(1, len(out) + 1)])
    out["duration_status"] = np.where(
        out[has_reopen_col], "observed", "right_censored"
    )
    out["duration_hours"] = (
        out[end_col] - out[start_col]
    ).dt.total_seconds().div(3600).where(out[has_reopen_col])
    return out


def apply_closure_to_weather(
    weather_df,
    intervals_df,
    weather_time_col="datetime",
    closure_label_col="closure",
    ambiguous_hours=24,
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
    known_reopen = intervals.loc[
        intervals["has_reopening_time"] & intervals["closure_end"].notna()
    ].copy()

    # Pass 1: mark ambiguous windows for missing-reopen cases
    for _, row in missing_reopen.iterrows():
        start = row["closure_start"].floor("h")
        na_start = start + pd.Timedelta(hours=1)
        na_end = start + pd.Timedelta(hours=ambiguous_hours - 1)

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


def make_future_road_status_target(
    df,
    time_col="date",
    closure_col="closure",
    horizon_hours=24,
):
    """Label whether the road is closed at any point in the future horizon.

    The current hour is excluded. A row is positive when at least one exact
    hourly timestamp from ``t + 1`` through ``t + horizon_hours`` is confirmed
    closed. It is negative only when every future hour is present and confirmed
    open. Otherwise the target is unknown.
    """
    _require_columns(df, [time_col, closure_col], df_name="df")
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive.")

    out = df.copy().sort_values(time_col).reset_index(drop=True)
    out[time_col] = _coerce_datetime(out[time_col])

    any_closed = pd.Series(False, index=out.index)
    any_unknown = pd.Series(False, index=out.index)

    for step in range(1, horizon_hours + 1):
        future_status = out[closure_col].shift(-step)
        future_time = out[time_col].shift(-step)
        exact_hour = future_time.sub(out[time_col]).eq(pd.Timedelta(hours=step))

        any_closed |= exact_hour & future_status.eq(1)
        any_unknown |= ~exact_hour | future_status.isna()

    target_col = f"road_closed_within_{horizon_hours}h"
    out[target_col] = np.where(any_closed, 1.0, np.where(any_unknown, np.nan, 0.0))
    return out

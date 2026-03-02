from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_year_folders(parent_directory):
    """
    Reads all subdirectories in the parent_directory and returns a list
    of names that appear to be four-digit year folders.

    Args:
        parent_directory (str): The path to the directory containing year folders.

    Returns:
        list: A list of strings, where each string is a year (e.g., "2023").
    """
    p = Path(parent_directory)
    # Use iterdir() to list items and filter for directories
    subdirectories = [x for x in p.iterdir() if x.is_dir()]
    
    year_folders = []
    for subdir in subdirectories:
        # Check if the folder name is a 4-digit number
        if subdir.name.isdigit() and len(subdir.name) == 4:
            year_folders.append(subdir.name)
            
    return year_folders

def create_year_folder(year, output_base="extracted_data"):
    """
    Creates the year folder that I will be extracting my data into."
    """
    output_dir = Path(output_base) / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_date(file_path: str) -> str:
    """
    Extracts date string (YYYY_MM_DD) from a PeMS .txt.gz file path.
    Going to use this as a helper with the extract_file function.
    """
    p = Path(file_path)
    # Remove .gz first
    stem = p.stem  # removes .gz → leaves ...2004_01_01.txt
    # Remove .txt
    stem = Path(stem).stem  # removes .txt
    # Date is last 3 underscore-separated parts
    date_parts = stem.split("_")[-3:]
    return "_".join(date_parts)

def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans dataframe and computes closure flags.
    Assumes raw PeMS-style columns already present.
    """
    stationIds = [
        3054021, 319677, 319673, 319675, 319680, 319674,
        3411021, 3411024, 3023124, 3023121, 319416, 318690,
        317786, 317791, 317789, 317787, 317788, 317797,
        317798, 3412081, 3412064, 3412061, 3047112, 3047111,
        3047113, 3047108, 3047101, 3412054, 3047097, 3047094,
        3047098, 3047084, 3047085, 3047081, 3047073, 3047072,
        3047075, 3047131, 3047042, 3047043, 314000, 316261,
        316249, 3412041, 316214, 316213, 3038021
    ]
    # Filter stations
    df = df[df["Station"].isin(stationIds)]
    # Timestamp handling
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Station", "Timestamp"])
    # Low flow
    df["low_flow"] = df["Total Flow"] <= 5
    # -----------------
    # OPTION A
    # -----------------
    df["closure_A"] = (
        df.groupby("Station")["low_flow"]
          .transform(lambda s: s.rolling(6, min_periods=6).sum() >= 5)
    )
    # -----------------
    # OPTION B
    # -----------------
    low_counts = df.groupby("Timestamp")["low_flow"].sum()
    multi_station_low = low_counts >= 2

    closure_B_times = (
        multi_station_low
            .rolling(3, min_periods=3)
            .sum() == 3
    )
    df["closure_B"] = (
        df["Timestamp"].map(closure_B_times).fillna(False)
        & df["low_flow"]
    )
    df["closure_flag"] = df["closure_A"] | df["closure_B"]
    # -----------------
    # Narrowing down to just what is closed and the months we want
    # -----------------
    df = df[((df["Timestamp"].dt.month > 9) | (df["Timestamp"].dt.month < 6)) & (df["closure_flag"] == True)]
    return df

def extract_closure_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts row-level closure_flag into event-level closure table
    with additional descriptive statistics.
    """

    df = df.sort_values(["Station", "Timestamp"]).copy()

    # Identify start of closure events
    df["prev_flag"] = df.groupby("Station")["closure_flag"].shift(fill_value=False)
    df["start_event"] = (~df["prev_flag"]) & (df["closure_flag"])

    # Event ID per station
    df["event_id"] = df.groupby("Station")["start_event"].cumsum()

    # Keep only closure rows
    closure_rows = df[df["closure_flag"]].copy()

    if closure_rows.empty:
        return pd.DataFrame(columns=[
            "Station",
            "closure_date",
            "start_time",
            "end_time",
            "duration_minutes",
            "n_intervals",
            "mean_flow",
            "mean_speed",
            "max_occupancy",
            "multi_station_event"
        ])

    # Detect multi-station timestamps
    station_counts = closure_rows.groupby("Timestamp")["Station"].nunique()
    multi_station_times = station_counts >= 2

    closure_rows["multi_station_time"] = (
        closure_rows["Timestamp"].map(multi_station_times).fillna(False)
    )

    # Aggregate events
    events = (
        closure_rows
        .groupby(["Station", "event_id"])
        .agg(
            start_time=("Timestamp", "min"),
            end_time=("Timestamp", "max"),
            n_intervals=("Timestamp", "count"),
            mean_flow=("Total Flow", "mean"),
            mean_speed=("Average Speed", "mean"),
            max_occupancy=("Avg Occupancy", "max"),
            multi_station_event=("multi_station_time", "any")
        )
        .reset_index()
    )

    # Duration (inclusive of last interval)
    events["duration_minutes"] = (
        (events["end_time"] - events["start_time"])
        .dt.total_seconds() / 60
        + 5
    )

    events["closure_date"] = events["start_time"].dt.date

    return events[
        [
            "Station",
            "closure_date",
            "start_time",
            "end_time",
            "duration_minutes",
            "n_intervals",
            "mean_flow",
            "mean_speed",
            "max_occupancy",
            "multi_station_event"
        ]
    ]

def extract_file(filePath):
    columnNames = [
        "Timestamp", "Station", "District#", "Freeway#",
        "Direction of Travel", "Lane Type", "Station Length",
        "Samples", "% Observed", "Total Flow",
        "Avg Occupancy", "Average Speed"
    ]

    unnamedColumns = ["unnamed" + str(i) for i in np.arange(40)]

    df = pd.read_csv(
        filePath,
        compression="gzip",
        header=None,
        names=columnNames + unnamedColumns
    )

    df = df.iloc[:, 0:12]
    df = df[(df["Freeway#"] == 80) & (df["Direction of Travel"] == "E")]
    df = df.dropna()

    df = clean_and_label(df)
    df = extract_closure_events(df)

    return df


rawFilePath = "/Users/hudson/Desktop/road_project/data/unextracted_data"
years = get_year_folders(rawFilePath)

all_events = []
total_events = 0

for year in years:

    print(f"\n===== Processing Year {year} =====")

    yearPath = Path(rawFilePath) / str(year)
    files = sorted(yearPath.rglob("*.gz"))

    year_event_count = 0

    for i, file in enumerate(files, 1):

        events = extract_file(file)

        if not events.empty:
            all_events.append(events)
            year_event_count += len(events)
            total_events += len(events)

        # Optional: lightweight progress indicator every 50 files
        if i % 50 == 0:
            print(f"  Processed {i}/{len(files)} files...")

    print(f"Year {year} complete.")
    print(f"  Closure events this year: {year_event_count}")
    print(f"  Running total events: {total_events}")

# Final concat
if all_events:
    final_events = pd.concat(all_events, ignore_index=True)
else:
    final_events = pd.DataFrame()

final_events.to_csv(
    "/Users/hudson/Desktop/road_project/data/ALL_CLOSURE_EVENTS.csv",
    index=False
)

print("\n===== DONE =====")
print(f"Total closure events across all years: {len(final_events)}")
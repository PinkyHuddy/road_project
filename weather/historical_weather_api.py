"""Download historical Donner Pass weather using only the requests library."""

import requests
from pathlib import Path


URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_PATH = Path(__file__).resolve().parent / "historical_weather_api.csv"

HOURLY_VARIABLES = [
    "temperature_2m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_high",
    "cloud_cover_mid",
    "wind_direction_100m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_speed_10m",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "apparent_temperature",
    "dew_point_2m",
    "relative_humidity_2m",
    "is_day",
    "snow_depth_water_equivalent",
    "sunshine_duration",
]

PARAMS = {
    "latitude": 39.342964,
    "longitude": -120.328979,
    "start_date": "2017-02-21",
    "end_date": "2026-03-04",
    "hourly": ",".join(HOURLY_VARIABLES),
    "timezone": "UTC",
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch",
}


def _csv_value(value):
    """Convert a scalar to a valid CSV field without importing csv."""
    if value is None:
        return ""
    text = str(value)
    if any(character in text for character in (",", '"', "\n", "\r")):
        return '"' + text.replace('"', '""') + '"'
    return text


def _utc_timestamp(value):
    """Match the UTC timestamp representation used by the previous pandas output."""
    timestamp = value.replace("T", " ")
    if len(timestamp) == 16:
        timestamp += ":00"
    if not timestamp.endswith(("Z", "+00:00")):
        timestamp += "+00:00"
    return timestamp.replace("Z", "+00:00")


def fetch_historical_weather():
    """Request and validate the historical hourly weather response."""
    response = requests.get(URL, params=PARAMS, timeout=120)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo response is missing hourly data: {payload}")

    hourly = payload["hourly"]
    if "time" not in hourly:
        raise ValueError("Open-Meteo hourly data is missing timestamps.")

    expected_rows = len(hourly["time"])
    for variable in HOURLY_VARIABLES:
        if variable not in hourly:
            raise ValueError(f"Open-Meteo hourly data is missing {variable!r}.")
        if len(hourly[variable]) != expected_rows:
            raise ValueError(
                f"Open-Meteo returned {len(hourly[variable])} values for {variable!r}; "
                f"expected {expected_rows}."
            )

    return payload


def save_hourly_csv(payload, output_path=OUTPUT_PATH):
    """Write the response using the original date-first column order."""
    hourly = payload["hourly"]
    columns = ["date", *HOURLY_VARIABLES]

    with Path(output_path).open("w", encoding="utf-8", newline="") as file:
        file.write(",".join(columns) + "\n")
        for row_number, timestamp in enumerate(hourly["time"]):
            row = [_utc_timestamp(timestamp)]
            row.extend(hourly[variable][row_number] for variable in HOURLY_VARIABLES)
            file.write(",".join(_csv_value(value) for value in row) + "\n")


def main():
    payload = fetch_historical_weather()
    save_hourly_csv(payload)

    print(f"Coordinates: {payload.get('latitude')}°N {payload.get('longitude')}°E")
    print(f"Elevation: {payload.get('elevation')} m asl")
    print(f"Timezone: {payload.get('timezone')} {payload.get('timezone_abbreviation')}")
    print(f"Timezone difference to GMT+0: {payload.get('utc_offset_seconds')}s")
    print(f"Saved {len(payload['hourly']['time'])} hourly rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 39.342964,
	"longitude": -120.328979,
	"start_date": "2017-02-21",
	"end_date": "2026-03-04",
	"hourly": ["temperature_2m", "cloud_cover", "cloud_cover_low", "cloud_cover_high", "cloud_cover_mid", "wind_direction_100m", "wind_direction_10m", "wind_speed_100m", "wind_speed_10m", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "precipitation", "apparent_temperature", "dew_point_2m", "relative_humidity_2m", "is_day", "snow_depth_water_equivalent", "sunshine_duration"],
	"timezone": "America/Los_Angeles",
	"temperature_unit": "fahrenheit",
	"wind_speed_unit": "mph",
	"precipitation_unit": "inch",
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(2).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(3).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(4).ValuesAsNumpy()
hourly_wind_direction_100m = hourly.Variables(5).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(6).ValuesAsNumpy()
hourly_wind_speed_100m = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
hourly_rain = hourly.Variables(9).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(10).ValuesAsNumpy()
hourly_snow_depth = hourly.Variables(11).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(12).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(13).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(14).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(15).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(16).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(17).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(18).ValuesAsNumpy()
hourly_is_day = hourly.Variables(19).ValuesAsNumpy()
hourly_snow_depth_water_equivalent = hourly.Variables(20).ValuesAsNumpy()
hourly_sunshine_duration = hourly.Variables(21).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["snow_depth"] = hourly_snow_depth
hourly_data["weather_code"] = hourly_weather_code
hourly_data["pressure_msl"] = hourly_pressure_msl
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["precipitation"] = hourly_precipitation
hourly_data["apparent_temperature"] = hourly_apparent_temperature
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["is_day"] = hourly_is_day
hourly_data["snow_depth_water_equivalent"] = hourly_snow_depth_water_equivalent
hourly_data["sunshine_duration"] = hourly_sunshine_duration

hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe.to_csv("/Users/hudson/Desktop/road_project/weather/weather_date.csv")
print("\nHourly data\n", hourly_dataframe)

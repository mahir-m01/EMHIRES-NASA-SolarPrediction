import pandas as pd
import requests
import time

# Geographic Centroids for all EMHIRES Countries
countries = {
    'AT': {'lat': 47.51, 'lon': 14.55}, 'BE': {'lat': 50.50, 'lon': 4.47},
    'BG': {'lat': 42.73, 'lon': 25.48}, 'CH': {'lat': 46.81, 'lon': 8.22},
    'CY': {'lat': 35.12, 'lon': 33.42}, 'CZ': {'lat': 49.81, 'lon': 15.47},
    'DE': {'lat': 51.16, 'lon': 10.45}, 'DK': {'lat': 56.26, 'lon': 9.50},
    'EE': {'lat': 58.59, 'lon': 25.01}, 'EL': {'lat': 39.07, 'lon': 21.82},
    'ES': {'lat': 40.46, 'lon': -3.74}, 'FI': {'lat': 61.92, 'lon': 25.74},
    'FR': {'lat': 46.22, 'lon': 2.21},  'HR': {'lat': 45.10, 'lon': 15.20},
    'HU': {'lat': 47.16, 'lon': 19.50}, 'IE': {'lat': 53.41, 'lon': -8.24},
    'IT': {'lat': 41.87, 'lon': 12.56}, 'LT': {'lat': 55.16, 'lon': 23.88},
    'LU': {'lat': 49.81, 'lon': 6.12},  'LV': {'lat': 56.87, 'lon': 24.60},
    'NL': {'lat': 52.13, 'lon': 5.29},  'NO': {'lat': 60.47, 'lon': 8.46},
    'PL': {'lat': 51.91, 'lon': 19.14}, 'PT': {'lat': 39.39, 'lon': -8.22},
    'RO': {'lat': 45.94, 'lon': 24.96}, 'SE': {'lat': 60.12, 'lon': 18.64},
    'SI': {'lat': 46.15, 'lon': 14.99}, 'SK': {'lat': 48.66, 'lon': 19.69},
    'UK': {'lat': 55.37, 'lon': -3.43}
}

all_data = []

print("Starting Data Fetch (2001-2015)... This might take atleast 20 minutes.")

for code, coords in countries.items():
    print(f"Fetching {code}")
    for year in range(2001, 2016):
        url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
               f"parameters=ALLSKY_SFC_SW_DWN,T2M,WS2M&community=RE&longitude={coords['lon']}"
               f"&latitude={coords['lat']}&start={year}0101&end={year}1231&format=JSON")
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                json_data = r.json()['properties']['parameter']
                df_year = pd.DataFrame(json_data)
                df_year['Country'] = code
                all_data.append(df_year)
                print(f"  {year} OK")
            else:
                print(f"  {year} Failed (Status: {r.status_code})")
            time.sleep(0.5) # Delay to respect rate limiting
        except Exception as e:
            print(f"  Error on {year}: {e}")

if all_data:
    final_weather = pd.concat(all_data)
    final_weather.index = pd.to_datetime(final_weather.index, format='%Y%m%d%H')
    final_weather.index.name = 'Timestamp'
    
    # Rename for consistency
    final_weather = final_weather.rename(columns={
        'ALLSKY_SFC_SW_DWN': 'Irradiance',
        'T2M': 'Temperature',
        'WS2M': 'Wind_Speed'
    })
    
    final_weather.to_csv('nasa_weather_master.csv')
    print("Done.")
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gc

df = pd.read_csv('EMHIRESPV_TSh_CF_Country_19862015.csv')


# Extracting the first 24 hours for a single country (ES - Spain for example)
one_day = df['ES'].iloc[0:24]

plt.figure(figsize=(10, 6))
plt.plot(one_day, marker='o', color='gold', linewidth=2, label='Solar Efficiency')
plt.fill_between(range(24), one_day, color='yellow', alpha=0.3)

plt.annotate('Night: No Sun', xy=(2, 0), xytext=(2, 0.1), arrowprops=dict(arrowstyle='->'))
plt.annotate('Sunrise: Production Starts', xy=(7, 0.05), xytext=(1, 0.3), arrowprops=dict(arrowstyle='->'))
plt.annotate('Solar Noon: Peak Power', xy=(13, one_day.max()), xytext=(15, 0.5), arrowprops=dict(arrowstyle='->'))

plt.title('A Single Day in the Life of a Solar Panel')
plt.ylabel('Efficiency (Capacity Factor 0.0 to 1.0)')
plt.xlabel('Hour of the Day (0-23)')
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Data Cleaning - 



# Check for NASA -999 flags
nasa_flags = (df == -999).sum().sum()

# Check for EMHIRES Null/NaN values
null_values = df.isnull().sum().sum()

print(f"NASA -999 Flags Found: {nasa_flags}")
print(f"Empty/NaN Cells Found: {null_values}")

# Create Verified Calendar (Starting 1986)
start_date = '1986-01-01 00:00:00'
timestamps = pd.date_range(start=start_date, periods=len(df), freq='h')
df['Timestamp'] = timestamps

# Reshaping / Melting
# Moving countries from columns to rows
df_long = pd.melt(df, id_vars=['Timestamp'], var_name='Country', value_name='Capacity_Factor')


# Remove 1986-2000 to match NASA's Hourly API availability

df_long = df_long[df_long['Timestamp'].dt.year >= 2001]

# Feature Extraction
df_long['Hour'] = df_long['Timestamp'].dt.hour
df_long['Month'] = df_long['Timestamp'].dt.month

print("New Data Shape (2001-2015):", df_long.shape)
print(df_long[['Timestamp', 'Country', 'Capacity_Factor', 'Hour', 'Month']].head(10))



# Merging - 


# Load NASA Weather Data
df_weather = pd.read_csv('nasa_weather_master.csv')
df_weather['Timestamp'] = pd.to_datetime(df_weather['Timestamp'])

# RAM Optimization: Downcast to 32-bit
float_cols = df_weather.select_dtypes(include=['float64']).columns
df_weather[float_cols] = df_weather[float_cols].astype('float32')

# The Master Merge
df_master = pd.merge(df_long, df_weather, on=['Timestamp', 'Country'], how='inner')


# Count NaN
null_counts = df_master[['Irradiance', 'Temperature', 'Wind_Speed']].isnull().sum()
print(f"Standard Null Values (NaN):\n{null_counts}\n")

# Count NASA Sentinels (-999.0)

sentinel_counts = (df_master[['Irradiance', 'Temperature', 'Wind_Speed']] == -999.0).sum()
print(f"NASA Sentinel Placeholders (-999.0):\n{sentinel_counts}")


# Filtering for columns and removing sensor errors
df_master = df_master[
    (df_master['Irradiance'] >= 0) & 
    (df_master['Temperature'] != -999.0) & 
    (df_master['Wind_Speed'] != -999.0)
]

# RAM Cleanup
del df_weather
gc.collect()

print(f"Post-cleaning row count: {len(df_master)}")
print(f"Final Shape: {df_master.shape}")
print(df_master.head())


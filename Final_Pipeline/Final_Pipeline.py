import pandas as pd
import numpy as np
import gc
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('EMHIRESPV_TSh_CF_Country_19862015.csv')

start_date = '1986-01-01 00:00:00'
timestamps = pd.date_range(start=start_date, periods=len(df), freq='h')
df['Timestamp'] = timestamps

df_long = pd.melt(df, id_vars=['Timestamp'], var_name='Country', value_name='Capacity_Factor')
df_long = df_long[df_long['Timestamp'].dt.year >= 2001]
df_long['Hour'] = df_long['Timestamp'].dt.hour
df_long['Month'] = df_long['Timestamp'].dt.month

df_weather = pd.read_csv('nasa_weather_master.csv')
df_weather['Timestamp'] = pd.to_datetime(df_weather['Timestamp'])
float_cols = df_weather.select_dtypes(include=['float64']).columns
df_weather[float_cols] = df_weather[float_cols].astype('float32')

df_master = pd.merge(df_long, df_weather, on=['Timestamp', 'Country'], how='inner')

df_master = df_master[
    (df_master['Irradiance'] >= 0) & 
    (df_master['Temperature'] != -999.0) & 
    (df_master['Wind_Speed'] != -999.0)
]

del df_weather
gc.collect()

df_master = pd.get_dummies(df_master, columns=['Country'], prefix='Country', dtype=int)
df_master.to_csv('merged_encoded.csv', index=False)

X = df_master.drop(columns=['Timestamp', 'Capacity_Factor'])
y = df_master['Capacity_Factor']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

print(f"""
Linear Regression Training:
Mean Absolute Error (MAE):      {mean_absolute_error(y_train, lr_train_pred):.4f}
Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_train, lr_train_pred)):.4f}
R-Squared Accuracy:             {r2_score(y_train, lr_train_pred):.4f}

Linear Regression Testing:
Mean Absolute Error (MAE):      {mean_absolute_error(y_test, lr_test_pred):.4f}
Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, lr_test_pred)):.4f}
R-Squared Accuracy:             {r2_score(y_test, lr_test_pred):.4f}
""")

joblib.dump(lr_model, 'solar_model_lr.pkl')

rfr_model = RandomForestRegressor(n_estimators=100, random_state=67, n_jobs=-1, max_depth=12)
rfr_model.fit(X_train, y_train)

rfr_train_pred = rfr_model.predict(X_train)
rfr_test_pred = rfr_model.predict(X_test)

print(f"""
Random Forest Training:
Mean Absolute Error (MAE):      {mean_absolute_error(y_train, rfr_train_pred):.4f}
Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_train, rfr_train_pred)):.4f}
R-Squared Accuracy:             {r2_score(y_train, rfr_train_pred):.4f}

Random Forest Testing:
Mean Absolute Error (MAE):      {mean_absolute_error(y_test, rfr_test_pred):.4f}
Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, rfr_test_pred)):.4f}
R-Squared Accuracy:             {r2_score(y_test, rfr_test_pred):.4f}
""")

joblib.dump(rfr_model, 'solar_model_rfr.pkl')
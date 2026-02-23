import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

# Loading the dataset
# (Here a mergesd and encoded dataset is loaded, which is the output of previous tasks. This dataset should have all the necessary features for training the model.)

df_master = pd.read_csv('merged_encoded.csv')
df_master['Timestamp'] = pd.to_datetime(df_master['Timestamp'])

# We drop 'Timestamp' because it's not a numeric feature, and 'Capacity_Factor' because it's the target.
X = df_master.drop(columns=['Timestamp', 'Capacity_Factor'])
y = df_master['Capacity_Factor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# Import solar_model_lr.pkl
model = joblib.load('solar_model_lr.pkl')


# Analysis Visualisation


# Figure 1: Model Accuracy & Error Analysis (Plots 1 and 2)

sns.set_theme(style="whitegrid")
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Actual vs Predicted Graph

sample_indices = np.random.choice(len(y_test), size=2000, replace=False)
y_test_sample = y_test.iloc[sample_indices]
y_pred_sample = model.predict(X_test.iloc[sample_indices])

axes1[0].scatter(y_test_sample, y_pred_sample, alpha=0.4, color='#2b5c8f')
axes1[0].plot([0, 1], [0, 1], '--', color='#d9534f', linewidth=2)
axes1[0].set_title('Linear Regression: Actual vs. Predicted', fontsize=14, fontweight='bold')
axes1[0].set_xlabel('Actual Solar Generation (Capacity Factor)')
axes1[0].set_ylabel('Predicted Solar Generation')

# Plot 2: Error Distribution (Residuals)
errors = y_test_sample - y_pred_sample
sns.histplot(errors, bins=50, kde=True, ax=axes1[1], color='#5cb85c')
axes1[1].set_title('Error Distribution (Residuals)', fontsize=14, fontweight='bold')
axes1[1].set_xlabel('Prediction Error')
axes1[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# Figure 2: Trends & Feature Relationships (Plots 3 and 4)

fig2, axes2 = plt.subplots(1, 2, figsize=(20, 7))

# Plot 3: Seasonality & Trends (1 Summer Week in Spain)
# Retreive 168 hours (1 week) of July data for Spain

spain_summer = df_master[(df_master['Country_ES'] == 1) & (df_master['Month'] == 7)].head(168)
X_spain = spain_summer.drop(columns=['Timestamp', 'Capacity_Factor'])
y_true_spain = spain_summer['Capacity_Factor']
y_pred_spain = model.predict(X_spain)

axes2[0].plot(spain_summer['Timestamp'], y_true_spain, label='Actual Generation', color='black', linewidth=2)
axes2[0].plot(spain_summer['Timestamp'], y_pred_spain, label='Model Forecast', color='#f0ad4e', linestyle='dashed', linewidth=2)
axes2[0].set_title('Seasonality & Trend: 1 Summer Week in Spain', fontsize=14, fontweight='bold')
axes2[0].set_xlabel('Timeline (Hours)')
axes2[0].set_ylabel('Power Output')
axes2[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10)) 
axes2[0].tick_params(axis='x', rotation=45)
axes2[0].legend()

# Plot 4: Correlation Heatmap (Feature Relationships)

# Exclude the 29+ country columns to keep the heatmap clean and readable
cols_to_check = ['Capacity_Factor', 'Hour', 'Month', 'Irradiance', 'Temperature', 'Wind_Speed']
corr_matrix = df_master[cols_to_check].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=axes2[1], cbar=True)
axes2[1].set_title('Correlation Heatmap: Feature Relationships', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

# Loading the dataset
# (Here a mergesd and encoded dataset is loaded, which is the output of previous tasks. This dataset should have all the necessary features for training the model.)

df_master2 = pd.read_csv('merged_encoded.csv')
df_master2['Timestamp'] = pd.to_datetime(df_master2['Timestamp'])

# We drop 'Timestamp' because it's not a numeric feature, and 'Capacity_Factor' because it's the target.
X = df_master2.drop(columns=['Timestamp', 'Capacity_Factor'])
y = df_master2['Capacity_Factor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# Loading the saved model
model = joblib.load('solar_model_rfr.pkl')


# Analysis Visualisation



# Setting up visual canvas
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Plot 1: Actual vs Predicted Graph
# 2000 Sample points

sample_indices = np.random.choice(len(y_test), size=2000, replace=False)
y_test_sample = y_test.iloc[sample_indices]
y_pred_sample = model.predict(X_test.iloc[sample_indices]) # Use RFR for predictions

axes[0].scatter(y_test_sample, y_pred_sample, alpha=0.4, color='#2b5c8f')
axes[0].plot([0, 1], [0, 1], '--', color='#d9534f', linewidth=2)
axes[0].set_title('Random Forest: Actual vs. Predicted', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Actual Solar Generation (Capacity Factor)')
axes[0].set_ylabel('Predicted Solar Generation')

# Plot 2: Error Distribution (Residuals)
errors = y_test_sample - y_pred_sample
sns.histplot(errors, bins=50, kde=True, ax=axes[1], color='#5cb85c')
axes[1].set_title('Error Distribution (Residuals)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Prediction Error')
axes[1].set_ylabel('Frequency')

# Plot 3: Seasonality & Trends (1 Summer Week in Spain)
# Retreive 168 hours (1 week) of July data for Spain

spain_summer = df_master2[(df_master2['Country_ES'] == 1) & (df_master2['Month'] == 7)].head(168)
X_spain = spain_summer.drop(columns=['Timestamp', 'Capacity_Factor'])
y_true_spain = spain_summer['Capacity_Factor']
y_pred_spain = model.predict(X_spain) 

axes[2].plot(spain_summer['Timestamp'], y_true_spain, label='Actual Generation', color='black', linewidth=2)
axes[2].plot(spain_summer['Timestamp'], y_pred_spain, label='Model Forecast', color='#f0ad4e', linestyle='dashed', linewidth=2)
axes[2].set_title('Seasonality & Trend: 1 Summer Week in Spain', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Timeline (Hours)')
axes[2].set_ylabel('Power Output')
axes[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=10)) 
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend()

plt.tight_layout()
plt.show()
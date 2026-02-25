import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Loading the dataset
# (Here a mergesd and encoded dataset is loaded, which is the output of previous tasks. This dataset should have all the necessary features for training the model.)

df_master2 = pd.read_csv('merged_encoded.csv')
print(df_master2.head(15))


# Training (using Random Forest Regressor) -


# We drop 'Timestamp' because it's not a numeric feature, and 'Capacity_Factor' because it's the target.
X = df_master2.drop(columns=['Timestamp', 'Capacity_Factor'])
y = df_master2['Capacity_Factor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# n_jobs=-1 utilizes all cpu cores
model = RandomForestRegressor(n_estimators=100, random_state=67, n_jobs=-1, max_depth=12)

model.fit(X_train, y_train)

ytrain_pred = model.predict(X_train)
ytest_pred = model.predict(X_test)

mae_train = mean_absolute_error(y_train, ytrain_pred)
mae_test = mean_absolute_error(y_test, ytest_pred)

rmse_train = np.sqrt(mean_squared_error(y_train, ytrain_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, ytest_pred))

r2_train = r2_score(y_train, ytrain_pred)
r2_test = r2_score(y_test, ytest_pred)  


print(f"""
Training:
      
Mean Absolute Error (MAE):      {mae_train:.4f}
Root Mean Squared Error (RMSE): {rmse_train:.4f}
R-Squared Accuracy:             {r2_train:.4f}
""")

print("")

print(f"""
Testing:
Mean Absolute Error (MAE):      {mae_test:.4f}
Root Mean Squared Error (RMSE): {rmse_test:.4f}
R-Squared Accuracy:             {r2_test:.4f}
""")

# Analysis of the model - 

# Extract feature importance from the Random Forest model
features = X.columns
importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})


importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("Top 10 Most Impactful Features")
print(importance_df.head(10))

# Exporting the model using joblib

joblib.dump(model, 'solar_model_rfr.pkl')
print("Model saved successfully as 'solar_model_rfr.pkl'")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Loading the dataset
# (Here a mergesd and encoded dataset is loaded, which is the output of previous tasks. This dataset should have all the necessary features for training the model.)

df_master2 = pd.read_csv('merged_encoded.csv')
print(df_master2.head(15))


# Training (using Linear Regression) -


# We drop 'Timestamp' because it's not a numeric feature, and 'Capacity_Factor' because it's the target.
X = df_master2.drop(columns=['Timestamp', 'Capacity_Factor'])
y = df_master2['Capacity_Factor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# Initialize Linear Regression model
model = LinearRegression()

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
TRAINING METRICS:
Mean Absolute Error (MAE):      {mae_train:.4f}
Root Mean Squared Error (RMSE): {rmse_train:.4f}
R-Squared Accuracy:             {r2_train:.4f}
""")

print(f"""
TESTING METRICS:
Mean Absolute Error (MAE):      {mae_test:.4f}
Root Mean Squared Error (RMSE): {rmse_test:.4f}
R-Squared Accuracy:             {r2_test:.4f}
""")

# Analysis of the model - 

# Extract feature importance from the Linear Regression model
features = X.columns
weights = model.coef_

importance_df = pd.DataFrame({
    'Feature': features,
    'Weight': weights
})

# Use absolute weights to determine overall impact
importance_df['Absolute_Weight'] = importance_df['Weight'].abs()
importance_df = importance_df.sort_values(by='Absolute_Weight', ascending=False)
importance_df = importance_df.drop(columns=['Absolute_Weight']).reset_index(drop=True)

print("Top 10 Most Impactful Features")
print(importance_df.head(10))

# Exporting the model using joblib

joblib.dump(model, 'solar_model_lr.pkl')
print("Model saved successfully as 'solar_model_lr.pkl'")
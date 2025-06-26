#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 09:41:20 2025

@author: Bronze
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
eth_data = pd.read_csv('ethereum.csv')
btc_data = pd.read_csv('bitcoin.csv')

# Convert Date column to datetime format
eth_data["date"] = pd.to_datetime(eth_data["date"])
btc_data["Date"] = pd.to_datetime(btc_data["Date"])

# View first 5 rows of both datasets
from IPython.display import display
display(btc_data.head(5))
display(eth_data.head(5))

# Check for missing values in Bitcoin data
bitcoin_missing = btc_data.isnull().sum()

# Check for missing values in Ethereum data
ethereum_missing = eth_data.isnull().sum()

# Print missing values for Bitcoin
print("Missing values in Bitcoin data:\n", bitcoin_missing)

# Print missing values for Ethereum
print("\Missing values in Ethereum data:\n", ethereum_missing)

# Define the Target Variable (Future Closing Price)
eth_data["Target"] = eth_data["Close"].shift(-1)
btc_data["Target"] = btc_data["Close"].shift(-1)


# View the last row of the datasets
eth_last_row = eth_data.iloc[[-1]]
btc_last_row = btc_data.iloc[[-1]]


# View first 8 rows of both datasets
from IPython.display import display
display(btc_data.head(8))
display(eth_data.head(8))


# Drop NaN values caused by rolling calculations
eth_data.dropna(inplace=True)
btc_data.dropna(inplace=True)

for df in [eth_data, btc_data]:
    df["Price_Spread"] = df["High"] - df["Low"]  # Daily price range
    df["Price_Change"] = df["Close"].pct_change()  # Percentage change in closing price


# Drop NaN values caused by rolling calculations
eth_data.dropna(inplace=True)
btc_data.dropna(inplace=True)
    
    
 # Filter Method- Using Chi Test
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import KBinsDiscretizer

# Define feature sets separately
btc_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'Price_Spread']
eth_features = ['Open', 'High', 'Low', 'Close', 'Price_Change', 'Price_Spread']
target = 'Target'

# Convert numerical features into categorical bins
discretizer_btc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
discretizer_eth = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
discretizer_target = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')  # For Target variable

btc_data_discretized = btc_data.copy()[btc_features + [target]]
eth_data_discretized = eth_data.copy()[eth_features + [target]]

btc_data_discretized[btc_features] = discretizer_btc.fit_transform(btc_data_discretized[btc_features])
eth_data_discretized[eth_features] = discretizer_eth.fit_transform(eth_data_discretized[eth_features])

# Discretize Target (Important Fix)
btc_data_discretized[target] = discretizer_target.fit_transform(btc_data_discretized[[target]])
eth_data_discretized[target] = discretizer_target.fit_transform(eth_data_discretized[[target]])

# Apply Chi-Square Test
selector_btc = SelectKBest(score_func=chi2, k='all')
selector_eth = SelectKBest(score_func=chi2, k='all')

btc_selected = selector_btc.fit(btc_data_discretized[btc_features], btc_data_discretized[target])
eth_selected = selector_eth.fit(eth_data_discretized[eth_features], eth_data_discretized[target])

# Display feature scores
btc_scores = pd.DataFrame({'Feature': btc_features, 'Chi2 Score': btc_selected.scores_})
eth_scores = pd.DataFrame({'Feature': eth_features, 'Chi2 Score': eth_selected.scores_})

print("\nBitcoin Feature Importance using Chi-Square Test:\n", btc_scores.sort_values(by='Chi2 Score', ascending=False))
print("\nEthereum Feature Importance using Chi-Square Test:\n", eth_scores.sort_values(by='Chi2 Score', ascending= False))

    
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define features and target
btc_features = ['Open', 'High', 'Low', 'Close']
eth_features = ['Open', 'High', 'Low', 'Close']
target = 'Target'



# Split data into training and testing sets (80% train, 20% test)
X_btc_train, X_btc_test, y_btc_train, y_btc_test = train_test_split(btc_data[btc_features], btc_data[target], test_size=0.2, random_state=42)
X_eth_train, X_eth_test, y_eth_train, y_eth_test = train_test_split(eth_data[eth_features], eth_data[target], test_size=0.2, random_state=42)

# Standardize the features
scaler_btc = StandardScaler()
scaler_eth = StandardScaler()
X_btc_train_scaled = scaler_btc.fit_transform(X_btc_train)
X_btc_test_scaled = scaler_btc.transform(X_btc_test)
X_eth_train_scaled = scaler_eth.fit_transform(X_eth_train)
X_eth_test_scaled = scaler_eth.transform(X_eth_test)

# Train Random Forest Regressor
rf_btc = RandomForestRegressor(n_estimators=100, random_state=42)
rf_eth = RandomForestRegressor(n_estimators=100, random_state=42)
rf_btc.fit(X_btc_train_scaled, y_btc_train)
rf_eth.fit(X_eth_train_scaled, y_eth_train)

# Predictions
y_btc_pred = rf_btc.predict(X_btc_test_scaled)
y_eth_pred = rf_eth.predict(X_eth_test_scaled)

# Evaluate the model
btc_rmse = np.sqrt(mean_squared_error(y_btc_test, y_btc_pred))
eth_rmse = np.sqrt(mean_squared_error(y_eth_test, y_eth_pred))
btc_r2 = r2_score(y_btc_test, y_btc_pred)
eth_r2 = r2_score(y_eth_test, y_eth_pred)

# Print evaluation results
print(f'Bitcoin Random Forest - RMSE: {btc_rmse:.4f}, R^2: {btc_r2:.4f}')
print(f'Ethereum Random Forest - RMSE: {eth_rmse:.4f}, R^2: {eth_r2:.4f}')

# Random Forest
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Bitcoin Plot
plt.subplot(1, 2, 1)
plt.scatter(y_btc_test, y_btc_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Bitcoin Prices")
plt.ylabel("Predicted Bitcoin Prices")
plt.title("Bitcoin: Actual vs. Predicted")
plt.plot([min(y_btc_test), max(y_btc_test)], [min(y_btc_test), max(y_btc_test)], color='red', linestyle='dashed')  # 45-degree line

# Ethereum Plot
plt.subplot(1, 2, 2)
plt.scatter(y_eth_test, y_eth_pred, alpha=0.5, color='green')
plt.xlabel("Actual Ethereum Prices")
plt.ylabel("Predicted Ethereum Prices")
plt.title("Ethereum: Actual vs. Predicted")
plt.plot([min(y_eth_test), max(y_eth_test)], [min(y_eth_test), max(y_eth_test)], color='red', linestyle='dashed')  # 45-degree line

plt.tight_layout()
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# Define features and target
btc_features = ['Open', 'High', 'Low', 'Close']
eth_features = ['Open', 'High', 'Low', 'Close']
target = 'Target'

# Split data into training and testing sets (80% train, 20% test)
X_btc_train, X_btc_test, y_btc_train, y_btc_test = train_test_split(btc_data[btc_features], btc_data[target], test_size=0.2, random_state=42)
X_eth_train, X_eth_test, y_eth_train, y_eth_test = train_test_split(eth_data[eth_features], eth_data[target], test_size=0.2, random_state=42)

# Standardize the features
scaler_btc = StandardScaler()
scaler_eth = StandardScaler()
X_btc_train_scaled = scaler_btc.fit_transform(X_btc_train)
X_btc_test_scaled = scaler_btc.transform(X_btc_test)
X_eth_train_scaled = scaler_eth.fit_transform(X_eth_train)
X_eth_test_scaled = scaler_eth.transform(X_eth_test)

# Train Random Forest Regressor
rf_btc = RandomForestRegressor(n_estimators=100, random_state=42)
rf_eth = RandomForestRegressor(n_estimators=100, random_state=42)
rf_btc.fit(X_btc_train_scaled, y_btc_train)
rf_eth.fit(X_eth_train_scaled, y_eth_train)

# Train Linear Regression
lr_btc = LinearRegression()
lr_eth = LinearRegression()
lr_btc.fit(X_btc_train_scaled, y_btc_train)
lr_eth.fit(X_eth_train_scaled, y_eth_train)

# Train KNN
knn_btc = KNeighborsRegressor(n_neighbors=5)
knn_eth = KNeighborsRegressor(n_neighbors=5)
knn_btc.fit(X_btc_train_scaled, y_btc_train)
knn_eth.fit(X_eth_train_scaled, y_eth_train)

# Predictions
y_btc_pred_rf = rf_btc.predict(X_btc_test_scaled)
y_eth_pred_rf = rf_eth.predict(X_eth_test_scaled)
y_btc_pred_lr = lr_btc.predict(X_btc_test_scaled)
y_eth_pred_lr = lr_eth.predict(X_eth_test_scaled)
y_btc_pred_knn = knn_btc.predict(X_btc_test_scaled)
y_eth_pred_knn = knn_eth.predict(X_eth_test_scaled)

# Evaluate the models
btc_rmse_rf = np.sqrt(mean_squared_error(y_btc_test, y_btc_pred_rf))
eth_rmse_rf = np.sqrt(mean_squared_error(y_eth_test, y_eth_pred_rf))
btc_r2_rf = r2_score(y_btc_test, y_btc_pred_rf)
eth_r2_rf = r2_score(y_eth_test, y_eth_pred_rf)

btc_rmse_lr = np.sqrt(mean_squared_error(y_btc_test, y_btc_pred_lr))
eth_rmse_lr = np.sqrt(mean_squared_error(y_eth_test, y_eth_pred_lr))
btc_r2_lr = r2_score(y_btc_test, y_btc_pred_lr)
eth_r2_lr = r2_score(y_eth_test, y_eth_pred_lr)

btc_rmse_knn = np.sqrt(mean_squared_error(y_btc_test, y_btc_pred_knn))
eth_rmse_knn = np.sqrt(mean_squared_error(y_eth_test, y_eth_pred_knn))
btc_r2_knn = r2_score(y_btc_test, y_btc_pred_knn)
eth_r2_knn = r2_score(y_eth_test, y_eth_pred_knn)

# Print evaluation results
print(f'Bitcoin Random Forest - RMSE: {btc_rmse_rf:.4f}, R^2: {btc_r2_rf:.4f}')
print(f'Ethereum Random Forest - RMSE: {eth_rmse_rf:.4f}, R^2: {eth_r2_rf:.4f}')
print(f'Bitcoin Linear Regression - RMSE: {btc_rmse_lr:.4f}, R^2: {btc_r2_lr:.4f}')
print(f'Ethereum Linear Regression - RMSE: {eth_rmse_lr:.4f}, R^2: {eth_r2_lr:.4f}')
print(f'Bitcoin KNN - RMSE: {btc_rmse_knn:.4f}, R^2: {btc_r2_knn:.4f}')
print(f'Ethereum KNN - RMSE: {eth_rmse_knn:.4f}, R^2: {eth_r2_knn:.4f}')

# Visualization of the models

models = ['Random Forest', 'Linear Regression', 'KNN ']
rmse_values_btc = [btc_rmse_rf, btc_rmse_lr, btc_rmse_knn]
rmse_values_eth = [eth_rmse_rf, eth_rmse_lr, eth_rmse_knn]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(models, rmse_values_btc, color=['blue', 'green', 'red'])
plt.xlabel("Models")
plt.ylabel("RMSE")
plt.title("Bitcoin Model Comparison")

plt.subplot(1, 2, 2)
plt.bar(models, rmse_values_eth, color=['blue', 'green', 'red'])
plt.xlabel("Models")
plt.ylabel("RMSE")
plt.title("Ethereum Model Comparison")

plt.tight_layout()
plt.show()


#  DataFrame for Bitcoin predictions
btc_predictions = pd.DataFrame({
    'Actual': y_btc_test,
    'Random Forest': y_btc_pred_rf,
    'Linear Regression': y_btc_pred_lr,
    'KNN Regression': y_btc_pred_knn
})

# DataFrame for Ethereum predictions
eth_predictions = pd.DataFrame({
    'Actual': y_eth_test,
    'Random Forest': y_eth_pred_rf,
    'Linear Regression': y_eth_pred_lr,
    'KNN Regression': y_eth_pred_knn
})

# Display first 10 rows
print("Bitcoin Predictions:\n", btc_predictions.head(10))
print("\nEthereum Predictions:\n", eth_predictions.head(10))
 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import timedelta
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Ensure the script's directory is the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Print the current working directory
print("Current working directory:", os.getcwd())

# Define constants
save_path = 'D:/bidirectional_lstm_model_results'
target_column = 'Greene County, PA'
data_file_path = 'County_zhvi_processed.xlsx'

# Load the model
model_path = os.path.join(save_path, f'{target_column}_Bidirectional_LSTM_model.h5')
model = load_model(model_path)

# Load the dataset
data = pd.read_excel(data_file_path)
data['Date'] = pd.to_datetime(data['Date'])
X = data.drop(columns=[target_column, 'Date']).values
y = data[target_column].values
dates = data['Date'].values

# Compute scalers from the dataset
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Reshape input to be 3D [samples, timesteps, features]
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Make predictions for training period
predictions = model.predict(X_scaled)
predictions_inv = scaler_y.inverse_transform(predictions).flatten()

# Generate future dates and prepare input data
last_date = pd.to_datetime(dates[-1])
future_dates = pd.date_range(last_date + timedelta(days=1), periods=36, freq='M').values  # Extending 3 years
future_X = np.tile(X[-1], (len(future_dates), 1))
future_X_scaled = scaler_X.transform(future_X)
future_X_scaled = np.reshape(future_X_scaled, (future_X_scaled.shape[0], 1, future_X_scaled.shape[1]))

# Make future predictions
future_predictions = model.predict(future_X_scaled)
future_predictions_inv = scaler_y.inverse_transform(future_predictions).flatten()

# Combine dates and predictions
all_dates = np.concatenate([dates, future_dates])
all_predictions = np.concatenate([predictions_inv, future_predictions_inv])

# Calculate confidence intervals (simple approach)
confidence_interval = 0.1  # Assuming 10% confidence interval for illustration
lower_bound = all_predictions * (1 - confidence_interval)
upper_bound = all_predictions * (1 + confidence_interval)

# Plot predictions with confidence intervals and actual data
plt.figure(figsize=(12, 6))
plt.plot(dates, y, label='Actual Value', color='green')
plt.plot(all_dates, all_predictions, label='Predicted Value', color='blue')
plt.fill_between(all_dates, lower_bound, upper_bound, color='b', alpha=0.2, label='Confidence Interval')
plt.axvline(x=last_date, color='r', linestyle='--', label='Training End Date')
plt.title('Predicted Values for Greene County, PA (Including Future Predictions)')
plt.xlabel('Date')
plt.ylabel('Predicted Value')
plt.legend()
plt.show()

# Save predictions to CSV
predictions_df = pd.DataFrame({'Date': all_dates, 'Predicted Value': all_predictions})
predictions_df.to_csv('predicted_values_extended.csv', index=False)

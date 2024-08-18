import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import csv
import gc

# Ensure the script's directory is the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Print the current working directory
print('Current working directory:', os.getcwd())

# Enable GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    for i, device in enumerate(physical_devices):
        print(f"GPU {i}: {device.name} is being used.")
else:
    print("No GPU found, using CPU.")

# Custom callback to compute and display additional metrics
class MetricsCallback(Callback):
    def __init__(self, X_train, y_train, X_val, y_val, scaler_y, log_path, county):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scaler_y = scaler_y
        self.log_path = log_path
        self.county = county

        # Initialize log file
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'County', 'Train RMSE', 'Validation RMSE', 'Train MAE', 'Validation MAE', 'Train R2', 'Validation R2'])

    def on_epoch_end(self, epoch, logs=None):
        y_pred_train = self.model.predict(self.X_train)
        y_pred_val = self.model.predict(self.X_val)

        # Inverse transform to get original scale
        y_train_inv = self.scaler_y.inverse_transform(self.y_train)
        y_val_inv = self.scaler_y.inverse_transform(self.y_val)
        y_pred_train_inv = self.scaler_y.inverse_transform(y_pred_train)
        y_pred_val_inv = self.scaler_y.inverse_transform(y_pred_val)

        rmse_train = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
        rmse_val = np.sqrt(mean_squared_error(y_val_inv, y_pred_val_inv))

        mae_train = mean_absolute_error(y_train_inv, y_pred_train_inv)
        mae_val = mean_absolute_error(y_val_inv, y_pred_val_inv)

        r2_train = r2_score(y_train_inv, y_pred_train_inv)
        r2_val = r2_score(y_val_inv, y_pred_val_inv)

        print(f"Epoch {epoch+1}:")
        print(f"County: {self.county}")
        print(f"Train RMSE: {rmse_train:.4f}, Validation RMSE: {rmse_val:.4f}")
        print(f"Train MAE: {mae_train:.4f}, Validation MAE: {mae_val:.4f}")
        print(f"Train R2: {r2_train:.4f}, Validation R2: {r2_val:.4f}")

        # Log the metrics to a file
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, self.county, rmse_train, rmse_val, mae_train, mae_val, r2_train, r2_val])

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train and evaluate the LSTM model
def train_and_evaluate_lstm(X, y, target_column, save_path):
    try:
        # Normalize the target variable
        scaler_y = MinMaxScaler()
        y = scaler_y.fit_transform(y.reshape(-1, 1))

        # Normalize the features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)

        # Reshape input to be 3D [samples, timesteps, features]
        X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Build and train the LSTM model
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

        # Callbacks
        log_path = os.path.join(save_path, f'{target_column}_training_log.csv')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
        metrics_callback = MetricsCallback(X_train, y_train, X_test, y_test, scaler_y, log_path, target_column)

        history = model.fit(
            X_train, y_train,
            epochs=100,
            validation_split=0.2,
            batch_size=8,
            verbose=2,
            callbacks=[early_stopping, reduce_lr, metrics_callback]
        )

        # Save the trained model
        model_save_path = os.path.join(save_path, f'{target_column}_LSTM_model.h5')
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        # Save the scalers
        scaler_X_path = os.path.join(save_path, f'{target_column}_scaler_X.pkl')
        scaler_y_path = os.path.join(save_path, f'{target_column}_scaler_y.pkl')
        joblib.dump(scaler_X, scaler_X_path)
        joblib.dump(scaler_y, scaler_y_path)
        print(f"Scalers saved to {scaler_X_path} and {scaler_y_path}")

        # After training and saving each model
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        # After clearing the session, add:
        gc.collect()

        return {
            'model_name': 'LSTM',
            'history': history.history
        }

    except Exception as e:
        return {
            'model_name': 'LSTM',
            'error': str(e)
        }

# Directory to save results
save_path = '/mnt/d/WSL2_bidirectional_lstm_model_results'
os.makedirs(save_path, exist_ok=True)

# Load the dataset
file_path = 'County_zhvi_processed.xlsx'
data = pd.read_excel(file_path)

# Get the list of files already present in the results directory
existing_files = os.listdir(save_path)

# Loop through each column to use it as the target
for column in data.columns:
    if column != 'Date':
        # Check if a file corresponding to this county already exists in the directory
        model_file = f"{column}_LSTM_model.h5"
        if model_file in existing_files:
            print(f"Skipping {column} as it has already been trained and evaluated.")
            continue
        
        X = data.drop(columns=[column, 'Date']).values  # all columns except the target column and 'Date' are features
        y = data[column].values  # target column

        result = train_and_evaluate_lstm(X, y, column, save_path)
        
        if 'error' in result:
            print(f"Error in training with target column {column}: {result['error']}")
        else:
            print(f"Training completed for target column {column}")

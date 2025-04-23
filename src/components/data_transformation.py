import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    cleaned_data_path: str = os.path.join("artifacts", "cleaned_power_consumption.csv")
    transformed_train_path: str = os.path.join("artifacts", "transformed_train.npz")
    transformed_test_path: str = os.path.join("artifacts", "transformed_test.npz")
    scaler_path: str = os.path.join("artifacts", "scaler.pkl")
    sequence_length: int = 30  # 30 days

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def create_sequences(self, data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    def initiate_data_transformation(self):
        logging.info("Starting data transformation...")

        try:
            df = pd.read_csv(self.config.cleaned_data_path, parse_dates=['datetime'], index_col='datetime')

            # Resample to daily consumption
            df_daily = df['Global_active_power'].resample('D').mean()
            df_daily.dropna(inplace=True)
            logging.info(f"Resampled daily data shape: {df_daily.shape}")

            # Normalize data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_daily.values.reshape(-1, 1))
            logging.info("Data normalization complete.")

            # Create sequences
            seq_len = self.config.sequence_length
            X, y = self.create_sequences(scaled_data, seq_len)
            logging.info(f"Generated sequences: X shape = {X.shape}, y shape = {y.shape}")

            # Train-test split (80-20)
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_test, y_test = X[split_idx:], y[split_idx:]

            # Save data
            np.savez_compressed(self.config.transformed_train_path, X=X_train, y=y_train)
            np.savez_compressed(self.config.transformed_test_path, X=X_test, y=y_test)
            save_object(self.config.scaler_path, scaler)

            logging.info(f"Train and test sequences saved. Scaler saved at {self.config.scaler_path}")
            return self.config.transformed_train_path, self.config.transformed_test_path

        except Exception as e:
            raise CustomException(e, sys)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import sys
import numpy as np
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import mlflow
import mlflow.tensorflow

from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "lstm_model.h5")
    input_shape: tuple = (30, 1)  # 30 days, 1 feature
    lstm_units: int = 64
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 20
    batch_size: int = 32

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def build_model(self):
        logging.info("Building LSTM model...")
        model = Sequential()
        model.add(LSTM(units=self.config.lstm_units, input_shape=self.config.input_shape))
        model.add(Dropout(self.config.dropout_rate))
        model.add(Dense(1))  # Regression output

        model.compile(optimizer=Adam(learning_rate=self.config.learning_rate), loss='mse', metrics=['mae'])
        return model

    def initiate_model_trainer(self, train_array_path: str, test_array_path: str):
        logging.info("Starting model training...")

        try:
            # Load data
            train_data = np.load(train_array_path, allow_pickle=True)
            test_data = np.load(test_array_path, allow_pickle=True)

            X_train, y_train = train_data['X'], train_data['y']
            X_test, y_test = test_data['X'], test_data['y']

            # Build and train model
            model = self.build_model()
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            mlflow.tensorflow.autolog()

            with mlflow.start_run():
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    callbacks=[early_stop]
                )

                # Save model
                model.save(self.config.model_path)
                logging.info(f"Model saved at {self.config.model_path}")

                # Log final loss
                final_val_loss = history.history['val_loss'][-1]
                logging.info(f"Final validation loss: {final_val_loss:.4f}")

                return final_val_loss

        except Exception as e:
            raise CustomException(e, sys)

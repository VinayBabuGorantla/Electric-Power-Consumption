import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "lstm_model.h5")
        self.scaler_path = os.path.join("artifacts", "scaler.pkl")
        self.sequence_length = 30

        # Load once
        self.model = load_model(self.model_path)
        self.scaler = load_object(self.scaler_path)

    def predict_next_day(self, recent_series: np.ndarray):
        try:
            logging.info("Starting prediction pipeline...")

            if len(recent_series) != self.sequence_length:
                raise ValueError(f"Input sequence must be exactly {self.sequence_length} values long.")

            # Reshape and scale
            scaled_input = self.scaler.transform(recent_series.reshape(-1, 1))
            input_sequence = np.expand_dims(scaled_input, axis=0)  # shape: (1, 30, 1)

            # Predict
            scaled_prediction = self.model.predict(input_sequence)[0][0]

            # Inverse transform to original scale
            predicted_value = self.scaler.inverse_transform([[scaled_prediction]])[0][0]
            logging.info(f"Prediction complete. Next day forecast: {predicted_value:.4f} kW")

            return predicted_value

        except Exception as e:
            raise CustomException(e, sys)

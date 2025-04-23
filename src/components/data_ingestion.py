import os
import sys
import pandas as pd
from dataclasses import dataclass
from urllib.request import urlretrieve
import zipfile

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    download_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    raw_data_dir: str = os.path.join("artifacts", "raw")
    raw_file_path: str = os.path.join("artifacts", "raw", "household_power_consumption.txt")
    cleaned_data_path: str = os.path.join("artifacts", "cleaned_power_consumption.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def download_and_extract(self):
        try:
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            zip_path = os.path.join(self.config.raw_data_dir, "data.zip")

            logging.info("Downloading dataset...")
            urlretrieve(self.config.download_url, zip_path)
            logging.info("Download complete.")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.raw_data_dir)
            logging.info("Extraction complete.")

        except Exception as e:
            raise CustomException(e, sys)

    def clean_and_save(self):
        try:
            logging.info("Reading raw data...")
            df = pd.read_csv(
                self.config.raw_file_path,
                sep=';',
                low_memory=False,
                na_values='?',
                parse_dates={'datetime': ['Date', 'Time']},
                infer_datetime_format=True
            )

            logging.info(f"Initial data shape: {df.shape}")
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            logging.info(f"Data shape after removing missing values: {df.shape}")

            df.to_csv(self.config.cleaned_data_path, index=False)
            logging.info(f"Cleaned data saved at {self.config.cleaned_data_path}")

            return self.config.cleaned_data_path

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process...")
        self.download_and_extract()
        return self.clean_and_save()

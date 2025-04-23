import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    try:
        logging.info(">>> Training pipeline started for Time Series Forecasting Project <<<")
        
        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        cleaned_data_path = ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        transformer = DataTransformation()
        transformed_train_path, transformed_test_path = transformer.initiate_data_transformation()

        # Step 3: Model Training
        trainer = ModelTrainer()
        final_val_loss = trainer.initiate_model_trainer(transformed_train_path, transformed_test_path)
        
        logging.info(f"Training pipeline completed successfully. Final validation loss: {final_val_loss:.4f}")

    except Exception as e:
        raise CustomException(e, sys)

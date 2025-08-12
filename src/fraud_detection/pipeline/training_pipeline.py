import sys
import os
import mlflow
from dotenv import load_dotenv

from src.fraud_detection.components.data_ingestion import DataIngestion
from src.fraud_detection.components.data_validation import DataValidation
from src.fraud_detection.components.data_transformation import DataTransformation
from src.fraud_detection.components.model_trainer import ModelTrainer

from src.fraud_detection.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.fraud_detection.logging.logger import logging
from src.fraud_detection.exception.exception import FraudDetectionException
from src.fraud_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact

# Load env file if available (local dev)
load_dotenv()

# Fetch MLflow credentials from env (works for local + CI/CD via GitHub Secrets)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not all([tracking_uri, username, password]):
    raise ValueError("Missing one or more MLFLOW environment variables.")

os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
os.environ['MLFLOW_TRACKING_USERNAME'] = username
os.environ['MLFLOW_TRACKING_PASSWORD'] = password
mlflow.set_tracking_uri(tracking_uri)


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=config)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=config)
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=config, data_transformation_artifact=data_transformation_artifact)
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def run_pipeline(self):
        try:
            di_artifact = self.start_data_ingestion()
            dv_artifact = self.start_data_validation(data_ingestion_artifact=di_artifact)
            dt_artifact = self.start_data_transformation(data_validation_artifact=dv_artifact)
            mt_artifact = self.start_model_trainer(data_transformation_artifact=dt_artifact)
            return mt_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

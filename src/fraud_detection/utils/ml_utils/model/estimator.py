from src.fraud_detection.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from src.fraud_detection.exception.exception import FraudDetectionException
from src.fraud_detection.logging.logger import logging

import sys

class ClassificationModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise FraudDetectionException(e, sys)
    
    def predict(self, X):
        try:
            X_transformed = self.preprocessor.transform(X)
            return self.model.predict(X_transformed)
        except Exception as e:
            raise FraudDetectionException(e, sys)
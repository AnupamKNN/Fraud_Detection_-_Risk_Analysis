import sys, os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek

from src.fraud_detection.entity.config_entity import DataTransformationConfig
from src.fraud_detection.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact

from src.fraud_detection.exception.exception import FraudDetectionException
from src.fraud_detection.logging.logger import logging
from src.fraud_detection.utils.main_utils.utils import save_object, save_numpy_array_data, read_yaml_file
from src.fraud_detection.constants.training_pipeline import SCHEMA_FILE_PATH, TARGET_COLUMN

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_tranformation_config= data_transformation_config

        except Exception as e:
            raise FraudDetectionException(e, sys)

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FraudDetectionException(e, sys)
        
    def get_data_transformer_object(self)-> Pipeline:
        try:
            """
            A Pipeline object that applies Simple Imputation and Standard Scaling (with first column dropped)
            """

            # Load schema configuration
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            # Identify numerical features to scale, pca features, and cyclic features
            num_features_to_scale = [list(d.keys())[0] for d in self.schema_config['features_to_scale']]
            pca_features = [list(d.keys())[0] for d in self.schema_config['pca_features']]
            cyclic_features = [list(d.keys())[0] for d in self.schema_config['cyclic_features']]

            # Define the numeric transformer
            numeric_transformer = StandardScaler()

            # Combine transformer and pass-through features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, num_features_to_scale),
                    ('pass_pca', 'passthrough', pca_features),
                    ('pass_cyclic', 'passthrough', cyclic_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise FraudDetectionException(e, sys)
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("Entered initiate_data_transformation method of Data Transformation class")
            
            # Read training and testing data
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Extract input_features and target feature for train and test dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis = 1)
            target_feature_train_df = train_df[TARGET_COLUMN].values.reshape(-1,1)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis = 1)
            target_feature_test_df = test_df[TARGET_COLUMN].values.reshape(-1,1)

            # Load schema and extract expected columns
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            expected_columns = [list(col.keys())[0] for col in self.schema_config["numerical_columns"]]

            # Handle and log any column mismatches
            missing_train = set(expected_columns) - set(input_feature_train_df.columns)
            extra_train = set(input_feature_train_df.columns) - set(expected_columns)
            if missing_train:
                logging.warning(f"Missing columns in training data: {missing_train}")
            if extra_train:
                logging.info(f"Extra columns in training data: {extra_train}")

            missing_test = set(expected_columns) - set (input_feature_test_df.columns)
            extra_test = set(input_feature_test_df.columns) - set(expected_columns)
            if missing_test:
                logging.warning(f"Missing columns in testing data: {missing_test}")
            if extra_test:
                logging.info(f"Extra columns in testing data: {extra_test}")

            # Drop extra columns to align with schema

            input_feature_train_df = input_feature_train_df[expected_columns]
            input_feature_test_df = input_feature_test_df[expected_columns]

            # Add columns like TransactionDate, Hour, ElapsedDays, Hour_sin, Hour_cos

            input_feature_train_df["TransactionDate"] = pd.to_datetime(input_feature_train_df["Time"], unit = "s")
            input_feature_test_df["TransactionDate"] = pd.to_datetime(input_feature_test_df["Time"], unit = "s")

            input_feature_train_df["Hour"] = input_feature_train_df["TransactionDate"].dt.hour
            input_feature_test_df["Hour"] = input_feature_test_df["TransactionDate"].dt.hour

            input_feature_train_df["ElapsedDays"] = (input_feature_train_df["TransactionDate"] - input_feature_train_df["TransactionDate"].min()).dt.days
            input_feature_test_df["ElapsedDays"] = (input_feature_test_df["TransactionDate"] - input_feature_test_df["TransactionDate"].min()).dt.days

            input_feature_train_df["Hour_sin"] = np.sin(2 * np.pi * input_feature_train_df["Hour"] / 24)
            input_feature_test_df["Hour_sin"] = np.sin(2 * np.pi * input_feature_test_df["Hour"] / 24)
            
            input_feature_train_df["Hour_cos"] = np.cos(2 * np.pi * input_feature_train_df["Hour"] / 24)
            input_feature_test_df["Hour_cos"] = np.cos(2 * np.pi * input_feature_test_df["Hour"] / 24)

            # Apply preprocessor
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Convert to dense arrays if sparse
            if hasattr(transformed_input_train_feature, "toarray"):
                transformed_input_train_feature = transformed_input_train_feature.toarray()
            if hasattr(transformed_input_test_feature, "toarray"):
                transformed_input_test_feature = transformed_input_test_feature.toarray()

            # Diagnostic Prints
            print("Shape of transformed_input_train_feature:", transformed_input_train_feature.shape)
            print("Shape of target_feature_train_df:", target_feature_train_df.shape)

            print("Shape of transformed_input_test_feature:", transformed_input_test_feature.shape)
            print("Shape of target_feature_test_df:", target_feature_test_df.shape)

            print(f"The type of transformed_input_train_feature is: {type(transformed_input_train_feature)}")
            print(f"The type of target_feature_train_df is: {type(target_feature_train_df)}")

            print(f"The type of transformed_input_test_feature is: {type(transformed_input_test_feature)}")
            print(f"The type of target_feature_test_df is: {type(target_feature_test_df)}")

            # Check the class distribution before balancing
            print("Class distribution before SMOTETomek:")
            unique, counts = np.unique(target_feature_train_df, return_counts=True)
            print(dict(zip(unique, counts)))

            # Apply SMOTETomek for balancing classes
            smt = SMOTETomek(random_state=42)
            resampled_input_train_feature, resampled_target_feature_train = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df.ravel()
                )
            
            # Check class distribution after balancing
            print("Class distribution after SMOTETomek:")
            unique, counts = np.unique(resampled_target_feature_train, return_counts=True)
            print(dict(zip(unique, counts)))

            # Stack feature and target column
            train_arr = np.c_[resampled_input_train_feature, resampled_target_feature_train]
            test_arr = np.c_[transformed_input_test_feature, target_feature_test_df]

            # Save transformed numpy arrays
            save_numpy_array_data(self.data_tranformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_tranformation_config.transformed_test_file_path, array=test_arr)

            # Save the preprocessing object
            save_object(self.data_tranformation_config.transformed_object_file_path, preprocessor_object)
            
            # Implement model pusher
            save_object("final_models/preprocessor.pkl", preprocessor_object)

            # Prepare and return artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path = self.data_tranformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_tranformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_tranformation_config.transformed_test_file_path
            )

            return data_transformation_artifact
    
        except Exception as e:
            raise FraudDetectionException(e, sys)
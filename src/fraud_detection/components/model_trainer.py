import os, sys

from src.fraud_detection.entity.config_entity import ModelTrainerConfig
from src.fraud_detection.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact

from src.fraud_detection.exception.exception import FraudDetectionException
from src.fraud_detection.logging.logger import logging

from src.fraud_detection.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from src.fraud_detection.utils.ml_utils.model.estimator import ClassificationModel
from src.fraud_detection.utils.ml_utils.metric.classification_metric import get_classification_score


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import mlflow
from dotenv import load_dotenv

load_dotenv()

# Safely fetch credentials from .env
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")


# Ensure they exist
if not all([tracking_uri, username, password]):
    raise ValueError("Missing one or more MLFLOW environment variables. Please check your .env file.")

# Set env vars only if present
os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
os.environ['MLFLOW_TRACKING_USERNAME'] = username
os.environ['MLFLOW_TRACKING_PASSWORD'] = password

mlflow.set_tracking_uri(tracking_uri)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def track_mlflow(self, best_model, train_metrics, test_metrics):
        with mlflow.start_run(run_name = "Classification Model"):

            # Train Metrics
            mlflow.log_metric("Train F1 Score", train_metrics.f1_score)
            mlflow.log_metric("Train Precision Score", train_metrics.precision_score)
            mlflow.log_metric("Train Recall Score", train_metrics.recall_score)
            mlflow.log_metric("Train Accuracy Score", train_metrics.accuracy_score)

            # Test Metrics
            mlflow.log_metric("Test F1 Score", test_metrics.f1_score)
            mlflow.log_metric("Test Precision Score", test_metrics.precision_score)
            mlflow.log_metric("Test Recall Score", test_metrics.recall_score)
            mlflow.log_metric("Test Accuracy Score", test_metrics.accuracy_score)

            # Save Model
            mlflow.sklearn.log_model(best_model, "model", registered_model_name = "Classification Model")

    def train_model(self, X_train, y_train, X_test, y_test)-> ClassificationModel:
        try:
            models = {
                    "Logistic Regression": LogisticRegression(),
                    "K-Neighbors Classifier": KNeighborsClassifier(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "AdaBoost Classifier": AdaBoostClassifier(),
                    "Gradient Boosting Classifier": GradientBoostingClassifier(),
                    "XGB Classifier": XGBClassifier(tree_method='hist', device = 'cuda'),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "CatBoost Classifier": CatBoostClassifier(task_type="GPU", devices='0')
                }
            
            params_grid = {
                    "Logistic Regression": [
                        # liblinear solver with l1 or l2 penalties
                        {
                            'solver': ['liblinear'],
                            'penalty': ['l1', 'l2'],
                            'C': [0.1, 1, 10],
                            'max_iter': [100, 200]
                        },
                        # lbfgs solver with l2 or None penalty
                        {
                            'solver': ['lbfgs'],
                            'penalty': ['l2', None],
                            'C': [0.1, 1, 10],
                            'max_iter': [100, 200]
                        },
                        # saga solver with l1, l2, and elasticnet penalties
                        {
                            'solver': ['saga'],
                            'penalty': ['l1', 'l2', 'elasticnet'],
                            'l1_ratio': [0, 0.5, 1],  # only for elasticnet penalty
                            'C': [0.1, 1, 10],
                            'max_iter': [100, 200]
                        }
                    ],

                    "K-Neighbors Classifier": {
                        'n_neighbors': [3, 5, 7],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan'],
                        'p': [1, 2]
                    },

                    "Decision Tree Classifier": {
                        'max_depth': [5, 10, 15],
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best'],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    },

                    "Random Forest Classifier": {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2],
                        'bootstrap': [True]
                    },

                    "AdaBoost Classifier": {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.01, 0.1]
                    },

                    "Gradient Boosting Classifier": {
                        'n_estimators': [100, 150],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5],
                        'subsample': [0.8, 1.0],
                        'min_samples_split': [2, 5]
                    },

                    "XGB Classifier": {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0],
                        'gamma': [0, 0.1]
                    },

                    "CatBoost Classifier": {
                        'iterations': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'depth': [4, 6],
                        'l2_leaf_reg': [3, 5]
                    }
                }

            
            model_report, best_models= evaluate_models(X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test, models = models, param= params_grid)

            # Extract test_accuracy scores from each model
            test_accuracy_scores = {model_name: metrics["test_accuracy"] for model_name, metrics in model_report.items()}

            # Get the best model based on test accuracy
            best_model_name = max(test_accuracy_scores, key = test_accuracy_scores.get)
            best_model = best_models[best_model_name]

            y_train_pred = best_model.predict(X_train)
            train_metrics = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            y_test_pred = best_model.predict(X_test)
            test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Track the experiments with MLFlow
            self.track_mlflow(best_model, train_metrics, test_metrics)

            preprocessor = load_object(file_path= self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            classification_model = ClassificationModel(preprocessor= preprocessor, model= best_model)
            save_object(file_path= self.model_trainer_config.trained_model_file_path, obj= classification_model)
            
            # Implement model pusher
            save_object("final_models/model.pkl", best_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                train_metric_artifact= train_metrics,
                test_metric_artifact= test_metrics
            )
            logging.info(f"Classification Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys)
        
    def initiate_model_trainer(self)-> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Loading train array and test array

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:1000, :-1],
                train_arr[:1000, -1],
                test_arr[:1000, :-1],
                test_arr[:1000, -1]
            )

            model_trainer_artifact = self.train_model(X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test)
            return model_trainer_artifact
        
        except Exception as e:
            raise FraudDetectionException(e, sys)
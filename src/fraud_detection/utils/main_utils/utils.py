import yaml
from src.fraud_detection.exception.exception import FraudDetectionException
from src.fraud_detection.logging.logger import logging
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import os, sys
import numpy as np
import pickle

def read_yaml_file(file_path:str)-> dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise FraudDetectionException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False)-> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        with open(file_path, "w") as file_obj:
            yaml.dump(content, file_obj)
    except Exception as e:
        raise FraudDetectionException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise FraudDetectionException(e, sys)
    
def save_object(file_path: str, obj: object)-> None:
    try:
        logging.info("Entered the save_object method of Main Utils class.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of Main Utils class.")
    except Exception as e:
        raise FraudDetectionException(e, sys)
    

def load_object(file_path: str)-> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist.")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise FraudDetectionException(e, sys)
    
def load_numpy_array_data(file_path: str)->np.array:
    """
    loads numpy array data from the file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise FraudDetectionException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param)-> dict:
    try:
        report = {}

        # Store best estimators
        best_models = {}
        
        for model_name in models:
            model = models[model_name]
            params = param[model_name]

            # RandomizedSearchCV
            cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state=42)
            if model_name in ["XGB Classifier", "CatBoost Classifier"]:
                n_jobs = 1
            else:
                n_jobs = 6
            rs = RandomizedSearchCV(model, params, cv=cv, n_iter=10, n_jobs=n_jobs, verbose=1, random_state=42)
            rs.fit(X_train, y_train)

            y_train_pred = rs.predict(X_train)
            y_test_pred = rs.predict(X_test)

            report[model_name] = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_f1_score": accuracy_score(y_train, y_train_pred),
                "train_precision_score": accuracy_score(y_train, y_train_pred),
                "train_recall_score": accuracy_score(y_train, y_train_pred),


                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "test_f1_score": accuracy_score(y_test, y_test_pred),
                "test_precision_score": accuracy_score(y_test, y_test_pred),
                "test_recall_score": accuracy_score(y_test, y_test_pred),
            }

            # Store best model
            best_models[model_name] = rs.best_estimator_

        return report, best_models
    
    except Exception as e:
        raise FraudDetectionException(e, sys)


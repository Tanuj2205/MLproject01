import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pickle


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        logging.error("Error saving object")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]] # Get hyperparameters for the model
            
            gs = GridSearchCV(model,para,cv=3) # Initialize GridSearchCV
            gs.fit(X_train, y_train) # Fit to training data
            model.set_params(**gs.best_params_)# Set best parameters to the model
            model.fit(X_train, y_train) # Train model
            y_train_pred = model.predict(X_train) # Predict train data
            y_test_pred = model.predict(X_test) # Predict test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        logging.error("Error evaluating models")
        raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
